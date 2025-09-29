from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, HttpUrl
import requests
from bs4 import BeautifulSoup
import json
import re
from typing import List, Dict, AsyncGenerator, Optional
import easyocr
from PIL import Image
import io
import base64
from mistralai import Mistral, DocumentURLChunk, ImageURLChunk
import logging
import asyncio
import os
from dotenv import load_dotenv
from groq import Groq
from datetime import datetime
import uuid
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv(override=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Amazon Product Scraper with Compliance Validation", version="T1.0.9")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Supabase Configuration
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if supabase_url and supabase_key:
    supabase: Client = create_client(supabase_url, supabase_key)
    logger.info("Supabase client initialized successfully.")
else:
    logger.warning("Supabase credentials not found. Database features will be disabled.")
    supabase = None

class ProductRequest(BaseModel):
    url: HttpUrl
    ocr_method: str

class ProductResponse(BaseModel):
    title: str
    images: List[Dict[str, str]]
    success: bool
    message: str = ""

# Initialize OCR readers and AI clients
try:
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    logger.warning(f"EasyOCR initialization failed: {e}")
    easyocr_reader = None

mistral_api_key = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=mistral_api_key) if mistral_api_key else None
if not mistral_api_key:
    logger.warning("MISTRAL_API_KEY not found. Mistral OCR will be unavailable.")

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
else:
    logger.warning("GROQ_API_KEY not found. Compliance validation will be limited.")
    groq_client = None

MIN_BBOX_THRESHOLD = 0
COMPLIANCE_FIELDS = ["MRP", "Net Quantity", "Expiry Date", "Manufacturer", "Country of Origin"]

async def log_to_supabase(product_report: dict):
    """
    Saves the product and scan data to the Supabase database.
    """
    if not supabase:
        logger.warning("Supabase not initialized. Skipping database logging.")
        return
        
    try:
        logger.info(f"Logging report to Supabase for product: {product_report.get('product_name')}")
        
        # 1. Upsert product information to avoid duplicates
        product_upsert_response = supabase.table('products').upsert({
            'product_id_custom': product_report.get('product_id'),
            'name': product_report.get('product_name'),
            'category': product_report.get('product_category'),
            'brand': product_report.get('brand_name'),
            'seller_name': product_report.get('seller_name'),
            'platform': product_report.get('platform'),
            'product_link': product_report.get('product_link'),
            'image_url': product_report.get('image_url')
        }, on_conflict='product_link').execute()

        if not product_upsert_response.data:
            raise Exception("Failed to upsert product data.")

        # Get the unique database ID of the product
        product_db_id = product_upsert_response.data[0]['id']
        logger.info(f"Product upserted with DB ID: {product_db_id}")

        # 2. Insert the new scan record linked to the product
        scan_insert_response = supabase.table('scans').insert({
            'product_id': product_db_id,
            'ocr_text_raw': product_report.get('ocr_text_raw'),
            'ocr_confidence_avg': product_report.get('ocr_confidence_avg'),
            'mrp_status': product_report.get('mrp_status'),
            'net_quantity_status': product_report.get('net_quantity_status'),
            'expiry_date_status': product_report.get('expiry_date_status'),
            'manufacturer_status': product_report.get('manufacturer_status'),
            'country_origin_status': product_report.get('country_of_origin_status'),
            'compliance_percent': int(product_report.get('compliance_percent', 0)),
            'compliance_status': product_report.get('compliance_status'),
            'actual_list': product_report.get('actual_list'),
            'expected_list': product_report.get('expected_list'),
            'violations_count': product_report.get('violations_count'),
            'geo_location': product_report.get('geo_location'),
            'crawl_date': product_report.get('crawl_date')
        }).execute()

        if not scan_insert_response.data:
            raise Exception("Failed to insert scan data.")

        logger.info(f"✅ Successfully logged scan to Supabase for product ID: {product_report.get('product_id')}")
    except Exception as e:
        logger.error(f"❌ Supabase logging failed: {str(e)}")

def check_text_density(image_url: str) -> tuple[int, float]:
    if not easyocr_reader:
        return 0, 0.0
        
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        results = easyocr_reader.readtext(image, detail=1)
        bbox_count = len(results)
        avg_confidence = sum([res[2] for res in results]) / bbox_count if bbox_count > 0 else 0.0
        logger.info(f"Image {image_url} has {bbox_count} text bounding boxes with average confidence {avg_confidence:.2f}")
        return bbox_count, avg_confidence
    except Exception as e:
        logger.error(f"Bounding box check failed for {image_url}: {str(e)}")
        return 0, 0.0

def extract_text_easyocr(image_url: str) -> tuple[str, float]:
    if not easyocr_reader:
        return "EasyOCR not available", 0.0
        
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        results = easyocr_reader.readtext(image, detail=1)
        text = ' '.join([res[1] for res in results]) if results else "No text detected"
        avg_confidence = sum([res[2] for res in results]) / len(results) if results else 0.0
        logger.info(f"EasyOCR processed image {image_url}: {text} (confidence: {avg_confidence:.2f})")
        return text.strip(), avg_confidence
    except Exception as e:
        logger.error(f"EasyOCR failed for {image_url}: {str(e)}")
        return f"EasyOCR text extraction failed: {str(e)}", 0.0

def extract_text_mistral(image_url: str) -> tuple[str, float]:
    try:
        if not mistral_client:
            return "Mistral client not initialized", 0.0
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(image_url, headers=headers, timeout=10)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            logger.error(f"Invalid image content type for {image_url}: {content_type}")
            return "Invalid image content type", 0.0
        encoded = base64.b64encode(response.content).decode('utf-8')
        base64_data_url = f"data:image/jpeg;base64,{encoded}"
        logger.info(f"Sending image to Mistral AI OCR: {image_url}")
        ocr_response = mistral_client.ocr.process(
            document=ImageURLChunk(image_url=base64_data_url),
            model="mistral-ocr-latest"
        )
        response_dict = json.loads(ocr_response.model_dump_json())
        text = ""
        if response_dict.get('pages') and len(response_dict['pages']) > 0:
            markdown = response_dict['pages'][0].get('markdown', '')
            if markdown:
                text = re.sub(r'!\[.*?\]\(.*?\)', '', markdown).strip()
                text = re.sub(r'\n\s*\n+', '\n', text).strip()
                logger.info(f"Mistral AI extracted markdown for {image_url}: {text}")
            if not text:
                logger.warning(f"No text detected by Mistral AI for {image_url}. Response: {response_dict}")
                return "No text detected by Mistral AI", 0.0
        logger.info(f"Mistral AI processed image {image_url}: {text}")
        return text, 0.9  # Default confidence for Mistral
    except Exception as e:
        logger.error(f"Mistral AI OCR failed for {image_url}: {str(e)}")
        if "401" in str(e):
            return "Mistral AI OCR failed: Invalid or missing API key.", 0.0
        return f"Mistral AI OCR text extraction failed: {str(e)}", 0.0

def validate_compliance_with_groq(title: str, ocr_texts: List[str]) -> Dict:
    if not groq_client:
        return {
            "actual_list": [],
            "expected_list": COMPLIANCE_FIELDS,
            "violations_count": len(COMPLIANCE_FIELDS),
            "compliance_percent": 0.0,
            "compliance_status": "Non-compliant",
            "field_status": {field: "missing" for field in COMPLIANCE_FIELDS}
        }
        
    try:
        combined_text = f"Product Title: {title}\n\nOCR Text:\n" + "\n".join(ocr_texts)
        logger.info(f"Input text for Groq validation: {combined_text}")
        prompt = f"""
        You are an AI assistant tasked with analyzing product information to check for compliance with specific guidelines. The required fields are: {', '.join(COMPLIANCE_FIELDS)}. Determine if each field is present in the provided text (product title and OCR-extracted text from images). A field is considered present if its information is explicitly mentioned. Recognize the following patterns:
        - MRP: "MRP: [currency][amount]", "Maximum Retail Price: [amount]", or similar.
        - Net Quantity: "Net Wt: [quantity][unit]", "Net: [quantity]", "Weight: [quantity]", or similar.
        - Expiry Date: "Best by: [date]", "Exp: [date]", "Use by: [date]", "Expires: [date]", or formats like "MM/YYYY", "MM-YY", "DD/MM/YYYY", "Month YYYY", "YYYY-MM-DD".
        - Manufacturer: "Manufactured by: [name]", "Made by: [name]", or similar.
        - Country of Origin: "Made in [country]", "Country of Origin: [country]", "Produce of [country]", or similar.

        Return a JSON object with the following structure, and **only** the JSON object, without any additional text or explanation:

        ```json
        {{
          "actual_list": ["field1", "field2", ...],
          "expected_list": ["MRP", "Net Quantity", "Expiry Date", "Manufacturer", "Country of Origin"],
          "violations_count": <integer>,
          "compliance_percent": <float>,
          "compliance_status": "<Full|Partial|Non-compliant>",
          "field_status": {{
            "MRP": "<present|missing>",
            "Net Quantity": "<present|missing>",
            "Expiry Date": "<present|missing>",
            "Manufacturer": "<present|missing>",
            "Country of Origin": "<present|missing>"
          }}
        }}
        ```

        Analyze the following text and return **only** the JSON object as a string, with proper formatting and no additional text:

        Text to analyze:
        {combined_text}
        """
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {"role": "system", "content": "You are an AI assistant that analyzes text for compliance and returns only a structured JSON object, without any additional text or explanation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        response_text = response.choices[0].message.content.strip()
        logger.info(f"Raw Groq API response: {response_text}")
        try:
            result = json.loads(response_text)
            logger.info(f"Groq compliance validation result: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse Groq response as JSON: {str(e)}")
            return {
                "actual_list": [],
                "expected_list": COMPLIANCE_FIELDS,
                "violations_count": len(COMPLIANCE_FIELDS),
                "compliance_percent": 0.0,
                "compliance_status": "Non-compliant",
                "field_status": {field: "missing" for field in COMPLIANCE_FIELDS}
            }
    except Exception as e:
        logger.error(f"Groq API compliance validation failed: {str(e)}")
        return {
            "actual_list": [],
            "expected_list": COMPLIANCE_FIELDS,
            "violations_count": len(COMPLIANCE_FIELDS),
            "compliance_percent": 0.0,
            "compliance_status": "Non-compliant",
            "field_status": {field: "missing" for field in COMPLIANCE_FIELDS}
        }

@app.get("/")
async def root():
    return {"message": "Compliance Checker API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Export the app for Vercel
handler = app
