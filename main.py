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

# Static and templates directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

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
    logger.info("EasyOCR initialized successfully")
except Exception as e:
    logger.error(f"EasyOCR initialization failed: {e}")
    easyocr_reader = None

mistral_api_key = os.getenv("MISTRAL_API_KEY")
mistral_client = Mistral(api_key=mistral_api_key) if mistral_api_key else None
if not mistral_api_key:
    logger.warning("MISTRAL_API_KEY not found. Mistral OCR will be unavailable.")

groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    groq_client = Groq(api_key=groq_api_key)
    logger.info("Groq client initialized successfully")
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
        logger.error(f"❌ Error details: {e}")
        # Log the specific data that failed
        logger.error(f"❌ Product data: {product_report.get('product_name')} - {product_report.get('product_link')}")

def check_text_density(image_url: str) -> tuple[int, float]:
    if not easyocr_reader:
        logger.warning("EasyOCR not available for text density check")
        return 1, 0.8  # Assume there's text to proceed with processing
        
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

def sanitize_json_string(json_str: str) -> str:
    json_str = re.sub(r'""(\w+\s*\w*)""', r'"\1"', json_str)
    json_str = re.sub(r'(\w+)\1\s*(of\s*of)?\s*(\w+)\3', r'\1 \3', json_str)
    json_str = re.sub(r',\s*present""\s*}', r'}', json_str)
    json_str = re.sub(r'"field_status":\s*{\s*{', r'"field_status": {', json_str)
    return json_str

def extract_json_from_response(response_text: str) -> Dict:
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if json_match:
        json_str = json_match.group(0)
        logger.info(f"Extracted JSON string: {json_str}")
        json_str = sanitize_json_string(json_str)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse sanitized JSON: {json_str}, error: {str(e)}")
            return None
    logger.error("No JSON object found in response")
    return None

def validate_compliance_with_groq(title: str, ocr_texts: List[str]) -> Dict:
    if not groq_client:
        logger.warning("Groq client not available. Returning default non-compliant result.")
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
        logger.debug(f"Sending Groq API request: model=meta-llama/llama-4-maverick-17b-128e-instruct, prompt length={len(prompt)}")
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
            logger.warning(f"Failed to parse Groq response as JSON: {str(e)}. Attempting to extract JSON block.")
            result = extract_json_from_response(response_text)
            if result:
                logger.info(f"Extracted and sanitized Groq compliance validation result: {result}")
                return result
            logger.error(f"Could not extract valid JSON from Groq response: {response_text}")
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

async def scrape_amazon_product(url: str, ocr_method: str) -> AsyncGenerator[str, None]:
    try:
        yield json.dumps({"step": "start", "message": f"Starting to scrape Amazon product page: {url}"})
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        title = "N/A"
        title_selectors = [
            '#productTitle', 'h1.a-size-large', 'h1#title', 'span#productTitle',
            'h1.a-spacing-none', '.a-size-extra-large'
        ]
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem:
                title = title_elem.get_text(strip=True)
                if title:
                    break
        yield json.dumps({"step": "title", "message": f"Extracted product title: {title}"})

        description = "N/A"
        desc_selectors = ['#productDescription', '.a-section .a-text-normal', '#detailBullets_feature_div']
        for selector in desc_selectors:
            desc_elem = soup.select_one(selector)
            if desc_elem:
                description = desc_elem.get_text(strip=True)
                break
        yield json.dumps({"step": "description_extracted", "message": f"Extracted product description: {description}"})

        images = re.findall(r'"hiRes":"(.+?)"', response.text)
        images = [img for img in images if img != 'null' and img.startswith('http') and 'amazon' in img.lower()]
        if not images:
            image_selectors = [
                '#landingImage', '#main-image', 'img#imgTagWrapperId', 'img[data-a-dynamic-image]',
                '.a-dynamic-image', '.imgTagWrapper img', '#imageBlock img'
            ]
            for selector in image_selectors:
                img_elements = soup.select(selector)
                for img in img_elements:
                    dynamic_data = img.get('data-a-dynamic-image')
                    if dynamic_data:
                        try:
                            data_json = json.loads(dynamic_data)
                            sorted_images = sorted(
                                data_json.keys(),
                                key=lambda x: (data_json[x][0] * data_json[x][1] if isinstance(data_json[x], list) else 0),
                                reverse=True
                            )
                            for img_url in sorted_images:
                                if img_url.startswith('http') and 'amazon' in img_url.lower():
                                    if not any(low_res in img_url for low_res in ['_SS', '_SL', '_SR']):
                                        images.append(img_url)
                        except:
                            pass
                    src = img.get('src') or img.get('data-src')
                    if src and src.startswith('http') and 'amazon' in src.lower():
                        if not any(low_res in src for low_res in ['_SS', '_SL', '_SR']):
                            images.append(src)
        images = list(set(images))[:5]
        yield json.dumps({"step": "images_detected", "message": f"Detected {len(images)} image(s)"})

        text_rich_images = []
        ocr_confidences = []
        for img_url in images:
            yield json.dumps({"step": "checking_density", "message": f"Checking text density for image: {img_url}"})
            bbox_count, avg_confidence = check_text_density(img_url)
            if bbox_count >= MIN_BBOX_THRESHOLD:
                text_rich_images.append(img_url)
                ocr_confidences.append(avg_confidence)
                yield json.dumps({"step": "density_result", "message": f"Image {img_url} has {bbox_count} text bounding boxes - proceeding with text extraction"})
            else:
                yield json.dumps({"step": "density_result", "message": f"Image {img_url} has {bbox_count} bounding boxes - proceeding with text extraction due to low threshold"})

        if not text_rich_images:
            ocr_texts = [description] if description != "N/A" else []
            yield json.dumps({
                "step": "complete",
                "title": title,
                "images": [],
                "success": True,
                "message": f"Found {len(images)} image(s), but none had sufficient text. Using product description for compliance check.",
                "compliance_result": {
                    "actual_list": [],
                    "expected_list": COMPLIANCE_FIELDS,
                    "violations_count": len(COMPLIANCE_FIELDS),
                    "compliance_percent": 0.0,
                    "compliance_status": "Non-compliant",
                    "field_status": {field: "missing" for field in COMPLIANCE_FIELDS}
                }
            })
            return

        image_data = []
        ocr_texts = []
        for img_url in text_rich_images:
            yield json.dumps({"step": "extracting_text", "message": f"Extracting text from image: {img_url} using {ocr_method}"})
            if ocr_method == "mistral" and mistral_client:
                text, confidence = extract_text_mistral(img_url)
                ocr_confidences.append(confidence)
            else:
                text, confidence = extract_text_easyocr(img_url)
                ocr_confidences.append(confidence)
            image_data.append({"url": img_url, "text": text})
            ocr_texts.append(text)
            yield json.dumps({"step": "text_extracted", "message": f"Text extracted for {img_url}", "image_data": {"url": img_url, "text": text}})

        if description != "N/A":
            ocr_texts.append(description)

        avg_ocr_confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0.0

        yield json.dumps({"step": "validating_compliance", "message": "Validating compliance using Groq Llama model"})
        compliance_result = validate_compliance_with_groq(title, ocr_texts)

        product_report = {
            "product_id": f"P{str(uuid.uuid4().hex)[:8].upper()}",
            "product_name": title,
            "product_category": "Unknown",
            "brand_name": "Unknown",
            "seller_name": "Unknown",
            "platform": "Amazon",
            "product_link": url,
            "image_url": text_rich_images[0] if text_rich_images else "",
            "ocr_text_raw": " | ".join(ocr_texts),
            "ocr_confidence_avg": round(avg_ocr_confidence, 2),
            "mrp_status": compliance_result["field_status"].get("MRP", "missing"),
            "net_quantity_status": compliance_result["field_status"].get("Net Quantity", "missing"),
            "expiry_date_status": compliance_result["field_status"].get("Expiry Date", "missing"),
            "manufacturer_status": compliance_result["field_status"].get("Manufacturer", "missing"),
            "country_of_origin_status": compliance_result["field_status"].get("Country of Origin", "missing"),
            "compliance_percent": compliance_result["compliance_percent"],
            "compliance_status": compliance_result["compliance_status"],
            "actual_list": json.dumps(compliance_result["actual_list"]),
            "expected_list": json.dumps(compliance_result["expected_list"]),
            "violations_count": compliance_result["violations_count"],
            "geo_location": "Pune, Maharashtra",
            "crawl_date": datetime.now().isoformat()
        }

        await log_to_supabase(product_report)

        yield json.dumps({
            "step": "complete",
            "title": title,
            "images": image_data,
            "success": True,
            "message": f"Scraped successfully. Report saved to Supabase.",
            "compliance_result": compliance_result
        })
    except requests.exceptions.RequestException as e:
        logger.error(f"Scraping failed for {url}: Network error - {str(e)}")
        yield json.dumps({
            "step": "error", "title": "N/A", "images": [], "success": False,
            "message": f"Scraping failed: Network error - {str(e)}",
            "compliance_result": { "actual_list": [], "expected_list": COMPLIANCE_FIELDS, "violations_count": len(COMPLIANCE_FIELDS), "compliance_percent": 0.0, "compliance_status": "Non-compliant", "field_status": {field: "missing" for field in COMPLIANCE_FIELDS}}
        })
    except Exception as e:
        logger.error(f"Scraping failed for {url}: {str(e)}")
        yield json.dumps({
            "step": "error", "title": "N/A", "images": [], "success": False,
            "message": f"Scraping failed: {str(e)}",
            "compliance_result": { "actual_list": [], "expected_list": COMPLIANCE_FIELDS, "violations_count": len(COMPLIANCE_FIELDS), "compliance_percent": 0.0, "compliance_status": "Non-compliant", "field_status": {field: "missing" for field in COMPLIANCE_FIELDS}}
        })

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Serves the main dashboard HTML file using templates."""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serves the main scanner page using templates."""
    return templates.TemplateResponse("app.html", {"request": request})

@app.get("/api/data")
async def get_all_data(
    q: Optional[str] = None,
    categories: List[str] = Query(None),
    brands: List[str] = Query(None),
    platforms: List[str] = Query(None),
    statuses: List[str] = Query(None),
    fields: List[str] = Query(None),  # e.g., ['mrp_status:present', 'net_quantity_status:present']
    compliance_min: Optional[float] = None,
    compliance_max: Optional[float] = None,
    ocr_min: Optional[float] = None,
    violations_min: Optional[int] = None,
    violations_max: Optional[int] = None,
    date_start: Optional[str] = None,  # YYYY-MM-DD
    date_end: Optional[str] = None,    # YYYY-MM-DD
    geos: List[str] = Query(None),
    sort: Optional[str] = "compliance_percent:desc",  # field:dir
    page: int = 1,
    limit: int = 10
):
    """
    API endpoint to fetch filtered and paginated compliance data from the Supabase database.
    Returns flattened data for easier client-side handling.
    """
    try:
        logger.info("API request received for /api/data with filters")
        query = supabase.table('scans').select('*, products(*)', count='exact')

        # Apply search filter (OR across multiple fields)
        if q:
            search_term = f"%{q.lower()}%"
            or_conditions = (
                f"products.name.ilike.{search_term},"
                f"products.brand.ilike.{search_term},"
                f"products.seller_name.ilike.{search_term},"
                f"ocr_text_raw.ilike.{search_term}"
            )
            query = query.or_(or_conditions)

        # Apply category filter
        if categories:
            query = query.in_("products.category", categories)

        # Apply brand filter
        if brands:
            query = query.in_("products.brand", brands)

        # Apply platform filter
        if platforms:
            query = query.in_("products.platform", platforms)

        # Apply status filter
        if statuses:
            query = query.in_("compliance_status", statuses)

        # Apply field presence filters (e.g., mrp_status == 'present')
        if fields:
            for field_str in fields:
                if ':' in field_str:
                    field, value = field_str.split(':')
                    query = query.eq(field, value)
                else:
                    logger.warning(f"Invalid field filter format: {field_str}")

        # Apply compliance percent range
        if compliance_min is not None:
            query = query.gte('compliance_percent', compliance_min)
        if compliance_max is not None:
            query = query.lte('compliance_percent', compliance_max)

        # Apply OCR confidence min
        if ocr_min is not None:
            query = query.gte('ocr_confidence_avg', ocr_min)

        # Apply violations count range
        if violations_min is not None:
            query = query.gte('violations_count', violations_min)
        if violations_max is not None:
            query = query.lte('violations_count', violations_max)

        # Apply crawl date range
        if date_start:
            query = query.gte('crawl_date', f"{date_start}T00:00:00")
        if date_end:
            query = query.lte('crawl_date', f"{date_end}T23:59:59")

        # Apply geo location filter
        if geos:
            query = query.in_("geo_location", geos)

        # Apply sorting
        if sort:
            field, direction = sort.split(':')
            desc = (direction.lower() == 'desc')
            # Adjust field if it's from products table
            if field in ['product_category', 'brand_name', 'product_name', 'seller_name', 'platform']:
                field = f"products.{field.replace('_name', '') if field in ['brand_name', 'product_name'] else field}"
            query = query.order(field, desc=desc)

        # Apply pagination (range is 0-based)
        start = (page - 1) * limit
        end = start + limit - 1
        query = query.range(start, end)

        # Execute query
        response = query.execute()
        
        # Log pagination info for debugging
        logger.info(f"API pagination: page={page}, limit={limit}, start={start}, end={end}")
        logger.info(f"Query returned {len(response.data) if response.data else 0} records, total count: {response.count}")

        if not response.data:
            return {"data": [], "total": 0, "page": page, "limit": limit, "returned_count": 0}

        # Flatten the data (merge products into scans)
        flat_data = []
        for scan in response.data:
            flat_item = scan.copy()
            product_info = flat_item.pop('products', {}) or {}
            flat_item.update(product_info)
            flat_data.append(flat_item)

        total_count = response.count or 0
        returned_count = len(flat_data)
        
        logger.info(f"API response: returning {returned_count} records out of {total_count} total")

        return {
            "data": flat_data, 
            "total": total_count,
            "page": page,
            "limit": limit,
            "returned_count": returned_count,
            "has_more": (start + returned_count) < total_count
        }
    except Exception as e:
        logger.error(f"API Error fetching filtered data: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch data from database.")

@app.get("/scrape")
async def scrape_product(url: str, ocr_method: str):
    if not any(domain in url.lower() for domain in ["amazon.in", "amazon.com", "amazon.co.uk"]):
        return StreamingResponse(
            iter([json.dumps({
                "step": "error",
                "title": "N/A",
                "images": [],
                "success": False,
                "message": "Please provide a valid Amazon product URL",
                "compliance_result": {
                    "actual_list": [],
                    "expected_list": COMPLIANCE_FIELDS,
                    "violations_count": len(COMPLIANCE_FIELDS),
                    "compliance_percent": 0.0,
                    "compliance_status": "Non-compliant",
                    "field_status": {field: "missing" for field in COMPLIANCE_FIELDS}
                }
            })]),
            media_type="text/event-stream"
        )
    if "/dp/" not in url and "/product/" not in url:
        return StreamingResponse(
            iter([json.dumps({
                "step": "error",
                "title": "N/A",
                "images": [],
                "success": False,
                "message": "Invalid Amazon product URL. Must contain /dp/ or /product/",
                "compliance_result": {
                    "actual_list": [],
                    "expected_list": COMPLIANCE_FIELDS,
                    "violations_count": len(COMPLIANCE_FIELDS),
                    "compliance_percent": 0.0,
                    "compliance_status": "Non-compliant",
                    "field_status": {field: "missing" for field in COMPLIANCE_FIELDS}
                }
            })]),
            media_type="text/event-stream"
        )
    if ocr_method not in ["easyocr", "mistral"]:
        return StreamingResponse(
            iter([json.dumps({
                "step": "error",
                "title": "N/A",
                "images": [],
                "success": False,
                "message": "Invalid OCR method. Choose 'easyocr' or 'mistral'.",
                "compliance_result": {
                    "actual_list": [],
                    "expected_list": COMPLIANCE_FIELDS,
                    "violations_count": len(COMPLIANCE_FIELDS),
                    "compliance_percent": 0.0,
                    "compliance_status": "Non-compliant",
                    "field_status": {field: "missing" for field in COMPLIANCE_FIELDS}
                }
            })]),
            media_type="text/event-stream"
        )

    async def stream_response() -> AsyncGenerator[str, None]:
        async for message in scrape_amazon_product(url, ocr_method):
            yield f"data: {message}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(stream_response(), media_type="text/event-stream")


@app.get("/analytics", response_class=HTMLResponse)
async def get_analytics(request: Request):
    """Serves the analytics dashboard HTML file using templates."""
    return templates.TemplateResponse("analytics.html", {"request": request})


@app.get("/api-docs", response_class=HTMLResponse)
async def get_api_docs(request: Request):
    """Serves the API documentation and try-out HTML page using templates."""
    return templates.TemplateResponse("api_docs.html", {"request": request})


@app.get("/scanner", response_class=HTMLResponse)
async def get_scanner(request: Request):
    """Serves the product scanner HTML page using templates."""
    return templates.TemplateResponse("scanproduct4.html", {"request": request})


@app.get("/home", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serves the landing page HTML file using templates."""
    return templates.TemplateResponse("compliance-landing-inline.html", {"request": request})

    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.index:app", host="0.0.0.0", port=8000, reload=True)
