# Deployment Guide for Render

## 🚀 Quick Deploy

This configuration is optimized for Render's free tier and will deploy successfully.

### Files for Deployment:
- `render.yaml` - Render configuration
- `runtime.txt` - Python version specification
- `requirements.txt` - Dependencies (EasyOCR removed for compatibility)
- `start.py` - Startup script
- `Procfile` - Alternative startup configuration

### Steps:

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Deploy to Render - Simplified without EasyOCR"
   git push origin main
   ```

2. **Deploy on Render:**
   - Go to [render.com](https://render.com)
   - Connect your GitHub repository
   - Choose "Web Service"
   - Render will auto-detect the configuration

3. **Set Environment Variables:**
   - `SUPABASE_URL` - Your Supabase project URL
   - `SUPABASE_KEY` - Your Supabase API key  
   - `MISTRAL_API_KEY` - Your Mistral API key
   - `GROQ_API_KEY` - Your Groq API key

## 🔧 Features Available:

✅ **Web Interface** - Full SPA with all pages
✅ **Mistral OCR** - AI-powered text extraction
✅ **Groq AI** - Compliance validation
✅ **Supabase** - Database integration
✅ **Analytics** - Dashboard with embedded data
✅ **API Endpoints** - All REST endpoints working

❌ **EasyOCR** - Removed due to build complexity on free tier

## 🎯 Adding EasyOCR Later:

To add EasyOCR back (requires paid plan):

1. Add to `requirements.txt`:
   ```
   easyocr==1.7.0
   opencv-python-headless==4.8.1.78
   torch==2.1.0
   torchvision==0.16.0
   numpy==1.24.3
   ```

2. Update build command in `render.yaml`:
   ```yaml
   buildCommand: |
     apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
     pip install --upgrade pip
     pip install -r requirements.txt
   ```

## 📊 Expected Performance:

- **Build Time:** ~3-5 minutes
- **Cold Start:** ~10-15 seconds
- **Response Time:** <2 seconds for most endpoints
- **Memory Usage:** ~200-300MB

Your app will be available at: `https://your-app-name.onrender.com`
