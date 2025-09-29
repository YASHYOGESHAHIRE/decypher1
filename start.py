#!/usr/bin/env python3
"""
Simple startup script for Render deployment
"""
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Import and run the FastAPI app
        from main import app
        import uvicorn
        
        # Get port from environment variable (Render sets this)
        port = int(os.environ.get("PORT", 8000))
        
        logger.info(f"Starting server on port {port}")
        
        # Run the server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            workers=1,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
