"""
YouTube Processing API using the robust multi-strategy yt-dlp approach.
Aligned with langchain_playground/Tools/YouTubeLoader/youtube.py implementation.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

# Import the robust YouTube processor
from langchain_playground.Tools.YouTubeLoader.youtube import youtube_loader

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube Processing API", description="Process YouTube videos with multi-strategy extraction and transcription", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class YouTubeRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")


class YouTubeResponse(BaseModel):
    success: bool
    message: str
    content: str = None
    error: str = None
    metadata: dict = None


@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {"status": "healthy", "service": "YouTube Processing API", "version": "2.0.0", "message": "Robust multi-strategy YouTube processor ready", "features": ["13+ extraction strategies for maximum compatibility", "Cloud environment optimization (Render, Vercel, etc.)", "iOS/Android client bypass methods", "Automatic audio transcription with Fal API", "Gemini LLM content formatting", "Manual and automatic subtitle support", "Comprehensive error handling"]}


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": "2025-08-12T03:12:00Z"}


@app.post("/api/youtube", response_model=YouTubeResponse)
async def process_youtube_video(request: YouTubeRequest):
    """
    Process a YouTube video using the robust multi-strategy approach.

    Features:
    - 13+ different extraction strategies optimized for cloud environments
    - iOS/Android client bypass methods
    - Automatic transcription with Fal API
    - Gemini LLM formatting and content enhancement
    - Manual subtitle extraction with language priority: zh-HK > zh-CN > en
    - Comprehensive error handling for all YouTube restrictions
    - Cloud environment detection and optimization
    """
    try:
        logger.info(f"🎬 Processing YouTube URL: {request.url}")

        # Validate URL format
        if not any(domain in request.url for domain in ["youtube.com", "youtu.be"]):
            raise HTTPException(status_code=400, detail="Invalid YouTube URL. Please provide a valid YouTube video URL (youtube.com or youtu.be).")

        # Process video using the robust youtube_loader
        logger.info("🚀 Starting video processing with multi-strategy approach...")
        content = youtube_loader(request.url)

        # Extract metadata from content for response
        lines = content.split("\n")
        title = "Unknown"
        author = "Unknown"

        for line in lines:
            if line.startswith("Title: "):
                title = line[7:]
            elif line.startswith("Author: "):
                author = line[8:]

        metadata = {"title": title, "author": author, "processing_method": "multi-strategy_extraction", "features_used": ["yt-dlp with 13+ strategies", "cloud environment optimization", "audio transcription", "LLM formatting"]}

        logger.info(f"✅ Video processed successfully: {title} by {author}")

        return YouTubeResponse(success=True, message="Video processed successfully using robust multi-strategy extraction", content=content, metadata=metadata)

    except HTTPException:
        # Re-raise HTTP exceptions (like invalid URL)
        raise

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ YouTube processing failed: {error_msg}")

        # Handle specific error types from the robust youtube.py implementation
        if "Sign in to confirm you're not a bot" in error_msg or "blocking automated requests" in error_msg:
            return YouTubeResponse(success=False, message="YouTube bot detection triggered", error="YouTube is temporarily blocking automated requests. This is often temporary - try again in a few minutes, or use a different video that's more publicly accessible.")
        elif "Private video" in error_msg:
            return YouTubeResponse(success=False, message="Video is private", error="This video is private and cannot be accessed. Please use a public video.")
        elif "Video unavailable" in error_msg:
            return YouTubeResponse(success=False, message="Video unavailable", error="This video is unavailable or has been removed from YouTube.")
        elif "age-restricted" in error_msg.lower():
            return YouTubeResponse(success=False, message="Age-restricted content", error="This video is age-restricted and cannot be processed without authentication.")
        elif "not available on this app" in error_msg or "Watch on the latest version" in error_msg:
            return YouTubeResponse(success=False, message="App restriction", error="This video is restricted by YouTube and cannot be processed through third-party applications. Please try a different video.")
        else:
            return YouTubeResponse(success=False, message="Processing failed", error=f"Failed to process video after trying multiple extraction strategies: {error_msg}")


@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is working"""
    return {"success": True, "message": "API is working correctly", "features": ["Multi-strategy YouTube extraction (13+ methods)", "Cloud environment optimization (Render, Vercel, AWS, etc.)", "iOS/Android client bypass", "Automatic audio transcription with Fal API", "Gemini LLM content formatting", "Manual subtitle extraction (zh-HK, zh-CN, en)", "Comprehensive error handling", "Progress logging and monitoring"], "environment": {"cloud_detected": any([os.getenv("VERCEL"), os.getenv("AWS_LAMBDA_FUNCTION_NAME"), os.getenv("FUNCTIONS_WORKER_RUNTIME"), os.getenv("GOOGLE_CLOUD_PROJECT"), os.getenv("RENDER"), "/tmp" in os.getcwd()]), "python_path": sys.path[:3]}}  # Show first 3 entries


@app.get("/api/status")
async def status_endpoint():
    """Detailed status endpoint for monitoring"""
    return {"service": "YouTube Processing API", "version": "2.0.0", "status": "operational", "capabilities": {"extraction_strategies": 13, "supported_languages": ["zh-HK", "zh-CN", "en"], "audio_transcription": "Fal API", "content_formatting": "Gemini LLM", "cloud_optimized": True}, "environment": {"platform": "cloud" if any([os.getenv("VERCEL"), os.getenv("RENDER"), "/tmp" in os.getcwd()]) else "local", "python_version": sys.version.split()[0]}}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
