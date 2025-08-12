"""
Clean YouTube Processing API with Web UI
Railway deployment-ready with 15+ extraction strategies
"""

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yt_dlp
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="YouTube Processor", description="Clean YouTube video processing with web UI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Models
class YouTubeRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    extract_audio: bool = Field(default=True, description="Extract audio for transcription")
    format_subtitles: bool = Field(default=True, description="Format subtitles with LLM")


class ProcessingResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    logs: List[str] = []


# YouTube Extraction Strategies
EXTRACTION_STRATEGIES = [
    # Android TV strategies
    {"client": "android_tv", "player_client": "android_tv"},
    {"client": "android_tv", "player_client": "android_tv", "use_oauth": True},
    # Android strategies
    {"client": "android", "player_client": "android"},
    {"client": "android", "player_client": "android", "use_oauth": True},
    {"client": "android_music", "player_client": "android_music"},
    # iOS strategies
    {"client": "ios", "player_client": "ios"},
    {"client": "ios_music", "player_client": "ios_music"},
    # Web strategies
    {"client": "web", "player_client": "web"},
    {"client": "web_music", "player_client": "web_music"},
    {"client": "web_embedded", "player_client": "web_embedded"},
    # TV strategies
    {"client": "tv", "player_client": "tv"},
    {"client": "tv_embedded", "player_client": "tv_embedded"},
    # Alternative strategies
    {"client": "mweb", "player_client": "mweb"},
    {"client": "web_safari", "player_client": "web_safari"},
    # Fallback strategies
    {"extractor_args": {"youtube": {"player_client": ["android", "web"]}}},
]


def extract_youtube_info(url: str, logs: List[str]) -> Optional[Dict[str, Any]]:
    """
    Extract YouTube video information using multiple strategies.
    Tries all strategies persistently without early stopping.
    """
    logs.append(f"🎯 Starting extraction for: {url}")
    logs.append(f"📋 Total strategies to try: {len(EXTRACTION_STRATEGIES)}")

    for i, strategy in enumerate(EXTRACTION_STRATEGIES, 1):
        try:
            logs.append(f"🔧 Strategy {i}/{len(EXTRACTION_STRATEGIES)}: {strategy}")

            # Base yt-dlp options
            ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": False, "writesubtitles": True, "writeautomaticsub": True, "subtitleslangs": ["en", "zh", "zh-cn", "zh-tw"], "format": "best[height<=720]/best", **strategy}

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

                if info and "title" in info:
                    logs.append(f"✅ Strategy {i} SUCCESS! Title: {info.get('title', 'Unknown')[:50]}...")

                    # Extract key information
                    result = {"title": info.get("title", "Unknown"), "description": info.get("description", ""), "duration": info.get("duration", 0), "uploader": info.get("uploader", "Unknown"), "upload_date": info.get("upload_date", ""), "view_count": info.get("view_count", 0), "like_count": info.get("like_count", 0), "subtitles": info.get("subtitles", {}), "automatic_captions": info.get("automatic_captions", {}), "url": url, "successful_strategy": i, "strategy_used": strategy}

                    logs.append(f"📊 Extracted: {len(result.get('subtitles', {}))} manual subs, {len(result.get('automatic_captions', {}))} auto subs")
                    return result
                else:
                    logs.append(f"❌ Strategy {i} failed: No valid info extracted")

        except Exception as e:
            error_msg = str(e)
            if "Video unavailable" in error_msg:
                logs.append(f"❌ Strategy {i} failed: Video unavailable")
            elif "Private video" in error_msg:
                logs.append(f"❌ Strategy {i} failed: Private video")
            elif "Sign in to confirm your age" in error_msg:
                logs.append(f"❌ Strategy {i} failed: Age restriction")
            else:
                logs.append(f"❌ Strategy {i} failed: {error_msg[:100]}")
            continue

    logs.append("🚫 All strategies exhausted - extraction failed")
    return None


def format_duration(seconds: int) -> str:
    """Format duration in human readable format."""
    if not seconds:
        return "Unknown"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def extract_subtitles_text(subtitles: Dict, auto_captions: Dict) -> str:
    """Extract and combine subtitle text from available sources."""
    text_content = []

    # Try manual subtitles first (higher quality)
    for lang in ["en", "zh", "zh-cn", "zh-tw"]:
        if lang in subtitles:
            for sub in subtitles[lang]:
                if "url" in sub:
                    text_content.append(f"Manual subtitles ({lang}) available")
                    break

    # Try automatic captions
    if not text_content:
        for lang in ["en", "zh", "zh-cn", "zh-tw"]:
            if lang in auto_captions:
                for sub in auto_captions[lang]:
                    if "url" in sub:
                        text_content.append(f"Automatic captions ({lang}) available")
                        break

    return " | ".join(text_content) if text_content else "No subtitles available"


# API Routes
@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve the web interface."""
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YouTube Processor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            max-width: 800px;
            width: 100%;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        
        .header h1 {
            color: #333;
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.1rem;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 600;
        }
        
        input[type="url"] {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 12px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        
        input[type="url"]:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .options {
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        input[type="checkbox"] {
            width: 20px;
            height: 20px;
        }
        
        .process-btn {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .process-btn:hover {
            transform: translateY(-2px);
        }
        
        .process-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            text-align: center;
            margin: 30px 0;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .result {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            margin-top: 30px;
        }
        
        .result h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .info-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .info-item strong {
            display: block;
            color: #333;
            margin-bottom: 5px;
        }
        
        .logs {
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
        }
        
        .error {
            background: #fed7d7;
            color: #c53030;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        
        .success {
            background: #c6f6d5;
            color: #2f855a;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 YouTube Processor</h1>
            <p>Clean extraction with 15+ persistent strategies</p>
        </div>
        
        <form id="processForm">
            <div class="form-group">
                <label for="url">YouTube URL</label>
                <input 
                    type="url" 
                    id="url" 
                    name="url" 
                    placeholder="https://www.youtube.com/watch?v=..." 
                    required
                >
            </div>
            
            <div class="options">
                <div class="checkbox-group">
                    <input type="checkbox" id="extractAudio" name="extract_audio" checked>
                    <label for="extractAudio">Extract Audio</label>
                </div>
                
                <div class="checkbox-group">
                    <input type="checkbox" id="formatSubs" name="format_subtitles" checked>
                    <label for="formatSubs">Format Subtitles</label>
                </div>
            </div>
            
            <button type="submit" class="process-btn" id="processBtn">
                🚀 Process Video
            </button>
        </form>
        
        <div id="loading" class="loading" style="display: none;">
            <div class="spinner"></div>
            <p>Processing video with persistent strategies...</p>
        </div>
        
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('processForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {
                url: formData.get('url'),
                extract_audio: formData.has('extract_audio'),
                format_subtitles: formData.has('format_subtitles')
            };
            
            const loadingDiv = document.getElementById('loading');
            const resultDiv = document.getElementById('result');
            const processBtn = document.getElementById('processBtn');
            
            // Show loading
            loadingDiv.style.display = 'block';
            resultDiv.innerHTML = '';
            processBtn.disabled = true;
            processBtn.textContent = '⏳ Processing...';
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                });
                
                const result = await response.json();
                
                loadingDiv.style.display = 'none';
                
                if (result.status === 'success') {
                    showResult(result);
                } else {
                    showError(result.message || 'Processing failed');
                }
                
            } catch (error) {
                loadingDiv.style.display = 'none';
                showError('Network error: ' + error.message);
            } finally {
                processBtn.disabled = false;
                processBtn.textContent = '🚀 Process Video';
            }
        });
        
        function showResult(result) {
            const resultDiv = document.getElementById('result');
            const data = result.data;
            
            resultDiv.innerHTML = `
                <div class="success">
                    <strong>✅ Processing completed successfully!</strong>
                </div>
                
                <div class="result">
                    <h3>📺 Video Information</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <strong>Title</strong>
                            ${data.title || 'Unknown'}
                        </div>
                        <div class="info-item">
                            <strong>Duration</strong>
                            ${formatDuration(data.duration)}
                        </div>
                        <div class="info-item">
                            <strong>Uploader</strong>
                            ${data.uploader || 'Unknown'}
                        </div>
                        <div class="info-item">
                            <strong>Views</strong>
                            ${formatNumber(data.view_count)}
                        </div>
                        <div class="info-item">
                            <strong>Strategy Used</strong>
                            ${data.successful_strategy}/${data.total_strategies}
                        </div>
                        <div class="info-item">
                            <strong>Subtitles</strong>
                            ${data.subtitle_info || 'None available'}
                        </div>
                    </div>
                    
                    ${data.description ? `
                        <div style="margin-top: 20px;">
                            <strong>Description:</strong>
                            <div style="background: white; padding: 15px; border-radius: 8px; margin-top: 10px; max-height: 150px; overflow-y: auto;">
                                ${data.description.substring(0, 500)}${data.description.length > 500 ? '...' : ''}
                            </div>
                        </div>
                    ` : ''}
                    
                    <div style="margin-top: 20px;">
                        <strong>📋 Processing Logs:</strong>
                        <div class="logs">${result.logs.join('\\n')}</div>
                    </div>
                </div>
            `;
        }
        
        function showError(message) {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = `
                <div class="error">
                    <strong>❌ Error:</strong> ${message}
                </div>
            `;
        }
        
        function formatDuration(seconds) {
            if (!seconds) return 'Unknown';
            
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = seconds % 60;
            
            if (hours > 0) {
                return `${hours}h ${minutes}m ${secs}s`;
            } else if (minutes > 0) {
                return `${minutes}m ${secs}s`;
            } else {
                return `${secs}s`;
            }
        }
        
        function formatNumber(num) {
            if (!num) return '0';
            return num.toLocaleString();
        }
    </script>
</body>
</html>
    """
    )


@app.post("/process", response_model=ProcessingResponse)
async def process_youtube_video(request: YouTubeRequest):
    """Process YouTube video with persistent extraction strategies."""
    logs = []

    try:
        logs.append(f"🎬 Processing request for: {request.url}")
        logs.append(f"⚙️ Options - Audio: {request.extract_audio}, Subtitles: {request.format_subtitles}")

        # Extract video information
        video_info = extract_youtube_info(request.url, logs)

        if not video_info:
            logs.append("❌ Extraction failed - all strategies exhausted")
            return ProcessingResponse(status="error", message="Failed to extract video information after trying all strategies", logs=logs)

        # Process extracted information
        result_data = {
            "title": video_info.get("title", "Unknown"),
            "description": video_info.get("description", ""),
            "duration": video_info.get("duration", 0),
            "uploader": video_info.get("uploader", "Unknown"),
            "upload_date": video_info.get("upload_date", ""),
            "view_count": video_info.get("view_count", 0),
            "like_count": video_info.get("like_count", 0),
            "successful_strategy": video_info.get("successful_strategy", 0),
            "total_strategies": len(EXTRACTION_STRATEGIES),
            "subtitle_info": extract_subtitles_text(video_info.get("subtitles", {}), video_info.get("automatic_captions", {})),
            "extraction_timestamp": datetime.now().isoformat(),
        }

        logs.append(f"✅ Successfully extracted video: {result_data['title'][:50]}...")
        logs.append(f"📊 Final result: Duration {format_duration(result_data['duration'])}, Views {result_data['view_count']:,}")

        return ProcessingResponse(status="success", message="Video processed successfully", data=result_data, logs=logs)

    except Exception as e:
        error_message = f"Processing error: {str(e)}"
        logs.append(f"❌ {error_message}")
        logger.error(error_message)

        return ProcessingResponse(status="error", message=error_message, logs=logs)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# For Railway deployment
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
