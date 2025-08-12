"""
Enhanced YouTube Processing API using existing youtube_loader
Railway deployment with web UI + full transcription & summarization
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google.genai import Client, types
from pydantic import BaseModel, Field

# Import the existing youtube_loader
from langchain_playground.Tools.YouTubeLoader import youtube_loader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="YouTube Processor", description="Full YouTube processing with transcription & summarization")

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
    generate_summary: bool = Field(default=True, description="Generate AI summary")


class ProcessingResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    logs: List[str] = []


def quick_summary(text: str) -> str:
    """Generate summary using Gemini 2.5-pro like the user's example."""
    try:
        client = Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=f"Summarize with list out of the key facts mentioned. Follow the language of the text.\n\n{text}",
            config=types.GenerateContentConfig(temperature=0),
        )
        return response.text
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return f"Summary generation failed: {str(e)}"


def extract_video_metadata(content: str) -> Dict[str, str]:
    """Extract title and author from youtube_loader output."""
    lines = content.split("\n")
    title = "Unknown"
    author = "Unknown"
    subtitle = ""

    for i, line in enumerate(lines):
        if line.startswith("Title: "):
            title = line.replace("Title: ", "")
        elif line.startswith("Author: "):
            author = line.replace("Author: ", "")
        elif line.startswith("subtitle:"):
            # Everything after "subtitle:" line
            subtitle = "\n".join(lines[i + 1 :])
            break

    return {"title": title, "author": author, "subtitle": subtitle}


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
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            max-width: 1000px;
            margin: 0 auto;
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
        
        .summary-section {
            background: #e6f3ff;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            border-left: 4px solid #4CAF50;
        }
        
        .transcript-section {
            background: #f0f8ff;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            border-left: 4px solid #2196F3;
            max-height: 400px;
            overflow-y: auto;
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
            <p>Extract • Transcribe • Summarize with AI</p>
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
                    <input type="checkbox" id="generateSummary" name="generate_summary" checked>
                    <label for="generateSummary">🤖 Generate AI Summary with Gemini</label>
                </div>
            </div>
            
            <button type="submit" class="process-btn" id="processBtn">
                🚀 Process Video
            </button>
        </form>
        
        <div id="loading" class="loading" style="display: none;">
            <div class="spinner"></div>
            <p>Processing video... This includes transcription and may take several minutes.</p>
        </div>
        
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('processForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const data = {
                url: formData.get('url'),
                generate_summary: formData.has('generate_summary')
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
            
            let summarySection = '';
            if (data.summary) {
                summarySection = `
                    <div class="summary-section">
                        <h4>🤖 AI Summary</h4>
                        <div style="white-space: pre-wrap; line-height: 1.6;">${data.summary}</div>
                    </div>
                `;
            }
            
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
                            <strong>Author</strong>
                            ${data.author || 'Unknown'}
                        </div>
                        <div class="info-item">
                            <strong>Processing Time</strong>
                            ${data.processing_time || 'N/A'}
                        </div>
                        <div class="info-item">
                            <strong>Status</strong>
                            Full transcription & analysis completed
                        </div>
                    </div>
                    
                    ${summarySection}
                    
                    <div class="transcript-section">
                        <h4>📝 Full Transcript</h4>
                        <div style="white-space: pre-wrap; line-height: 1.6; font-size: 14px;">${data.subtitle || 'No transcript available'}</div>
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
    </script>
</body>
</html>
    """
    )


@app.post("/process", response_model=ProcessingResponse)
async def process_youtube_video(request: YouTubeRequest):
    """Process YouTube video using the existing youtube_loader infrastructure."""
    start_time = datetime.now()
    logs = []

    try:
        logs.append(f"🎬 Processing: {request.url}")
        logs.append(f"🤖 Summary generation: {'Enabled' if request.generate_summary else 'Disabled'}")

        # Use the existing youtube_loader function
        logs.append("🔄 Starting full YouTube processing pipeline...")
        youtube_content = youtube_loader(request.url)

        # Extract metadata from the content
        metadata = extract_video_metadata(youtube_content)

        # Generate summary if requested
        summary = None
        if request.generate_summary:
            logs.append("🤖 Generating AI summary with Gemini 2.5-pro...")
            summary = quick_summary(youtube_content)
            logs.append("✅ Summary generation completed")

        # Calculate processing time
        processing_time = datetime.now() - start_time

        result_data = {
            "title": metadata["title"],
            "author": metadata["author"],
            "subtitle": metadata["subtitle"],
            "summary": summary,
            "full_content": youtube_content,
            "processing_time": f"{processing_time.total_seconds():.1f}s",
            "url": request.url,
            "timestamp": datetime.now().isoformat(),
        }

        logs.append(f"✅ Processing completed in {processing_time.total_seconds():.1f}s")

        return ProcessingResponse(status="success", message="Video processed successfully with full transcription and summarization", data=result_data, logs=logs)

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

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
