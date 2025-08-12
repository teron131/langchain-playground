# Stage 2: YouTube processing without browser automation
import os
import random
import requests
import yt_dlp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google.genai import Client, types
from opencc import OpenCC
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="YouTube Summarizer - Stage 2", description="YouTube processing without browser automation")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# User agents for anti-detection
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]


# Pydantic models
class YouTubeRequest(BaseModel):
    url: str


class SummaryResponse(BaseModel):
    summary: str
    title: str
    author: str
    url: str
    extraction_method: str = "simple_yt_dlp"


# Utility functions
def quick_summary(text: str) -> str:
    """Generate summary using Gemini API."""
    try:
        client = Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-05-20",
            contents=f"Summarize with list out of the key facts mentioned\n\n{text}",
            config=types.GenerateContentConfig(temperature=0),
        )
        return response.text
    except Exception as e:
        print(f"Summary generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


def extract_video_simple(url: str) -> dict:
    """Simple YouTube video extraction using yt-dlp without cookies."""
    print(f"🔄 Starting simple YouTube extraction for: {url}")

    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "user_agent": random.choice(user_agents),
            "socket_timeout": 30,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        print("✅ Simple extraction successful!")
        return info

    except Exception as e:
        print(f"❌ Simple extraction failed: {e}")
        raise RuntimeError(f"YouTube extraction failed: {e}")


# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "YouTube Summarizer Stage 2 is running", "features": ["youtube_processing", "gemini_ai", "simple_extraction"], "port": os.getenv("PORT", "unknown")}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Enhanced HTML interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YouTube Summarizer - Stage 2</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 8px; }
            input[type="url"] { width: 100%; padding: 10px; margin: 10px 0; border: 1px solid #ddd; border-radius: 4px; }
            button { background: #007cba; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; margin: 5px; }
            button:hover { background: #005a8c; }
            button:disabled { background: #ccc; cursor: not-allowed; }
            .result { margin-top: 20px; padding: 15px; background: white; border-radius: 4px; border: 1px solid #ddd; }
            .error { background: #ffe6e6; color: #d00; }
            .success { background: #e6ffe6; color: #060; }
            .stage-badge { 
                background: #28a745; 
                color: white; 
                padding: 5px 15px; 
                border-radius: 20px; 
                font-size: 14px; 
                font-weight: bold; 
            }
            .method-info { 
                background: #e1f5fe; 
                padding: 15px; 
                border-radius: 8px; 
                margin: 20px 0;
                border-left: 4px solid #0277bd;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 YouTube Summarizer <span class="stage-badge">Stage 2</span></h1>
            <p>Testing YouTube processing with container deployment</p>
            
            <div class="method-info">
                <h3>🔧 Current Stage: YouTube Processing</h3>
                <p><strong>✅ Working:</strong> Basic FastAPI deployment, YouTube extraction with yt-dlp, Gemini AI summarization</p>
                <p><strong>🔄 Next:</strong> Add browser automation for cookie extraction (Stage 3)</p>
                <p><strong>🎯 Goal:</strong> Solve "Sign in to confirm you're not a bot" errors</p>
            </div>
            
            <input type="url" id="urlInput" placeholder="https://youtube.com/watch?v=..." style="font-size: 16px; padding: 15px;" />
            
            <div style="text-align: center; margin: 25px 0;">
                <button onclick="processVideo()" id="processBtn" style="background: #28a745; font-size: 18px; padding: 15px 30px;">
                    🔄 Process YouTube Video
                </button>
            </div>
            
            <div id="result"></div>
        </div>

        <script>
            function showResult(content, isError = false) {
                const resultDiv = document.getElementById('result');
                
                if (isError) {
                    resultDiv.innerHTML = '<div class="result error">❌ ' + content + '</div>';
                } else {
                    resultDiv.innerHTML = '<div class="result success">' +
                        '<h3>📺 ' + content.title + '</h3>' +
                        '<p><strong>👤 Channel:</strong> ' + content.author + '</p>' +
                        '<p><strong>🔗 URL:</strong> <a href="' + content.url + '" target="_blank">' + content.url + '</a></p>' +
                        '<p><strong>🔧 Method:</strong> ' + content.extraction_method + '</p>' +
                        '<h4>📝 Summary:</h4>' +
                        '<div style="white-space: pre-wrap;">' + content.summary + '</div>' +
                        '</div>';
                }
            }
            
            async function processVideo() {
                const url = document.getElementById('urlInput').value;
                const processBtn = document.getElementById('processBtn');
                
                if (!url.trim()) {
                    showResult('Please enter a YouTube URL', true);
                    return;
                }
                
                processBtn.disabled = true;
                processBtn.innerHTML = '⏳ Processing...';
                
                try {
                    const response = await fetch('/api/summarize-simple', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ url: url }),
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        showResult(data, false);
                    } else {
                        throw new Error(data.detail || 'Processing failed');
                    }
                    
                } catch (error) {
                    showResult('Processing failed: ' + error.message, true);
                } finally {
                    processBtn.disabled = false;
                    processBtn.innerHTML = '🔄 Process YouTube Video';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/summarize-simple", response_model=SummaryResponse)
async def summarize_simple(request: YouTubeRequest):
    """Process YouTube video using simple yt-dlp extraction."""

    try:
        print(f"🚀 Processing YouTube URL: {request.url}")

        # Simple yt-dlp extraction
        info = extract_video_simple(request.url)

        # Process the video info
        title = info.get("title", "Unknown")
        author = info.get("uploader", "Unknown")
        duration = info.get("duration", "Unknown")
        description = info.get("description", "No description available")

        print(f"✅ Video accessible:")
        print(f"   📺 Title: {title}")
        print(f"   👤 Author: {author}")
        print(f"   ⏱️  Duration: {duration}s")

        # Extract subtitle content
        subtitle_text = ""

        # Check for manual subtitles first
        subtitles = info.get("subtitles", {})
        if subtitles:
            print(f"Available manual subtitles: {list(subtitles.keys())}")
            for lang in ["en", "zh-HK", "zh-CN"]:
                if lang in subtitles and subtitles[lang]:
                    print(f"Found manual subtitle in {lang}")
                    subtitle_url = subtitles[lang][0]["url"]
                    try:
                        response = requests.get(subtitle_url)
                        subtitle_text = response.text
                        print("✅ Downloaded subtitle text")
                        break
                    except Exception as e:
                        print(f"Failed to download subtitle: {e}")

        # If no subtitles, use description
        if not subtitle_text:
            print("ℹ️  No accessible subtitles - using video description")
            subtitle_text = f"Video Description: {description[:2000]}"

        # Format content for summarization
        youtube_content = "\n".join(
            [
                "YouTube Video Content:",
                f"Title: {title}",
                f"Author: {author}",
                f"Duration: {duration} seconds",
                f"Content:\n{subtitle_text[:3000]}",
            ]
        )

        print("✅ YouTube content extracted successfully")

        # Generate summary
        summary = quick_summary(youtube_content)
        print("✅ Summary generated successfully")

        return SummaryResponse(summary=summary, title=title, author=author, url=request.url, extraction_method="simple_yt_dlp")

    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
