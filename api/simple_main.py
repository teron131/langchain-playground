# Minimal FastAPI app for Render.com testing
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Simple FastAPI app without complex dependencies
app = FastAPI(title="YouTube Summarizer - Simple", description="Simplified version for Render.com testing")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class YouTubeRequest(BaseModel):
    url: str


class SummaryResponse(BaseModel):
    summary: str
    title: str
    author: str
    url: str
    extraction_method: str = "simple"


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "YouTube Summarizer API is running", "port": os.getenv("PORT", "unknown")}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple HTML interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YouTube Summarizer - Simple Test</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .container { background: #f5f5f5; padding: 20px; border-radius: 8px; }
            .status { background: #e6ffe6; color: #060; padding: 10px; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 YouTube Summarizer - Test Version</h1>
            <div class="status">
                ✅ <strong>Render.com deployment successful!</strong><br>
                🔧 This is a simplified version to test the deployment.<br>
                📊 Once this works, we'll add back the YouTube processing features.
            </div>
            
            <h2>Service Status</h2>
            <p><strong>Health Endpoint:</strong> <a href="/health">/health</a></p>
            <p><strong>Environment:</strong> Render.com Container</p>
            <p><strong>Purpose:</strong> Test deployment before adding complex features</p>
            
            <h2>Next Steps</h2>
            <ol>
                <li>✅ Basic FastAPI deployment working</li>
                <li>🔄 Add back YouTube processing</li>
                <li>🔄 Add browser automation</li>
                <li>🔄 Test YouTube cookie extraction</li>
            </ol>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/test")
async def test_endpoint(request: YouTubeRequest):
    """Simple test endpoint."""
    return {"status": "success", "message": "Basic API working", "url": request.url, "next_step": "Add YouTube processing functionality"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
