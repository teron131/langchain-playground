import asyncio
import json
import os
import random
import re
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, cast

import fal_client
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

# Try to check if pyppeteer is available for Chromium-based extraction
CHROMIUM_AVAILABLE = False


def _check_chromium_availability():
    """Check if Chromium extraction is available without importing heavy dependencies."""
    global CHROMIUM_AVAILABLE
    try:
        import importlib
        import sys

        # Check if pyppeteer is available without importing it
        if "pyppeteer" in sys.modules or importlib.util.find_spec("pyppeteer") is not None:
            CHROMIUM_AVAILABLE = True
            print("✅ Pyppeteer available - Chromium extraction enabled")
        else:
            print("⚠️ Pyppeteer not available - Chromium extraction disabled")
            CHROMIUM_AVAILABLE = False
    except Exception as e:
        print(f"⚠️ Error checking Pyppeteer availability: {e}")
        CHROMIUM_AVAILABLE = False


# Check availability on startup
_check_chromium_availability()


# Create FastAPI app
app = FastAPI(title="YouTube Summarizer", description="AI-powered YouTube video summarizer with Chromium-based extraction for serverless environments")


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
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
]


# ================ FULL CHROMIUM EXTRACTION FUNCTIONS (Container optimized) ================
import uuid
from pathlib import Path


def is_container_env() -> bool:
    """Detect if running in a container environment (Render.com, Docker, etc.)."""
    return any(
        [
            os.getenv("RENDER"),
            os.getenv("DOCKER_CONTAINER"),
            Path("/.dockerenv").exists(),
            os.getenv("KUBERNETES_SERVICE_HOST"),
        ]
    )


def get_browser_executable() -> Optional[str]:
    """Get the best available browser executable."""
    # Check for system Chrome/Chromium
    chrome_paths = [
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/opt/google/chrome/chrome",
    ]

    for path in chrome_paths:
        if Path(path).exists():
            print(f"✅ Found Chrome at: {path}")
            return path

    print("⚠️ No system Chrome found, using Pyppeteer bundled version")
    return None


class BrowserSessionManager:
    """Manage persistent browser sessions for better YouTube authentication."""

    def __init__(self):
        self.cache_dir = Path("/app/.cache/browser-profiles") if is_container_env() else Path.cwd() / ".cache" / "browser-profiles"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.active_sessions = {}

    def get_profile_path(self, session_id: str = "default") -> Path:
        """Get browser profile path for session persistence."""
        profile_path = self.cache_dir / f"profile_{session_id}"
        profile_path.mkdir(parents=True, exist_ok=True)
        return profile_path

    async def get_browser_instance(self, session_id: str = "youtube_auth") -> tuple:
        """Get or create a browser instance with persistent profile."""
        try:
            from pyppeteer import launch

            # Use persistent profile for better authentication
            profile_path = self.get_profile_path(session_id)

            # Enhanced launch arguments for container environment
            launch_args = [
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-setuid-sandbox",
                "--disable-accelerated-2d-canvas",
                "--no-first-run",
                "--no-zygote",
                "--disable-gpu",
                "--disable-extensions",
                "--disable-default-apps",
                "--disable-translate",
                "--disable-sync",
                "--hide-scrollbars",
                "--metrics-recording-only",
                "--mute-audio",
                "--no-default-browser-check",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-features=TranslateUI,VizDisplayCompositor",
                "--disable-blink-features=AutomationControlled",
                "--disable-infobars",
                f"--user-data-dir={profile_path}",
            ]

            # Try to use system Chrome if available
            chrome_executable = get_browser_executable()

            launch_kwargs = {
                "headless": True,
                "args": launch_args,
                "ignoreHTTPSErrors": True,
                "defaultViewport": {"width": 1920, "height": 1080},
            }

            if chrome_executable:
                launch_kwargs["executablePath"] = chrome_executable

            browser = await asyncio.wait_for(launch(**launch_kwargs), timeout=20.0)

            # Create a new page with enhanced stealth
            page = await browser.newPage()

            # Enhanced stealth configuration
            await page.setUserAgent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

            # Remove webdriver detection
            await page.evaluateOnNewDocument(
                """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en'],
                });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
            """
            )

            return browser, page

        except Exception as e:
            print(f"❌ Failed to create browser instance: {e}")
            raise


# Initialize global session manager
browser_manager = BrowserSessionManager()


async def extract_youtube_cookies_with_chromium() -> List[dict]:
    """
    Enhanced persistent Chromium cookie extraction for container environments.
    """
    print("🚀 Starting container-optimized Chromium extraction...")

    browser = None
    page = None
    session_id = f"youtube_{uuid.uuid4().hex[:8]}"

    try:
        # Get browser instance with persistent profile
        print("🔧 Creating persistent browser session...")
        browser, page = await browser_manager.get_browser_instance(session_id)
        print("✅ Browser instance created with persistent profile")

        # Set additional headers for authenticity
        await page.setExtraHTTPHeaders(
            {
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Cache-Control": "max-age=0",
            }
        )

        # Navigate to YouTube home page first
        print("🎬 Navigating to YouTube...")
        try:
            await asyncio.wait_for(page.goto("https://www.youtube.com", {"waitUntil": "networkidle2", "timeout": 30000}), timeout=35.0)
            print("✅ Successfully navigated to YouTube")
        except asyncio.TimeoutError:
            print("⚠️ Navigation timeout, trying with domcontentloaded...")
            await asyncio.wait_for(page.goto("https://www.youtube.com", {"waitUntil": "domcontentloaded"}), timeout=20.0)
            print("✅ Navigation completed with domcontentloaded")

        # Handle consent/privacy dialogs
        print("🔒 Handling consent dialogs...")
        try:
            # Wait for consent dialog with multiple selectors
            consent_selectors = [
                'button[aria-label*="Accept"]',
                'button[aria-label*="I agree"]',
                'button:has-text("Accept all")',
                'button:has-text("I agree")',
                '[data-testid*="accept"], [data-test-id*="accept"]',
                '.VfPpkd-LgbsSe[aria-label*="Accept"]',
                'form[action*="consent"] button',
            ]

            for selector in consent_selectors:
                try:
                    consent_element = await asyncio.wait_for(page.waitForSelector(selector, {"timeout": 3000}), timeout=4.0)
                    if consent_element:
                        await consent_element.click()
                        print(f"✅ Clicked consent button: {selector}")
                        await asyncio.sleep(2)
                        break
                except (asyncio.TimeoutError, Exception):
                    continue

        except Exception as e:
            print(f"ℹ️ No consent dialog found or error: {e}")

        # Browse a few videos to establish better session
        print("🎥 Establishing session by browsing sample videos...")
        sample_videos = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",  # Rick Roll
            "https://www.youtube.com/watch?v=kJQP7kiw5Fk",  # Despacito
        ]

        for video_url in sample_videos:
            try:
                await asyncio.wait_for(page.goto(video_url, {"waitUntil": "domcontentloaded"}), timeout=15.0)
                print(f"✅ Browsed: {video_url}")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"⚠️ Failed to browse {video_url}: {e}")
                continue

        # Go back to main page
        await page.goto("https://www.youtube.com")
        await asyncio.sleep(2)

        # Extract all cookies
        print("🍪 Extracting authentication cookies...")
        cookies = await page.cookies()
        print(f"✅ Successfully extracted {len(cookies)} total cookies")

        # Enhanced cookie filtering for authentication
        youtube_cookies = []
        essential_cookies = {
            "VISITOR_INFO1_LIVE",
            "YSC",
            "CONSENT",
            "__Secure-YEC",
            "LOGIN_INFO",
            "PREF",
            "GPS",
            "HSID",
            "APISID",
            "SSID",
            "SID",
            "SAPISID",
            "__Secure-3PAPISID",
            "__Secure-3PSID",
            "__Secure-3PSIDCC",
            "SIDCC",
            "NID",
            "1P_JAR",
            "__Secure-1PAPISID",
            "__Secure-1PSID",
            "__Secure-1PSIDCC",
            "OTZ",
            "AEC",
            "DV",
        }

        for cookie in cookies:
            cookie_domain = cookie.get("domain", "")
            cookie_name = cookie.get("name", "")

            # Include YouTube/Google domain cookies or essential authentication cookies
            if cookie_domain.endswith("youtube.com") or cookie_domain.endswith("google.com") or cookie_domain.endswith("googleusercontent.com") or cookie_domain.endswith("googlevideo.com") or cookie_name in essential_cookies:

                youtube_cookies.append(
                    {
                        "name": cookie_name,
                        "value": cookie.get("value", ""),
                        "domain": cookie_domain,
                        "path": cookie.get("path", "/"),
                        "secure": cookie.get("secure", False),
                        "httpOnly": cookie.get("httpOnly", False),
                        "sameSite": cookie.get("sameSite", "Lax"),
                    }
                )
                print(f"  📌 Found cookie: {cookie_name} from {cookie_domain}")

        print(f"✅ Filtered to {len(youtube_cookies)} YouTube-relevant cookies")

        # Validate essential cookies
        found_essential = [c["name"] for c in youtube_cookies if c["name"] in essential_cookies]
        print(f"🔍 Found essential cookies: {found_essential}")

        if len(youtube_cookies) == 0:
            print("⚠️ No YouTube cookies found - session establishment failed")
            return []

        # Store session for potential reuse
        print(f"💾 Session {session_id} established with {len(youtube_cookies)} cookies")
        return youtube_cookies

    except Exception as e:
        print(f"❌ Error during Chromium extraction: {type(e).__name__}: {str(e)}")
        import traceback

        print(f"📍 Traceback: {traceback.format_exc()}")
        return []

    finally:
        # Clean shutdown
        try:
            if page:
                print("🧹 Closing page...")
                await asyncio.wait_for(page.close(), timeout=5.0)
                print("✅ Page closed")
        except Exception as e:
            print(f"⚠️ Error closing page: {e}")

        try:
            if browser:
                print("🧹 Closing browser...")
                await asyncio.wait_for(browser.close(), timeout=10.0)
                print("✅ Browser closed successfully")
        except Exception as e:
            print(f"⚠️ Error closing browser: {e}")


# ================ PYDANTIC MODELS ================
class YouTubeRequest(BaseModel):
    url: str


class ChromiumExtractionRequest(BaseModel):
    """Request model for Chromium-based cookie extraction"""

    url: str


class SummaryResponse(BaseModel):
    summary: str
    title: str
    author: str
    url: str
    extraction_method: str = "unknown"


# ================ UTILITY FUNCTIONS ================
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


# ================ API ENDPOINTS ================
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the enhanced HTML interface with client-side processing."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YouTube Summarizer - Chromium Enhanced</title>
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
            .method-badge { 
                display: inline-block; 
                padding: 2px 8px; 
                border-radius: 12px; 
                font-size: 12px; 
                font-weight: bold; 
                margin-left: 10px;
            }
            .chromium-badge { background: #6f42c1; color: white; }
            .progress { margin: 10px 0; }
            .step { margin: 5px 0; padding: 5px; border-left: 3px solid #ddd; padding-left: 10px; }
            .step.active { border-left-color: #007cba; background: #f0f8ff; }
            .step.complete { border-left-color: #4CAF50; background: #f0fff0; }
            .step.error { border-left-color: #f44336; background: #fff0f0; }
            .primary-method { 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 25px; 
                border-radius: 15px; 
                margin: 20px 0; 
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                text-align: center;
            }
            .primary-button { 
                background: #28a745 !important; 
                font-size: 20px !important; 
                padding: 18px 40px !important;
                font-weight: bold !important;
                box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4);
                border: none !important;
                border-radius: 10px !important;
                margin: 15px 0 !important;
                transition: all 0.3s ease;
            }
            .primary-button:hover { 
                background: #218838 !important; 
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(40, 167, 69, 0.6);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎬 YouTube Summarizer - Chromium Enhanced</h1>
            <p>Next-generation YouTube processing with automated Chromium-based serverless extraction</p>
            
            <div class="primary-method">
                <h2 style="margin-top: 0; font-size: 28px;">🚀 Container-Optimized Chromium Extraction</h2>
                <p style="font-size: 18px; margin: 20px 0; line-height: 1.6;">
                    <strong>🎯 ENTERPRISE-GRADE METHOD!</strong><br>
                    This launches full Chromium in a persistent container environment 
                    with session management and authentication persistence. <strong>No manual steps required!</strong>
                </p>
                
                <div style="background: rgba(255,255,255,0.2); padding: 20px; border-radius: 10px; margin: 20px 0; text-align: left;">
                    <strong style="font-size: 16px;">✅ Render.com Container Advantages:</strong>
                    <ul style="margin: 15px 0; padding-left: 25px; font-size: 16px;">
                        <li>🔒 Persistent browser sessions with authentication</li>
                        <li>🌐 Stable IP addresses (not flagged by YouTube)</li>
                        <li>💾 Cookie persistence across requests</li>
                        <li>⚡ Full Chromium (not serverless-limited)</li>
                        <li>🛡️ Advanced anti-bot detection evasion</li>
                    </ul>
                </div>
                
                <div style="background: rgba(255,255,255,0.15); padding: 15px; border-radius: 8px; font-size: 14px; text-align: left;">
                    <strong>Container Architecture:</strong> Persistent Docker container → Full Chrome installation → 
                    Session profiles → Enhanced stealth → Authentication cookies → Reliable extraction
                </div>
            </div>
            
            <input type="url" id="urlInput" placeholder="https://youtube.com/watch?v=..." style="font-size: 16px; padding: 15px;" />
            
            <div style="text-align: center; margin: 25px 0;">
                <button onclick="summarizeWithChromium()" id="chromiumBtn" class="primary-button">
                    🚀 Process with Chromium
                </button>
            </div>
            
            <div id="progress"></div>
            <div id="result"></div>
        </div>

        <script>
            function updateProgress(steps) {
                const progressDiv = document.getElementById('progress');
                progressDiv.innerHTML = '<div class="progress">' + 
                    steps.map(step => '<div class="step ' + step.status + '">' + step.text + '</div>').join('') + 
                '</div>';
            }

            function showResult(content, isError = false, method = 'unknown') {
                const resultDiv = document.getElementById('result');
                const methodBadge = '<span class="method-badge chromium-badge">' + method + '</span>';
                
                if (isError) {
                    resultDiv.innerHTML = '<div class="result error">❌ ' + content + methodBadge + '</div>';
                } else {
                    resultDiv.innerHTML = '<div class="result success">' +
                        '<h3>📺 ' + content.title + methodBadge + '</h3>' +
                        '<p><strong>👤 Channel:</strong> ' + content.author + '</p>' +
                        '<p><strong>🔗 URL:</strong> <a href="' + content.url + '" target="_blank">' + content.url + '</a></p>' +
                        '<h4>📝 Summary:</h4>' +
                        '<div style="white-space: pre-wrap;">' + content.summary + '</div>' +
                        '</div>';
                }
            }
            
            async function summarizeWithChromium() {
                const url = document.getElementById('urlInput').value;
                const chromiumBtn = document.getElementById('chromiumBtn');
                
                if (!url.trim()) {
                    showResult('Please enter a YouTube URL', true, 'chromium');
                    return;
                }
                
                chromiumBtn.disabled = true;
                chromiumBtn.innerHTML = '⏳ Processing...';
                
                const steps = [
                    { text: '🚀 Launching headless Chromium browser...', status: 'active' },
                    { text: '🎬 Navigating to YouTube and handling consent...', status: '' },
                    { text: '🍪 Extracting cookies from browser context...', status: '' },
                    { text: '📋 Processing video with extracted cookies...', status: '' },
                    { text: '🤖 Generating summary...', status: '' },
                ];
                
                try {
                    updateProgress(steps);
                    
                    const response = await fetch('/api/summarize-with-chromium', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            url: url
                        }),
                    });
                    
                    steps.forEach((s, i) => {
                        if (i < 4) s.status = 'complete';
                        else s.status = 'active';
                    });
                    updateProgress(steps);
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        steps.forEach(s => s.status = 'complete');
                        updateProgress(steps);
                        showResult(data, false, 'chromium_serverless');
                    } else {
                        throw new Error(data.detail || 'Chromium-based processing failed');
                    }
                    
                } catch (error) {
                    const errorSteps = steps.map(s => ({ ...s, status: s.status === 'active' ? 'error' : s.status }));
                    updateProgress(errorSteps);
                    showResult('Chromium-based processing failed: ' + error.message, true, 'chromium');
                } finally {
                    chromiumBtn.disabled = false;
                    chromiumBtn.innerHTML = '🚀 Process with Chromium';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/summarize-with-chromium", response_model=SummaryResponse)
async def summarize_with_chromium(request: ChromiumExtractionRequest):
    """Process YouTube video using Chromium-based cookie extraction for serverless environments."""

    if not CHROMIUM_AVAILABLE:
        raise HTTPException(status_code=503, detail="Chromium extractor not available. Please install pyppeteer and @sparticuz/chromium.")

    try:
        print(f"🚀 Processing YouTube URL with Chromium-based extraction: {request.url}")

        # Use Chromium-based extraction
        info = await extract_video_with_chromium_cookies(request.url)

        # Process the video info
        title = info.get("title", "Unknown")
        author = info.get("uploader", "Unknown")
        duration = info.get("duration", "Unknown")
        description = info.get("description", "No description available")

        print(f"✅ Video accessible with Chromium cookies:")
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

        print("✅ YouTube content extracted successfully with Chromium")

        # Generate summary
        summary = quick_summary(youtube_content)
        print("✅ Summary generated successfully")

        return SummaryResponse(summary=summary, title=title, author=author, url=request.url, extraction_method="chromium_serverless")

    except Exception as e:
        print(f"❌ Error processing video with Chromium: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing video with Chromium: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "YouTube Summarizer API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
