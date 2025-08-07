"""
Chromium-based cookie extraction for serverless YouTube processing.

This module uses @sparticuz/chromium with Playwright to extract YouTube cookies
in serverless environments like Vercel where yt-dlp's --cookies-from-browser doesn't work.
"""

import asyncio
import os
import random
from pathlib import Path
from typing import Dict, List

import yt_dlp
from playwright.async_api import async_playwright


# Detect if running in serverless environment
def is_serverless_env() -> bool:
    """Detect if running in a serverless environment."""
    return any(
        [
            os.getenv("VERCEL"),
            os.getenv("AWS_LAMBDA_FUNCTION_NAME"),
            os.getenv("FUNCTIONS_WORKER_RUNTIME"),  # Azure Functions
            os.getenv("GOOGLE_CLOUD_PROJECT"),  # Google Cloud
            "/tmp" in os.getcwd(),  # Common in serverless
        ]
    )


async def extract_youtube_cookies_with_chromium() -> List[Dict]:
    """
    Extract YouTube cookies using headless Chromium in serverless environment.

    This function:
    1. Launches headless Chromium using @sparticuz/chromium (via Playwright)
    2. Navigates to YouTube to establish a session
    3. Extracts cookies that can be used with yt-dlp
    4. Returns cookies in a format compatible with yt-dlp

    Returns:
        List[Dict]: Cookies in browser format for yt-dlp
    """
    print("🚀 Starting Chromium-based YouTube cookie extraction...")

    # User agents for anti-detection
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    selected_user_agent = random.choice(user_agents)

    try:
        async with async_playwright() as p:
            # Configure Chromium for serverless environment
            browser_args = [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-accelerated-2d-canvas",
                "--no-first-run",
                "--no-zygote",
                "--single-process",
                "--disable-gpu",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-features=TranslateUI",
                "--disable-ipc-flooding-protection",
                "--disable-web-security",
                "--disable-features=VizDisplayCompositor",
                f"--user-agent={selected_user_agent}",
            ]

            # Launch browser with serverless-optimized configuration
            if is_serverless_env():
                print("🌐 Detected serverless environment - using optimized Chromium config")
                browser_args.extend(
                    [
                        "--memory-pressure-off",
                        "--max_old_space_size=512",
                        "--disable-background-networking",
                        "--disable-background-sync",
                        "--disable-client-side-phishing-detection",
                        "--disable-component-extensions-with-background-pages",
                        "--disable-default-apps",
                        "--disable-extensions",
                        "--disable-hang-monitor",
                        "--disable-popup-blocking",
                        "--disable-prompt-on-repost",
                        "--disable-sync",
                        "--metrics-recording-only",
                        "--no-default-browser-check",
                        "--safebrowsing-disable-auto-update",
                    ]
                )

            # Use @sparticuz/chromium path in serverless environments
            executable_path = None
            if is_serverless_env():
                try:
                    # Try to use @sparticuz/chromium if available
                    import subprocess

                    result = subprocess.run(["node", "-e", "console.log(require('@sparticuz/chromium').executablePath)"], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0 and result.stdout.strip():
                        executable_path = result.stdout.strip()
                        print(f"📦 Using @sparticuz/chromium: {executable_path}")
                except Exception as e:
                    print(f"⚠️  Could not load @sparticuz/chromium: {e}")
                    print("🔄 Falling back to system Chromium")

            browser = await p.chromium.launch(
                headless=True,
                executable_path=executable_path,
                args=browser_args,
                timeout=60000,  # 60 second timeout
            )

            print("✅ Chromium browser launched successfully")

            # Create new context with realistic settings
            context = await browser.new_context(
                user_agent=selected_user_agent,
                viewport={"width": 1920, "height": 1080},
                device_scale_factor=1,
                is_mobile=False,
                has_touch=False,
                locale="en-US",
                timezone_id="America/New_York",
                permissions=["geolocation"],
                extra_http_headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                },
            )

            page = await context.new_page()

            # Navigate to YouTube homepage to establish session
            print("🎬 Navigating to YouTube...")
            await page.goto("https://www.youtube.com", wait_until="networkidle", timeout=30000)

            # Wait for page to fully load
            await page.wait_for_timeout(3000)

            # Handle potential consent dialogs
            try:
                # Look for various consent/cookie buttons
                consent_selectors = [
                    'button[aria-label*="Accept"]',
                    'button[aria-label*="consent"]',
                    'button:has-text("Accept all")',
                    'button:has-text("I agree")',
                    'button:has-text("Accept")',
                    '[data-testid="accept-button"]',
                    "#acceptButton",
                ]

                for selector in consent_selectors:
                    try:
                        consent_button = await page.wait_for_selector(selector, timeout=2000)
                        if consent_button:
                            await consent_button.click()
                            print("✅ Clicked consent dialog")
                            await page.wait_for_timeout(2000)
                            break
                    except:
                        continue

            except Exception as e:
                print(f"ℹ️  No consent dialog found or error handling it: {e}")

            # Wait a bit more for any additional loading
            await page.wait_for_timeout(2000)

            # Extract cookies from the browser context
            print("🍪 Extracting cookies from browser context...")
            cookies = await context.cookies()

            # Filter for YouTube-relevant cookies
            youtube_cookies = []
            relevant_cookie_names = {"VISITOR_INFO1_LIVE", "YSC", "CONSENT", "PREF", "GPS", "SAPISID", "APISID", "HSID", "SSID", "SID", "__Secure-1PAPISID", "__Secure-3PAPISID", "__Secure-1PSID", "__Secure-3PSID", "__Secure-1PSIDCC", "__Secure-3PSIDCC", "LOGIN_INFO", "SIDCC", "SESSION_TOKEN"}

            for cookie in cookies:
                # Include all YouTube/Google domain cookies and specifically named ones
                if cookie["domain"] in [".youtube.com", ".google.com", "youtube.com", "google.com"] or cookie["name"] in relevant_cookie_names:
                    youtube_cookies.append(
                        {
                            "name": cookie["name"],
                            "value": cookie["value"],
                            "domain": cookie["domain"],
                            "path": cookie["path"],
                            "secure": cookie.get("secure", False),
                            "httpOnly": cookie.get("httpOnly", False),
                            "sameSite": cookie.get("sameSite", "Lax"),
                        }
                    )

            print(f"✅ Extracted {len(youtube_cookies)} YouTube-relevant cookies")

            # Close browser
            await context.close()
            await browser.close()

            return youtube_cookies

    except Exception as e:
        print(f"❌ Chromium cookie extraction failed: {e}")
        raise RuntimeError(f"Failed to extract cookies with Chromium: {e}")


def cookies_to_netscape_format(cookies: List[Dict]) -> str:
    """Convert browser cookies to Netscape format for yt-dlp."""
    if not cookies:
        return ""

    header = "# Netscape HTTP Cookie File\n# Generated by Chromium-based cookie extraction\n\n"

    cookie_lines = []
    for cookie in cookies:
        # Format: domain flag path secure expiration name value
        domain = cookie.get("domain", ".youtube.com")
        if not domain.startswith("."):
            domain = "." + domain

        line = "\t".join([domain, "TRUE", cookie.get("path", "/"), "TRUE" if cookie.get("secure", True) else "FALSE", "0", cookie.get("name", ""), cookie.get("value", "")])  # domain flag  # expiration (0 = session cookie)
        cookie_lines.append(line)

    return header + "\n".join(cookie_lines)


async def extract_video_with_chromium_cookies(url: str) -> Dict:
    """
    Extract YouTube video info using Chromium-extracted cookies.

    This is the main function that combines cookie extraction with yt-dlp processing.

    Args:
        url (str): YouTube video URL

    Returns:
        Dict: Video information extracted with authenticated cookies
    """
    print(f"🔄 Starting Chromium-based YouTube extraction for: {url}")

    try:
        # Step 1: Extract cookies using Chromium
        cookies = await extract_youtube_cookies_with_chromium()

        if not cookies:
            raise RuntimeError("No cookies extracted from Chromium browser")

        print(f"🍪 Using {len(cookies)} extracted cookies for yt-dlp")

        # Step 2: Convert cookies to Netscape format
        netscape_cookies = cookies_to_netscape_format(cookies)

        # Step 3: Create temporary cookie file
        cookie_file_path = f"/tmp/chromium_youtube_cookies_{random.randint(1000, 9999)}.txt"

        try:
            # Write cookies to file
            with open(cookie_file_path, "w") as f:
                f.write(netscape_cookies)

            print(f"📝 Wrote cookies to temporary file: {cookie_file_path}")

            # Step 4: Use yt-dlp with extracted cookies
            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            ]

            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": False,
                "cookiefile": cookie_file_path,
                "user_agent": random.choice(user_agents),
                "referer": "https://www.youtube.com/",
                "headers": {
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "DNT": "1",
                    "Upgrade-Insecure-Requests": "1",
                },
                "sleep_interval": random.uniform(1, 3),
                "max_sleep_interval": 5,
                "writesubtitles": False,
                "writeautomaticsub": False,
                "socket_timeout": 30,
            }

            print("🎬 Extracting video info with Chromium cookies...")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            print("✅ Chromium-based extraction successful!")
            return info

        finally:
            # Clean up cookie file
            try:
                Path(cookie_file_path).unlink()
                print("🧹 Cleaned up temporary cookie file")
            except:
                pass

    except Exception as e:
        print(f"❌ Chromium-based extraction failed: {e}")
        raise RuntimeError(f"Failed to extract video with Chromium cookies: {e}")


# Synchronous wrapper for use in existing codebase
def extract_video_with_chromium_cookies_sync(url: str) -> Dict:
    """Synchronous wrapper for Chromium-based extraction."""
    try:
        return asyncio.run(extract_video_with_chromium_cookies(url))
    except Exception as e:
        print(f"❌ Sync wrapper failed: {e}")
        raise


if __name__ == "__main__":
    # Test the implementation
    import sys

    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    else:
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll for testing

    print(f"🧪 Testing Chromium extraction with: {test_url}")

    try:
        result = extract_video_with_chromium_cookies_sync(test_url)
        print(f"✅ Success! Title: {result.get('title', 'Unknown')}")
        print(f"📊 Duration: {result.get('duration', 'Unknown')} seconds")
        print(f"👤 Uploader: {result.get('uploader', 'Unknown')}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
