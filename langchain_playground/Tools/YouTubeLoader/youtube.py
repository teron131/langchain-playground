import io

import requests
import yt_dlp
from dotenv import load_dotenv
from pydub import AudioSegment

from langchain_playground.Tools.YouTubeLoader.llm_formatter import llm_format
from langchain_playground.Tools.YouTubeLoader.utils import (
    parse_youtube_json_captions,
    s2hk,
    srt_to_txt,
    whisper_result_to_txt,
)
from langchain_playground.Tools.YouTubeLoader.Whisper import whisper_fal

load_dotenv()


def extract_video_info(url: str) -> dict:
    """Extract video information using yt-dlp with optimized container-friendly strategies."""
    import os
    import random
    import time

    # Detect if running in cloud environment (Vercel, etc.)
    is_cloud_env = any(
        [
            os.getenv("VERCEL"),
            os.getenv("AWS_LAMBDA_FUNCTION_NAME"),
            os.getenv("FUNCTIONS_WORKER_RUNTIME"),  # Azure Functions
            os.getenv("GOOGLE_CLOUD_PROJECT"),  # Google Cloud
            os.getenv("RENDER"),  # Render.com
            "/tmp" in os.getcwd(),  # Common in serverless
        ]
    )

    if is_cloud_env:
        print("🌐 Detected cloud environment - using optimized container strategies")

    # Rotate through different user agents to avoid detection patterns
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Linux; Android 10; SM-T870) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    # Optimized strategies based on container testing results
    if is_cloud_env:
        # Container-optimized strategies (prioritize proven working methods)
        strategies = [
            # Strategy 1: Android TV client (proven to work in containers)
            {
                "user_agent": "Mozilla/5.0 (Linux; Android 10; SM-T870) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Safari/537.36",
                "extractor_args": {"youtube": {"player_client": ["android_tv"]}},
            },
            # Strategy 2: Android client (reliable backup)
            {
                "user_agent": random.choice(user_agents),
                "extractor_args": {"youtube": {"player_client": ["android"]}},
            },
            # Strategy 3: Android embedded (alternative method)
            {
                "user_agent": random.choice(user_agents),
                "extractor_args": {"youtube": {"player_client": ["android_embedded"]}},
            },
            # Strategy 4: Web client with Android user agent
            {
                "user_agent": "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.91 Mobile Safari/537.36",
                "extractor_args": {"youtube": {"player_client": ["web"]}},
            },
            # Strategy 5: Multiple client fallback
            {
                "user_agent": random.choice(user_agents),
                "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            },
        ]
    else:
        # Local environment strategies (include browser cookies for local development)
        strategies = [
            # Strategy 1: Try with cookies from Chrome (local environments)
            {
                "cookies_from_browser": ["chrome"],
                "user_agent": random.choice(user_agents),
                "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            },
            # Strategy 2: Try with cookies from Firefox (local environments)
            {
                "cookies_from_browser": ["firefox"],
                "user_agent": random.choice(user_agents),
                "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            },
            # Strategy 3: Android TV client (proven to work)
            {
                "user_agent": "Mozilla/5.0 (Linux; Android 10; SM-T870) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Safari/537.36",
                "extractor_args": {"youtube": {"player_client": ["android_tv"]}},
            },
            # Strategy 4: Android client
            {
                "user_agent": random.choice(user_agents),
                "extractor_args": {"youtube": {"player_client": ["android"]}},
            },
            # Strategy 5: Web client fallback
            {
                "user_agent": random.choice(user_agents),
                "extractor_args": {"youtube": {"player_client": ["web"]}},
            },
        ]

    last_error = None

    for i, strategy in enumerate(strategies):
        print(f"Trying strategy {i + 1}/{len(strategies)}...")

        # Build yt-dlp options with current strategy
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "user_agent": strategy["user_agent"],
            "referer": "https://www.youtube.com/",
            "headers": {
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
                "Accept-Encoding": "gzip, deflate, br",
                "Cache-Control": "max-age=0",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "DNT": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "sec-ch-ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Linux"',
                "X-Forwarded-For": f"{random.randint(100, 199)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
            },
            # Optimized rate limiting
            "sleep_interval": random.uniform(0.5, 1.5),
            "max_sleep_interval": 4,
            "writesubtitles": False,
            "writeautomaticsub": False,
            # Container-optimized options
            "prefer_insecure": False,
            "no_check_certificate": False,
            "geo_bypass": True,
            "socket_timeout": 30,
        }

        # Add container-specific optimizations for strategy 2+
        if is_cloud_env and i >= 1:
            ydl_opts.update(
                {
                    "extractor_retries": 2,
                    "fragment_retries": 3,
                    "retry_sleep": "linear",
                    "force_json": True,
                    "no_color": True,
                    "youtube_include_dash_manifest": False,
                    "mark_watched": False,
                }
            )

        # Add strategy-specific options
        if "cookies_from_browser" in strategy:
            ydl_opts["cookiesfrombrowser"] = strategy["cookies_from_browser"]
            print(f"Using cookies from: {strategy['cookies_from_browser']}")

        if "extractor_args" in strategy:
            ydl_opts["extractor_args"] = strategy["extractor_args"]
            print(f"Using extractor args: {strategy['extractor_args']}")

        # Reduced delays for faster processing
        if i > 0:
            delay = random.uniform(1, 2)  # Much shorter delays
            print(f"Waiting {delay:.1f}s before retry...")
            time.sleep(delay)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                print(f"✅ Strategy {i + 1} succeeded!")
                return info

        except Exception as e:
            error_msg = str(e)
            last_error = e
            print(f"❌ Strategy {i + 1} failed: {error_msg}")

            # If this is a non-bot-detection error, fail fast
            if any(keyword in error_msg.lower() for keyword in ["private video", "video unavailable", "age-restricted", "not available on this app", "watch on the latest version"]):
                print(f"Non-retriable error detected, stopping attempts")
                break

            # Continue to next strategy for bot detection and other retriable errors
            continue

    # All strategies failed, raise the final error
    if last_error:
        error_msg = str(last_error)
        print(f"All strategies failed. Final error: {error_msg}")

        # Enhanced error handling with more helpful messages
        if "Sign in to confirm you're not a bot" in error_msg or "--cookies-from-browser" in error_msg:
            print(f"⚠️ YouTube bot detection triggered")
            raise RuntimeError("YouTube is blocking automated requests. This is often temporary. Try again in a few minutes, or use a video that's more publicly accessible.")
        elif "not available on this app" in error_msg or "Watch on the latest version of YouTube" in error_msg:
            print(f"⚠️ YouTube app restriction")
            raise RuntimeError("This video is restricted by YouTube and cannot be processed through third-party applications. Please try a different video.")
        elif "Private video" in error_msg:
            raise RuntimeError("This video is private and cannot be accessed. Please use a public video.")
        elif "Video unavailable" in error_msg:
            raise RuntimeError("This video is unavailable or has been removed from YouTube.")
        elif "age-restricted" in error_msg.lower():
            raise RuntimeError("This video is age-restricted and cannot be processed without authentication.")
        else:
            raise RuntimeError(f"Failed to access video after trying multiple methods: {error_msg}")
    else:
        raise RuntimeError("Unknown error occurred while processing YouTube video.")


def download_audio_from_url(audio_url: str) -> bytes:
    """Download audio from URL and return as bytes with progress logging."""
    try:
        print(f"🌐 Starting download from: {audio_url[:50]}...")
        response = requests.get(audio_url, stream=True)
        response.raise_for_status()

        # Get file size if available
        total_size = response.headers.get("content-length")
        if total_size:
            total_size = int(total_size)
            print(f"📊 File size: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
        else:
            print("📊 File size: Unknown")

        # Download with progress tracking
        downloaded = 0
        chunks = []
        chunk_size = 8192  # 8KB chunks

        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                chunks.append(chunk)
                downloaded += len(chunk)

                # Show progress every MB
                if downloaded % (1024 * 1024) < chunk_size:
                    if total_size:
                        progress = (downloaded / total_size) * 100
                        print(f"⬇️  Downloaded: {downloaded:,} bytes ({progress:.1f}%)")
                    else:
                        print(f"⬇️  Downloaded: {downloaded:,} bytes")

        audio_data = b"".join(chunks)
        print(f"✅ Download complete: {len(audio_data):,} bytes")
        return audio_data

    except Exception as e:
        print(f"❌ Download failed: {e}")
        raise RuntimeError(f"Failed to download audio: {e}")


def download_audio_bytes(info: dict) -> bytes:
    """Download audio from YouTube video info and convert to MP3 bytes."""
    # Get best audio format
    formats = info.get("formats", [])
    audio_format = None

    # Look for audio-only formats first
    for fmt in formats:
        if fmt.get("vcodec") == "none" and fmt.get("acodec") != "none":
            audio_format = fmt
            break

    # Fallback to any format with audio
    if not audio_format:
        for fmt in formats:
            if fmt.get("acodec") != "none":
                audio_format = fmt
                break

    if not audio_format:
        raise RuntimeError("No audio format available")

    # Download audio from URL
    audio_url = audio_format["url"]
    audio_data = download_audio_from_url(audio_url)

    if not audio_data:
        raise RuntimeError("Downloaded audio data is empty")

    # Try to process with AudioSegment and convert to MP3
    try:
        with io.BytesIO(audio_data) as in_memory_file:
            try:
                # Try to detect format from audio_format info
                format_name = audio_format.get("ext", "mp4")
                audio_segment = AudioSegment.from_file(in_memory_file, format=format_name)
            except Exception:
                # Fallback: let AudioSegment auto-detect
                in_memory_file.seek(0)
                audio_segment = AudioSegment.from_file(in_memory_file)

        # Export to MP3 bytes
        with io.BytesIO() as output_buffer:
            audio_segment.export(output_buffer, format="mp3", bitrate="32k", parameters=["-ac", "1"])
            return output_buffer.getvalue()

    except Exception as e:
        print(f"AudioSegment conversion failed: {e}")
        print("Falling back to raw audio bytes (Fal API can handle various formats)")
        # Return raw audio bytes - Fal API can handle various audio formats
        return audio_data


def get_subtitle_from_captions(info: dict) -> str:
    """Try to get existing subtitle from yt-dlp extracted info."""
    # Check for manual subtitles only (skip automatic captions)
    subtitles = info.get("subtitles", {})
    print(f"Available manual subtitles: {list(subtitles.keys())}")

    # Priority languages for manual subtitles only
    for lang in ["zh-HK", "zh-CN", "en"]:
        if lang in subtitles:
            print(f"Found manual subtitle in {lang}")
            subtitle_info = subtitles[lang][0]  # Get first subtitle format

            # Download subtitle content
            subtitle_url = subtitle_info["url"]
            print(f"🔗 Downloading subtitle from: {subtitle_url[:50]}...")
            response = requests.get(subtitle_url)
            raw_content = response.text

            # Convert zh-CN to zh-HK if needed
            if lang == "zh-CN":
                raw_content = s2hk(raw_content)
                print("Converted zh-CN to zh-HK")

            # Detect format and convert to plain text
            print("🔄 Converting to plain text...")
            if raw_content.strip().startswith("{"):
                # JSON format from YouTube timedtext API
                print("📋 Detected JSON format, parsing...")
                plain_text = parse_youtube_json_captions(raw_content)
            else:
                # SRT format
                print("📋 Detected SRT format, converting...")
                plain_text = srt_to_txt(raw_content)

            print(f"📝 Converted text (first 200 chars): {plain_text[:200]}...")

            return plain_text

    # Skip automatic captions - they're often unreliable
    # Only show relevant automatic caption languages to avoid overwhelming output
    automatic_captions = info.get("automatic_captions", {})
    relevant_auto_langs = [lang for lang in ["en", "en-orig", "zh-Hans", "zh-Hant"] if lang in automatic_captions]
    print(f"Available automatic captions (relevant): {relevant_auto_langs} (skipping - will transcribe instead)")

    print("No suitable manual captions found, will transcribe audio")
    return None


def youtube_loader(url: str) -> str:
    """Load and process a YouTube video's subtitle, title, and author information.

    Process:
    1. Extract video info using yt-dlp
    2. Check for existing captions in [zh-HK, zh-CN, en]
    3. If no captions, download audio and transcribe with Fal
    4. Format subtitle with Gemini LLM
    5. Return formatted content

    Args:
        url (str): The YouTube video URL to load

    Returns:
        str: Formatted string containing the video title, author and subtitle
    """
    print(f"\n🎬 Loading YouTube video: {url}")
    print("=" * 50)

    # Extract video information using yt-dlp
    print("📋 Extracting video information...")
    info = extract_video_info(url)

    title = info.get("title", "Unknown")
    author = info.get("uploader", "Unknown")
    duration = info.get("duration", "Unknown")

    print(f"✅ Video accessible:")
    print(f"   📺 Title: {title}")
    print(f"   👤 Author: {author}")
    print(f"   ⏱️  Duration: {duration}s")

    # Try to get existing subtitle first
    subtitle = get_subtitle_from_captions(info)
    audio_bytes = None

    if not subtitle:
        print("🎯 No suitable captions found, transcribing audio...")

        # Download audio
        print("📥 Downloading audio...")
        audio_bytes = download_audio_bytes(info)
        print(f"Downloaded {len(audio_bytes)} bytes of audio")

        # Detect language for transcription
        automatic_captions = info.get("automatic_captions", {})
        language = "en" if "a.en" in automatic_captions or "en" in automatic_captions else "zh"
        print(f"Transcribing in language: {language}")

        # Transcribe with Fal
        print("🎤 Transcribing with Fal API...")
        result = whisper_fal(audio_bytes, language)
        subtitle = whisper_result_to_txt(result)
        print("✅ Transcription completed")
    else:
        print("✅ Using existing captions")
        # Still need audio for LLM formatting
        print("📥 Downloading audio for LLM formatting...")
        audio_bytes = download_audio_bytes(info)

    # Format subtitle with LLM
    print("🤖 Formatting subtitle with Gemini...")
    formatted_subtitle = llm_format(subtitle, audio_bytes)
    print("✅ Subtitle formatted")

    # Return formatted content
    content = [
        "Answer the user's question based on the full content.",
        f"Title: {title}",
        f"Author: {author}",
        f"subtitle:\n{formatted_subtitle}",
    ]

    print("✅ SUCCESS: Video processed successfully!")
    return "\n".join(content)


if __name__ == "__main__":
    url = "https://youtu.be/6Nn4MJYmv4A"
    print(youtube_loader(url))
