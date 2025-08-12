"""
Standalone YouTube Processing API - No External Package Dependencies
All functions copied directly to avoid import chain issues
"""

import asyncio
import io
import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

import fal_client
import requests
import yt_dlp
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from google.genai import Client, types
from opencc import OpenCC
from pydantic import BaseModel, Field
from pydub import AudioSegment

load_dotenv()

# Configure logging for Railway stdout visibility
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


def log_and_print(message: str):
    """Log and print message to ensure visibility in Railway."""
    print(message, flush=True)
    logger.info(message)
    sys.stdout.flush()


# Initialize FastAPI
app = FastAPI(title="YouTube Summarizer", description="Standalone YouTube processing with transcription & summarization")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=2)


# Request Models
class YouTubeRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    generate_summary: bool = Field(default=True, description="Generate AI summary")


class ProcessingResponse(BaseModel):
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    logs: List[str] = []


# Utility Functions (copied from utils.py)
def s2hk(content: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese."""
    return OpenCC("s2hk").convert(content)


def whisper_result_to_txt(result: dict) -> str:
    """Convert Whisper transcription result to plain text."""
    txt_content = "\n".join(chunk["text"].strip() for chunk in result["chunks"])
    return s2hk(txt_content)


def parse_youtube_json_captions(json_content: str) -> str:
    """Parse YouTube's JSON timedtext format and extract plain text."""
    try:
        data = json.loads(json_content)
        text_parts = []
        if "events" in data:
            for event in data["events"]:
                if "segs" in event:
                    for seg in event["segs"]:
                        if "utf8" in seg:
                            text_parts.append(seg["utf8"])
        full_text = "".join(text_parts)
        return full_text.strip()
    except (json.JSONDecodeError, KeyError, TypeError):
        return json_content


def srt_to_txt(srt_content: str) -> str:
    """Convert SRT format content to plain text."""
    lines = []
    for line in srt_content.splitlines():
        line = line.strip()
        if line and not line.isdigit() and "-->" not in line:
            lines.append(line)
    return s2hk("\n".join(lines))


# YouTube Processing Functions (copied from youtube.py)
def extract_video_info(url: str) -> dict:
    """Extract video information using yt-dlp with Railway-optimized strategies."""
    # Railway cloud environment detection
    is_cloud_env = any(
        [
            os.getenv("RAILWAY_STATIC_URL"),
            os.getenv("RAILWAY_PROJECT_ID"),
            os.getenv("PORT") and not os.getenv("DEVELOPMENT"),
            os.getenv("RENDER"),
            "/tmp" in os.getcwd(),
        ]
    )

    if is_cloud_env:
        log_and_print("🌐 Detected cloud environment - using optimized strategies")

    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Linux; Android 10; SM-T870) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Safari/537.36",
    ]

    # Enhanced strategies to avoid 403 errors
    strategies = [
        # Strategy 1: Android TV with specific params
        {
            "user_agent": "com.google.android.youtube/19.09.37 (Linux; U; Android 11) gzip",
            "extractor_args": {"youtube": {"player_client": ["android_tv"], "player_skip": ["configs"], "skip": ["hls", "dash"]}},
        },
        # Strategy 2: Android mobile client
        {
            "user_agent": "com.google.android.youtube/19.09.37 (Linux; U; Android 11) gzip",
            "extractor_args": {
                "youtube": {
                    "player_client": ["android"],
                    "player_skip": ["configs"],
                }
            },
        },
        # Strategy 3: iOS client
        {
            "user_agent": "com.google.ios.youtube/19.09.3 (iPhone14,3; U; CPU iOS 15_6 like Mac OS X)",
            "extractor_args": {
                "youtube": {
                    "player_client": ["ios"],
                    "player_skip": ["configs"],
                }
            },
        },
        # Strategy 4: Android embedded
        {
            "user_agent": "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Mobile Safari/537.36",
            "extractor_args": {
                "youtube": {
                    "player_client": ["android_embedded"],
                    "player_skip": ["configs", "webpage"],
                }
            },
        },
        # Strategy 5: Web fallback with age bypass
        {
            "user_agent": random.choice(user_agents),
            "extractor_args": {
                "youtube": {
                    "player_client": ["web"],
                    "player_skip": ["configs"],
                }
            },
            "age_limit": 999,  # Bypass age restrictions
        },
    ]

    last_error = None

    for i, strategy in enumerate(strategies):
        log_and_print(f"Trying strategy {i + 1}/{len(strategies)}...")

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "user_agent": strategy["user_agent"],
            "referer": "https://www.youtube.com/",
            "extractor_args": strategy["extractor_args"],
            "socket_timeout": 30,
            "http_headers": {
                "User-Agent": strategy["user_agent"],
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-us,en;q=0.5",
                "Sec-Fetch-Mode": "navigate",
            },
        }

        # Add age_limit if specified in strategy
        if "age_limit" in strategy:
            ydl_opts["age_limit"] = strategy["age_limit"]

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                log_and_print(f"✅ Strategy {i + 1} succeeded!")
                return info
        except Exception as e:
            error_msg = str(e)
            last_error = e
            log_and_print(f"❌ Strategy {i + 1} failed: {error_msg}")

            if any(keyword in error_msg.lower() for keyword in ["private video", "video unavailable"]):
                break
            continue

    if last_error:
        raise RuntimeError(f"Failed to access video: {str(last_error)}")
    else:
        raise RuntimeError("Unknown error occurred while processing YouTube video.")


def download_audio_bytes(info: dict) -> bytes:
    """Download audio from YouTube video info with optimized format selection for size."""
    formats = info.get("formats", [])

    # Categorize formats by efficiency (smaller size, good quality)
    high_efficiency_formats = []  # Opus, AAC - best compression
    medium_efficiency_formats = []  # MP4A, other modern codecs
    fallback_formats = []  # Older/larger formats

    for fmt in formats:
        if fmt.get("vcodec") == "none" and fmt.get("acodec") != "none":
            # Audio-only format - preferred
            acodec = fmt.get("acodec", "").lower()
            filesize = fmt.get("filesize", 0) or fmt.get("filesize_approx", 0) or 0

            # Prioritize by codec efficiency for transcription
            if any(codec in acodec for codec in ["opus"]):
                # Opus is most efficient for speech
                high_efficiency_formats.append((fmt, filesize))
            elif any(codec in acodec for codec in ["aac", "mp4a"]):
                # AAC is good balance of quality/size
                medium_efficiency_formats.append((fmt, filesize))
            else:
                # Other audio codecs
                fallback_formats.append((fmt, filesize))
        elif fmt.get("acodec") != "none":
            # Has audio track (video+audio) - lower priority
            filesize = fmt.get("filesize", 0) or fmt.get("filesize_approx", 0) or 0
            fallback_formats.append((fmt, filesize))

    # Sort each category by filesize (prefer smaller)
    high_efficiency_formats.sort(key=lambda x: x[1] if x[1] > 0 else float("inf"))
    medium_efficiency_formats.sort(key=lambda x: x[1] if x[1] > 0 else float("inf"))
    fallback_formats.sort(key=lambda x: x[1] if x[1] > 0 else float("inf"))

    # Try formats in order of efficiency and size
    audio_format = None
    selected_from = "unknown"

    for fmt_list, category in [(high_efficiency_formats, "high_efficiency"), (medium_efficiency_formats, "medium_efficiency"), (fallback_formats, "fallback")]:
        if fmt_list:
            # Select the smallest format from each category
            audio_format, size = fmt_list[0]
            selected_from = category
            break

    # If no suitable format found, try any available format
    if not audio_format:
        all_formats = [fmt for fmt_list in [high_efficiency_formats, medium_efficiency_formats, fallback_formats] for fmt, _ in fmt_list]
        if all_formats:
            audio_format = all_formats[0]  # Take first available
            selected_from = "any_available"
        else:
            raise RuntimeError("No audio format available")

    # Log format details for debugging
    filesize_mb = (audio_format.get("filesize") or audio_format.get("filesize_approx") or 0) / 1024 / 1024
    log_and_print(f"Selected format: {audio_format.get('format_id')} ({selected_from}) - {audio_format.get('acodec')} - {filesize_mb:.1f}MB estimated")

    # Download audio with better error handling
    audio_url = audio_format["url"]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36", "Referer": "https://www.youtube.com/"}

    try:
        response = requests.get(audio_url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        if "403" in str(e):
            raise RuntimeError(f"YouTube blocked audio download (HTTP 403). Try again later or use a different video.")
        else:
            raise RuntimeError(f"Failed to download audio: {e}")

    # Download in chunks with progress
    audio_data = b""
    for chunk in response.iter_content(chunk_size=32768):
        if chunk:
            audio_data += chunk

    log_and_print(f"Downloaded {len(audio_data)} bytes of audio")

    # Skip conversion entirely - use raw audio for transcription
    log_and_print("🔄 Skipping FFmpeg conversion, using raw audio for FAL")

    # Log file size info but don't impose limits
    audio_size_mb = len(audio_data) / 1024 / 1024
    log_and_print(f"✅ Using raw audio: {len(audio_data)} bytes ({audio_size_mb:.1f}MB)")
    return audio_data


def get_subtitle_from_captions(info: dict) -> str:
    """Get existing subtitles from video info."""
    subtitles = info.get("subtitles", {})

    for lang in ["zh-HK", "zh-CN", "en"]:
        if lang in subtitles:
            subtitle_info = subtitles[lang][0]
            response = requests.get(subtitle_info["url"])
            raw_content = response.text

            if lang == "zh-CN":
                raw_content = s2hk(raw_content)

            if raw_content.strip().startswith("{"):
                return parse_youtube_json_captions(raw_content)
            else:
                return srt_to_txt(raw_content)

    return None


def optimize_audio_for_transcription(audio_bytes: bytes, target_size_mb: int = 25) -> bytes:
    """Optimize audio for transcription with intelligent compression strategies."""
    raw_size_mb = len(audio_bytes) / 1024 / 1024
    log_and_print(f"🎵 Audio size check: {raw_size_mb:.1f}MB")

    # If audio is reasonably sized, use as-is
    if raw_size_mb <= target_size_mb:
        log_and_print(f"✅ Using raw audio for transcription ({raw_size_mb:.1f}MB)")
        return audio_bytes

    log_and_print(f"🔄 Audio size is {raw_size_mb:.1f}MB, applying intelligent compression...")

    try:
        # Try to load audio with pydub for compression
        audio_io = io.BytesIO(audio_bytes)

        # Try different audio formats in order of likelihood
        audio_segment = None
        for fmt in ["mp3", "mp4", "m4a", "webm", "ogg"]:
            try:
                audio_io.seek(0)
                audio_segment = AudioSegment.from_file(audio_io, format=fmt)
                log_and_print(f"✅ Successfully loaded audio as {fmt}")
                break
            except Exception as e:
                log_and_print(f"⚠️ Failed to load as {fmt}: {str(e)[:100]}")
                continue

        if not audio_segment:
            # Fallback: try without specifying format
            try:
                audio_io.seek(0)
                audio_segment = AudioSegment.from_file(audio_io)
                log_and_print("✅ Successfully loaded audio with auto-detection")
            except Exception as e:
                log_and_print(f"❌ Failed to load audio for compression: {e}")
                # If compression fails, return raw audio
                log_and_print(f"⚠️ Using raw audio despite size ({raw_size_mb:.1f}MB)")
                return audio_bytes

        # Calculate duration for intelligent bitrate selection
        duration_minutes = len(audio_segment) / 60000  # Convert ms to minutes
        log_and_print(f"📊 Audio duration: {duration_minutes:.1f} minutes")

        # Dynamic compression strategies based on file size and duration
        compression_strategies = []

        if raw_size_mb > 100:  # Very large files
            compression_strategies.extend(
                [
                    {"bitrate": "24k", "channels": 1, "frame_rate": 16000, "description": "Aggressive (24kbps, mono, 16kHz)"},
                    {"bitrate": "32k", "channels": 1, "frame_rate": 22050, "description": "High compression (32kbps, mono, 22kHz)"},
                ]
            )
        elif raw_size_mb > 50:  # Large files
            compression_strategies.extend(
                [
                    {"bitrate": "32k", "channels": 1, "frame_rate": 22050, "description": "High compression (32kbps, mono, 22kHz)"},
                    {"bitrate": "48k", "channels": 1, "frame_rate": None, "description": "Mono + 48kbps"},
                ]
            )
        else:  # Medium files
            compression_strategies.extend(
                [
                    {"bitrate": "48k", "channels": 1, "frame_rate": None, "description": "Mono + 48kbps"},
                    {"bitrate": "64k", "channels": None, "frame_rate": None, "description": "Lower bitrate (64kbps)"},
                ]
            )

        # Always add fallback strategies
        compression_strategies.extend(
            [
                {"bitrate": "32k", "channels": 1, "frame_rate": 22050, "description": "Fallback compression"},
                {"bitrate": "24k", "channels": 1, "frame_rate": 16000, "description": "Last resort compression"},
            ]
        )

        for strategy in compression_strategies:
            try:
                log_and_print(f"🔄 Trying compression: {strategy['description']}")

                # Apply transformations
                compressed_audio = audio_segment

                # Convert to mono if specified
                if strategy["channels"] == 1:
                    compressed_audio = compressed_audio.set_channels(1)

                # Change sample rate if specified
                if strategy["frame_rate"]:
                    compressed_audio = compressed_audio.set_frame_rate(strategy["frame_rate"])

                # Export with specified bitrate
                output_buffer = io.BytesIO()
                compressed_audio.export(output_buffer, format="mp3", bitrate=strategy["bitrate"], parameters=["-ac", "1"] if strategy["channels"] == 1 else [])
                compressed_bytes = output_buffer.getvalue()
                compressed_size_mb = len(compressed_bytes) / 1024 / 1024

                log_and_print(f"📊 Compressed to {compressed_size_mb:.1f}MB (reduction: {((raw_size_mb - compressed_size_mb) / raw_size_mb * 100):.1f}%)")

                # Check if compression is good enough (achieved reasonable reduction)
                if compressed_size_mb <= target_size_mb or compressed_size_mb < raw_size_mb * 0.7:
                    log_and_print(f"✅ Compression successful with {strategy['description']}")
                    return compressed_bytes

            except Exception as e:
                log_and_print(f"⚠️ Compression strategy failed: {str(e)[:100]}")
                continue

        # If all compression strategies failed, try basic MP3 export
        log_and_print("🔄 All compression strategies failed, trying basic MP3 export...")
        try:
            output_buffer = io.BytesIO()
            audio_segment.export(output_buffer, format="mp3", bitrate="32k")
            fallback_bytes = output_buffer.getvalue()
            fallback_size_mb = len(fallback_bytes) / 1024 / 1024

            log_and_print(f"✅ Fallback compression: {fallback_size_mb:.1f}MB")
            return fallback_bytes
        except Exception as e:
            log_and_print(f"❌ Fallback compression failed: {e}")

        # Last resort: use raw audio
        log_and_print(f"⚠️ Using raw audio as last resort ({raw_size_mb:.1f}MB)")
        return audio_bytes

    except Exception as e:
        error_msg = str(e)
        log_and_print(f"❌ Audio optimization error: {error_msg}")
        # Return raw audio if compression fails
        log_and_print(f"⚠️ Using raw audio after optimization failure ({raw_size_mb:.1f}MB)")
        return audio_bytes


def transcribe_with_fal(audio_bytes: bytes) -> str:
    """Transcribe audio using FAL API - matches original whisper_fal.py."""
    try:
        log_and_print("🎤 Starting FAL transcription...")

        # Check API key
        fal_key = os.getenv("FAL_KEY")
        if not fal_key:
            return "[FAL_KEY not configured]"

        # Upload audio with better content type detection
        log_and_print("📤 Uploading audio to FAL...")

        # Detect content type based on magic bytes
        content_type = "audio/mpeg"  # Default to generic audio
        if audio_bytes.startswith(b"ID3") or audio_bytes.startswith(b"\xff\xfb"):
            content_type = "audio/mpeg"  # MP3
        elif audio_bytes.startswith(b"fLaC"):
            content_type = "audio/flac"
        elif audio_bytes.startswith(b"OggS"):
            content_type = "audio/ogg"
        elif audio_bytes.startswith(b"\x00\x00\x00\x18ftypmp4") or audio_bytes.startswith(b"\x00\x00\x00\x20ftypM4A"):
            content_type = "audio/mp4"
        elif audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:12]:
            content_type = "audio/wav"
        elif audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):  # WebM/Matroska
            content_type = "audio/webm"

        log_and_print(f"🔍 Detected format: {content_type}")

        try:
            url = fal_client.upload(data=audio_bytes, content_type=content_type)
            log_and_print(f"✅ Upload successful to FAL")
        except Exception as upload_error:
            log_and_print(f"❌ Upload failed: {upload_error}")
            # Try with generic audio type as fallback
            log_and_print("🔄 Retrying upload with generic audio type...")
            url = fal_client.upload(data=audio_bytes, content_type="audio/mpeg")
            log_and_print(f"✅ Upload successful (fallback)")

        # Transcribe using original FAL API structure (no timeout parameter)
        log_and_print("🔄 Starting transcription...")

        def on_queue_update(update):
            """Handle FAL queue updates."""
            if isinstance(update, fal_client.InProgress):
                for log_entry in update.logs:
                    log_and_print(f"FAL: {log_entry['message']}")

        try:
            result = fal_client.subscribe(
                "fal-ai/whisper",
                arguments={
                    "audio_url": url,
                    "task": "transcribe",
                    "language": None,  # Auto-detect language
                },
                with_logs=True,
                on_queue_update=on_queue_update,
            )

            log_and_print("✅ Transcription completed")
            return whisper_result_to_txt(result)

        except Exception as transcribe_error:
            error_msg = str(transcribe_error)
            log_and_print(f"❌ FAL transcription failed: {error_msg}")

            # Check for specific error types
            if "403" in error_msg or "forbidden" in error_msg.lower():
                return "[FAL API access denied (403). Check API key permissions.]"
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                return "[FAL API quota exceeded]"
            elif "timeout" in error_msg.lower():
                return "[FAL API timeout - audio may be too long]"
            else:
                return f"[FAL transcription failed: {error_msg}]"

    except Exception as e:
        error_msg = str(e)
        log_and_print(f"❌ General transcription error: {error_msg}")
        return f"[Transcription error: {error_msg}]"


def process_youtube_video_sync(url: str, generate_summary: bool = True) -> dict:
    """Synchronous processing function to run in thread pool."""
    try:
        print(f"\n🎬 Processing: {url}")

        # Extract video info with timeout
        info = extract_video_info(url)
        title = info.get("title", "Unknown")
        author = info.get("uploader", "Unknown")

        print(f"✅ Video: {title} by {author}")

        # Try captions first (fast)
        subtitle = get_subtitle_from_captions(info)

        if not subtitle:
            print("🎯 No captions found, transcribing audio...")
            try:
                audio_bytes = download_audio_bytes(info)

                # Always attempt transcription with optimization
                subtitle = transcribe_with_fal(audio_bytes)
            except Exception as e:
                subtitle = f"[Audio processing failed: {str(e)}]"
        else:
            print("✅ Using existing captions")

        formatted_subtitle = simple_format_subtitle(subtitle)

        # Generate summary if requested and subtitle is valid
        summary = None
        if generate_summary and not subtitle.startswith("["):
            try:
                print("🤖 Generating summary...")
                full_content = f"Title: {title}\nAuthor: {author}\nTranscript:\n{formatted_subtitle}"
                summary = quick_summary(full_content)
            except Exception as e:
                summary = f"[Summary generation failed: {str(e)}]"

        return {"title": title, "author": author, "subtitle": formatted_subtitle, "summary": summary, "url": url, "status": "success"}

    except Exception as e:
        return {"title": "Unknown", "author": "Unknown", "subtitle": f"[Processing failed: {str(e)}]", "summary": None, "url": url, "status": "error", "error": str(e)}


def simple_format_subtitle(subtitle: str) -> str:
    """Simple subtitle formatting without complex LLM chains."""
    # Basic formatting - add periods, capitalize
    lines = subtitle.split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if line:
            # Capitalize first letter
            if line[0].islower():
                line = line[0].upper() + line[1:]
            # Add period if missing
            if not line.endswith((".", "!", "?")):
                line += "."
            formatted_lines.append(line)

    return "\n".join(formatted_lines)


def quick_summary(text: str) -> str:
    """Generate summary using Gemini."""
    try:
        client = Client(api_key=os.getenv("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=f"Summarize with list out of the key facts mentioned. Follow the language of the text.\n\n{text}",
            config=types.GenerateContentConfig(temperature=0),
        )
        return response.text
    except Exception as e:
        return f"Summary generation failed: {str(e)}"


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
    <title>YouTube Summarizer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        .container {
            background: white; border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px; max-width: 1000px; margin: 0 auto;
        }
        .header { text-align: center; margin-bottom: 40px; }
        .header h1 { color: #333; font-size: 2.5rem; margin-bottom: 10px; }
        .header p { color: #666; font-size: 1.1rem; }
        .form-group { margin-bottom: 25px; }
        label { display: block; margin-bottom: 8px; color: #333; font-weight: 600; }
        input[type="url"] {
            width: 100%; padding: 15px; border: 2px solid #e1e5e9;
            border-radius: 12px; font-size: 16px;
        }
        input[type="url"]:focus { outline: none; border-color: #667eea; }
        .options { margin-bottom: 30px; }
        .checkbox-group { display: flex; align-items: center; gap: 8px; }
        input[type="checkbox"] { width: 20px; height: 20px; }
        .process-btn {
            width: 100%; padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border: none; border-radius: 12px;
            font-size: 18px; font-weight: 600; cursor: pointer;
        }
        .process-btn:hover { transform: translateY(-2px); }
        .process-btn:disabled { opacity: 0.6; cursor: not-allowed; }
        .loading { text-align: center; margin: 30px 0; display: none; }
        .spinner {
            border: 4px solid #f3f3f3; border-top: 4px solid #667eea;
            border-radius: 50%; width: 50px; height: 50px;
            animation: spin 1s linear infinite; margin: 0 auto 20px;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .result { background: #f8f9fa; border-radius: 12px; padding: 25px; margin-top: 30px; }
        .result h3 { color: #333; margin-bottom: 15px; font-size: 1.3rem; }
        .info-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin-bottom: 20px;
        }
        .info-item {
            background: white; padding: 15px; border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .info-item strong { display: block; color: #333; margin-bottom: 5px; }
        .summary-section {
            background: #e6f3ff; border-radius: 12px; padding: 20px;
            margin: 20px 0; border-left: 4px solid #4CAF50;
        }
        .transcript-section {
            background: #f0f8ff; border-radius: 12px; padding: 20px;
            margin: 20px 0; border-left: 4px solid #2196F3;
            max-height: 400px; overflow-y: auto;
        }
        .error { background: #fed7d7; color: #c53030; padding: 15px; border-radius: 8px; margin-top: 20px; }
        .success { background: #c6f6d5; color: #2f855a; padding: 15px; border-radius: 8px; margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 YouTube Summarizer</h1>
            <p>Standalone • Extract • Transcribe • Summarize</p>
        </div>
        
        <form id="processForm">
            <div class="form-group">
                <label for="url">YouTube URL</label>
                <input type="url" id="url" name="url" placeholder="https://www.youtube.com/watch?v=..." required>
            </div>
            
            <div class="options">
                <div class="checkbox-group">
                    <input type="checkbox" id="generateSummary" name="generate_summary" checked>
                    <label for="generateSummary">🤖 Generate AI Summary</label>
                </div>
            </div>
            
            <button type="submit" class="process-btn" id="processBtn">🚀 Process Video</button>
        </form>
        
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p>Processing video... This may take several minutes.</p>
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
            
            loadingDiv.style.display = 'block';
            resultDiv.innerHTML = '';
            processBtn.disabled = true;
            processBtn.textContent = '⏳ Processing...';
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
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
                <div class="success"><strong>✅ Processing completed!</strong></div>
                <div class="result">
                    <h3>📺 Video Information</h3>
                    <div class="info-grid">
                        <div class="info-item"><strong>Title</strong>${data.title}</div>
                        <div class="info-item"><strong>Author</strong>${data.author}</div>
                        <div class="info-item"><strong>Processing Time</strong>${data.processing_time}</div>
                    </div>
                    ${summarySection}
                    <div class="transcript-section">
                        <h4>📝 Transcript</h4>
                        <div style="white-space: pre-wrap; line-height: 1.6;">${data.subtitle}</div>
                    </div>
                </div>
            `;
        }
        
        function showError(message) {
            document.getElementById('result').innerHTML = 
                `<div class="error"><strong>❌ Error:</strong> ${message}</div>`;
        }
    </script>
</body>
</html>
    """
    )


@app.post("/process", response_model=ProcessingResponse)
async def process_youtube_video(request: YouTubeRequest):
    """Process YouTube video with robust error handling and visible logging."""
    start_time = datetime.now()
    logs = [f"🎬 Starting processing: {request.url}"]

    try:
        log_and_print("📋 Step 1: Extracting video info...")
        logs.append("📋 Step 1: Extracting video info...")

        # Extract video info
        info = extract_video_info(request.url)
        title = info.get("title", "Unknown")
        author = info.get("uploader", "Unknown")

        log_and_print(f"✅ Video found: {title} by {author}")
        logs.append(f"✅ Video found: {title} by {author}")

        # Try captions first (fast and reliable)
        log_and_print("📋 Step 2: Checking for existing captions...")
        logs.append("📋 Step 2: Checking for existing captions...")
        subtitle = get_subtitle_from_captions(info)

        if subtitle:
            log_and_print("✅ Found existing captions - skipping transcription")
            logs.append("✅ Found existing captions - skipping transcription")
            formatted_subtitle = simple_format_subtitle(subtitle)
        else:
            log_and_print("🎯 No captions found")
            logs.append("🎯 No captions found")

            try:
                log_and_print("📋 Step 3: Downloading audio...")
                logs.append("📋 Step 3: Downloading audio...")
                audio_bytes = download_audio_bytes(info)

                # Log audio size info
                audio_size_mb = len(audio_bytes) / 1024 / 1024
                log_and_print(f"📊 Audio size: {audio_size_mb:.1f}MB")
                logs.append(f"📊 Audio size: {audio_size_mb:.1f}MB")

                log_and_print("📋 Step 4: Starting audio optimization...")
                logs.append("📋 Step 4: Starting audio optimization...")

                try:
                    # Optimize audio before transcription
                    optimized_audio = optimize_audio_for_transcription(audio_bytes)
                    optimized_size_mb = len(optimized_audio) / 1024 / 1024
                    log_and_print(f"🎵 Optimized to {optimized_size_mb:.1f}MB")
                    logs.append(f"🎵 Optimized to {optimized_size_mb:.1f}MB")

                    # Attempt transcription
                    try:
                        # Check FAL_KEY before attempting transcription
                        if not os.getenv("FAL_KEY"):
                            log_and_print("❌ FAL_KEY not configured")
                            logs.append("❌ FAL_KEY not configured")
                            formatted_subtitle = "[FAL_KEY not configured]"
                        else:
                            log_and_print("📋 Step 5: Starting FAL transcription...")
                            logs.append("📋 Step 5: Starting FAL transcription...")

                            # Transcribe the optimized audio
                            subtitle = transcribe_with_fal(optimized_audio)
                            formatted_subtitle = simple_format_subtitle(subtitle)
                            log_and_print("✅ Transcription completed")
                            logs.append("✅ Transcription completed")
                    except Exception as fal_error:
                        log_and_print(f"❌ FAL transcription failed: {fal_error}")
                        logs.append(f"❌ FAL transcription failed: {fal_error}")
                        formatted_subtitle = f"[Transcription failed: {str(fal_error)}]"

                except Exception as opt_error:
                    log_and_print(f"❌ Audio optimization failed: {opt_error}")
                    logs.append(f"❌ Audio optimization failed: {opt_error}")
                    formatted_subtitle = f"[Audio optimization failed: {str(opt_error)}]"

            except Exception as audio_error:
                error_msg = f"❌ Audio processing failed: {str(audio_error)}"
                log_and_print(error_msg)
                logs.append(error_msg)
                formatted_subtitle = f"[Audio processing failed: {str(audio_error)}]"

        # Generate summary if requested
        summary = None
        if request.generate_summary and not formatted_subtitle.startswith("["):
            try:
                log_and_print("📋 Step 5: Generating summary...")
                logs.append("📋 Step 5: Generating summary...")
                full_content = f"Title: {title}\nAuthor: {author}\nTranscript:\n{formatted_subtitle}"
                summary = quick_summary(full_content)
                log_and_print("✅ Summary generated")
                logs.append("✅ Summary generated")
            except Exception as summary_error:
                error_msg = f"❌ Summary generation failed: {str(summary_error)}"
                log_and_print(error_msg)
                logs.append(error_msg)
                summary = f"[Summary generation failed: {str(summary_error)}]"

        processing_time = datetime.now() - start_time
        completion_msg = f"✅ Processing completed in {processing_time.total_seconds():.1f}s"
        log_and_print(completion_msg)
        logs.append(completion_msg)

        result_data = {
            "title": title,
            "author": author,
            "subtitle": formatted_subtitle,
            "summary": summary,
            "processing_time": f"{processing_time.total_seconds():.1f}s",
            "url": request.url,
        }

        return ProcessingResponse(status="success", message="Video processed successfully", data=result_data, logs=logs)

    except Exception as e:
        processing_time = datetime.now() - start_time
        error_message = f"Processing error: {str(e)}"
        failure_msg = f"💔 Failed after {processing_time.total_seconds():.1f}s"

        log_and_print(f"❌ {error_message}")
        log_and_print(failure_msg)
        logs.append(f"❌ {error_message}")
        logs.append(failure_msg)

        return ProcessingResponse(status="error", message=error_message, logs=logs)


# Add a simple health check endpoint for debugging
@app.get("/test")
async def test_endpoint():
    """Simple test endpoint to verify the service is working."""
    return {"status": "healthy", "message": "Service is running", "timestamp": datetime.now().isoformat(), "memory_usage": "OK"}


# For Railway deployment
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
