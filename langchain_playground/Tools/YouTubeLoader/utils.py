import json
import os
import re
import subprocess
import time
from functools import lru_cache
from typing import Optional, Tuple, cast

from dotenv import load_dotenv
from opencc import OpenCC

load_dotenv()

# Cache for tokens with expiration
_token_cache = {"tokens": None, "timestamp": 0, "expires_in": 3600}  # 1 hour


def get_cached_tokens() -> Optional[Tuple[str, str]]:
    """Get cached tokens if they haven't expired."""
    if _token_cache["tokens"] and (time.time() - _token_cache["timestamp"]) < _token_cache["expires_in"]:
        return _token_cache["tokens"]
    return None


def set_cached_tokens(tokens: Tuple[str, str]) -> None:
    """Cache tokens with current timestamp."""
    _token_cache["tokens"] = tokens
    _token_cache["timestamp"] = time.time()


def generate_tokens_from_cli() -> Tuple[str, str]:
    """Generate tokens using the CLI tool."""
    try:
        result = subprocess.run(
            ["youtube-po-token-generator"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            tokens = json.loads(result.stdout.strip())
            return tokens["visitorData"], tokens["poToken"]
        else:
            raise RuntimeError(f"CLI tool failed: {result.stderr}")
    except Exception as e:
        print(f"CLI token generation failed: {e}")
        # Fallback to node.js method
        result = subprocess.run(
            [
                "node",
                "-e",
                "const{generate}=require('youtube-po-token-generator');generate().then(t=>console.log(JSON.stringify(t)));",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        tokens = json.loads(result.stdout)
        return tokens["visitorData"], tokens["poToken"]


def po_token_verifier() -> Tuple[str, str]:
    """Get YouTube authentication tokens with proper caching and fallback strategies."""
    # Check for environment variables first
    visitor_data = os.getenv("YOUTUBE_VISITOR_DATA")
    po_token = os.getenv("YOUTUBE_PO_TOKEN")

    if visitor_data and po_token:
        print("Using tokens from environment variables")
        return visitor_data, po_token

    # Check cache
    cached_tokens = get_cached_tokens()
    if cached_tokens:
        print("Using cached tokens")
        return cached_tokens

    # Generate new tokens
    print("Generating new tokens...")
    try:
        tokens = generate_tokens_from_cli()
        set_cached_tokens(tokens)
        print("Successfully generated and cached new tokens")
        return tokens
    except Exception as e:
        print(f"Token generation failed: {e}")
        # Return empty tokens as fallback - pytubefix will work without them sometimes
        return "", ""


@lru_cache(maxsize=None)
def s2hk(content: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese.

    Args:
        content (str): The content to convert

    Returns:
        str: The converted content
    """
    return OpenCC("s2hk").convert(content)


def convert_time_to_hms(seconds_float: float) -> str:
    """Converts a time in seconds to 'hh:mm:ss,ms' format for SRT.

    Args:
        seconds_float (float): Time in seconds.

    Returns:
        str: Time in 'hh:mm:ss,ms' format.
    """
    hours, remainder = divmod(seconds_float, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"


def whisper_result_to_srt(result: dict) -> str:
    """Convert the specific transcription  API response into SRT format string.

    Args:
        result (Dict): The transcription result from the Whisper model.

    Returns:
        str: SRT formatted string
    """
    srt_entries = []
    for counter, chunk in enumerate(result["chunks"], 1):
        chunk: dict
        timestamp: list[float] = chunk.get("timestamp")
        subtitle: str = chunk.get("text")

        start_time = timestamp[0]
        end_time = timestamp[1] if timestamp[1] is not None else start_time + 2.0

        start_time_hms = convert_time_to_hms(start_time)
        end_time_hms = convert_time_to_hms(end_time)

        subtitle = subtitle.strip()
        srt_entry = f"{counter}\n{start_time_hms} --> {end_time_hms}\n{subtitle}\n\n"
        srt_entries.append(srt_entry)
    srt_content = "".join(srt_entries)
    return s2hk(srt_content)


def whisper_result_to_txt(result: dict) -> str:
    """Convert the specific transcription API response into plain text format.

    Args:
        result (Dict): The transcription result from the Whisper model.

    Returns:
        str: Plain text formatted string with each chunk on a new line
    """
    txt_content = "\n".join(cast(str, chunk["text"]).strip() for chunk in result["chunks"])
    return s2hk(txt_content)


def parse_youtube_json_captions(json_content: str) -> str:
    """Parse YouTube's JSON timedtext format and extract plain text.

    Args:
        json_content: Raw JSON string from YouTube timedtext API

    Returns:
        Plain text without timestamps
    """
    try:
        # Parse the JSON
        data = json.loads(json_content)

        # Extract text from events
        text_parts = []
        if "events" in data:
            for event in data["events"]:
                if "segs" in event:
                    for seg in event["segs"]:
                        if "utf8" in seg:
                            text_parts.append(seg["utf8"])

        # Join all text parts
        full_text = "".join(text_parts)

        # Clean up the text (remove extra whitespace, normalize)
        full_text = re.sub(r"\s+", " ", full_text).strip()

        return full_text

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"⚠️ Failed to parse JSON captions: {e}")
        return json_content  # Return original if parsing fails


def srt_to_txt(srt_content: str) -> str:
    """Convert SRT format content to plain text.

    Args:
        srt_content (str): The SRT formatted content

    Returns:
        str: Plain text with only the transcript lines
    """
    txt_content = "\n".join(line.strip() for line in srt_content.splitlines() if not line.strip().isdigit() and "-->" not in line and line.strip())
    return s2hk(txt_content)
