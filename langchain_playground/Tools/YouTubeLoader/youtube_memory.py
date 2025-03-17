import io
import json
import subprocess
from typing import Literal, Tuple

from dotenv import load_dotenv
from pydub import AudioSegment
from pytubefix import Buffer, YouTube

from .llm_formatter import llm_format_txt
from .utils import response_to_txt
from .Whisper import whisper_transcribe

load_dotenv()


# YouTube video processing functions


def get_audio_bytes(youtube: YouTube) -> bytes:
    """
    Get audio bytes from YouTube object.
    Args:
        youtube (YouTube): YouTube object
    Returns:
        bytes: Audio bytes
    """
    # Get audio stream
    youtube_stream = youtube.streams.get_audio_only()

    # Download audio bytes (MP4/DASH container) from stream
    buffer = Buffer()
    buffer.download_in_buffer(youtube_stream)
    audio_data = buffer.read()

    # Use AudioSegment to load from memory buffer
    with io.BytesIO(audio_data) as in_memory_file:
        audio_segment: AudioSegment = AudioSegment.from_file(in_memory_file)

    # Export audio bytes (MP3 format with ID3 metadata)
    with io.BytesIO() as output_buffer:
        audio_segment.export(output_buffer, format="mp3")
        return output_buffer.getvalue()


def url_to_subtitles(youtube: YouTube, whisper_model: str) -> str:
    """Process a YouTube video: download audio and handle subtitles."""
    try:
        audio_bytes = get_audio_bytes(youtube)
        response = whisper_transcribe(audio_bytes, whisper_model)
        subtitle = response_to_txt(response)
        formatted_content = llm_format_txt(subtitle)
        print(f"Formatted TXT: {youtube.title}")
        return formatted_content

    except Exception as e:
        error_message = f"Error processing video {youtube.title}: {str(e)}"
        print(error_message)
        return error_message


# Main function


# npm install -g npm@11.1.0
def po_token_verifier() -> Tuple[str, str]:
    """Get YouTube authentication tokens using node.js generator and return as tuple."""
    result = subprocess.run(
        [
            "node",
            "-e",
            "const{generate}=require('youtube-po-token-generator');generate().then(t=>console.log(JSON.stringify(t)));",
        ],
        capture_output=True,
        text=True,
    )
    tokens = json.loads(result.stdout)
    return tokens["visitorData"], tokens["poToken"]


def youtubeloader(url: str, whisper_model: Literal["fal", "replicate", "hf"] = "fal") -> str:
    """Load and process a YouTube video's subtitles, title, and author information from a URL. Accepts various YouTube URL formats including standard watch URLs and shortened youtu.be links.

    Args:
        url (str): The YouTube video URL to load

    Returns:
        str: Formatted string containing the video title, author and subtitles
    """
    yt = YouTube(
        url,
        use_po_token=True,
        po_token_verifier=po_token_verifier,
    )
    content = [
        "Answer the user's question based on the full content.",
        f"Title: {yt.title}",
        f"Author: {yt.author}",
        "Subtitles:",
        url_to_subtitles(yt, whisper_model),
    ]
    return "\n".join(content)
