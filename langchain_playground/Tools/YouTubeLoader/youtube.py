import io
from typing import Literal

from dotenv import load_dotenv
from pydub import AudioSegment
from pytubefix import Buffer, YouTube

from .llm_formatter import llm_format
from .utils import po_token_verifier, result_to_txt
from .Whisper import whisper_fal

load_dotenv()


# YouTube video processing functions
def youtube_to_audio_bytes(youtube: YouTube) -> bytes:
    """Get audio bytes from YouTube object.
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

    # Use AudioSegment to decode from memory buffer
    with io.BytesIO(audio_data) as in_memory_file:
        audio_segment: AudioSegment = AudioSegment.from_file(in_memory_file)

    # Export audio bytes (MP3 format with ID3 metadata)
    with io.BytesIO() as output_buffer:
        audio_segment.export(output_buffer, format="mp3", bitrate="16k")
        return output_buffer.getvalue()


def youtube_to_subtitle(youtube: YouTube, language: str = None) -> str:
    """Process a YouTube video: download audio and handle subtitle."""
    audio_bytes = youtube_to_audio_bytes(youtube)
    result = whisper_fal(audio_bytes, language)
    subtitle = result_to_txt(result)
    formatted_subtitle = llm_format(subtitle, audio_bytes)
    return formatted_subtitle


# Main function
def youtube_loader(url: str) -> str:
    """Load and process a YouTube video's subtitle, title, and author information from a URL. Accepts various YouTube URL formats including standard watch URLs and shortened youtu.be links.

    Args:
        url (str): The YouTube video URL to load

    Returns:
        str: Formatted string containing the video title, author and subtitle
    """
    youtube = YouTube(
        url,
        use_po_token=True,
        po_token_verifier=po_token_verifier,
    )

    formatted_subtitle = youtube_to_subtitle(youtube, language=None)

    content = [
        "Answer the user's question based on the full content.",
        f"Title: {youtube.title}",
        f"Author: {youtube.author}",
        f"subtitle:\n{formatted_subtitle}",
    ]
    return "\n".join(content)
