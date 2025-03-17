import json
import subprocess
from pathlib import Path
from typing import Literal, Tuple

from dotenv import load_dotenv
from pytubefix import YouTube

from .llm_formatter import llm_format_txt
from .utils import response_to_srt, response_to_txt, s2hk, srt_to_txt
from .Whisper import whisper_transcribe

load_dotenv()

BASE_CACHE_DIR = Path(".cache")


def read_file(file_path: Path | str) -> str:
    """Read text from a file with UTF-8 encoding.

    Args:
        file_path (Path | str): Path to the file to read

    Returns:
        str: The file contents
    """
    return Path(file_path).read_text(encoding="utf-8")


def write_file(file_path: Path | str, content: str) -> None:
    """Write text to a file with UTF-8 encoding.

    Args:
        file_path (Path | str): Path to the file to write
        content (str): Content to write to the file
    """
    Path(file_path).write_text(content, encoding="utf-8")


# YouTube video processing functions


def download_audio(youtube: YouTube, cache_dir: Path, output_path: Path) -> None:
    mp3_path = output_path.with_suffix(".mp3")
    if mp3_path.exists():
        print(f"Audio file already exists: {mp3_path}")
        return
    youtube.streams.get_audio_only().download(output_path=str(cache_dir), filename=youtube.video_id + ".mp3")
    print(f"Downloaded audio: {mp3_path}")


def download_subtitles(youtube: YouTube, output_path: Path) -> None:
    srt_path = output_path.with_suffix(".srt")

    if srt_path.exists():
        print(f"SRT already exists: {srt_path}")
        return

    # Implicit priority
    for lang in ["zh-HK", "zh-CN", "en"]:
        if lang in youtube.captions:
            youtube.captions[lang].save_captions(filename=str(srt_path))
            print(f"Downloaded subtitle: {srt_path}")
            if lang == "zh-CN":
                content = s2hk(read_file(srt_path))
                write_file(srt_path, content)
                print(f"Converted subtitle: {srt_path}")
            return

    print("No suitable subtitles found for download.")


def process_subtitles(youtube: YouTube, output_path: Path, whisper_model: str) -> None:
    """
    Process subtitles: download or transcribe as needed.
    Give preference to the uploader's existing manual captions. If unavailable, use Whisper to transcribe the video, as English automatic captions are bad and nonexistent for Chinese.
    """
    mp3_path = output_path.with_suffix(".mp3")
    srt_path = output_path.with_suffix(".srt")
    txt_path = output_path.with_suffix(".txt")
    available_subtitles = youtube.captions
    print(f"Available subtitles: {available_subtitles}")

    if available_subtitles and any(lang in available_subtitles for lang in ["en", "zh-HK", "zh-CN"]):
        download_subtitles(youtube, output_path)

    if srt_path.exists() and not txt_path.exists():
        write_file(txt_path, srt_to_txt(read_file(srt_path)))
        print(f"Converted TXT: {txt_path}")
        return

    # Assume 'a.en' always exists if it is English
    transcribe_language = "en"
    if "a.en" in available_subtitles:
        transcribe_language = "en"
    elif not available_subtitles or any(lang in available_subtitles for lang in ["zh-HK", "zh-CN"]):
        transcribe_language = "zh"

    response = whisper_transcribe(mp3_path, whisper_model, transcribe_language)

    write_file(srt_path, response_to_srt(response))
    print(f"Transcribed SRT: {srt_path}")

    write_file(txt_path, response_to_txt(response))
    print(f"Transcribed TXT: {txt_path}")


def url_to_subtitles(youtube: YouTube, whisper_model: str) -> str:
    """Process a YouTube video: download audio and handle subtitles."""
    try:
        cache_dir = BASE_CACHE_DIR / youtube.video_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / youtube.video_id
        txt_path = output_path.with_suffix(".txt")

        if txt_path.exists():
            print(f"Subtitle txt file already exists: {txt_path}")
            return read_file(txt_path)

        download_audio(youtube, cache_dir, output_path)
        process_subtitles(youtube, output_path, whisper_model)

        content = read_file(txt_path)
        formatted_content = llm_format_txt(content)
        write_file(txt_path, formatted_content)
        print(f"Formatted TXT: {txt_path}")

        return read_file(txt_path)

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


def youtubeloader(url: str, whisper_model: Literal["fal", "hf", "replicate"] = "fal") -> str:
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
