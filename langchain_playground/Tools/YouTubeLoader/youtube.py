import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from opencc import OpenCC
from pytubefix import YouTube

from .whisper_fal import whisper_fal_transcribe
from .whisper_hf import whisper_hf_transcribe
from .whisper_replicate import whisper_replicate_transcribe

load_dotenv()

BASE_CACHE_DIR = Path(".cache")


# File handling functions


def create_cache_dir(video_id: str) -> Path:
    cache_dir = BASE_CACHE_DIR / video_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_output_path(cache_dir: Path, video_id: str) -> Path:
    """Get the output path for the given cache directory and video ID."""
    return cache_dir / video_id


def read_file(file_path: Path) -> str:
    return file_path.read_text(encoding="utf-8")


def write_file(file_path: Path, content: str) -> None:
    file_path.write_text(content, encoding="utf-8")


# Subtitle preprocessing functions


@lru_cache(maxsize=None)
def s2hk(content: str) -> str:
    return OpenCC("s2hk").convert(content)


def llm_format_txt(txt_filepath: Path, chunk_size: int = 1000) -> None:
    """Format subtitles using LLM."""
    preprocess_subtitles_chain = (
        hub.pull("preprocess_subtitles")
        | ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )
        | StrOutputParser()
        | RunnableLambda(s2hk)
    )

    subtitles = read_file(txt_filepath)
    chunked_subtitles = [subtitles[i : i + chunk_size] for i in range(0, len(subtitles), chunk_size)]

    formatted_subtitles = preprocess_subtitles_chain.batch([{"subtitles": chunk} for chunk in chunked_subtitles])
    formatted_subtitles = "".join(formatted_subtitles)

    write_file(txt_filepath, formatted_subtitles)
    print(f"Formatted TXT: {txt_filepath}")


# Utility functions


def convert_time_to_hms(seconds_float: float) -> str:
    """
    Converts a time in seconds to 'hh:mm:ss,ms' format for SRT.

    Args:
        seconds_float (float): Time in seconds.

    Returns:
        str: Time in 'hh:mm:ss,ms' format.
    """
    hours, remainder = divmod(seconds_float, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"


def response_to_srt(result: Dict, srt_path: Path) -> None:
    """
    Process the transcription result into SRT format and write to a file.

    Args:
        result (Dict): The transcription result from the Whisper model.
        srt_path (Path): The path to the output SRT file.

    Returns:
        None
    """
    with srt_path.open("w", encoding="utf-8") as srt_file:
        for counter, chunk in enumerate(result["chunks"], 1):
            start_time = chunk.get("timestamp", [0])[0]
            end_time = chunk.get("timestamp", [0, start_time + 2.0])[1]  # Add 2 seconds to fade out
            start_time_hms = convert_time_to_hms(start_time)
            end_time_hms = convert_time_to_hms(end_time)
            transcript = chunk["text"].strip()
            transcript = s2hk(transcript)
            srt_entry = f"{counter}\n{start_time_hms} --> {end_time_hms}\n{transcript}\n\n"
            srt_file.write(srt_entry)


def response_to_txt(result: Dict, txt_path: Path) -> None:
    """
    Process the transcription result into a plain text format and write to a file.

    Args:
        result (Dict): The transcription result from the Whisper model.
        txt_path (Path): The path to the output text file.

    Returns:
        None
    """
    with txt_path.open("w", encoding="utf-8") as txt_file:
        txt_file.write("\n".join(s2hk(chunk["text"].strip()) for chunk in result["chunks"]))


def srt_to_txt(srt_path: Path) -> None:
    txt_path = srt_path.with_suffix(".txt")
    content = read_file(srt_path)

    txt_content = "\n".join(s2hk(line.strip()) for line in content.splitlines() if not line.strip().isdigit() and "-->" not in line and line.strip())

    write_file(txt_path, txt_content)
    print(f"Converted TXT: {txt_path}")


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


def process_subtitles(youtube: YouTube, output_path: Path, whisper_model: str = "fal") -> None:
    """Process subtitles: download or transcribe as needed."""
    mp3_path = output_path.with_suffix(".mp3")
    srt_path = output_path.with_suffix(".srt")
    txt_path = output_path.with_suffix(".txt")
    available_subtitles = youtube.captions
    print(f"Available subtitles: {available_subtitles}")

    if available_subtitles and any(lang in available_subtitles for lang in ["en", "zh-HK", "zh-CN"]):
        download_subtitles(youtube, output_path)

    if srt_path.exists() and not txt_path.exists():
        srt_to_txt(srt_path)
        return

    # Assume 'a.en' always exists if it is English
    transcribe_language = "en"
    if "a.en" in available_subtitles:
        transcribe_language = "en"
    elif not available_subtitles or any(lang in available_subtitles for lang in ["zh-HK", "zh-CN"]):
        transcribe_language = "zh"

    if whisper_model == "fal":
        response = whisper_fal_transcribe(str(mp3_path), language=transcribe_language)
    elif whisper_model == "hf":
        response = whisper_hf_transcribe(str(mp3_path))
    elif whisper_model == "replicate":
        response = whisper_replicate_transcribe(str(mp3_path))
    else:
        raise ValueError(f"Unsupported whisper model: {whisper_model}")

    response_to_srt(response, srt_path)
    print(f"Transcribed SRT: {srt_path}")

    response_to_txt(response, txt_path)
    print(f"Transcribed TXT: {txt_path}")


def url_to_subtitles(youtube: YouTube, whisper_model: str = "fal") -> str:
    """Process a YouTube video: download audio and handle subtitles."""
    try:
        cache_dir = create_cache_dir(youtube.video_id)
        output_path = get_output_path(cache_dir, youtube.video_id)
        txt_path = output_path.with_suffix(".txt")

        if txt_path.exists():
            print(f"Subtitle txt file already exists: {txt_path}")
            return read_file(txt_path)

        download_audio(youtube, cache_dir, output_path)
        process_subtitles(youtube, output_path, whisper_model)
        llm_format_txt(txt_path)

        return read_file(txt_path)

    except Exception as e:
        error_message = f"Error processing video {youtube.title}: {str(e)}"
        print(error_message)
        return error_message


# Main function


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


def youtubeloader(url: str, whisper_model: str = "fal") -> str:
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
