from functools import lru_cache
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai.chat_models.base import ChatOpenAI
from opencc import OpenCC
from pytubefix import YouTube

from .whisper_fal import whisper_fal_transcribe
from .whisper_hf import whisper_hf_transcribe

load_dotenv()

# File handling functions


def create_cache_dir(video_id: str) -> Path:
    cache_dir = Path(f".cache/{video_id}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_output_path(cache_dir: Path, video_id: str) -> Path:
    """Get the output path for the given cache directory and video ID."""
    return Path(cache_dir / video_id)


def read_file(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def write_file(file_path: Path, content: str) -> None:
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


# Subtitle preprocessing functions


@lru_cache(maxsize=None)
def s2hk(content: str) -> str:
    return OpenCC("s2hk").convert(content)


def llm_format_txt(txt_filepath: str, chunk_size: int = 1000) -> None:
    """Format subtitles using LLM."""
    txt_path = Path(txt_filepath).with_suffix(".txt")

    preprocess_subtitles_chain = (
        hub.pull("preprocess_subtitles")
        | ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )
        | StrOutputParser()
        | RunnableLambda(s2hk)
    )

    subtitles = read_file(txt_path)
    chunked_subtitles = [subtitles[i : i + chunk_size] for i in range(0, len(subtitles), chunk_size)]

    formatted_subtitles = preprocess_subtitles_chain.batch([{"subtitles": chunk} for chunk in chunked_subtitles])
    formatted_subtitles = "".join(formatted_subtitles)

    write_file(txt_path, formatted_subtitles)
    print(f"Formatted TXT: {txt_path}")


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


def response_to_srt(result: Dict, srt_path: str) -> None:
    """
    Process the transcription result into SRT format and write to a file.

    Args:
        result (Dict): The transcription result from the Whisper model.
        srt_path (str): The path to the output SRT file.

    Returns:
        None
    """
    with open(srt_path, "w", encoding="utf-8") as srt_file:
        for counter, chunk in enumerate(result["chunks"], 1):
            start_time = chunk.get("timestamp", [0])[0]
            end_time = chunk.get("timestamp", [0, start_time + 2.0])[1]  # Add 2 seconds to fade out
            start_time_hms = convert_time_to_hms(start_time)
            end_time_hms = convert_time_to_hms(end_time)
            transcript = chunk["text"].strip()
            transcript = s2hk(transcript)
            srt_entry = f"{counter}\n{start_time_hms} --> {end_time_hms}\n{transcript}\n\n"
            srt_file.write(srt_entry)


def response_to_txt(result: Dict, txt_path: str) -> None:
    """
    Process the transcription result into a plain text format and write to a file.

    Args:
        result (Dict): The transcription result from the Whisper model.
        txt_path (str): The path to the output text file.

    Returns:
        None
    """
    with open(txt_path, "w", encoding="utf-8") as txt_file:
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
    youtube.streams.get_audio_only().download(output_path=str(cache_dir), filename=youtube.video_id, mp3=True)
    print(f"Downloaded audio: {mp3_path}")


def download_subtitles(youtube: YouTube, output_path: Path) -> None:
    srt_path = output_path.with_suffix(".srt")

    if srt_path.exists():
        print(f"SRT already exists: {srt_path}")
        return

    # Implicit priority
    for lang in ["zh-HK", "zh-CN", "en"]:
        if lang in youtube.captions:
            youtube.captions[lang].save_captions(filename=srt_path)
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
    else:
        raise ValueError(f"Unsupported whisper model: {whisper_model}")

    response_to_srt(response, str(srt_path))
    print(f"Transcribed SRT: {srt_path}")

    response_to_txt(response, str(txt_path))
    print(f"Transcribed TXT: {txt_path}")


def url_to_subtitles(url: str, whisper_model: str = "fal") -> str:
    """Process a YouTube video: download audio and handle subtitles."""
    try:
        youtube = YouTube(url)
        cache_dir = create_cache_dir(youtube.video_id)
        output_path = get_output_path(cache_dir, youtube.video_id)
        txt_path = output_path.with_suffix(".txt")

        if txt_path.exists():
            print(f"Subtitle txt file already exists: {txt_path}")
            return read_file(txt_path)

        download_audio(youtube, cache_dir, output_path)
        process_subtitles(youtube, output_path, whisper_model)
        llm_format_txt(str(txt_path))

        return read_file(txt_path)

    except Exception as e:
        error_message = f"Error processing video {url}: {str(e)}"
        print(error_message)
        return error_message
