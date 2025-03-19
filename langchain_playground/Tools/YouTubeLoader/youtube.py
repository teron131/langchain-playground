from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pytubefix import YouTube

from .llm_formatter import llm_format_text, llm_format_text_audio
from .utils import po_token_verifier, result_to_srt, result_to_txt, s2hk, srt_to_txt
from .Whisper import whisper_transcribe

load_dotenv()

BASE_CACHE_DIR = Path(".cache")


@dataclass
class FilePaths:
    cache_dir: Path
    output_path: Path
    mp3_path: Path
    srt_path: Path
    txt_path: Path

    @classmethod
    def from_youtube(cls, youtube: YouTube) -> "FilePaths":
        cache_dir = BASE_CACHE_DIR / youtube.video_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / youtube.video_id
        mp3_path = output_path.with_suffix(".mp3")
        srt_path = output_path.with_suffix(".srt")
        txt_path = output_path.with_suffix(".txt")
        return cls(
            cache_dir=cache_dir,
            output_path=output_path,
            mp3_path=mp3_path,
            srt_path=srt_path,
            txt_path=txt_path,
        )


def read_text_file(file_path: Path) -> str:
    """
    Read text from a file with UTF-8 encoding.

    Args:
        file_path (Path): Path to the text file to read

    Returns:
        str: The file contents
    """
    return file_path.read_text(encoding="utf-8")


def write_text_file(file_path: Path, content: str) -> None:
    """
    Write text to a file with UTF-8 encoding.

    Args:
        file_path (Path): Path to the text file to write
        content (str): Content to write to the text file
    """
    file_path.write_text(content, encoding="utf-8")


# YouTube video processing functions


def download_audio(youtube: YouTube) -> None:
    paths = FilePaths.from_youtube(youtube)
    if paths.mp3_path.exists():
        print(f"Audio file already exists: {paths.mp3_path}")
        return
    youtube.streams.get_audio_only().download(output_path=str(paths.cache_dir), filename=youtube.video_id + ".mp3")
    print(f"Downloaded audio: {paths.mp3_path}")


def download_subtitles(youtube: YouTube) -> None:
    paths = FilePaths.from_youtube(youtube)
    if paths.srt_path.exists():
        print(f"SRT already exists: {paths.srt_path}")
        return

    # Implicit priority
    for lang in ["zh-HK", "zh-CN", "en"]:
        if lang in youtube.captions:
            youtube.captions[lang].save_captions(filename=str(paths.srt_path))
            print(f"Downloaded subtitle: {paths.srt_path}")
            if lang == "zh-CN":
                content = s2hk(read_text_file(paths.srt_path))
                write_text_file(paths.srt_path, content)
                print(f"Converted subtitle: {paths.srt_path}")
            return

    print("No suitable subtitle found for download.")


def process_subtitles(youtube: YouTube, whisper_model: str) -> None:
    """
    Process subtitle: download or transcribe as needed.
    Give preference to the uploader's existing manual captions. If unavailable, use Whisper to transcribe the video, as English automatic captions are bad and nonexistent for Chinese.
    """
    paths = FilePaths.from_youtube(youtube)
    available_subtitles = youtube.captions
    print(f"Available subtitle: {available_subtitles}")

    if available_subtitles and any(lang in available_subtitles for lang in ["en", "zh-HK", "zh-CN"]):
        download_subtitles(youtube)

    if paths.srt_path.exists() and not paths.txt_path.exists():
        write_text_file(paths.txt_path, srt_to_txt(read_text_file(paths.srt_path)))
        print(f"Converted TXT: {paths.txt_path}")
        return

    transcribe_language = "en" if "a.en" in available_subtitles else "zh"
    result = whisper_transcribe(paths.mp3_path, whisper_model, transcribe_language)

    write_text_file(paths.srt_path, result_to_srt(result))
    print(f"Transcribed SRT: {paths.srt_path}")

    write_text_file(paths.txt_path, result_to_txt(result))
    print(f"Transcribed TXT: {paths.txt_path}")


def youtube_to_subtitle(
    youtube: YouTube,
    whisper_model: Literal["fal", "hf", "replicate"] = "fal",
) -> str:
    """Process a YouTube video: download audio and handle subtitle."""
    paths = FilePaths.from_youtube(youtube)
    if paths.txt_path.exists():
        print(f"Subtitle txt file already exists: {paths.txt_path}")
        return read_text_file(paths.txt_path)

    download_audio(youtube)
    process_subtitles(youtube, whisper_model)

    with open(paths.mp3_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    subtitle = read_text_file(paths.txt_path)
    formatted_subtitle = llm_format_text_audio(subtitle, audio_bytes)
    write_text_file(paths.txt_path, formatted_subtitle)
    print(f"Formatted TXT: {paths.txt_path}")

    return read_text_file(paths.txt_path)


def youtubeloader(
    url: str,
    whisper_model: Literal["fal", "hf", "replicate"] = "fal",
) -> str:
    """
    Load and process a YouTube video's subtitle, title, and author information from a URL. Accepts various YouTube URL formats including standard watch URLs and shortened youtu.be links.

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
    paths = FilePaths.from_youtube(youtube)
    if paths.txt_path.exists():
        print(f"Subtitle txt file already exists: {paths.txt_path}")
        return read_text_file(paths.txt_path)

    download_audio(youtube)
    process_subtitles(youtube, whisper_model)

    with open(paths.mp3_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    subtitle = read_text_file(paths.txt_path)
    formatted_subtitle = llm_format_text_audio(subtitle, audio_bytes)
    write_text_file(paths.txt_path, formatted_subtitle)
    print(f"Formatted TXT: {paths.txt_path}")

    content = [
        "Answer the user's question based on the full content.",
        f"Title: {youtube.title}",
        f"Author: {youtube.author}",
        f"subtitle:\n{formatted_subtitle}",
    ]
    return "\n".join(content)
