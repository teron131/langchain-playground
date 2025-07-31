from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pytubefix import YouTube

from .llm_formatter import llm_format
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


class FileUtils:
    @staticmethod
    def read_text(file_path: Path) -> str:
        """Read text from a file with UTF-8 encoding.

        Args:
            file_path (Path): Path to the text file to read

        Returns:
            str: The file contents
        """
        return file_path.read_text(encoding="utf-8")

    @staticmethod
    def write_text(file_path: Path, content: str) -> str:
        """Write text to a file with UTF-8 encoding.

        Args:
            file_path (Path): Path to the text file to write
            content (str): Content to write to the text file
        """
        file_path.write_text(content, encoding="utf-8")
        return content


class YouTubeProcessor:
    def __init__(self, youtube: YouTube, whisper_model: Literal["fal", "hf", "replicate"] = "fal"):
        self.youtube: YouTube = youtube
        self.paths: FilePaths = FilePaths.from_youtube(youtube)
        self.whisper_model: Literal["fal", "hf", "replicate"] = whisper_model

    def download_audio(self) -> Optional[Path]:
        """Download audio from YouTube video if not already cached."""
        if self.paths.mp3_path.exists():
            print(f"Audio file already exists: {self.paths.mp3_path}")
            return self.paths.mp3_path

        self.youtube.streams.get_audio_only().download(output_path=str(self.paths.cache_dir), filename=self.youtube.video_id + ".mp3")
        print(f"Downloaded audio: {self.paths.mp3_path}")
        return self.paths.mp3_path

    def download_subtitle(self) -> Optional[Path]:
        """Download subtitle from YouTube if available."""
        if self.paths.srt_path.exists():
            print(f"SRT already exists: {self.paths.srt_path}")
            return self.paths.srt_path

        # Implicit priority
        for lang in ["zh-HK", "zh-CN", "en"]:
            if lang in self.youtube.captions:
                self.youtube.captions[lang].save_captions(filename=str(self.paths.srt_path))
                print(f"Downloaded subtitle: {self.paths.srt_path}")
                if lang == "zh-CN":
                    content = s2hk(FileUtils.read_text(self.paths.srt_path))
                    FileUtils.write_text(self.paths.srt_path, content)
                    print(f"Converted subtitle: {self.paths.srt_path}")
                return self.paths.srt_path

        print("No suitable subtitle found for download.")
        return None

    def process_subtitle(self) -> str:
        """Process subtitle: download or transcribe as needed.

        Give preference to the uploader's existing manual captions. If unavailable, use Whisper to transcribe the video, as English automatic captions are bad and nonexistent for Chinese.

        Args:
            whisper_model (Literal["fal", "hf", "replicate"]): The Whisper model to use

        Returns:
            str: The formatted subtitle
        """
        available_subtitles = self.youtube.captions
        print(f"Available subtitle: {available_subtitles}")

        if available_subtitles and any(lang in available_subtitles for lang in ["en", "zh-HK", "zh-CN"]):
            self.download_subtitle()

        if self.paths.srt_path.exists() and not self.paths.txt_path.exists():
            subtitle = srt_to_txt(FileUtils.read_text(self.paths.srt_path))
            FileUtils.write_text(self.paths.txt_path, subtitle)
            print(f"Converted TXT: {self.paths.txt_path}")
            return subtitle

        # Ensure audio is downloaded before transcription
        if not self.paths.mp3_path.exists():
            self.download_audio()

        transcribe_language = "en" if "a.en" in available_subtitles else "zh"
        result = whisper_transcribe(self.paths.mp3_path, self.whisper_model, transcribe_language)

        print(result)

        FileUtils.write_text(self.paths.srt_path, result_to_srt(result))
        print(f"Transcribed SRT: {self.paths.srt_path}")

        subtitle = result_to_txt(result)
        FileUtils.write_text(self.paths.txt_path, subtitle)
        print(f"Transcribed TXT: {self.paths.txt_path}")

        return subtitle

    def get_formatted_subtitle(self) -> str:
        """Get formatted subtitle from YouTube video.

        Args:
            whisper_model (Literal["fal", "hf", "replicate"]): The Whisper model to use

        Returns:
            str: The formatted subtitle
        """
        if self.paths.txt_path.exists():
            print(f"Subtitle txt file already exists: {self.paths.txt_path}")
            return FileUtils.read_text(self.paths.txt_path)

        self.download_audio()
        self.process_subtitle()

        with open(self.paths.mp3_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        subtitle = FileUtils.read_text(self.paths.txt_path)
        formatted_subtitle = llm_format(subtitle, audio_bytes)
        FileUtils.write_text(self.paths.txt_path, formatted_subtitle)
        print(f"Formatted TXT: {self.paths.txt_path}")

        return FileUtils.read_text(self.paths.txt_path)


def youtube_loader(
    url: str,
    whisper_model: Literal["fal", "hf", "replicate"] = "fal",
) -> str:
    """Load and process a YouTube video's subtitle, title, and author information from a URL.

    This function handles the entire process:
    1. Creates a YouTube object from the URL
    2. Downloads audio if needed
    3. Gets subtitles (from existing captions or transcription)
    4. Formats the subtitle with LLM
    5. Returns a formatted string with title, author and subtitle

    Accepts various YouTube URL formats including standard watch URLs and shortened youtu.be links.

    Args:
        url (str): The YouTube video URL to load
        whisper_model (Literal["fal", "hf", "replicate"], optional): The Whisper model to use. Defaults to "fal".

    Returns:
        str: Formatted string containing the video title, author and subtitle
    """
    youtube = YouTube(
        url,
        # use_po_token=True,
        # po_token_verifier=po_token_verifier,
    )

    processor = YouTubeProcessor(youtube, whisper_model)
    formatted_subtitle = processor.get_formatted_subtitle()

    content = [
        "Answer the user's question based on the full content.",
        f"Title: {youtube.title}",
        f"Author: {youtube.author}",
        f"subtitle:\n{formatted_subtitle}",
    ]
    return "\n".join(content)
