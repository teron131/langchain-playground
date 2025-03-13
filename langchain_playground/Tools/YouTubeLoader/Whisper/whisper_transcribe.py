from pathlib import Path

from .whisper_fal import whisper_fal
from .whisper_hf import whisper_hf
from .whisper_replicate import whisper_replicate


def whisper_transcribe(mp3_path: str | Path, whisper_model: str = "fal", language: str = "en") -> tuple[Path, Path]:
    """Transcribe audio file using specified whisper model and save as SRT and TXT files.

    Args:
        mp3_path (str | Path): Path to the audio file
        whisper_model (str, optional): Whisper model to use. Defaults to "fal".
        language (str, optional): Language to transcribe. Defaults to "en".

    Returns:
        tuple[Path, Path]: Paths to the generated SRT and TXT files
    """
    mp3_path = Path(mp3_path)
    if whisper_model == "fal":
        response = whisper_fal(str(mp3_path), language=language)
    elif whisper_model == "hf":
        response = whisper_hf(str(mp3_path))
    elif whisper_model == "replicate":
        response = whisper_replicate(str(mp3_path))
    else:
        raise ValueError(f"Unsupported whisper model: {whisper_model}")
    return response
