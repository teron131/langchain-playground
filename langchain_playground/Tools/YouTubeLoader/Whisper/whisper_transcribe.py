from pathlib import Path
from typing import Literal

from .whisper_fal import whisper_fal
from .whisper_hf import whisper_hf
from .whisper_replicate import whisper_replicate


def whisper_transcribe(
    audio: Path | bytes,
    whisper_model: Literal["fal", "replicate", "hf"] = "fal",
    language: str = None,
) -> dict[str, str | list[dict[str, tuple[float] | str]]]:
    """Transcribe audio file using specified whisper model and save as SRT and TXT files.

    Args:
        audio (Path | bytes): The audio file / data to be transcribed.
        whisper_model (str, optional): Whisper model to use. Defaults to "fal".
        language (str, optional): Language to transcribe. Defaults to "en".

    Returns:
        dict: A dictionary containing the transcription result with the following structure:
            {
                "text": str,    # Full transcribed text
                "chunks": [     # List of transcription chunks
                    # Each chunk is a dictionary with:
                    {
                        "timestamp": tuple[float],  # Start and end time of the chunk
                        "text": str,               # Transcribed text for this chunk
                    },
                ]
            }
    """
    if whisper_model == "fal":
        response = whisper_fal(audio, language=language)
    elif whisper_model == "replicate":
        response = whisper_replicate(audio)
    elif whisper_model == "hf":
        response = whisper_hf(audio)
    else:
        raise ValueError(f"Unsupported whisper model: {whisper_model}")
    return response
