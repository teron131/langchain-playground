import base64
from pathlib import Path

import replicate
from dotenv import load_dotenv

load_dotenv()


def whisper_replicate(audio: Path | bytes) -> dict[str, str | list[dict[str, tuple[float] | str]]]:
    """
    Transcribe an audio file using Replicate model.
    https://replicate.com/vaibhavs10/incredibly-fast-whisper

    This function converts the audio file to base64 URI, and returns the transcription result.

    Args:
        audio (Path | bytes): The audio file / data to be transcribed.
    Returns:
        dict: A dictionary containing the transcription result with the following structure:
            {
                "text": str,    # Full transcribed text
                "chunks": [     # List of transcription chunks
                    # Each chunk is a dictionary with:
                    {
                        "timestamp": tuple[float],  # Start and end time of the chunk
                        "text": str,                # Transcribed text for this chunk
                    },
                ]
            }
    """
    if isinstance(audio, Path):
        with open(audio, "rb") as file:
            audio_bytes = file.read()
    elif isinstance(audio, bytes):
        audio_bytes = audio
    else:
        raise ValueError("Invalid audio type. Must be Path or bytes.")

    # Convert audio to base64 data URI
    data = base64.b64encode(audio_bytes).decode("utf-8")
    audio = f"data:application/octet-stream;base64,{data}"

    input = {"audio": audio, "batch_size": 64}
    output = replicate.run(
        "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
        input=input,
    )
    return output
