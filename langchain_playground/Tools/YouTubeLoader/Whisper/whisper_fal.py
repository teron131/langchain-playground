from pathlib import Path

import fal_client
from dotenv import load_dotenv

load_dotenv()


def whisper_fal(
    audio: Path | bytes,
    language: str = None,
) -> dict:
    """Transcribe an audio file using Fal model.
    https://fal.ai/models/fal-ai/whisper

    This function uploads the audio file, subscribes to the transcription service,
    and returns the transcription result.

    It defaults at English.

    Args:
        audio (Path | bytes): The audio file / data to be transcribed.
        language (str): The language of the audio file. Defaults to "en".
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

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(log["message"])

    if isinstance(audio, Path):
        url = fal_client.upload_file(audio)
    elif isinstance(audio, bytes):
        url = fal_client.upload(data=audio, content_type="audio/mp3")
    else:
        raise ValueError("Invalid audio type")

    result = fal_client.subscribe(
        # "fal-ai/wizper",
        "fal-ai/whisper",
        arguments={
            "audio_url": url,
            "task": "transcribe",
            "language": language,
        },
        with_logs=True,
        on_queue_update=on_queue_update,
    )
    return result
