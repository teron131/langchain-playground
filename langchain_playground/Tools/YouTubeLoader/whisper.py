import base64
import warnings
from pathlib import Path

import fal_client
import replicate
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore")

load_dotenv()


def whisper_hf_transcribe(audio_path: str) -> dict:
    """
    Transcribe audio file using whisper-large-v3-turbo model with Hugging Face optimization.

    Returns:
        dict: A dictionary containing the transcription result with the following structure:
            {
                "text": str,  # Full transcribed text
                "chunks": [
                    {
                        "timestamp": Tuple[float],  # Start and end time of the chunk
                        "text": str,  # Transcribed text for this chunk
                    }
                ]
            }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Using device: {device} ({torch_dtype})")

    model_id = "openai/whisper-large-v3-turbo"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    result = pipe(audio_path)
    return result


def whisper_fal_transcribe(audio_path: str, language: str = "en") -> dict:
    """
    Transcribe an audio file using Fal model.
    https://fal.ai/models/fal-ai/whisper

    This function uploads the audio file, subscribes to the transcription service,
    and returns the transcription result.

    It defaults at English.

    Args:
        audio_path (str): The path to the audio file to be transcribed.
        language (str): The language of the audio file. Defaults to "en".
    Returns:
        dict: A dictionary containing the transcription result with the following structure:
            {
                "text": str,  # Full transcribed text
                "chunks": List[dict],  # List of transcription chunks
                    # Each chunk is a dictionary with:
                    {
                        "timestamp": List[float],  # Start and end time of the chunk
                        "text": str,  # Transcribed text for this chunk
                    }
            }
    """

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(log["message"])

    url = fal_client.upload_file(audio_path)
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


def whisper_replicate_transcribe(audio_path: str) -> dict:
    """
    Transcribe an audio file using Replicate model.
    https://replicate.com/vaibhavs10/incredibly-fast-whisper

    This function converts the audio file to base64 URI, and returns the transcription result.

    Args:
        audio_path (str): The path to the audio file to be transcribed.
    Returns:
        dict: A dictionary containing the transcription result with the following structure:
            {
                "text": str,  # Full transcribed text
                "chunks": List[dict],  # List of transcription chunks
                    # Each chunk is a dictionary with:
                    {
                        "timestamp": List[float],  # Start and end time of the chunk
                        "text": str,  # Transcribed text for this chunk
                    }
            }
    """
    with open(audio_path, "rb") as file:
        data = base64.b64encode(file.read()).decode("utf-8")
        audio = f"data:application/octet-stream;base64,{data}"

    input = {"audio": audio, "batch_size": 64}
    output = replicate.run(
        "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
        input=input,
    )
    return output


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
        response = whisper_fal_transcribe(str(mp3_path), language=language)
    elif whisper_model == "hf":
        response = whisper_hf_transcribe(str(mp3_path))
    elif whisper_model == "replicate":
        response = whisper_replicate_transcribe(str(mp3_path))
    else:
        raise ValueError(f"Unsupported whisper model: {whisper_model}")
    return response
