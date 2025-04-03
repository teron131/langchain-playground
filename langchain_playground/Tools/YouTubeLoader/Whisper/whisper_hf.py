import warnings
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore")

load_dotenv()


def whisper_hf(audio: str | bytes) -> dict:
    """Transcribe audio file using whisper-large-v3-turbo model with Hugging Face optimization.

    Args:
        audio (str | bytes): The audio file path string or bytes data to be transcribed.
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
    result = pipe(audio)
    return result