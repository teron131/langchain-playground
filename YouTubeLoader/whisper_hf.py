import warnings

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore")


def whisper_hf_transcribe(audio_path: str):
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
