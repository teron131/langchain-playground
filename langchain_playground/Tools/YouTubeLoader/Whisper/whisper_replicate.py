import base64

import replicate


def whisper_replicate(audio_path: str) -> dict:
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
