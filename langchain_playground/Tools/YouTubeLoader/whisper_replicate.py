import base64

import replicate


def whisper_replicate_transcribe(audio_path: str) -> dict:
    with open(audio_path, "rb") as file:
        data = base64.b64encode(file.read()).decode("utf-8")
        audio = f"data:application/octet-stream;base64,{data}"

    input = {"audio": audio, "batch_size": 64}
    output = replicate.run(
        "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
        input=input,
    )
    return output
