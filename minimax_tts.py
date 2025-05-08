import os
import re
import tarfile
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()


OUTPUT_DIR = Path("audio")
OUTPUT_DIR.mkdir(exist_ok=True)


def split_text(text: str) -> list[str]:
    """
    Split text based on the primary language and its punctuation characteristics:
    - For primarily Chinese text: splits at Chinese punctuation
    - For primarily English text: splits at English punctuation followed by space
    - Determines primary language based on character count

    Args:
        text (str): The text to split.

    Returns:
        list: A list of split text parts.
    """
    if not text:
        return []

    # Define regex patterns
    CHINESE_CHARACTERS = r"[\u4e00-\u9fff]"
    ENGLISH_CHARACTERS = r"[a-zA-Z]"

    CHINESE_PUNCTUATIONS = r"，。、；：？！"
    ENGLISH_PUNCTUATIONS = r",;:.?!"

    CHINESE_PATTERNS = f"(?<=[{CHINESE_PUNCTUATIONS}])|\\s+"
    ENGLISH_PATTERNS = f"(?<=[{ENGLISH_PUNCTUATIONS}])\\s+"

    # Count characters to determine language
    CHINESE_COUNT = len(re.findall(CHINESE_CHARACTERS, text))
    ENGLISH_COUNT = len(re.findall(ENGLISH_CHARACTERS, text))

    # Choose splitting strategy based on language composition
    if CHINESE_COUNT > 0 and (CHINESE_COUNT >= ENGLISH_COUNT or any(p in text for p in CHINESE_PUNCTUATIONS)):
        # For mixed or primarily Chinese text, handle both punctuation types
        parts = []
        # First split by Chinese punctuation
        initial_parts = re.split(CHINESE_PATTERNS, text)

        # Then process each part for English punctuation
        for part in initial_parts:
            if any(p in part for p in ENGLISH_PUNCTUATIONS):
                parts.extend(re.split(ENGLISH_PATTERNS, part))
            else:
                parts.append(part)
    else:
        # Split by English punctuation
        parts = re.split(ENGLISH_PATTERNS, text)

    return [part.strip() for part in parts if part.strip()]


def run_tts(
    text: str,
    speed: float = 1.2,
    voice_id: str = "English_Trustworth_Man",
    emotion: str = "neutral",
    language_boost: str = "English",
    **kwargs,
) -> bytes:
    """Run TTS (Text-to-Speech) using Minimax API."""
    group_id = os.getenv("MINIMAX_GROUP_ID")
    api_key = os.getenv("MINIMAX_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    def _check(response, context):
        try:
            response.raise_for_status()
        except requests.HTTPError:
            raise RuntimeError(f"{context} failed: {response.text}")
        data = response.json()
        base = data.get("base_resp", {})
        if base.get("status_code") != 0:
            raise RuntimeError(f"{context} API error: {base.get('status_msg', 'Unknown error')}")
        return data

    # Step 1: Create speech generation task
    payload = {
        "model": kwargs.get("model", "speech-02-hd"),
        "text": text,
        "voice_setting": {
            "speed": speed,
            "vol": kwargs.get("vol", 1.0),
            "pitch": kwargs.get("pitch", 0),
            "voice_id": voice_id,
            "emotion": emotion,
            "english_normalization": kwargs.get("english_normalization", False),
        },
        "audio_setting": {
            "sample_rate": kwargs.get("sample_rate", 32000),
            "bitrate": kwargs.get("bitrate", 128000),
            "format": kwargs.get("format", "mp3"),
            "channel": kwargs.get("channel", 1),
        },
        "pronunciation_dict": {"tone": kwargs.get("pronunciation_tone", [])},
        "stream": kwargs.get("stream", False),
        "language_boost": language_boost,
        "subtitle_enable": kwargs.get("subtitle_enable", False),
    }
    create_url = f"https://api.minimaxi.chat/v1/t2a_async_v2?GroupId={group_id}"
    data = _check(requests.post(create_url, headers=headers, json=payload), "Create task")
    task_id, file_id = data.get("task_id"), data.get("file_id")
    if not (task_id and file_id):
        raise RuntimeError(f"Missing task_id or file_id: {data}")
    print(f"Task created with ID: {task_id}, file_id: {file_id}")

    # Step 2: Poll task status
    query_url = f"https://api.minimaxi.chat/v1/query/t2a_async_query_v2?GroupId={group_id}&task_id={task_id}"
    waiting_time = 0
    while True:
        data = _check(requests.get(query_url, headers=headers), "Query task")
        status = data.get("status")
        if status == "Success":
            print("Task completed successfully!")
            break
        if status in ("Failed", "Expired"):
            raise RuntimeError(f"Task {status.lower()}: {data}")
        time.sleep(10)
        waiting_time += 10
        print(f"Task status: {status}, waiting... {waiting_time}s")

    # Step 3: Retrieve download URL
    retrieve_url = f"https://api.minimaxi.chat/v1/files/retrieve?GroupId={group_id}&file_id={file_id}"
    data = _check(requests.get(retrieve_url, headers=headers), "Retrieve file")
    download_url = data.get("file", {}).get("download_url")
    if not download_url:
        raise RuntimeError(f"Missing download URL: {data}")
    print(f"Got download URL: {download_url}")

    # Step 4: Download and extract MP3
    tar_path = OUTPUT_DIR / f"{file_id}.tar"
    resp = requests.get(download_url, stream=True)
    resp.raise_for_status()
    with tar_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if member.name.endswith(".mp3"):
                extracted = tar.extractfile(member)
                if not extracted:
                    continue
                dest = OUTPUT_DIR / Path(member.name).name
                with dest.open("wb") as out:
                    out.write(extracted.read())
    tar_path.unlink()

    mp3_files = list(OUTPUT_DIR.glob("*.mp3"))
    if not mp3_files:
        raise RuntimeError("No mp3 files found after extraction")
    return mp3_files[0].read_bytes()


text = """

"""

if __name__ == "__main__":
    run_tts(text)
