import os
import re
import time
from pathlib import Path
from typing import Literal

import requests
from dotenv import load_dotenv
from IPython import get_ipython
from IPython.display import Audio, display

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


def add_pause(text: str, pause_sec: float = 0.5) -> str:
    """Add pause to the text sentence by sentence based on the delimiter.

    https://www.minimax.io/audio

    Args:
        text (str): The text to add pause to.
        pause_sec (float): The pause time in seconds.

    Returns:
        str: The text with pause added.
    """
    DELIMITER = f"<#{pause_sec}#>"
    text_parts = split_text(text)
    return DELIMITER.join(text_parts)


def run_tts(
    text: str,
    speed: float = 1.25,
    voice_id: str = "Cantonese_WiselProfessor",
    emotion: Literal["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"] = "neutral",
    language_boost: str = "Chinese,Yue",
    pause_sec: float = 0.4,
    **kwargs,
) -> bytes:
    """Run TTS (Text-to-Speech) using Minimax API.

    https://www.minimax.io/platform/document/T2A%20V2?key=66719005a427f0c8a5701643

    Args:
        text (str): The text to convert to speech.

    Returns:
        bytes: The audio data in bytes.
    """
    MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID")
    MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY")

    url = f"https://api.minimaxi.chat/v1/t2a_v2?GroupId={MINIMAX_GROUP_ID}"

    headers = {
        "Authorization": f"Bearer {MINIMAX_API_KEY}",
        "Content-Type": "application/json",
    }

    data = {
        "model": kwargs.get("model", "speech-01-hd"),
        "text": add_pause(text, pause_sec=pause_sec / speed),
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
        "pronunciation_dict": {
            "tone": kwargs.get("pronunciation_tone", []),
        },
        "stream": kwargs.get("stream", False),
        "language_boost": language_boost,
        "subtitle_enable": kwargs.get("subtitle_enable", False),
        "output_format": kwargs.get("output_format", "hex"),
    }

    response = requests.post(url, headers=headers, json=data)

    # Extract the audio data from the response
    audio_data_hex = response.json()["data"]["audio"]
    audio_binary = bytes.fromhex(audio_data_hex)

    # Display the audio in ipynb for playback, if not in ipynb, save to file
    if get_ipython() is not None:
        display(Audio(data=audio_binary, rate=32000))  # Default rate is 32000
    else:
        timestamp = int(time.time())
        file_name = OUTPUT_DIR / f"output_{timestamp}.mp3"
        with open(file_name, "wb") as file:
            file.write(audio_binary)
        print(f"Audio saved to {file_name}")


text = """即使你降生的時世沒選擇, 人間再蒼白, 容得下想法, 世界藍圖只等你畫上恐龍和巨塔, 天下仍是你畫冊"""

if __name__ == "__main__":
    run_tts(text)
