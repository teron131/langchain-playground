from functools import lru_cache
from typing import Dict

from opencc import OpenCC


@lru_cache(maxsize=None)
def s2hk(content: str) -> str:
    return OpenCC("s2hk").convert(content)


def convert_time_to_hms(seconds_float: float) -> str:
    """
    Converts a time in seconds to 'hh:mm:ss,ms' format for SRT.

    Args:
        seconds_float (float): Time in seconds.

    Returns:
        str: Time in 'hh:mm:ss,ms' format.
    """
    hours, remainder = divmod(seconds_float, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"


def response_to_srt(result: Dict) -> str:
    """
    Convert the specific transcription  API response into SRT format string.

    Args:
        result (Dict): The transcription result from the Whisper model.

    Returns:
        str: SRT formatted string
    """
    srt_entries = []
    for counter, chunk in enumerate(result["chunks"], 1):
        start_time = chunk.get("timestamp", [0])[0]
        end_time = chunk.get("timestamp", [0, start_time + 2.0])[1]  # Add 2 seconds to fade out
        start_time_hms = convert_time_to_hms(start_time)
        end_time_hms = convert_time_to_hms(end_time)
        transcript = chunk["text"].strip()
        transcript = s2hk(transcript)
        srt_entry = f"{counter}\n{start_time_hms} --> {end_time_hms}\n{transcript}\n\n"
        srt_entries.append(srt_entry)
    return "".join(srt_entries)


def response_to_txt(result: Dict) -> str:
    """
    Convert the specific transcription API response into plain text format.

    Args:
        result (Dict): The transcription result from the Whisper model.

    Returns:
        str: Plain text formatted string with each chunk on a new line
    """
    return "\n".join(s2hk(chunk["text"].strip()) for chunk in result["chunks"])


def srt_to_txt(srt_content: str) -> str:
    """
    Convert SRT format content to plain text.

    Args:
        srt_content (str): The SRT formatted content

    Returns:
        str: Plain text with only the transcript lines
    """
    return "\n".join(s2hk(line.strip()) for line in srt_content.splitlines() if not line.strip().isdigit() and "-->" not in line and line.strip())
