from pathlib import Path

from pytubefix import YouTube


def download_youtube_audio(url: str) -> str:
    """Download audio from a YouTube video and save it to the 'audio' directory.

    Args:
        url (str): The YouTube video URL

    Returns:
        str: Path to the downloaded audio file
    """
    # Create audio directory if it doesn't exist
    audio_dir = Path("audio")
    audio_dir.mkdir(exist_ok=True)

    # Initialize YouTube object
    youtube = YouTube(url)

    # Get video ID for filename
    video_title = youtube.title
    mp3_path = audio_dir / f"{video_title}.mp3"

    # Check if file already exists
    if mp3_path.exists():
        print(f"Audio file already exists: {mp3_path}")
        return str(mp3_path)

    # Download audio
    youtube.streams.get_audio_only().download(output_path=str(audio_dir), filename=f"{video_title}.mp3")
    print(f"Downloaded audio: {mp3_path}")

    return str(mp3_path)


if __name__ == "__main__":
    download_youtube_audio("https://youtu.be/o-BtD9uO31Y")
