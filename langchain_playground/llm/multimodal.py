"""Multimodal input utilities for images, PDFs, markdown, audio, and video files with OpenRouter."""

import base64
from pathlib import Path

from langchain_core.messages import HumanMessage


def _normalize_paths(paths: str | Path | list[str | Path]) -> list[Path]:
    """Normalize single path or list of paths, into list of Path objects."""
    return [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]


def _encode_file_to_base64(path: Path) -> str:
    """Encode file to base64 string."""
    with open(path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def _get_image_mime_type(path: Path) -> str:
    """Get MIME type for image file based on extension.

    The Responses API expects a data URL string in the `image_url` field, so we keep
    the MIME type around to build that.
    """
    suffix = path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(suffix, "image/jpeg")


def _get_audio_format(path: Path) -> str:
    """Get audio format for file based on extension.

    Returns:
        Audio format string: "wav" or "mp3"
    """
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "wav"
    elif suffix == ".mp3":
        return "mp3"
    else:
        raise ValueError(f"Unsupported audio format: {suffix}. Supported formats: .wav, .mp3")


def _get_video_mime_type(path: Path) -> str:
    """Get MIME type for video file based on extension."""
    suffix = path.suffix.lower()
    mime_types = {
        ".mp4": "video/mp4",
        ".mpeg": "video/mpeg",
        ".mov": "video/mov",
        ".webm": "video/webm",
    }
    return mime_types.get(suffix, "video/mp4")


class ImageMessage(HumanMessage):
    """HumanMessage with image file content encoded as base64."""

    def __init__(
        self,
        paths: str | Path | list[str | Path],
        description: str = "",
    ):
        """Initialize ImageMessage with image file(s).

        Args:
            paths: Single image path or list of image paths
            description: Description of the image
        """
        paths: list[Path] = _normalize_paths(paths)

        content = []
        for path in paths:
            image_base64 = _encode_file_to_base64(path)
            mime_type = _get_image_mime_type(path)
            data_url = f"data:{mime_type};base64,{image_base64}"
            # Responses API uses `input_image` blocks instead of legacy chat `image_url`
            content.append({"type": "input_image", "image_url": data_url, "detail": "auto"})

        if description:
            # Text blocks also use the `input_text` type in Responses API
            content.append({"type": "input_text", "text": description})

        super().__init__(content=content)


class PDFMessage(HumanMessage):
    """HumanMessage with PDF file content encoded as base64.

    Uses OpenRouter's PDF format: base64-encoded data URLs in the file content type.
    See https://openrouter.ai/docs/features/multimodal/pdfs for details.

    Note: Some models (e.g. Gemini) may not support PDFs through OpenRouter. If you encounter "invalid_prompt" errors, try using a different model (e.g., GPT).
    """

    def __init__(
        self,
        paths: str | Path | list[str | Path],
        description: str = "",
    ):
        """Initialize PDFMessage with PDF file(s).

        Args:
            paths: Single PDF path or list of PDF paths
            description: Description of the PDF
        """
        paths: list[Path] = _normalize_paths(paths)

        content = []
        for path in paths:
            pdf_base64 = _encode_file_to_base64(path)
            data_url = f"data:application/pdf;base64,{pdf_base64}"
            # OpenRouter format: https://openrouter.ai/docs/features/multimodal/pdfs
            content.append(
                {
                    "type": "file",
                    "file": {"filename": path.name, "file_data": data_url},
                }
            )

        if description:
            content.append({"type": "text", "text": description})

        super().__init__(content=content)


class MarkdownMessage(HumanMessage):
    """HumanMessage with markdown file content loaded as text string."""

    def __init__(
        self,
        paths: str | Path | list[str | Path],
        description: str = "",
    ):
        """Initialize MarkdownMessage with markdown file(s).

        Args:
            paths: Single markdown path or list of markdown paths
            description: Description of the markdown
        """
        paths: list[Path] = _normalize_paths(paths)

        content = []
        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                file_content = f.read()
            content.append({"type": "text", "text": file_content})

        if description:
            content.append({"type": "text", "text": description})

        super().__init__(content=content)


class AudioMessage(HumanMessage):
    """HumanMessage with audio file content encoded as base64.

    Uses OpenRouter's audio format: base64-encoded data with format specification.
    See https://openrouter.ai/docs/features/multimodal/audio for details.

    Supported formats: wav, mp3
    """

    def __init__(
        self,
        paths: str | Path | list[str | Path],
        description: str = "",
    ):
        """Initialize AudioMessage with audio file(s).

        Args:
            paths: Single audio path or list of audio paths (supports .wav, .mp3)
            description: Description of the audio
        """
        paths: list[Path] = _normalize_paths(paths)

        content = []
        for path in paths:
            audio_format = _get_audio_format(path)
            audio_base64 = _encode_file_to_base64(path)
            # OpenRouter format: https://openrouter.ai/docs/features/multimodal/audio
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_base64, "format": audio_format},
                }
            )

        if description:
            content.append({"type": "text", "text": description})

        super().__init__(content=content)


class VideoMessage(HumanMessage):
    """HumanMessage with video file content encoded as base64.

    Uses OpenRouter's video format: video_url with base64 data URL.
    See https://openrouter.ai/docs/features/multimodal/videos for details.

    Supported formats: mp4, mpeg, mov, webm
    """

    def __init__(
        self,
        paths: str | Path | list[str | Path],
        description: str = "",
    ):
        """Initialize VideoMessage with video file(s).

        Args:
            paths: Single video path or list of video paths (supports .mp4, .mpeg, .mov, .webm)
            description: Description of the video
        """
        paths: list[Path] = _normalize_paths(paths)

        content = []
        for path in paths:
            video_base64 = _encode_file_to_base64(path)
            mime_type = _get_video_mime_type(path)
            data_url = f"data:{mime_type};base64,{video_base64}"
            # OpenRouter format: https://openrouter.ai/docs/features/multimodal/videos
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": data_url},
                }
            )

        if description:
            content.append({"type": "text", "text": description})

        super().__init__(content=content)
