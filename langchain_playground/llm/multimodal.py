"""Multimodal input utilities for images, PDFs, markdown, audio, and video files using LangChain content blocks."""

import base64
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.messages.content import (
    create_audio_block,
    create_file_block,
    create_image_block,
    create_plaintext_block,
    create_text_block,
    create_video_block,
)

# Extension to (file_type, mime_type) mapping
EXTENSION_MAP: dict[str, tuple[str, str]] = {
    # Images
    ".jpg": ("image", "image/jpeg"),
    ".jpeg": ("image", "image/jpeg"),
    ".png": ("image", "image/png"),
    ".gif": ("image", "image/gif"),
    ".webp": ("image", "image/webp"),
    # Videos
    ".mp4": ("video", "video/mp4"),
    ".mpeg": ("video", "video/mpeg"),
    ".mov": ("video", "video/quicktime"),
    ".webm": ("video", "video/webm"),
    # Audio
    ".mp3": ("audio", "audio/mpeg"),
    ".wav": ("audio", "audio/wav"),
    # PDF
    ".pdf": ("pdf", "application/pdf"),
    # Text
    ".txt": ("text", "text/plain"),
    ".md": ("text", "text/markdown"),
}


def _create_content_block(path: Path) -> dict:
    """Create appropriate content block based on file extension.

    Args:
        path: File path (must exist)

    Returns:
        Content block dictionary

    Raises:
        ValueError: If file type is unsupported
    """
    suffix = path.suffix.lower()

    if suffix not in EXTENSION_MAP:
        supported = ", ".join(sorted(EXTENSION_MAP.keys()))
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {supported}")

    file_type, mime_type = EXTENSION_MAP[suffix]

    # Text files read as text
    if file_type == "text":
        text_content = path.read_text(encoding="utf-8")
        if suffix == ".md":
            return create_plaintext_block(text=text_content, mime_type=mime_type)
        return create_text_block(text=text_content)

    # Binary files encode as base64
    data = base64.b64encode(path.read_bytes()).decode("utf-8")

    if file_type == "image":
        return create_image_block(base64=data, mime_type=mime_type)
    elif file_type == "video":
        return create_video_block(base64=data, mime_type=mime_type)
    elif file_type == "audio":
        return create_audio_block(base64=data, mime_type=mime_type)
    elif file_type == "pdf":
        return create_file_block(base64=data, mime_type=mime_type)

    raise ValueError(f"Unhandled file type: {file_type}")


class MediaMessage(HumanMessage):
    """HumanMessage with media file content using LangChain content blocks.

    Automatically detects file type and uses appropriate content block helpers.
    Supports images, videos, audio, PDFs, and text files.
    """

    def __init__(
        self,
        paths: str | Path | list[str | Path],
        description: str = "",
    ):
        """Initialize MediaMessage with media file(s).

        Args:
            paths: Single file path or list of file paths
            description: Optional description text to include with the media

        Raises:
            ValueError: If file type is unsupported
            FileNotFoundError: If any file path does not exist
        """
        path_list = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]

        content_blocks = []
        for path in path_list:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            content_blocks.append(_create_content_block(path))

        if description:
            content_blocks.append(create_text_block(text=description))

        super().__init__(content_blocks=content_blocks)
