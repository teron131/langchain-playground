"""Multimodal input utilities for OpenAI Response API.

Creates content blocks compatible with OpenAI's Response API format.
Reference: https://platform.openai.com/docs/api-reference/responses

Supported content block types:
    - input_text:  {"type": "input_text", "text": "..."}
    - input_image: {"type": "input_image", "image_url": "data:mime;base64,..."}
    - input_audio: {"type": "input_audio", "data": "base64...", "format": "mp3|wav"}
    - input_video: {"type": "input_video", "video_url": "data:mime;base64,..."}
    - input_file:  {"type": "input_file", "file_data": "data:mime;base64,..."}
"""

import base64
from pathlib import Path

from langchain_core.messages import HumanMessage

# Supported file types: extension -> (category, mime_type)
SUPPORTED_EXTENSIONS: dict[str, tuple[str, str]] = {
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
    # Documents
    ".pdf": ("file", "application/pdf"),
    # Text
    ".txt": ("text", "text/plain"),
    ".md": ("text", "text/markdown"),
}


class MediaMessage(HumanMessage):
    """HumanMessage with media content for OpenAI Response API.

    Automatically detects file types and creates appropriate content blocks.
    Supports: images (.jpg, .png, .gif, .webp), videos (.mp4, .mov, .webm),
    audio (.mp3, .wav), documents (.pdf), and text (.txt, .md).

    Example:
        >>> msg = MediaMessage("image.png", "What's in this image?")
        >>> msg = MediaMessage(["doc.pdf", "chart.png"], "Summarize these")
    """

    def __init__(
        self,
        paths: str | Path | list[str | Path],
        description: str = "",
    ):
        """Initialize MediaMessage with file(s) and optional description.

        Args:
            paths: Single path or list of paths to media files
            description: Text prompt to include after media content

        Raises:
            FileNotFoundError: If any file does not exist
            ValueError: If any file type is unsupported
        """
        path_list = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]

        content_blocks = []
        for path in path_list:
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                supported = ", ".join(sorted(SUPPORTED_EXTENSIONS.keys()))
                raise ValueError(f"Unsupported: {suffix}. Supported: {supported}")

            category, mime_type = SUPPORTED_EXTENSIONS[suffix]

            if category == "text":
                content_blocks.append({"type": "input_text", "text": path.read_text(encoding="utf-8")})
            else:
                data = base64.b64encode(path.read_bytes()).decode("utf-8")
                if category == "image":
                    content_blocks.append({"type": "input_image", "image_url": f"data:{mime_type};base64,{data}"})
                elif category == "video":
                    content_blocks.append({"type": "input_video", "video_url": f"data:{mime_type};base64,{data}"})
                elif category == "audio":
                    content_blocks.append({"type": "input_audio", "data": data, "format": "wav" if suffix == ".wav" else "mp3"})
                elif category == "file":
                    content_blocks.append({"type": "input_file", "file_data": f"data:{mime_type};base64,{data}"})
                else:
                    raise ValueError(f"Unhandled category: {category}")

        if description:
            content_blocks.append({"type": "input_text", "text": description})

        super().__init__(content=content_blocks)
