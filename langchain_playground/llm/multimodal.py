"""Multimodal input utilities for OpenAI Chat Completions API.

Creates content blocks compatible with OpenAI's Chat Completions API format (used by OpenRouter).
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
    # Videos (mapped to image_url for vision models)
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
    """HumanMessage with media content for Chat Completions API.

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
                # Plain text content
                content_blocks.append({"type": "text", "text": path.read_text(encoding="utf-8")})
            else:
                data = base64.b64encode(path.read_bytes()).decode("utf-8")
                data_url = f"data:{mime_type};base64,{data}"

                if category == "image":
                    # Standard image_url format for Chat Completions API
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
                elif category == "video":
                    # Video as image_url (some vision models support this)
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
                elif category == "audio":
                    # Audio input (model-specific support)
                    content_blocks.append(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": data, "format": "wav" if suffix == ".wav" else "mp3"},
                        }
                    )
                elif category == "file":
                    # OpenRouter file format for PDFs
                    content_blocks.append(
                        {
                            "type": "file",
                            "file": {"filename": path.name, "file_data": data_url},
                        }
                    )
                else:
                    raise ValueError(f"Unhandled category: {category}")

        if description:
            content_blocks.append({"type": "text", "text": description})

        super().__init__(content=content_blocks)
