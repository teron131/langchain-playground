from concurrent.futures import ThreadPoolExecutor
import os

from docling.document_converter import DocumentConverter
from dotenv import load_dotenv

load_dotenv()


def webloader_docling(urls: list[str]) -> list[str | None]:
    """Load and process website content from URLs into markdown."""
    converter = DocumentConverter()

    def _convert(url: str) -> str | None:
        try:
            return converter.convert(url).document.export_to_markdown()
        except Exception:
            return None

    max_workers = min(len(urls), os.cpu_count())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_convert, urls))
