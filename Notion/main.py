from typing import List

from formatters import BaseFormatter, LatexFormatter, Rephraser
from notion_api import NotionAPI


def process_with_formatters(blocks: List[dict], formatters: List[BaseFormatter]) -> None:
    """
    Process blocks through a sequence of formatters.

    Args:
        blocks (List[dict]): Notion blocks to process
        formatters (List[BaseFormatter]): List of formatters to apply
    """
    for formatter in formatters:
        formatter.process_blocks(blocks)


if __name__ == "__main__":
    notion_api = NotionAPI()
    blocks = notion_api.read_blocks()

    formatters = [LatexFormatter(notion_api), Rephraser(notion_api)]

    process_with_formatters(blocks, formatters)
