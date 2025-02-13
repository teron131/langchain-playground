from dotenv import load_dotenv

load_dotenv()

from typing import List

from formatters import BaseFormatter, LatexFormatter
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
    notion_api = NotionAPI(PAGE_ID="196bb2c6d133804a910bddd4596647db")
    blocks = notion_api.read_blocks()
    process_with_formatters(blocks, [LatexFormatter(notion_api)])
    notion_api.write_blocks(blocks)
