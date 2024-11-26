from typing import List

from formatters import BaseFormatter, LatexFormatter, Rephraser
from notion_api import NotionAPI
from utils import blocks_to_markdown, markdown_to_blocks


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
    # blocks = notion_api.read_blocks()

    markdown = """
- **Inequality constraints**: \\( A_{ineq} x \\leq b_{ineq} \\)\n- \\( A_{eq} \\) and \\( A_{ineq} \\) are matrices defining the linear constraints.
"""
    # markdown = notion_api.read_blocks_markdown()
    blocks = markdown_to_blocks(markdown)
    print(blocks)
    notion_api.write_blocks(blocks)
