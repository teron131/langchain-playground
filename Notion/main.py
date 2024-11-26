from typing import List

from IPython.display import display

from formatters import BaseFormatter, LatexFormatter, Rephraser
from notion_api import NotionAPI
from markdown import blocks_to_markdown, markdown_to_blocks


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
\[ x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}. \)
"""
    display(markdown)
    # markdown = notion_api.read_blocks_markdown()
    blocks = markdown_to_blocks(markdown)
    print(blocks)
    notion_api.write_blocks(blocks)
