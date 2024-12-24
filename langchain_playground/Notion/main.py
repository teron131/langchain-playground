from typing import List

from IPython.display import display

from .formatters import BaseFormatter, LatexFormatter, Rephraser
from .markdown import blocks_to_markdown, markdown_to_blocks
from .notion_api import NotionAPI


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
    rephraser = Rephraser(notion_api)
    blocks = notion_api.read_blocks()
    markdown = blocks_to_markdown(blocks)
    rephrased_markdown = rephraser.rephrase_text(markdown)
    rephrased_blocks = markdown_to_blocks(rephrased_markdown)
    notion_api.write_blocks(rephrased_blocks)
