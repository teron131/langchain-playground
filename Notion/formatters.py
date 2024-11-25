import re
from typing import Dict, List

from tqdm import tqdm

from notion_api import NotionAPI
from utils import is_rich_text_block, text_to_equation, text_to_text


class LatexFormatter:
    LATEX_PATTERN = re.compile(r"(.*?)(\\\(|\\\[|\$\$)(.*?)(\\\)|\\\]|\$\$)|(.+)$")

    def __init__(self, notionapi: NotionAPI):
        self.notionapi = notionapi

    def _convert_rich_text(self, rich_text: Dict) -> List[Dict]:
        """
        Reference: https://developers.notion.com/reference/rich-text

        Process rich text content to extract LaTeX equations and convert them to Notion equation blocks.

        Takes a rich text object containing potential LaTeX equations and splits it into separate text and equation objects while preserving formatting. Equations are identified by LaTeX delimiters (\\(\\), \\[\\], or $$) and converted to Notion's native equation format.

        Args:
            rich_text (Dict): A Notion rich text object containing text content and formatting

        Returns:
            List[Dict]: List of text and equation objects with preserved formatting. Each object is either a text object containing regular text or an equation object containing the LaTeX expression.

        Example:
            Input: rich_text with text "x = \\(a + b\\) where a,b > 0"
            Returns: [
                {type: "text", text: {content: "x = "}, ...},
                {type: "equation", equation: {expression: "a + b"}, ...},
                {type: "text", text: {content: " where a,b > 0"}, ...}
            ]
        """
        content = rich_text["text"]["content"]
        result = []

        for match in self.LATEX_PATTERN.finditer(content):
            before_text, start_delim, equation_content, end_delim, remaining_text = match.groups()
            if before_text:
                result.append(text_to_text(rich_text, before_text))
            if equation_content:
                result.append(text_to_equation(rich_text, equation_content))
            if remaining_text:
                result.append(text_to_text(rich_text, remaining_text))
                break

        return result

    def convert_block(self, block: Dict) -> Dict:
        """
        Reference: https://developers.notion.com/reference/block

        Processes a block containing LaTeX equations delimited by \\( \\), \\[ \\], or $$ and converts them into Notion's native equation format. Preserves text formatting.

        Args:
            block (Dict): A Notion block object that may contain LaTeX equations in its rich text content.

        Returns:
            Dict: The block with any LaTeX equations converted to Notion's native equation format.
        """
        block_type = block["type"]
        if not is_rich_text_block(block_type):
            return block

        rich_text_list = block[block_type].get("rich_text")
        new_rich_text_list = []

        for rich_text in rich_text_list:
            if rich_text["type"] == "text":
                new_rich_text_list.extend(self._convert_rich_text(rich_text))
            else:
                new_rich_text_list.append(rich_text)

        block[block_type]["rich_text"] = new_rich_text_list

        return block

    def convert_blocks(self, blocks: List[Dict]) -> None:
        """
        Convert LaTeX equations in blocks and update them in Notion.

        Args:
            blocks (List[Dict]): A list of Notion blocks that may contain LaTeX equations.
        """
        with tqdm(blocks, desc="Converting LaTeX equations") as pbar:
            for block in pbar:
                block_type = block["type"]
                if is_rich_text_block(block_type):
                    converted_block = self.convert_block(block)
                    new_rich_text = converted_block[block_type]["rich_text"]
                    self.notionapi.update_block_rich_text(block, new_rich_text)
