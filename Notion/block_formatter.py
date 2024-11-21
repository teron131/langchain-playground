import re
from typing import Dict, List


def text_to_text(rich_text: Dict, new_content: str) -> Dict:
    """
    Create a text rich_text content from content while preserving other rich_text properties.

    Modifies only the content and type, keeping other properties from the original rich_text object.
    """
    return {
        "type": "text",
        "text": {"content": new_content, "link": rich_text["text"]["link"]},
        "plain_text": new_content,
        "annotations": rich_text["annotations"],
        "href": rich_text["href"],
    }


def text_to_equation(rich_text: Dict, new_content: str) -> Dict:
    """
    Create an equation rich_text content from content while preserving other rich_text properties.

    Modifies only the content and type, keeping other properties from the original rich_text object.
    """
    return {
        "type": "equation",
        "equation": {"expression": new_content.strip()},
        "plain_text": new_content,
        "annotations": rich_text["annotations"],
        "href": rich_text["href"],
    }


def convert_latex_block(block: Dict) -> Dict:
    """
    Reference: https://developers.notion.com/reference/block

    Processes a block containing LaTeX equations delimited by \\( \\), \\[ \\], or $$ and converts them
    into Notion's native equation format. Preserves text formatting.

    Args:
        block (Dict): A Notion block object containing rich text with LaTeX equations

    Returns:
        Dict: The processed block with LaTeX equations converted to Notion equation blocks.
              Preserves original block structure and formatting.

    Example:
        Input text: "A function \\(f(x)\\) is continuous"
        Output: Two rich_text objects:
            1. Text object with "A function "
            2. Equation object with "f(x)"
            3. Text object with " is continuous"
    """

    def _convert_latex_rich_text(rich_text: Dict) -> List[Dict]:
        """
        Reference: https://developers.notion.com/reference/rich-text

        Process rich text content to extract LaTeX equations and convert them to Notion equation blocks.

        Takes a rich text object containing potential LaTeX equations and splits it into separate
        text and equation objects while preserving formatting. Equations are identified by LaTeX
        delimiters (\\(\\), \\[\\], or $$) and converted to Notion's native equation format.

        Args:
            rich_text (Dict): A Notion rich text object containing text content and formatting

        Returns:
            List[Dict]: List of text and equation objects with preserved formatting. Each object is
                       either a text object containing regular text or an equation object containing
                       the LaTeX expression.

        Example:
            Input rich_text with content "x = \\(a + b\\) where a,b > 0"
            Returns: [
                {type: "text", text: {content: "x = "}, ...},
                {type: "equation", equation: {expression: "a + b"}, ...},
                {type: "text", text: {content: " where a,b > 0"}, ...}
            ]
        """
        content = rich_text["text"]["content"]
        result = []

        # Use a single regex pattern to match LaTeX equations and text
        pattern = r"(.*?)(\\\(|\\\[|\$\$)(.*?)(\\\)|\\\]|\$\$)|(.+)$"

        for match in re.finditer(pattern, content):
            before_text, start_delim, equation_content, end_delim, remaining_text = match.groups()

            # Add text before equation if exists
            if before_text:
                result.append(text_to_text(rich_text, before_text))

            # Add equation part if exists
            if equation_content:
                result.append(text_to_equation(rich_text, equation_content))

            # Add remaining text if exists
            if remaining_text:
                result.append(text_to_text(rich_text, remaining_text))
                break

        return result

    rich_text_list = block.get(block["type"]).get("rich_text")
    # Assume equations only exist in the block types that have rich_text
    if rich_text_list:
        result = []
        for rich_text in rich_text_list:
            if rich_text["type"] == "text":
                result.extend(_convert_latex_rich_text(rich_text))
            else:
                result.append(rich_text)
        block[block["type"]]["rich_text"] = result
    return block


def convert_headers(block: Dict) -> Dict:
    pass
