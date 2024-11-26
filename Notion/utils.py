import re
from typing import Dict, List, Tuple

# Constants
RICH_TEXT_BLOCKS = frozenset({"paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item", "to_do", "toggle", "quote", "callout"})

DEFAULT_ANNOTATIONS = {
    "bold": False,
    "italic": False,
    "strikethrough": False,
    "underline": False,
    "code": False,
    "color": "default",
}


def is_rich_text_block(block_type: str) -> bool:
    """
    Check if a block type supports rich text according to Notion API.

    Args:
        block_type (str): The type of the block to check

    Returns:
        bool: True if the block type supports rich text, False otherwise
    """
    return block_type in RICH_TEXT_BLOCKS


## BLOCK TO MARKDOWN
def blocks_to_markdown(blocks: List[Dict], level: int = 0, number_stack: List[int] = None) -> str:
    """
    Convert Notion blocks to markdown while maintaining proper list numbering and nesting.
    """
    markdown_list = []
    if number_stack is None:
        number_stack = [0] * 10  # Support up to 10 levels of nesting

    for block in blocks:
        block_type = block["type"]
        if is_rich_text_block(block_type):
            content = ""
            indent = "  " * level

            # Add markdown formatting based on block type
            if block_type.startswith("heading_"):
                content = "#" * int(block_type[-1]) + " "
            elif block_type == "bulleted_list_item":
                content = indent + "- "
            elif block_type == "numbered_list_item":
                number_stack[level] += 1
                content = f"{indent}{number_stack[level]}. "

            # Get the text content with annotations
            for rich_text in block[block_type]["rich_text"]:
                text = ""
                if rich_text["type"] == "text":
                    text = rich_text["text"]["content"]
                elif rich_text["type"] == "equation":
                    expression = rich_text["equation"]["expression"]
                    text = f"\\({expression}\\)"

                # Apply annotations
                annotations = rich_text["annotations"]
                if annotations["code"]:
                    text = f"`{text}`"
                if annotations["bold"]:
                    text = f"**{text}**"
                if annotations["italic"]:
                    text = f"*{text}*"
                if annotations["strikethrough"]:
                    text = f"~~{text}~~"
                if annotations["underline"]:
                    text = f"__{text}__"

                content += text

            markdown_list.append(content)

            # Process child blocks recursively if they exist
            if block.get("has_children") and "children" in block:
                if level + 1 < len(number_stack):
                    number_stack[level + 1] = 0
                child_content = blocks_to_markdown(block["children"], level + 1, number_stack)
                markdown_list.append(child_content)

    return "\n".join(filter(None, markdown_list))


## MARKDOWN TO BLOCK
# New class for math handling
class MathParser:
    """Handles parsing and processing of mathematical expressions in text."""

    def __init__(self):
        self.inline_math_pattern = r"\\\((.*?)\\\)|\$([^$\n]+?)\$"
        self.block_math_pattern = r"\\\[(.*?)\\\]|\$\$([^$\n]+?)\$\$"
        self._math_pattern = f"({self.inline_math_pattern}|{self.block_math_pattern})"

    def split_text(self, text: str) -> List[str]:
        """Split text by math expressions."""
        return re.split(f"({self._math_pattern})", text)

    def is_math_expression(self, text: str) -> bool:
        """Check if text is a math expression."""
        return bool(re.match(self.inline_math_pattern, text) or re.match(self.block_math_pattern, text))

    def extract_expression(self, text: str) -> str:
        """Extract the math expression from text."""
        match = re.match(self.inline_math_pattern, text) or re.match(self.block_math_pattern, text)
        return next((g for g in match.groups() if g is not None), "") if match else ""


# Rich text creation helpers
def create_rich_text(content: str, annotations: Dict = None) -> Dict:
    if annotations is None:
        annotations = {
            "bold": False,
            "italic": False,
            "strikethrough": False,
            "underline": False,
            "code": False,
            "color": "default",
        }

    return {
        "type": "text",
        "text": {"content": content, "link": None},
        "plain_text": content,
        "annotations": annotations,
        "href": None,
    }


def create_rich_text_equation(expr: str) -> Dict:
    return {
        "type": "equation",
        "equation": {"expression": expr},
        "annotations": {
            "bold": False,
            "italic": False,
            "strikethrough": False,
            "underline": False,
            "code": False,
            "color": "default",
        },
        "href": None,
    }


def create_equation_block(lines: List[str], start_index: int) -> Dict:
    equation_lines = []
    i = start_index + 1
    while i < len(lines) and lines[i] != "\\]" and lines[i] != "\\)":
        equation_lines.append(lines[i])
        i += 1
    expression = "\n".join(equation_lines)
    return {
        "type": "paragraph",
        "paragraph": {
            "rich_text": [create_rich_text_equation(expression)],
            "color": "default",
        },
    }


# Text parsing functions
def annotate_text(text: str, added_equations: set) -> List[Dict]:
    """
    Parse text for markdown formatting annotations and convert to rich text objects, handling bold formatting and other annotations.

    Args:
        text (str): The text to parse for formatting annotations
        added_equations (set): Set of equations that have already been processed to avoid duplicates

    Returns:
        List[Dict]: List of rich text objects with appropriate formatting annotations
    """
    rich_text_list = []
    bold_parts = re.split(r"(\*\*.*?\*\*)", text)

    for bold_part in bold_parts:
        if not bold_part:
            continue

        if bold_part.startswith("**") and bold_part.endswith("**"):
            text_content = bold_part[2:-2]
            rich_text_list.append(create_rich_text(text_content, {"bold": True}))
            continue

        # Handle other annotations
        annotations = {
            "bold": False,
            "italic": False,
            "strikethrough": False,
            "underline": False,
            "code": False,
            "color": "default",
        }
        current_text = bold_part

        for marker, format_type, length in [("`", "code", 1), ("*", "italic", 1), ("~~", "strikethrough", 2), ("__", "underline", 2)]:
            if current_text.startswith(marker) and current_text.endswith(marker):
                current_text = current_text[length:-length]
                annotations[format_type] = True

        if current_text and current_text not in added_equations:
            rich_text_list.append(create_rich_text(current_text, annotations))

    return rich_text_list


def markdown_to_rich_text(text: str) -> List[Dict]:
    """
    Parse inline markdown formatting and convert to rich text objects.

    Args:
        text (str): The markdown text to parse

    Returns:
        List[Dict]: List of rich text objects with appropriate formatting
    """
    rich_text_list = []
    added_equations = set()
    math_parser = MathParser()

    parts = math_parser.split_text(text)

    for part in parts:
        if not part:
            continue

        if math_parser.is_math_expression(part):
            expr = math_parser.extract_expression(part)
            if expr and expr not in added_equations:
                rich_text_list.append(create_rich_text_equation(expr))
                added_equations.add(expr)
            continue

        rich_text_list.extend(annotate_text(part, added_equations))

    return rich_text_list


# Block conversion functions
def markdown_to_block(line: str) -> Dict:
    """
    Parse a single line of markdown text and convert it to a Notion block.

    Args:
        line (str): A line of markdown text to parse

    Returns:
        Dict: A Notion block object representing either a bulleted list item, numbered list item, or paragraph
    """
    if line.startswith("- "):
        content = line[2:]
        return {
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": markdown_to_rich_text(content), "color": "default"},
        }
    elif re.match(r"^\d+\.\s+(.+)$", line):
        content = re.match(r"^\d+\.\s+(.+)$", line).group(1)
        return {
            "type": "numbered_list_item",
            "numbered_list_item": {"rich_text": markdown_to_rich_text(content), "color": "default"},
        }
    else:
        return {
            "type": "paragraph",
            "paragraph": {"rich_text": markdown_to_rich_text(line), "color": "default"},
        }


def markdown_to_blocks(markdown: str) -> List[Dict]:
    """
    Convert markdown text to Notion blocks.

    Args:
        markdown (str): The markdown text to convert

    Returns:
        List[Dict]: List of Notion blocks
    """
    blocks = []
    lines = markdown.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue

        if line == "\\[":
            blocks.append(create_equation_block(lines, i))
            while i < len(lines) and lines[i] != "\\]":
                i += 1
            i += 1
            continue

        blocks.append(markdown_to_block(line))
        i += 1

    return blocks
