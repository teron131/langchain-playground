import re
from typing import Dict, List, Optional, Tuple

from utils import is_rich_text_block


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

    # Class level patterns
    INLINE_MATH_PATTERN = r"(?:\\\(.*?\\\)|\$[^$\n]+?\$)"
    BLOCK_MATH_PATTERN = r"(?:\\\[.*?\\\]|\$\$.*?\$\$)"
    ENV_PATTERN = r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}"
    _MATH_PATTERN = f"{INLINE_MATH_PATTERN}|{BLOCK_MATH_PATTERN}|{ENV_PATTERN}"

    @classmethod
    def split_text(cls, text: str) -> List[str]:
        """Split text by math expressions."""
        return re.split(f"({cls._MATH_PATTERN})", text, flags=re.DOTALL)

    @classmethod
    def is_math_expression(cls, text: str) -> bool:
        """Check if text is a math expression."""
        return bool(re.match(f"^(?:{cls._MATH_PATTERN})$", text, re.DOTALL))

    @classmethod
    def extract_expression(cls, text: str) -> str:
        """Extract the math expression from text."""
        # Extract content between delimiters
        if text.startswith("\\begin{"):
            match = re.match(r"\\begin\{([^}]+)\}(.*?)\\end\{\1\}", text, re.DOTALL)
            return match.group(2).strip() if match else ""
        elif text.startswith("\\(") and text.endswith("\\)"):
            return text[2:-2].strip()
        elif text.startswith("\\[") and text.endswith("\\]"):
            return text[2:-2].strip()
        elif text.startswith("$") and text.endswith("$"):
            if text.startswith("$$") and text.endswith("$$"):
                return text[2:-2].strip()
            return text[1:-1].strip()
        return ""

    @classmethod
    def is_math_environment(cls, text: str) -> bool:
        """Check if text contains a math environment."""
        return bool(re.match(r"\\begin\{[^}]+\}.*?\\end\{[^}]+\}", text, re.DOTALL))


class RichTextAnnotator:
    """Handles rich text annotation processing and creation."""

    DEFAULT_ANNOTATIONS = {
        "bold": False,
        "italic": False,
        "strikethrough": False,
        "underline": False,
        "code": False,
        "color": "default",
    }

    ANNOTATION_MARKERS = [("`", "code", 1), ("*", "italic", 1), ("~~", "strikethrough", 2), ("__", "underline", 2)]

    @classmethod
    def create_text(cls, content: str, annotations: Dict = None) -> Dict:
        """Create a rich text object with given content and annotations."""
        return {
            "type": "text",
            "text": {"content": content, "link": None},
            "plain_text": content,
            "annotations": annotations or cls.DEFAULT_ANNOTATIONS.copy(),
            "href": None,
        }

    @classmethod
    def create_equation(cls, expr: str) -> Dict:
        """Create a rich text equation object."""
        return {
            "type": "equation",
            "equation": {"expression": expr},
            "annotations": cls.DEFAULT_ANNOTATIONS.copy(),
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
            "rich_text": [RichTextAnnotator.create_equation(expression)],
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
            rich_text_list.append(RichTextAnnotator.create_text(text_content, {"bold": True}))
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
            rich_text_list.append(RichTextAnnotator.create_text(current_text, annotations))

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

    parts = MathParser.split_text(text)

    for part in parts:
        if not part:
            continue

        if MathParser.is_math_expression(part):
            if MathParser.is_math_environment(part):
                # Skip environment processing here as it will be handled at block level
                continue

            expr = MathParser.extract_expression(part)
            if expr and expr not in added_equations:
                rich_text_list.append(RichTextAnnotator.create_equation(expr))
                added_equations.add(expr)
            continue

        rich_text_list.extend(annotate_text(part, added_equations))

    return rich_text_list


class EquationDelimiters:
    """Handles equation delimiters and their relationships."""

    START_DELIMITERS = frozenset({"\\[", "\\(", "\\\\[", "\\\\(", "$$"})
    DELIMITER_PAIRS = {"\\[": "\\]", "\\(": "\\)", "\\\\[": "\\\\]", "\\\\(": "\\\\)", "$$": "$$"}
    END_DELIMITERS = frozenset(DELIMITER_PAIRS.values())

    @classmethod
    def is_start_delimiter(cls, text: str) -> bool:
        """Check if text is a start delimiter."""
        return text in cls.START_DELIMITERS

    @classmethod
    def is_end_delimiter(cls, text: str) -> bool:
        """Check if text is an end delimiter."""
        return text in cls.END_DELIMITERS

    @classmethod
    def get_closing_delimiter(cls, start_delimiter: str) -> str:
        """Get the corresponding closing delimiter for a start delimiter."""
        return cls.DELIMITER_PAIRS.get(start_delimiter)


class EquationBlockParser:
    """Handles equation block parsing and conversion."""

    @staticmethod
    def parse_math_environment(text: str) -> List[Dict]:
        """
        Parse a LaTeX math environment and convert it to equation blocks.

        Args:
            text (str): The text containing the math environment

        Returns:
            List[Dict]: List of equation blocks, one for each line
        """
        math_parser = MathParser()
        if not math_parser.is_math_environment(text):
            return []

        # Extract the content between \begin and \end
        content = math_parser.extract_expression(text)

        # Split into lines and clean up
        equations = [eq.strip() for eq in content.split("\\\\") if eq.strip()]

        # Create equation blocks
        blocks = []
        for eq in equations:
            blocks.append({"type": "equation", "equation": {"expression": eq, "color": "default"}})

        return blocks

    @staticmethod
    def parse_equation_block(lines: List[str], start_index: int) -> Tuple[Optional[Dict], int]:
        """
        Parse an equation block from the given lines starting at start_index.

        Args:
            lines (List[str]): List of text lines
            start_index (int): Starting index in the lines list

        Returns:
            Tuple[Optional[Dict], int]: Tuple of (equation block dict or None if invalid, next index to process)
        """
        if start_index >= len(lines):
            return None, start_index

        # Check for math environment
        current_line = lines[start_index]
        if current_line.strip().startswith("\\begin{"):
            # Collect all lines until \end
            env_lines = []
            i = start_index
            while i < len(lines) and "\\end{" not in lines[i]:
                env_lines.append(lines[i])
                i += 1
            if i < len(lines):
                env_lines.append(lines[i])  # Include the \end line

            full_env = "\n".join(env_lines)
            equation_blocks = EquationBlockParser.parse_math_environment(full_env)
            if equation_blocks:
                return equation_blocks[0], i + 1  # Return first block and remaining will be processed later

        start_line = lines[start_index].strip()
        if not EquationDelimiters.is_start_delimiter(start_line):
            return None, start_index

        closing_delimiter = EquationDelimiters.get_closing_delimiter(start_line)
        equation_lines = []
        current_index = start_index + 1
        found_closing = False

        while current_index < len(lines):
            current_line = lines[current_index].strip()
            if current_line == closing_delimiter:
                found_closing = True
                break
            if EquationDelimiters.is_start_delimiter(current_line) or EquationDelimiters.is_end_delimiter(current_line):
                # Found a mismatched delimiter
                break
            equation_lines.append(lines[current_index])
            current_index += 1

        if not found_closing:
            # No matching closing delimiter found, treat as regular text
            return {
                "type": "paragraph",
                "paragraph": {"rich_text": markdown_to_rich_text(start_line), "color": "default"},
            }, start_index + 1

        # Found matching closing delimiter
        expression = "\n".join(equation_lines)
        return {
            "type": "paragraph",
            "paragraph": {
                "rich_text": [RichTextAnnotator.create_equation(expression)],
                "color": "default",
            },
        }, current_index + 1


class TextBlockParser:
    """Handles markdown to text block parsing and conversion."""

    @staticmethod
    def parse_header(line: str) -> Optional[Dict]:
        """Parse header markdown and return Notion block."""
        match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if not match:
            return None

        level = len(match.group(1))
        content = match.group(2)
        if level <= 3:
            return {
                "type": f"heading_{level}",
                f"heading_{level}": {"rich_text": markdown_to_rich_text(content), "color": "default", "is_toggleable": False},
            }
        return {
            "type": "paragraph",
            "paragraph": {"rich_text": markdown_to_rich_text(line), "color": "default"},
        }

    @staticmethod
    def parse_bullet_list(line: str) -> Optional[Dict]:
        """Parse bullet list markdown and return Notion block."""
        match = re.match(r"^(\s*)-\s+(.+)$", line)
        if not match:
            return None

        content = match.group(2)
        return {
            "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": markdown_to_rich_text(content), "color": "default"},
        }

    @staticmethod
    def parse_numbered_list(line: str) -> Optional[Dict]:
        """Parse numbered list markdown and return Notion block."""
        match = re.match(r"^(\s*)\d+\.\s+(.+)$", line)
        if not match:
            return None

        content = match.group(2)
        return {
            "type": "numbered_list_item",
            "numbered_list_item": {"rich_text": markdown_to_rich_text(content), "color": "default"},
        }


def markdown_to_block(line: str) -> Dict:
    """
    Parse a single line of markdown text and convert it to a Notion block.

    Args:
        line (str): A line of markdown text to parse

    Returns:
        Dict: A Notion block object representing a header (h1, h2, h3), bullet list item, numbered list item, or paragraph
    """
    if EquationDelimiters.is_start_delimiter(line.strip()):
        return None

    # Try parsing different block types
    for parser in [TextBlockParser.parse_header, TextBlockParser.parse_bullet_list, TextBlockParser.parse_numbered_list]:
        block = parser(line)
        if block is not None:
            return block

    # Default to paragraph
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
    lines = markdown.split(r"\n" if r"\n" in markdown else "\n")
    i = 0

    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        # Check for math environment
        if lines[i].strip().startswith("\\begin{"):
            env_lines = []
            start_i = i
            while i < len(lines) and "\\end{" not in lines[i]:
                env_lines.append(lines[i])
                i += 1
            if i < len(lines):
                env_lines.append(lines[i])
                full_env = "\n".join(env_lines)
                equation_blocks = EquationBlockParser.parse_math_environment(full_env)
                blocks.extend(equation_blocks)
                i += 1
                continue

        # Try parsing as equation block first
        equation_block, next_index = EquationBlockParser.parse_equation_block(lines, i)
        if equation_block is not None:
            blocks.append(equation_block)
            i = next_index
            continue

        # Parse as regular markdown block
        block = markdown_to_block(lines[i])
        if block:
            blocks.append(block)
        i += 1

    return blocks
