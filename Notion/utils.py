from typing import Dict, List


def is_rich_text_block(block_type: str) -> bool:
    """
    Check if a block type supports rich text according to Notion API.

    Args:
        block_type (str): The type of the block to check

    Returns:
        bool: True if the block type supports rich text, False otherwise
    """
    RICH_TEXT_BLOCKS = set({"paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item", "numbered_list_item", "to_do", "toggle", "quote", "callout"})
    return block_type in RICH_TEXT_BLOCKS


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
