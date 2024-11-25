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


def blocks_to_str(blocks: List[Dict]) -> str:
    content_list = []

    for block in blocks:
        block_type = block["type"]
        if is_rich_text_block(block_type):
            content = ""
            for rich_text in block[block_type]["rich_text"]:
                rich_text_type = rich_text["type"]
                if rich_text_type == "text":
                    content += rich_text["text"]["content"]
                elif rich_text_type == "equation":
                    content += rich_text["equation"]["expression"]
            content_list.append(content)

    return "\n".join(content_list)
