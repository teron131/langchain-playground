from typing import Dict


def create_text_part(rich_text: Dict, new_content: str) -> Dict:
    """
    Create a text rich_text content from content while preserving other rich_text properties.

    Modifies only the content and type, keeping other properties from the original rich_text object.
    """
    return {
        "type": "text",
        "text": {"content": new_content, "link": None},
        "plain_text": new_content,
        "annotations": rich_text["annotations"],
        "href": rich_text["href"],
    }


def create_equation_part(rich_text: Dict, new_content: str) -> Dict:
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
