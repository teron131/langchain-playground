RICH_TEXT_BLOCKS = frozenset(
    {
        "paragraph",
        "heading_1",
        "heading_2",
        "heading_3",
        "bulleted_list_item",
        "numbered_list_item",
        "to_do",
        "toggle",
        "quote",
        "callout",
    }
)


def is_rich_text_block(block_type: str) -> bool:
    """
    Check if a block type supports rich text according to Notion API.

    Args:
        block_type (str): The type of the block to check

    Returns:
        bool: True if the block type supports rich text, False otherwise
    """
    return block_type in RICH_TEXT_BLOCKS
