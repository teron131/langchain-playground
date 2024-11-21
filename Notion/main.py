from block_formatter import convert_latex_block
from notion_io import NotionAPI


def convert_latex_blocks():
    print("Starting to fetch Notion data...")

    notion_api = NotionAPI(PAGE_ID="145bb2c6d1338027830cd4a587baf1fc")
    page_content = notion_api.read_blocks()
    print(f"Retrieved {len(page_content)} blocks from Notion page")

    # Filter blocks that have rich_text
    blocks_with_rich_text = [block for block in page_content if block.get(block["type"]).get("rich_text")]
    print(f"Found {len(blocks_with_rich_text)} blocks with rich_text")

    # Convert and update each block with rich_text
    for block in blocks_with_rich_text:
        converted_block = convert_latex_block(block)
        rich_text = converted_block[block["type"]]["rich_text"]

        # Use the notion_api instance to call update_block_rich_text
        response = notion_api.update_block_rich_text(block, rich_text)
        print(f"response: {response}")


if __name__ == "__main__":
    convert_latex_blocks()
