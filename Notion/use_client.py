import os

from dotenv import load_dotenv

from block_formatter import convert_latex_block
from notion_client import Client

load_dotenv()

notion = Client(auth=os.getenv("NOTION_TOKEN"))
page_id = "145bb2c6d1338027830cd4a587baf1fc"


def get_all_blocks(page_id):
    blocks = []
    start_cursor = None

    while True:
        response = notion.blocks.children.list(page_id, start_cursor=start_cursor)
        blocks.extend(response["results"])

        # Check if there are more pages
        if not response.get("has_more"):
            break

        # Update the start_cursor for the next page
        start_cursor = response.get("next_cursor")

    return blocks


# Convert LaTeX in blocks and update them
all_blocks = get_all_blocks(page_id)
for block in all_blocks:
    block_type = block.get("type")
    rich_text_list = block.get(block_type).get("rich_text")

    if rich_text_list:
        converted_block = convert_latex_block(block)
        notion.blocks.update(block["id"], **converted_block)
        print(f"Block {block['id']} updated with LaTeX conversion.")
