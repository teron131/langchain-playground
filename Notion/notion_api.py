import os
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from markdown import blocks_to_markdown
from utils import is_rich_text_block

load_dotenv()


class NotionAPI:
    def __init__(self, PAGE_ID: str = None):
        """
        Initialize NotionAPI client with authentication token and page ID.
        Make sure to connect the page to the integration associated with the token.
        The schemas of the APIs are from https://developers.notion.com/reference

        Args:
            TOKEN (str): Notion API authentication token, from https://developers.notion.com
            PAGE_ID (str): ID of the Notion page to interact with, from the last part (without title) of the page URL, e.g. https://www.notion.so/USERNAME/TITLE-PAGE_ID
        """
        self.token = os.getenv("NOTION_TOKEN")
        self.headers = {"Authorization": f"Bearer {self.token}", "Notion-Version": "2022-06-28"}
        self.page_id = os.getenv("PAGE_ID") if PAGE_ID is None else PAGE_ID

    def read_blocks(self, block_id: Optional[str] = None) -> List[Dict]:
        """
        Recursively reads all blocks and their children from a Notion page using cursor-based pagination.
        """
        if block_id is None:
            block_id = self.page_id

        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        blocks = []
        next_cursor = None

        while True:
            params = {"start_cursor": next_cursor} if next_cursor else {}
            response = requests.get(url, headers=self.headers, params=params)
            data = response.json()
            results = data.get("results", [])

            for block in results:
                if block.get("has_children"):
                    children = self.read_blocks(block["id"])
                    if children:
                        block["children"] = children
                blocks.append(block)

            if not data.get("has_more"):
                break

            next_cursor = data.get("next_cursor")

        return blocks

    def read_blocks_markdown(self, block_id: Optional[str] = None) -> str:
        """
        Read all text content from a Notion page.

        Returns:
            str: A string containing the text content of the page.
        """
        if block_id is None:
            block_id = self.page_id

        blocks = self.read_blocks(block_id)

        return blocks_to_markdown(blocks)

    def write_blocks(self, new_blocks: List[Dict]) -> Dict:
        """
        Reference: https://developers.notion.com/reference/patch-block-children

        Write blocks to a Notion page using the Notion API.
        Makes a PATCH request to append blocks as children of the specified page.

        Args:
            blocks (List[Dict]): A list of block objects to write to the page. Each block should follow the Notion API block object format.

        Returns:
            Dict: A paginated list of newly created first level children block objects.
        """
        supported_blocks = []
        for block in new_blocks:
            # Skip link_preview blocks as they're not directly supported
            if block.get(block["type"]).get("rich_text"):
                supported_blocks.append(block)

        url = f"https://api.notion.com/v1/blocks/{self.page_id}/children"
        payload = {"children": supported_blocks}
        response = requests.patch(url, json=payload, headers=self.headers)
        return response.json()

    def update_block_rich_text(self, block: Dict, new_rich_text: List[Dict]) -> Dict:
        """
        Reference: https://developers.notion.com/reference/update-a-block

        Updates the rich_text field of a block according to its type.

        Args:
            block (Dict): The block to update, containing 'id' and 'type' fields
            rich_text (List[Dict]): The new rich_text content to set

        Returns:
            Dict: The updated block object.
        """
        block_id = block["id"]
        block_type = block["type"]
        url = f"https://api.notion.com/v1/blocks/{block_id}"
        payload = {block_type: {"rich_text": new_rich_text}}
        response = requests.patch(url, json=payload, headers=self.headers)
        return response.json()
