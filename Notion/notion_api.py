import os
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class NotionAPI:
    def __init__(self, PAGE_ID: str):
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
        if block_id is None:
            block_id = self.page_id
        """
        Reference: https://developers.notion.com/reference/get-block-children

        Recursively read all blocks and their children from a Notion page.
        Makes paginated API requests to fetch blocks from a Notion page, including all nested child blocks.
        Uses cursor-based pagination to handle large pages.

        Args:
            block_id (str): The ID of the Notion block/page to read from

        Returns:
            List[Dict]: A paginated list of block objects.
        """
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        blocks = []
        start_cursor = None

        while True:
            params = {"start_cursor": start_cursor} if start_cursor else {}
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code != 200:
                print(f"Error fetching blocks: {response.status_code} - {response.text}")
                return []

            data = response.json()
            results = data.get("results", [])
            blocks.extend(results)

            # Get child blocks recursively
            child_blocks = [child_block for block in results if block.get("has_children") for child_block in self.read_blocks(block["id"])]
            blocks.extend(child_blocks)

            if not data.get("has_more"):
                break

            start_cursor = data.get("start_cursor")

        return blocks

    def write_blocks(self, new_blocks: List[Dict]) -> Dict:
        """
        Reference: https://developers.notion.com/reference/patch-block-children

        Write blocks to a Notion page using the Notion API.
        Makes a PATCH request to append blocks as children of the specified page.

        Args:
            blocks (List[Dict]): A list of block objects to write to the page.
                               Each block should follow the Notion API block object format.

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
