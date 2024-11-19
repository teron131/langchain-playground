import os
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from notion_client import Client

load_dotenv()


class NotionBlockManager:
    def __init__(self):
        self.notion_token = os.getenv("NOTION_TOKEN")
        self.page_id = os.getenv("PAGE_ID")
        self.headers = {"Authorization": f"Bearer {self.notion_token}", "Notion-Version": "2022-06-28", "Content-Type": "application/json"}
        self.notion = Client(auth=self.notion_token)

    def get_all_blocks(self, block_id: str) -> List[Dict]:
        """Recursively fetch all blocks including children"""
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        blocks = []
        has_more = True
        start_cursor = None

        while has_more:
            params = {"start_cursor": start_cursor} if start_cursor else {}
            response = requests.get(url, headers=self.headers, params=params)

            if response.status_code != 200:
                print(f"Error fetching blocks: {response.status_code} - {response.text}")
                return []

            data = response.json()
            blocks.extend(data.get("results", []))

            # Recursively get child blocks
            for block in data.get("results", []):
                if block.get("has_children", False):
                    blocks.extend(self.get_all_blocks(block["id"]))

            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor")

        return blocks

    def retrieve_page(self, page_id: str) -> Optional[Dict]:
        """Retrieve a Notion page by ID"""
        url = f"https://api.notion.com/v1/pages/{page_id}"
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            print(f"Error retrieving page: {response.status_code} - {response.text}")
            return None

        return response.json()

    def blocks_to_dataframe(self, blocks: List[Dict]) -> pd.DataFrame:
        """Convert Notion blocks to pandas DataFrame"""
        data = []
        for block in blocks:
            block_type = block["type"]
            content = self._extract_block_content(block, block_type)
            data.append({"id": block["id"], "type": block_type, "content": content})

        return pd.DataFrame(data)

    def _extract_block_content(self, block: Dict, block_type: str) -> str:
        """Extract content from different block types"""
        content = ""

        if "rich_text" in block.get(block_type, {}):
            content = self._process_rich_text(block[block_type]["rich_text"])
        elif block_type == "code":
            content = block["code"]["text"][0]["text"]["content"]
        elif block_type == "quote":
            content = self._process_rich_text(block["quote"]["rich_text"])

        return content

    def _process_rich_text(self, rich_text: List[Dict]) -> str:
        """Process rich text content including equations"""
        content = ""
        for item in rich_text:
            if item["type"] == "text":
                content += item["text"]["content"]
            elif item["type"] == "equation":
                content += f"$$ {item['equation']['expression']} $$"
        return content

    def format_content_for_notion(self, block: Any) -> List[Dict]:
        """Format content for Notion API"""
        if isinstance(block, str):
            parts = block.split("$$")
            formatted_parts = []

            for i, part in enumerate(parts):
                if i % 2 == 1:  # Equation parts
                    formatted_parts.append({"type": "equation", "equation": {"expression": part.strip()}})
                elif part:  # Text parts
                    formatted_parts.append({"type": "text", "text": {"content": part}})

            return formatted_parts
        return block

    def combine_text_and_equations(self, df: pd.DataFrame) -> List[Dict]:
        """Combine text and equations into Notion blocks"""
        combined_blocks = [{"type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": "Hi! 👋"}}]}}]

        block_type_handlers = {
            "divider": lambda content: {"type": "divider", "divider": {}},
            "heading_1": lambda content: {"type": "heading_1", "heading_1": {"rich_text": content}},
            "heading_2": lambda content: {"type": "heading_2", "heading_2": {"rich_text": content}},
            "heading_3": lambda content: {"type": "heading_3", "heading_3": {"rich_text": content}},
            "quote": lambda content: {"type": "quote", "quote": {"rich_text": content}},
            "paragraph": lambda content: {"type": "paragraph", "paragraph": {"rich_text": content}},
            "code": lambda content: {"type": "code", "code": {"text": content, "language": "python"}},
            "bulleted_list_item": lambda content: {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": content}},
        }

        for _, row in df.iterrows():
            content = self.format_content_for_notion(row["content"])
            if handler := block_type_handlers.get(row["type"]):
                if content or row["type"] == "divider":
                    combined_blocks.append(handler(content))

        return combined_blocks

    def upload_to_notion(self, page_id: str, combined_blocks: List[Dict]) -> Dict:
        """Upload blocks to Notion page"""
        url = f"https://api.notion.com/v1/blocks/{page_id}/children"
        payload = {"children": combined_blocks}
        response = requests.patch(url, json=payload, headers=self.headers)
        return response.json()


def main():
    manager = NotionBlockManager()
    print("Starting to fetch Notion data...")

    page_data = manager.retrieve_page(manager.page_id)
    if not page_data:
        print("Failed to retrieve page")
        return

    print("Page retrieved successfully")
    page_content = manager.get_all_blocks(manager.page_id)
    print(f"Retrieved {len(page_content)} blocks from Notion page")

    df = manager.blocks_to_dataframe(page_content)
    combined_data = manager.combine_text_and_equations(df)
    print("Processed and combined text and equations")

    response = manager.upload_to_notion(manager.page_id, combined_data)
    print("Upload complete!")

    # Debug prints
    print(f"NOTION_TOKEN: {manager.notion_token}")
    print(f"PAGE_ID: {manager.page_id}")


if __name__ == "__main__":
    main()
