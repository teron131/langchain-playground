import os
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from notion_client import Client

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
PAGE_ID = "143bb2c6d13380459053f33d84fd6cdb"


class NotionAPI:
    def __init__(self, TOKEN: str, PAGE_ID: str):
        self.token = TOKEN
        self.page_id = PAGE_ID
        self.HEADERS = {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": "2022-06-28",
        }
        self.client = Client(auth=self.token)

    def read_blocks(self, block_id: str) -> List[Dict]:
        """Recursively read all blocks including children from a page by ID"""
        url = f"https://api.notion.com/v1/blocks/{block_id}/children"
        blocks = []
        start_cursor = None

        while True:
            response = requests.get(url, headers=self.HEADERS, params={"start_cursor": start_cursor} if start_cursor else {})

            if response.status_code != 200:
                print(f"Error fetching blocks: {response.status_code} - {response.text}")
                return []

            data = response.json()
            results = data.get("results", [])
            blocks.extend(results)

            # Recursively get child blocks
            for block in results:
                if block.get("has_children", False):
                    blocks.extend(self.read_blocks(block["id"]))

            if not data.get("has_more"):
                break

            start_cursor = data.get("start_cursor")

        return blocks

    def write_blocks(self, blocks: List[Dict]) -> Dict:
        """Write blocks to Notion page"""
        url = f"https://api.notion.com/v1/blocks/{self.page_id}/children"
        payload = {"children": blocks}
        response = requests.patch(url, json=payload, headers=self.HEADERS)
        return response.json()


class BlockProcessor:
    def convert_latex(blocks):
        LATEX_DELIMITERS = [("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$")]

        def process_text_content(rich_text):
            def extract_text_part(content, rich_text):
                return {
                    "type": "text",
                    "text": {"content": content, "link": None},
                    "plain_text": content,
                    "annotations": rich_text["annotations"],
                    "href": rich_text["href"],
                }

            def extract_equation_part(content, rich_text):
                return {
                    "type": "equation",
                    "equation": {"expression": content.strip()},
                    "plain_text": content,
                    "annotations": rich_text["annotations"],
                    "href": rich_text["href"],
                }

            content = rich_text["text"]["content"]
            start_idx = 0
            result = []

            while start_idx < len(content):
                found_delim = False

                for start_delim, end_delim in LATEX_DELIMITERS:
                    start_pos = content.find(start_delim, start_idx)
                    if start_pos == -1:
                        continue

                    end_pos = content.find(end_delim, start_pos + len(start_delim))
                    if end_pos == -1:
                        continue

                    # Add text before equation if exists
                    if start_pos > start_idx:
                        before_text = content[start_idx:start_pos]
                        result.append(extract_text_part(before_text, rich_text))

                    # Add equation part
                    equation_content = content[start_pos + len(start_delim) : end_pos]
                    result.append(extract_equation_part(equation_content, rich_text))

                    start_idx = end_pos + len(end_delim)
                    found_delim = True
                    break

                # Add remaining text
                if not found_delim:
                    remaining_text = content[start_idx:]
                    result.append(extract_text_part(remaining_text, rich_text))
                    break

            return result

        def process_rich_text(rich_text_list):
            result = []
            for rich_text in rich_text_list:
                if rich_text["type"] == "text":
                    result.extend(process_text_content(rich_text))
                else:
                    result.append(rich_text)
            return result

        for block in blocks:
            rich_text = block[block["type"]]["rich_text"]
            if rich_text:
                block[block["type"]]["rich_text"] = process_rich_text(rich_text)
        return blocks


def main():
    print("Starting to fetch Notion data...")

    notion_api = NotionAPI(NOTION_TOKEN, PAGE_ID)
    page_content = notion_api.read_blocks(PAGE_ID)
    print(f"Retrieved {len(page_content)} blocks from Notion page")

    converted_blocks = BlockProcessor.convert_latex(page_content)
    print("Converted LaTeX equations to Notion equation blocks")

    response = notion_api.write_blocks(converted_blocks)
    print("Upload complete!")


if __name__ == "__main__":
    main()
