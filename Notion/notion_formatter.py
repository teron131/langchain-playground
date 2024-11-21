import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from notion_client import Client

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
PAGE_ID = "144bb2c6d1338068befdfb9e08242891"


class NotionAPI:
    def __init__(self, TOKEN: str, PAGE_ID: str):
        """
        Initialize NotionAPI client with authentication token and page ID.
        Make sure to connect the page to the integration associated with the token.
        The schemas of the APIs are from https://developers.notion.com/reference

        Args:
            TOKEN (str): Notion API authentication token, from https://developers.notion.com
            PAGE_ID (str): ID of the Notion page to interact with, from the last part (without title) of the page URL,
                           e.g. https://www.notion.so/USERNAME/TITLE-PAGE_ID
        """
        self.token = TOKEN
        self.page_id = PAGE_ID
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Notion-Version": "2022-06-28",
        }
        self.client = Client(auth=self.token)

    def read_blocks(self, block_id: str) -> List[Dict]:
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

    def write_blocks(self, blocks: List[Dict]) -> Dict:
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
        for block in blocks:
            # Skip link_preview blocks as they're not directly supported
            if block.get(block["type"]).get("rich_text"):
                supported_blocks.append(block)

        url = f"https://api.notion.com/v1/blocks/{self.page_id}/children"
        payload = {"children": supported_blocks}
        response = requests.patch(url, json=payload, headers=self.headers)
        return response.json()

    def update_block_rich_text(self, block: Dict, rich_text: List[Dict]) -> Dict:
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
        payload = {block_type: {"rich_text": rich_text}}
        response = requests.patch(url, json=payload, headers=self.headers)
        return response.json()


class BlockProcessor:
    def convert_latex_block(block: Dict) -> Dict:
        """
        Reference: https://developers.notion.com/reference/block

        Processes a block containing LaTeX equations delimited by \\( \\), \\[ \\], or $$ and converts them
        into Notion's native equation format. Preserves text formatting.

        Args:
            block (Dict): A Notion block object containing rich text with LaTeX equations

        Returns:
            Dict: The processed block with LaTeX equations converted to Notion equation blocks.
                  Preserves original block structure and formatting.

        Example:
            Input text: "A function \\(f(x)\\) is continuous"
            Output: Two rich_text objects:
                1. Text object with "A function "
                2. Equation object with "f(x)"
                3. Text object with " is continuous"
        """
        LATEX_DELIMITERS = set([("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$")])

        def process_text_content(rich_text: Dict) -> List[Dict]:
            """
            Reference: https://developers.notion.com/reference/rich-text

            Process rich text content to extract LaTeX equations and convert them to Notion equation blocks.

            Takes a rich text object containing potential LaTeX equations and splits it into separate
            text and equation objects while preserving formatting. Equations are identified by LaTeX
            delimiters (\\(\\), \\[\\], or $$) and converted to Notion's native equation format.

            Args:
                rich_text (Dict): A Notion rich text object containing text content and formatting

            Returns:
                List[Dict]: List of text and equation objects with preserved formatting. Each object is
                           either a text object containing regular text or an equation object containing
                           the LaTeX expression.

            Example:
                Input rich_text with content "x = \\(a + b\\) where a,b > 0"
                Returns: [
                    {type: "text", text: {content: "x = "}, ...},
                    {type: "equation", equation: {expression: "a + b"}, ...},
                    {type: "text", text: {content: " where a,b > 0"}, ...}
                ]
            """

            def extract_text_part(content, rich_text):
                """
                Create a text rich_text content from content while preserving other rich_text properties.

                Modifies only the content and type, keeping other properties from the original rich_text object.
                """
                return {
                    "type": "text",
                    "text": {"content": content, "link": None},
                    "plain_text": content,
                    "annotations": rich_text["annotations"],
                    "href": rich_text["href"],
                }

            def extract_equation_part(content, rich_text):
                """
                Create an equation rich_text content from content while preserving other rich_text properties.

                Modifies only the content and type, keeping other properties from the original rich_text object.
                """
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

        rich_text_list = block.get(block["type"]).get("rich_text")
        if rich_text_list:
            result = []
            # Assume equations only exist in the block types that have rich_text
            for rich_text in rich_text_list:
                if rich_text["type"] == "text":
                    result.extend(process_text_content(rich_text))
                else:
                    result.append(rich_text)
            block[block["type"]]["rich_text"] = result
        return block

    @staticmethod
    def loop_all_blocks(blocks: List[Dict], func: Callable) -> List[Dict]:
        return [func(block) for block in blocks]


def main():
    print("Starting to fetch Notion data...")

    notion_api = NotionAPI(NOTION_TOKEN, PAGE_ID)
    page_content = notion_api.read_blocks(PAGE_ID)
    print(f"Retrieved {len(page_content)} blocks from Notion page")

    # Filter blocks that have rich_text
    blocks_with_rich_text = [block for block in page_content if block.get(block["type"]).get("rich_text")]
    print(f"Found {len(blocks_with_rich_text)} blocks with rich_text")

    # Convert and update each block with rich_text
    for block in blocks_with_rich_text:
        converted_block = BlockProcessor.convert_latex_block(block)
        rich_text = converted_block[block["type"]]["rich_text"]
        print(rich_text)
        # Use the notion_api instance to call update_block_rich_text
        response = notion_api.update_block_rich_text(block, rich_text)
        print(f"Updated block {block['id']}")


if __name__ == "__main__":
    main()
