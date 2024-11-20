import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv
from notion_client import Client

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
PAGE_ID = "143bb2c6d13380459053f33d84fd6cdb"
HEADERS = {"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": "2022-06-28", "Content-Type": "application/json"}
notion = Client(auth=NOTION_TOKEN)

LATEX_DELIMITERS = [("\\(", "\\)"), ("\\[", "\\]"), ("$$", "$$")]


class NotionAPI:
    def __init__(self, TOKEN: str, PAGE_ID: str):
        self.TOKEN = TOKEN
        self.PAGE_ID = PAGE_ID
        self.HEADERS = {
            "Authorization": f"Bearer {TOKEN}",
            "Notion-Version": "2022-06-28",
        }
        self.client = Client(auth=TOKEN)

    # def read_page(self) -> Optional[Dict]:
    #     """Read a page by ID"""
    #     try:
    #         url = f"https://api.notion.com/v1/pages/{self.PAGE_ID}"
    #         response = requests.get(url, headers=self.HEADERS)
    #         response.raise_for_status()
    #         return response.json()
    #     except requests.exceptions.RequestException as e:
    #         print(f"Error retrieving page: {str(e)}")
    #         return None

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
        url = f"https://api.notion.com/v1/blocks/{self.PAGE_ID}/children"
        payload = {"children": blocks}
        response = requests.patch(url, json=payload, headers=self.HEADERS)
        return response.json()


class BlockProcessor:
    @staticmethod
    def blocks_to_dataframe(blocks: List[Dict]) -> pd.DataFrame:
        """Convert Notion blocks to pandas DataFrame"""
        data = []
        for block in blocks:
            block_type = block["type"]
            content = BlockProcessor._extract_block_content(block, block_type)
            data.append({"id": block["id"], "type": block_type, "content": content})

        return pd.DataFrame(data)

    # @staticmethod
    # def _extract_block_content(block: Dict, block_type: str) -> str:
    #     """Extract content from different block types"""
    #     content = ""

    #     if block_type == "equation":
    #         content = block["equation"]["expression"]
    #     elif block_type in ["bulleted_list_item", "numbered_list_item"]:
    #         content = BlockProcessor._rich_text_to_string(block[block_type]["rich_text"])
    #     elif "rich_text" in block.get(block_type, {}):
    #         content = BlockProcessor._rich_text_to_string(block[block_type]["rich_text"])
    #     elif block_type == "code":
    #         content = block["code"]["text"][0]["text"]["content"]
    #     elif block_type == "quote":
    #         content = BlockProcessor._rich_text_to_string(block["quote"]["rich_text"])

    #     return content

    # @staticmethod
    # def _rich_text_to_string(rich_text: List[Dict]) -> str:
    #     """Process rich text content including equations"""
    #     content = ""
    #     for item in rich_text:
    #         if item["type"] == "text":
    #             content += item["text"]["content"]
    #         elif item["type"] == "equation":
    #             expr = item["equation"]["expression"]
    #             if "\n" in expr:
    #                 content += f"\\[\n{expr}\n\\]"
    #             else:
    #                 content += f"\\( {expr} \\)"
    #     return content


class EquationProcessor:
    @staticmethod
    def string_to_rich_text(block: Any) -> List[Dict]:
        """Format content for Notion API by converting text with equations into rich text format."""
        if not isinstance(block, str):
            return block

        if not block.strip():
            return [{"type": "text", "text": {"content": block}}]

        return EquationProcessor._process_content_lines(block.split("\n"))

    @staticmethod
    def _process_content_lines(lines: List[str]) -> List[Dict]:
        """Process content lines and convert to rich text format."""
        formatted_parts = []
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            equation_result = EquationProcessor._process_equations(line, lines, i)

            if equation_result.is_equation:
                formatted_parts.extend(equation_result.parts)
                i = equation_result.next_index
            elif line:
                formatted_parts.append({"type": "text", "text": {"content": line}})

            if i < len(lines) - 1:
                formatted_parts.append({"type": "text", "text": {"content": "\n"}})

            i += 1

        return formatted_parts if formatted_parts else [{"type": "text", "text": {"content": ""}}]

    @staticmethod
    def _process_equations(line: str, lines: List[str], current_index: int) -> Any:
        """Process equations in the content and return processing results."""
        from collections import namedtuple

        Result = namedtuple("Result", ["is_equation", "parts", "next_index"])

        normalized_line = line.replace("$$", "\\[")
        if normalized_line != line:
            for i in range(current_index + 1, len(lines)):
                if "$$" in lines[i]:
                    lines[i] = lines[i].replace("$$", "\\]")
                    break

        for start_delim, end_delim in LATEX_DELIMITERS:
            if start_delim in normalized_line and end_delim in normalized_line:
                try:
                    parts = EquationProcessor._process_inline_equation(normalized_line, start_delim, end_delim)
                    return Result(True, parts, current_index)
                except ValueError:
                    continue
            elif normalized_line.strip() == start_delim.strip("\\"):
                parts, end_index = EquationProcessor._process_multiline_equation(lines, current_index + 1, end_delim)
                if parts:
                    return Result(True, parts, end_index)

        return Result(False, [], current_index)

    @staticmethod
    def _process_inline_equation(line: str, start_delim: str, end_delim: str) -> List[Dict]:
        """Process a single-line equation and return rich text parts."""
        parts = []
        start_idx = line.index(start_delim)
        end_idx = line.index(end_delim, start_idx + len(start_delim))
        equation = line[start_idx + len(start_delim) : end_idx].strip()

        if equation:
            if start_idx > 0:
                parts.append({"type": "text", "text": {"content": line[:start_idx]}})

            parts.append({"type": "equation", "equation": {"expression": equation}})

            remaining = line[end_idx + len(end_delim) :].strip()
            if remaining:
                parts.append({"type": "text", "text": {"content": " " + remaining}})

        return parts

    @staticmethod
    def _process_multiline_equation(lines: List[str], start_index: int, end_delim: str) -> Tuple[List[Dict], int]:
        """Process a multi-line equation and return rich text parts and ending index."""
        equation_lines = []
        j = start_index

        while j < len(lines):
            if lines[j].strip() == end_delim.strip("\\"):
                if equation_lines:
                    return ([{"type": "equation", "equation": {"expression": "\n".join(equation_lines)}}], j)
                return ([], j)
            equation_lines.append(lines[j])
            j += 1

        return ([], start_index - 1)


def combine_text_and_equations(df: pd.DataFrame) -> List[Dict]:
    """Combine text and equations into Notion blocks"""
    combined_blocks = []

    for _, row in df.iterrows():
        content = row["content"]
        block_type = row["type"]

        if block_type == "equation":
            combined_blocks.append({"type": "equation", "equation": {"expression": content.strip("\\[\\]\n")}})
        else:
            rich_text = EquationProcessor.string_to_rich_text(content)
            if isinstance(rich_text, dict) and rich_text["type"] == "equation":
                combined_blocks.append(rich_text)
            elif rich_text:
                combined_blocks.append({"type": block_type, block_type: {"rich_text": rich_text}})

    return combined_blocks


def main():
    print("Starting to fetch Notion data...")

    notion_api = NotionAPI(NOTION_TOKEN, PAGE_ID)
    # page_data = notion_api.read_page()
    # if not page_data:
    #     print("Failed to retrieve page")
    #     return

    print("Page retrieved successfully")
    page_content = notion_api.read_blocks(PAGE_ID)
    print(f"Retrieved {len(page_content)} blocks from Notion page")

    df = BlockProcessor.blocks_to_dataframe(page_content)
    combined_data = combine_text_and_equations(df)
    print("Processed and combined text and equations")

    response = notion_api.write_blocks(combined_data)
    print("Upload complete!")


if __name__ == "__main__":
    # Test read_page functionality
    test_api = NotionAPI(NOTION_TOKEN, PAGE_ID)
    page = test_api.read_blocks(PAGE_ID)
    print(page)
