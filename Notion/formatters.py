import re
from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from utils import is_rich_text_block
from notion_api import NotionAPI


def text_to_text(rich_text: Dict, new_content: str) -> Dict:
    return {
        "type": "text",
        "text": {"content": new_content, "link": rich_text["text"]["link"]},
        "plain_text": new_content,
        "annotations": rich_text["annotations"],
        "href": rich_text["href"],
    }


def text_to_equation(rich_text: Dict, new_content: str) -> Dict:
    return {
        "type": "equation",
        "equation": {"expression": new_content},
        "plain_text": new_content,
        "annotations": rich_text["annotations"],
        "href": rich_text["href"],
    }


class BaseFormatter:
    def __init__(self, notionapi: NotionAPI):
        self.notionapi = notionapi

    def process_rich_text(self, rich_text: Dict) -> List[Dict]:
        """Abstract method meant to be overridden by child classes (LatexFormatter and Rephraser)."""
        raise NotImplementedError

    def process_block(self, block: Dict) -> Dict:
        """
        Process a block's content while preserving the block structure and formatting.

        Args:
            block (Dict): A Notion block object to be processed

        Returns:
            Dict: The processed block while preserving the original structure
        """
        block_type = block["type"]
        if not is_rich_text_block(block_type):
            return block

        rich_text_list = block[block_type].get("rich_text", [])
        new_rich_text_list = []

        for rich_text in rich_text_list:
            if rich_text["type"] == "text":
                new_rich_text_list.extend(self.process_rich_text(rich_text))
            else:
                new_rich_text_list.append(rich_text)

        block[block_type]["rich_text"] = new_rich_text_list
        return block

    def process_blocks(self, blocks: List[Dict]) -> None:
        """
        Process blocks and update them in Notion.

        Args:
            blocks (List[Dict]): A list of Notion blocks to be processed
        """
        for block in tqdm(blocks, desc=self.progress_description):
            block_type = block["type"]
            if is_rich_text_block(block_type):
                processed_block = self.process_block(block)
                new_rich_text = processed_block[block_type]["rich_text"]
                self.notionapi.update_block_rich_text(block, new_rich_text)


class LatexFormatter(BaseFormatter):
    def __init__(self, notionapi: NotionAPI):
        super().__init__(notionapi)
        self.inline_pattern = r"(.*?)(\\\(|\$)(.*?)(\\\)|\$)|(.+)$"
        self.block_pattern = r"(.*?)(\\\[|\$\$)(.*?)(\\\]|\$\$)|(.+)$"
        self.LATEX_PATTERN = re.compile(self.inline_pattern)

    @property
    def progress_description(self) -> str:
        return "Converting LaTeX equations"

    def process_rich_text(self, rich_text: Dict) -> List[Dict]:
        content = rich_text["text"]["content"]
        result = []

        for match in self.LATEX_PATTERN.finditer(content):
            before_text, start_delim, equation_content, end_delim, remaining_text = match.groups()
            if before_text:
                result.append(text_to_text(rich_text, before_text))
            if equation_content:
                result.append(text_to_equation(rich_text, equation_content))
            if remaining_text:
                result.append(text_to_text(rich_text, remaining_text))
                break

        return result


class Rephraser(BaseFormatter):
    def __init__(self, notionapi: NotionAPI):
        super().__init__(notionapi)
        self.chain = self.create_chain()

    @property
    def progress_description(self) -> str:
        return "Rephrasing blocks"

    def create_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Rephrase the following text, primarily on grammar and clarity. Do not change the specific terms.Do not change if it is already good."),
                ("human", "{text}"),
            ]
        )
        model = ChatOpenAI(model="gpt-4o-mini")
        return prompt | model

    def rephrase_text(self, text: str) -> str:
        return self.chain.invoke({"text": text}).content

    def process_rich_text(self, rich_text: Dict) -> List[Dict]:
        """
        Process rich text content by rephrasing it using a language model.

        Args:
            rich_text (Dict): A Notion rich text object containing text content and formatting

        Returns:
            List[Dict]: List containing the rephrased text object with preserved formatting
        """
        content = rich_text["text"]["content"]
        rephrased_content = self.rephrase_text(content)
        return [self.text_to_text(rich_text, rephrased_content)]


if __name__ == "__main__":
    notion_api = NotionAPI()
    rephraser = Rephraser(notion_api)
    rephraser.process_blocks(notion_api.read_blocks())
