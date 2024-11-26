import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from notion_api import NotionAPI
from utils import is_rich_text_block, text_to_equation, text_to_text


@dataclass
class Annotations:
    """Represents text formatting annotations in Notion."""

    bold: bool = False
    italic: bool = False
    strikethrough: bool = False
    underline: bool = False
    code: bool = False
    color: str = "default"


@dataclass
class RichTextContent:
    """Represents the content of a rich text block in Notion."""

    content: str
    link: Optional[str] = None


@dataclass
class RichText:
    """Represents a rich text object in Notion."""

    type: str
    text: RichTextContent
    annotations: Optional[Annotations] = None
    plain_text: Optional[str] = None
    href: Optional[str] = None


class NotionBlock(TypedDict):
    """Represents a block object in Notion."""

    id: str
    type: str
    has_children: bool


class BaseFormatter:
    """Base class for Notion block formatters."""

    def __init__(self, notionapi: NotionAPI):
        """
        Initialize the formatter with a Notion API instance.

        Args:
            notionapi (NotionAPI): Instance of NotionAPI for interacting with Notion
        """
        self.notionapi = notionapi

    @property
    def progress_description(self) -> str:
        """
        Returns the description to be shown in the progress bar.

        Returns:
            str: Progress bar description
        """
        raise NotImplementedError("Subclasses must implement progress_description")

    def process_block(self, block: NotionBlock) -> NotionBlock:
        """
        Process a single block.

        Args:
            block (NotionBlock): The block to process

        Returns:
            NotionBlock: The processed block
        """
        raise NotImplementedError("Subclasses must implement process_block")

    def process_blocks(self, blocks: List[NotionBlock]) -> None:
        """
        Process multiple blocks and update them in Notion.

        Args:
            blocks (List[NotionBlock]): List of blocks to process
        """
        with tqdm(blocks, desc=self.progress_description) as pbar:
            for block in pbar:
                block_type = block["type"]
                if is_rich_text_block(block_type):
                    processed_block = self.process_block(block)
                    new_rich_text = processed_block[block_type]["rich_text"]
                    self.notionapi.update_block_rich_text(block, new_rich_text)


class LatexFormatter(BaseFormatter):
    """Formatter for converting LaTeX equations in Notion blocks to native equation blocks."""

    LATEX_PATTERN = re.compile(r"(.*?)(\\\(|\\\[|\$\$)(.*?)(\\\)|\\\]|\$\$)|(.+)$")

    def __init__(self, notionapi: NotionAPI):
        """
        Initialize the LaTeX formatter.

        Args:
            notionapi (NotionAPI): Instance of NotionAPI for interacting with Notion
        """
        super().__init__(notionapi)

    @property
    def progress_description(self) -> str:
        return "Converting LaTeX equations"

    def _convert_rich_text(self, rich_text: RichText) -> List[RichText]:
        """
        Process rich text content to extract LaTeX equations and convert them to Notion equation blocks.

        Args:
            rich_text (RichText): A Notion rich text object containing text content and formatting

        Returns:
            List[RichText]: List of text and equation objects with preserved formatting
        """
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

    def process_block(self, block: NotionBlock) -> NotionBlock:
        """
        Reference: https://developers.notion.com/reference/block

        Processes a block containing LaTeX equations delimited by \\( \\), \\[ \\], or $$ and converts them into Notion's native equation format. Preserves text formatting.

        Args:
            block (NotionBlock): A Notion block object that may contain LaTeX equations in its rich text content.

        Returns:
            NotionBlock: The block with any LaTeX equations converted to Notion's native equation format.
        """
        block_type = block["type"]
        if not is_rich_text_block(block_type):
            return block

        rich_text_list = block[block_type].get("rich_text")
        new_rich_text_list = []

        for rich_text in rich_text_list:
            if rich_text["type"] == "text":
                new_rich_text_list.extend(self._convert_rich_text(rich_text))
            else:
                new_rich_text_list.append(rich_text)

        block[block_type]["rich_text"] = new_rich_text_list

        return block


class Rephraser(BaseFormatter):
    """Formatter for rephrasing text content in Notion blocks."""

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
        """
        Rephrase the given text using the language model.

        Args:
            text (str): Text to rephrase

        Returns:
            str: Rephrased text
        """
        return self.chain.invoke({"text": text}).content

    def process_block(self, block: NotionBlock) -> NotionBlock:
        """
        Rephrase the content of a block using a language model.

        Args:
            block (NotionBlock): A Notion block object containing text to be rephrased

        Returns:
            NotionBlock: The block with rephrased content
        """
        block_type = block["type"]
        if not is_rich_text_block(block_type):
            return block

        rich_text_list = block[block_type].get("rich_text", [])
        new_rich_text_list = []

        for rich_text in rich_text_list:
            if rich_text["type"] == "text":
                content = rich_text["text"]["content"]
                rephrased_content = self.rephrase_text(content)
                new_rich_text = text_to_text(rich_text, rephrased_content)
                new_rich_text_list.append(new_rich_text)
            else:
                new_rich_text_list.append(rich_text)

        block[block_type]["rich_text"] = new_rich_text_list
        return block


if __name__ == "__main__":
    notion_api = NotionAPI()
    rephraser = Rephraser(notion_api)
    rephraser.process_blocks(notion_api.read_blocks())
