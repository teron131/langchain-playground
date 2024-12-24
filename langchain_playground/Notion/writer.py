import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_openai import ChatOpenAI
from markdown import markdown_to_blocks
from notion_api import NotionAPI

if __name__ == "__main__":
    notion_api = NotionAPI()
    chain = ChatOpenAI(model="gpt-4o-mini")
    question = """
Write some math with inline, block, and multi-line equations in markdown.
    """
    print(f"Question:\n{question}")
    print()
    response = chain.invoke(question).content
    print(f"Response:\n{response}")
    print()
    blocks = markdown_to_blocks(response)
    print(blocks)
    notion_api.write_blocks(blocks)
