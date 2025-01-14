import re
from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import BaseTool, tool
from pytubefix import YouTube

from ..Tools.WebSearch import websearch
from ..Tools.YouTubeLoader import youtubeloader


@tool
def websearch_tool(query: str) -> str:
    """Search the web for information based on the query."""
    return websearch(query)


def webloader(url: str) -> str:
    """Load and process the content of a website from URL into formatted text. The function loads the webpage content, cleans up excessive newlines, and prepares a formatted output with the URL and content.

    Args:
        url (str): The URL of the website to load

    Returns:
        str: Formatted string containing the website URL followed by the processed content
    """
    docs = WebBaseLoader(url).load()
    docs = [re.sub(r"\n{3,}", r"\n\n", re.sub(r" {2,}", " ", doc.page_content)) for doc in docs]
    content = [
        f"Website: {url}",
        *docs,
    ]
    return "\n\n".join(content)


@tool
def webloader_tool(url: str) -> str:
    """Load the content of a website from url to text."""
    return webloader(url)


@tool
def youtubeloader_tool(url: str) -> str:
    """Load the subtitles of a YouTube video by url in form such as: https://www.youtube.com/watch?v=..., https://youtu.be/..., or more."""
    return youtubeloader(url)


def print_tool_info(tool_func: BaseTool) -> None:
    print(f"Tool:{tool_func.name}")
    print(f"Description: {tool_func.description}")
    print(f"Arguments: {tool_func.args}\n")


def get_tools() -> List[BaseTool]:
    """Get the list of available tools for the UniversalChain.

    Returns:
        List[BaseTool]: List of tool functions.
    """
    tools = [webloader_tool, youtubeloader_tool]

    # Print info for all tools
    for tool_func in tools:
        print_tool_info(tool_func)

    return tools


if __name__ == "__main__":
    get_tools()
