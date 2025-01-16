import re
from functools import wraps
from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import BaseTool, tool

from langchain_playground.Tools.WebSearch import websearch
from langchain_playground.Tools.YouTubeLoader import youtubeloader


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


def print_tool_info(tool_func: BaseTool) -> None:
    print(f"Tool:{tool_func.name}")
    print(f"Description: {tool_func.description}")
    print(f"Arguments: {tool_func.args}\n")


def get_tools() -> List[BaseTool]:
    """Get the list of available tools for the UniversalChain. The tools are wrapped with their original docstrings and registered as langchain tools.

    Returns:
        List[BaseTool]: List of tool functions with their docstrings preserved
    """

    @tool
    @wraps(websearch)
    def websearch_tool(query: str) -> str:
        return websearch(query)

    @tool
    @wraps(webloader)
    def webloader_tool(url: str) -> str:
        return webloader(url)

    @tool
    @wraps(youtubeloader)
    def youtubeloader_tool(url: str) -> str:
        return youtubeloader(url)

    tools = [websearch_tool, webloader_tool, youtubeloader_tool]

    # Print info for all tools
    for tool_func in tools:
        print_tool_info(tool_func)

    return tools
