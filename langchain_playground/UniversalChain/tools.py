from functools import wraps
from typing import List

from langchain_core.tools import BaseTool, tool

from ..Tools import webloader, websearch, youtubeloader


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
