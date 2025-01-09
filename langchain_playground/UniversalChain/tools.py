import re
from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import BaseTool, tool
from pytubefix import YouTube

from ..YouTubeLoader import url_to_subtitles


def webloader(url: str) -> str:
    """Load and process the content of a website from URL into formatted text. The function loads the webpage content, cleans up excessive newlines, and prepares a formatted output with the URL and content.

    Args:
        url (str): The URL of the website to load

    Returns:
        str: Formatted string containing the website URL followed by the processed content
    """
    docs = WebBaseLoader(url).load()
    docs = [re.sub(r"\n{3,}", r"\n\n", doc.page_content) for doc in docs]
    content = [
        f"Website: {url}",
        *docs,
    ]
    return "\n\n".join(content)


@tool
def webloader_tool(url: str) -> str:
    """Load the content of a website from url to text."""
    return webloader(url)


def youtube_loader(url: str) -> str:
    """Load and process a YouTube video's subtitles, title, and author information from a URL. Accepts various YouTube URL formats including standard watch URLs and shortened youtu.be links.

    Args:
        url (str): The YouTube video URL to load

    Returns:
        str: Formatted string containing the video title, author and subtitles
    """
    yt = YouTube(url)
    content = [
        "Answer the user's question based on the full content.",
        f"Title: {yt.title}",
        f"Author: {yt.author}",
        "Subtitles:",
        url_to_subtitles(url),
    ]
    return "\n".join(content)


@tool
def youtube_loader_tool(url: str) -> str:
    """Load the subtitles of a YouTube video by url in form such as: https://www.youtube.com/watch?v=..., https://youtu.be/..., or more."""
    return youtube_loader(url)


def print_tool_info(tool_func: BaseTool) -> None:
    print(f"Tool:{tool_func.name}")
    print(f"Description: {tool_func.description}")
    print(f"Arguments: {tool_func.args}\n")


def get_tools() -> List[BaseTool]:
    """Get the list of available tools for the UniversalChain.

    Returns:
        List[BaseTool]: List of tool functions.
    """
    tools = [webloader_tool, youtube_loader_tool]

    # Print info for all tools
    for tool_func in tools:
        print_tool_info(tool_func)

    return tools
