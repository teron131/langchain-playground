import re

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from pytubefix import YouTube

from ..YouTubeLoader import url_to_subtitles


def get_tools():
    """Get the list of available tools for the UniversalChain.

    Returns:
        list: List of tool functions.
    """

    @tool
    def webloader(url: str) -> str:
        """Load the content of a website from url to text."""
        docs = WebBaseLoader(url).load()
        docs = [re.sub(r"\n{3,}", r"\n\n", doc.page_content) for doc in docs]
        docs_string = f"Website: {url}" + "\n\n".join(docs)
        return docs_string

    @tool
    def youtube_loader(url: str) -> str:
        """Load the subtitles of a YouTube video by url in form such as: https://www.youtube.com/watch?v=..., https://youtu.be/..., or more."""
        yt = YouTube(url)
        return f"Answer the user's question based on the full content.\nTitle: {yt.title}\nAuthor: {yt.author}\nSubtitles:\n\n{url_to_subtitles(url)}"

    return [webloader, youtube_loader]
