import os
import re

from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from tavily import TavilyClient

load_dotenv()


def webloader_langchain(url: str) -> str:
    """Load and process the content of a website from URL into formatted text. The function loads the webpage content, cleans up excessive newlines, and prepares a formatted output with the URL and content.

    Args:
        url (str): The URL of the website to load

    Returns:
        str: Formatted string containing the website URL followed by the processed content
    """
    docs = WebBaseLoader(url).load()
    docs = [re.sub(r"\n{3,}", r"\n\n", re.sub(r" {2,}", " ", doc.page_content)) for doc in docs]
    return "\n\n".join(docs)


def webloader_docling(url: str) -> str:
    """Load and process the content of a website from URL into a rich unified markdown representation.

    Args:
        url (str): The URL of the website to load

    Returns:
        str: Formatted string containing the website URL followed by the processed content
    """
    converter = DocumentConverter()
    result = converter.convert(url)
    return result.document.export_to_markdown()


def webloader_tavily(url: str) -> str:
    """Execute a Tavily search query.
    https://docs.tavily.com/documentation/api-reference/endpoint/search

    args:
        websearch_args (WebSearchArgs): Search arguments containing query and max_results

    Returns:
        dict: Raw Tavily API response
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    response = tavily_client.extract(
        urls=url,
        include_images=False,
        search_depth="advanced",
    )
    return response["results"][0]["raw_content"]


def webloader(url: str) -> str:
    """Load and process the content of a website from URL into a rich unified markdown representation.

    Args:
        url (str): The URL of the website to load

    Returns:
        str: Formatted string in markdown containing the website URL followed by the processed content
    """
    # Try each loader in sequence, falling back to the next if one fails
    webloaders = [webloader_docling, webloader_langchain, webloader_tavily]

    for loader in webloaders:
        try:
            return loader(url)
        except Exception:
            continue

    # If all loaders fail, return an error message
    return f"Failed to load content from {url} using all available methods."
