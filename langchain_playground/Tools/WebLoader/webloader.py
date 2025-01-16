import re

from langchain_community.document_loaders import WebBaseLoader


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
