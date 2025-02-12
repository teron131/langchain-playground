import os

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent, tool

from langchain_playground.Tools import webloader, websearch, youtubeloader

load_dotenv()


@tool
def web_search(query: str) -> str:
    """
    Search the web for information based on the query.

    Args:
        query: Search query string
    """
    return websearch(query)


@tool
def web_loader(url: str) -> str:
    """
    Load and process the content of a website from URL into a rich unified markdown representation.

    Args:
        url: The URL of the website to load
    """
    return webloader(url)


@tool
def youtube_loader(url: str) -> str:
    """
    Load and process a YouTube video's subtitles, title, and author information from a URL. Accepts various YouTube URL formats including standard watch URLs and shortened youtu.be links.

    Args:
        url: The YouTube video URL to load
    """
    return youtubeloader(url)


model = LiteLLMModel(
    model_id="openrouter/openai/gpt-4o-mini",
    api_base="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
agent = CodeAgent(
    tools=[web_search, web_loader, youtube_loader],
    model=model,
    add_base_tools=False,
)


def invoke(question: str):
    return agent.run(question)
