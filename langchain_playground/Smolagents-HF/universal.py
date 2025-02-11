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


if __name__ == "__main__":
    question = """
Cyberpunk 2077 \u2014 Never Fade Away by P. T. Adamczyk & Olga Jankowska (SAMURAI Cover)\n\nWe lost everything\nWe had to pay the price\nYeah we lost everything\nWe had to pay the price\nI saw in you what life was missing\nYou lit a flame that consumed my hate\nI'm not one for reminiscing but\nI'd trade it all for your sweet embrace\nYeah\n'Cause we lost everything\nWe had to pay the price\nThere's a canvas with two faces\nOf fallen angels who loved and lost\nIt was a passion for the ages\nAnd in the end guess we paid the cost\nA thing of beauty, I know\nWill never fade away\nWhat you did to me, I know\nSaid what you had to say\nBut a thing of beauty\nWill never fade away\nWill never fade away\nWill never fade away\nI see your eyes, I know you see me\nYou're like a ghost how you're everywhere\nI'm your demon never leaving\nA metal soul of rage and fear\nThat one thing that changed it all\nThat one sin that caused the fall\nA thing of beauty, I know\nWill never fade away\nWhat you did to me, I know\nSaid what you had to say\nBut a thing of beauty, I know\nWill never fade away\nAnd I'll do my duty, I know\nSomehow I'll find a way\nBut a thing of beauty\nWill never fade away\nAnd I'll do my duty\nYeah\nWe'll never fade away\nWe'll never fade away\nWe'll never fade away\nWe'll never fade away\n\nWritten and composed by Johnny Silverhand, \"Never Fade Away\" was part of his 2013 album A Cool Metal Fire under the label of Universal Media.[1][2] Some time afterwards he released the song again in an album titled Never Fade Away. Finally, after the Samurai reunion, another version with the rest of the band was recorded. By 2077, the Samurai version was usually played on 107.3 Morro Rock Radio.\n\nCan you analyze this lyrics?
"""
    response = invoke(question)
    print(response)
