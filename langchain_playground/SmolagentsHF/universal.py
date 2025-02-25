import os

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel, ToolCallingAgent, tool

from langchain_playground.Tools import webloader, websearch, youtubeloader

load_dotenv()


class UniversalAgent:
    def __init__(self, model_id: str):
        self.model = self.create_model(model_id)
        self.agent = self.create_agent()

    def create_model(self, model_id: str):
        return LiteLLMModel(
            model_id=f"openrouter/{model_id}",
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def create_agent(self):
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

        return CodeAgent(
            tools=[web_search, web_loader, youtube_loader],
            model=self.model,
            add_base_tools=False,
        )

    def invoke(self, question: str):
        self.agent.visualize()
        return self.agent.run(question)
