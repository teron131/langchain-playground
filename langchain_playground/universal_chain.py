import os
import re
from typing import Generator, Iterator, List, Union

import opencc
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pytubefix import YouTube

from .YouTubeLoader import url_to_subtitles


class UniversalChain:
    def __init__(
        self,
        model_name: str,
        use_history: bool = False,
        max_token_limit: int = 128000,
    ):
        self.llm = self.get_llm(model_name)
        self.tools = self.get_tools()
        self.use_history = use_history
        self.max_token_limit = max_token_limit
        self.checkpointer = MemorySaver() if use_history else None
        self.chain = self.create_chain()

    def get_llm(self, model_id: str):
        try:
            if "azure" in model_id:
                llm = AzureChatOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                )
            elif "gpt" in model_id or "o1" in model_id:
                llm = ChatOpenAI(model=model_id, api_key=os.getenv("OPENAI_API_KEY"))
            elif "gemini" in model_id:
                llm = ChatGoogleGenerativeAI(model=model_id, api_key=os.getenv("GOOGLE_API_KEY"))
            elif "claude" in model_id:
                llm = ChatOpenAI(
                    model=f"anthropic/{model_id}",  # Avoid making model_id with '/', otherwise it will mess up the FastAPI URL
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
            elif "deepseek" in model_id:
                llm = ChatOpenAI(
                    model=f"deepseek/{model_id}",  # Avoid making model_id with '/', otherwise it will mess up the FastAPI URL
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
            else:
                llm = init_chat_model(model=model_id)
        except Exception as e:
            raise ValueError(f"Invalid model_id: {model_id}\n{e}")
        return llm

    def get_tools(self):

        @tool
        def webloader(url: str) -> str:
            """Load the content of a website from url to text."""
            docs = WebBaseLoader(url).load()
            docs = [re.sub(r"\n{3,}", r"\n\n", doc.page_content) for doc in docs]
            docs_string = f"Website: {url}" + "\n\n".join(docs)
            return docs_string

        @tool
        def youtube_loader(url: str) -> str:
            # https://github.com/JuanBindez/pytubefix/blob/main/pytubefix/__main__.py
            """Load the subtitles of a YouTube video by url in form such as: https://www.youtube.com/watch?v=..., https://youtu.be/..., or more."""
            yt = YouTube(url)
            return f"Answer the user's question based on the full content.\nTitle: {yt.title}\nAuthor: {yt.author}\nSubtitles:\n\n{url_to_subtitles(url)}"

        return [webloader, youtube_loader]

    def create_chain(self):
        agent = create_react_agent(
            self.llm,
            self.tools,
            checkpointer=self.checkpointer,
        )

        return agent

    def generate_response(self, input_text: str):
        """
        Generate a response to the given input text.

        Args:
            input_text (str): The input text.

        Returns:
            str: The response.
        """
        # Configure thread ID for memory persistence
        config = {"configurable": {"thread_id": "universal-chain-session"}}

        response = self.chain.invoke(
            {"messages": [("user", input_text)]},
            config,
        )

        return response["messages"][-1].content

    def s2hk(content: str) -> str:
        converter = opencc.OpenCC("s2hk")
        return converter.convert(content)


if __name__ == "__main__":
    chain = UniversalChain("gpt-4o", use_history=True)
    question = """
Explain convex problem and how to solve it.
"""
    print(f"Question:\n{question}")
    print()
    print(f"Response:\n{chain.generate_response(question)}")
    print()
