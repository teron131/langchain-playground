import os
import re
from typing import Generator, Iterator, List, Union

import opencc
from langchain import hub
from langchain.agents.agent import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain.chat_models.base import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.chat_models.base import ChatOpenAI
from pytubefix import YouTube

from YouTubeLoader.youtube import url_to_subtitles


class UniversalChain:
    def __init__(self, model_name: str, use_history: bool = False):
        self.llm = self.get_llm(model_name)
        self.tools = self.get_tools()
        self.use_history = use_history
        self.history = InMemoryChatMessageHistory(session_id="universal-chain-session")
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
            elif "gemini" in model_id:
                llm = ChatGoogleGenerativeAI(model=model_id, api_key=os.getenv("GEMINI_API_KEY"))
            elif "claude" in model_id:
                llm = ChatOpenAI(
                    model=f"anthropic/{model_id}",  # Avoid making model_id with '/', otherwise it will mess up the FastAPI URL
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
            elif "gpt" in model_id:
                llm = ChatOpenAI(model=model_id, api_key=os.getenv("OPENAI_API_KEY"))
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
        tool_agent_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(self.llm, self.tools, tool_agent_prompt)

        react_prompt = hub.pull("hwchase17/react")
        agent = create_react_agent(self.llm, self.tools, react_prompt)

        agent_executor = AgentExecutor(agent=agent, tools=self.tools)

        if self.use_history:
            return RunnableWithMessageHistory(
                agent_executor,
                # This is needed because in most real world scenarios, a session id is needed
                # It isn't really used here because we are using a simple in memory ChatMessageHistory
                lambda session_id: self.history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

        return agent_executor

    def generate_response(self, input_text: str):
        config = {"configurable": {"session_id": "universal-chain-session"}}
        return self.chain.invoke({"input": input_text}, config)["output"]

    def s2hk(content: str) -> str:
        converter = opencc.OpenCC("s2hk")
        return converter.convert(content)


if __name__ == "__main__":
    chain = UniversalChain("gpt-4o-mini", use_history=True)
    questions = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/ Can you read the content?",
        "https://youtu.be/7J_Ugp8ZB4E Summarize",
    ]
    for i, question in enumerate(questions, 1):
        print(f"Question {i}:\n{question}")
        response = chain.generate_response(question)
        print()
        print(f"Response {i}:")
        print(response)
        if i < len(questions):
            print("---")
