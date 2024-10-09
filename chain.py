import os
from typing import Generator, Iterator, Union

import opencc
from dotenv import load_dotenv
from langchain import hub
from langchain.agents.agent import AgentExecutor
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain.chat_models.base import init_chat_model
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.chat_models.base import ChatOpenAI

load_dotenv()


class UniversalChain:
    class LLM:
        def __init__(self, model_name: str):
            self.llm_initializers = {
                "azure": self._init_azure_openai,
                "gemini": self._init_gemini,
                "claude": self._init_claude,
                "gpt": self._init_openai,
            }
            try:
                for key, initializer in self.llm_initializers.items():
                    if key in model_name.lower():
                        return initializer(model_name)
                return init_chat_model(model=model_name)
            except Exception as e:
                print(f"Error initializing LLM: {e}")
                raise e

        @staticmethod
        def _init_azure_openai(model_name: str):
            return AzureChatOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            )

        @staticmethod
        def _init_gemini(model_name: str):
            return ChatGoogleGenerativeAI(model=model_name, api_key=os.getenv("GEMINI_API_KEY"))

        @staticmethod
        def _init_claude(model_name: str):
            return ChatOpenAI(
                model=f"anthropic/{model_name}",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )

        @staticmethod
        def _init_openai(model_name: str):
            return ChatOpenAI(model=model_name, api_key=os.getenv("OPENAI_API_KEY"))

    def __init__(self, model_name: str, use_history: bool = False):
        self.llm = self.LLM(model_name)
        self.tools = self.get_tools()
        self.use_history = use_history
        self.history = InMemoryChatMessageHistory(session_id="universal-chain-session")
        self.chain = self.create_chain()

    def get_tools(self):
        @tool
        def add(a: float, b: float) -> float:
            """Adds a and b."""
            return a + b

        @tool
        def multiply(a: float, b: float) -> float:
            """Multiplies a and b."""
            return a * b

        return [add, multiply]

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

    def invoke(self, input_text: str):
        config = {"configurable": {"session_id": "universal-chain-session"}}
        return self.chain.invoke({"input": input_text}, config)

    @staticmethod
    def s2hk(content: str) -> str:
        converter = opencc.OpenCC("s2hk")
        return converter.convert(content)

    @staticmethod
    def display_response(response: Union[str, Generator, Iterator], stream: bool = False) -> None:
        response = response["output"]
        response = UniversalChain.s2hk(response)
        if stream:
            for chunk in response:
                print(chunk, end="")
        else:
            print(response)


if __name__ == "__main__":
    chain = UniversalChain("gpt-4o-mini", use_history=False)
    response1 = chain.invoke("My name is Andy")
    chain.display_response(response1, stream=True)
    print("---")
    response2 = chain.invoke("What is my name?")
    chain.display_response(response2, stream=True)
