import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from .Tools import get_tools


class UniversalChain:
    def __init__(self, model_id: str, llm: BaseChatModel = None):
        """
        Initialize the UniversalChain with a language model.

        Args:
            model_id (str): The model ID in OpenRouter format.
            llm (BaseChatModel, optional): For overriding the default BaseChatModel.
        """
        self.llm = llm or ChatOpenAI(
            model=model_id,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        self.chain: CompiledGraph = self.create_chain()
        self.result: dict[str, list[BaseMessage]] = {}

    def create_chain(self) -> CompiledGraph:
        """
        Create a chain with the configured LLM and tools.

        Args:
            model_id (str): ID of the language model to use.

        Returns:
            Agent: The created agent chain with an attached invoke method.
        """
        tools = get_tools()
        return create_react_agent(
            self.llm,
            tools,
            checkpointer=MemorySaver(),
            store=InMemoryStore(),
        )

    def get_response(
        self,
        user_input: str,
        message_history: list[BaseMessage] = None,
    ) -> dict[str, list[BaseMessage]]:
        """Generate a response to the given input text."""
        config = {"configurable": {"thread_id": "universal-chain-session"}}

        # Include message history if provided, otherwise just the current input
        # Specialized for Open WebUI as the LangChain memory would get lost, likely due to session management, but it has a variable: messages (list[tuple[str, str]])
        messages = message_history or []
        messages.append(("user", user_input))

        self.result = self.chain.invoke(
            {"messages": messages},
            config,
        )
        return self.result

    def extract_ans_str(self) -> str:
        """Extract the answer string from the last message."""
        return self.result["messages"][-1].content

    def extract_history_msgs(self) -> list[BaseMessage]:
        """Extract the history messages from the result."""
        return self.result["messages"][:-1]

    def invoke(
        self,
        user_input: str,
        message_history: list[BaseMessage] = None,
    ) -> str:
        """
        Invoke the chain with the given input and message history.

        Args:
            user_input (str): The input text.
            message_history (list[BaseMessage], optional): The message history. Defaults to None.

        Returns:
            str: The response string.
        """
        self.get_response(user_input, message_history)
        return self.extract_ans_str()
