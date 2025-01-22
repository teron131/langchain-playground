from typing import Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from .llm import get_llm
from .Tools import get_tools


class UniversalChain:
    def __init__(self, model_id: str):
        self.chain = self.create_chain(model_id)
        self.result = {}

    def create_chain(self, model_id: str):
        """Create a chain with the configured LLM and tools.

        Args:
            model_id (str): ID of the language model to use.

        Returns:
            Agent: The created agent chain with an attached invoke method.
        """
        llm = get_llm(model_id)
        tools = get_tools()
        return create_react_agent(
            llm,
            tools,
            checkpointer=MemorySaver(),
            store=InMemoryStore(),
        )

    def get_response(self, user_input: str, message_history: list = None) -> Dict:
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
        return self.result["messages"][-1].content

    def extract_history_msgs(self) -> list[str]:
        return self.result["messages"][:-1]

    def invoke(self, user_input: str, message_history: list = None) -> str:
        self.get_response(user_input, message_history)
        return self.extract_ans_str()
