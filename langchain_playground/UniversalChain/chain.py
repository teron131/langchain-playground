from typing import Any, Dict, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from .llm import get_llm
from .tools import get_tools


class UniversalChain:
    def __init__(self, model_id: str, state_modifier: Optional[Any] = None):
        self.chain = self.create_chain(model_id, state_modifier)
        self.result = {}

    def create_chain(self, model_id: str, state_modifier: Optional[Any] = None):
        """Create a chain with the configured LLM and tools.

        Args:
            provider (str): Provider of the language model.
            model_id (str): ID of the language model to use.
            state_modifier (Optional[Any]): State modifier to use. Defaults to None.

        Returns:
            Agent: The created agent chain with an attached invoke method.
        """
        llm = get_llm(model_id)
        tools = get_tools()
        return create_react_agent(
            llm,
            tools,
            state_modifier=state_modifier,
            checkpointer=MemorySaver(),
            store=InMemoryStore(),
        )

    def get_response(self, user_input: str) -> Dict:
        """Generate a response to the given input text."""
        config = {"configurable": {"thread_id": "universal-chain-session"}}
        self.result = self.chain.invoke(
            {"messages": [("user", user_input)]},
            config,
        )
        return self.result

    def extract_ans_str(self) -> str:
        return self.result["messages"][-1].content

    def extract_history_msgs(self) -> list[str]:
        return self.result["messages"][:-1]

    def invoke(self, user_input: str) -> str:
        self.get_response(user_input)
        return self.extract_ans_str()
