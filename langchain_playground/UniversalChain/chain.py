from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from .llm import get_llm
from .tools import get_tools


class UniversalChain:
    def __init__(self, provider: str, model_id: str):
        self.chain = self.create_chain(provider, model_id)

    def create_chain(self, provider: str, model_id: str, state_modifier: Optional[Any] = None):
        """Create a chain with the configured LLM and tools.

        Args:
            provider (str): Provider of the language model.
            model_id (str): ID of the language model to use.
            use_history (bool): Whether to use conversation history. Defaults to True.

        Returns:
            Agent: The created agent chain with an attached invoke method.
        """
        llm = get_llm(provider, model_id)
        tools = get_tools()
        chain = create_react_agent(
            llm,
            tools,
            state_modifier=state_modifier,
            checkpointer=MemorySaver(),
            store=InMemoryStore(),
        )

        return chain

    def invoke(self, user_input: str) -> str:
        """Generate a response to the given input text."""
        config = {"configurable": {"thread_id": "universal-chain-session"}}
        result = self.chain.invoke(
            {"messages": [("user", user_input)]},
            config,
        )
        return result["messages"][-1].content
