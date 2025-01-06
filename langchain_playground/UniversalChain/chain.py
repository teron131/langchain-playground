from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .llm import get_llm
from .tools import get_tools


class UniversalChain:
    """A class that handles creation and execution of language model chains."""

    def __init__(self, model_name: str, use_history: bool = True):
        """Initialize the UniversalChain.

        Args:
            model_name (str): Name of the language model to use.
            use_history (bool): Whether to use conversation history. Defaults to True.
        """
        self.chain = self._create_chain(model_name, use_history)

    def _create_chain(self, model_name: str, use_history: bool = True):
        """Initialize a chain with the configured LLM and tools.

        Args:
            model_name (str): Name of the language model to use.
            use_history (bool): Whether to use conversation history. Defaults to True.

        Returns:
            Agent: The created agent chain.
        """
        llm = get_llm(model_name)
        tools = get_tools()
        checkpointer = MemorySaver() if use_history else None
        agent = create_react_agent(
            llm,
            tools,
            checkpointer=checkpointer,
        )
        return agent

    def invoke(self, input_text: str) -> str:
        """Generate a response to the given input text.

        Args:
            input_text (str): The input text.

        Returns:
            str: The generated response.
        """
        config = {"configurable": {"thread_id": "universal-chain-session"}}
        response = self.chain.invoke(
            {"messages": [("user", input_text)]},
            config,
        )
        return response["messages"][-1].content
