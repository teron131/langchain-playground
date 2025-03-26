import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph, add_messages
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel

from .Tools import get_tools


class State(BaseModel):
    model_id: str
    input_content: str
    response: str = ""  # Set default empty string to make it optional during initialization


def run_graph(state: State) -> State:
    chain = UniversalChain(state.model_id)
    state.response = chain.invoke(input_content=state.input_content)
    return state


builder = StateGraph(State)
builder.add_node("run_graph", run_graph)
builder.add_edge(START, "run_graph")
builder.add_edge("run_graph", END)
graph = builder.compile()


class UniversalChain:
    def __init__(self, model_id: str, llm: BaseChatModel = None):
        """Initialize the UniversalChain with a language model.

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
        self.result: MessagesState = {}

    def create_chain(self) -> CompiledGraph:
        """Create a chain with the configured LLM and tools.

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
            version="v2",
        )

    def get_response(
        self,
        input_content: str,
        message_history: list[BaseMessage] = None,
    ) -> MessagesState:
        """Generate a response to the given input text."""
        # Required for checkpointer and store
        config = {"configurable": {"thread_id": "universal-chain-session"}}

        # Include message history if provided, otherwise just the current input
        # Specialized for Open WebUI as the LangChain memory would get lost, likely due to session management, but it has a variable: messages (list[tuple[str, str]])
        messages = message_history or []
        messages = add_messages(messages, HumanMessage(content=input_content))
        return self.chain.invoke({"messages": messages}, config)

    def invoke(
        self,
        input_content: str,
        message_history: list[BaseMessage] = None,
    ) -> str:
        """Invoke the chain with the given input and message history.

        Args:
            input_content (str): The input text.
            message_history (list[BaseMessage], optional): The message history. Defaults to None.

        Returns:
            str: The response string.
        """
        self.get_response(input_content, message_history)
        return _extract_answer_message(self.result).content


def _extract_answer_message(messages: MessagesState) -> BaseMessage:
    """Extract the answer message from the last message."""
    return messages["messages"][-1]


def _extract_history_messages(messages: MessagesState) -> list[BaseMessage]:
    """Extract the history messages from the result."""
    return messages["messages"][:-1]
