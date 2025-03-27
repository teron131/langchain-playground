import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from rich import print

from .Tools import get_tools
from .utils import load_image_base64, s2hk


class AgentState(MessagesState):
    model_id: str = "google/gemini-2.0-flash-001"


def invoke_react_agent(state: AgentState) -> AgentState:
    """Create the ReAct agent and invoke it with the messages.

    Args:
        state (State): The state of the messages.

    Returns:
        State: The updated state of the messages.
    """
    state: AgentState
    llm = ChatOpenAI(
        model=state["model_id"],
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    chain = create_react_agent(
        llm,
        tools=get_tools(),
        version="v2",
    )
    response = chain.invoke({"messages": state["messages"]})
    return {"messages": response["messages"]}


def create_graph() -> CompiledStateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("invoke_react_agent", invoke_react_agent)
    builder.add_edge(START, "invoke_react_agent")
    builder.add_edge("invoke_react_agent", END)
    return builder.compile()


graph = create_graph()


class UniversalChain:
    """A chain implementation that wraps the compiled state graph with manual memory management outside LangGraph deployment, with contrast to the automatic memory management in deployment."""

    def __init__(self, model_id: str, llm: BaseChatModel = None):
        """Initialize the UniversalChain with a language model.

        Args:
            model_id (str): The model ID in OpenRouter format.
            llm (BaseChatModel, optional): For overriding the default BaseChatModel.
        """
        self.model_id = model_id
        self.graph: CompiledStateGraph = create_graph()
        self.result: MessagesState = {"messages": []}

    @property
    def messages(self) -> list[BaseMessage]:
        """Get the messages from the result."""
        return self.result["messages"]

    def invoke(
        self,
        text: str,
        image: str = None,
        history_messages: list[BaseMessage] = None,
    ) -> MessagesState:
        """Invoke and get the response from the ReAct agent.

        Args:
            text (str): The input text.
            image (str, optional): Path or URL to an image to include.
            history_messages (list[BaseMessage], optional): Previous conversation messages.

        Returns:
            MessagesState: The updated message state after processing.
        """
        config = {"configurable": {"thread_id": "universal-chain-session"}}

        messages = history_messages or self.messages

        message = _create_message(text, image)
        messages = add_messages(messages, message)

        self.result = self.graph.invoke(
            {
                "model_id": self.model_id,
                "messages": messages,
            },
            config,
        )
        return self.result

    def invoke_as_str(
        self,
        text: str,
        image: str = None,
        history_messages: list[BaseMessage] = None,
    ) -> str:
        """Invoke and get the string response from the ReAct agent.

        Args:
            text (str): The input text.
            image (str, optional): Path or URL to an image to include.
            history_messages (list[BaseMessage], optional): Previous conversation messages.

        Returns:
            str: The response string.
        """
        self.invoke(text, image, history_messages)
        return self.messages[-1].content


def _create_message(text: str, image: str = None) -> HumanMessage:
    """Create a human message with text and optional image content.

    Args:
        text (str): The text input from the user
        image (str, optional): Path or URL to an image to include

    Returns:
        HumanMessage: A formatted message object containing text and optional image
    """
    content = []
    if text:
        content.append({"type": "text", "text": text})
    if image:
        try:
            image_base64 = load_image_base64(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )
        except Exception as e:
            print(f"Error loading image: {e}")
            print("Skipping image")
    return HumanMessage(content)
