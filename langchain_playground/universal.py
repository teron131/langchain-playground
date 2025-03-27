import os
from typing import Optional

from dotenv import load_dotenv
from langchain_community.chat_models import ChatLiteLLM
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph, add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.pregel import RetryPolicy
from rich import print

from .Tools import get_tools
from .utils import load_image_base64

load_dotenv()


class AgentState(MessagesState):
    """State for the agent, extending MessagesState with model information.

    Args:
        model_id (str): The model ID in LiteLLM format.
        messages (list[BaseMessage]): The list of messages, inherited from MessagesState.
    """

    model_id: str = "openrouter/google/gemini-2.0-flash-001"


def invoke_react_agent(state: AgentState) -> AgentState:
    """Create the ReAct agent and invoke it with the messages.

    Args:
        state (AgentState): The state of the messages.

    Returns:
        AgentState: The updated state of the messages.
    """
    llm = ChatLiteLLM(
        model=state["model_id"],
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    agent = create_react_agent(
        model=llm,
        tools=get_tools(),
        version="v2",
    )

    response = agent.invoke({"messages": state["messages"]})
    # In deployment, the history is automatically handled by the add_messages function in between the nodes
    # In chain, the history is manually handled by the calling the add_messages function
    return {"messages": response["messages"]}


def create_graph() -> CompiledStateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("invoke_react_agent", invoke_react_agent, retry=RetryPolicy(max_attempts=3))
    builder.add_edge(START, "invoke_react_agent")
    builder.add_edge("invoke_react_agent", END)
    return builder.compile()


graph = create_graph()


class UniversalChain:
    """A chain implementation that wraps the compiled state graph with manual memory management outside LangGraph deployment, with contrast to the automatic memory management in deployment."""

    def __init__(self, model_id: str, llm: Optional[BaseChatModel] = None):
        """Initialize the UniversalChain with a language model.

        Args:
            model_id (str): The model ID in LiteLLM format.
            llm (BaseChatModel, optional): For overriding the default BaseChatModel.
        """
        self.model_id = model_id
        self.graph: CompiledStateGraph = create_graph()
        self.result: MessagesState = {"messages": []}

    @property
    def messages(self) -> list[BaseMessage]:
        """Get the messages from the result."""
        return self.result["messages"]

    def invoke(self, text: str, image: Optional[str] = None) -> MessagesState:
        """Invoke and get the response from the ReAct agent.

        Args:
            text (str): The input text.
            image (str, optional): Path or URL to an image to include.

        Returns:
            MessagesState: The updated message state after processing.
        """
        config = {"configurable": {"thread_id": "universal-chain-session"}}

        message = create_message(text, image)
        # Merges message into existing messages without duplicates due to auto ID handling
        messages = add_messages(self.messages, message)

        # Update the result for storing history
        self.result = self.graph.invoke(
            {
                "model_id": self.model_id,
                "messages": messages,
            },
            config,
        )
        return self.result

    def invoke_as_str(self, text: str, image: Optional[str] = None) -> str:
        """Invoke and get the string response from the ReAct agent.

        Args:
            text (str): The input text.
            image (str, optional): Path or URL to an image to include.

        Returns:
            str: The response string.
        """
        self.invoke(text, image)
        return self.messages[-1].content


def create_message(text: str, image: Optional[str] = None) -> HumanMessage:
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
    return HumanMessage(content=content)
