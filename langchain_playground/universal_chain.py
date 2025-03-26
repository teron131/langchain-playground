import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, add_messages
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore

from .Tools import get_tools
from .utils import load_image_base64


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

    def create_input_message(
        self,
        input_text: str,
        input_image: str = None,
    ) -> HumanMessage:
        """Create a human message with text and optional image content.

        Args:
            input_text (str): The text input from the user
            input_image (str, optional): Path or URL to an image to include

        Returns:
            HumanMessage: A formatted message object containing text and optional image
        """
        content = []
        if input_text:
            content.append({"type": "text", "text": input_text})
        if input_image:
            try:
                image_base64 = load_image_base64(input_image)
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

    def get_response(
        self,
        input_text: str,
        input_image: str = None,
        history_messages: list[BaseMessage] = None,
    ) -> MessagesState:
        """Generate a response from the ReAct chain.

        Args:
            input_text (str): The input text.
            input_image (str, optional): Path or URL to an image to include.
            history_messages (list[BaseMessage], optional): Previous conversation messages.

        Returns:
            MessagesState: The updated message state after processing.
        """
        config = {"configurable": {"thread_id": "universal-chain-session"}}
        # Include message history if provided, otherwise just the current input
        # Specialized for Open WebUI as the LangChain memory would get lost, likely due to session management, but it has a variable: messages (list[tuple[str, str]])
        messages = history_messages or []

        input_message = self.create_input_message(input_text, input_image)
        messages = add_messages(messages, input_message)
        return self.chain.invoke({"messages": messages}, config)

    def invoke(
        self,
        input_text: str,
        input_image: str = None,
        history_messages: list[BaseMessage] = None,
    ) -> str:
        """Invoke the chain with the given input and message history.

        Args:
            input_text (str): The input text.
            input_image (str, optional): Path or URL to an image to include.
            history_messages (list[BaseMessage], optional): Previous conversation messages.

        Returns:
            str: The response string.
        """
        self.get_response(input_text, input_image, history_messages)
        return _extract_answer_message(self.result).content


def _extract_answer_message(messages: MessagesState) -> BaseMessage:
    """Extract the answer message from the last message."""
    return messages["messages"][-1]


def _extract_history_messages(messages: MessagesState) -> list[BaseMessage]:
    """Extract the history messages from the result."""
    return messages["messages"][:-1]
