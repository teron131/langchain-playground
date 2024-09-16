import os
from typing import Any, Dict, List, Optional, Union

import opencc
from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models.base import init_chat_model
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts.structured import StructuredPrompt
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai.chat_models.base import ChatOpenAI

messages = []
chat_history = InMemoryChatMessageHistory(messages=messages[:-1])


def get_session_history():
    return chat_history


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Let's inspect some of the attributes associated with the tool.
print(multiply.name)
print(multiply.description)
print(multiply.args)
load_dotenv()

# Universal Initiator of OpenRouter
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    model_name="anthropic/claude-3.5-sonnet",
    temperature=1,
)


def s2hk(content: str) -> str:
    converter = opencc.OpenCC("s2hk")
    return converter.convert(content)


prompt = hub.pull("hardkothari/prompt-maker")
prompt = StructuredPrompt(
    input_variables=["input1", "input2"],
    messages=[
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=[],
                template="System prompt",
            )
        ),
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=["input1", "input2"],
                template="Human prompt",
            )
        ),
    ],
    schema_={
        "type": "object",
        "title": "extract",
        "required": ["output1", "output2"],
        "properties": {
            "output1": {
                "enum": ["option1", "option2"],
                "type": "string",
                "description": "...",
            },
            "output2": {
                "type": "string",
                "description": "...",
            },
        },
        "description": "...",
    },
)
prompt = ChatPromptTemplate(
    messages=[
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
    ]
)
llm = init_chat_model(model="gpt-4o-mini", model_provider="openai", temperature=1)
tools = [multiply]
llm_with_tools = llm.bind_tools(tools)
chain = prompt | llm_with_tools | StrOutputParser() | RunnableLambda(s2hk)
chain_with_history = RunnableWithMessageHistory(chain, get_session_history)


if __name__ == "__main__":
    response = chain.invoke({"input": "What is 999.999*11.1111"})
    print(response)
