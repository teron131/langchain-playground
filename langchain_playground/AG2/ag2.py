import os
from datetime import datetime
from typing import Callable, Dict, Literal, Optional, Union

from autogen import (
    AssistantAgent,
    UserProxyAgent,
    config_list_from_json,
    register_function,
)
from autogen.agentchat import (
    Agent,
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    register_function,
)
from autogen.agentchat.contrib import agent_builder
from autogen.agentchat.contrib.capabilities import teachability
from autogen.cache.cache import Cache
from autogen.coding import DockerCommandLineCodeExecutor, LocalCommandLineCodeExecutor
from config import llm_config
from dotenv import load_dotenv
from typing_extensions import Annotated

load_dotenv()

assistant = ConversableAgent(name="assistant", llm_config=llm_config)
messages = [{"role": "user", "content": "Explain gradient descent"}]
assistant.generate_oai_reply(messages)

recipient = ConversableAgent(name="recipient", llm_config=llm_config)

with Cache.disk() as cache:
    assistant.initiate_chat(recipient=recipient, message=messages[0], cache=cache)


assistant.print_usage_summary()
