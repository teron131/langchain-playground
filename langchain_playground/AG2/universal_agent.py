import os

import agentops
from autogen import AssistantAgent, UserProxyAgent, filter_config, register_function
from autogen.agentchat import ChatResult
from autogen.cache import Cache
from autogen.coding import LocalCommandLineCodeExecutor
from config import llm_config
from dotenv import load_dotenv

from langchain_playground.UniversalChain.tools import (
    webloader,
    websearch,
    youtubeloader,
)

load_dotenv()

filter_dict = {"model": ["gpt-4o-mini"]}
llm_config["config_list"] = filter_config(llm_config["config_list"], filter_dict)

# Setting up code executor
os.makedirs("coding", exist_ok=True)
code_executor = LocalCommandLineCodeExecutor(work_dir="coding")

user_proxy = UserProxyAgent(
    name="User",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"executor": code_executor},
    llm_config=llm_config,
)

assistant = AssistantAgent(
    name="Assistant",
    system_message="Only use the tools you have been provided with. Reply TERMINATE when the task is done.",
    llm_config=llm_config,
)

register_function(
    websearch,
    caller=assistant,
    executor=user_proxy,
    description="Search the web for information based on the query.",
)

register_function(
    webloader,
    caller=assistant,
    executor=user_proxy,
    description="Load the content of a website from url to text.",
)

register_function(
    youtubeloader,
    caller=assistant,
    executor=user_proxy,
    description="Load the subtitles of a YouTube video by url in form such as: https://www.youtube.com/watch?v=..., https://youtu.be/..., or more.",
)


def get_result(question: str) -> ChatResult:
    agentops.init()

    with Cache.disk(cache_seed=43) as cache:
        result = user_proxy.initiate_chat(
            assistant,
            message=question,
            cache=cache,
        )

    agentops.end_session("Success")
    return result


def invoke(question: str) -> str:
    result = get_result(question)
    return result.summary
