import os

import agentops
from autogen import AssistantAgent, UserProxyAgent, register_function
from autogen.cache import Cache
from autogen.coding import LocalCommandLineCodeExecutor
from config import llm_config
from dotenv import load_dotenv

from langchain_playground.UniversalChain.tools import youtubeloader

load_dotenv()
agentops.init(
    api_key=os.environ["AGENTOPS_API_KEY"],
    default_tags=["langchain-playground"],
)

# NOTE: this ReAct prompt is adapted from Langchain's ReAct agent: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/react/agent.py#L79
react_prompt = """
Answer the following questions as best you can. You have access to tools provided.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take
Action Input: the input to the action
Observation: the result of the action
... (this process can repeat multiple times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
"""


# Define the ReAct prompt message. Assuming a "question" field is present in the context
def react_prompt_message(sender, recipient, context):
    return react_prompt.format(input=context["question"])


# Setting up code executor.
os.makedirs("coding", exist_ok=True)
# Use docker executor for running code in a container if you have docker installed.
# code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
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
    youtubeloader,
    caller=assistant,
    executor=user_proxy,
    description="Load the subtitles of a YouTube video by url in form such as: https://www.youtube.com/watch?v=..., https://youtu.be/..., or more.",
)


def invoke(question: str):
    # Cache LLM responses. To get different responses, change the cache_seed value.
    with Cache.disk(cache_seed=43) as cache:
        user_proxy.initiate_chat(
            assistant,
            message=react_prompt_message,
            question=question,
            cache=cache,
        )


agentops.end_session("Success")
