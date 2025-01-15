import autogen
from autogen import GroupChat, GroupChatManager, filter_config
from autogen.agentchat import ChatResult
from autogen.agentchat.contrib.agent_builder import AgentBuilder
from config import llm_config
from dotenv import load_dotenv

load_dotenv()

filter_dict = {"model": ["gpt-4o-mini"]}
llm_config["config_list"] = filter_config(llm_config["config_list"], filter_dict)

builder = AgentBuilder(
    config_file_or_env="OAI_CONFIG_LIST",
    builder_model="gpt-4o-mini",
    agent_model="gpt-4o-mini",
)


def get_result(task: str) -> ChatResult:
    agent_list, agent_configs = builder.build(task, llm_config, coding=True)

    group_chat = GroupChat(
        agents=agent_list,
        messages=[],
        max_round=12,
    )
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )

    chat_result = agent_list[0].initiate_chat(
        manager,
        message=task,
    )

    return chat_result


def invoke(task: str) -> str:
    result = get_result(task)
    return result.summary
