import agentops
from autogen import (
    AssistantAgent,
    ConversableAgent,
    GroupChat,
    GroupChatManager,
    UserProxyAgent,
    config_list_from_json,
    register_function,
)
from autogen.agentchat import ChatResult
from autogen.agentchat.contrib.agent_builder import AgentBuilder
from config import llm_config
from dotenv import load_dotenv

from langchain_playground.UniversalChain.tools import (
    webloader,
    websearch,
    youtubeloader,
)

load_dotenv()

config_list = config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model": ["gpt-4o-mini"]},
)
llm_config["config_list"] = config_list


def build_agents(task: str) -> tuple[list[ConversableAgent], UserProxyAgent]:
    builder = AgentBuilder(
        config_file_or_env="OAI_CONFIG_LIST",
        builder_model="gpt-4o-mini",
        agent_model="gpt-4o-mini",
    )

    agent_list, _ = builder.build(
        building_task=task,
        default_llm_config=llm_config,
        coding=True,
        max_agents=5,
    )

    user_proxy = agent_list[-1] if isinstance(agent_list[-1], UserProxyAgent) else next(agent for agent in agent_list if isinstance(agent, UserProxyAgent))

    return agent_list, user_proxy


def create_tools_assistant(agent_list: list[ConversableAgent], user_proxy: UserProxyAgent) -> list[ConversableAgent]:
    tools = [
        (websearch, "Search the web for information based on the query."),
        (webloader, "Load the content of a website from url to text."),
        (youtubeloader, "Load the subtitles of a YouTube video by url in form such as: https://www.youtube.com/watch?v=..., https://youtu.be/..., or more."),
    ]

    tools_assistant = AssistantAgent(
        name="Tools_Assistant",
        system_message="Only use the tools you have been provided with. Reply TERMINATE when the task is done.",
        description="""The only assistant in the team that has access to the tools. The available tools are:
1. Searching the web for relevant information
2. Loading and processing web content
3. Extracting information from YouTube videos""",
        llm_config=llm_config,
    )

    for tool, description in tools:
        register_function(
            tool,
            caller=tools_assistant,
            executor=user_proxy,
            description=description,
        )

    agent_list.append(tools_assistant)
    return agent_list


def setup_group_chat(agent_list):
    group_chat = GroupChat(
        agents=agent_list,
        messages=[],
        # max_round=12,
        speaker_selection_method="auto",
    )

    manager = GroupChatManager(
        groupchat=group_chat,
        name="Task Manager",
        system_message="You are the manager of the team. You are responsible for the overall task and the progress of the team. You are also responsible for the communication between agents.",
        llm_config=llm_config,
    )
    return manager


def create_team(task: str) -> tuple[GroupChatManager, list[ConversableAgent]]:
    agent_list, user_proxy = build_agents(task)
    agent_list = create_tools_assistant(agent_list, user_proxy)
    manager = setup_group_chat(agent_list)
    return manager, agent_list


def get_result(task: str) -> ChatResult:
    manager, agent_list = create_team(task)

    # User proxy agent is auto created by the builder
    # It is usually the last agent in the list
    user_proxy = agent_list[-1] if isinstance(agent_list[-1], UserProxyAgent) else next(agent for agent in agent_list if isinstance(agent, UserProxyAgent))

    agentops.init()
    chat_result = user_proxy.initiate_chat(
        manager,
        message=task,
    )
    agentops.end_session("Success")
    return chat_result


def invoke(task: str) -> str:
    chat_result = get_result(task)
    return chat_result.summary
