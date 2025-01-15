import agentops
from autogen import GroupChat, GroupChatManager, config_list_from_json
from autogen.agentchat import ChatResult
from autogen.agentchat.contrib.agent_builder import AgentBuilder
from config import llm_config
from dotenv import load_dotenv

load_dotenv()

config_list = config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={"model": ["gpt-4o-mini"]},
)
llm_config["config_list"] = config_list


builder = AgentBuilder(
    config_file_or_env="OAI_CONFIG_LIST",
    builder_model="gpt-4o-mini",
    agent_model="gpt-4o-mini",
)


def invoke(task: str) -> ChatResult:
    agent_list, agent_configs = builder.build(task, llm_config, coding=True)

    group_chat = GroupChat(
        agents=agent_list,
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin",
    )
    manager = GroupChatManager(
        groupchat=group_chat,
        name="Task Manager",
        system_message="""You are the manager of the team. You are responsible for the overall task and the progress of the team. You are also responsible for the communication between agents.""",
        llm_config=llm_config,
    )

    agentops.init()
    chat_result = agent_list[0].initiate_chat(
        manager,
        message=task,
    )
    agentops.end_session("Success")
    return chat_result
