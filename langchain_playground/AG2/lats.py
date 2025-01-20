import agentops
from autogen import ConversableAgent, ReasoningAgent, UserProxyAgent, filter_config
from autogen.agentchat import ChatResult
from config import llm_config
from dotenv import load_dotenv

load_dotenv()

llm_config["config_list"] = filter_config(
    config_list=llm_config["config_list"],
    filter_dict={"model": ["gpt-4o-mini"]},
)


def last_meaningful_msg(sender: ConversableAgent, recipient: ConversableAgent, summary_args: dict):
    import warnings

    if sender == recipient:
        return "TERMINATE"

    summary = ""
    chat_messages = recipient.chat_messages[sender]

    for msg in reversed(chat_messages):
        try:
            content = msg["content"]
            if isinstance(content, str):
                summary = content.replace("TERMINATE", "")
            elif isinstance(content, list):
                # Remove the `TERMINATE` word in the content list.
                summary = "\n".join(x["text"].replace("TERMINATE", "") for x in content if isinstance(x, dict) and "text" in x)
            if summary.strip().rstrip():
                return summary
        except (IndexError, AttributeError) as e:
            warnings.warn(f"Cannot extract summary using last_msg: {e}. Using an empty str as summary.", UserWarning)
    return summary


lats_agent = ReasoningAgent(
    name="LATS_Agent",
    llm_config=llm_config,
    verbose=False,
    reason_config={
        "method": "lats",
        "nsim": 3,
        "forest_size": 3,
        "max_depth": 3,
    },
)


user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
    max_consecutive_auto_reply=10,
)


def get_result(question: str) -> ChatResult:
    agentops.init()
    result = user_proxy.initiate_chat(
        lats_agent,
        message=question,
        summary_method=last_meaningful_msg,
    )
    agentops.end_session("Success")
    return result


def invoke(question: str) -> str:
    result = get_result(question)
    return result.summary


def find_best_path(node: dict) -> list:
    """
    Find the path to the node with highest score and depth.

    Args:
        node (dict): Current node in the tree

    Returns:
        list: List of nodes representing the best path
    """
    if not node.get("children"):
        return [node]

    # Find child with best score/depth using max() with a tuple key
    best_child = max(
        ((child, find_best_path(child)) for child in node["children"]),
        key=lambda x: (x[0].get("value", 0), len(x[1])),
    )[1]

    return [node] + best_child


def print_best_path(tree: dict):
    """
    Print the path with highest score and depth from root to leaf.

    Args:
        tree (dict): Dictionary representing the tree structure
    """
    best_path = find_best_path(tree)

    print("\nBest Reasoning Path:")
    print("=" * 70)

    for depth, node in enumerate(best_path):
        indent = "  " * depth
        content = node["content"]

        # For the final node (leaf), print full content after the tree
        if depth == len(best_path) - 1:
            print("\nFinal Detailed Answer:")
            print("=" * 70)
            print(content)
            continue

        print(f"{indent}├─ {content}")
        if node.get("value"):
            print(f"{indent}│  Value: {node['value']:.3f}")


def invoke_with_path(question: str) -> str:
    result = get_result(question)
    data = lats_agent._root.to_dict()
    print_best_path(data)
    return result.summary
