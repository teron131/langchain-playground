import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph, add_messages
from langgraph.prebuilt import create_react_agent

from .Tools import get_tools


class State(MessagesState):
    model_id: str = "google/gemini-2.0-flash-001"


def invoke_react_agent(state: State) -> State:
    print(state)
    llm = ChatOpenAI(
        model=state["model_id"],
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    chain = create_react_agent(
        llm,
        tools=get_tools(),
        version="v2",
    )
    # Invoke the chain with the messages
    response = chain.invoke({"messages": state["messages"]})
    # Return a dictionary with the updated state
    print(response)
    return {"messages": response["messages"]}


builder = StateGraph(State)
builder.add_node("invoke_react_agent", invoke_react_agent)
builder.add_edge(START, "invoke_react_agent")
builder.add_edge("invoke_react_agent", END)
graph = builder.compile()

# Example usage:
# graph.invoke(
#     {
#         "model_id": "google/gemini-2.0-flash-001",
#         "messages": [HumanMessage(content="Hello, world!")],
#     }
# )
