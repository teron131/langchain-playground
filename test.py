from langchain.agents.agent import AgentExecutor
from langchain.agents.tool_calling_agent.base import create_tool_calling_agent
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def add(a: float, b: float) -> float:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiplies a and b."""
    return a * b


llm = ChatOpenAI(model="gpt-4o-mini")
tools = [add, multiply]
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

query = "What is 223423.34534 + 2122321.1231245 and 223423.12312 * 2122321.3454334?"
stream = True
if stream:
    for chunk in agent_executor.stream({"input": query}):
        if "output" in chunk:
            for c in chunk["output"]:
                print(c, end="", flush=True)
    print()
else:
    print(agent_executor.invoke({"input": query})["output"])
