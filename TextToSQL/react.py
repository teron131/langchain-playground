import re

from langchain import hub
from langchain.agents import AgentExecutor
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from TextToSQL.utils import data_to_table, format_query


def text_to_sql_react(user_message: str) -> str:
    """
    Query a SQLite database using natural language and return formatted results.

    Uses a prebuilt ReAct agent with SQLDatabaseToolkit to:
    1. Parse natural language into SQL queries
    2. Execute queries against the database
    3. Format results into a natural language response with query details
    4. Handle error cases gracefully

    Args:
        user_message (str): Natural language question about the database

    Returns:
        str: Formatted response containing:
            - Natural language answer
            - SQL query used
            - Query results in table format
    """
    db = SQLDatabase.from_uri("sqlite:///databases/Chinook.db")

    sql_prompt = hub.pull("langchain-ai/sql-agent-system-prompt")
    # react_prompt = hub.pull("hwchase17/react")
    # combined_prompt = ChatPromptTemplate.from_messages(
    #     [
    #         *sql_prompt.messages,
    #         ("system", react_prompt.template),
    #     ]
    # )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_react_agent(llm, toolkit.get_tools(), state_modifier=sql_prompt.messages[0].prompt.template)
    # agent_executor = AgentExecutor(
    #     agent=agent,
    #     tools=toolkit.get_tools(),
    #     return_intermediate_steps=True,
    #     handle_parsing_errors=True,
    # )

    config = {"configurable": {"session_id": "text-to-sql-react-chain-session"}}
    dialect = "SQLite"
    top_k = 20

    try:
        response = agent.invoke(
            {
                "messages": [("user", user_message)],
                "dialect": dialect,
                "top_k": top_k,
            },
            config,
        )

        answer = next(msg for msg in reversed(response["messages"]) if isinstance(msg, AIMessage)).content
        query = next(msg for msg in reversed(response["messages"]) if isinstance(msg, ToolMessage) and msg.name == "sql_db_query").content
        data = next(msg for msg in reversed(response["messages"]) if isinstance(msg, ToolMessage) and msg.name == "sql_db_query_checker").content

        last_step = response[]AIMessage
        last_step = response["intermediate_steps"][-1]
        if last_step[0].tool == "sql_db_query":
            query = last_step[0].tool_input
            data = last_step[1]
            table_name = re.search(r"FROM\s+(\w+)", query, re.IGNORECASE).group(1)
        return response
    #             return f"""
    # {response["output"]}
    # ```sql
    # {format_query(query)}

    # {table_name}:
    # {data_to_table(query, data)}
    # ```
    # """
    #         else:
    #             return response["output"]

    except Exception as e:
        return f"{e}"
