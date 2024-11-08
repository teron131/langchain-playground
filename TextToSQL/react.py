import re

from langchain import hub
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

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
    sql_system_prompt = sql_prompt.messages[0].prompt.template

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent = create_react_agent(llm, toolkit.get_tools(), state_modifier=sql_system_prompt)

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

        # Trace the query execution in reverse manner
        reversed_messages = reversed(response["messages"])

        answer_msg = next((msg for msg in reversed_messages if isinstance(msg, AIMessage)), None)
        answer = answer_msg.content if answer_msg else ""

        data_msg = next((msg for msg in reversed_messages if isinstance(msg, ToolMessage) and msg.name == "sql_db_query"), None)
        data = data_msg.content if data_msg else ""
        query_call_id = data_msg.tool_call_id if data_msg else ""

        # Find the query execution matching the query call id
        query_msg = next((msg for msg in reversed_messages if isinstance(msg, AIMessage) and msg.tool_calls and any(call["id"] == query_call_id for call in msg.tool_calls)), None)
        query = query_msg.tool_calls[0]["args"]["query"] if query_msg else ""

        table_name = re.search(r"FROM\s+(\w+)", query, re.IGNORECASE).group(1) if query else ""

        return f"""
    {answer}
    ```sql
    {format_query(query)}

    {table_name}:
    {data_to_table(query, data)}
    ```
    """

    except Exception as e:
        return f"{e}"
