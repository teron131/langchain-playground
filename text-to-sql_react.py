import ast
import os
import re
from typing import Generator, Iterator, List, Union

from langchain import hub
from langchain.agents.agent import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI
from pydantic import BaseModel
from tabulate import tabulate


class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
        LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "")
        pass

    def __init__(self):
        self.name = "Text-to-SQL ReAct"
        self.valves = self.Valves()
        self.db = SQLDatabase.from_uri("sqlite:///databases/Chinook.db")
        self.chain = self.create_chain()
        pass

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    def clean_sql(self, sql):
        sql = sql.strip("`").split("\n", 1)[-1].rsplit("\n", 1)[0]
        return sql

    def create_chain(self):
        sql_prompt = hub.pull("langchain-ai/sql-agent-system-prompt")
        react_prompt = hub.pull("hwchase17/react")
        combined_prompt = ChatPromptTemplate.from_messages(
            [
                *sql_prompt.messages,
                ("system", react_prompt.template),
            ]
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        toolkit = SQLDatabaseToolkit(db=self.db, llm=llm)

        agent = create_react_agent(llm, toolkit.get_tools(), combined_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=toolkit.get_tools(), return_intermediate_steps=True)

        return agent_executor

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        def format_sql_query(query: str) -> str:
            """Format SQL query with proper line breaks and indentation."""
            keywords = ["SELECT", "FROM", "JOIN", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT", "OFFSET"]
            query = " ".join(query.strip().split())  # Normalize whitespace

            # Convert keywords to uppercase and add line breaks
            for keyword in keywords:
                query = re.sub(rf"\b{keyword}\b", keyword, query, flags=re.IGNORECASE)
                query = query.replace(f" {keyword} ", f"\n{keyword} ")

            # Special handling for JOIN...ON
            query = query.replace(" ON ", "\n    ON ")

            return query

        def data_to_table(query: str, data: str) -> str:
            """Convert SQL query results to a formatted table."""
            # Extract column headers from SELECT clause
            SELECT_clause = re.search(r"SELECT\s+(.*?)\s+FROM", query, re.IGNORECASE).group(1)
            headers = [col.strip() for col in SELECT_clause.split(",")]
            headers = [re.split(r"\s+as\s+", col.strip(), flags=re.IGNORECASE)[-1] for col in headers]

            # Parse data string into list of tuples
            data_list = ast.literal_eval(data) if isinstance(data, str) else []

            return tabulate(data_list, headers, tablefmt="pretty", showindex=True)

        config = {"configurable": {"session_id": "text-to-sql-react-chain-session"}}
        dialect = "SQLite"
        top_k = 20

        try:
            response = self.chain.invoke({"input": user_message, "dialect": dialect, "top_k": top_k}, config)
            last_step = response["intermediate_steps"][-1]
            if last_step[0].tool == "sql_db_query":
                query = last_step[0].tool_input
                data = last_step[1]
                table_name = re.search(r"FROM\s+(\w+)", query, re.IGNORECASE).group(1)

                return f"""
{response["output"]}
```sql
{format_sql_query(query)}

{table_name}:
{data_to_table(query, data)}
```
"""

        except Exception as e:
            return f"Error: {e}"
