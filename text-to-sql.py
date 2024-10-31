import os
from typing import Generator, Iterator, List, Union

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools.sql_database.tool import (
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.prompts.chat import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai.chat_models.base import ChatOpenAI
from pydantic import BaseModel


class Pipeline:
    class Valves(BaseModel):
        OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
        LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
        LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
        LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "")
        pass

    def __init__(self):
        self.name = "Text-to-SQL"
        self.valves = self.Valves()
        self.db = SQLDatabase.from_uri("sqlite:///databases/Chinook.db")
        pass

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        prompt = PromptTemplate.from_template(
            """
You are a helpful assistant that can answer questions with reference to the SQL query result from the database.
Question:
{question}
Query:
{query}
Result:
{result}
Answer:
"""
        )

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        write_query = create_sql_query_chain(llm, self.db)
        execute_query = QuerySQLDataBaseTool(db=self.db)

        def clean_query(sql):
            cleaned_sql = QuerySQLCheckerTool(db=self.db, llm=llm).invoke(sql)
            return cleaned_sql.replace("```sql", "").replace("```", "").strip()

        chain = (
            RunnableParallel(
                query=write_query,
                question=RunnablePassthrough(),
            )
            | RunnablePassthrough.assign(question=lambda x: x["question"]["question"])
            | RunnablePassthrough.assign(query=lambda x: clean_query(x["query"]))
            | RunnablePassthrough.assign(result=lambda x: execute_query.run(x["query"]))
            | prompt
            | llm
        )

        try:
            result = chain.invoke({"question": user_message})
            return result.content
        except Exception as e:
            return f"{e}"
