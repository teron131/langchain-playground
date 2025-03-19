from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.tools import QuerySQLCheckerTool, QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI

from .utils import data_to_table, format_query


def text_to_sql(user_message: str) -> str:
    """
    Query a SQLite database using natural language and return formatted results.

    Uses a simple pipeline chain that:
    1. Converts natural language to SQL query
    2. Cleans and validates the SQL query
    3. Executes the query against the database
    4. Formats the results into a natural language response

    Args:
        user_message (str): Natural language question about the database

    Returns:
        str: Natural language response with query results
    """
    db = SQLDatabase.from_uri("sqlite:///databases/Chinook.db")

    prompt = PromptTemplate.from_template(
        """You are a helpful assistant that can answer questions with reference to the SQL query result from the database.
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

    write_query = create_sql_query_chain(llm, db, k=20)
    execute_query = QuerySQLDataBaseTool(db=db)

    def clean_query(sql):
        """Clean and validate SQL query using QuerySQLCheckerTool.

        Args:
            sql (str): Raw SQL query string that may contain markdown formatting

        Returns:
            str: Cleaned SQL query with markdown and extra whitespace removed
        """
        cleaned_sql = QuerySQLCheckerTool(db=db, llm=llm).invoke(sql)
        return cleaned_sql.replace("```sql", "").replace("```", "").strip()

    chain = (
        RunnableParallel(
            query=write_query,
            question=RunnablePassthrough(),
        )
        | RunnablePassthrough.assign(question=lambda x: x["question"]["question"])
        | RunnablePassthrough.assign(query=lambda x: clean_query(x["query"]))
        | RunnablePassthrough.assign(result=lambda x: execute_query.run(x["query"]))
        | RunnablePassthrough.assign(
            prompt=lambda x: prompt.format(
                question=x["question"],
                query=x["query"],
                result=x["result"],
            )
        )
        | RunnablePassthrough.assign(answer=lambda x: llm.invoke(x["prompt"]))
    )

    try:
        result = chain.invoke({"question": user_message})
        return f"""
{result["answer"].content}

```sql
{format_query(result["query"])}

{data_to_table(result["query"], result["result"])}
```
"""
    except Exception as e:
        return f"{e}"
