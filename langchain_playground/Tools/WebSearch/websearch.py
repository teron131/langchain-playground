import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()


@dataclass
class WebSearchArgs:
    query: str
    max_results: int = 5
    filter_score: float = 0.5
    summarize_content: bool = False
    suggested_answer: bool = False


def tavily_search(websearch_args: WebSearchArgs) -> dict:
    """Execute a Tavily search query.
    https://docs.tavily.com/documentation/api-reference/endpoint/search

    args:
        websearch_args (WebSearchArgs): Search arguments containing query and max_results

    Returns:
        dict: Raw Tavily API response
    """
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return tavily_client.search(
        websearch_args.query,
        max_results=websearch_args.max_results,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=True,
        include_image=False,
    )


def filter_garbage(text: str) -> str:
    """Removes garbage characters while keeping printable ASCII, Chinese characters, and emojis intact.

    args:
        text (str): The input string containing various characters

    Returns:
        str: A cleaned string with garbage characters removed
    """
    PRINTABLE_ASCII = lambda c: 32 <= ord(c) <= 126
    CONTROL_CHARS = "\n\t\r"
    CHINESE = lambda c: "\u4e00" <= c <= "\u9fff"
    MISC_SYMBOLS = lambda c: "\u2600" <= c <= "\u26ff"
    DINGBATS = lambda c: "\u2700" <= c <= "\u27bf"
    EMOJIS = lambda c: "\U0001f300" <= c <= "\U0001f9ff"

    def is_valid_char(c):
        return PRINTABLE_ASCII(c) or c in CONTROL_CHARS or CHINESE(c) or MISC_SYMBOLS(c) or DINGBATS(c) or EMOJIS(c)

    cleaned_text = "".join(char for char in text if is_valid_char(char))
    cleaned_text = re.sub(r"\n{3,}", r"\n\n", cleaned_text)
    cleaned_text = re.sub(r" {2,}", " ", cleaned_text)

    return cleaned_text


def summarize_content(raw_content_list: list[str]) -> list[str]:
    """Batch summarize content.

    Args:
        raw_content_list (list[str]): List of raw content strings to summarize

    Returns:
        list[str]: List of summarized content strings
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize the following content in 200 words."),
            ("human", "{raw_content}"),
        ]
    )

    llm = ChatOpenAI(
        model="google/gemini-2.0-flash-001",
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    chain = prompt | llm

    results = chain.batch(raw_content_list)
    return [result.content for result in results]


def process_response(response: dict, websearch_args: WebSearchArgs) -> dict:
    """Process Tavily search response by filtering results, cleaning content, and summarizing.

    args:
        response (dict): Raw Tavily API response
        websearch_args (WebSearchArgs): Search arguments containing processing options

    Returns:
        dict: Processed response
    """
    # Filter results by score
    response["results"] = [result for result in response["results"] if result["score"] >= websearch_args.filter_score]

    # Filter garbage characters
    for result in response["results"]:
        if result["content"] is not None:
            result["content"] = filter_garbage(result["content"])
        if result["raw_content"] is not None:
            result["raw_content"] = filter_garbage(result["raw_content"])

    # Summarize "raw_content" to replace "content"
    if websearch_args.summarize_content:
        raw_content_list = [result["raw_content"] for result in response["results"]]
        none_idx = [i for i, content in enumerate(raw_content_list) if content is None]

        summary_list = summarize_content(raw_content_list)
        content_list = [None if i in none_idx else summary for i, summary in enumerate(summary_list)]

        for i, result in enumerate(response["results"]):
            if content_list[i] is not None:
                result["content"] = content_list[i]

    return response


def format_response(response: dict, websearch_args: WebSearchArgs) -> str:
    """Format the processed response into a string.

    Args:
        response (dict): Processed response
        websearch_args (WebSearchArgs): Search arguments

    Returns:
        str: Formatted string containing filtered and processed search results
    """
    formatted_output = [
        f"Query: {response['query']}\n",
        "Sources:\n",
    ]

    for result in response["results"]:
        formatted_output.extend(
            [
                f"Relevance Score: {result['score']}",
                f"URL: {result['url']}",
                f"Content: {result['content']}\n",
            ]
        )

    if websearch_args.suggested_answer:
        formatted_output.append(f"Suggested Answer: {response['answer']}")

    return "\n".join(formatted_output)


def websearch(query: str) -> str:
    """Search the web for information based on the query.

    Args:
        query (str): Search query string

    Returns:
        str: Processed and formatted search results
    """
    websearch_args = WebSearchArgs(
        query=query,
        max_results=5,
        filter_score=0.5,
        summarize_content=True,
        suggested_answer=True,
    )
    response = tavily_search(websearch_args)
    processed_response = process_response(response, websearch_args)
    formatted_response = format_response(processed_response, websearch_args)
    return formatted_response
