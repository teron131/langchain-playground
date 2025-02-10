import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv
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

    args:
        websearch_args (WebSearchArgs): Search arguments containing query and max_results

    Returns:
        dict: Raw Tavily API response
    """
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
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
    MISC_SYMBOLS = lambda c: "\u2600" <= c <= "\u26FF"
    DINGBATS = lambda c: "\u2700" <= c <= "\u27BF"
    EMOJIS = lambda c: "\U0001F300" <= c <= "\U0001F9FF"

    def is_valid_char(c):
        return PRINTABLE_ASCII(c) or c in CONTROL_CHARS or CHINESE(c) or MISC_SYMBOLS(c) or DINGBATS(c) or EMOJIS(c)

    cleaned_text = "".join(char for char in text if is_valid_char(char))
    cleaned_text = re.sub(r"\n{3,}", r"\n\n", cleaned_text)
    cleaned_text = re.sub(r" {2,}", " ", cleaned_text)

    return cleaned_text


def _summarize_content(response: dict) -> None:
    """Helper function to summarize content using LLM."""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    summarizer = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize the following content in 200 words: {raw_content}"),
        ]
    )
    chain = prompt | summarizer

    raw_content_list = [result["raw_content"] for result in response["results"]]
    none_idx = [i for i, content in enumerate(raw_content_list) if content is None]

    summarized_content_list = chain.batch(raw_content_list)
    content_list = [None if i in none_idx else msg.content for i, msg in enumerate(summarized_content_list)]

    for i, result in enumerate(response["results"]):
        if content_list[i] is not None:
            result["content"] = content_list[i]


def process_response(response: dict, websearch_args: WebSearchArgs) -> str:
    """Process Tavily search response by filtering results and cleaning content.

    args:
        response (dict): Raw Tavily API response
        websearch_args (WebSearchArgs): Search arguments containing processing options

    Returns:
        str: Formatted string containing filtered and processed search results
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
        _summarize_content(response)

    content = [
        f"Query: {response['query']}\n",
        "Sources:\n",
    ]

    for result in response["results"]:
        content.extend(
            [
                f"Relevance Score: {result['score']}",
                f"URL: {result['url']}",
                f"Content: {result['content']}\n",
            ]
        )

    if websearch_args.suggested_answer:
        content.append(f"Suggested Answer: {response['answer']}")

    return "\n".join(content)


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
    return process_response(response, websearch_args)
