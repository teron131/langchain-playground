from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from .utils import s2hk


def llm_format_txt(content: str, chunk_size: int = 1000) -> str:
    """Format text content using LLM.
    Note that Chinese are longer than it seems.
    Args:
        content (str): The text content to format
        chunk_size (int, optional): Size of chunks to process. Defaults to 1000.

    Returns:
        str: The formatted text content
    """
    preprocess_subtitles_chain = (
        hub.pull("preprocess_subtitles")
        | ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )
        | StrOutputParser()
        | RunnableLambda(s2hk)
    )

    chunked_subtitles = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    formatted_subtitles = preprocess_subtitles_chain.batch([{"subtitles": chunk} for chunk in chunked_subtitles])
    return "".join(formatted_subtitles)
