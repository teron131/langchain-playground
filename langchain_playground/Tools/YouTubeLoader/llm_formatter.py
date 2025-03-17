import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from .utils import s2hk

load_dotenv()


def llm_format_txt(content: str, chunk_size: int = 1000) -> str:
    """Format text content using LLM.
    Note that Chinese are longer than it seems.
    Args:
        content (str): The text content to format
        chunk_size (int, optional): Size of chunks to process. Defaults to 1000.

    Returns:
        str: The formatted text content
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert subtitles editor. Your task is to refine a sequence of piecemeal subtitles derived from transcription. These subtitles may contain typos and lack proper punctuation. Follow the guidelines below to ensure high-quality subtitles:

Instructions:
1. Make minimal contextual changes.
2. Only make contextual changes if you are highly confident.
3. Add punctuation appropriately.
4. Separate into paragraphs appropriately.

Example:
Original Subtitle: welcome back fellow history enthusiasts to our channel today we embark on a thrilling expedition
Refined Subtitle: Welcome back, fellow history enthusiasts, to our channel! Today, we embark on a thrilling expedition.""",
            ),
            ("human", "{subtitles}"),
        ]
    )

    llm = ChatOpenAI(
        model="google/gemini-2.0-flash-001",
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    preprocess_subtitles_chain = prompt | llm | StrOutputParser() | RunnableLambda(s2hk)

    chunked_subtitles = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    formatted_subtitles = preprocess_subtitles_chain.batch([{"subtitles": chunk} for chunk in chunked_subtitles])
    return "".join(formatted_subtitles)
