import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from .utils import s2hk

load_dotenv()

PROMPT = """You are an expert subtitles editor. Your task is to refine a sequence of piecemeal subtitles derived from transcription. These subtitles may contain typos and lack proper punctuation. Follow the guidelines below to ensure high-quality subtitles:

Instructions:
1. Make minimal contextual changes.
2. Only make contextual changes if you are highly confident.
3. Add punctuation appropriately.
4. Separate into paragraphs appropriately.

Example:
Original Subtitle: welcome back fellow history enthusiasts to our channel today we embark on a thrilling expedition
Refined Subtitle: Welcome back, fellow history enthusiasts, to our channel! Today, we embark on a thrilling expedition."""


def llm_format_text(subtitles: str, chunk_size: int = 1000) -> str:
    """
    Format subtitles using LLM.
    Note that Chinese are longer than it seems.

    Args:
        subtitles (str): The subtitles to format
        chunk_size (int, optional): Size of chunks to process. Defaults to 1000.

    Returns:
        str: The formatted subtitles
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PROMPT),
            ("human", "{subtitles}"),
        ]
    )

    llm = ChatOpenAI(
        model="google/gemini-2.0-flash-001",
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    chain = prompt | llm | StrOutputParser() | RunnableLambda(s2hk)

    chunked_subtitles = [subtitles[i : i + chunk_size] for i in range(0, len(subtitles), chunk_size)]
    formatted_subtitles = chain.batch([{"subtitles": chunk} for chunk in chunked_subtitles])
    return "".join(formatted_subtitles)


import io

from google import genai

from .utils import s2hk


def llm_format_text_audio(subtitles: str, audio_bytes: bytes) -> str:
    """
    Format subtitles using LLM.

    Args:
        subtitles (str): The subtitles to format
        audio_bytes (bytes): The audio bytes to format

    Returns:
        str: The formatted subtitles
    """
    client = genai.Client()

    with io.BytesIO(audio_bytes) as in_memory_file:
        audio_file = client.files.upload(
            file=in_memory_file,
            config={"mimeType": "audio/mp3"},
        )

    prompt = PROMPT + "\n\nWith reference to the audio, refine the subtitles if necessary."

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, subtitles, audio_file],
    )
    client.files.delete(name=audio_file.name)

    return s2hk(response.text)
