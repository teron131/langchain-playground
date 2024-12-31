import asyncio
from typing import List

import openai
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from .config import long_context_llm
from .models import WikiSection

MAX_RETRIES = 3
RETRY_DELAY = 2

# Initialize embeddings and vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = InMemoryVectorStore(embeddings)
retriever = vectorstore.as_retriever(k=3)


async def initialize_vectorstore(references: dict):
    print("\n📚 Initializing vector store with references...")
    reference_docs = [Document(page_content=v, metadata={"source": k}) for k, v in references.items()]
    await vectorstore.aadd_documents(reference_docs)
    print(f"✅ Added {len(reference_docs)} documents to vector store")


async def test_retriever():
    return await retriever.ainvoke("What's a long context LLM anyway?")


# Generate Sections
section_writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert Wikipedia writer. Complete your assigned WikiSection from the following outline:
{outline}

Cite your sources, using the following references:
<Documents>
{docs}
</Documents>
""",
        ),
        ("user", "Write the full WikiSection for the {section} section."),
    ]
)


async def retrieve(inputs: dict):
    print(f"\n🔍 Retrieving relevant documents for section: {inputs['section']}")
    docs = await retriever.ainvoke(inputs["topic"] + ": " + inputs["section"])
    formatted = "\n".join([f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>' for doc in docs])
    print(f"✅ Found {len(docs)} relevant documents")
    return {"docs": formatted, **inputs}


section_writer = retrieve | section_writer_prompt | long_context_llm.with_structured_output(WikiSection)


async def write_section(outline, section_title, topic):
    print(f"\n📝 Writing section: {section_title}")
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            section = await section_writer.ainvoke(
                {
                    "outline": outline.as_str,
                    "section": section_title,
                    "topic": topic,
                }
            )
            print(f"✅ Completed section ({len(section.content)} chars)")
            return section
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n⚠️ Section writing attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"\n❌ Failed to write section {section_title}")
                # Create a basic section as fallback
                return WikiSection(section_title=section_title, content=f"Content generation failed for this section. Please refer to the outline:\n\n{outline.as_str}", citations=[])


# Generate final article
writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert Wikipedia author. Write the complete wiki article on {topic} using the following section drafts:
{draft}

Strictly follow Wikipedia format guidelines, but keep the content concise and focused.
Aim to maintain clarity while being efficient with the content length.
""",
        ),
        (
            "user",
            'Write the complete Wiki article using markdown format. Organize citations using footnotes like "[1]",' " avoiding duplicates in the footer. Include URLs in the footer.",
        ),
    ]
)

# Simpler fallback prompt for when the main prompt fails
simple_writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Write a clear and concise Wikipedia article on {topic}. Use the following content as your source:
{draft}

Focus on presenting the key information in a straightforward manner.
Keep the content brief but informative.
""",
        ),
        (
            "user",
            "Write the article in markdown format with simple citation numbers [1] and a list of sources at the end.",
        ),
    ]
)


async def generate_with_fallback(inputs: dict) -> str:
    """Generate article with fallback to simpler format if needed."""
    max_retries = 3
    retry_delay = 5  # seconds

    async def attempt_generation(prompt):
        for attempt in range(max_retries):
            try:
                return await (prompt | long_context_llm | StrOutputParser()).ainvoke(inputs)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"\n⚠️ Generation attempt {attempt + 1} failed: {str(e)}")
                    print(f"Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise

    try:
        # Try with main prompt
        return await attempt_generation(writer_prompt)
    except Exception as e:
        print("\n⚠️ Main generation failed, using simplified format...")
        try:
            # Try with simplified prompt
            return await attempt_generation(simple_writer_prompt)
        except Exception as fallback_e:
            print(f"\n❌ All generation attempts failed: {str(fallback_e)}")
            # Return a basic formatted version of the draft as last resort
            return f"# {inputs['topic']}\n\n{inputs['draft']}"


writer = RunnableLambda(generate_with_fallback)


async def stream_writer(topic, section):
    print("\n📖 Generating final article...")
    try:
        result = await writer.ainvoke({"topic": topic, "draft": section.as_str})
        print(result)
        print("\n✅ Article generation complete")
    except Exception as e:
        print(f"\n⚠️ Error during article generation: {str(e)}")
        print("\n🔄 Retrying with simplified format...")
        # Final fallback: generate a basic article
        simple_result = await (simple_writer_prompt | long_context_llm | StrOutputParser()).ainvoke({"topic": topic, "draft": section.as_str})
        print(simple_result)
        print("\n✅ Article generation complete (simplified format)")
