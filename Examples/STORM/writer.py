"""Writer module for the STORM pipeline."""

import asyncio

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from .config import config
from .models import WikiSection
from .utils import ProgressTracker, RetryError, with_retries

# Initialize embeddings and vectorstore with timeout
embeddings = OpenAIEmbeddings(
    model=config.embedding_model,
    openai_api_key=config.openai_api_key,
    timeout=config.request_timeout,
)
vectorstore = InMemoryVectorStore(embeddings)
retriever = vectorstore.as_retriever(k=config.vector_store_k)


async def initialize_vectorstore(references: dict):
    """Initialize the vector store with reference documents."""
    print("\n📚 Initializing vector store with references...")
    reference_docs = [Document(page_content=v, metadata={"source": k}) for k, v in references.items()]

    # Process documents in batches to avoid timeouts
    batch_size = config.max_concurrent_requests
    for i in range(0, len(reference_docs), batch_size):
        batch = reference_docs[i : i + batch_size]
        try:
            await with_retries(
                vectorstore.aadd_documents,
                batch,
                max_retries=config.max_retries,
                initial_delay=config.initial_retry_delay,
                error_message=f"Failed to add document batch {i//batch_size + 1}",
                success_message=f"Added batch {i//batch_size + 1} ({len(batch)} documents)",
            )
        except RetryError:
            print(f"\n⚠️ Failed to add batch {i//batch_size + 1}. Some documents may be missing.")


async def test_retriever():
    """Test the retriever functionality."""
    return await retriever.ainvoke("What's a long context LLM anyway?")


# Generate Sections
section_writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert Wikipedia writer. Complete your assigned WikiSection from the following outline:
{outline}

Cite your sources, using the following references:
<Documents>
{docs}
</Documents>""",
        ),
        ("user", "Write the full WikiSection for the {section} section."),
    ]
)


async def retrieve(inputs: dict):
    """Retrieve relevant documents for a section."""
    print(f"\n🔍 Retrieving relevant documents for section: {inputs['section']}")
    try:
        async with asyncio.timeout(config.request_timeout):
            docs = await with_retries(
                retriever.ainvoke,
                inputs["topic"] + ": " + inputs["section"],
                max_retries=config.max_retries,
                initial_delay=config.initial_retry_delay,
                error_message="Failed to retrieve documents",
            )
            formatted = "\n".join([f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>' for doc in docs])
            print(f"✅ Found {len(docs)} relevant documents")
            return {"docs": formatted, **inputs}
    except asyncio.TimeoutError:
        print("\n⚠️ Document retrieval timed out, proceeding with empty results")
        return {"docs": "", **inputs}
    except RetryError:
        print("\n⚠️ Document retrieval failed, proceeding with empty results")
        return {"docs": "", **inputs}


section_writer = retrieve | section_writer_prompt | config.long_context_llm.with_structured_output(WikiSection)


async def write_section(outline, section_title: str, topic: str):
    """Write a single section of the article."""
    print(f"\n📝 Writing section: {section_title}")
    try:
        async with asyncio.timeout(config.request_timeout * 1.5):  # 1.5x timeout for section writing
            section = await with_retries(
                section_writer.ainvoke,
                {
                    "outline": outline.as_str,
                    "section": section_title,
                    "topic": topic,
                },
                max_retries=config.max_retries,
                initial_delay=config.initial_retry_delay,
                error_message=f"Failed to write section: {section_title}",
                success_message=f"Completed section: {section_title}",
            )
            return section
    except (asyncio.TimeoutError, RetryError):
        # Create a basic section as fallback
        return WikiSection(
            section_title=section_title,
            content=f"Content generation failed for this section (timeout/error). Please refer to the outline:\n\n{outline.as_str}",
            citations=[],
        )


async def write_sections(state):
    """Write all sections of the article."""
    print("\n📝 Writing article sections...")
    outline = state["outline"]
    topic = state["topic"]

    # Initialize progress tracker
    progress = ProgressTracker(len(outline.sections), "Section Writing")

    # Prepare inputs for all sections
    section_inputs = [
        {
            "outline": outline.as_str,
            "section": section.section_title,
            "topic": topic,
        }
        for section in outline.sections
    ]

    try:
        # Process sections in batches to avoid overwhelming the API
        sections = []
        batch_size = config.max_concurrent_requests
        for i in range(0, len(section_inputs), batch_size):
            batch = section_inputs[i : i + batch_size]
            try:
                async with asyncio.timeout(config.request_timeout * 2):  # Double timeout for batch processing
                    batch_results = await section_writer.abatch(batch)
                    sections.extend(batch_results)
                    for section in batch_results:
                        progress.step(f"Completed section: {section.section_title}")
            except asyncio.TimeoutError:
                print(f"\n⚠️ Batch {i//batch_size + 1} timed out, using fallback sections")
                # Add fallback sections for the batch
                sections.extend(
                    [
                        WikiSection(
                            section_title=input["section"],
                            content=f"Content generation timed out. Original outline:\n\n{input['outline']}",
                            citations=[],
                        )
                        for input in batch
                    ]
                )

        print(f"\n✅ Completed {len(sections)} sections")
        return {**state, "sections": sections}
    except Exception as e:
        print(f"\n⚠️ Section writing failed: {str(e)}")
        # Return basic sections as fallback
        return {
            **state,
            "sections": [
                WikiSection(
                    section_title=section.section_title,
                    content=f"Content generation failed. Original description: {section.description}",
                    citations=[],
                )
                for section in outline.sections
            ],
        }


# Generate final article
writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert Wikipedia author. Write the complete wiki article on {topic} using the following section drafts:
{draft}

Strictly follow Wikipedia format guidelines, but keep the content concise and focused.
Aim to maintain clarity while being efficient with the content length.""",
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
            """Write a clear and concise Wikipedia article on {topic}. Use the following content as your source:
{draft}

Focus on presenting the key information in a straightforward manner.
Keep the content brief but informative.""",
        ),
        (
            "user",
            "Write the article in markdown format with simple citation numbers [1] and a list of sources at the end.",
        ),
    ]
)


async def generate_with_fallback(inputs: dict) -> str:
    """Generate article with fallback to simpler format if needed."""

    async def attempt_generation(prompt, attempt_num=1):
        chain = prompt | config.long_context_llm | StrOutputParser()
        try:
            # Set timeout for the entire operation
            async with asyncio.timeout(config.request_timeout * 2):  # Double timeout for generation
                return await with_retries(
                    chain.ainvoke,
                    inputs,
                    max_retries=config.max_retries,
                    initial_delay=config.initial_retry_delay * attempt_num,  # Increase delay for subsequent attempts
                    exponential_base=config.retry_exponential_base,
                    error_message=f"Article generation failed (Attempt {attempt_num})",
                )
        except asyncio.TimeoutError:
            if attempt_num < 3:  # Max 3 major attempts
                print(f"\n⏱️ Generation timed out, retrying with longer timeout (Attempt {attempt_num + 1})...")
                await asyncio.sleep(config.initial_retry_delay * attempt_num)
                return await attempt_generation(prompt, attempt_num + 1)
            raise RetryError(Exception("Timeout"), attempt_num)
        except RetryError as e:
            if "Connection error" in str(e.original_error) and attempt_num < 3:
                print(f"\n🔄 Connection error, retrying with longer delay (Attempt {attempt_num + 1})...")
                await asyncio.sleep(config.initial_retry_delay * attempt_num * 2)
                return await attempt_generation(prompt, attempt_num + 1)
            raise

    try:
        # Try with main prompt
        return await attempt_generation(writer_prompt)
    except (RetryError, asyncio.TimeoutError):
        print("\n⚠️ Main generation failed, using simplified format...")
        try:
            # Try with simplified prompt
            return await attempt_generation(simple_writer_prompt)
        except (RetryError, asyncio.TimeoutError):
            print("\n❌ All generation attempts failed")
            # Return a basic formatted version of the draft as last resort
            return f"# {inputs['topic']}\n\n{inputs['draft']}"


writer = RunnableLambda(generate_with_fallback)


async def stream_writer(topic: str, section: WikiSection):
    """Stream the article writing process with progress updates."""
    print("\n📖 Generating final article...")
    try:
        async with asyncio.timeout(config.request_timeout * 3):  # Triple timeout for final article
            result = await with_retries(
                writer.ainvoke,
                {"topic": topic, "draft": section.as_str},
                max_retries=config.max_retries,
                initial_delay=config.initial_retry_delay,
                error_message="Article generation failed",
                success_message="Article generation complete",
            )
            print(result)
            return result
    except (RetryError, asyncio.TimeoutError):
        print("\n🔄 Retrying with simplified format...")
        # Final fallback: generate a basic article
        try:
            async with asyncio.timeout(config.request_timeout * 2):
                simple_result = await with_retries(
                    (simple_writer_prompt | config.long_context_llm | StrOutputParser()).ainvoke,
                    {"topic": topic, "draft": section.as_str},
                    max_retries=config.max_retries,
                    initial_delay=config.initial_retry_delay * 2,  # Double delay for fallback
                    error_message="Simplified article generation failed",
                    success_message="Article generation complete (simplified format)",
                )
                print(simple_result)
                return simple_result
        except (RetryError, asyncio.TimeoutError):
            # Ultimate fallback: return formatted draft
            print("\n❌ All article generation attempts failed")
            return f"# {topic}\n\n{section.as_str}"
