from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from .config import long_context_llm
from .models import WikiSection

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
    section = await section_writer.ainvoke(
        {
            "outline": outline.as_str,
            "section": section_title,
            "topic": topic,
        }
    )
    print(f"✅ Completed section ({len(section.content)} chars)")
    return section


# Generate final article
writer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert Wikipedia author. Write the complete wiki article on {topic} using the following section drafts:
{draft}

Strictly follow Wikipedia format guidelines.
""",
        ),
        (
            "user",
            'Write the complete Wiki article using markdown format. Organize citations using footnotes like "[1]",' " avoiding duplicates in the footer. Include URLs in the footer.",
        ),
    ]
)

writer = writer_prompt | long_context_llm | StrOutputParser()


async def stream_writer(topic, section):
    print("\n📖 Generating final article...")
    async for tok in writer.astream({"topic": topic, "draft": section.as_str}):
        print(tok, end="")
    print("\n✅ Article generation complete")
