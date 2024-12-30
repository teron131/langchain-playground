from langchain_community.retrievers.wikipedia import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain as as_runnable

from .config import fast_llm, long_context_llm
from .models import Outline, Perspectives, RelatedSubjects

# Generate Initial Outline
direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Wikipedia writer. Write an outline for a Wikipedia page about a user-provided topic. Be comprehensive and specific.",
        ),
        ("user", "{topic}"),
    ]
)

generate_outline_direct = direct_gen_outline_prompt | fast_llm.with_structured_output(Outline)


async def get_initial_outline(topic: str):
    print("\n🔍 Generating initial outline...")
    outline = await generate_outline_direct.ainvoke({"topic": topic})
    print("✅ Initial outline generated")
    return outline


# Expand Topics
gen_related_topics_prompt = ChatPromptTemplate.from_template(
    """
I'm writing a Wikipedia page for a topic mentioned below. Please identify and recommend some Wikipedia pages on closely related subjects. I'm looking for examples that provide insights into interesting aspects commonly associated with this topic, or examples that help me understand the typical content and structure included in Wikipedia pages for similar topics.

Please list the as many subjects and urls as you can.

Topic of interest: {topic}
"""
)

expand_chain = gen_related_topics_prompt | fast_llm.with_structured_output(RelatedSubjects)


async def get_related_subjects(topic):
    print("\n🔍 Finding related topics...")
    subjects = await expand_chain.ainvoke({"topic": topic})
    print(f"✅ Found {len(subjects.topics)} related topics")
    return subjects


# Generate Perspectives
gen_perspectives_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You need to select a diverse (and distinct) group of Wikipedia editors who will work together to create a comprehensive article on the topic. Each of them represents a different perspective, role, or affiliation related to this topic.
You can use other Wikipedia pages of related topics for inspiration. For each editor, add a description of what they will focus on.

IMPORTANT: Editor names must:
- Only contain letters, numbers, underscores, and hyphens (no spaces or periods)
- Use underscores or hyphens instead of spaces (e.g., 'jonathan_ross' or 'jensen-huang')
- Be between 1 and 64 characters long

Wiki page outlines of related topics for inspiration:
{examples}
""",
        ),
        ("user", "Topic of interest: {topic}"),
    ]
)

gen_perspectives_chain = gen_perspectives_prompt | fast_llm.with_structured_output(Perspectives)

# Wikipedia retrieval and formatting
wikipedia_retriever = WikipediaRetriever(load_all_available_meta=True, top_k_results=1)


def format_doc(doc, max_length=1000):
    related = "- ".join(doc.metadata["categories"])
    return f"### {doc.metadata['title']}\n\nSummary: {doc.page_content}\n\nRelated\n{related}"[:max_length]


def format_docs(docs):
    return "\n\n".join(format_doc(doc) for doc in docs)


@as_runnable
async def survey_subjects(topic: str):
    print("\n🔍 Surveying Wikipedia for related content...")
    related_subjects = await expand_chain.ainvoke({"topic": topic})
    print(f"📚 Retrieving {len(related_subjects.topics)} Wikipedia articles...")
    retrieved_docs = await wikipedia_retriever.abatch(related_subjects.topics, return_exceptions=True)
    all_docs = []
    for docs in retrieved_docs:
        if isinstance(docs, BaseException):
            continue
        all_docs.extend(docs)
    print(f"✅ Retrieved {len(all_docs)} articles successfully")
    formatted = format_docs(all_docs)
    print("\n🤔 Generating editor perspectives...")
    perspectives = await gen_perspectives_chain.ainvoke({"examples": formatted, "topic": topic})
    print(f"✅ Generated {len(perspectives.editors)} editor perspectives")
    return perspectives


# Refine Outline
refine_outline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Wikipedia writer. You have gathered information from experts and search engines. Now, you are refining the outline of the Wikipedia page.
You need to make sure that the outline is comprehensive and specific.
Topic you are writing about: {topic} 

Old outline:

{old_outline}
""",
        ),
        (
            "user",
            "Refine the outline based on your conversations with subject-matter experts:\n\nConversations:\n\n{conversations}\n\nWrite the refined Wikipedia outline:",
        ),
    ]
)

refine_outline_chain = refine_outline_prompt | long_context_llm.with_structured_output(Outline)


async def get_refined_outline(topic, initial_outline, final_state):
    print("\n📝 Refining outline based on expert interviews...")
    refined = await refine_outline_chain.ainvoke(
        {
            "topic": topic,
            "old_outline": initial_outline.as_str,
            "conversations": "\n\n".join(f"### {m.name}\n\n{m.content}" for m in final_state["messages"]),
        }
    )
    print("✅ Outline refinement complete")
    return refined
