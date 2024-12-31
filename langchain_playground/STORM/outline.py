"""Outline generation module for the STORM pipeline."""

from langchain_community.retrievers.wikipedia import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain as as_runnable

from .config import config
from .models import Outline, Perspectives, RelatedSubjects
from .utils import RetryError, with_retries

# Generate Initial Outline
direct_gen_outline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert Wikipedia writer. Create a detailed outline for a Wikipedia article about the given topic. Your outline should:
- Follow Wikipedia's standard article structure
- Include all major aspects of the topic
- Break down complex subjects into clear subsections
- Ensure logical flow and progression of information
- Cover historical context, key developments, and current significance
- Include sections for real-world applications or impact where relevant
- Add sections for criticism or controversies if applicable

Focus on creating a comprehensive yet well-organized structure that will guide the development of an authoritative article.
""",
        ),
        ("user", "{topic}"),
    ]
)

generate_outline_direct = direct_gen_outline_prompt | config.fast_llm.with_structured_output(Outline)


async def get_initial_outline(topic: str):
    """Generate initial outline for the article."""
    print("\n🔍 Generating initial outline...")
    try:
        outline = await with_retries(generate_outline_direct.ainvoke, {"topic": topic}, max_retries=config.max_retries, initial_delay=config.initial_retry_delay, error_message="Failed to generate initial outline", success_message="Initial outline generated")
        return outline
    except RetryError:
        # Return a basic outline as fallback
        return Outline(
            page_title=topic,
            sections=[
                {
                    "section_title": "Introduction",
                    "description": "Overview of the topic",
                },
                {
                    "section_title": "Background",
                    "description": "Historical context and development",
                },
                {
                    "section_title": "Key Concepts",
                    "description": "Main ideas and principles",
                },
                {
                    "section_title": "Applications",
                    "description": "Real-world uses and implementations",
                },
                {
                    "section_title": "See also",
                    "description": "Related topics and further reading",
                },
            ],
        )


# Expand Topics
gen_related_topics_prompt = ChatPromptTemplate.from_template(
    """
You are an expert Wikipedia researcher. Identify and recommend Wikipedia pages that are closely related to the given topic. Focus on:
- Core concepts and foundational topics that provide essential context
- Notable examples and applications that demonstrate real-world relevance
- Related fields or domains that intersect with the main topic
- Historical developments or predecessor topics that shaped its evolution
- Contemporary trends or emerging areas connected to the topic

For each recommended page, provide both the subject and its Wikipedia URL. Aim to be comprehensive in your recommendations while ensuring each suggestion has a clear, meaningful connection to the main topic.

Topic: {topic}
"""
)

expand_chain = gen_related_topics_prompt | config.fast_llm.with_structured_output(RelatedSubjects)


async def get_related_subjects(topic: str):
    """Get related subjects for research."""
    print("\n🔍 Finding related topics...")
    try:
        subjects = await with_retries(expand_chain.ainvoke, {"topic": topic}, max_retries=config.max_retries, initial_delay=config.initial_retry_delay, error_message="Failed to find related topics", success_message=lambda result: f"Found {len(result.topics)} related topics")
        return subjects
    except RetryError:
        # Return minimal related subjects as fallback
        return RelatedSubjects(topics=[topic])


# Generate Perspectives
gen_perspectives_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert Wikipedia editor assembling a diverse team of editors to create a comprehensive article. Your task is to select 3-5 editors with distinct perspectives, expertise, and backgrounds related to the topic.

Each editor should represent a unique viewpoint such as:
- Academic/theoretical perspective
- Industry/practical experience
- Historical/evolutionary context
- Social/cultural impact
- Technical/implementation details
- Critical/analytical stance

For each editor, provide:
1. A unique identifier following these rules:
   - Use only letters, numbers, underscores, and hyphens
   - Replace spaces with underscores or hyphens (e.g., 'jonathan_ross', 'jensen-huang') 
   - Keep length between 1-64 characters
2. Their specific expertise and background
3. What aspects of the topic they will focus on investigating

Use the following Wikipedia articles as inspiration for different perspectives and areas to cover:
{examples}

Ensure the editors' combined expertise will result in balanced, comprehensive coverage of the topic.
""",
        ),
        ("user", "Topic of interest: {topic}"),
    ]
)

gen_perspectives_chain = gen_perspectives_prompt | config.fast_llm.with_structured_output(Perspectives)

# Wikipedia retrieval and formatting
wikipedia_retriever = WikipediaRetriever(load_all_available_meta=True, top_k_results=1)


def format_doc(doc, max_length=1000):
    """Format a Wikipedia document for use in prompts."""
    related = "- ".join(doc.metadata["categories"])
    return f"### {doc.metadata['title']}\n\nSummary: {doc.page_content}\n\nRelated\n{related}"[:max_length]


def format_docs(docs):
    """Format multiple Wikipedia documents."""
    return "\n\n".join(format_doc(doc) for doc in docs)


@as_runnable
async def survey_subjects(topic: str):
    """Survey Wikipedia for related content and generate editor perspectives."""
    print("\n🔍 Surveying Wikipedia for related content...")
    try:
        related_subjects = await get_related_subjects(topic)
        print(f"📚 Retrieving {len(related_subjects.topics)} Wikipedia articles...")

        retrieved_docs = await with_retries(
            wikipedia_retriever.abatch,
            related_subjects.topics,
            return_exceptions=True,
            max_retries=config.max_retries,
            initial_delay=config.initial_retry_delay,
            error_message="Failed to retrieve Wikipedia articles",
        )

        all_docs = []
        for docs in retrieved_docs:
            if not isinstance(docs, BaseException):
                all_docs.extend(docs)

        print(f"✅ Retrieved {len(all_docs)} articles successfully")
        formatted = format_docs(all_docs)

        print("\n🤔 Generating editor perspectives...")
        perspectives = await with_retries(gen_perspectives_chain.ainvoke, {"examples": formatted, "topic": topic}, max_retries=config.max_retries, initial_delay=config.initial_retry_delay, error_message="Failed to generate editor perspectives", success_message=lambda result: f"Generated {len(result.editors)} editor perspectives")
        return perspectives
    except RetryError:
        # Return minimal perspectives as fallback
        return Perspectives(editors=[{"name": "general_editor", "affiliation": "Wikipedia", "role": "General Editor", "description": "Focuses on creating a balanced, comprehensive article."}])


# Refine Outline
refine_outline_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Wikipedia editor refining an article outline based on expert research and interviews. Your goal is to create a comprehensive, well-structured outline that will guide the article writing process.

Consider these key aspects:
- Ensure all major topics and subtopics from the original outline are preserved or improved
- Add new sections based on expert insights and research findings
- Organize sections in a logical flow from general to specific
- Include sections for background, technical details, applications, and impact
- Maintain Wikipedia's neutral point of view and emphasis on verifiable information

Topic: {topic}

Original outline for reference:
{old_outline}
""",
        ),
        (
            "user",
            """
Review the expert conversations below and refine the outline to incorporate their insights:

Conversations:
{conversations}

Create a refined outline that:
- Integrates key points and examples from the expert discussions
- Maintains a balanced coverage of different perspectives
- Ensures technical accuracy while remaining accessible
- Follows Wikipedia's structure and style guidelines

Write the refined outline now:
""",
        ),
    ]
)

refine_outline_chain = refine_outline_prompt | config.long_context_llm.with_structured_output(Outline)


async def get_refined_outline(topic: str, initial_outline: Outline, final_state: dict):
    """Refine the outline based on expert interviews."""
    print("\n📝 Refining outline based on expert interviews...")
    try:
        refined = await with_retries(
            refine_outline_chain.ainvoke,
            {
                "topic": topic,
                "old_outline": initial_outline.as_str,
                "conversations": "\n\n".join(f"### {m.name}\n\n{m.content}" for m in final_state["messages"]),
            },
            max_retries=config.max_retries,
            initial_delay=config.initial_retry_delay,
            error_message="Failed to refine outline",
            success_message="Outline refinement complete",
        )
        return refined
    except RetryError:
        print("\n⚠️ Outline refinement failed, using initial outline")
        return initial_outline
