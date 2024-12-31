from langchain_community.retrievers.wikipedia import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain as as_runnable

from .config import fast_llm, long_context_llm
from .models import Outline, Perspectives, RelatedSubjects

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

generate_outline_direct = direct_gen_outline_prompt | fast_llm.with_structured_output(Outline)


async def get_initial_outline(topic: str):
    print("\n🔍 Generating initial outline...")
    outline = await generate_outline_direct.ainvoke({"topic": topic})
    print("✅ Initial outline generated")
    return outline


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
