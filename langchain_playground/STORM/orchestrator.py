"""Orchestrator module for the STORM pipeline."""

import asyncio
from typing import List, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .config import config
from .interview import interview_graph
from .models import Editor, Outline, WikiSection
from .outline import get_initial_outline, refine_outline_chain, survey_subjects
from .utils import ProgressTracker, RetryError, with_retries
from .writer import initialize_vectorstore, section_writer, writer


class ResearchState(TypedDict):
    """State for the research pipeline."""
    topic: str
    outline: Outline
    editors: List[Editor]
    interview_results: List[dict]  # List of InterviewState
    sections: List[WikiSection]
    article: str


async def initialize_research(state: ResearchState):
    """Initialize the research process by generating outline and surveying subjects."""
    topic = state["topic"]
    print("\n🚀 Starting research process for topic:", topic)
    
    try:
        # Run initial research tasks in parallel
        outline, editors = await asyncio.gather(
            get_initial_outline(topic),
            survey_subjects.ainvoke(topic)
        )
        print("\n✅ Initial research phase complete")
        return {
            **state,
            "outline": outline,
            "editors": editors.editors,
        }
    except Exception as e:
        print(f"\n⚠️ Research initialization failed: {str(e)}")
        # Return basic state with fallback values
        return {
            **state,
            "outline": Outline(
                page_title=topic,
                sections=[{"section_title": "Overview", "description": "General overview of the topic"}]
            ),
            "editors": [Editor(
                name="general_editor",
                affiliation="Wikipedia",
                role="General Editor",
                description="Focuses on creating a balanced, comprehensive article."
            )],
        }


async def conduct_interviews(state: ResearchState):
    """Conduct interviews with all editors."""
    topic = state["topic"]
    editors = state["editors"]
    print(f"\n👥 Starting interviews with {len(editors)} editors...")
    
    # Initialize progress tracker
    progress = ProgressTracker(len(editors), "Interviews")
    
    # Prepare initial states for all interviews
    initial_states = [
        {
            "editor": editor,
            "messages": [
                AIMessage(
                    content=f"So you said you were writing an article on {topic}?",
                    name="expert_bot",
                )
            ],
        }
        for editor in editors
    ]
    
    # Run interviews in parallel
    try:
        interview_results = await interview_graph.abatch(initial_states)
        for i, result in enumerate(interview_results):
            progress.step(f"Completed interview with {editors[i].name}")
        print("\n✅ All interviews completed")
        return {**state, "interview_results": interview_results}
    except Exception as e:
        print(f"\n⚠️ Interview process failed: {str(e)}")
        # Return state with empty interview results
        return {**state, "interview_results": []}


def format_conversation(interview_state: dict) -> str:
    """Format an interview conversation for display."""
    messages = []
    for m in interview_state["messages"]:
        if isinstance(m, (AIMessage, HumanMessage)):
            messages.append(m)
        else:
            # Convert dict to appropriate Message object
            if m["type"] == "ai":
                messages.append(AIMessage(content=m["content"], name=m["name"]))
            elif m["type"] == "human":
                messages.append(HumanMessage(content=m["content"]))

    convo = "\n".join(f"{m.name if hasattr(m, 'name') else 'User'}: {m.content}" for m in messages)
    return f'Conversation with {interview_state["editor"].name}\n\n' + convo


async def refine_outline(state: ResearchState):
    """Refine the outline based on interview results."""
    convos = "\n\n".join([format_conversation(interview_state) for interview_state in state["interview_results"]])
    
    try:
        updated_outline = await with_retries(
            refine_outline_chain.ainvoke,
            {
                "topic": state["topic"],
                "old_outline": state["outline"].as_str,
                "conversations": convos,
            },
            max_retries=config.max_retries,
            initial_delay=config.initial_retry_delay,
            error_message="Failed to refine outline",
            success_message="Outline refinement complete"
        )
        return {**state, "outline": updated_outline}
    except RetryError:
        print("\n⚠️ Outline refinement failed, keeping original outline")
        return state


async def index_references(state: ResearchState):
    """Index all references from interviews."""
    print("\n📚 Indexing references from all interviews...")
    all_docs = {}
    for interview_state in state["interview_results"]:
        all_docs.update(interview_state.get("references", {}))
    
    if all_docs:
        await initialize_vectorstore(all_docs)
        print(f"✅ Indexed {len(all_docs)} total references")
    else:
        print("\n⚠️ No references found to index")
    
    return state


async def write_sections(state: ResearchState):
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
        # Write sections in parallel
        sections = await section_writer.abatch(section_inputs)
        for i, section in enumerate(sections):
            progress.step(f"Completed section: {section.section_title}")
        
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
                    citations=[]
                )
                for section in outline.sections
            ]
        }


async def write_article(state: ResearchState):
    """Generate the final article."""
    print("\n📖 Generating final article...")
    topic = state["topic"]
    sections = state["sections"]
    draft = "\n\n".join([section.as_str for section in sections])
    
    try:
        article = await with_retries(
            writer.ainvoke,
            {"topic": topic, "draft": draft},
            max_retries=config.max_retries,
            initial_delay=config.initial_retry_delay,
            error_message="Failed to generate final article",
            success_message="Article generation complete"
        )
        return {**state, "article": article}
    except RetryError:
        # Return formatted draft as fallback
        return {**state, "article": f"# {topic}\n\n{draft}"}


# Create the graph
builder = StateGraph(ResearchState)

nodes = [
    ("init_research", initialize_research),
    ("conduct_interviews", conduct_interviews),
    ("refine_outline", refine_outline),
    ("index_references", index_references),
    ("write_sections", write_sections),
    ("write_article", write_article),
]

for i in range(len(nodes)):
    name, node = nodes[i]
    builder.add_node(name, node)
    if i > 0:
        builder.add_edge(nodes[i - 1][0], name)

builder.add_edge(START, nodes[0][0])
builder.add_edge(nodes[-1][0], END)

storm = builder.compile(checkpointer=MemorySaver())


def generate_article(topic: str) -> str:
    """Generate a complete article on the given topic."""
    print("\n🌟 Starting STORM pipeline for topic:", topic)
    config_dict = {"configurable": {"thread_id": "my-thread"}}

    async def _generate():
        async for _ in storm.astream({"topic": topic}, config_dict):
            pass
        checkpoint = storm.get_state(config_dict)
        return checkpoint.values["article"]

    article = asyncio.run(_generate())
    print("\n🎉 STORM pipeline complete!")
    return str(article)
