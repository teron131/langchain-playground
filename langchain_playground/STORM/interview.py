"""Interview module for the STORM pipeline."""

import asyncio
import json
from typing import Annotated, List, Optional

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import chain as as_runnable
from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper
from langgraph.graph import END, START, StateGraph
from langgraph.pregel import RetryPolicy
from typing_extensions import TypedDict

from .config import config
from .models import AnswerWithCitations, Editor, Queries
from .utils import RetryError, format_conversation, with_retries, with_fallback


# Interview State
def add_messages(left, right):
    if not isinstance(left, list):
        left = [left]
    if not isinstance(right, list):
        right = [right]
    return left + right


def update_references(references, new_references):
    if not references:
        references = {}
    references.update(new_references)
    return references


def update_editor(editor, new_editor):
    # Can only set at the outset
    if not editor:
        return new_editor
    return editor


class InterviewState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    references: Annotated[Optional[dict], update_references]
    editor: Annotated[Optional[Editor], update_editor]


# Dialog Roles
gen_question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an experienced Wikipedia editor conducting research for an article. You have a specific perspective and expertise that informs your research approach. Your goal is to gather detailed, accurate information through an interview with a subject matter expert.

Ask focused, insightful questions that:
- Build on previous responses to explore topics in depth
- Seek concrete examples and evidence
- Challenge assumptions and probe for nuance
- Draw out unique insights based on the expert's experience
- Align with your specialized perspective: {persona}

Guidelines:
- Ask one clear, specific question at a time
- Avoid repeating questions already asked
- Stay focused on information relevant to the Wikipedia article
- When you have gathered sufficient information, end with "Thank you so much for your help!"

Remember to maintain a professional, curious tone while representing your unique editorial perspective.
""",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

# Fallback prompt for when the full prompt fails
simple_question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a Wikipedia researcher gathering information for an article. Ask one clear, focused question about the topic that will help provide accurate, verifiable content. Avoid questions that have already been answered. When you have gathered sufficient information, end with 'Thank you so much for your help!' Your questions should aim to uncover specific details, examples, or evidence that would strengthen the article.
""",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)


def tag_with_name(ai_message: AIMessage, name: str):
    ai_message.name = name
    return ai_message


def swap_roles(state: InterviewState, name: str):
    converted = []
    for message in state["messages"]:
        if isinstance(message, AIMessage) and message.name != name:
            message = HumanMessage(**message.model_dump(exclude={"type"}))
        converted.append(message)
    return {"messages": converted}


@as_runnable
async def generate_question(state: InterviewState):
    editor = state["editor"]
    print(f"\n💭 {editor.name} is thinking of a question...")

    async def _generate_with_prompt(prompt):
        gen_question_chain = (
            RunnableLambda(swap_roles).bind(name=editor.name) |
            prompt |
            config.fast_llm |
            RunnableLambda(tag_with_name).bind(name=editor.name)
        )
        return await gen_question_chain.ainvoke(state)

    try:
        # First try with full persona-based prompt
        result = await with_retries(
            _generate_with_prompt,
            gen_question_prompt.partial(persona=editor.persona),
            max_retries=config.max_retries,
            initial_delay=config.initial_retry_delay,
            error_message="Failed to generate question with persona",
        )
    except RetryError:
        print("\n⚠️ Using simplified prompt due to failures...")
        try:
            # Fallback to simpler prompt
            result = await with_retries(
                _generate_with_prompt,
                simple_question_prompt,
                max_retries=config.max_retries,
                initial_delay=config.initial_retry_delay,
                error_message="Failed to generate question with simple prompt",
            )
        except RetryError:
            # Final fallback
            return {"messages": [AIMessage(name=editor.name, content="Could you provide more information about this topic?")]}

    print(f"❓ {editor.name} asked: {result.content[:100]}...")
    return {"messages": [result]}


async def get_initial_question(topic: str, editor: Editor):
    messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
    try:
        return await with_retries(
            generate_question.ainvoke,
            {"editor": editor, "messages": messages},
            max_retries=config.max_retries,
            initial_delay=config.initial_retry_delay,
            error_message="Failed to generate initial question",
        )
    except RetryError:
        return {"messages": [AIMessage(name=editor.name, content=f"Could you tell me more about {topic}?")]}


# Answer questions
gen_queries_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful research assistant. Query the search engine to answer the user's questions.",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)
gen_queries_chain = gen_queries_prompt | config.fast_llm.with_structured_output(Queries, include_raw=True)


async def get_queries(question_content: str):
    try:
        return await with_retries(
            gen_queries_chain.ainvoke,
            {"messages": [HumanMessage(content=question_content)]},
            max_retries=config.max_retries,
            initial_delay=config.initial_retry_delay,
            error_message="Failed to generate search queries",
        )
    except RetryError:
        return {
            "parsed": Queries(queries=[f"What is {question_content}?"]),
            "raw": AIMessage(content=f"Searching for: {question_content}")
        }


gen_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert researcher and communicator engaging with a Wikipedia writer. Your role is to provide comprehensive, well-supported responses using gathered information to help create an accurate and authoritative Wikipedia article.

When provided with source materials, analyze them carefully and synthesize the information into clear, factual statements. Every claim should be supported by relevant citations using footnote format (e.g. [1]). Focus on extracting key facts, statistics, expert opinions, and important context from the sources.

If no sources are found (indicated by "no_results"), draw upon your broad knowledge to provide a balanced, factual response while maintaining academic rigor. Even without citations, ensure your response is informative, nuanced, and aligned with Wikipedia's neutral point of view.

Present information in a clear, organized manner that the writer can easily incorporate into the article. After your response, list all cited URLs in a numbered reference format that matches your footnotes.
""",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

gen_answer_chain = (
    gen_answer_prompt |
    config.fast_llm.with_structured_output(AnswerWithCitations, include_raw=True)
).with_config(run_name="GenerateAnswer")


# Initialize search engine
search_engine = GoogleSearchAPIWrapper(
    google_api_key=config.google_api_key,
    google_cse_id=config.google_cse_id,
)


@tool
async def search_engine_tool(query: str):
    """Search engine to the internet."""
    async def _search():
        results = search_engine.results(query, num_results=config.max_search_results)
        return [{"content": r["snippet"], "url": r["link"]} for r in results]
    
    try:
        return await with_retries(
            _search,
            max_retries=config.max_retries,
            initial_delay=config.search_rate_limit_delay,
            error_message=f"Search failed for query: {query}",
            success_message=f"Search completed for: {query}"
        )
    except RetryError:
        return []  # Return empty results after all retries fail


async def gen_answer(
    state: InterviewState,
    config_dict: Optional[dict] = None,
    name: str = "expert_bot",
    max_str_len: int = 15000,
):
    print("\n🔍 Expert is researching the answer...")
    swapped_state = swap_roles(state, name)

    try:
        # Generate queries
        queries = await with_retries(
            gen_queries_chain.ainvoke,
            swapped_state,
            max_retries=config.max_retries,
            initial_delay=config.initial_retry_delay,
            error_message="Failed to generate queries",
        )
    except RetryError:
        return {
            "messages": [AIMessage(name=name, content="I apologize, but I'm having trouble processing your question. Could you please rephrase it?")],
            "references": {}
        }

    # Search for answers
    print(f"🌐 Searching web for {len(queries['parsed'].queries)} queries...")
    query_results = await search_engine_tool.abatch(queries["parsed"].queries, config_dict, return_exceptions=True)
    successful_results = [res for res in query_results if not isinstance(res, Exception) and res]
    all_query_results = {res["url"]: res["content"] for results in successful_results for res in results}
    print(f"✅ Found {len(all_query_results)} relevant sources")

    if not all_query_results:
        all_query_results = {"no_results": "No specific sources found. Providing a general response based on existing knowledge."}

    # Prepare state for answer generation
    dumped = json.dumps(all_query_results)[:max_str_len]
    ai_message: AIMessage = queries["raw"]
    tool_call = queries["raw"].tool_calls[0]
    tool_id = tool_call["id"]
    tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
    swapped_state["messages"].extend([ai_message, tool_message])

    try:
        # Generate answer
        print("\n💭 Expert is formulating response...")
        generated = await with_retries(
            gen_answer_chain.ainvoke,
            swapped_state,
            max_retries=config.max_retries,
            initial_delay=config.initial_retry_delay,
            error_message="Failed to generate answer",
        )
        
        cited_urls = set(generated["parsed"].cited_urls)
        cited_references = {k: v for k, v in all_query_results.items() if k in cited_urls}
        formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
        print(f"💬 Expert responded with {len(formatted_message.content)} characters")
        return {"messages": [formatted_message], "references": cited_references}
    except RetryError:
        return {
            "messages": [AIMessage(name=name, content="I apologize, but I'm having trouble formulating a detailed response. Here's what I found in my search: " + "\n\n".join(all_query_results.values()))],
            "references": all_query_results
        }


async def get_example_answer(question_content: str):
    return await gen_answer({"messages": [HumanMessage(content=question_content)]})


# Construct the Interview Graph
def route_messages(state: InterviewState, name: str = "expert_bot"):
    messages = state["messages"]
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])
    if num_responses >= config.max_interview_turns:
        print("\n⏰ Maximum turns reached, ending conversation")
        return END
    last_question = messages[-2]
    if last_question.content.endswith("Thank you so much for your help!"):
        print("\n👋 Editor thanked expert, ending conversation")
        return END
    return "ask_question"


builder = StateGraph(InterviewState)

builder.add_node("ask_question", generate_question, retry=RetryPolicy(max_attempts=5))
builder.add_node("answer_question", gen_answer, retry=RetryPolicy(max_attempts=5))
builder.add_conditional_edges("answer_question", route_messages)
builder.add_edge("ask_question", "answer_question")

builder.add_edge(START, "ask_question")
interview_graph = builder.compile(checkpointer=False).with_config(run_name="Conduct Interviews")


async def run_interview(topic: str, editor: Editor):
    print(f"\n🎭 Starting interview with editor {editor.name}...")
    final_step = None
    initial_state = {
        "editor": editor,
        "messages": [
            AIMessage(
                content=f"So you said you were writing an article on {topic}?",
                name="expert_bot",
            )
        ],
    }
    async for step in interview_graph.astream(initial_state):
        name = next(iter(step))
        print(f"📍 Current step: {name}")
        final_step = step

    print(f"✅ Completed interview with {editor.name}")
    return next(iter(final_step.values()))
