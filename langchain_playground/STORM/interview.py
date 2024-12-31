import asyncio
import json
import os
import time
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

from .config import fast_llm
from .models import AnswerWithCitations, Editor, Queries


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
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            # First try with the full persona-based prompt
            try:
                gen_question_chain = RunnableLambda(swap_roles).bind(name=editor.name) | gen_question_prompt.partial(persona=editor.persona) | fast_llm | RunnableLambda(tag_with_name).bind(name=editor.name)
                result = await gen_question_chain.ainvoke(state)
            except Exception as e:
                if "model produced invalid content" in str(e) or "InternalServerError" in str(e):
                    print("\n⚠️ Using simplified prompt due to model error...")
                    # Fallback to a simpler prompt without complex persona
                    simple_question_prompt = ChatPromptTemplate.from_messages(
                        [
                            (
                                "system",
                                """You are researching for a Wikipedia article. Ask a relevant question about the topic.
When you have no more questions, say "Thank you so much for your help!"
Keep questions focused and specific.""",
                            ),
                            MessagesPlaceholder(variable_name="messages", optional=True),
                        ]
                    )
                    gen_question_chain = RunnableLambda(swap_roles).bind(name=editor.name) | simple_question_prompt | fast_llm | RunnableLambda(tag_with_name).bind(name=editor.name)
                    result = await gen_question_chain.ainvoke(state)
                else:
                    raise

            print(f"❓ {editor.name} asked: {result.content[:100]}...")
            return {"messages": [result]}
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n⚠️ Question generation attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("\n❌ Failed to generate question")
                # Return a basic question as fallback
                return {"messages": [AIMessage(name=editor.name, content="Could you provide more information about this topic?")]}


async def get_initial_question(topic, editor):
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
            return await generate_question.ainvoke(
                {
                    "editor": editor,
                    "messages": messages,
                }
            )
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n⚠️ Initial question generation attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("\n❌ Failed to generate initial question")
                # Return a basic question as fallback
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
gen_queries_chain = gen_queries_prompt | fast_llm.with_structured_output(Queries, include_raw=True)


async def get_queries(question_content):
    max_retries = 3
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            return await gen_queries_chain.ainvoke({"messages": [HumanMessage(content=question_content)]})
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n⚠️ Query generation attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("\n❌ Failed to generate queries")
                # Return a basic query as fallback
                return {"parsed": Queries(queries=[f"What is {question_content}?"]), "raw": AIMessage(content=f"Searching for: {question_content}")}


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

gen_answer_chain = gen_answer_prompt | fast_llm.with_structured_output(AnswerWithCitations, include_raw=True).with_config(run_name="GenerateAnswer")

'''
# Tavily is typically a better search engine, but your free queries are limited
search_engine = TavilySearchResults(max_results=4)


@tool
async def search_engine_tool(query: str):
    """Search engine to the internet."""
    results = search_engine.invoke(query)
    return [{"content": r["content"], "url": r["url"]} for r in results]
'''


# Use Google Search API
search_engine = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID"),
)


# Track last query time for rate limiting
last_query_time = 0
RATE_LIMIT_DELAY = 2  # Delay in seconds between queries


@tool
async def search_engine_tool(query: str):
    """Search engine to the internet."""
    global last_query_time

    # Add delay if needed to respect rate limits
    current_time = time.time()
    time_since_last_query = current_time - last_query_time
    if time_since_last_query < RATE_LIMIT_DELAY:
        delay = RATE_LIMIT_DELAY - time_since_last_query
        await asyncio.sleep(delay)

    try:
        results = search_engine.results(query, num_results=4)
        last_query_time = time.time()
        return [{"content": r["snippet"], "url": r["link"]} for r in results]
    except Exception as e:
        if "429" in str(e):
            print(f"Rate limit exceeded, waiting {RATE_LIMIT_DELAY} seconds before retry...")
            await asyncio.sleep(RATE_LIMIT_DELAY)
            try:
                results = search_engine.results(query, num_results=4)
                last_query_time = time.time()
                return [{"content": r["snippet"], "url": r["link"]} for r in results]
            except Exception as retry_e:
                print(f"Search error after retry: {str(retry_e)}")
                return []
        else:
            print(f"Search error: {str(e)}")
            return []


async def gen_answer(
    state: InterviewState,
    config: Optional[dict] = None,
    name: str = "expert_bot",
    max_str_len: int = 15000,
):
    print("\n🔍 Expert is researching the answer...")
    max_retries = 3
    retry_delay = 5  # seconds

    swapped_state = swap_roles(state, name)  # Convert all other AI messages

    # Generate queries with retry
    for attempt in range(max_retries):
        try:
            queries = await gen_queries_chain.ainvoke(swapped_state)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n⚠️ Query generation attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("\n❌ Failed to generate queries")
                # Return a basic response as fallback
                return {
                    "messages": [AIMessage(name=name, content="I apologize, but I'm having trouble processing your question. Could you please rephrase it?")],
                    "references": {}
                }

    print(f"🌐 Searching web for {len(queries['parsed'].queries)} queries...")
    query_results = await search_engine_tool.abatch(queries["parsed"].queries, config, return_exceptions=True)
    successful_results = [res for res in query_results if not isinstance(res, Exception) and res]  # Filter out empty results
    all_query_results = {res["url"]: res["content"] for results in successful_results for res in results}
    print(f"✅ Found {len(all_query_results)} relevant sources")

    # Handle case when no results are found
    if not all_query_results:
        all_query_results = {"no_results": "No specific sources found. Providing a general response based on existing knowledge."}

    # We could be more precise about handling max token length if we wanted to here
    dumped = json.dumps(all_query_results)[:max_str_len]
    ai_message: AIMessage = queries["raw"]
    tool_call = queries["raw"].tool_calls[0]
    tool_id = tool_call["id"]
    tool_message = ToolMessage(tool_call_id=tool_id, content=dumped)
    swapped_state["messages"].extend([ai_message, tool_message])
    
    # Generate answer with retry
    print("\n💭 Expert is formulating response...")
    retry_delay = 5  # Reset delay for new operation
    for attempt in range(max_retries):
        try:
            # Only update the shared state with the final answer to avoid
            # polluting the dialogue history with intermediate messages
            generated = await gen_answer_chain.ainvoke(swapped_state)
            cited_urls = set(generated["parsed"].cited_urls)
            # Save the retrieved information to a the shared state for future reference
            cited_references = {k: v for k, v in all_query_results.items() if k in cited_urls}
            formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
            print(f"💬 Expert responded with {len(formatted_message.content)} characters")
            return {"messages": [formatted_message], "references": cited_references}
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n⚠️ Answer generation attempt {attempt + 1} failed: {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("\n❌ Failed to generate answer")
                # Return a basic response as fallback
                return {
                    "messages": [AIMessage(name=name, content="I apologize, but I'm having trouble formulating a detailed response. Here's what I found in my search: " + "\n\n".join(all_query_results.values()))],
                    "references": all_query_results
                }


async def get_example_answer(question_content):
    return await gen_answer({"messages": [HumanMessage(content=question_content)]})


# Construct the Interview Graph
max_num_turns = 5


def route_messages(state: InterviewState, name: str = "expert_bot"):
    messages = state["messages"]
    num_responses = len([m for m in messages if isinstance(m, AIMessage) and m.name == name])
    if num_responses >= max_num_turns:
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


async def run_interview(topic, editor):
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
