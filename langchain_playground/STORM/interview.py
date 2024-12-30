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
gen_qn_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an experienced Wikipedia writer and want to edit a specific page.
Besides your identity as a Wikipedia writer, you have a specific focus when researching the topic.
Now, you are chatting with an expert to get information. Ask good questions to get more useful information.

When you have no more questions to ask, say "Thank you so much for your help!" to end the conversation.
Please only ask one question at a time and don't ask what you have asked before.
Your questions should be related to the topic you want to write.
Be comprehensive and curious, gaining as much unique insight from the expert as possible.

Stay true to your specific perspective:

{persona}
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
    gn_chain = RunnableLambda(swap_roles).bind(name=editor.name) | gen_qn_prompt.partial(persona=editor.persona) | fast_llm | RunnableLambda(tag_with_name).bind(name=editor.name)
    result = await gn_chain.ainvoke(state)
    print(f"❓ {editor.name} asked: {result.content[:100]}...")
    return {"messages": [result]}


async def get_initial_question(topic, editor):
    messages = [HumanMessage(f"So you said you were writing an article on {topic}?")]
    return await generate_question.ainvoke(
        {
            "editor": editor,
            "messages": messages,
        }
    )


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
    return await gen_queries_chain.ainvoke({"messages": [HumanMessage(content=question_content)]})


gen_answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert who can use information effectively. You are chatting with a Wikipedia writer who wants to write a Wikipedia page on the topic you know. You have gathered the related information and will now use the information to form a response.

Make your response as informative as possible. When specific sources are available, make sure every sentence is supported by the gathered information and include citations.
If no specific sources are found (indicated by a "no_results" key), provide a general response based on your knowledge while maintaining factual accuracy. In this case, you can omit citations but should still provide valuable insights.

When sources are available, back up each claim with a citation from a reliable source, formatted as a footnote, reproducing the URLs after your response.
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
    swapped_state = swap_roles(state, name)  # Convert all other AI messages
    queries = await gen_queries_chain.ainvoke(swapped_state)
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
    print("\n💭 Expert is formulating response...")
    # Only update the shared state with the final answer to avoid
    # polluting the dialogue history with intermediate messages
    generated = await gen_answer_chain.ainvoke(swapped_state)
    cited_urls = set(generated["parsed"].cited_urls)
    # Save the retrieved information to a the shared state for future reference
    cited_references = {k: v for k, v in all_query_results.items() if k in cited_urls}
    formatted_message = AIMessage(name=name, content=generated["parsed"].as_str)
    print(f"💬 Expert responded with {len(formatted_message.content)} characters")
    return {"messages": [formatted_message], "references": cited_references}


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
