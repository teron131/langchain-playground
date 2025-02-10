import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from llm_stats import get_all_model_stats
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()


class State(TypedDict):
    input: str
    model_type: Literal["chat", "reasoning"] = None
    complexity: Literal["low", "high"] = None
    ranked_models: list[str] = None
    response: str = None


def analyzer(state: State) -> State:
    """Analyze input to determine model type and complexity."""

    class AnalyzerOutput(BaseModel):
        model_type: Literal["chat", "reasoning"] = Field(..., description="Type of model needed - 'chat' for simple Q&A or 'reasoning' for complex tasks")
        complexity: Literal["low", "high"] = Field(..., description="Complexity level of the task")

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert at analyzing questions and tasks. Analyze the input and determine:

1. Model type needed:
   - "chat" for simple Q&A, factual queries, or basic conversation
   - "reasoning" for complex problem solving, multi-step tasks, or analysis

2. Complexity level:
   - "low" for simple facts, definitions, or basic questions
   - "high" for complex reasoning, detailed analysis, creative tasks, or multi-part questions""",
            ),
            ("human", "{input}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm.with_structured_output(AnalyzerOutput)

    # Get structured response
    response = chain.invoke({"input": state["input"]})
    state["model_type"] = response.model_type
    state["complexity"] = response.complexity

    return state


def ranker(state: State) -> State:
    """Route to appropriate models based on model_type and complexity."""
    # Scoring weights based on complexity
    weights = {
        "low": {
            "price": 0.4,
            "speed": 0.4,
            "elo": 0.2,
        },
        "high": {
            "price": 0.2,
            "speed": 0.2,
            "elo": 0.6,
        },
    }

    # Get all model stats and filter by model_type
    stats = get_all_model_stats()
    filtered_stats = [model for model in stats if model["model_type"] == state["model_type"]]

    # Calculate normalized scores for each metric
    def normalize_array(arr, reverse=False):
        min_val = min(arr)
        max_val = max(arr)
        if min_val == max_val:
            return [1.0] * len(arr)
        if reverse:  # Lower is better (for price, latency)
            return [(max_val - x) / (max_val - min_val) for x in arr]
        return [(x - min_val) / (max_val - min_val) for x in arr]

    # Extract metrics
    elos = [m["elo"] for m in filtered_stats]
    latencies = [m["latency"] for m in filtered_stats]
    throughputs = [m["throughput"] for m in filtered_stats]
    prices = [(m["input_price_per_1k"] + m["output_price_per_1k"]) for m in filtered_stats]

    # Normalize scores (0-1 range)
    elo_scores = normalize_array(elos)
    latency_scores = normalize_array(latencies, reverse=True)
    throughput_scores = normalize_array(throughputs)
    price_scores = normalize_array(prices, reverse=True)

    # Calculate performance scores (combine latency and throughput)
    speed_scores = [(l + t) / 2 for l, t in zip(latency_scores, throughput_scores)]

    # Get weights for current complexity
    w = weights[state["complexity"]]

    # Calculate final scores
    model_scores = []
    for i in range(len(filtered_stats)):
        score = w["price"] * price_scores[i] + w["speed"] * speed_scores[i] + w["elo"] * elo_scores[i]
        model_scores.append(
            {
                "model": filtered_stats[i]["model"],
                "score": score,
            }
        )

    # Sort models by score in descending order
    ranked_models = sorted(model_scores, key=lambda x: x["score"], reverse=True)

    # Select best model and store ranked list
    state["selected_model"] = ranked_models[0]["model"]
    state["ranked_models"] = ranked_models

    return state


def router(state: State) -> State:
    """Try each model in ranked order until one succeeds."""
    for model in state["ranked_models"]:
        try:

            llm = ChatOpenAI(
                model=model["model"],
                api_key=os.getenv("OPENROUTER_API_KEY"),
                base_url="https://openrouter.ai/api/v1",
            )
            print(f"Model: {model['model']}")
            response = llm.invoke(state["input"])
            state["response"] = response.content
            return state
        except Exception:
            continue

    raise RuntimeError("All models failed")


builder = StateGraph(State)

builder.add_node("analyzer", analyzer)
builder.add_node("ranker", ranker)
builder.add_node("router", router)

builder.add_edge(START, "analyzer")
builder.add_edge("analyzer", "ranker")
builder.add_edge("ranker", "router")
builder.add_edge("router", END)

router_graph = builder.compile()

if __name__ == "__main__":
    result = router_graph.invoke({"input": "What is the capital of France?"})
    print(result)
