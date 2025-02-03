import os
from typing import Literal

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client, evaluate
from pydantic import BaseModel, Field

from questions import questions

load_dotenv()

client = Client()


try:
    # Create or get existing dataset
    dataset = client.create_dataset(
        dataset_name="model_comparison",
        description="Questions for comparing model responses",
    )
    # Add questions to dataset
    for question in questions:
        client.create_example(
            inputs={"question": question},
            dataset_id=dataset.id,
        )
except Exception:
    pass


# Evaluators
def ranked_preference(inputs: dict, outputs: list[dict]) -> list:
    """Evaluate and compare two AI responses to determine which is preferred.
    Returns:
        List of scores (1 for preferred response, 0 for non-preferred)
    """

    class PreferenceResult(BaseModel):
        """Result of the preference evaluation between two AI responses"""

        preferred_assistant: Literal["A", "B", "Tie"] = Field(..., description="Which assistant provided the better response - A, B, or Tie if equal")
        explanation: str = Field(..., description="Detailed explanation of the reasoning behind the preference, analyzing the quality, accuracy, and effectiveness of the responses")

    # See the prompt: https://smith.langchain.com/hub/langchain-ai/pairwise-evaluation-2
    # prompt = hub.pull("langchain-ai/pairwise-evaluation-2")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a harsh but fair judge evaluating responses from two AI assistants. Your task is to critically analyze their answers and identify flaws, inaccuracies, and shortcomings. Consider factors like correctness, relevance, precision, completeness, and practical usefulness. You must point out any errors, logical fallacies, or missing key information. While both responses may have merits, focus on finding meaningful differences in quality. Do not give credit for superficial elements like length or style. Be ruthlessly objective and do not hesitate to declare neither response satisfactory if they fail to properly address the question. Your evaluation must be independent and unbiased by assistant names or response order. Provide a detailed critical analysis explaining your decision."),
            ("human", "[User Question] {question}\n[The Start of Assistant A's Answer] {answer_a} [The End of Assistant A's Answer]\nThe Start of Assistant B's Answer] {answer_b} [The End of Assistant B's Answer]"),
        ]
    )

    model = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | model.with_structured_output(PreferenceResult)
    response = chain.invoke(
        {
            "question": inputs["question"],
            "answer_a": outputs[0].get("answer", "N/A"),
            "answer_b": outputs[1].get("answer", "N/A"),
        }
    )

    if response.preferred_assistant == "A":
        scores = [1, 0]
    elif response.preferred_assistant == "B":
        scores = [0, 1]
    else:  # Tie
        scores = [0.5, 0.5]
    return scores


def task_fulfillment_evaluator_pairwise(inputs: dict, outputs: list[dict]) -> list:
    """Evaluate how well the response fulfills the user's task instructions."""

    class TaskExistence(BaseModel):
        task_exists: bool = Field(..., description="Whether the question includes explicit task instructions that the answer should fulfill")

    condition_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Determine if the following question includes explicit task instructions that the answer should fulfill. Answer 'yes' if it does, otherwise answer 'no'."),
            ("human", "{question}"),
        ]
    )
    condition_model = ChatOpenAI(model="gpt-4o-mini")
    condition_chain = condition_prompt | condition_model.with_structured_output(TaskExistence)
    condition_response = condition_chain.invoke({"question": inputs["question"]})
    if condition_response.task_exists:
        return [1, 1]

    def task_fulfillment_evaluator(question: str, answer: str) -> int:
        """Evaluate if the response fulfills the task instructions adequately."""

        class TaskFulfillmentResult(BaseModel):
            rating: int = Field(..., description="How well the response fulfills the task instructions, from 0 to 10", ge=0, le=10)
            explanation: str = Field(..., description="Explanation of the evaluation of task fulfillment")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an expert evaluator assessing how effectively an AI response fulfills the user's task instructions. Rate the response on a scale of 0-10 based on the following criteria:

Completeness (0-4 points):
- 0: Completely fails to address any requirements
- 1: Addresses only a small portion of requirements
- 2: Addresses about half of requirements
- 3: Addresses most but not all requirements
- 4: Fully addresses all requirements with no omissions

Directness (0-3 points): 
- 0: Completely off-topic or irrelevant
- 1: Mostly indirect with excessive tangents
- 2: Somewhat direct but with unnecessary content
- 3: Perfectly direct and focused

Clarity (0-3 points):
- 0: Incomprehensible or nonsensical
- 1: Major clarity issues that impede understanding
- 2: Minor clarity issues but generally understandable
- 3: Crystal clear with no ambiguity

Your final score should be the sum of these criteria. Be extremely critical and only award full points when the response is truly exceptional. Most responses should score in the lower half of the range.
""",
                ),
                ("human", "[Question:] {question}\n[Answer:] {answer}"),
            ]
        )

        model = ChatOpenAI(model="gpt-4o-mini")
        chain = prompt | model.with_structured_output(TaskFulfillmentResult)
        response = chain.invoke({"question": question, "answer": answer})
        return response.rating

    rating_a = task_fulfillment_evaluator(inputs["question"], outputs[0].get("answer", "N/A"))
    rating_b = task_fulfillment_evaluator(inputs["question"], outputs[1].get("answer", "N/A"))
    return [rating_a / 10, rating_b / 10]


def valid_reasoning_evaluator_pairwise(inputs: dict, outputs: list[dict]) -> list:
    """Evaluate the validity of the reasoning."""

    def valid_reasoning_evaluator(question: str, answer: str) -> int:
        """Evaluate if the reasoning is valid."""

        class ValidReasoningResult(BaseModel):
            """Result of the preference evaluation between two AI responses"""

            rating: int = Field(..., description="How well the reasoning is valid, from 0 to 10", ge=0, le=10)
            explanation: str = Field(..., description="Explanation of the reasoning behind the rating")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an expert evaluator assessing the quality of reasoning in AI responses. Rate the response on a scale of 0-10 based on the following strict criteria:

Logical Coherence (0-3 points):
- 0: Incoherent, riddled with contradictions and fallacies
- 1: Major logical flaws and disconnected arguments
- 2: Generally logical but with noticeable inconsistencies
- 3: Impeccable logic with clear cause-effect relationships

Depth and Completeness (0-4 points):
- 0: Extremely shallow, misses most key concepts
- 1: Barely scratches the surface, major omissions
- 2: Moderate depth but significant gaps remain
- 3: Good depth but lacks advanced connections
- 4: Exceptional depth with sophisticated insights

Handling of Nuance (0-3 points):
- 0: Completely black and white thinking
- 1: Minimal recognition of complexity
- 2: Addresses some nuances but misses critical ones
- 3: Masterful handling of subtleties and edge cases

Your final score should be the sum of the three criteria. Be extremely critical and focus on:
- Identifying ANY logical fallacies or inconsistencies
- Penalizing superficial explanations that lack rigor
- Demanding sophisticated treatment of edge cases
- Requiring concrete examples and evidence

Most responses should score in the lower half of the range. Reserve high scores for truly exceptional answers only.
""",
                ),
                ("human", "[Question:] {question}\n[Answer:] {answer}"),
            ]
        )

        model = ChatOpenAI(model="gpt-4o-mini")
        chain = prompt | model.with_structured_output(ValidReasoningResult)
        response = chain.invoke(
            {
                "question": question,
                "answer": answer,
            }
        )
        return response.rating

    rating_a = valid_reasoning_evaluator(inputs["question"], outputs[0].get("answer", "N/A"))
    rating_b = valid_reasoning_evaluator(inputs["question"], outputs[1].get("answer", "N/A"))

    return [rating_a / 10, rating_b / 10]


def style_evaluator_pairwise(inputs: dict, outputs: list[dict]) -> list:
    """Evaluate the response for formatting, readability, style matching, originality, and redundancy."""

    def style_evaluator(question: str, answer: str) -> int:
        """Evaluate the presentation aspects of the response."""

        class StyleReadabilityResult(BaseModel):
            rating: int = Field(..., description="Rating for formatting, readability, style matching, originality, and non-redundancy on a scale of 0 to 10", ge=0, le=10)
            explanation: str = Field(..., description="Explanation of the evaluation regarding presentation aspects")

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are an expert evaluator assessing the overall presentation of an AI response. Evaluate the answer on a scale of 0-10 based on the following criteria:

Formatting & Readability (0-4 points):
- 0: Unreadable mess with no structure or organization
- 1: Major formatting issues that significantly impair readability
- 2: Basic formatting but with distracting issues
- 3: Generally good formatting with minor flaws
- 4: Impeccable formatting and organization

Style Matching (0-2 points):
- 0: Completely ignores or contradicts requested style/tone
- 1: Inconsistent style with frequent lapses
- 2: Perfect adherence to requested style/tone

Originality vs Redundancy (0-4 points):
- 0: Mindless repetition and copied content
- 1: Heavy redundancy with minimal original thought
- 2: Mix of unoriginal and original content
- 3: Mostly original with slight repetition
- 4: Entirely original and concise

Be extremely critical. Most responses should score in the bottom half of each range. Reserve top scores for truly exceptional cases only.
""",
                ),
                ("human", "[Question:] {question}\n[Answer:] {answer}"),
            ]
        )

        model = ChatOpenAI(model="gpt-4o-mini")
        chain = prompt | model.with_structured_output(StyleReadabilityResult)
        response = chain.invoke({"question": question, "answer": answer})
        return response.rating

    rating_a = style_evaluator(inputs["question"], outputs[0].get("answer", "N/A"))
    rating_b = style_evaluator(inputs["question"], outputs[1].get("answer", "N/A"))
    return [rating_a / 10, rating_b / 10]


def weighted_average(inputs: dict, outputs: list[dict]) -> list:
    """Calculate the weighted average of the scores from other evaluators."""
    # Get scores from each evaluator
    preference_scores = ranked_preference(inputs, outputs)
    task_scores = task_fulfillment_evaluator_pairwise(inputs, outputs)
    reasoning_scores = valid_reasoning_evaluator_pairwise(inputs, outputs)
    style_scores = style_evaluator_pairwise(inputs, outputs)

    # Calculate weighted average for each model
    weights = [0.4, 0.3, 0.2, 0.1]  # Preference, Task, Reasoning, Style weights
    scores = [preference_scores, task_scores, reasoning_scores, style_scores]
    model_a_score = sum(score[0] * weight for score, weight in zip(scores, weights))
    model_b_score = sum(score[1] * weight for score, weight in zip(scores, weights))

    return [model_a_score, model_b_score]


# Define model response generators
llm1 = ChatOpenAI(model="gpt-4o-mini")
llm2 = ChatOpenAI(
    model="anthropic/claude-3.5-sonnet",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


# Models generate responses
def model_a_chain(inputs: dict) -> dict:
    """Generate response using first model."""
    prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
    chain = prompt | llm1
    response = chain.invoke({"question": inputs["question"]})
    return {"answer": response.content}


def model_b_chain(inputs: dict) -> dict:
    """Generate response using second model."""
    prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
    chain = prompt | llm2
    response = chain.invoke({"question": inputs["question"]})
    return {"answer": response.content}


# First evaluate each model separately to create experiments
model_a_results = evaluate(
    model_a_chain,
    data="model_comparison",
    experiment_prefix=llm1.model_name,
    max_concurrency=4,
)

model_b_results = evaluate(
    model_b_chain,
    data="model_comparison",
    experiment_prefix=llm2.model_name,
    max_concurrency=4,
)

# Then compare the two existing experiments
evaluate(
    [model_a_results.experiment_name, model_b_results.experiment_name],
    evaluators=[weighted_average],
    experiment_prefix="model-compare",
    max_concurrency=4,
)
