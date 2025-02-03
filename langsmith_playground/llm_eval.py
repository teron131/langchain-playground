from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client
from pydantic import BaseModel, Field

load_dotenv()

client = Client()

# For other dataset creation methods, see:
# https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_programmatically
# https://docs.smith.langchain.com/evaluation/how_to_guides/manage_datasets_in_application

# Create inputs and reference outputs
examples = [
    (
        "Which country is Mount Kilimanjaro located in?",
        "Mount Kilimanjaro is located in Tanzania.",
    ),
    (
        "What is Earth's lowest point?",
        "Earth's lowest point is The Dead Sea.",
    ),
]

questions = [{"question": question} for question, _ in examples]
answers = [{"answer": answer} for _, answer in examples]

try:
    # Programmatically create a dataset in LangSmith
    dataset = client.create_dataset(dataset_name="Sample dataset", description="A sample dataset in LangSmith.")

    # Add examples to the dataset
    client.create_examples(inputs=questions, outputs=answers, dataset_id=dataset.id)
except Exception as e:
    pass


# Define the application logic you want to evaluate inside a target function
# The SDK will automatically send the inputs from the dataset to your target function
def target(inputs: dict) -> dict:
    prompt = ChatPromptTemplate.from_messages([("user", "{question}")])
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm
    response = chain.invoke({"question": inputs["question"]})
    return {"attempted_answer": response.content}


# Define instructions for the LLM judge evaluator
instructions = """Evaluate Student Answer against Ground Truth for conceptual similarity and classify true or false: 
- False: No conceptual match and similarity
- True: Most or full conceptual match and similarity
- Key criteria: Concept should match, not exact wording.
"""


# Define output schema for the LLM judge
class Grade(BaseModel):
    score: bool = Field(description="Boolean that indicates whether the attempted answer is accurate relative to the ground truth answer")
    explanation: str = Field(description="Explanation of the grading decision")


# Define LLM judge that grades the accuracy of the response relative to reference output
def accuracy(outputs: dict, reference_outputs: dict) -> list[dict]:
    """Evaluate the accuracy of the attempted answer relative to the ground truth answer with explanation.

    Returns:
        list[dict]: Contains the boolean score and explanation from the LLM judge
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", instructions),
            ("user", "Ground Truth: {ground_truth}; Attempted Answer: {attempted_answer}"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm.with_structured_output(Grade)
    response = chain.invoke({"ground_truth": reference_outputs["answer"], "attempted_answer": outputs["attempted_answer"]})

    return [
        {"key": "accuracy", "score": response.score},
        {"key": "explanation", "value": response.explanation},
    ]


# After running the evaluation, a link will be provided to view the results in langsmith
experiment_results = client.evaluate(
    target,
    data="Sample dataset",
    evaluators=[
        accuracy,
        # can add multiple evaluators here
    ],
    experiment_prefix="first-eval-in-langsmith",
    max_concurrency=2,
)
