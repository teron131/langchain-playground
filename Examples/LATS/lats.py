import math
import os
from collections import defaultdict, deque
from typing import Literal, Optional

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables import chain as as_runnable
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv()


class Reflection(BaseModel):
    reflections: str = Field(description="The critique and reflections on the sufficiency, superfluency, and general quality of the response.")
    score: int = Field(description="Score from 0-10 on the quality of the candidate response.", gte=0, lte=10)
    found_solution: bool = Field(description="Whether the response has fully solved the question or task.")

    def as_message(self):
        return HumanMessage(content=f"Reasoning: {self.reflections}\nScore: {self.score}")

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


class Node:
    def __init__(
        self,
        messages: list[BaseMessage],
        reflection: Reflection,
        parent: Optional["Node"] = None,
    ):
        self.messages = messages
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        self._is_solved = reflection.found_solution if reflection else False
        if self._is_solved:
            self._mark_tree_as_solved()
        self.backpropagate(reflection.normalized_score)

    def __repr__(self) -> str:
        return f"<Node value={self.value}, visits={self.visits}," f" solution={self.messages} reflection={self.reflection}/>"

    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages.extend(node.get_messages(include_reflections=include_reflections)[::-1])
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[::-1]  # root solution, reflection, child 1, ...

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            # We filter out all non-terminal, non-solution trajectories
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent


class Step(BaseModel):
    description: str = Field(description="Description of the step")
    tool: str = Field(description="Tool to use (Search or LLM)")
    tool_input: str = Field(description="Input for the tool")


class Plan(BaseModel):
    steps: list[Step] = Field(description="List of steps to solve the problem")
    current_step: int = Field(description="Current step being executed (1-based)", default=1)
    is_complete: bool = Field(description="Whether all steps have been completed", default=False)

    def next_step(self) -> Optional[Step]:
        if self.current_step > len(self.steps):
            self.is_complete = True
            return None
        return self.steps[self.current_step - 1]

    def advance(self):
        self.current_step += 1
        if self.current_step > len(self.steps):
            self.is_complete = True

    def get_current_context(self) -> str:
        """Get context of current step execution for reflection."""
        if self.current_step > len(self.steps):
            return "Final solution verification"
        step = self.steps[self.current_step - 1]
        return f"Step {self.current_step}: {step.description}"


class TreeState(TypedDict):
    # The full tree
    root: Node
    # The original input
    input: str
    # The execution plan
    plan: Plan


llm = ChatOpenAI(
    model="google/gemini-2.0-flash-001",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a thoughtful evaluator. Your task is to carefully analyze and grade the assistant's response to the user's question. Consider:
1. Accuracy and correctness of the information
2. Completeness of the answer
3. Clarity and coherence
4. Appropriate use of available tools
5. Whether the response fully addresses the user's needs""",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="candidate"),
    ]
)

reflection_llm_chain = prompt | llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(run_name="Reflection") | PydanticToolsParser(tools=[Reflection])


@as_runnable
def reflection_chain(inputs) -> Reflection:
    tool_choices = reflection_llm_chain.invoke(inputs)
    reflection = tool_choices[0]
    if not isinstance(inputs["candidate"][-1], AIMessage):
        reflection.found_solution = False
    return reflection


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AI assistant.",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)


initial_answer_chain = prompt_template | llm.with_config(run_name="GenerateInitialCandidate")


parser = JsonOutputToolsParser(return_id=True)


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a thoughtful planner. Your task is to break down the given problem into clear, logical steps.

Begin! 
Describe your steps with rich details. Each Step should be followed by only one #E.

Task: {input}""",
        ),
        MessagesPlaceholder(variable_name="messages", optional=True),
    ]
)

planner_chain = planner_prompt | llm.bind_tools(tools=[Step, Plan], tool_choice="Plan").with_config(run_name="Planner") | PydanticToolsParser(tools=[Plan])


# Define the node we will add to the graph
def generate_initial_response(state: TreeState) -> dict:
    """Generate the initial plan and start executing it."""
    print("\n🎯 Generating initial plan...")
    # First generate a plan
    plan_result = planner_chain.invoke({"input": state["input"]})
    plan = plan_result[0]

    print("\n📋 Generated Plan:")
    for i, step in enumerate(plan.steps, 1):
        print(f"\nStep {i}:")
        print(f"Description: {step.description}")
        print(f"Tool: {step.tool}[{step.tool_input}]")

    # Get the first step
    first_step = plan.next_step()
    if not first_step:
        print("\n⚠️ No steps in plan!")
        return {**state, "plan": plan}

    print(f"\n🚀 Starting execution of Step 1: {first_step.description}")
    print(f"Using {first_step.tool}: {first_step.tool_input}")

    # Generate initial response for first step
    res = initial_answer_chain.invoke({"input": first_step.tool_input})
    parsed = parser.invoke(res)
    output_messages = [res]

    print("\n📝 Generated outputs:")
    for msg in output_messages:
        print(f"\n{msg.content}")

    print("\n💭 Reflecting on step execution...")
    reflection = reflection_chain.invoke({"context": plan.get_current_context(), "input": state["input"], "candidate": output_messages})
    print(f"Score: {reflection.score}/10")
    print(f"Reflection: {reflection.reflections}")
    print(f"Solution found: {'Yes' if reflection.found_solution else 'No'}")

    root = Node(output_messages, reflection=reflection)
    return {
        **state,
        "root": root,
        "plan": plan,
    }


# This generates N candidate values
# for a single input to sample actions from the environment
def generate_candidates(messages: ChatPromptValue, config: RunnableConfig):
    n = config["configurable"].get("N", 5)
    chat_result = llm.generate(
        [messages.to_messages()],
        n=n,
        callbacks=config["callbacks"],
        run_name="GenerateCandidates",
    )
    return [gen.message for gen in chat_result.generations[0]]


expansion_chain = prompt_template | generate_candidates


def select(root: Node) -> dict:
    """Starting from the root node a child node is selected at each tree level until a leaf node is reached."""

    if not root.children:
        return root

    node = root
    while node.children:
        max_child = max(node.children, key=lambda child: child.upper_confidence_bound())
        node = max_child

    return node


def expand(state: TreeState, config: RunnableConfig) -> dict:
    """Execute the next step in the plan using tree search."""
    root = state["root"]
    plan = state["plan"]
    current_step = plan.next_step()

    print(f"\n📍 Current progress:")
    print(f"Step {plan.current_step}/{len(plan.steps)}: {current_step.description}")
    print(f"Tree height: {root.height}")
    print(f"Current step solved: {'Yes' if root.is_solved else 'No'}")

    # If current step is complete, advance to next step
    if root.is_solved:
        plan.advance()
        if plan.is_complete:
            print("\n✅ All steps completed!")
            return state

        # Get next step
        next_step = plan.next_step()
        if next_step:
            print(f"\n🚀 Moving to Step {plan.current_step}: {next_step.description}")
            print(f"Using {next_step.tool}: {next_step.tool_input}")

            # Generate initial response for next step
            res = initial_answer_chain.invoke({"input": next_step.tool_input})
            parsed = parser.invoke(res)
            output_messages = [res]

            print("\n📝 Generated outputs:")
            for msg in output_messages:
                print(f"\n{msg.content}")

            print("\n💭 Reflecting on step execution...")
            reflection = reflection_chain.invoke({"context": plan.get_current_context(), "input": state["input"], "candidate": output_messages})
            print(f"Score: {reflection.score}/10")
            print(f"Reflection: {reflection.reflections}")
            print(f"Solution found: {'Yes' if reflection.found_solution else 'No'}")

            root = Node(output_messages, reflection=reflection)
            state["root"] = root
            return state

    # Otherwise continue expanding current step
    print("\n🔄 Continuing current step exploration...")
    best_candidate: Node = select(root)
    messages = best_candidate.get_trajectory()

    # Generate N candidates from the single child candidate
    print(f"\n🌱 Generating new candidates for step {plan.current_step}...")
    new_candidates = expansion_chain.invoke({"input": current_step.tool_input, "messages": messages}, config)
    parsed = parser.batch(new_candidates)
    flattened = [(i, tool_call) for i, tool_calls in enumerate(parsed) for tool_call in tool_calls]
    collected_responses = defaultdict(list)

    output_messages = []
    for i, candidate in enumerate(new_candidates):
        output_messages.append([candidate] + collected_responses[i])

    print("\n📝 Generated outputs:")
    for i, msgs in enumerate(output_messages, 1):
        print(f"\nCandidate {i}:")
        for msg in msgs:
            print(f"\n{msg.content}")

    # Reflect on each candidate
    print("\n💭 Evaluating candidates...")
    reflections = reflection_chain.batch(
        [{"context": plan.get_current_context(), "input": state["input"], "candidate": msges} for msges in output_messages],
        config,
    )

    for i, reflection in enumerate(reflections):
        print(f"\nCandidate {i+1}:")
        print(f"Score: {reflection.score}/10")
        print(f"Solution found: {'Yes' if reflection.found_solution else 'No'}")

    # Grow tree
    child_nodes = [
        Node(
            cand,
            parent=best_candidate,
            reflection=reflection,
        )
        for cand, reflection in zip(output_messages, reflections)
    ]
    best_candidate.children.extend(child_nodes)
    return state


def should_loop(state: TreeState):
    """Determine whether to continue the tree search."""
    root = state["root"]
    plan = state["plan"]

    if plan.is_complete and root.is_solved:
        print("\n🎉 Task completed successfully!")
        return END
    if root.height > 5:
        print("\n⚠️ Maximum tree height reached!")
        return END
    return "expand"


builder = StateGraph(TreeState)
builder.add_node("start", generate_initial_response)
builder.add_node("expand", expand)
builder.add_edge(START, "start")


builder.add_conditional_edges(
    "start",
    # Either expand/rollout or finish
    should_loop,
    ["expand", END],
)
builder.add_conditional_edges(
    "expand",
    # Either continue to rollout or finish
    should_loop,
    ["expand", END],
)

graph = builder.compile()


# Example usage
def invoke(question: str):
    print("\n🎮 Starting execution with question:", question)
    last_step = None
    for step in graph.stream({"input": question, "plan": None}):
        last_step = step
        step_name, step_state = next(iter(step.items()))
        print("\n" + "=" * 50)
        print(f"Current phase: {step_name}")
        print(f"Tree height: {step_state['root'].height if 'root' in step_state else 0}")
        print(f"Current step: {step_state['plan'].current_step if step_state.get('plan') else 'Planning'}")
        print("=" * 50)

    # Get the root node from the last step regardless of which node it ended on
    step_name, step_state = next(iter(last_step.items()))
    solution_node = step_state["root"].get_best_solution()
    best_trajectory = solution_node.get_trajectory(include_reflections=True)

    print("\n📊 Final Solution Path:")
    print("=" * 50)
    for i, message in enumerate(best_trajectory):
        if isinstance(message, HumanMessage):
            print(f"\n💭 Reflection {i}:")
            print(message.content)
        else:
            print(f"\n🤖 Step {i}:")
            print(message.content)
    print("=" * 50)

    print("\n🎯 Final Answer:")
    print(best_trajectory[-1].content)


if __name__ == "__main__":
    question = """
An store sells a batch of 60 iPods, 10 of which are defective. Suppose that you purchase 2 iPods from
this store for your FYP project.
Suppose that if any of the iPods that you purchased were defective, then you returned them to the
store to swap for new iPods and you can only go back to the store for one time.
What is the probability that both the 2 iPods are non-defective?
    """
    invoke(question)
