import re
from typing import List, Optional

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

planner_prompt = """
For the following task, make plans that can solve the problem step by step. For each plan, indicate which external tool together with tool input to retrieve evidence. You can store the evidence into a variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

Tools can be one of the following:
(1) Google[input]: Worker that searches results from Google. Useful when you need to find short and succinct answers about a specific topic. The input should be a search query.
(2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general world knowledge and common sense. Prioritize it when you are confident in solving the problem yourself. Input can be any instruction.

For example,
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours less than Toby. How many hours did Rebecca work?
Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve with Wolfram Alpha. #E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: Find out the number of hours Thomas worked. #E2 = LLM[What is x, given #E1]
Plan: Calculate the number of hours Rebecca worked. #E3 = Calculator[(2 ∗ #E2 − 10) − 8]

Begin! 
Describe your plans with rich details. Each Plan should be followed by only one #E.

Task: {task}
"""

solver_prompt = """
Solve the following task or problem. To solve the problem, we have made step-by-step Plan and retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might contain irrelevant information.

Plan: {plan}

Now solve the question or task according to provided Evidence above. Respond with the answer directly with no extra words.

Task: {task}
Response:
"""


class ReWOO(TypedDict):
    task: str
    plan_string: str
    steps: List
    results: dict
    result: str


class ReWOOGraph:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name)
        self.search = TavilySearchResults()

    def get_plan(self, state: ReWOO) -> dict:
        prompt = ChatPromptTemplate.from_messages([("user", planner_prompt)])
        planner = prompt | self.llm

        task = state["task"]
        result = planner.invoke({"task": task})

        regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"
        matches = re.findall(regex_pattern, result.content)

        return {"steps": matches, "plan_string": result.content}

    def _get_current_task(self, state: ReWOO) -> Optional[int]:
        if "results" not in state or state["results"] is None:
            return 1
        if len(state["results"]) == len(state["steps"]):
            return None
        return len(state["results"]) + 1

    def tool_execution(self, state: ReWOO) -> dict:
        """Worker node that executes the tools of a given plan."""
        step_num = self._get_current_task(state)
        _, step_name, tool, tool_input = state["steps"][step_num - 1]

        results = (state["results"] or {}) if "results" in state else {}
        for k, v in results.items():
            tool_input = tool_input.replace(k, v)

        if tool == "Google":
            result = self.search.invoke(tool_input)
        elif tool == "LLM":
            result = self.llm.invoke(tool_input)
        else:
            raise ValueError(f"Unknown tool: {tool}")

        results[step_name] = str(result)
        return {"results": results}

    def solve(self, state: ReWOO) -> dict:
        plan_steps = []
        for _plan, step_name, tool, tool_input in state["steps"]:
            results = (state["results"] or {}) if "results" in state else {}
            for k, v in results.items():
                tool_input = tool_input.replace(k, v)
                step_name = step_name.replace(k, v)
            plan_steps.append(f"Plan: {_plan}\n{step_name} = {tool}[{tool_input}]")

        plan = "\n".join(plan_steps)
        prompt = solver_prompt.format(plan=plan, task=state["task"])
        result = self.llm.invoke(prompt)
        return {"result": result.content}

    def _route(self, state: ReWOO) -> str:
        step_num = self._get_current_task(state)
        if step_num is None:
            return "solve"
        return "tool"

    def create_graph(self) -> StateGraph:
        graph = StateGraph(ReWOO)

        # Add nodes
        graph.add_node("plan", self.get_plan)
        graph.add_node("tool", self.tool_execution)
        graph.add_node("solve", self.solve)

        # Add edges
        graph.add_edge("plan", "tool")
        graph.add_edge("solve", END)
        graph.add_conditional_edges("tool", self._route)
        graph.add_edge(START, "plan")

        return graph.compile()


if __name__ == "__main__":
    task = "what is the exact hometown of the 2024 mens australian open winner?"
    rewoo = ReWOOGraph()
    graph = rewoo.create_graph()
    for s in graph.stream({"task": task}):
        print(s)
        print("---")
    print(s["solve"]["result"])
