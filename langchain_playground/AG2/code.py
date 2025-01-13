import io
import os
import re
from datetime import datetime

import agentops
import autogen
from autogen.agentchat import ChatResult, UserProxyAgent
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from config import llm_config
from PIL import Image

filter_dict = {"model": ["gpt-4o-mini"]}
filtered_config = autogen.filter_config(llm_config["config_list"], filter_dict)
llm_config["config_list"] = filtered_config

code_assistant = GPTAssistantAgent(
    name="Code Assistant",
    llm_config=llm_config,
    assistant_config={"tools": [{"type": "code_interpreter"}]},
    instructions="""You are an expert at both mathematics and programming. You can:
- Solve complex mathematical problems by writing and executing code
- Create data visualizations and charts using Python libraries
- Perform numerical calculations and analysis
- Write efficient Python code to solve any given problem
Reply TERMINATE when the task is solved and there are no remaining questions.
""",
)

user_proxy = UserProxyAgent(
    name="User",
    is_termination_msg=lambda msg: "TERMINATE" in msg["content"],
    code_execution_config={"work_dir": "coding", "use_docker": True},
    human_input_mode="NEVER",
)


def extract_file_id(result: ChatResult):
    """Extract OpenAI file ID from text using regex pattern."""
    file_id_pattern = r"file-[A-Za-z0-9]{22}"
    match = re.search(file_id_pattern, result.summary)
    return match.group(0) if match else None


def get_result(question: str) -> ChatResult:
    agentops.init()

    result = user_proxy.initiate_chat(
        code_assistant,
        message=question,
    )

    agentops.end_session("Success")
    return result


def invoke(question: str) -> str:
    result = get_result(question)

    # Handle plot if exists
    # Assume the chat always gives the file id (from examples)
    file_id = extract_file_id(result)
    if file_id:
        os.makedirs("plots", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plot_{timestamp}.png"
        filepath = f"plots/{filename}"

        # Retrieve the image
        image_data = code_assistant.openai_client.files.content(file_id)
        image_data_bytes = image_data.read()
        image = Image.open(io.BytesIO(image_data_bytes))

        # For ipynb
        try:
            from IPython.display import display

            display(image)
        except:
            pass

        image.save(filepath)
        print(f"Image saved to: {filepath}")

    return result.summary


question1 = "The volume of a cube is increasing at the rate of 16 cm3/s. At what rate is its total surface area increasing when the length of an edge is 6 cm?"
question2 = "Draw a line chart to show the population trend in US. Show how you solved it with code."

if __name__ == "__main__":
    invoke(question1)
    invoke(question2)
