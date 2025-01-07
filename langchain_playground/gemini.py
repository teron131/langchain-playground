from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    Tool,
    ToolCodeExecution,
)

load_dotenv()

client = genai.Client()

model_id = "gemini-2.0-flash-exp"
google_search_tool = Tool(google_search=GoogleSearch())
code_execution_tool = ToolCodeExecution()  # Add the tool execution tool

tools = [google_search_tool, code_execution_tool]  # Include both tools


def invoke(question: str):
    return client.models.generate_content(
        model=model_id,
        contents=question,
        config=GenerateContentConfig(
            tools=tools,
            response_modalities=["TEXT"],
        ),
    )


response = invoke("When is the next total solar eclipse in the United States?")

for each in response.candidates[0].content.parts:
    print(each.text)

# Example response:
# The next total solar eclipse visible in the contiguous United States will be on ...

# To get grounding metadata as web content.
print(response.candidates[0].grounding_metadata.search_entry_point.rendered_content)
