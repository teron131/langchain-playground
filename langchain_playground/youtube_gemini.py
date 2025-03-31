from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    Tool,
    ToolCodeExecution,
)

load_dotenv()

client = genai.Client()

response = client.models.generate_content(
    model="models/gemini-2.0-flash",
    contents=types.Content(
        parts=[
            types.Part(text="Can you summarize this video?"),
            types.Part(file_data=types.FileData(file_uri="https://www.youtube.com/watch?v=dNntYdZQ_mk")),
        ],
    ),
)

print(response)
