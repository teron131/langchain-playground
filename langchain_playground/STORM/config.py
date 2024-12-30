from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize LLMs
# We will have a faster LLM do most of the work, but a slower, long-context model
# to distill the conversations and write the final report.
fast_llm = ChatOpenAI(model="gpt-4o-mini")
long_context_llm = ChatOpenAI(model="gpt-4o-mini")
