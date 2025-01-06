import os

from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI


def get_llm(model_id: str):
    """Initialize and return a language model based on the model ID.

    Args:
        model_id (str): Identifier for the language model to initialize.

    Returns:
        BaseChatModel: Initialized language model instance.

    Raises:
        ValueError: If the model_id is invalid or initialization fails.
    """
    try:
        if "azure" in model_id:
            llm = AzureChatOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            )
        elif "gpt" in model_id or "o1" in model_id:
            llm = ChatOpenAI(model=model_id, api_key=os.getenv("OPENAI_API_KEY"))
        elif "gemini" in model_id:
            llm = ChatGoogleGenerativeAI(model=model_id, api_key=os.getenv("GOOGLE_API_KEY"))
        elif "claude" in model_id:
            llm = ChatOpenAI(
                model=f"anthropic/{model_id}",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
        elif "deepseek" in model_id:
            llm = ChatOpenAI(
                model=f"deepseek/{model_id}",
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
        else:
            llm = init_chat_model(model=model_id)
    except Exception as e:
        raise ValueError(f"Invalid model_id: {model_id}\n{e}")
    return llm
