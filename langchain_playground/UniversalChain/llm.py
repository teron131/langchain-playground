import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_together import ChatTogether


def _init_azure_openai(model_id: str) -> BaseChatModel:
    return AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )


def _init_openai(model_id: str) -> BaseChatModel:
    return ChatOpenAI(
        model=model_id,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _init_gemini(model_id: str) -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=model_id,
        api_key=os.getenv("GOOGLE_API_KEY"),
    )


def _init_openrouter(model_id: str) -> BaseChatModel:
    return ChatOpenAI(
        model=model_id,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


def _init_nvidia(model_id: str) -> BaseChatModel:
    return ChatNVIDIA(
        model=model_id,
        api_key=os.getenv("NVIDIA_API_KEY"),
    )


def _init_together(model_id: str) -> BaseChatModel:
    return ChatTogether(
        model=model_id,
        api_key=os.getenv("TOGETHER_API_KEY"),
    )


def _init_groq(model_id: str) -> BaseChatModel:
    return ChatGroq(
        model=model_id,
        api_key=os.getenv("GROQ_API_KEY"),
    )


PROVIDERS = {
    "azure": _init_azure_openai,
    "openai": _init_openai,
    "gemini": _init_gemini,
    "openrouter": _init_openrouter,
    "nvidia": _init_nvidia,
    "together": _init_together,
    "groq": _init_groq,
}


def get_llm(provider: str = "openai", model_id: str = "gpt-4o-mini") -> BaseChatModel:
    """Initialize and return a language model based on the model ID and provider.

    Args:
        provider (str): Explicit provider specification.
        model_id (str): Identifier for the language model to initialize.

    Returns:
        BaseChatModel: Initialized language model instance.

    Raises:
        ValueError: If the model_id is invalid, provider is invalid, or initialization fails.
    """
    try:
        if provider:
            if provider not in PROVIDERS:
                raise ValueError(f"Invalid provider: {provider}. Available providers: {list(PROVIDERS.keys())}")
            return PROVIDERS[provider](model_id)

        # Fallback to generic initialization
        return init_chat_model(model=model_id)

    except Exception as e:
        raise ValueError(f"Failed to initialize model {model_id}: {str(e)}")
