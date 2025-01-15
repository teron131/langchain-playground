import os

from dotenv import load_dotenv

load_dotenv()


# The `price` field must be in USD per 1k tokens.
def M2k(price_per_M: float) -> float:
    """Convert price per million tokens to price per thousand tokens."""
    return price_per_M / 1000


llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": os.environ["OPENAI_API_KEY"],
        },
        {
            "model": "gpt-4o",
            "api_key": os.environ["OPENAI_API_KEY"],
        },
        {
            "model": "o1-mini",
            "api_key": os.environ["OPENAI_API_KEY"],
        },
        {
            "model": "openai/gpt-4o-mini",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "price": [M2k(0.15), M2k(0.6)],
        },
        {
            "model": "openai/gpt-4o",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "price": [M2k(2.5), M2k(10)],
        },
        {
            "model": "openai/o1-mini",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "price": [M2k(3), M2k(12)],
        },
        {
            "model": "anthropic/claude-3.5-sonnet",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "price": [M2k(3), M2k(15)],
        },
        {
            "model": "google/gemini-2.0-flash-exp:free",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "price": [M2k(0), M2k(0)],
        },
        {
            "model": "google/gemini-2.0-flash-thinking-exp:free",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "price": [M2k(0), M2k(0)],
        },
        {
            "model": "google/gemini-exp-1206:free",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "price": [M2k(0), M2k(0)],
        },
        {
            "model": "deepseek/deepseek-chat",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "price": [M2k(0.14), M2k(0.28)],
        },
    ],
}

# Filtering example
# import autogen

# filter_dict = {"model": ["google/gemini-2.0-flash-exp:free"]}
# llm_config["config_list"] = autogen.filter_config(llm_config["config_list"], filter_dict)
# llm_config
