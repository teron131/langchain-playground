import os

from dotenv import load_dotenv

load_dotenv()

llm_config = {
    "config_list": [
        {
            "model": "gpt-4o-mini",
            "api_key": os.environ["OPENAI_API_KEY"],
        },
        {
            "model": "anthropic/claude-3.5-sonnet",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "price": [0.00375, 0.015],
        },
        {
            "model": "deepseek/deepseek-chat",
            "api_key": os.environ["OPENROUTER_API_KEY"],
            "base_url": "https://openrouter.ai/api/v1",
            "price": [0.00014, 0.00028],
        },
    ],
}

# Filtering example
# import autogen

# filter_dict = {"model": ["gpt-4o-mini"]}

# config_list = autogen.filter_config(llm_config["config_list"], filter_dict)
# llm_config["config_list"] = config_list
# llm_config
