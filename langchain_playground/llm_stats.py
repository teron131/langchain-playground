import os
import re
import warnings
from typing import ClassVar, Dict, List, Literal, Union

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

load_dotenv()


ELO_FILE = "model_elo.pkl"


def load_elo_data() -> pd.DataFrame:
    if os.path.exists(ELO_FILE):
        return pd.read_pickle(ELO_FILE)
    else:
        dataset = load_dataset("mathewhe/chatbot-arena-elo", split="train")
        df = dataset.to_pandas()

        df = df[["Rank* (UB)", "Model", "Arena Score"]]
        df.loc[:, "Model"] = df["Model"].str.lower()

        # Remove parentheses content and trailing date patterns
        df.loc[:, "Model"] = df["Model"].apply(lambda x: re.sub(r"\s*\([^)]*\)|\s*-\d{2}-\d{2}(?!\d)|\s*-\d{4}-\d{2}-\d{2}|\s*-\d{4}|\s*-\d{6}|\s*-\d{8}", "", x).strip())

        df.loc[:, "Model"] = df["Model"].apply(lambda x: x.replace(" ", "-"))

        # Merge duplicate rows (if any) by grouping on Rank* (UB) and Model
        # and taking the maximum Arena Score per group
        df = df.groupby(["Model"], as_index=False).agg({"Arena Score": "max", "Rank* (UB)": "min"})

        df = df.sort_values(by="Rank* (UB)").reset_index(drop=True)
        df = df[["Rank* (UB)", "Model", "Arena Score"]]
        df.to_pickle(ELO_FILE)
        return df


df = load_elo_data()


def lookup_elo(model: str) -> int:
    matches = df[df["Model"] == model.lower()]
    if len(matches) == 0:
        return None
    return int(matches["Arena Score"].iloc[0])


class BaseLLMwithStats(ChatOpenAI):
    """Base class for GPT models with embedded stats"""

    model_name: ClassVar[str]
    model_type: ClassVar[Literal["chat", "reasoning"]]
    elo: ClassVar[int]
    latency: ClassVar[float]
    throughput: ClassVar[float]
    input_price_per_1k: ClassVar[float]
    output_price_per_1k: ClassVar[float]

    def __init__(self, **kwargs):
        super().__init__(
            model=self.model_name,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            **kwargs,
        )

    @property
    def model_stats(self) -> Dict[str, Union[str, float]]:
        return {
            "model": self.model_name,
            "model_type": self.model_type,
            "elo": self.elo,
            "latency": self.latency,
            "throughput": self.throughput,
            "input_price_per_1k": self.input_price_per_1k,
            "output_price_per_1k": self.output_price_per_1k,
        }


MODEL_CONFIGS = {
    # OpenAI
    "gpt4omini": {
        "model_name": "openai/gpt-4o-mini",
        "model_type": "chat",
        "elo": lookup_elo("gpt-4o-mini"),
        "latency": 0.36,
        "throughput": 62.23,
        "input_price_per_1k": 0.15,
        "output_price_per_1k": 0.6,
    },
    "gpt4o": {
        "model_name": "openai/gpt-4o",
        "model_type": "chat",
        "elo": lookup_elo("gpt-4o"),
        "latency": 0.32,
        "throughput": 67.58,
        "input_price_per_1k": 2.5,
        "output_price_per_1k": 10,
    },
    "o1mini": {
        "model_name": "openai/o1-mini",
        "model_type": "reasoning",
        "elo": lookup_elo("o1-mini"),
        "latency": 1.17,
        "throughput": 183.9,
        "input_price_per_1k": 1.1,
        "output_price_per_1k": 4.4,
    },
    # Google
    "geminiexp": {
        "model_name": "google/gemini-exp-1206:free",
        "model_type": "chat",
        "elo": lookup_elo("gemini-exp-1206"),
        "latency": 0.79,
        "throughput": 44.97,
        "input_price_per_1k": 0,
        "output_price_per_1k": 0,
    },
    "gemini20flash": {
        "model_name": "google/gemini-2.0-flash-exp:free",
        "model_type": "chat",
        "elo": lookup_elo("gemini-2.0-flash-exp"),
        "latency": 1.05,
        "throughput": 120.5,
        "input_price_per_1k": 0,
        "output_price_per_1k": 0,
    },
    "gemini20flashthinking": {
        "model_name": "google/gemini-2.0-flash-thinking-exp:free",
        "model_type": "reasoning",
        "elo": lookup_elo("gemini-2.0-flash-thinking-exp"),
        "latency": 4.45,
        "throughput": 228.4,
        "input_price_per_1k": 0,
        "output_price_per_1k": 0,
    },
    # Anthropic
    "claude35sonnet": {
        "model_name": "anthropic/claude-3.5-sonnet",
        "model_type": "chat",
        "elo": lookup_elo("claude-3.5-sonnet"),
        "latency": 1.48,
        "throughput": 55.66,
        "input_price_per_1k": 3,
        "output_price_per_1k": 15,
    },
    # DeepSeek
    "deepseekr1distillqwen32b": {
        "model_name": "deepseek/deepseek-r1-distill-qwen-32b",
        "model_type": "reasoning",
        "elo": lookup_elo("deepseek-r1-distill-qwen-32b"),
        "latency": 0.18,
        "throughput": 34.53,
        "input_price_per_1k": 0.12,
        "output_price_per_1k": 0.18,
    },
    "deepseekr1distillllama70b": {
        "model_name": "deepseek/deepseek-r1-distill-llama-70b",
        "model_type": "reasoning",
        "elo": lookup_elo("deepseek-r1-distill-llama-70b"),
        "latency": 0.36,
        "throughput": 31.18,
        "input_price_per_1k": 0.23,
        "output_price_per_1k": 0.69,
    },
    "deepseekr1": {
        "model_name": "deepseek/deepseek-r1",
        "model_type": "reasoning",
        "elo": lookup_elo("deepseek-r1"),
        "latency": 0.72,
        "throughput": 10.54,
        "input_price_per_1k": 0.8,
        "output_price_per_1k": 2.4,
    },
}


def create_model_class(name: str, config: Dict) -> type:
    return type(
        name,
        (BaseLLMwithStats,),
        {
            "model_name": config["model_name"],
            "model_type": config["model_type"],
            "elo": config["elo"],
            "latency": config["latency"],
            "throughput": config["throughput"],
            "input_price_per_1k": config["input_price_per_1k"],
            "output_price_per_1k": config["output_price_per_1k"],
        },
    )


MODEL_CLASSES = {name: create_model_class(name, config) for name, config in MODEL_CONFIGS.items()}


def get_all_model_stats() -> List[Dict[str, Union[str, float]]]:
    """
    Get stats for all available models.

    Returns:
        List[Dict[str, Union[str, float]]]: List of all model stats
    """
    return [model_cls().model_stats for model_cls in MODEL_CLASSES.values()]


if __name__ == "__main__":
    print(get_all_model_stats())
