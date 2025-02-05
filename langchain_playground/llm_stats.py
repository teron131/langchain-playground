import os
from typing import ClassVar, Dict, Literal, Union

from langchain_openai import ChatOpenAI


class BaseLLMwithStats(ChatOpenAI):
    """Base class for GPT models with embedded stats"""

    model_name: ClassVar[str]
    model_type: ClassVar[Literal["chat", "reasoning"]]
    elo: ClassVar[int]
    latency: ClassVar[float]
    throughput: ClassVar[float]
    speed: ClassVar[int]
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
            "speed": self.speed,
            "input_price_per_1k": self.input_price_per_1k,
            "output_price_per_1k": self.output_price_per_1k,
        }


##### OpenAI #####


class gpt4omini(BaseLLMwithStats):
    model_name: ClassVar[str] = "openai/gpt-4o-mini"
    model_type: ClassVar[Literal["chat", "reasoning"]] = "chat"
    elo: ClassVar[int] = 1273
    latency: ClassVar[float] = 0.36
    throughput: ClassVar[float] = 62.23
    input_price_per_1k: ClassVar[float] = 0.15
    output_price_per_1k: ClassVar[float] = 0.6


class gpt4o(BaseLLMwithStats):
    model_name: ClassVar[str] = "openai/gpt-4o"
    model_type: ClassVar[Literal["chat", "reasoning"]] = "chat"
    elo: ClassVar[int] = 1365
    latency: ClassVar[float] = 0.32
    throughput: ClassVar[float] = 67.58
    input_price_per_1k: ClassVar[float] = 2.5
    output_price_per_1k: ClassVar[float] = 10


class o1mini(BaseLLMwithStats):
    model_name: ClassVar[str] = "openai/o1-mini"
    model_type: ClassVar[Literal["chat", "reasoning"]] = "reasoning"
    elo: ClassVar[int] = 1305
    latency: ClassVar[float] = 1.17
    throughput: ClassVar[float] = 183.9
    input_price_per_1k: ClassVar[float] = 1.1
    output_price_per_1k: ClassVar[float] = 4.4


# class o3mini(BaseLLMwithStats):
#     model_name: ClassVar[str] = "openai/o3-mini"
#     model_type: ClassVar[Literal["chat", "reasoning"]] = "reasoning"
#     elo: ClassVar[int] = 1365
#     latency: ClassVar[float] = 3.46
#     throughput: ClassVar[float] = 4648
#     input_price_per_1k: ClassVar[float] = 1.1
#     output_price_per_1k: ClassVar[float] = 4.4


##### Google #####


class geminiexp(BaseLLMwithStats):
    model_name: ClassVar[str] = "google/gemini-exp-1206:free"
    model_type: ClassVar[Literal["chat", "reasoning"]] = "chat"
    elo: ClassVar[int] = 1373
    latency: ClassVar[float] = 0.79
    throughput: ClassVar[float] = 44.97
    input_price_per_1k: ClassVar[float] = 0
    output_price_per_1k: ClassVar[float] = 0


class gemini20flash(BaseLLMwithStats):
    model_name: ClassVar[str] = "google/gemini-2.0-flash-exp:free"
    model_type: ClassVar[Literal["chat", "reasoning"]] = "chat"
    elo: ClassVar[int] = 1356
    latency: ClassVar[float] = 1.05
    throughput: ClassVar[float] = 120.5
    input_price_per_1k: ClassVar[float] = 0
    output_price_per_1k: ClassVar[float] = 0


class gemini20flashthinking(BaseLLMwithStats):
    model_name: ClassVar[str] = "google/gemini-2.0-flash-thinking-exp:free"
    model_type: ClassVar[Literal["chat", "reasoning"]] = "reasoning"
    elo: ClassVar[int] = 1384
    latency: ClassVar[float] = 4.45
    throughput: ClassVar[float] = 228.4
    input_price_per_1k: ClassVar[float] = 0
    output_price_per_1k: ClassVar[float] = 0


##### Anthropic #####


class claude35sonnet(BaseLLMwithStats):
    model_name: ClassVar[str] = "anthropic/claude-3.5-sonnet"
    model_type: ClassVar[Literal["chat", "reasoning"]] = "chat"
    elo: ClassVar[int] = 1283
    latency: ClassVar[float] = 1.48
    throughput: ClassVar[float] = 55.66
    input_price_per_1k: ClassVar[float] = 3
    output_price_per_1k: ClassVar[float] = 15


##### DeepSeek #####


class deepseekr1distillqwen32b(BaseLLMwithStats):
    model_name: ClassVar[str] = "deepseek/deepseek-r1-distill-qwen-32b"
    model_type: ClassVar[Literal["chat", "reasoning"]] = "reasoning"
    elo: ClassVar[int] = 1305
    latency: ClassVar[float] = 0.18
    throughput: ClassVar[float] = 34.53
    input_price_per_1k: ClassVar[float] = 0.12
    output_price_per_1k: ClassVar[float] = 0.18


class deepseekr1distillllama70b(BaseLLMwithStats):
    model_name: ClassVar[str] = "deepseek/deepseek-r1-distill-llama-70b"
    model_type: ClassVar[Literal["chat", "reasoning"]] = "reasoning"
    elo: ClassVar[int] = 1305
    latency: ClassVar[float] = 0.36
    throughput: ClassVar[float] = 31.18
    input_price_per_1k: ClassVar[float] = 0.23
    output_price_per_1k: ClassVar[float] = 0.69


class deepseekr1(BaseLLMwithStats):
    model_name: ClassVar[str] = "deepseek/deepseek-r1"
    model_type: ClassVar[Literal["chat", "reasoning"]] = "reasoning"
    elo: ClassVar[int] = 1361
    latency: ClassVar[float] = 0.72
    throughput: ClassVar[float] = 10.54
    input_price_per_1k: ClassVar[float] = 0.8
    output_price_per_1k: ClassVar[float] = 2.4
