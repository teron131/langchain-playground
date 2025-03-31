"""Define the configurable parameters for the agent."""

from __future__ import annotations

from typing import Literal, Optional

from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import BaseModel, Field

MODELS = sorted(
    [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "openai/o3-mini",
        "openai/o3-mini-high",
        "anthropic/claude-3.7-sonnet",
        "anthropic/claude-3.7-sonnet:thinking",
        "google/gemini-2.0-flash-001",
        "google/gemini-2.0-flash-thinking-exp:free",
        "google/gemini-2.5-pro-exp-03-25:free",
    ]
)


class Configuration(BaseModel):
    """The configuration for the agent."""

    suggested_model: Literal[*MODELS] = Field(  # type: ignore
        default="openai/gpt-4o-mini",
        description="The model ID in OpenRouter format to use for the agent's main interactions.",
    )

    custom_model: str = Field(
        default="",
        description="The model ID in OpenRouter format to use for the agent's main interactions.",
    )

    system_prompt: str = Field(
        default="",
        description="The system prompt to use for the agent's interactions. This prompt sets the context and behavior for the agent.",
    )

    max_search_results: int = Field(
        default=10,
        description="The maximum number of search results to return for each search query.",
    )

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig object."""
        config: dict = ensure_config(config)
        configurable: dict = config.get("configurable", {})
        valid_fields = {k: v for k, v in configurable.items() if k in cls.model_fields}
        return cls(**valid_fields)
