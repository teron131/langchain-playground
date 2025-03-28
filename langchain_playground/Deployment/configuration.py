"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import BaseModel, Field

MODELS = sorted(
    [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "anthropic/claude-3.7-sonnet",
    ]
)


class Configuration(BaseModel):
    """The configuration for the agent."""

    provider: Literal["OpenAI", "Google", "OpenRouter"] = Field(
        default="OpenAI",
        description="The provider to use for the agent's interactions. Should be one of: OpenAI, Google, OpenRouter.",
    )

    model: Literal[*MODELS] = Field(  # type: ignore
        default="openai/gpt-4o-mini",
        description="The name of the language model to use for the agent's main interactions. Should be in the form: provider/model-name.",
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
        configurable = config.get("configurable", {})
        valid_fields = {k: v for k, v in configurable.items() if k in cls.model_fields}
        return cls(**valid_fields)
