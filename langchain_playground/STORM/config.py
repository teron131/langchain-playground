"""Configuration settings for the STORM pipeline."""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()


@dataclass
class STORMConfig:
    """Configuration for the STORM pipeline."""

    # LLM Models
    fast_llm_model: str = "gpt-4o-mini"
    long_context_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # API Keys (loaded from environment)
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    google_cse_id: str = field(default_factory=lambda: os.getenv("GOOGLE_CSE_ID", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    # Interview Settings
    max_interview_turns: int = 5
    max_search_results: int = 4
    search_rate_limit_delay: float = 2.0

    # Retry Settings
    max_retries: int = 3
    initial_retry_delay: float = 5.0
    retry_exponential_base: float = 2.0

    # Vector Store Settings
    vector_store_k: int = 3

    # HTTP Settings
    user_agent: str = field(default="STORM-Wikipedia-Article-Generator/1.0 (Research Project)", metadata={"help": "User agent string for API requests"})
    request_timeout: float = 120.0  # Timeout for API requests in seconds
    max_concurrent_requests: int = 5  # Maximum number of concurrent API requests

    # Cached LLM instances
    _fast_llm: Optional[BaseChatModel] = None
    _long_context_llm: Optional[BaseChatModel] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        required_vars = {
            "OPENAI_API_KEY": self.openai_api_key,
        }

        missing_vars = [var for var, value in required_vars.items() if not value]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Set user agent environment variable
        os.environ["USER_AGENT"] = self.user_agent

    @property
    def fast_llm(self) -> BaseChatModel:
        """Get or create the fast LLM instance."""
        if self._fast_llm is None:
            self._fast_llm = ChatOpenAI(
                model=self.fast_llm_model,
                openai_api_key=self.openai_api_key,
                request_timeout=self.request_timeout,
                max_retries=self.max_retries,
            )
        return self._fast_llm

    @property
    def long_context_llm(self) -> BaseChatModel:
        """Get or create the long context LLM instance."""
        if self._long_context_llm is None:
            self._long_context_llm = ChatOpenAI(
                model=self.long_context_model,
                openai_api_key=self.openai_api_key,
                request_timeout=self.request_timeout * 2,  # Double timeout for long context
                max_retries=self.max_retries,
            )
        return self._long_context_llm


# Create default configuration instance
config = STORMConfig()

# Export configured LLM instances
fast_llm = config.fast_llm
long_context_llm = config.long_context_llm
