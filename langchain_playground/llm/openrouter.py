"""OpenRouter LLM client initialization and configuration."""

import os
from typing import Any, Generator, Literal, Optional

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

load_dotenv()


def ChatOpenRouter(
    model: str,
    temperature: float = 0.0,
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = None,
    provider_sort: Literal["throughput", "price", "latency"] = "throughput",
    pdf_engine: Optional[Literal["mistral-ocr", "pdf-text", "native"]] = None,
    **kwargs,
) -> BaseChatModel:
    """Initialize an OpenRouter model with sensible defaults.

    Uses OpenAI Responses API and reasoning is returned by default.

    Args:
        model: Model identifier (PROVIDER/MODEL)
        temperature: Sampling temperature (0.0-2.0)
        reasoning_effort: Level of reasoning effort for reasoning models.
            Can be "minimal", "low", "medium", or "high".
        provider_sort: Routing preference for OpenRouter (default: "throughput")
        pdf_engine: PDF processing engine for file attachments. Options:
            - "mistral-ocr": Best for scanned documents ($2 per 1,000 pages)
            - "pdf-text": Best for well-structured PDFs (Free)
            - "native": Use model's native file processing (if available)
            If None, OpenRouter will auto-select based on model capabilities.
        **kwargs: Additional config (e.g. max_tokens, extra_body, etc.)

    Note: Some models (e.g., google/gemini-2.5-pro) may not support PDFs through OpenRouter in the same way as others. If you encounter "invalid_prompt" errors with PDFs, try a different model.
    """
    extra_body = _build_extra_body(
        base_extra_body=kwargs.pop("extra_body", {}) or {},
        provider_sort=provider_sort,
        pdf_engine=pdf_engine,
        reasoning_effort=reasoning_effort,
    )

    return init_chat_model(
        model=model,
        model_provider="openai",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=temperature,
        use_responses_api=True,
        extra_body=extra_body or None,
        **kwargs,
    )


def _build_extra_body(
    *,
    base_extra_body: dict[str, Any],
    provider_sort: Literal["throughput", "price", "latency"],
    pdf_engine: Optional[Literal["mistral-ocr", "pdf-text", "native"]],
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]],
) -> dict[str, Any]:
    """Build OpenRouter extra_body config."""
    return {
        **base_extra_body,
        **({"provider": {"sort": provider_sort}} if provider_sort and "provider" not in base_extra_body else {}),
        **({"plugins": [{"id": "file-parser", "pdf": {"engine": pdf_engine}}]} if pdf_engine else {}),
        "reasoning": {"exclude": False, **({"effort": reasoning_effort} if reasoning_effort else {})},
    }


# ============================================================================
# Response parsing utilities
# ============================================================================


def _extract_reasoning(content_blocks: list[dict]) -> str | None:
    """Extract reasoning from response content_blocks."""
    if not content_blocks:
        return None

    block = content_blocks[0]
    if reasoning := block.get("reasoning"):
        return reasoning

    if extras := block.get("extras"):
        if isinstance(extras, dict) and (content := extras.get("content")):
            if isinstance(content, list) and content and isinstance(content[-1], dict):
                return content[-1].get("text")

    return None


def parse_invoke(
    response: AIMessage,
    include_reasoning: bool = False,
) -> str | tuple[str | None, str]:
    """Parse response to extract answer and optionally reasoning."""
    answer = response.content_blocks[-1]["text"]
    if include_reasoning:
        reasoning = _extract_reasoning(response.content_blocks)
        return reasoning, answer
    return answer


def parse_batch(
    responses: list[AIMessage],
    include_reasoning: bool = False,
) -> list[str] | list[tuple[str | None, str]]:
    """Parse batched responses, optionally with reasoning."""
    return [parse_invoke(response, include_reasoning) for response in responses]


def get_stream_generator(
    stream: Generator[AIMessage, None, None],
    include_reasoning: bool = False,
) -> Generator[str | tuple[str, str | None], None, None]:
    """Yield streaming chunks, optionally with reasoning."""
    reasoning_yielded = False

    for chunk in stream:
        if not (blocks := getattr(chunk, "content_blocks", None)):
            continue

        if include_reasoning and not reasoning_yielded and (reasoning := _extract_reasoning(blocks)):
            reasoning_yielded = True
            yield (reasoning, None)

        if answer := blocks[-1].get("text"):
            yield (None, answer) if include_reasoning else answer


def parse_stream(
    stream: Generator[AIMessage, None, None],
    include_reasoning: bool = False,
) -> str | tuple[str | None, str]:
    """Print streamed chunks and return the final result."""
    reasoning = None
    answer_parts: list[str] = []

    for item in get_stream_generator(stream, include_reasoning):
        if isinstance(item, tuple):
            reasoning_chunk, answer_chunk = item
            if reasoning_chunk is not None:
                reasoning = reasoning_chunk
                print(f"Reasoning: {reasoning}", flush=True)
            if answer_chunk is not None:
                answer_parts.append(answer_chunk)
                print(answer_chunk, end="", flush=True)
        else:
            answer_parts.append(item)
            print(item, end="", flush=True)

    answer = "".join(answer_parts)
    return (reasoning, answer) if include_reasoning else answer
