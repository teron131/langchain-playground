"""Utility functions and classes for the STORM pipeline."""

import asyncio
import functools
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


class RetryError(Exception):
    """Exception raised when all retry attempts fail."""

    def __init__(self, original_error: Exception, attempts: int):
        self.original_error = original_error
        self.attempts = attempts
        super().__init__(f"Failed after {attempts} attempts. Last error: {original_error}")


async def with_retries(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    initial_delay: float = 5.0,
    exponential_base: float = 2.0,
    error_message: str = "Operation failed",
    success_message: Optional[str] = None,
    **kwargs: Any,
) -> T:
    """Execute an async function with exponential backoff retry logic.

    Args:
        func: The async function to execute
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        exponential_base: Base for exponential backoff
        error_message: Message to print on error
        success_message: Optional message to print on success
        **kwargs: Keyword arguments for the function

    Returns:
        The result of the function execution

    Raises:
        RetryError: If all retry attempts fail
    """
    retry_delay = initial_delay

    for attempt in range(max_retries):
        try:
            result = await func(*args, **kwargs)
            if success_message:
                if callable(success_message):
                    try:
                        msg = success_message(result)
                    except Exception:
                        msg = "Operation completed successfully"
                else:
                    msg = str(success_message)
                print(f"\n✅ {msg}")
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n⚠️ {error_message} (Attempt {attempt + 1}): {str(e)}")
                print(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= exponential_base
            else:
                print(f"\n❌ {error_message}: All attempts failed")
                raise RetryError(e, max_retries)


def with_fallback(fallback_value: T) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that provides a fallback value if the function fails.

    Args:
        fallback_value: Value to return if the function fails

    Returns:
        Decorated function that returns fallback_value on failure
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                print(f"\n⚠️ Function {func.__name__} failed: {str(e)}")
                print(f"Using fallback value: {fallback_value}")
                return fallback_value

        return wrapper

    return decorator


class ProgressTracker:
    """Track progress of multi-step operations."""

    def __init__(self, total_steps: int, description: str = "Operation"):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description

    def step(self, message: str = ""):
        """Increment progress and print status."""
        self.current_step += 1
        progress = (self.current_step / self.total_steps) * 100
        print(f"\n📊 {self.description}: {progress:.1f}% complete - {message}")

    @property
    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        return self.current_step >= self.total_steps


def format_conversation(messages: list) -> str:
    """Format a conversation for display."""
    formatted = []
    for msg in messages:
        name = msg.get("name", "User") if isinstance(msg, dict) else getattr(msg, "name", "User")
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        formatted.append(f"{name}: {content}")
    return "\n".join(formatted)
