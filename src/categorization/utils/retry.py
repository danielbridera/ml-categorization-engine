"""Retry utility with exponential backoff for OpenAI API calls."""

import time
from typing import Callable, TypeVar

from openai import APIConnectionError, APITimeoutError, OpenAIError, RateLimitError

from categorization.settings.log import logger

T = TypeVar("T")


def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
) -> T:
    """
    Retry function with exponential backoff.

    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry

    Returns:
        Result from successful function call

    Raises:
        OpenAIError: If all retries exhausted
    """
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            return func()
        except RateLimitError:
            if attempt == max_retries:
                logger.error(f"Rate limit exceeded after {max_retries} retries")
                raise
            logger.warning(
                f"Rate limit hit. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
            delay *= backoff_factor
        except (APIConnectionError, APITimeoutError) as e:
            if attempt == max_retries:
                logger.error(f"Network error after {max_retries} retries: {e}")
                raise
            logger.warning(
                f"Network error. Retrying in {delay}s... (attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
            delay *= backoff_factor
        except OpenAIError as e:
            # Don't retry other OpenAI errors (invalid requests, auth, etc.)
            logger.error(f"OpenAI API error: {e}")
            raise
