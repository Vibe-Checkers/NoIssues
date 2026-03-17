"""LLM client wrapper for BuildAgent v2.0.

Creates two ChatOpenAI instances (nano + chat) pointed at OpenRouter.
Both go through the global rate limiter. Retry on 429/5xx.

Environment variables:
    OPENROUTER_API_KEY       — OpenRouter API key
    OPENROUTER_MODEL_NANO    — model for agent steps (e.g. 'google/gemini-2.0-flash-001')
    OPENROUTER_MODEL_CHAT    — model for lesson extraction (e.g. 'anthropic/claude-sonnet-4')
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import RateLimitError, APIConnectionError, APIStatusError

from parallel.rate_limiter import GlobalRateLimiter

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMResponse:
    """Unified response from any LLM call."""

    content: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient:
    """Wrapper around two OpenRouter models (nano + chat) with rate limiting."""

    def __init__(self, rate_limiter: GlobalRateLimiter, worker_id: int = 0):  # noqa: ARG002
        self.limiter = rate_limiter
        api_key = os.environ["OPENROUTER_API_KEY"]

        self.nano = ChatOpenAI(
            model=os.environ["OPENROUTER_MODEL_NANO"],
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )
        self.chat = ChatOpenAI(
            model=os.environ["OPENROUTER_MODEL_CHAT"],
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )

    def call_nano(self, messages: list[dict], estimated_tokens: int = 2000) -> LLMResponse:
        """Call nano model through the rate limiter with retry."""
        return self._call(self.nano, messages, estimated_tokens)

    def call_chat(self, messages: list[dict], estimated_tokens: int = 2000) -> LLMResponse:
        """Call chat model through the rate limiter with retry."""
        return self._call(self.chat, messages, estimated_tokens)

    def _call(self, client: ChatOpenAI, messages: list[dict], estimated_tokens: int) -> LLMResponse:
        """Internal call with rate limiting and retry."""
        return _llm_call_with_retry(self.limiter, client, messages, estimated_tokens)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIStatusError)),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        "LLM call attempt %d failed, retrying: %s",
        retry_state.attempt_number,
        retry_state.outcome.exception(),
    ),
)
def _llm_call_with_retry(
    limiter: GlobalRateLimiter,
    client: ChatOpenAI,
    messages: list[dict],
    estimated_tokens: int,
) -> LLMResponse:
    """Execute a single LLM call with rate limiting and retry logic."""
    limiter.acquire(estimated_tokens)
    try:
        response = client.invoke(messages)

        # Extract token usage from response metadata
        usage = response.usage_metadata or {}
        prompt_tokens = usage.get("input_tokens", 0)
        completion_tokens = usage.get("output_tokens", 0)
        actual_tokens = prompt_tokens + completion_tokens

        limiter.release(actual_tokens)

        return LLMResponse(
            content=response.content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except RateLimitError as e:
        retry_after = getattr(e, "retry_after", None)
        limiter.backoff(retry_after)
        raise
