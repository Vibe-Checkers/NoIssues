"""LLM client wrapper for BuildAgent v2.0.

Creates two AzureChatOpenAI instances (nano + chat) from env vars.
Both go through the global rate limiter. Retry on 429/5xx.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from langchain_openai import AzureChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import RateLimitError, APIConnectionError, APIStatusError

from parallel.rate_limiter import GlobalRateLimiter

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Unified response from any LLM call."""

    content: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient:
    """Wrapper around two Azure OpenAI deployments (nano + chat) with rate limiting.

    Environment variables:
        AZURE_OPENAI_ENDPOINT          — Azure endpoint URL (nano; also chat fallback)
        AZURE_OPENAI_API_KEY           — API key (nano; also chat fallback)
        AZURE_OPENAI_API_VERSION       — API version (default: 2024-02-15-preview)
        AZURE_OPENAI_DEPLOYMENT_NANO   — deployment name for gpt5-nano
        AZURE_OPENAI_DEPLOYMENT_CHAT   — deployment name for gpt5-chat
        AZURE_OPENAI_ENDPOINT_CHAT     — (optional) separate endpoint for chat model
        AZURE_OPENAI_API_KEY_CHAT      — (optional) separate API key for chat model
    """

    def __init__(self, rate_limiter: GlobalRateLimiter):
        self.limiter = rate_limiter

        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        self.nano = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NANO"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=api_version,
        )
        # Chat may use a separate Azure resource (optional override vars)
        self.chat = AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_CHAT"],
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT_CHAT", os.environ["AZURE_OPENAI_ENDPOINT"]),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY_CHAT", os.environ["AZURE_OPENAI_API_KEY"]),
            api_version=api_version,
        )

    def call_nano(self, messages: list[dict], estimated_tokens: int = 2000) -> LLMResponse:
        """Call gpt5-nano through the rate limiter with retry."""
        return self._call(self.nano, messages, estimated_tokens)

    def call_chat(self, messages: list[dict], estimated_tokens: int = 2000) -> LLMResponse:
        """Call gpt5-chat through the rate limiter with retry."""
        return self._call(self.chat, messages, estimated_tokens)

    def _call(self, client: AzureChatOpenAI, messages: list[dict], estimated_tokens: int) -> LLMResponse:
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
    client: AzureChatOpenAI,
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
