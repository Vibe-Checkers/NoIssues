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

import httpx

from langchain_openai import ChatOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception,
)
from openai import RateLimitError, APIConnectionError, APIStatusError

from parallel.rate_limiter import GlobalRateLimiter

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
OPENROUTER_RETRY_ATTEMPTS = int(os.environ.get("OPENROUTER_RETRY_ATTEMPTS", "4"))
OPENROUTER_RETRY_MULTIPLIER = float(os.environ.get("OPENROUTER_RETRY_MULTIPLIER_SECONDS", "0.5"))
OPENROUTER_RETRY_MAX_WAIT = float(os.environ.get("OPENROUTER_RETRY_MAX_WAIT_SECONDS", "8"))


def _extract_api_error_details(exc: Exception) -> dict[str, str | int | None]:
    """Extract status/content-type/body preview headers from API exceptions for observability."""
    status_code = getattr(exc, "status_code", None)
    content_type = None
    body_preview = None
    cf_ray = None
    request_id = None

    response = getattr(exc, "response", None)
    if response is not None:
        status_code = status_code or getattr(response, "status_code", None)
        headers = getattr(response, "headers", {})
        content_type = headers.get("content-type")
        cf_ray = headers.get("cf-ray")
        request_id = headers.get("x-request-id") or headers.get("openai-request-id")
        text = getattr(response, "text", None)
        if text:
            body_preview = text.replace("\n", " ")[:200]

    return {
        "status_code": status_code,
        "content_type": content_type,
        "body_preview": body_preview,
        "cf_ray": cf_ray,
        "request_id": request_id,
    }


def _is_retryable_exception(exc: Exception) -> bool:
    """Return whether an exception is safe/expected to retry for OpenRouter calls."""
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", None)
        if status_code is None and getattr(exc, "response", None) is not None:
            status_code = getattr(exc.response, "status_code", None)
        return status_code in RETRYABLE_STATUS_CODES
    return False


def _log_before_sleep(retry_state) -> None:
    """Structured retry log with upstream diagnostics for incident triage."""
    exc = retry_state.outcome.exception()
    details = _extract_api_error_details(exc)
    sleep_s = retry_state.next_action.sleep if retry_state.next_action else 0
    logger.warning(
        "LLM call attempt %d/%d failed; retrying in %.2fs: %s",
        retry_state.attempt_number,
        OPENROUTER_RETRY_ATTEMPTS,
        sleep_s,
        exc,
        extra={
            "status_code": details["status_code"],
            "content_type": details["content_type"],
            "cf_ray": details["cf_ray"],
            "request_id": details["request_id"],
            "body_preview": details["body_preview"],
        },
    )


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
        request_timeout = float(os.environ.get("OPENROUTER_TIMEOUT_SECONDS", "90"))
        connect_timeout = float(os.environ.get("OPENROUTER_CONNECT_TIMEOUT_SECONDS", "3"))
        read_timeout = float(os.environ.get("OPENROUTER_READ_TIMEOUT_SECONDS", "60"))
        write_timeout = float(os.environ.get("OPENROUTER_WRITE_TIMEOUT_SECONDS", "10"))
        timeout_cfg = httpx.Timeout(
            timeout=request_timeout,
            connect=connect_timeout,
            read=read_timeout,
            write=write_timeout,
        )
        # Keep SDK retries disabled by default to avoid retry storms and duplicate backoff loops.
        # We do retries centrally in _llm_call_with_retry.
        max_retries = int(os.environ.get("OPENROUTER_SDK_MAX_RETRIES", "0"))

        self.nano = ChatOpenAI(
            model=os.environ["OPENROUTER_MODEL_NANO"],
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=timeout_cfg,
            max_retries=max_retries,
        )
        self.chat = ChatOpenAI(
            model=os.environ["OPENROUTER_MODEL_CHAT"],
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            timeout=timeout_cfg,
            max_retries=max_retries,
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
    stop=stop_after_attempt(OPENROUTER_RETRY_ATTEMPTS),
    wait=wait_random_exponential(multiplier=OPENROUTER_RETRY_MULTIPLIER, max=OPENROUTER_RETRY_MAX_WAIT),
    retry=retry_if_exception(_is_retryable_exception),
    reraise=True,
    before_sleep=_log_before_sleep,
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
