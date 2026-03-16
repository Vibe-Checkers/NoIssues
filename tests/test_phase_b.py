"""Tests for Phase B: LLM Client + Rate Limiter (no Docker)."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parallel.rate_limiter import GlobalRateLimiter


# ═══════════════════════════════════════════════════════
# B1: Rate Limiter
# ═══════════════════════════════════════════════════════

class TestRateLimiter:
    @pytest.fixture
    def limiter(self):
        rl = GlobalRateLimiter(rpm=10, tpm=10_000, headroom=1.0)
        yield rl
        rl.shutdown()

    def test_acquire_release_basic(self, limiter):
        """Single acquire/release should not block."""
        limiter.acquire(100)
        limiter.release(100)

    def test_rpm_limiting(self):
        """RPM semaphore blocks when limit is reached."""
        limiter = GlobalRateLimiter(rpm=3, tpm=100_000, headroom=1.0)
        try:
            # 3 acquires should succeed
            for _ in range(3):
                limiter.acquire(10)

            # 4th should block — test with a timeout
            blocked = threading.Event()

            def try_acquire():
                limiter.acquire(10)
                blocked.set()

            t = threading.Thread(target=try_acquire)
            t.start()
            t.join(timeout=0.5)

            assert not blocked.is_set(), "4th acquire should have blocked (RPM=3)"

            # Manually refill to unblock
            limiter._refill()
            t.join(timeout=2.0)
            assert blocked.is_set(), "Should unblock after refill"
        finally:
            limiter.shutdown()

    def test_tpm_limiting(self):
        """TPM budget blocks when exhausted."""
        limiter = GlobalRateLimiter(rpm=100, tpm=500, headroom=1.0)
        try:
            limiter.acquire(400)

            blocked = threading.Event()

            def try_acquire():
                limiter.acquire(400)
                blocked.set()

            t = threading.Thread(target=try_acquire)
            t.start()
            t.join(timeout=0.5)

            assert not blocked.is_set(), "Should block when TPM budget exhausted"

            # Refill restores budget
            limiter._refill()
            t.join(timeout=2.0)
            assert blocked.is_set(), "Should unblock after TPM refill"
        finally:
            limiter.shutdown()

    def test_concurrent_acquire_release(self):
        """10 threads acquiring/releasing — RPM never exceeded."""
        limiter = GlobalRateLimiter(rpm=20, tpm=100_000, headroom=1.0)
        acquired_count = 0
        max_concurrent = 0
        count_lock = threading.Lock()
        errors = []

        def worker():
            nonlocal acquired_count, max_concurrent
            try:
                for _ in range(3):
                    limiter.acquire(100)
                    with count_lock:
                        acquired_count += 1
                        if acquired_count > max_concurrent:
                            max_concurrent = acquired_count
                    time.sleep(0.01)
                    limiter.release(100)
                    with count_lock:
                        acquired_count -= 1
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        limiter.shutdown()

        assert not errors, f"Concurrent access errors: {errors}"
        assert max_concurrent <= 20, f"RPM exceeded: max_concurrent={max_concurrent}"

    def test_backoff(self, limiter):
        """Backoff sets a delay and reduces TPM budget."""
        original_tpm = limiter._tpm_limit
        limiter.backoff(0.2)

        assert limiter._budget_reduced is True
        assert limiter._tpm_limit < original_tpm

        # Subsequent acquire should be delayed
        t0 = time.monotonic()
        limiter.acquire(10)
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.1, "Backoff delay should be applied"

    def test_headroom_applied(self):
        """Headroom reduces effective limits."""
        limiter = GlobalRateLimiter(rpm=100, tpm=100_000, headroom=0.5)
        try:
            assert limiter._rpm_limit == 50
            assert limiter._tpm_limit == 50_000
        finally:
            limiter.shutdown()

    def test_stats(self, limiter):
        stats = limiter.stats
        assert "rpm_used" in stats
        assert "rpm_limit" in stats
        assert "tpm_remaining" in stats
        assert "tpm_limit" in stats
        assert stats["rpm_used"] == 0

    def test_env_var_defaults(self):
        """Falls back to env vars or defaults."""
        with patch.dict(os.environ, {}, clear=False):
            # Remove any Azure env vars that might interfere
            env = {k: v for k, v in os.environ.items() if not k.startswith("AZURE_OPENAI")}
            with patch.dict(os.environ, env, clear=True):
                limiter = GlobalRateLimiter()
                try:
                    # Default: 60 RPM * 0.85 = 51
                    assert limiter._rpm_limit == 51
                finally:
                    limiter.shutdown()


# ═══════════════════════════════════════════════════════
# B2: LLM Client
# ═══════════════════════════════════════════════════════

class TestLLMClient:
    def test_llm_response_dataclass(self):
        from agent.llm import LLMResponse
        r = LLMResponse(content="hello", prompt_tokens=10, completion_tokens=5)
        assert r.content == "hello"
        assert r.prompt_tokens == 10

    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT_NANO": "gpt5-nano",
        "AZURE_OPENAI_DEPLOYMENT_CHAT": "gpt5-chat",
        "AZURE_OPENAI_API_VERSION": "2024-02-15-preview",
    })
    def test_client_initialization(self):
        """LLMClient creates two separate deployment clients."""
        from agent.llm import LLMClient
        limiter = GlobalRateLimiter(rpm=10, tpm=10_000, headroom=1.0)
        try:
            client = LLMClient(limiter)
            assert client.nano.deployment_name == "gpt5-nano"
            assert client.chat.deployment_name == "gpt5-chat"
        finally:
            limiter.shutdown()

    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT_NANO": "gpt5-nano",
        "AZURE_OPENAI_DEPLOYMENT_CHAT": "gpt5-chat",
    })
    def test_call_nano_with_mock(self):
        """call_nano routes through rate limiter and returns LLMResponse."""
        from agent.llm import LLMClient, LLMResponse

        limiter = GlobalRateLimiter(rpm=10, tpm=10_000, headroom=1.0)
        try:
            client = LLMClient(limiter)

            mock_response = MagicMock()
            mock_response.content = "test response"
            mock_response.usage_metadata = {"input_tokens": 100, "output_tokens": 50}

            # Replace the entire Pydantic client with a MagicMock
            mock_nano = MagicMock()
            mock_nano.invoke.return_value = mock_response
            client.nano = mock_nano

            result = client.call_nano([{"role": "user", "content": "hello"}])

            assert isinstance(result, LLMResponse)
            assert result.content == "test response"
            assert result.prompt_tokens == 100
            assert result.completion_tokens == 50
            mock_nano.invoke.assert_called_once()
        finally:
            limiter.shutdown()

    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT_NANO": "gpt5-nano",
        "AZURE_OPENAI_DEPLOYMENT_CHAT": "gpt5-chat",
    })
    def test_call_chat_uses_chat_deployment(self):
        """call_chat routes to the chat deployment."""
        from agent.llm import LLMClient

        limiter = GlobalRateLimiter(rpm=10, tpm=10_000, headroom=1.0)
        try:
            client = LLMClient(limiter)

            mock_response = MagicMock()
            mock_response.content = "chat response"
            mock_response.usage_metadata = {"input_tokens": 200, "output_tokens": 100}

            mock_chat = MagicMock()
            mock_chat.invoke.return_value = mock_response
            mock_nano = MagicMock()  # should NOT be called
            client.chat = mock_chat
            client.nano = mock_nano

            result = client.call_chat([{"role": "user", "content": "analyze"}])
            assert result.content == "chat response"
            mock_chat.invoke.assert_called_once()
            mock_nano.invoke.assert_not_called()
        finally:
            limiter.shutdown()

    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT_NANO": "gpt5-nano",
        "AZURE_OPENAI_DEPLOYMENT_CHAT": "gpt5-chat",
    })
    def test_retry_on_rate_limit(self):
        """Retries on RateLimitError up to 3 attempts."""
        from agent.llm import LLMClient
        from openai import RateLimitError

        limiter = GlobalRateLimiter(rpm=100, tpm=100_000, headroom=1.0)
        try:
            client = LLMClient(limiter)

            mock_response = MagicMock()
            mock_response.content = "ok"
            mock_response.usage_metadata = {"input_tokens": 10, "output_tokens": 5}

            error = RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            )
            mock_nano = MagicMock()
            mock_nano.invoke.side_effect = [error, error, mock_response]
            client.nano = mock_nano

            result = client.call_nano([{"role": "user", "content": "test"}])
            assert result.content == "ok"
            assert mock_nano.invoke.call_count == 3
        finally:
            limiter.shutdown()

    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
        "AZURE_OPENAI_DEPLOYMENT_NANO": "gpt5-nano",
        "AZURE_OPENAI_DEPLOYMENT_CHAT": "gpt5-chat",
    })
    def test_retry_exhausted_raises(self):
        """After 3 failed attempts, the error propagates."""
        from agent.llm import LLMClient
        from openai import RateLimitError

        limiter = GlobalRateLimiter(rpm=100, tpm=100_000, headroom=1.0)
        try:
            client = LLMClient(limiter)

            error = RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            )
            mock_nano = MagicMock()
            mock_nano.invoke.side_effect = error
            client.nano = mock_nano

            with pytest.raises(RateLimitError):
                client.call_nano([{"role": "user", "content": "test"}])

            assert mock_nano.invoke.call_count == 3
        finally:
            limiter.shutdown()
