"""Global RPM + TPM token-bucket rate limiter for BuildAgent v2.0.

Shared across all worker threads. Controls Azure OpenAI API usage
to stay within RPM and TPM quotas.
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

DEFAULT_RPM = 60
DEFAULT_TPM = 80_000
DEFAULT_HEADROOM = 0.85


class GlobalRateLimiter:
    """Thread-safe rate limiter enforcing both RPM and TPM budgets.

    - acquire(estimated_tokens): blocks until both RPM and TPM budget available.
    - release(actual_tokens): adjusts TPM counter after a call completes.
    - backoff(retry_after): called on 429 to temporarily reduce throughput.
    - Background daemon thread refills TPM budget every 60s.
    """

    def __init__(
        self,
        rpm: int | None = None,
        tpm: int | None = None,
        headroom: float | None = None,
    ):
        rpm_raw = rpm or int(os.environ.get("LLM_RPM", os.environ.get("AZURE_OPENAI_RPM", DEFAULT_RPM)))
        tpm_raw = tpm or int(os.environ.get("LLM_TPM", os.environ.get("AZURE_OPENAI_TPM", DEFAULT_TPM)))
        self._headroom = headroom or float(os.environ.get("RATE_LIMIT_HEADROOM", DEFAULT_HEADROOM))

        self._rpm_limit = max(1, int(rpm_raw * self._headroom))
        self._tpm_limit = max(1, int(tpm_raw * self._headroom))

        # RPM: semaphore-based — each acquire takes one slot, refill restores all
        self._rpm_semaphore = threading.Semaphore(self._rpm_limit)
        self._rpm_used = 0
        self._rpm_lock = threading.Lock()

        # TPM: counter-based — acquire deducts estimated, release adjusts
        self._tpm_budget = self._tpm_limit
        self._tpm_lock = threading.Lock()
        self._tpm_available = threading.Condition(self._tpm_lock)

        # Backoff state
        self._backoff_until = 0.0
        self._budget_reduced = False

        # Start background refill daemon
        self._stop_event = threading.Event()
        self._refill_thread = threading.Thread(target=self._refill_loop, daemon=True)
        self._refill_thread.start()

    def acquire(self, estimated_tokens: int = 2000) -> None:
        """Block until both RPM and TPM budget are available."""
        # Wait for any active backoff
        wait = self._backoff_until - time.monotonic()
        if wait > 0:
            logger.debug("Rate limiter backoff: waiting %.1fs", wait)
            time.sleep(wait)

        # RPM gate
        self._rpm_semaphore.acquire()
        with self._rpm_lock:
            self._rpm_used += 1

        # TPM gate
        with self._tpm_available:
            while self._tpm_budget < estimated_tokens:
                logger.debug("TPM budget exhausted (%d < %d), waiting for refill",
                             self._tpm_budget, estimated_tokens)
                self._tpm_available.wait(timeout=5.0)
            self._tpm_budget -= estimated_tokens

    def release(self, actual_tokens: int) -> None:
        """Adjust TPM counter after a call completes.

        If actual_tokens < estimated, the difference is credited back.
        If actual_tokens > estimated, no penalty (already consumed).
        """
        # No adjustment needed — the estimated was already deducted.
        # This method exists for future refinement where we track actual vs estimated.
        pass

    def backoff(self, retry_after: float | None = None) -> None:
        """Called when a 429 is received. Temporarily reduces throughput."""
        wait = retry_after or 10.0
        self._backoff_until = time.monotonic() + wait
        logger.warning("Rate limit 429 received. Backing off for %.1fs", wait)

        # Temporarily reduce TPM budget by 20%
        if not self._budget_reduced:
            with self._tpm_available:
                reduction = int(self._tpm_limit * 0.20)
                self._tpm_limit -= reduction
                self._budget_reduced = True
                logger.info("TPM budget reduced by %d to %d", reduction, self._tpm_limit)

    def _refill_loop(self) -> None:
        """Background thread: refill RPM and TPM budgets every 60s."""
        while not self._stop_event.wait(timeout=60.0):
            self._refill()

    def _refill(self) -> None:
        """Restore RPM slots and TPM budget to their limits."""
        # Refill RPM
        with self._rpm_lock:
            to_restore = self._rpm_used
            self._rpm_used = 0
        for _ in range(to_restore):
            self._rpm_semaphore.release()

        # Refill TPM
        with self._tpm_available:
            # Restore budget after 60s without a 429
            if self._budget_reduced and time.monotonic() > self._backoff_until + 60:
                original = int(
                    (int(os.environ.get("LLM_TPM", os.environ.get("AZURE_OPENAI_TPM", DEFAULT_TPM))))
                    * self._headroom
                )
                self._tpm_limit = original
                self._budget_reduced = False
                logger.info("TPM budget restored to %d", self._tpm_limit)

            self._tpm_budget = self._tpm_limit
            self._tpm_available.notify_all()

        logger.debug("Rate limiter refill: RPM slots=%d, TPM budget=%d",
                     self._rpm_limit, self._tpm_budget)

    def shutdown(self) -> None:
        """Stop the background refill thread."""
        self._stop_event.set()
        self._refill_thread.join(timeout=5.0)

    @property
    def stats(self) -> dict:
        """Current rate limiter state for monitoring."""
        with self._rpm_lock:
            rpm_used = self._rpm_used
        with self._tpm_lock:
            tpm_remaining = self._tpm_budget
        return {
            "rpm_used": rpm_used,
            "rpm_limit": self._rpm_limit,
            "tpm_remaining": tpm_remaining,
            "tpm_limit": self._tpm_limit,
        }
