"""Docker build, container run, and cleanup operations for BuildAgent v2.0.

Uses a semaphore for build concurrency control, with transient error
and cache corruption detection + retry.
"""

from __future__ import annotations

import logging
import os
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

TRANSIENT_PATTERNS = [
    "connection reset by peer",
    "temporary failure resolving",
    "could not resolve host",
    "TLS handshake timeout",
]

CACHE_CORRUPT_PATTERNS = [
    "parent snapshot does not exist",
    "failed to calculate checksum",
    "failed to compute cache key",
]


class DockerOps:
    """Docker build/run/cleanup with concurrency control and retry logic."""

    def __init__(
        self,
        build_semaphore: threading.Semaphore | None = None,
        timeout: int = 600,
    ):
        concurrency = int(os.environ.get("DOCKER_BUILD_CONCURRENCY", "2"))
        self.semaphore = build_semaphore or threading.Semaphore(concurrency)
        self.timeout = timeout

    def build(self, context_dir: str, image_name: str) -> tuple[bool, str, int]:
        """Build a Docker image from context_dir/Dockerfile.

        Returns (success, error_output, duration_ms).
        """
        with self.semaphore:
            return self._do_build(context_dir, image_name)

    def _do_build(self, context_dir: str, image_name: str) -> tuple[bool, str, int]:
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                ["docker", "build", "-t", image_name,
                 "-f", f"{context_dir}/Dockerfile", context_dir],
                capture_output=True, text=True, timeout=self.timeout,
            )
            duration = int((time.monotonic() - t0) * 1000)

            if proc.returncode == 0:
                return True, "", duration

            error = proc.stderr or proc.stdout

            # Transient error → wait + 1 retry
            if self._is_transient(error):
                logger.info("Transient build error detected, retrying after 5s")
                time.sleep(5)
                return self._retry_build(context_dir, image_name)

            # Cache corruption → prune + 1 retry
            if self._is_cache_corrupt(error):
                logger.info("Cache corruption detected, pruning and retrying")
                self._prune_cache()
                return self._retry_build(context_dir, image_name)

            return False, error, duration

        except subprocess.TimeoutExpired:
            duration = int((time.monotonic() - t0) * 1000)
            return False, f"Build timed out after {self.timeout}s", duration

    def _retry_build(self, context_dir: str, image_name: str) -> tuple[bool, str, int]:
        """Single retry attempt."""
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                ["docker", "build", "-t", image_name,
                 "-f", f"{context_dir}/Dockerfile", context_dir],
                capture_output=True, text=True, timeout=self.timeout,
            )
            duration = int((time.monotonic() - t0) * 1000)
            if proc.returncode == 0:
                return True, "", duration
            return False, proc.stderr or proc.stdout, duration
        except subprocess.TimeoutExpired:
            duration = int((time.monotonic() - t0) * 1000)
            return False, f"Build timed out after {self.timeout}s (retry)", duration

    def run_container(
        self, image_name: str, command: str, timeout: int = 30,
    ) -> tuple[int, str, bool]:
        """Run a command in a container.

        Returns (exit_code, output, timed_out).
        """
        try:
            proc = subprocess.run(
                ["docker", "run", "--rm", "--entrypoint", "",
                 image_name, "sh", "-c", command],
                capture_output=True, text=True, timeout=timeout,
            )
            output = (proc.stdout + proc.stderr)[:2000]
            return proc.returncode, output, False
        except subprocess.TimeoutExpired:
            return -1, f"Timed out after {timeout}s", True

    def cleanup(self, image_name: str) -> None:
        """Remove a Docker image."""
        subprocess.run(
            ["docker", "rmi", "-f", image_name],
            capture_output=True,
        )

    def prune_cache(self, keep_storage_gb: int | None = None) -> None:
        """Prune Docker builder cache."""
        keep = keep_storage_gb or int(os.environ.get("DOCKER_KEEP_STORAGE_GB", "10"))
        subprocess.run(
            ["docker", "builder", "prune", f"--keep-storage={keep}GB", "-f"],
            capture_output=True,
        )

    def _prune_cache(self) -> None:
        """Internal prune for cache corruption recovery."""
        subprocess.run(
            ["docker", "builder", "prune", "-f"],
            capture_output=True,
        )

    @staticmethod
    def _is_transient(error: str) -> bool:
        error_lower = error.lower()
        return any(p in error_lower for p in TRANSIENT_PATTERNS)

    @staticmethod
    def _is_cache_corrupt(error: str) -> bool:
        error_lower = error.lower()
        return any(p in error_lower for p in CACHE_CORRUPT_PATTERNS)
