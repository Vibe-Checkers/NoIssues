"""Disk space monitor with backpressure for BuildAgent v2.0.

Blocks worker threads when free disk space drops below threshold,
forcing Docker cleanup until space is recovered.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD_GB = 5


class DiskSpaceMonitor:
    """Monitors disk space and blocks workers when below threshold."""

    def __init__(
        self,
        threshold_gb: float | None = None,
        check_path: str = "/",
    ):
        gb = threshold_gb or float(os.environ.get("DISK_SPACE_THRESHOLD_GB", DEFAULT_THRESHOLD_GB))
        self.threshold_bytes = int(gb * 1e9)
        self.check_path = check_path

    def check_or_wait(self) -> None:
        """Block until disk space is above threshold.

        While waiting, runs Docker prune to free space.
        """
        while self._get_free_space() < self.threshold_bytes:
            free_gb = self._get_free_space() / 1e9
            logger.warning(
                "Disk below %.1fGB threshold (%.1fGB free), pruning and waiting...",
                self.threshold_bytes / 1e9, free_gb,
            )
            self._prune()
            time.sleep(30)

    def _get_free_space(self) -> int:
        """Return free bytes on the disk containing check_path."""
        usage = shutil.disk_usage(self.check_path)
        return usage.free

    @staticmethod
    def _prune() -> None:
        """Run Docker prune to reclaim space."""
        subprocess.run(
            ["docker", "image", "prune", "-f"],
            capture_output=True,
        )
        subprocess.run(
            ["docker", "container", "prune", "-f"],
            capture_output=True,
        )
