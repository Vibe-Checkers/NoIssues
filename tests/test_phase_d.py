"""Tests for Phase D: Docker Operations + Disk Monitor.

Docker tests require a running Docker daemon. They are skipped if Docker is unavailable.
Disk monitor tests use mocked shutil.disk_usage.
"""

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.docker_ops import DockerOps
from parallel.disk_monitor import DiskSpaceMonitor


def docker_available() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


skip_no_docker = pytest.mark.skipif(
    not docker_available(), reason="Docker daemon not available"
)


# ═══════════════════════════════════════════════════════
# D1: Docker Operations
# ═══════════════════════════════════════════════════════

class TestDockerOps:
    @pytest.fixture
    def ops(self):
        return DockerOps(build_semaphore=threading.Semaphore(2), timeout=120)

    @pytest.fixture
    def good_context(self, tmp_path):
        """Create a trivial Dockerfile that succeeds."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM alpine:3.19\nRUN echo ok\n")
        return str(tmp_path)

    @skip_no_docker
    def test_build_success(self, ops, good_context):
        success, error, duration = ops.build(good_context, "buildagent-test-ok")
        try:
            assert success is True
            assert error == ""
            assert duration > 0
        finally:
            ops.cleanup("buildagent-test-ok")

    @skip_no_docker
    def test_build_failure(self, ops, tmp_path):
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM alpine:3.19\nRUN exit 1\n")

        success, error, duration = ops.build(str(tmp_path), "buildagent-test-fail")
        try:
            assert success is False
            assert duration > 0
        finally:
            ops.cleanup("buildagent-test-fail")

    @skip_no_docker
    def test_run_container(self, ops, good_context):
        ops.build(good_context, "buildagent-test-run")
        try:
            exit_code, output, timed_out = ops.run_container(
                "buildagent-test-run", "echo hello"
            )
            assert exit_code == 0
            assert "hello" in output
            assert timed_out is False
        finally:
            ops.cleanup("buildagent-test-run")

    @skip_no_docker
    def test_run_container_timeout(self, ops, good_context):
        ops.build(good_context, "buildagent-test-timeout")
        try:
            exit_code, output, timed_out = ops.run_container(
                "buildagent-test-timeout", "sleep 999", timeout=2
            )
            assert timed_out is True
            assert exit_code == -1
        finally:
            ops.cleanup("buildagent-test-timeout")

    @skip_no_docker
    def test_cleanup_removes_image(self, ops, good_context):
        ops.build(good_context, "buildagent-test-cleanup")
        ops.cleanup("buildagent-test-cleanup")

        result = subprocess.run(
            ["docker", "image", "inspect", "buildagent-test-cleanup"],
            capture_output=True,
        )
        assert result.returncode != 0  # image should not exist

    @skip_no_docker
    def test_build_timeout(self, tmp_path):
        """Build that exceeds timeout is killed."""
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM alpine:3.19\nRUN sleep 999\n")

        ops = DockerOps(build_semaphore=threading.Semaphore(1), timeout=5)
        success, error, duration = ops.build(str(tmp_path), "buildagent-test-btimeout")
        try:
            assert success is False
            assert "timed out" in error.lower()
        finally:
            ops.cleanup("buildagent-test-btimeout")

    def test_transient_error_detection(self):
        assert DockerOps._is_transient("error: connection reset by peer")
        assert DockerOps._is_transient("Temporary failure resolving 'archive.ubuntu.com'")
        assert not DockerOps._is_transient("normal build error")

    def test_cache_corruption_detection(self):
        assert DockerOps._is_cache_corrupt("failed to solve: parent snapshot does not exist")
        assert not DockerOps._is_cache_corrupt("normal build error")


# ═══════════════════════════════════════════════════════
# D2: Disk Monitor
# ═══════════════════════════════════════════════════════

class TestDiskMonitor:
    def test_enough_space_passes(self):
        """check_or_wait returns immediately when space is sufficient."""
        monitor = DiskSpaceMonitor(threshold_gb=1.0)

        # Mock sufficient space
        fake_usage = MagicMock()
        fake_usage.free = 50 * 1e9  # 50 GB

        with patch("parallel.disk_monitor.shutil.disk_usage", return_value=fake_usage):
            monitor.check_or_wait()  # should not block

    def test_low_space_blocks_then_unblocks(self):
        """check_or_wait blocks when space is low, unblocks when restored."""
        monitor = DiskSpaceMonitor(threshold_gb=10.0)

        call_count = [0]

        def mock_disk_usage(path):
            call_count[0] += 1
            usage = MagicMock()
            if call_count[0] <= 1:
                usage.free = 2 * 1e9  # 2 GB — below threshold
            else:
                usage.free = 20 * 1e9  # 20 GB — above threshold
            return usage

        with patch("parallel.disk_monitor.shutil.disk_usage", side_effect=mock_disk_usage):
            with patch("parallel.disk_monitor.time.sleep"):  # don't actually sleep
                with patch("parallel.disk_monitor.subprocess.run"):
                    monitor.check_or_wait()

        # Called 3 times: while-check (low), in-loop log (high), while-check again (high)
        assert call_count[0] >= 2

    def test_prune_called_when_low(self):
        """Docker prune commands are called when space is low."""
        monitor = DiskSpaceMonitor(threshold_gb=10.0)

        call_count = [0]

        def mock_disk_usage(path):
            call_count[0] += 1
            usage = MagicMock()
            if call_count[0] <= 1:
                usage.free = 2 * 1e9
            else:
                usage.free = 20 * 1e9
            return usage

        prune_calls = []

        def mock_run(cmd, **kwargs):
            prune_calls.append(cmd)

        with patch("parallel.disk_monitor.shutil.disk_usage", side_effect=mock_disk_usage):
            with patch("parallel.disk_monitor.time.sleep"):
                with patch("parallel.disk_monitor.subprocess.run", side_effect=mock_run):
                    monitor.check_or_wait()

        # Should have called image prune and container prune
        prune_cmds = [" ".join(c) for c in prune_calls]
        assert any("image prune" in c for c in prune_cmds)
        assert any("container prune" in c for c in prune_cmds)

    def test_threshold_from_env(self):
        with patch.dict(os.environ, {"DISK_SPACE_THRESHOLD_GB": "20"}):
            monitor = DiskSpaceMonitor()
            assert monitor.threshold_bytes == int(20 * 1e9)
