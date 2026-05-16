"""BuildKit-rootless backend for BuildAgent, drop-in for DockerOps.

Used when the agent runs in a container without privileges (e.g. Modal).

Image storage strategy: each built image is exported to an OCI-layout
directory under $BUILDKIT_IMAGE_STORE. Subsequent run_container calls
build a synthetic Dockerfile (FROM <img>\\nRUN <cmd>) that references the
prior image via `--opt context:<img>=oci-layout://<store>/<img>`.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_ADDR = os.environ.get("BUILDKIT_ADDR", "")

# buildkitd-rootless puts its socket at $XDG_RUNTIME_DIR/buildkit/buildkitd.sock
# but the fallback when XDG_RUNTIME_DIR is unset (or points at a missing dir)
# is /run/buildkit/buildkitd.sock. Probe both.
SOCKET_CANDIDATES = [
    "/run/user/0/buildkit/buildkitd.sock",
    "/run/buildkit/buildkitd.sock",
]
DEFAULT_STORE = os.environ.get(
    "BUILDKIT_IMAGE_STORE", "/tmp/buildagent-images"
)

# Sentinel markers wrapped around smoke command RUN steps so we can
# distinguish "the command itself exited non-zero" from "the buildkit
# step machinery failed (oom, network, …)".
SMOKE_START = "::BUILDAGENT_SMOKE_START::"
SMOKE_END = "::BUILDAGENT_SMOKE_END::"


class BuildKitOps:
    """BuildKit-rootless ops with the same public surface as DockerOps."""

    def __init__(
        self,
        build_semaphore: threading.Semaphore | None = None,
        timeout: int = 600,
    ):
        concurrency = int(os.environ.get("DOCKER_BUILD_CONCURRENCY", "2"))
        self.semaphore = build_semaphore or threading.Semaphore(concurrency)
        self.timeout = timeout
        self.addr = DEFAULT_ADDR
        self.store = Path(DEFAULT_STORE)
        self.store.mkdir(parents=True, exist_ok=True)
        self._daemon_lock = threading.Lock()
        self._ensure_daemon()

    # ── Daemon management ───────────────────────────────────────────────

    def _ensure_daemon(self) -> None:
        """Start buildkitd --rootless if no socket yet, probing both
        common rootless socket paths."""
        with self._daemon_lock:
            candidates = [self.addr.removeprefix("unix://")] if self.addr else []
            candidates += [s for s in SOCKET_CANDIDATES if s not in candidates]

            existing = next((s for s in candidates if Path(s).exists()), None)
            if existing:
                self.addr = f"unix://{existing}"
                return

            log_path = "/tmp/buildkitd.log"
            logger.info("Starting buildkitd --rootless")
            subprocess.Popen(
                ["buildkitd", "--rootless"],
                stdout=open(log_path, "ab"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

            for _ in range(60):
                for sock in candidates:
                    if not Path(sock).exists():
                        continue
                    addr = f"unix://{sock}"
                    r = subprocess.run(
                        ["buildctl", "--addr", addr, "debug", "info"],
                        capture_output=True, text=True, timeout=5,
                    )
                    if r.returncode == 0:
                        self.addr = addr
                        logger.info("buildkitd ready at %s", addr)
                        return
                time.sleep(0.5)

            log_tail = ""
            try:
                log_tail = "\n".join(Path(log_path).read_text().splitlines()[-30:])
            except OSError:
                pass
            raise RuntimeError(
                f"buildkitd failed to start. Probed sockets: {candidates}. "
                f"Log tail:\n{log_tail}"
            )

    # ── build / run / cleanup ───────────────────────────────────────────

    def build(self, context_dir: str, image_name: str) -> tuple[bool, str, int]:
        with self.semaphore:
            return self._do_build(context_dir, image_name)

    def _do_build(self, context_dir: str, image_name: str) -> tuple[bool, str, int]:
        t0 = time.monotonic()
        oci_dir = self.store / image_name
        if oci_dir.exists():
            shutil.rmtree(oci_dir)
        oci_dir.mkdir(parents=True)

        cmd = [
            "buildctl", "--addr", self.addr, "build",
            "--frontend", "dockerfile.v0",
            "--local", f"context={context_dir}",
            "--local", f"dockerfile={context_dir}",
            "--output",
            f"type=oci,dest={oci_dir}.tar,tar=true,name={image_name}",
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout,
            )
            duration = int((time.monotonic() - t0) * 1000)
            if proc.returncode != 0:
                return False, proc.stderr or proc.stdout, duration
            # extract tar into oci_dir for use as oci-layout context
            tar_path = f"{oci_dir}.tar"
            try:
                subprocess.run(
                    ["tar", "-xf", tar_path, "-C", str(oci_dir)],
                    check=True, capture_output=True,
                )
            finally:
                try:
                    os.unlink(tar_path)
                except OSError:
                    pass
            return True, "", duration
        except subprocess.TimeoutExpired:
            duration = int((time.monotonic() - t0) * 1000)
            return False, f"Build timed out after {self.timeout}s", duration

    def run_container(
        self, image_name: str, command: str, timeout: int = 30,
    ) -> tuple[int, str, bool]:
        oci_dir = self.store / image_name
        if not oci_dir.exists():
            return -1, f"Image {image_name} not found in OCI store", False

        ctx = Path(tempfile.mkdtemp(prefix="bk-smoke-"))
        try:
            # Synthetic Dockerfile that runs the smoke command.
            # Single RUN: a non-zero exit fails the build. Sentinel markers
            # in stdout let us slice the smoke output out of the buildkit log.
            # set -e -o pipefail ensures any failed stage in a pipeline
            # (or a missing binary) bubbles up as non-zero exit, instead of
            # silently passing on the LLM's `|| echo FAIL` patterns.
            df = (
                f"FROM base\n"
                f"RUN /bin/sh -c 'set -e -o pipefail; echo {SMOKE_START}; "
                f"{command}; rc=$?; echo {SMOKE_END}; exit $rc'\n"
            )
            (ctx / "Dockerfile").write_text(df)

            cmd = [
                "buildctl", "--addr", self.addr, "build",
                "--frontend", "dockerfile.v0",
                "--local", f"context={ctx}",
                "--local", f"dockerfile={ctx}",
                "--local", f"layout={oci_dir}",
                "--opt", "context:base=oci-layout://layout",
                "--progress", "plain",
                # no --output: we only care about exit code + log
            ]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout,
                )
                # buildctl writes progress to stderr in plain mode
                log = (proc.stderr or "") + (proc.stdout or "")
                output = _extract_smoke_output(log)
                # rc 0 means RUN succeeded; non-zero means RUN exit was non-zero
                # or buildkit itself failed. Pass through the rc.
                return proc.returncode, output[:2000], False
            except subprocess.TimeoutExpired:
                return -1, f"Timed out after {timeout}s", True
        finally:
            shutil.rmtree(ctx, ignore_errors=True)

    def cleanup(self, image_name: str) -> None:
        oci_dir = self.store / image_name
        shutil.rmtree(oci_dir, ignore_errors=True)
        try:
            os.unlink(f"{oci_dir}.tar")
        except OSError:
            pass

    def prune_cache(self, keep_storage_gb: int | None = None) -> None:
        subprocess.run(
            ["buildctl", "--addr", self.addr, "prune"],
            capture_output=True,
        )

    def _prune_cache(self) -> None:
        self.prune_cache()


def _extract_smoke_output(log: str) -> str:
    """Pull the lines between SMOKE_START and SMOKE_END markers out of
    buildkit's verbose plain-progress log. Falls back to a tail of the
    whole log if markers are missing (build failed before they printed)."""
    start = log.find(SMOKE_START)
    end = log.find(SMOKE_END)
    if start != -1 and end != -1 and end > start:
        chunk = log[start + len(SMOKE_START):end]
        # Strip buildkit's "#N <step>" line prefixes when present
        lines = []
        for line in chunk.splitlines():
            line = line.strip()
            if not line:
                continue
            # buildkit prefixes RUN output with e.g. "#7 0.234 actual-line"
            parts = line.split(" ", 2)
            if (
                len(parts) == 3
                and parts[0].startswith("#")
                and parts[1].replace(".", "", 1).isdigit()
            ):
                line = parts[2]
            lines.append(line)
        return "\n".join(lines).strip()
    # No markers — return last chunk of log for debugging
    return "\n".join(log.splitlines()[-30:])
