"""Factory for selecting the container-ops backend at runtime.

Set BUILDAGENT_CONTAINER_BACKEND=buildkit to use the rootless BuildKit
backend (for unprivileged sandboxes like Modal). Anything else (default:
"docker") uses the standard Docker daemon backend.
"""

from __future__ import annotations

import os
import threading


def get_ops(build_semaphore: threading.Semaphore | None = None, timeout: int = 600):
    backend = os.environ.get("BUILDAGENT_CONTAINER_BACKEND", "docker").lower()
    if backend == "buildkit":
        from agent.buildkit_ops import BuildKitOps
        return BuildKitOps(build_semaphore=build_semaphore, timeout=timeout)
    from agent.docker_ops import DockerOps
    return DockerOps(build_semaphore=build_semaphore, timeout=timeout)
