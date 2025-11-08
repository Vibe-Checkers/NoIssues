#!/usr/bin/env python3
"""
Builder Agent - Dockerized, Hardened, and Fully Logged

Executes build instructions by running terminal commands in an isolated Docker container.
Adds robust logging, environment sanitization, output redaction, container restarts on timeout,
and guardrails for destructive commands. Includes a post-run integrity check to ensure
all steps were logged.

Usage:
    python builder.py <repository_path> <instructions_file> [github_url]

Environment overrides (optional):
    OPENAI_API_KEY=...                    # required
    OPENAI_MODEL=gpt-4o-mini              # default
    BUILDER_USE_DOCKER=1                  # default 1 (on)
    BUILDER_DOCKER_IMAGE=auto             # auto chooses a base image from repo, or set explicit image (e.g., node:20-bullseye)
    BUILDER_DOCKER_NETWORK=bridge         # or "none" to disable egress
    BUILDER_DOCKER_MEMORY=4g              # memory limit
    BUILDER_DOCKER_CPUS=2                 # number of CPUs (integer or float)
    BUILDER_DOCKER_READONLY=0             # 1 enables read-only rootfs (may break some builds)
"""

import os
import io
import re
import json
import errno
import logging
import shutil
import threading
import subprocess
import hashlib
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import docker
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_classic.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)

# =============================================================================
# Global configuration & logging
# =============================================================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("builder")

# Suppress verbose HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Globals set during runtime
BUILD_WORKING_DIRECTORY: Optional[str] = None
BUILD_STATE = None
USE_DOCKER: bool = os.getenv("BUILDER_USE_DOCKER", "1") == "1"
DOCKER_SANDBOX = None

# =============================================================================
# Constants and helpers
# =============================================================================

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SENSITIVE_NAME_PATTERNS = re.compile(r"(SECRET|TOKEN|KEY|PASS|PWD|PASSWORD|CREDENTIAL|COOKIE|SESSION)", re.IGNORECASE)
EXPLICIT_DENY = {
    "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY",
    "GITHUB_TOKEN", "GITLAB_TOKEN",
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
}

DANGEROUS_REGEX = re.compile(
    r"(?:\brm\s+-rf\s+(?:/|--no-preserve-root|\.|~)\b)|"
    r"(?:\bmkfs\b)|(?:\bdd\s+if=)|(?:\bshred\b)|"
    r"(?:\bmount\b)|(?:\bumount\b)|"
    r"(?:\bcurl\b.*\|\s*(?:sh|bash))|(?:\bwget\b.*\|\s*(?:sh|bash))|"
    r"(?:\:\(\)\{\:\|\:\&\}\;\:)",  # fork bomb pattern
    re.IGNORECASE
)

CRITICAL_PATHS = {"/", "/root", "/home", "/Users", "/etc", "/var", "/usr", "C:\\"}

SUPPORTED_STACK_KEYS = ("node", "python", "rust", "go", "java")

STEP_TIMEOUT = int(os.getenv("BUILDER_STEP_TIMEOUT", "900"))

PROVISION_CONTEXT: Dict[str, Any] = {}


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _slug(s: str) -> str:
    s = s.strip().replace("/", "_").replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "", s)[:60] or "cmd"


def _init_run_logs(repo_path: str) -> str:
    root = os.path.join(repo_path, "builder_logs", _now_stamp())
    os.makedirs(root, exist_ok=True)
    os.makedirs(os.path.join(root, "commands"), exist_ok=True)
    return root


def _safe_rmtree(path: str, allowed_base: Optional[str] = None):
    path = os.path.abspath(path)
    normalized_critical = {os.path.abspath(p) for p in CRITICAL_PATHS}
    if path in normalized_critical:
        raise ValueError(f"Refusing to remove critical path: {path}")
    if allowed_base:
        base = os.path.abspath(allowed_base)
        if not (path == base or path.startswith(base + os.sep)):
            raise ValueError(f"Refusing to remove path outside allowed base ({base}): {path}")
    try:
        shutil.rmtree(path)
    except OSError as exc:
        if exc.errno != errno.ENOTEMPTY:
            raise
        # macOS Finder can recreate .DS_Store between rmtree steps; remove leftovers and retry once
        try:
            for entry in os.listdir(path):
                if entry == ".DS_Store":
                    try:
                        os.remove(os.path.join(path, entry))
                    except FileNotFoundError:
                        pass
        except FileNotFoundError:
            return
        shutil.rmtree(path)


def _select_base_image(repo_path: str) -> str:
    override = os.getenv("BUILDER_DOCKER_IMAGE")
    if override and override.lower() != "auto":
        return override
    p = Path(repo_path)
    if (p / "package.json").exists():
        return "node:20-bullseye"
    if (p / "pyproject.toml").exists() or (p / "requirements.txt").exists():
        return "python:3.11-slim"
    if (p / "Cargo.toml").exists():
        return "rust:1-bullseye"
    if (p / "go.mod").exists():
        return "golang:1.22-bullseye"
    if (p / "pom.xml").exists():
        return "maven:3-eclipse-temurin-17"
    if (p / "build.gradle").exists() or (p / "gradlew").exists():
        return "gradle:8-jdk17"
    return "ubuntu:22.04"


def detect_stack(repo_path: str) -> Dict[str, bool]:
    """Return a dictionary describing which language toolchains a repo needs."""
    repo = Path(repo_path)
    stack = {key: False for key in SUPPORTED_STACK_KEYS}

    if (repo / "package.json").exists() or any(repo.glob("*/package.json")):
        stack["node"] = True

    python_markers = [
        "pyproject.toml",
        "requirements.txt",
        "Pipfile",
        "setup.py",
        "setup.cfg",
    ]
    if any((repo / marker).exists() for marker in python_markers):
        stack["python"] = True

    if (repo / "Cargo.toml").exists() or any(repo.glob("*/Cargo.toml")):
        stack["rust"] = True

    if (repo / "go.mod").exists() or any(repo.glob("*/go.mod")):
        stack["go"] = True

    java_markers = ["pom.xml", "build.gradle", "build.gradle.kts", "gradlew"]
    if any((repo / marker).exists() for marker in java_markers):
        stack["java"] = True

    return stack


def _merge_stack(base_stack: Dict[str, bool], extras: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
    """Merge boolean language flags from extras into a copy of base_stack."""
    merged = {key: bool(base_stack.get(key)) for key in SUPPORTED_STACK_KEYS}
    if extras:
        for key in SUPPORTED_STACK_KEYS:
            if bool(extras.get(key)):
                merged[key] = True
    return merged


def _normalize_language_flags(source: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """Extract language flags from planner metadata or similar structures."""
    flags = {key: False for key in SUPPORTED_STACK_KEYS}
    if not isinstance(source, dict):
        return flags

    language_section = source.get("languages") if isinstance(source.get("languages"), dict) else None
    candidates = [language_section, source]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for key in SUPPORTED_STACK_KEYS:
            if key in candidate:
                flags[key] = flags[key] or bool(candidate[key])
    return flags


def _dedupe_string_list(values: Optional[Any]) -> List[str]:
    items: List[str] = []
    if isinstance(values, (list, tuple)):
        seen: set[str] = set()
        for value in values:
            text = str(value)
            if text in seen:
                continue
            seen.add(text)
            items.append(text)
    return items


def _extras_from_planner(source: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    extras: Dict[str, Any] = {}
    if not isinstance(source, dict):
        return extras
    for list_key in ("apt_packages", "pip_packages"):
        collected = _dedupe_string_list(source.get(list_key))
        if collected:
            extras[list_key] = collected
    return extras


def generate_provisioning_dockerfile(
    base_image: str,
    stack: Dict[str, bool],
    uid: int,
    gid: int,
    extras: Optional[Dict[str, Any]] = None,
) -> str:
    """Return a Dockerfile string that provisions required toolchains."""
    extras = extras or {}
    merged_stack = _merge_stack(stack, extras)

    base_packages = [
        "git",
        "curl",
        "ca-certificates",
        "build-essential",
        "pkg-config",
        "libssl-dev",
        "python3",
        "python3-pip",
        "python3-venv",
    ]

    if merged_stack["node"]:
        base_packages.extend(["nodejs", "npm"])
    if merged_stack["go"]:
        base_packages.append("golang-go")
    if merged_stack["java"]:
        base_packages.extend(["openjdk-17-jdk", "maven", "gradle"])
    apt_extras = extras.get("apt_packages")
    if isinstance(apt_extras, (list, tuple)):
        base_packages.extend(apt_extras)

    apt_install = " ".join(sorted(dict.fromkeys(base_packages)))

    dockerfile_lines = [
        f"FROM {base_image}",
        "ENV DEBIAN_FRONTEND=noninteractive",
        textwrap.dedent(
            f"""
            RUN set -eux; \
                apt-get update; \
                apt-get install -y --no-install-recommends {apt_install}; \
                rm -rf /var/lib/apt/lists/*
            """.strip()
        ),
        textwrap.dedent(
            f"""
            RUN set -eux; \
                if ! getent group {gid} >/dev/null 2>&1; then \
                    groupadd -g {gid} app; \
                fi; \
                if id -u app >/dev/null 2>&1; then \
                    usermod -u {uid} -g {gid} app; \
                else \
                    useradd -m -u {uid} -g {gid} app; \
                fi; \
                mkdir -p /workspace /home/app/.npm /home/app/.cache/pip /home/app/.cargo /home/app/.rustup /home/app/go; \
                chown -R app:$(id -gn app) /workspace /home/app
            """.strip()
        ),
    ]

    if merged_stack["node"]:
        dockerfile_lines.append(
            textwrap.dedent(
                """
                RUN set -eux; \
                    npm install -g corepack >/dev/null 2>&1 || true; \
                    corepack enable || true
                """.strip()
            )
        )

    if merged_stack["rust"]:
        dockerfile_lines.append(
            textwrap.dedent(
                """
                RUN set -eux; \
                    su app -c "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable"; \
                    su app -c "/home/app/.cargo/bin/rustup component add rustfmt clippy"
                """.strip()
            )
        )

    pip_extras = extras.get("pip_packages")
    if isinstance(pip_extras, (list, tuple)) and pip_extras:
        pkgs = " ".join(pip_extras)
        dockerfile_lines.append(
            textwrap.dedent(
                f"""
                RUN set -eux; \
                    pip3 install --no-cache-dir {pkgs}
                """.strip()
            )
        )

    dockerfile_lines.extend(
        [
            "ENV HOME=/home/app",
            "ENV CARGO_HOME=/home/app/.cargo",
            "ENV RUSTUP_HOME=/home/app/.rustup",
            "ENV GOPATH=/home/app/go",
            "ENV PATH=/home/app/.cargo/bin:/home/app/go/bin:$PATH",
            "USER app",
            "WORKDIR /workspace",
        ]
    )

    return "\n".join(dockerfile_lines) + "\n"


def build_provisioned_image(
    repo_path: str,
    base_image: str,
    stack: Dict[str, bool],
    extras: Optional[Dict[str, Any]] = None,
) -> str:
    """Build (or rebuild) the provisioned Docker image and return its tag."""
    repo = Path(repo_path)
    ctx_dir = repo / ".builder_ctx"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    stack = {key: bool(stack.get(key)) for key in SUPPORTED_STACK_KEYS}
    extras = extras or {}

    try:
        uid = os.getuid()
        gid = os.getgid()
    except AttributeError:
        uid = 1000
        gid = 1000

    dockerfile_content = generate_provisioning_dockerfile(base_image, stack, uid, gid, extras)
    dockerfile_path = ctx_dir / "Dockerfile"
    dockerfile_path.write_text(dockerfile_content, encoding="utf-8")

    # Compose a deterministic tag based on base image + stack + extras + uid/gid to enable caching
    tag_payload = json.dumps(
        {
            "base": base_image,
            "stack": stack,
            "extras": extras or {},
            "uid": uid,
            "gid": gid,
        },
        sort_keys=True,
    )
    digest = hashlib.sha256(tag_payload.encode("utf-8")).hexdigest()[:12]
    tag = f"builder-provisioned:{digest}"

    client = docker.from_env()
    try:
        client.images.pull(base_image)
    except Exception as exc:  # pragma: no cover - pull failures are non-fatal
        logger.warning(f"Failed to pull base image {base_image}: {exc}")

    try:
        client.images.build(path=str(ctx_dir), tag=tag, rm=True)
    except Exception as exc:
        raise RuntimeError(f"Provisioning image build failed: {exc}")

    return tag

def _allowed_env_base() -> set:
    return {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "SHELL",
        "USER",
        "TMPDIR",
        "TERM",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "GOPROXY",
        "GOSUMDB",
        "NPM_CONFIG_REGISTRY",
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "CARGO_HOME",
        "RUSTUP_HOME",
    }


def make_sanitized_env(base: Dict[str, str]) -> Dict[str, str]:
    """Return a sanitized environment for subprocess/containers (no secrets)."""
    env: Dict[str, str] = {}
    allow = _allowed_env_base()
    for k, v in base.items():
        if k in allow:
            env[k] = v
        elif k in EXPLICIT_DENY or SENSITIVE_NAME_PATTERNS.search(k):
            continue
    extras = ["/usr/local/bin", "/usr/bin", "/bin", "/usr/local/sbin", "/usr/sbin", "/sbin"]
    env["PATH"] = ":".join(dict.fromkeys(env.get("PATH", "").split(":") + extras))
    if BUILD_WORKING_DIRECTORY:
        env.setdefault("HOME", BUILD_WORKING_DIRECTORY)
        env.setdefault("TMPDIR", os.path.join(BUILD_WORKING_DIRECTORY, ".tmp"))
        os.makedirs(env["TMPDIR"], exist_ok=True)
    return env


def _compiled_secret_values(env: Dict[str, str]) -> List[str]:
    vals: List[str] = []
    # From sanitized env (names only may remain, but keep heuristic)
    for k, v in env.items():
        if v and (k in EXPLICIT_DENY or SENSITIVE_NAME_PATTERNS.search(k)):
            vals.append(v)
    # Also add from real process env (explicit deny only)
    for k in EXPLICIT_DENY:
        v = os.environ.get(k)
        if v and v not in vals:
            vals.append(v)
    # Keep only reasonable length strings
    return [s for s in vals if 4 <= len(s) <= 200]


def redact(text: str, secrets: List[str]) -> str:
    out = text
    for s in secrets:
        out = out.replace(s, "****REDACTED****")
    return out


# =============================================================================
# Docker sandbox
# =============================================================================

class DockerSandbox:
    """
    Manages a persistent Docker container to run all build commands safely.
    - Runs as host user (non-root) when possible.
    - Drops capabilities, sets resource limits.
    - Restarts on command timeout to guarantee a clean state.
    """

    def __init__(self, image: str, mount_host_dir: str, workdir: str = "/workspace"):
        self.client = docker.from_env()
        self.image = image
        self.workdir = workdir
        self.mount_host_dir = mount_host_dir
        self.container = None

        # Configurable via env
        self.network_mode = os.getenv("BUILDER_DOCKER_NETWORK", "bridge")
        self.mem_limit = os.getenv("BUILDER_DOCKER_MEMORY", "4g")
        cpus_env = os.getenv("BUILDER_DOCKER_CPUS", "2")
        try:
            # Docker Python SDK expects nano_cpus as int (e.g., 2 CPUs => 2e9)
            self.nano_cpus = int(float(cpus_env) * 1_000_000_000)
        except Exception:
            self.nano_cpus = 2_000_000_000
        self.readonly_root = os.getenv("BUILDER_DOCKER_READONLY", "0") == "1"

    def start(self):
        # Pull image (best effort)
        try:
            self.client.images.pull(self.image)
        except Exception as e:
            logger.warning(f"Image pull warning for {self.image}: {e}")

        # Use host UID:GID to avoid root-owned files on the volume
        uidgid = None
        try:
            uidgid = f"{os.getuid()}:{os.getgid()}"  # Linux/Unix
        except Exception:
            uidgid = None  # On Windows/macOS (no getuid) -> fall back to default

        cache_root = os.path.join(self.mount_host_dir, ".builder_caches")
        cache_map = {
            os.path.join(cache_root, "npm"): "/home/app/.npm",
            os.path.join(cache_root, "pip"): "/home/app/.cache/pip",
            os.path.join(cache_root, "cargo"): "/home/app/.cargo",
            os.path.join(cache_root, "rustup"): "/home/app/.rustup",
            os.path.join(cache_root, "go"): "/home/app/go",
        }
        try:
            os.makedirs(cache_root, exist_ok=True)
            for host_path in cache_map:
                os.makedirs(host_path, exist_ok=True)
        except OSError as exc:
            logger.warning(f"Failed to prepare cache directories: {exc}")

        volumes: Dict[str, Dict[str, str]] = {
            self.mount_host_dir: {"bind": self.workdir, "mode": "rw"}
        }
        for host_path, container_path in cache_map.items():
            volumes[host_path] = {"bind": container_path, "mode": "rw"}

        kwargs = {
            "image": self.image,
            "command": "sleep infinity",
            "detach": True,
            "working_dir": self.workdir,
            "volumes": volumes,
            "network_mode": self.network_mode,
            "mem_limit": self.mem_limit,
            "nano_cpus": self.nano_cpus,
            "tty": False,
            "security_opt": ["no-new-privileges"],
            "cap_drop": ["ALL"],
            "read_only": self.readonly_root,
            "environment": {"HOME": "/home/app"},
            "user": "app",
        }
        if uidgid:
            kwargs["user"] = uidgid
        if self.readonly_root:
            kwargs["tmpfs"] = {"/tmp": "", "/run": ""}

        self.container = self.client.containers.run(**kwargs)
        logger.info(f"Docker started: {self.container.short_id} ({self.image}) -> {self.mount_host_dir}:/workspace")

    def stop(self):
        if self.container:
            try:
                self.container.kill()
            except Exception:
                pass
            try:
                self.container.remove(force=True)
            except Exception:
                pass
            logger.info("Docker sandbox stopped")
            self.container = None

    def exec(self, cmd: str, timeout: int = STEP_TIMEOUT, environment: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
        """Run a shell command inside the container with a timeout; auto-restart on timeout."""
        assert self.container, "Sandbox not started"
        full_cmd = f"bash -lc {repr(cmd)}"
        result_buf = io.StringIO()
        exit_code: Optional[int] = None
        done = threading.Event()

        def run():
            nonlocal exit_code
            try:
                exec_result = self.container.exec_run(
                    full_cmd, stdout=True, stderr=True, demux=True, environment=environment
                )
                # docker SDK returns an object with `exit_code` & `output` when demux=False,
                # with demux=True we get exit_code & (stdout, stderr)
                if hasattr(exec_result, "exit_code"):
                    exit_code = exec_result.exit_code
                    out, err = exec_result.output if hasattr(exec_result, "output") else (b"", b"")
                else:
                    # older SDK tuple shape
                    exit_code, (out, err) = exec_result
                text = ""
                if out:
                    text += out.decode("utf-8", errors="ignore")
                if err:
                    text += err.decode("utf-8", errors="ignore")
                text = ANSI_RE.sub("", text)
                result_buf.write(text)
            finally:
                done.set()

        t = threading.Thread(target=run, daemon=True)
        t.start()
        if not done.wait(timeout):
            # Hard reset container to guarantee no runaway processes
            try:
                self.stop()
            finally:
                try:
                    self.start()
                except Exception as _:
                    pass
            text = result_buf.getvalue() + f"\n[TIMEOUT after {timeout}s - container restarted]"
            return 124, text

        return exit_code or 0, result_buf.getvalue()


# =============================================================================
# Build State (logging & stats)
# =============================================================================

class BuildState:
    """Tracks build state, environment, and logs per-command outputs and indices."""
    
    def __init__(self, working_dir: str, logs_dir: Optional[str] = None):
        self.working_dir = working_dir
        self.logs_dir = logs_dir or _init_run_logs(working_dir)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(os.path.join(self.logs_dir, "commands"), exist_ok=True)
        os.makedirs(os.path.join(self.working_dir, ".tmp"), exist_ok=True)

        # Host base env (sanitized) and user-provided overrides tracked separately
        self.host_env = make_sanitized_env(os.environ.copy())
        self.env_overrides: Dict[str, str] = {}
        self.start_time = datetime.now()
        self.command_seq = 0
        self.commands_executed: List[Dict[str, Any]] = []
        self.commands_failed: List[Dict[str, Any]] = []

        # Files we maintain:
        self.index_path = os.path.join(self.logs_dir, "commands.jsonl")
        self.trace_path = os.path.join(self.logs_dir, "agent_trace.log")
        self.token_usage_path = os.path.join(self.logs_dir, "token_usage.json")

    def _write(self, path: str, text: str, append: bool = True):
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(text)

    def record(self, command: str, success: bool, output: str, return_code: int):
        """Record a command execution and write per-step logs + index entry."""
        self.command_seq += 1
        record = {
            "seq": self.command_seq,
            "command": command,
            "success": success,
            "return_code": return_code,
            "timestamp": datetime.now().isoformat()
        }
        # Keep aggregated stats in memory
        (self.commands_executed if success else self.commands_failed).append(record)

        # Write full output to a dedicated log file
        cmd_slug = _slug(command)
        step_log_path = os.path.join(self.logs_dir, "commands", f"{self.command_seq:03d}_{cmd_slug}.log")
        with open(step_log_path, "w", encoding="utf-8") as f:
            f.write(output)

        # Append to commands.jsonl index (easy to query later)
        index_entry = {**record, "log_file": step_log_path}
        self._write(self.index_path, json.dumps(index_entry) + "\n", append=True)

    def summary_dict(self) -> Dict[str, Any]:
        duration = (datetime.now() - self.start_time).total_seconds()
        return {
            "working_directory": self.working_dir,
            "duration_seconds": round(duration, 2),
            "commands_succeeded": len(self.commands_executed),
            "commands_failed": len(self.commands_failed),
            "total_commands": len(self.commands_executed) + len(self.commands_failed)
        }

    def get_summary(self) -> str:
        return json.dumps(self.summary_dict(), indent=2)


# =============================================================================
# File-based agent trace handler
# =============================================================================

class FormattedOutputHandler(BaseCallbackHandler):
    """Pretty prints agent Thought/Action/Observation and writes to agent_trace.log + token usage."""

    def __init__(self, logs_dir: Optional[str] = None):
        super().__init__()
        self.logs_dir = logs_dir
        self.token_usage = {"input": 0, "output": 0, "total": 0}

    def _append_trace(self, text: str):
        if not self.logs_dir:
            return
        trace_path = os.path.join(self.logs_dir, "agent_trace.log")
        with open(trace_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def on_agent_action(self, action, **kwargs):
        block = (
            f"\n{'─'*70}\n"
            f"[THOUGHT] {action.log.split('Action:')[0].strip() if 'Action:' in action.log else action.log.strip()}\n\n"
            f"[ACTION] {action.tool}\n"
            f"[INPUT] {action.tool_input}\n"
            f"{'─'*70}"
        )
        print(block)
        self._append_trace(block)

    def on_tool_end(self, output, **kwargs):
        output_str = str(output)
        # Print and log full output (no truncation)
        print(f"\n[OBSERVATION] {output_str}\n")
        self._append_trace(f"\n[OBSERVATION] {output_str}\n")

    def on_chain_error(self, error, **kwargs):
        msg = f"\n[PARSE/CHAIN ERROR] {error}\n"
        print(msg)
        self._append_trace(msg)

    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            if usage:
                self.token_usage["input"] += usage.get('prompt_tokens', 0)
                self.token_usage["output"] += usage.get('completion_tokens', 0)
                self.token_usage["total"] += usage.get('total_tokens', 0)
                if self.logs_dir:
                    with open(os.path.join(self.logs_dir, "token_usage.json"), "w", encoding="utf-8") as f:
                        json.dump(self.token_usage, f, indent=2)


class LoopGuard(BaseCallbackHandler):
    """Stops execution if the agent repeats the same tool + input too many times."""

    def __init__(self, max_repeats: int = 3):
        super().__init__()
        self.max_repeats = max_repeats
        self._recent: List[Tuple[str, str]] = []

    def on_agent_action(self, action, **kwargs):
        tool = getattr(action, "tool", "")
        tool_input = str(getattr(action, "tool_input", "")).strip()
        self._recent.append((tool, tool_input))
        if len(self._recent) > self.max_repeats:
            self._recent.pop(0)
        if (
            len(self._recent) == self.max_repeats
            and len({tuple(item) for item in self._recent}) == 1
        ):
            raise RuntimeError(
                f"Loop detected: repeated {tool} with identical input {self.max_repeats} times"
            )


class LenientReActOutputParser(ReActSingleInputOutputParser):
    """ReAct parser that tolerates tool name decorations like ExecuteCommand(...)."""

    def parse(self, text: str) -> AgentAction | AgentFinish:
        parsed = super().parse(text)
        if isinstance(parsed, AgentAction):
            cleaned = self._clean_tool_name(parsed.tool)
            return AgentAction(cleaned, parsed.tool_input, parsed.log)
        return parsed

    @staticmethod
    def _clean_tool_name(tool: str) -> str:
        candidate = tool.strip()
        for sep in ("(", ":"):
            if sep in candidate:
                candidate = candidate.split(sep, 1)[0].strip()
        return candidate


# =============================================================================
# Tool functions
# =============================================================================

def execute_command(command: str) -> str:
    """
    Execute a shell command in the build working directory (Docker if enabled).

    Returns a JSON string with fields: success, command, rc, output (redacted, truncated for JSON),
    and working_directory.
    """
    global BUILD_STATE, DOCKER_SANDBOX, USE_DOCKER
    
    try:
        working_dir = BUILD_WORKING_DIRECTORY or os.getcwd()
        logger.info(f"Executing command in {working_dir}: {command}")

        if BUILD_STATE:
            combined_env = dict(BUILD_STATE.host_env)
            combined_env.update(BUILD_STATE.env_overrides)
        else:
            combined_env = dict(os.environ)
        
        # Deny obviously dangerous shapes
        if command.strip().startswith("sudo "):
            err = "sudo commands are blocked - cannot run interactive commands that require password"
            logger.warning(err)
            if BUILD_STATE:
                BUILD_STATE.record(command, False, err, -1)
            return json.dumps({"success": False, "command": command, "error": err, "rc": -1}, indent=2)
        if DANGEROUS_REGEX.search(command):
            err = f"Command rejected for safety: {command}"
            logger.warning(err)
            if BUILD_STATE:
                BUILD_STATE.record(command, False, err, -1)
            return json.dumps({"success": False, "command": command, "error": err, "rc": -1}, indent=2)

        # Exec in Docker or locally
        if USE_DOCKER and DOCKER_SANDBOX:
            # Pass only explicit overrides; do not clobber container default PATH
            env_for_container = (BUILD_STATE.env_overrides if BUILD_STATE and BUILD_STATE.env_overrides else None)
            rc, combined = DOCKER_SANDBOX.exec(command, timeout=STEP_TIMEOUT, environment=env_for_container)
            success = (rc == 0)
            output = combined
            result_code = rc
        else:
            # Local fallback (rare): we keep it simple but safe
            try:
                # Use Popen to be able to enforce timeouts and potentially kill group if extended
                proc = subprocess.run(
                    command,
                    shell=True,
                    executable="/bin/bash",
                    cwd=working_dir,
                    capture_output=True,
                    text=True,
                    timeout=STEP_TIMEOUT,
                    env=combined_env,
                )
                success = proc.returncode == 0
                output = proc.stdout + proc.stderr
                result_code = proc.returncode
            except subprocess.TimeoutExpired:
                success = False
                result_code = 124
                output = f"[TIMEOUT after {STEP_TIMEOUT}s] {command}"

        # Redact secrets from outputs before logging
        secrets = _compiled_secret_values(combined_env)
        output_redacted = redact(output, secrets)

        # Truncate extremely large output for the JSON return; keep full in file log
        json_preview = output_redacted
        if len(json_preview) > 50_000:
            json_preview = json_preview[:25_000] + "\n\n... [OUTPUT TRUNCATED] ...\n\n" + json_preview[-25_000:]

        # Record to state (writes per-command file + index)
        if BUILD_STATE:
            BUILD_STATE.record(command, success, output_redacted, result_code)
        
        response = {
            "success": success,
            "command": command,
            "rc": result_code,
            "output": json_preview,
            "working_directory": working_dir
        }
        logger.info(f"Command {'succeeded' if success else 'failed'} with return code {result_code}")
        return json.dumps(response, indent=2)
        
    except Exception as e:
        msg = f"Error executing command: {e}"
        logger.error(msg)
        if BUILD_STATE:
            BUILD_STATE.record(command, False, msg, -1)
        return json.dumps({"success": False, "command": command, "error": str(e), "rc": -1}, indent=2)


def set_environment_variable(input_str: str) -> str:
    """
    Set environment variables for subsequent commands.
    Input format: "KEY=VALUE" or "KEY=VALUE,KEY2=VALUE2"
    """
    global BUILD_STATE
    try:
        if not BUILD_STATE:
            return json.dumps({"success": False, "error": "Build state not initialized"}, indent=2)

        env_pairs_masked: List[Dict[str, str]] = []
        # Parse and set
        for pair in filter(None, [p.strip() for p in input_str.split(",")]):
            if "=" not in pair:
                return json.dumps({"success": False, "error": f"Invalid format: {pair}. Use KEY=VALUE"}, indent=2)
            key, value = pair.split("=", 1)
            key, value = key.strip(), value.strip()
            BUILD_STATE.env_overrides[key] = value
            masked = "****REDACTED****" if SENSITIVE_NAME_PATTERNS.search(key) else value
            env_pairs_masked.append({"key": key, "value": masked})
            logger.info(f"Set environment variable: {key}={masked}")

        # Persist the action log
        if BUILD_STATE.logs_dir:
            try:
                path = os.path.join(BUILD_STATE.logs_dir, f"env-{datetime.now().strftime('%H%M%S')}.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"action": "set_env", "variables": env_pairs_masked, "ts": datetime.now().isoformat()}, f, indent=2)
            except Exception:
                pass

        return json.dumps({"success": True, "variables_set": env_pairs_masked}, indent=2)
        
    except Exception as e:
        logger.error(f"Error setting environment variable: {e}")
        return json.dumps({"success": False, "error": str(e)}, indent=2)


def get_build_status(_: str = "") -> str:
    """Return current build status and summary as a JSON string."""
    global BUILD_STATE
    if not BUILD_STATE:
        return json.dumps({"error": "Build state not initialized"}, indent=2)
    return BUILD_STATE.get_summary()


def provision_packages(spec_json: str) -> str:
    """Rebuild the provisioned image with extra packages or languages and restart the sandbox."""
    global PROVISION_CONTEXT, DOCKER_SANDBOX

    if not USE_DOCKER:
        return json.dumps({"success": False, "error": "Docker provisioning is disabled"}, indent=2)

    if not PROVISION_CONTEXT:
        return json.dumps({"success": False, "error": "Provisioning context unavailable"}, indent=2)

    try:
        spec = json.loads(spec_json) if spec_json else {}
    except Exception as exc:
        return json.dumps({"success": False, "error": f"Invalid JSON: {exc}"}, indent=2)

    if not isinstance(spec, dict):
        return json.dumps({"success": False, "error": "Provision spec must be a JSON object"}, indent=2)

    existing_extras = dict(PROVISION_CONTEXT.get("extras", {}))

    # Merge boolean language flags
    for key in SUPPORTED_STACK_KEYS:
        if key in spec:
            existing_extras[key] = bool(spec[key])

    # Merge optional package lists (pip, apt)
    for list_key in ("apt_packages", "pip_packages"):
        if list_key in spec:
            value = spec[list_key]
            if isinstance(value, (list, tuple)):
                current = list(existing_extras.get(list_key, []))
                current.extend(str(item) for item in value)
                # Remove duplicates while preserving order
                seen = set()
                deduped = []
                for item in current:
                    if item in seen:
                        continue
                    seen.add(item)
                    deduped.append(item)
                existing_extras[list_key] = deduped

    combined_stack = _merge_stack(PROVISION_CONTEXT.get("stack", {}), existing_extras)

    if (
        combined_stack == PROVISION_CONTEXT.get("stack")
        and existing_extras == PROVISION_CONTEXT.get("extras", {})
    ):
        return json.dumps(
            {"success": True, "image": PROVISION_CONTEXT.get("image_tag")},
            indent=2,
        )

    try:
        new_tag = build_provisioned_image(
            PROVISION_CONTEXT["repo_path"],
            PROVISION_CONTEXT["base_image"],
            combined_stack,
            existing_extras,
        )
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)}, indent=2)

    # Restart sandbox with new image
    if DOCKER_SANDBOX:
        try:
            DOCKER_SANDBOX.stop()
        except Exception:
            pass

    DOCKER_SANDBOX = DockerSandbox(
        image=new_tag,
        mount_host_dir=BUILD_WORKING_DIRECTORY,
        workdir="/workspace",
    )
    DOCKER_SANDBOX.start()

    PROVISION_CONTEXT.update(
        {
            "extras": existing_extras,
            "stack": combined_stack,
            "image_tag": new_tag,
        }
    )

    return json.dumps({"success": True, "image": new_tag}, indent=2)


# =============================================================================
# Agent creation
# =============================================================================

def create_builder_agent(
    max_iterations: int = 20,
    verbose: bool = True,
    working_directory: Optional[str] = None,
    use_docker: bool = True,
    docker_image: Optional[str] = None,
    logs_dir: Optional[str] = None,
    planner_context: Optional[Dict[str, Any]] = None,
) -> Tuple[AgentExecutor, Optional[FormattedOutputHandler]]:
    """
    Prepare the builder agent (LLM + tools + docker sandbox + logging).
    Returns (AgentExecutor, FormattedOutputHandler).
    """
    global BUILD_WORKING_DIRECTORY, BUILD_STATE, USE_DOCKER, DOCKER_SANDBOX

    # Set working directory & build state
    BUILD_WORKING_DIRECTORY = os.path.abspath(working_directory or os.getcwd())
    # Initialize logs directory if not provided
    logs_dir = logs_dir or _init_run_logs(BUILD_WORKING_DIRECTORY)
    BUILD_STATE = BuildState(BUILD_WORKING_DIRECTORY, logs_dir=logs_dir)
    logger.info(f"Working directory: {BUILD_WORKING_DIRECTORY}")
    logger.info(f"Logs directory:    {logs_dir}")

    detected_stack = detect_stack(BUILD_WORKING_DIRECTORY)
    planner_flags = _normalize_language_flags(planner_context)
    chosen_image = docker_image or _select_base_image(BUILD_WORKING_DIRECTORY)

    merged_stack = {key: planner_flags.get(key, False) or detected_stack.get(key, False) for key in SUPPORTED_STACK_KEYS}
    planner_extras = _extras_from_planner(planner_context)

    PROVISION_CONTEXT.clear()
    PROVISION_CONTEXT.update(
        {
            "repo_path": BUILD_WORKING_DIRECTORY,
            "base_image": chosen_image,
            "stack": merged_stack,
            "extras": planner_extras,
            "image_tag": chosen_image,
        }
    )

    USE_DOCKER = use_docker
    if USE_DOCKER:
        try:
            provisioned_image = build_provisioned_image(
                BUILD_WORKING_DIRECTORY,
                chosen_image,
                PROVISION_CONTEXT["stack"],
                PROVISION_CONTEXT["extras"],
            )
            PROVISION_CONTEXT["image_tag"] = provisioned_image
            DOCKER_SANDBOX = DockerSandbox(
                image=provisioned_image,
                mount_host_dir=BUILD_WORKING_DIRECTORY,
                workdir="/workspace",
            )
            DOCKER_SANDBOX.start()
            logger.info(
                "Docker sandbox started with image %s, mounting %s -> /workspace",
                provisioned_image,
                BUILD_WORKING_DIRECTORY,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to start Docker sandbox: {e}")
    else:
        DOCKER_SANDBOX = None
        logger.info("Docker disabled; running commands on host (not recommended).")
    
    # LLM
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable. Please check your .env file.")
    llm = ChatOpenAI(model=model, api_key=api_key, temperature=0)

    # Tools
    tools = [
        Tool(
            name="ExecuteCommand",
            func=execute_command,
            description="Execute a shell command in the working directory. Input: command string (e.g., 'npm install', 'cargo build --release', 'make', 'python -m pytest'). Returns JSON with success, rc, and output."
        ),
        Tool(
            name="SetEnvironmentVariable",
            func=set_environment_variable,
            description="Set environment variables for subsequent commands. Input: 'KEY=VALUE' or 'KEY=VALUE,KEY2=VALUE2'. Sensitive keys are redacted from logs."
        ),
        Tool(
            name="GetBuildStatus",
            func=get_build_status,
            description="Get current build summary. No input."
        ),
        Tool(
            name="ProvisionPackages",
            func=provision_packages,
            description="Request additional tooling. Input: JSON, e.g. '{\"rust\": true}' to install languages or lists like 'apt_packages'."
        )
    ]
    
    # Prompt
    template = """You are a hardened build engineer operating inside an isolated Docker sandbox.

CONTEXT:
- The repository is already cloned at {working_directory}. Never run git clone or modify the workspace owner.
- Tooling is provisioned *before* you start. Runtime privilege escalations (sudo, apt-get, curl|bash, etc.) are forbidden.
- Available tools: {tool_names}

{tools}

MANDATORY PREFLIGHT:
- For every relevant toolchain, first run deterministic version checks:
  * Node: `node --version`, `npm --version`
  * Python: `python3 --version`, `pip3 --version`
  * Rust: `cargo --version`, `rustc --version`
  * Go: `go version`
  * Java: `mvn -v` or `./gradlew -v`
- If any command is missing, call ProvisionPackages with JSON, e.g. `{{"rust": true}}` or `{{"apt_packages": ["git-lfs"]}}`.

EXECUTION GUIDELINES:
- Use deterministic build/test commands:
  * Node: `npm ci` then `npm test` (or project scripts)
  * Python: `python3 -m pip install -r requirements.txt` then `python3 -m pytest`
  * Rust: `cargo build --locked` then `cargo test --locked`
  * Go: `go build ./...` then `go test ./...`
  * Java: `mvn -B verify` or `./gradlew test`
- Use SetEnvironmentVariable for configuration (e.g., `SetEnvironmentVariable` with `KEY=VALUE`).
- Never install packages inside the running container beyond these commands; rely on ProvisionPackages.
- Keep actions focused, inspect results, and retry intelligently.

FORMAT RULES (MUST FOLLOW EXACTLY):
1. Your response is either (Thought + Action + Action Input) or (Thought + Final Answer).
2. After "Action:", immediately write "Action Input:" on the next line.
3. The action input must be raw text without quotes.
4. Stop immediately after the action input; the system supplies observations.
5. Do not write "Observation:" yourself.

INVALID EXAMPLE (never do this):
Action: ExecuteCommand(command: "npm ci")
Action Input: npm ci

VALID EXAMPLE:
Action: ExecuteCommand
Action Input: npm ci

Build Instructions (skip any clone/checkout steps):
{input}

Thought:{agent_scratchpad}"""
    prompt = PromptTemplate.from_template(template)
    
    # Agent
    parser = LenientReActOutputParser()
    agent = create_react_agent(llm, tools, prompt, output_parser=parser)
    
    # Callback
    callback_handler = FormattedOutputHandler(logs_dir=logs_dir) if verbose else None
    loop_guard = LoopGuard()
    callbacks = [cb for cb in (callback_handler, loop_guard) if cb]
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        callbacks=callbacks,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
        max_execution_time=None,
        early_stopping_method="force",
        return_intermediate_steps=True
    )
    
    logger.info("Builder agent created successfully")
    return agent_executor, callback_handler


# =============================================================================
# Logging integrity check
# =============================================================================

def verify_logging_integrity(logs_dir: str, expected_seq: int) -> Dict[str, Any]:
    """
    Verify that:
      - commands.jsonl has 'expected_seq' lines,
      - each index entry's log_file exists,
      - agent_trace.log exists and is non-empty,
      - token_usage.json exists,
      - summary files are present.
    Returns a dict with results.
    """
    results = {
        "ok": True,
        "issues": [],
        "counts": {}
    }

    idx_path = os.path.join(logs_dir, "commands.jsonl")
    trace_path = os.path.join(logs_dir, "agent_trace.log")
    token_path = os.path.join(logs_dir, "token_usage.json")
    summary_path = os.path.join(logs_dir, "build_summary.json")
    execres_path = os.path.join(logs_dir, "execution_result.json")

    # commands.jsonl check
    idx_lines: List[str] = []
    if os.path.exists(idx_path):
        with open(idx_path, "r", encoding="utf-8") as f:
            idx_lines = [ln for ln in f.read().splitlines() if ln.strip()]
        results["counts"]["commands_indexed"] = len(idx_lines)
        if len(idx_lines) != expected_seq:
            results["ok"] = False
            results["issues"].append(f"commands.jsonl line count {len(idx_lines)} != expected {expected_seq}")
    else:
        results["ok"] = False
        results["issues"].append("Missing commands.jsonl")

    # Log files existence
    missing_logs = 0
    for ln in idx_lines:
        try:
            entry = json.loads(ln)
            log_file = entry.get("log_file")
            if not log_file or not os.path.exists(log_file):
                missing_logs += 1
        except Exception:
            missing_logs += 1
    if missing_logs:
        results["ok"] = False
        results["issues"].append(f"{missing_logs} command log files missing")

    # Trace
    if not (os.path.exists(trace_path) and os.path.getsize(trace_path) > 0):
        results["ok"] = False
        results["issues"].append("agent_trace.log missing or empty")

    # Token usage
    if not os.path.exists(token_path):
        results["issues"].append("token_usage.json missing (not fatal)")

    # Summaries
    if not os.path.exists(summary_path):
        results["issues"].append("build_summary.json missing (not fatal)")
    if not os.path.exists(execres_path):
        results["issues"].append("execution_result.json missing (not fatal)")

    return results


# =============================================================================
# Main execution function
# =============================================================================

def execute_build(
    repository_path: str,
    instructions_file: str,
    github_url: Optional[str] = None,
    max_iterations: int = 20,
    verbose: bool = True,
    use_docker: bool = True,
    docker_image: Optional[str] = None,
    planner_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute build instructions using the builder agent.
    
    Returns a dict with final status, output, build_summary, token_usage, intermediate_steps, paths, and logging_integrity.
    """
    global DOCKER_SANDBOX, USE_DOCKER, BUILD_STATE

    run_logs_dir = None
    result: Optional[Dict[str, Any]] = None

    try:
        # Optionally clone repository into repository_path (safe delete if necessary)
        if github_url:
            logger.info(f"Cloning repository from {github_url} to {repository_path}")
            
            if os.path.exists(repository_path):
                if os.path.isdir(repository_path) and os.listdir(repository_path):
                    logger.warning(f"Directory {repository_path} already exists and is not empty. Removing safely...")
                    _safe_rmtree(repository_path, allowed_base=os.path.dirname(repository_path))
                elif os.path.isfile(repository_path):
                    raise ValueError(f"Path exists but is a file, not a directory: {repository_path}")
            
            parent_dir = os.path.dirname(repository_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            clone_result = subprocess.run(
                ["git", "clone", github_url, repository_path],
                capture_output=True, text=True, timeout=300
            )
            if clone_result.returncode != 0:
                raise ValueError(f"Failed to clone repository: {clone_result.stderr}")
            logger.info(f"Successfully cloned repository to {repository_path}")
        
        # Validate paths
        if not os.path.exists(repository_path):
            raise ValueError(f"Repository path does not exist: {repository_path}")
        if not os.path.isdir(repository_path):
            raise ValueError(f"Repository path is not a directory: {repository_path}")
        if not os.path.exists(instructions_file):
            raise ValueError(f"Instructions file does not exist: {instructions_file}")
        
        # Read instructions
        logger.info(f"Reading instructions from: {instructions_file}")
        with open(instructions_file, "r", encoding="utf-8") as f:
            instructions = f.read()
        if not instructions.strip():
            raise ValueError(f"Instructions file is empty: {instructions_file}")
        
        logger.info(f"Starting build execution in {repository_path}")
        logger.info(f"Instructions length: {len(instructions)} characters")

        # Prepare logs directory per run now (so handler can write there)
        run_logs_dir = _init_run_logs(repository_path)
        
        # Create agent
        agent_executor, callback_handler = create_builder_agent(
            max_iterations=max_iterations,
            verbose=verbose,
            working_directory=repository_path,
            use_docker=use_docker,
            docker_image=docker_image,
            logs_dir=run_logs_dir,
            planner_context=planner_context,
        )
        
        # Execute build
        print("\n" + "=" * 70)
        print("BUILDER AGENT - Starting Execution")
        if github_url:
            print(f"Cloned from: {github_url}")
        print(f"Repository: {repository_path}")
        print(f"Instructions: {instructions_file}")
        print("=" * 70 + "\n")
        
        try:
            result = agent_executor.invoke({
                "input": instructions,
                "working_directory": repository_path
            })
        finally:
            # Ensure Docker sandbox is stopped
            if USE_DOCKER and DOCKER_SANDBOX:
                try:
                    DOCKER_SANDBOX.stop()
                except Exception:
                    pass

        print("\n" + "=" * 70)
        print("BUILDER AGENT - Execution Complete")
        print("=" * 70 + "\n")

        # Build summary
        build_summary = json.loads(get_build_status())  # get_build_status returns JSON string

        # Compute success from actual execution: at least one command ran and none failed
        ran_any = bool(BUILD_STATE and getattr(BUILD_STATE, "command_seq", 0) > 0)
        num_failed = int(BUILD_STATE and len(getattr(BUILD_STATE, "commands_failed", [])) or 0)
        no_failures = num_failed == 0
        success_flag = ran_any and no_failures
        
        # Compile results
        resolved_output = result.get("output", "") if isinstance(result, dict) else ""
        resolved_steps = len(result.get("intermediate_steps", [])) if isinstance(result, dict) else 0

        execution_result = {
            "success": success_flag,
            "output": resolved_output,
            "build_summary": build_summary,
            "token_usage": (callback_handler.token_usage if callback_handler else {}),
            "intermediate_steps": resolved_steps,
            "repository_path": repository_path,
            "instructions_file": instructions_file,
            "github_url": github_url,
            "logs_dir": run_logs_dir
        }

        # Persist summaries
        try:
            with open(os.path.join(run_logs_dir, "build_summary.json"), "w", encoding="utf-8") as f:
                json.dump(build_summary, f, indent=2)
            with open(os.path.join(run_logs_dir, "token_usage.json"), "w", encoding="utf-8") as f:
                json.dump(callback_handler.token_usage if callback_handler else {}, f, indent=2)
            with open(os.path.join(run_logs_dir, "execution_result.json"), "w", encoding="utf-8") as f:
                json.dump(execution_result, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist summary files: {e}")

        # Logging integrity check
        integrity = verify_logging_integrity(run_logs_dir, BUILD_STATE.command_seq if BUILD_STATE else 0)
        try:
            with open(os.path.join(run_logs_dir, "logging_integrity.json"), "w", encoding="utf-8") as f:
                json.dump(integrity, f, indent=2)
        except Exception:
            pass
        execution_result["logging_integrity"] = integrity
        # If integrity failed, mark as unsuccessful
        if not integrity.get("ok", False):
            execution_result["success"] = False

        # If not successful, attach a clear error reason
        if not execution_result["success"]:
            reasons = []
            if not ran_any:
                reasons.append("no commands executed")
            if not no_failures:
                reasons.append(f"{num_failed} command(s) failed")
            if not integrity.get("ok", False):
                reasons.append("logging integrity failed")
            execution_result["error"] = "; ".join(reasons) or "unsuccessful"

        # Print summary
        print("\n[BUILD SUMMARY]")
        print(json.dumps(build_summary, indent=2))
        if callback_handler:
            print(f"\n[TOKEN USAGE]")
            print(f"Input tokens: {callback_handler.token_usage.get('input', 0)}")
            print(f"Output tokens: {callback_handler.token_usage.get('output', 0)}")
            print(f"Total tokens: {callback_handler.token_usage.get('total', 0)}")

        if execution_result.get("success"):
            logger.info("Build execution completed successfully")
        else:
            logger.error(f"Build execution failed: {execution_result.get('error', 'unspecified')}")
        return execution_result
        
    except Exception as e:
        logger.error(f"Build execution failed: {e}")
        # Try to include partial summary if available
        partial_summary = {}
        try:
            partial_summary = json.loads(get_build_status()) if BUILD_STATE else {}
        except Exception:
            partial_summary = {}
        # Attempt to stop sandbox
        try:
            if USE_DOCKER and DOCKER_SANDBOX:
                DOCKER_SANDBOX.stop()
        except Exception:
            pass
        err_result = {
            "success": False,
            "error": str(e),
            "build_summary": partial_summary,
            "repository_path": repository_path if 'repository_path' in locals() else None,
            "instructions_file": instructions_file if 'instructions_file' in locals() else None,
            "logs_dir": run_logs_dir
        }
        # Persist failure
        try:
            if run_logs_dir:
                with open(os.path.join(run_logs_dir, "execution_result.json"), "w", encoding="utf-8") as f:
                    json.dump(err_result, f, indent=2)
        except Exception:
            pass
        return err_result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python builder.py <repository_path> <instructions_file> [github_url]")
        print("\nArguments:")
        print("  repository_path    - Path where repository should be/is located")
        print("  instructions_file  - Path to markdown file with build instructions")
        print("  github_url         - (Optional) GitHub URL to clone before building")
        print("\nExamples:")
        print('  # Build existing repository:')
        print('  python builder.py /path/to/repo /path/to/instructions.md')
        print()
        print('  # Clone and build:')
        print('  python builder.py /path/to/repo /path/to/instructions.md https://github.com/user/repo.git')
        sys.exit(1)
    
    repo_path = sys.argv[1]
    instructions_path = sys.argv[2]
    github_url = sys.argv[3] if len(sys.argv) > 3 else None
    
    result = execute_build(
        repository_path=repo_path,
        instructions_file=instructions_path,
        github_url=github_url,
        verbose=True,
        use_docker=USE_DOCKER,
        docker_image=os.getenv("BUILDER_DOCKER_IMAGE")  # "auto" or explicit
    )
    
    if result.get("success"):
        print("\n✓ Build completed successfully!")
        sys.exit(0)
    else:
        print(f"\n✗ Build failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
