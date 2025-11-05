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
import logging
import shutil
import threading
import subprocess
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
    if path in CRITICAL_PATHS or any(path == p or path.startswith(p + os.sep) for p in CRITICAL_PATHS):
        raise ValueError(f"Refusing to remove critical path: {path}")
    if allowed_base:
        base = os.path.abspath(allowed_base)
        if not (path == base or path.startswith(base + os.sep)):
            raise ValueError(f"Refusing to remove path outside allowed base ({base}): {path}")
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


def _allowed_env_base() -> set:
    return {"PATH", "HOME", "LANG", "LC_ALL", "SHELL", "USER", "TMPDIR", "TERM"}


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

        kwargs = {
            "image": self.image,
            "command": "sleep infinity",
            "detach": True,
            "working_dir": self.workdir,
            "volumes": {self.mount_host_dir: {"bind": self.workdir, "mode": "rw"}},
            "network_mode": self.network_mode,
            "mem_limit": self.mem_limit,
            "nano_cpus": self.nano_cpus,
            "tty": False,
            "security_opt": ["no-new-privileges"],
            "cap_drop": ["ALL"],
            "read_only": self.readonly_root,
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

    def exec(self, cmd: str, timeout: int = 600, environment: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
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
            rc, combined = DOCKER_SANDBOX.exec(command, timeout=600, environment=env_for_container)
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
                    timeout=600,
                    env={(BUILD_STATE.host_env if BUILD_STATE else os.environ) | (BUILD_STATE.env_overrides if BUILD_STATE else {})}
                )
                success = proc.returncode == 0
                output = proc.stdout + proc.stderr
                result_code = proc.returncode
            except subprocess.TimeoutExpired:
                success = False
                result_code = 124
                output = f"[TIMEOUT after 600s] {command}"

        # Redact secrets from outputs before logging
        combined_env = {}
        if BUILD_STATE:
            combined_env = (BUILD_STATE.host_env | BUILD_STATE.env_overrides)
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


# =============================================================================
# Agent creation
# =============================================================================

def create_builder_agent(
    max_iterations: int = 20,
    verbose: bool = True,
    working_directory: Optional[str] = None,
    use_docker: bool = True,
    docker_image: Optional[str] = None,
    logs_dir: Optional[str] = None
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

    # Docker
    USE_DOCKER = use_docker
    if USE_DOCKER:
        try:
            chosen_image = docker_image or _select_base_image(BUILD_WORKING_DIRECTORY)
            DOCKER_SANDBOX = DockerSandbox(image=chosen_image, mount_host_dir=BUILD_WORKING_DIRECTORY, workdir="/workspace")
            DOCKER_SANDBOX.start()
            logger.info(f"Docker sandbox started with image {chosen_image}, mounting {BUILD_WORKING_DIRECTORY} -> /workspace")
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
        )
    ]

    # Prompt
    template = """You are a builder agent that executes build instructions to compile and build software projects.

IMPORTANT CONTEXT:
- The repository is ALREADY CLONED at: {working_directory}
- You are working inside the repository - DO NOT clone it again
- Start directly with installation, dependency setup, or build commands

Available tools: {tool_names}

{tools}

BUILD EXECUTION APPROACH:
1. SET ENVIRONMENT: Use SetEnvironmentVariable if build requires specific env vars
2. RUN COMMANDS: Use ExecuteCommand to run build steps (install, compile, test, etc.)
3. HANDLE ERRORS: If a command fails, analyze the error output and try to resolve
4. CHECK STATUS: Use GetBuildStatus to review progress when needed
5. REPORT: Provide a clear Final Answer with the outcome

CRITICAL FORMAT RULES (FOLLOW EXACTLY):
1. After "Action:", write "Action Input:" on the next line
2. After "Action Input:", write the input WITHOUT quotes
3. After "Action Input:", STOP IMMEDIATELY - do not write anything else
4. Do NOT write "Observation:" - the system provides it
5. Each response: EITHER (Thought + Action + Action Input) OR (Thought + Final Answer), NEVER BOTH

COMMAND EXECUTION GUIDELINES:
- The repository is at {working_directory}
- Run commands one at a time and check output
- If 'command not found' (rc=127), try 'which <command>'
- Common issues: missing dependencies, wrong env vars, permissions

Example:
Thought: I should verify Node.js is installed.
Action: ExecuteCommand
Action Input: node --version

Now begin.

Build Instructions (NOTE: Repository is already cloned at {working_directory}, skip any clone/checkout steps):
{input}

Thought:{agent_scratchpad}"""
    prompt = PromptTemplate.from_template(template)

    # Agent
    agent = create_react_agent(llm, tools, prompt)

    # Callback
    callback_handler = FormattedOutputHandler(logs_dir=logs_dir) if verbose else None
    callbacks = [callback_handler] if callback_handler else []

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        callbacks=callbacks,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
        max_execution_time=None,
        early_stopping_method="generate",
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
    docker_image: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute build instructions using the builder agent.

    Returns a dict with final status, output, build_summary, token_usage, intermediate_steps, paths, and logging_integrity.
    """
    global DOCKER_SANDBOX, USE_DOCKER, BUILD_STATE

    run_logs_dir = None

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
            logs_dir=run_logs_dir
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
        execution_result = {
            "success": success_flag,
            "output": result.get("output", ""),
            "build_summary": build_summary,
            "token_usage": (callback_handler.token_usage if callback_handler else {}),
            "intermediate_steps": len(result.get("intermediate_steps", [])),
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
