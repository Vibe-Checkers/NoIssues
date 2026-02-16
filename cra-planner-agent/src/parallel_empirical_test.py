#!/usr/bin/env python3
"""
Parallel Empirical Testing Script
Run 4 agents in parallel, test Dockerfiles, and log everything with unified format.
Combines parallel execution with Docker build testing for maximum speed.

MEMORY MANAGEMENT:
- Aggressive cleanup after each test: removes repos, Docker images, truncates large variables
- Emergency cleanup in finally block: ensures cleanup even on exceptions
- Final cleanup: removes any remaining files and images
- Explicit garbage collection after each test
- Safe for long-running tests with 100+ repositories
"""

print("[STARTUP] Parallel empirical test script starting...")
print("[STARTUP] Loading imports...")

import os
import sys
import json
import time
import threading
import traceback
import concurrent.futures
import gc
import shutil
import ssl
import certifi
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse
from dotenv import load_dotenv


# Load environment variables FIRST before any other imports
print("[STARTUP] Loading environment variables...")
# Look for .env in the project root (parent of src/ directory if running from src/)
dotenv_path = Path(__file__).parent.parent / '.env'
if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    print(f"[STARTUP] Loaded .env from: {dotenv_path}")
else:
    # Fallback: try current directory
    load_dotenv()
    print("[STARTUP] Loaded .env from current directory")

# Verify critical environment variables are loaded
endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
if endpoint:
    print(f"[STARTUP] Azure endpoint configured: {endpoint}")
else:
    print("[WARNING] AZURE_OPENAI_ENDPOINT not found in environment!")

print("[STARTUP] Loading agent and testing modules...")

# Support both direct execution (python src/parallel_empirical_test.py)
# and module execution (python -m src.parallel_empirical_test)
# Add src directory to Python path for direct execution
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from .agent.core import _get_host_platform
    from .agent.validation import DockerBuildTester
except ImportError:
    from agent.core import _get_host_platform
    from agent.validation import DockerBuildTester

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

print("[STARTUP] All imports loaded successfully!")

# Token pricing (USD per 1K tokens). Override via env to match your Azure SKU.
# Agent model (gpt-5-nano)
NANO_PROMPT_COST_PER_1K = float(os.getenv("AZURE_GPT5NANO_PROMPT_COST_PER_1K", "0"))
NANO_COMPLETION_COST_PER_1K = float(os.getenv("AZURE_GPT5NANO_COMPLETION_COST_PER_1K", "0"))
# Analysis model (gpt-5) - used for preparation phase
GPT5_PROMPT_COST_PER_1K = float(os.getenv("AZURE_GPT5_PROMPT_COST_PER_1K", "0"))
GPT5_COMPLETION_COST_PER_1K = float(os.getenv("AZURE_GPT5_COMPLETION_COST_PER_1K", "0"))

# Backward compat
PROMPT_COST_PER_1K = NANO_PROMPT_COST_PER_1K
COMPLETION_COST_PER_1K = NANO_COMPLETION_COST_PER_1K


def repo_slug(repo_url: str) -> str:
    """
    Generate a unique, filesystem-safe slug from a repository URL.

    Prevents collisions for forks and same-name repos across different owners.
    Format: owner__repo (e.g., "facebook__react", "myuser__react")

    Args:
        repo_url: GitHub repository URL

    Returns:
        Sanitized slug string safe for filesystem and Docker image names

    Examples:
        >>> repo_slug("https://github.com/facebook/react")
        'facebook__react'
        >>> repo_slug("https://github.com/myuser/react.git")
        'myuser__react'
    """
    try:
        path = urlparse(repo_url).path.strip("/")
        if path.endswith(".git"):
            path = path[:-4]

        # Split path into owner/repo
        parts = path.split("/")
        if len(parts) >= 2:
            owner, repo = parts[0], parts[1]
        else:
            # Fallback: use just the repo name
            owner = "unknown"
            repo = parts[0] if parts else "repo"

        # Create slug with owner__repo format
        slug = f"{owner}__{repo}".lower()

        # Sanitize: allow only alphanumeric, underscore, dash, dot
        slug = re.sub(r"[^a-z0-9_.-]+", "-", slug)

        return slug
    except Exception:
        # Ultimate fallback: hash the URL
        import hashlib
        return f"repo_{hashlib.md5(repo_url.encode()).hexdigest()[:12]}"


def compute_token_cost(token_usage: Dict) -> Dict[str, float]:
    """Calculate cost from token usage using configured per-1K prices (agent/nano model)."""
    prompt = token_usage.get("input", 0)
    completion = token_usage.get("output", 0)
    cost_prompt = (prompt / 1000.0) * NANO_PROMPT_COST_PER_1K
    cost_completion = (completion / 1000.0) * NANO_COMPLETION_COST_PER_1K
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": token_usage.get("total", prompt + completion),
        "prompt_cost_usd": cost_prompt,
        "completion_cost_usd": cost_completion,
        "total_cost_usd": cost_prompt + cost_completion,
    }


def compute_preparation_cost(prep_token_usage: Dict) -> Dict[str, float]:
    """Calculate cost from preparation-phase token usage (GPT-5 model)."""
    prompt = prep_token_usage.get("input", 0)
    completion = prep_token_usage.get("output", 0)
    cost_prompt = (prompt / 1000.0) * GPT5_PROMPT_COST_PER_1K
    cost_completion = (completion / 1000.0) * GPT5_COMPLETION_COST_PER_1K
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prep_token_usage.get("total", prompt + completion),
        "prompt_cost_usd": cost_prompt,
        "completion_cost_usd": cost_completion,
        "total_cost_usd": cost_prompt + cost_completion,
        "calls": prep_token_usage.get("calls", []),
    }


def parse_test_results(output: str) -> dict:
    """
    Deterministically parse test framework output into structured pass/fail/skip counts.
    Handles: Maven/JUnit, Cargo, Jest, Mocha, Go test, pytest, CTest/automake.

    Order matters: most-specific patterns are checked first to avoid false matches.
    E.g. Cargo and Maven must be checked before pytest (r"\d+ passed" matches both).
    """
    counts = {"framework": "unknown", "passed": 0, "failed": 0, "skipped": 0, "errors": 0, "total": 0}

    # Maven Surefire (most distinct): "Tests run: 25, Failures: 0, Errors: 0, Skipped: 0"
    m = re.search(
        r'Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+)',
        output
    )
    if m:
        counts["framework"] = "junit/maven"
        counts["total"] = int(m.group(1))
        counts["failed"] = int(m.group(2))
        counts["errors"] = int(m.group(3))
        counts["skipped"] = int(m.group(4))
        counts["passed"] = counts["total"] - counts["failed"] - counts["errors"] - counts["skipped"]
        return counts

    # Cargo (distinct "test result:" prefix): "test result: ok. 42 passed; 0 failed; 3 ignored"
    m = re.search(r'test result:.*?(\d+) passed;\s*(\d+) failed;\s*(\d+) ignored', output)
    if m:
        counts["framework"] = "cargo"
        counts["passed"] = int(m.group(1))
        counts["failed"] = int(m.group(2))
        counts["skipped"] = int(m.group(3))
        counts["total"] = counts["passed"] + counts["failed"] + counts["skipped"]
        return counts

    # Jest (distinct "Tests:" label at line start): "Tests: 3 failed, 9 passed, 12 total"
    if re.search(r'^Tests:\s+', output, re.MULTILINE):
        counts["framework"] = "jest"
        for key, pattern in [("passed", r'(\d+) passed'), ("failed", r'(\d+) failed'),
                              ("skipped", r'(\d+) skipped'), ("total", r'(\d+) total')]:
            m = re.search(pattern, output)
            if m:
                counts[key] = int(m.group(1))
        return counts

    # Mocha (uses "passing"/"failing", not "passed"/"failed"):
    # "  12 passing (88ms)"  and  "  3 failing"
    m_pass = re.search(r'(\d+) passing', output)
    if m_pass:
        counts["framework"] = "mocha"
        counts["passed"] = int(m_pass.group(1))
        m_fail = re.search(r'(\d+) failing', output)
        if m_fail:
            counts["failed"] = int(m_fail.group(1))
        counts["total"] = counts["passed"] + counts["failed"]
        return counts

    # Go test: lines starting "ok  " or "FAIL  " per package
    go_ok = len(re.findall(r'^ok\s+', output, re.MULTILINE))
    go_fail = len(re.findall(r'^FAIL\s+', output, re.MULTILINE))
    if go_ok + go_fail > 0:
        counts["framework"] = "go test"
        counts["passed"] = go_ok
        counts["failed"] = go_fail
        counts["total"] = go_ok + go_fail
        return counts

    # pytest (generic "\d+ passed"): "5 passed, 2 failed, 1 skipped in 3.22s"
    if re.search(r'\d+ passed', output):
        counts["framework"] = "pytest"
        for key, pattern in [("passed", r'(\d+) passed'), ("failed", r'(\d+) failed'),
                              ("skipped", r'(\d+) skipped'), ("errors", r'(\d+) error')]:
            m = re.search(pattern, output)
            if m:
                counts[key] = int(m.group(1))
        counts["total"] = counts["passed"] + counts["failed"] + counts["skipped"] + counts["errors"]
        return counts

    # CTest summary line: "X tests passed, Y tests failed"
    m = re.search(r'(\d+) tests? passed', output, re.IGNORECASE)
    if m:
        counts["framework"] = "ctest"
        counts["passed"] = int(m.group(1))
        m_fail = re.search(r'(\d+) tests? failed', output, re.IGNORECASE)
        if m_fail:
            counts["failed"] = int(m_fail.group(1))
        counts["total"] = counts["passed"] + counts["failed"]
        return counts

    # automake: count "PASS:" / "FAIL:" lines
    pass_lines = len(re.findall(r'^PASS:', output, re.MULTILINE))
    fail_lines = len(re.findall(r'^FAIL:', output, re.MULTILINE))
    if pass_lines + fail_lines > 0:
        counts["framework"] = "automake"
        counts["passed"] = pass_lines
        counts["failed"] = fail_lines
        counts["total"] = pass_lines + fail_lines
        return counts

    return counts


class LLMFunctionalVerifier:
    """
    DEPRECATED — kept as dead code for reference only.
    Replaced by real test suite execution via run_tests.sh + parse_test_results().
    """
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )

    def generate_verification_command(self, dockerfile_content: str, repo_name: str) -> Tuple[str, Dict]:
        """
        Ask LLM to suggest a safe, non-interactive smoke test command.
        Returns: (command_string, log_data)
        """
        prompt = f"""You are a QA Engineer. Suggest a SINGLE, ONE-LINE shell command to verify that this application container is ACTUALLY FUNCTIONAL.

CONTEXT: This command will run INSIDE the Docker container using "docker run --rm <image> sh -c <YOUR_COMMAND>".
Your job is to provide the command that runs INSIDE the container, NOT to run Docker itself.

REPO NAME: {repo_name}
DOCKERFILE:
{dockerfile_content[:1500]}...

CRITICAL RULES - THE COMMAND MUST ACTUALLY EXECUTE CODE:
1. NEVER suggest "docker run" or any Docker commands - those run on the HOST, not inside the container!
2. NEVER use file existence checks (ls, test -f, [ -e ], stat, find) - these can be faked with 'touch' or empty files
3. MUST actually EXECUTE the application, binary, or import the library to verify it works
4. For Python apps/libraries: Use 'python -c "import mypackage; print(mypackage.__version__)"' NOT 'python --version'
5. For Java: Run 'java -jar /path/to/app.jar --version' or test actual class loading
6. For C/C++ binaries: Execute the actual compiled binary with --version or --help (NOT just check if file exists)
7. For Node.js: Use 'node -e "require(\"mypackage\")"' NOT just 'node --version'
8. For compiled libraries (.so, .a): Try to load them with ldd or language-specific import
9. Command must be non-interactive (no user input)
10. Command must return exit code 0 on success
11. DO NOT try to curl localhost or start servers (container may not have network/curl)
12. Return ONLY the command string, no markdown, no quotes, no explanations

GOOD EXAMPLES (Actually test functionality):
- python -c "import flask; print(flask.__version__)"
- node -e "const express = require('express'); console.log(express.version || 'ok')"
- java -cp /app/lib/* com.example.Main --version
- /usr/local/bin/myapp --version  (executes the actual binary)
- ruby -e "require 'rails'; puts Rails::VERSION::STRING"

BAD EXAMPLES (DO NOT DO THIS):
- ls /usr/local/lib/libwebview.so* (can be faked with 'touch /usr/local/lib/libwebview.so')
- test -f /usr/bin/myapp (can be faked with 'touch /usr/bin/myapp')
- [ -e /usr/local/bin/tool ] && echo ok (can be faked with empty file)
- python --version (only tests Python is installed, not the app)
- java -version (only tests Java is installed, not the app)
- docker run --rm myapp (this runs on HOST, not in container - will fail!)
- curl http://localhost:8080 (container may not have curl or network)
"""
        log_entry = {
            "component": "LLMFunctionalVerifier",
            "method": "generate_verification_command",
            "prompt": prompt,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            log_entry["response"] = response.content
            if hasattr(response, 'response_metadata'):
                log_entry["token_usage"] = response.response_metadata.get("token_usage", {})
            
            return response.content.strip().replace('`', '').strip(), log_entry
        except Exception as e:
            print(f"[Verifier] LLM generation failed: {e}")
            log_entry["error"] = str(e)
            return "true", log_entry # Fallback: always pass

    def verify_output(self, command: str, output: str, exit_code: int) -> Tuple[Dict, Dict]:
        """
        Ask LLM if the command output indicates success.
        Returns: (result_dict, log_data)
        """
        if exit_code != 0:
            return {
                "success": False,
                "reason": f"Command returned exit code {exit_code}",
                "analysis": "Exit code non-zero indicates failure."
            }, {}

        prompt = f"""You are analyzing the output of a smoke test command run inside a Docker container.

COMMAND: {command}
EXIT CODE: {exit_code}
OUTPUT:
{output[:2000]}

Did this command succeed? 
- It succeeded if it printed help text, version info, or expected program output.
- It failed if it printed a traceback, error message, or nothing (if output was expected).

Return strictly JSON:
{{
    "success": true/false,
    "reason": "Short explanation"
}}
"""
        log_entry = {
            "component": "LLMFunctionalVerifier",
            "method": "verify_output",
            "prompt": prompt,
            "timestamp": datetime.now().isoformat()
        }

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            log_entry["response"] = response.content
            if hasattr(response, 'response_metadata'):
                log_entry["token_usage"] = response.response_metadata.get("token_usage", {})
                
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content), log_entry
        except Exception as e:
            log_entry["error"] = str(e)
            return {"success": True, "reason": "LLM verification failed, assuming success based on exit code 0."}, log_entry


class LLMErrorAnalyzer:
    """
    Analyzes build errors using an LLM to provide structured, intelligent feedback.
    Replaces static regex matching with semantic understanding.
    """
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )

    def analyze_error(self, error_log: str, failed_command: str,
                      dockerfile_content: str = "") -> Tuple[Dict, Dict]:
        """
        Analyze a build error and return structured diagnosis.
        Now accepts full Dockerfile content for context-aware fixes.
        Returns: (analysis_dict, log_data)
        """
        system_prompt = """You are a Docker Build Expert and Linux System Administrator.
Your goal is to analyze a failed Docker build log and identify the EXACT valid fix.

You will receive the FULL Dockerfile content and the FULL error log.
Use both to provide precise, line-specific fixes.

Output must be valid JSON with this structure:
{
    "cause": "Concise explanation of what went wrong (e.g. 'Missing compiled extension dependencies')",
    "missing_packages": ["list", "of", "likely", "missing", "linux", "packages"],
    "suggested_fix": "Concrete, single-line command to fix the issue (e.g. 'RUN apt-get update && apt-get install -y libxml2-dev')",
    "dockerfile_fix": "Show the EXACT Dockerfile lines to change — include line numbers and before/after",
    "search_keywords": "Optimized keywords for web search if the fix is uncertain"
}

Rules:
1. If the error is a missing file (e.g. 'requirements.txt not found'), suggest checking file locations.
2. If it's a network error, suggest a retry or check proxies.
3. Be specific with package names (e.g. 'python3-dev' instead of just 'dev headers').
4. Do NOT hallucinate packages. If unsure, assume a web search is needed.
5. When you have the Dockerfile, reference exact lines and show corrected versions.
6. Consider cascading errors — the FIRST error in the log is usually the root cause.
"""

        # Pass full error log — let the LLM see everything
        error_display = error_log
        if len(error_log) > 20000:
            error_display = (
                error_log[:5000]
                + "\n\n... [middle truncated] ...\n\n"
                + error_log[-10000:]
            )

        dockerfile_section = ""
        if dockerfile_content:
            # Add line numbers for precise reference
            numbered_lines = []
            for i, line in enumerate(dockerfile_content.split('\n'), 1):
                numbered_lines.append(f"{i:3d} | {line}")
            dockerfile_section = (
                "\n\nDOCKERFILE (with line numbers):\n"
                + "\n".join(numbered_lines)
            )

        user_prompt = f"""FAILED COMMAND: {failed_command}
{dockerfile_section}

FULL ERROR LOG:
{error_display}

Analyze this error and determine the fix."""

        log_entry = {
            "component": "LLMErrorAnalyzer",
            "method": "analyze_error",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "timestamp": datetime.now().isoformat()
        }

        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            log_entry["response"] = response.content
            if hasattr(response, 'response_metadata'):
                log_entry["token_usage"] = response.response_metadata.get("token_usage", {})
            
            # Parse JSON from content (handle potential markdown fences)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            return json.loads(content), log_entry
        except Exception as e:
            print(f"[LLM Analyzer] Failed to analyze error: {e}")
            log_entry["error"] = str(e)
            return {
                "cause": "LLM Analysis Failed",
                "missing_packages": [],
                "suggested_fix": "Please analyze the error log manually and try to search for the specific error message.",
                "search_keywords": f"docker build error {failed_command}"
            }, log_entry


class LLMDockerfileValidator:
    """
    Uses LLM to evaluate whether a Dockerfile is suitable for the application.
    Provides explanation of what was verified and whether the Dockerfile is appropriate.
    """
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )

    def validate_dockerfile_suitability(self, dockerfile_content: str, repo_name: str,
                                       build_success: bool, smoke_test_passed: bool) -> Tuple[Dict, Dict]:
        """
        Evaluate if the Dockerfile is actually suitable for the application.

        Args:
            dockerfile_content: Full Dockerfile content
            repo_name: Repository name
            build_success: Whether Docker build succeeded
            smoke_test_passed: Whether smoke test passed

        Returns:
            (validation_dict, log_data)
        """
        prompt = f"""You are a Docker expert evaluating whether a Dockerfile is suitable for building a real application.

REPOSITORY: {repo_name}
BUILD STATUS: {'SUCCESS' if build_success else 'FAILED'}
SMOKE TEST: {'PASSED' if smoke_test_passed else 'FAILED'}

DOCKERFILE:
{dockerfile_content}

Your task is to evaluate:
1. Does this Dockerfile actually build the application from source code?
2. Are the dependencies appropriate for this type of project?
3. Does it follow Docker best practices (multi-stage builds, non-root user, etc.)?
4. Is this a legitimate production-worthy Dockerfile or just a minimal container?

Return ONLY valid JSON in this format:
{{
    "is_suitable": true/false,
    "explanation": "2-3 sentences explaining why this Dockerfile is or isn't suitable for the actual application. Mention what was built, what dependencies were installed, and whether it's production-ready.",
    "confidence": "high/medium/low",
    "concerns": ["list", "of", "any", "concerns", "or", "improvements", "needed"]
}}

Be critical but fair. A Dockerfile is suitable if it:
- Actually compiles/builds the application (not just installs language runtime)
- Includes necessary build and runtime dependencies
- Follows reasonable Docker practices
- Creates a functional container

A Dockerfile is NOT suitable if it:
- Only installs the base language without building anything (e.g., just 'FROM python:3.9' with no build steps)
- Missing critical dependencies for the application type
- Has obvious security or architectural flaws
"""

        log_entry = {
            "component": "LLMDockerfileValidator",
            "method": "validate_dockerfile_suitability",
            "prompt": prompt,
            "timestamp": datetime.now().isoformat()
        }

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            log_entry["response"] = response.content
            if hasattr(response, 'response_metadata'):
                log_entry["token_usage"] = response.response_metadata.get("token_usage", {})

            # Parse JSON from response
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content), log_entry
        except Exception as e:
            print(f"[Dockerfile Validator] LLM validation failed: {e}")
            log_entry["error"] = str(e)
            # Default to assuming it's suitable if LLM fails
            return {
                "is_suitable": True,
                "explanation": f"Dockerfile validation could not be completed due to error: {e}. Assuming suitable based on build and smoke test results.",
                "confidence": "low",
                "concerns": ["LLM validation failed"]
            }, log_entry


class ParallelEmpiricalTester:
    """Parallel empirical tester with unified logging."""

    def __init__(self, results_dir: str = "./parallel_empirical_results", max_workers: int = 4):
        """
        Initialize parallel empirical tester.

        Args:
            results_dir: Directory to store all results
            max_workers: Number of parallel workers
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers

        # Create subdirectories with clear organization
        self.repos_dir = self.results_dir / "repositories"
        self.transcripts_dir = self.results_dir / "agent_transcripts"  # Full agent conversation logs
        self.artifacts_dir = self.results_dir / "artifacts"  # Dockerfiles and build artifacts
        self.structured_logs_dir = self.results_dir / "structured_logs"  # JSON structured logs

        self.repos_dir.mkdir(exist_ok=True)
        self.transcripts_dir.mkdir(exist_ok=True)
        self.artifacts_dir.mkdir(exist_ok=True)
        self.structured_logs_dir.mkdir(exist_ok=True)
        
        # Unique cache directory for local buildx cache isolation
        self.caches_dir = self.results_dir / "caches"
        self.caches_dir.mkdir(exist_ok=True)

        self.docker_tester = DockerBuildTester(timeout=1200, serialize_builds=False)

        # Thread-safe console output
        self.console_lock = threading.Lock()

        # Initialize results storage
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_jsonl_file = self.results_dir / f"results_{self.timestamp}.jsonl"  # Append-only
        self.results_json_file = self.results_dir / f"results_{self.timestamp}.json"   # Final snapshot
        self.summary_file = self.results_dir / f"summary_{self.timestamp}.txt"
        self.master_log_file = self.results_dir / f"master_log_{self.timestamp}.txt"
        self.progress_file = self.results_dir / f"progress_{self.timestamp}.txt"  # NEW: Real-time progress

        # Open master log
        self.master_log = open(self.master_log_file, 'w', encoding='utf-8')
        self.master_log_lock = threading.Lock()

        # Progress tracking
        self.progress_lock = threading.Lock()
        self.completed_count = 0
        self.total_count = 0

        self.results = []
        self.results_lock = threading.Lock()

    def log(self, repo_name: str, message: str, to_console: bool = True):
        """Thread-safe logging to master log and console."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_msg = f"[{timestamp}] [{repo_name}] {message}"

        # Write to master log
        with self.master_log_lock:
            self.master_log.write(formatted_msg + '\n')
            self.master_log.flush()

        # Write to console
        if to_console:
            with self.console_lock:
                print(formatted_msg)

    def update_progress(self, repo_name: str, status: str):
        """Update progress file with current status."""
        with self.progress_lock:
            self.completed_count += 1
            progress_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {self.completed_count}/{self.total_count} - {repo_name}: {status}\n"

            try:
                with open(self.progress_file, 'a', encoding='utf-8') as f:
                    f.write(progress_msg)
                    f.flush()
            except Exception as e:
                print(f"[WARNING] Failed to write progress: {e}")

    def save_artifacts(self, slug: str, repo_path: str, result: Dict):
        """Save all artifacts for empirical study reproducibility."""
        try:
            artifact_dir = self.artifacts_dir / slug
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # List of files to save
            files_to_save = [
                ("Dockerfile", "Dockerfile"),
                (".dockerignore", ".dockerignore"),
                ("run_tests.sh", "run_tests.sh"),
                ("test_output.log", "test_output.log"),
            ]

            saved_files = []
            for src_name, dst_name in files_to_save:
                src_path = Path(repo_path) / src_name
                if src_path.exists():
                    dst_path = artifact_dir / dst_name
                    shutil.copy2(src_path, dst_path)
                    saved_files.append(dst_name)
                    self.log(slug, f"Saved {dst_name} to {dst_path}", to_console=False)

            # Save result metadata as JSON (comprehensive for empirical analysis)
            metadata_file = artifact_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "repo_url": result.get("repo_url"),
                    "repo_name": result.get("repo_name"),
                    "repo_slug": result.get("repo_slug"),
                    "detected_language": result.get("detected_language"),
                    "success": result.get("success"),
                    "timestamp": result.get("timestamp"),
                    "total_duration_seconds": result.get("total_duration_seconds"),
                    "token_usage": result.get("token_usage"),
                    "cost": result.get("cost"),
                    "saved_artifacts": saved_files,
                    "dockerfile_test": result.get("dockerfile_test"),
                    "agent_analysis": {
                        "status": result.get("agent_analysis", {}).get("status"),
                        "phase": result.get("agent_analysis", {}).get("phase"),
                        "attempts": result.get("agent_analysis", {}).get("attempts"),
                        "error": result.get("agent_analysis", {}).get("error"),
                    } if result.get("agent_analysis") else None
                }, f, indent=2)

            self.log(slug, f"Saved {len(saved_files)} artifacts + metadata to {artifact_dir}", to_console=False)
            return str(artifact_dir)

        except Exception as e:
            self.log(slug, f"Failed to save artifacts: {e}", to_console=False)
            return None

    def test_single_repository(self, repo_url: str, worker_id: int) -> Dict:
        """
        Test a single repository using the Learner Agent.
        """
        slug = repo_slug(repo_url)
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        result = {
            "repo_url": repo_url,
            "repo_slug": slug,
            "repo_name": repo_name,
            "task_id": worker_id,
            "timestamp": datetime.now().isoformat(),
            "agent_analysis": {},
            "success": False
        }

        try:
            thread_id = threading.get_ident()
            test_start_time = time.time()
            self.log(repo_name, f"[Task {worker_id}] Starting test", to_console=True)

            # Step 1: Clone
            self.log(repo_name, "Cloning repository...", to_console=True)
            clone_start = time.time()
            # Define simple clone wrapper
            def _clone(url, target_base):
                import subprocess
                slug = repo_slug(url)
                target = Path(target_base) / slug
                if target.exists():
                    import shutil
                    shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "clone", "--depth", "1", "--recursive", url, str(target)], check=True, capture_output=True)
                return str(target)

            try:
                repo_path = _clone(repo_url, self.repos_dir)
                clone_duration = time.time() - clone_start
                result["clone"] = {"success": True, "duration": clone_duration}
            except Exception as e:
                result["clone"] = {"success": False, "error": str(e)}
                self.log(repo_name, f"Clone failed: {e}")
                return result

            # Step 2: Run Learner Agent with Validation Callback
            from agent.workflow import run_learner_agent
            from agent.tools import FormattedOutputHandler

            # Setup per-repo logging
            transcript_file = self.transcripts_dir / f"{slug}.log"
            structured_log_file = self.structured_logs_dir / f"{slug}.json"

            def validation_callback(path):
                """Callback for agent to verify its own work via Docker build."""
                dockerfile_path = Path(path) / "Dockerfile"
                if not dockerfile_path.exists():
                    self.log(repo_name, "Validation: Dockerfile not found", to_console=False)
                    return {"success": False, "error": "Dockerfile not found at root of repository"}

                self.log(repo_name, "Validation: Building Docker image...", to_console=False)
                image_name = f"learner-{slug}:latest"
                cache_dir = self.caches_dir / slug
                build_res = self.docker_tester.build_dockerfile(
                    str(dockerfile_path),
                    path,
                    image_name,
                    cache_dir=str(cache_dir)
                )

                if build_res["success"]:
                    self.log(repo_name, "Validation: Build succeeded", to_console=False)
                    self.docker_tester.cleanup_image(image_name)
                    return {"success": True}
                else:
                    self.log(repo_name, f"Validation: Build failed at stage {build_res.get('stage')}", to_console=False)

                    # Check for multi-stage hallucinations
                    multistage_errors = self._validate_multistage_references(dockerfile_path)
                    extra_feedback = ""
                    if multistage_errors['invalid_refs']:
                        extra_feedback = f"\n\nCRITICAL: You are COPYing from stage(s) {multistage_errors['invalid_refs']} which DO NOT EXIST. Defined stages are: {multistage_errors['defined_stages']}. FIX THIS!"

                    # Provide detailed feedback for the refinement loop
                    error_snippet = build_res.get('error_snippet', 'Unknown error')
                    error_stage = build_res.get('stage', 'UNKNOWN')
                    full_error = build_res.get('error_message', '')
                    return {
                        "success": False,
                        "error": f"Stage: {error_stage}\nError: {error_snippet}\nDetails: {full_error}{extra_feedback}"
                    }

            self.log(repo_name, "Running Learner Agent...", to_console=True)
            start_time = time.time()
            
            # Define Verification Tool for the Agent
            from langchain_core.tools import Tool

            def verify_build_tool_func(input_str: str) -> str:
                """
                Verifies the Dockerfile by building it, then runs run_tests.sh if present.

                Flow:
                  1. Build Dockerfile (environment setup)
                  2. If run_tests.sh exists: docker run <image> bash /app/run_tests.sh
                     Parse test output for pass/fail counts.
                  3. If no run_tests.sh: return build success only.
                """
                self.log(repo_name, "[VerifyBuild] Starting Docker build verification", to_console=False)

                dockerfile_path = Path(repo_path) / "Dockerfile"
                run_tests_path = Path(repo_path) / "run_tests.sh"

                # 1. Existence check
                if not dockerfile_path.exists():
                    self.log(repo_name, "[VerifyBuild] Dockerfile not found", to_console=False)
                    return json.dumps({
                        "status": "error",
                        "message": "Dockerfile not found. You must create it first using WriteToFile."
                    }, indent=2)

                # 2. Pre-build: validate multi-stage COPY references
                self.log(repo_name, "[VerifyBuild] Running pre-build validation", to_console=False)
                multistage_errors = self._validate_multistage_references(dockerfile_path)
                if multistage_errors['invalid_refs']:
                    defined = ", ".join(multistage_errors['defined_stages']) or "None"
                    self.log(repo_name, f"[VerifyBuild] Invalid stage references: {multistage_errors['invalid_refs']}", to_console=False)
                    return json.dumps({
                        "status": "failed",
                        "stage": "PRE_BUILD_CHECK",
                        "error_type": "INVALID_STAGE_REFERENCE",
                        "message": (
                            f"CRITICAL: COPYing from stage(s) {multistage_errors['invalid_refs']} "
                            f"which DO NOT EXIST. Defined stages: {defined}. Fix before building."
                        )
                    }, indent=2)

                self.log(repo_name, "[VerifyBuild] Pre-build validation passed", to_console=False)

                verify_image = f"verify-{slug}:latest"

                # 3. Build image
                self.log(repo_name, "[VerifyBuild] Building Docker image", to_console=False)
                cache_dir = self.caches_dir / slug
                build_result = self.docker_tester.build_dockerfile(
                    str(dockerfile_path), repo_path, verify_image, cache_dir=str(cache_dir)
                )

                if not build_result.get("success"):
                    self.log(repo_name, f"[VerifyBuild] ✗ Build FAILED: {build_result.get('stage')}", to_console=False)
                    try:
                        self.docker_tester.cleanup_image(verify_image)
                    except Exception:
                        pass

                    full_error = build_result.get('error_message', '')

                    # Read current Dockerfile for context-aware error analysis
                    current_dockerfile = ""
                    try:
                        if dockerfile_path.exists():
                            current_dockerfile = dockerfile_path.read_text(encoding='utf-8')
                    except Exception:
                        pass

                    analyzer = LLMErrorAnalyzer()
                    analysis, _ = analyzer.analyze_error(
                        full_error,
                        build_result.get('failed_command', ''),
                        dockerfile_content=current_dockerfile
                    )
                    self.log(repo_name, f"[VerifyBuild] Error cause: {analysis.get('cause', 'Unknown')}", to_console=False)

                    # Pass FULL error to agent — no aggressive truncation
                    # Keep first 1000 chars (FROM line / early context) + last 6000 chars (actual error)
                    if len(full_error) > 8000:
                        tail_lines = (
                            full_error[:1000]
                            + "\n\n... [middle truncated — showing first 1000 + last 6000 chars] ...\n\n"
                            + full_error[-6000:]
                        )
                    else:
                        tail_lines = full_error

                    return json.dumps({
                        "status": "failed",
                        "stage": build_result.get('stage', 'UNKNOWN'),
                        "failed_command": build_result.get('failed_command', 'Unknown command'),
                        "error_snippet": build_result.get('error_snippet', ''),
                        "error_analysis": {
                            "cause": analysis.get("cause"),
                            "suggested_fix": analysis.get("suggested_fix"),
                            "missing_packages": analysis.get("missing_packages", []),
                            "dockerfile_fix": analysis.get("dockerfile_fix", ""),
                        },
                        "tail_lines": tail_lines,
                        "search_keywords": analysis.get(
                            "search_keywords",
                            f"docker build error {build_result.get('failed_command', '')}"
                        ),
                    }, indent=2)

                # 4. Build succeeded — run tests if run_tests.sh present
                self.log(repo_name, "[VerifyBuild] ✓ Build succeeded", to_console=False)

                if run_tests_path.exists():
                    self.log(repo_name, "[VerifyBuild] run_tests.sh found — running test suite (timeout=300s)", to_console=False)
                    run_result = self.docker_tester.run_container(
                        verify_image, "bash /app/run_tests.sh", timeout=300
                    )
                    self.docker_tester.cleanup_image(verify_image)

                    raw_output = (run_result.get("output") or "").strip()
                    exit_code = run_result.get("exit_code", -1)
                    test_counts = parse_test_results(raw_output)
                    # Pass much more test output to agent for better diagnosis
                    # Keep first 1000 chars (setup/compilation) + last 7000 chars (actual test results)
                    if len(raw_output) > 10000:
                        display_output = (
                            raw_output[:1000]
                            + "\n\n... [middle truncated — showing first 1000 + last 7000 chars] ...\n\n"
                            + raw_output[-7000:]
                        )
                    else:
                        display_output = raw_output

                    # Save full test output to persistent file on host
                    try:
                        test_output_file = Path(repo_path) / "test_output.log"
                        with open(test_output_file, 'w', encoding='utf-8') as f:
                            f.write(f"# Test run at {datetime.now().isoformat()}\n")
                            f.write(f"# Exit code: {exit_code}\n")
                            f.write(f"# Framework: {test_counts.get('framework', 'unknown')}\n")
                            f.write(f"# Passed: {test_counts.get('passed', 0)}, Failed: {test_counts.get('failed', 0)}\n\n")
                            f.write(raw_output)
                        self.log(repo_name, f"[VerifyBuild] Saved test output to {test_output_file}", to_console=False)
                    except Exception as e:
                        self.log(repo_name, f"[VerifyBuild] Warning: Could not save test output: {e}", to_console=False)

                    self.log(
                        repo_name,
                        f"[VerifyBuild] Test exit={exit_code} counts={test_counts}",
                        to_console=False
                    )

                    if exit_code == 0:
                        return json.dumps({
                            "status": "success",
                            "message": "✓ Build AND test suite passed!",
                            "test_results": test_counts,
                            "test_output": display_output,
                            "STOP": "Task complete. Give your Final Answer now. Do NOT call any more tools.",
                        }, indent=2)
                    else:
                        return json.dumps({
                            "status": "failed",
                            "stage": "TEST_SUITE",
                            "message": "Build succeeded but test suite failed (exit code {}).".format(exit_code),
                            "test_results": test_counts,
                            "test_output": display_output,
                            "NEXT_STEP": (
                                "You MUST call DiagnoseTestFailure BEFORE attempting any fix. "
                                "Pass the test_output above, plus ReadLocalFile('Dockerfile') "
                                "and ReadLocalFile('run_tests.sh') as inputs. "
                                "DiagnoseTestFailure will tell you whether to fix the Dockerfile or run_tests.sh."
                            ),
                            "note": (
                                "Fix missing dependencies or configuration in Dockerfile / run_tests.sh. "
                                "Do NOT modify test files or replace the test command with a no-op."
                            ),
                        }, indent=2)
                else:
                    # No run_tests.sh — build-only, NOT complete
                    self.docker_tester.cleanup_image(verify_image)
                    return json.dumps({
                        "status": "incomplete",
                        "message": (
                            "⚠ Build succeeded but run_tests.sh is MISSING. "
                            "You MUST create run_tests.sh with WriteToFile and then call VerifyBuild again. "
                            "The task is NOT complete until tests pass."
                        ),
                    }, indent=2)

            verify_tool = Tool(
                name="VerifyBuild",
                func=verify_build_tool_func,
                description="Verifies the 'Dockerfile' by running a real Docker build. usage: VerifyBuild('')"
            )

            # RunInContainer tool — cheap diagnostic commands inside the built image
            def run_in_container_func(input_str: str) -> str:
                """
                Run an arbitrary command inside the last-built Docker image.
                Use for quick diagnostics before committing to a full VerifyBuild cycle.
                
                Examples:
                  RunInContainer({"command": "pip list"})
                  RunInContainer({"command": "pytest --co -q"})
                  RunInContainer({"command": "which make && make --version"})
                """
                try:
                    # Parse input
                    if isinstance(input_str, str):
                        try:
                            parsed = json.loads(input_str)
                            command = parsed.get("command", input_str)
                        except json.JSONDecodeError:
                            command = input_str
                    elif isinstance(input_str, dict):
                        command = input_str.get("command", str(input_str))
                    else:
                        command = str(input_str)
                    
                    command = command.strip()
                    if not command:
                        return json.dumps({"status": "error", "message": "Empty command"})
                    
                    self.log(repo_name, f"[RunInContainer] Running: {command[:100]}", to_console=False)
                    
                    # Check if the image exists, build if needed
                    container_image = f"verify-{slug}:latest"
                    dockerfile_path_ric = Path(repo_path) / "Dockerfile"
                    
                    if not dockerfile_path_ric.exists():
                        return json.dumps({
                            "status": "error",
                            "message": "No Dockerfile found. Create it first with WriteToFile."
                        })
                    
                    # Quick build (uses cache if nothing changed)
                    cache_dir_ric = self.caches_dir / slug
                    build_result = self.docker_tester.build_dockerfile(
                        str(dockerfile_path_ric), repo_path, container_image, cache_dir=str(cache_dir_ric)
                    )
                    if not build_result.get("success"):
                        return json.dumps({
                            "status": "error",
                            "message": f"Docker build failed: {build_result.get('error_snippet', 'unknown')}"
                        })
                    
                    # Run command
                    run_result = self.docker_tester.run_container(
                        container_image, command, timeout=60
                    )
                    self.docker_tester.cleanup_image(container_image)
                    
                    output = run_result.get("output", "")
                    exit_code = run_result.get("exit_code", -1)
                    
                    # Pass more output for better diagnosis
                    if len(output) > 8000:
                        output = output[:2000] + "\n\n... (truncated) ...\n\n" + output[-5000:]
                    
                    return json.dumps({
                        "status": "success" if run_result.get("success") else "failed",
                        "exit_code": exit_code,
                        "output": output
                    }, indent=2)
                    
                except Exception as e:
                    return json.dumps({"status": "error", "message": str(e)})
            
            run_in_container_tool = Tool(
                name="RunInContainer",
                func=run_in_container_func,
                description=(
                    "Run a command inside the Docker container for quick diagnostics. "
                    "Usage: RunInContainer({\"command\": \"pip list\"}) — cheaper than VerifyBuild."
                )
            )

            # Prepare callback handler with per-repo logging
            callback_handler = FormattedOutputHandler(log_file=str(transcript_file))

            # Run Learner Agent with Validation Tool
            self.log(repo_name, "Agent execution started", to_console=True)
            agent_result = run_learner_agent(
                repo_path=repo_path,
                repo_name=repo_name,
                repo_url=repo_url,
                max_retries=5,  # 5 attempts with feedback injection
                callback_handler=callback_handler,
                validation_callback=validation_callback,
                extra_tools=[verify_tool, run_in_container_tool]
            )

            duration = time.time() - start_time # Use start_time from original code
            result["agent_analysis"] = agent_result
            result["total_duration"] = duration
            result["detected_language"] = agent_result.get("language", "Unknown")

            # Capture token usage and calculate cost (agent model - nano)
            token_usage = callback_handler.token_usage
            cost_data = compute_token_cost(token_usage)
            result["token_usage"] = token_usage
            result["cost"] = cost_data

            # Capture preparation-phase token usage (analysis model - GPT-5)
            prep_tokens = agent_result.get("preparation_token_usage", {})
            if prep_tokens:
                prep_cost = compute_preparation_cost(prep_tokens)
                result["preparation_token_usage"] = prep_tokens
                result["preparation_cost"] = prep_cost
                # Combined total
                result["total_cost_usd"] = (
                    cost_data["total_cost_usd"] + prep_cost["total_cost_usd"]
                )
                self.log(repo_name,
                    f"Costs - Agent(nano): ${cost_data['total_cost_usd']:.4f} | "
                    f"Prep(GPT-5): ${prep_cost['total_cost_usd']:.4f} | "
                    f"Total: ${result['total_cost_usd']:.4f}",
                    to_console=True
                )
            else:
                result["total_cost_usd"] = cost_data["total_cost_usd"]

            # Save structured agent transcript with all artifacts for empirical analysis
            try:
                transcript = callback_handler.get_transcript()

                # Read generated files for inclusion in structured log
                dockerfile_content = None
                run_tests_content = None
                test_output_content = None

                dockerfile_path = Path(repo_path) / "Dockerfile"
                if dockerfile_path.exists():
                    dockerfile_content = dockerfile_path.read_text(encoding='utf-8')

                run_tests_path = Path(repo_path) / "run_tests.sh"
                if run_tests_path.exists():
                    run_tests_content = run_tests_path.read_text(encoding='utf-8')

                test_output_path = Path(repo_path) / "test_output.log"
                if test_output_path.exists():
                    test_output_content = test_output_path.read_text(encoding='utf-8')

                # Extract test results from transcript if available
                test_results = None
                for step in reversed(transcript):
                    obs = step.get("observation", "")
                    if '"status": "success"' in obs and '"test_results"' in obs:
                        try:
                            obs_data = json.loads(obs)
                            test_results = obs_data.get("test_results")
                            break
                        except:
                            pass

                with open(structured_log_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "repo_url": repo_url,
                        "repo_name": repo_name,
                        "repo_slug": slug,
                        "detected_language": result["detected_language"],
                        "timestamp": result["timestamp"],
                        "total_duration_seconds": duration,
                        "token_usage": token_usage,
                        "cost": cost_data,
                        "final_result": {
                            "status": agent_result["status"],
                            "attempts": agent_result.get("attempts", 0),
                            "phase": agent_result.get("phase"),
                            "error": agent_result.get("error")
                        },
                        "test_results": test_results,
                        "generated_files": {
                            "dockerfile": dockerfile_content,
                            "run_tests_sh": run_tests_content,
                            "test_output": test_output_content
                        },
                        "transcript": transcript
                    }, f, indent=2)
                self.log(repo_name, f"Saved structured transcript to {structured_log_file}", to_console=False)
            except Exception as e:
                self.log(repo_name, f"Failed to save transcript: {e}", to_console=False)

            # Add dockerfile_test structure for compatibility with summary
            # Note: workflow returns attempts as {"build": N, "test": M} dict
            raw_attempts = agent_result.get("attempts", 0)
            if isinstance(raw_attempts, dict):
                total_attempts = sum(raw_attempts.values())
            else:
                total_attempts = int(raw_attempts) if raw_attempts else 0

            result["dockerfile_test"] = {
                "success": agent_result["status"] == "success",
                "attempts": total_attempts,
                "final_iteration": total_attempts if agent_result["status"] == "success" else 0,
                "iterations": []  # Could populate from agent metrics if available
            }

            if agent_result["status"] == "success":
                result["success"] = True
                status_msg = f"SUCCESS! Dockerfile built in {duration:.1f}s (Cost: ${cost_data['total_cost_usd']:.4f})"
                self.log(repo_name, status_msg, to_console=True)

                # Save artifacts
                artifact_path = self.save_artifacts(slug, repo_path, result)
                if artifact_path:
                    result["artifact_path"] = artifact_path

                # Update progress
                self.update_progress(repo_name, "✓ Success")

            else:
                # Populate failure info
                result["dockerfile_test"]["final_result"] = {
                    "stage": "AGENT_FAILURE",
                    "failed_command": "N/A",
                    "error": agent_result.get("error", "Unknown error")
                }
                failure_msg = f"FAILURE: {agent_result.get('error')} (Cost: ${cost_data['total_cost_usd']:.4f})"
                self.log(repo_name, failure_msg, to_console=True)

                # Save artifacts even on failure for debugging
                artifact_path = self.save_artifacts(slug, repo_path, result)
                if artifact_path:
                    result["artifact_path"] = artifact_path

                # Update progress
                self.update_progress(repo_name, "✗ Failed")

        except Exception as e:
            result["exception"] = str(e) # Mark exception for cleanup logic
            self.log(repo_name, f"Test crashed: {e}", to_console=True)
            traceback.print_exc()
        
        finally:
            # Calculate total duration
            result["total_duration_seconds"] = time.time() - test_start_time

            # Always cleanup to prevent disk/memory blowups
            try:
                self._aggressive_cleanup(repo_name, repo_path, result)
            except Exception as e:
                self.log(repo_name, f"Cleanup failed: {e}", to_console=False)

            # Force garbage collection after each test
            gc.collect()

            # Save result
            with self.results_lock:
                self.results.append(result)
                self._save_incremental_results(result)  # Append to JSONL

        return result

    def _get_recommended_action(self, build_result: Dict) -> str:
        """Generate recommended next action based on build failure."""
        stage = build_result.get('stage', '')

        recommendations = {
            'IMAGE_PULL': 'Use SearchDockerError to find correct base image or verify tag exists',
            'PLATFORM_INCOMPATIBLE': 'Add --platform=linux/amd64 to FROM statement',
            'FILE_COPY_MISSING': 'Use ListDirectory to check actual file names, fix COPY paths',
            'DEPENDENCY_BUILD_TOOLS': 'Install build dependencies: apt-get install -y build-essential python3-dev',
            'BUILD_TOOL_MISSING': 'SearchDockerError to find how to install the missing tool',
            'DOCKERFILE_SYNTAX': 'Check Dockerfile syntax - unknown instruction or malformed command',
            'UNKNOWN': 'Use SearchDockerError with error keywords from tail_lines'
        }

        return recommendations.get(stage, 'Use SearchDockerError with error keywords to find solution')

    def _validate_multistage_references(self, dockerfile_path: Path) -> Dict[str, list]:
        """
        Validate that COPY --from=<stage> references actually exist as build stages.

        Common issue: COPY --from=builder when no "FROM ... AS builder" exists,
        causing Docker to try pulling "builder:latest" from registry.

        Args:
            dockerfile_path: Path to Dockerfile

        Returns:
            Dict with 'invalid_refs' (list of missing stage names) and 'defined_stages' (list of actual stages)
        """
        try:
            with open(dockerfile_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all defined stages: FROM ... AS <name>
            # Pattern now handles --platform flags and multi-word base images
            defined_stages = set()
            stage_pattern = r'FROM\s+(?:--platform=[^\s]+\s+)?(.+?)\s+AS\s+([^\s]+)'
            for match in re.finditer(stage_pattern, content, re.IGNORECASE):
                stage_name = match.group(2).strip()
                defined_stages.add(stage_name)

            # Find all COPY --from=<stage> references
            invalid_refs = []
            copy_from_pattern = r'COPY\s+--from=([^\s]+)'
            for match in re.finditer(copy_from_pattern, content, re.IGNORECASE):
                ref = match.group(1).strip()
                # Skip numeric stage indexes (e.g., --from=0)
                if ref.isdigit():
                    continue
                # Check if this stage was defined
                if ref not in defined_stages:
                    invalid_refs.append(ref)

            return {
                'invalid_refs': invalid_refs,
                'defined_stages': list(defined_stages)
            }

        except Exception as e:
            print(f"[WARNING] Could not validate multi-stage references: {e}")
            return {'invalid_refs': [], 'defined_stages': []}

    # def _detect_fake_binaries(self, dockerfile_path: Path) -> Dict[str, any]:
    #     """
    #     Detect if Dockerfile creates fake/dummy executables or empty files to game smoke tests.

    #     Args:
    #         dockerfile_path: Path to Dockerfile

    #     Returns:
    #         Dict with 'violations' (list of suspicious patterns found) and 'is_suspicious' (bool)
    #     """
    #     try:
    #         with open(dockerfile_path, 'r', encoding='utf-8') as f:
    #             content = f.read()

    #         violations = []

    #         # Pattern 1: Creating executables with printf/echo (fake version scripts)
    #         if re.search(r'(printf|echo).*["\'].*version.*["\'].*>.*/(bin|sbin)/', content, re.IGNORECASE):
    #             violations.append("Creating fake executable with hardcoded version output")

    #         # Pattern 2: Using 'touch' to create files in bin, lib, or include directories
    #         touch_patterns = [
    #             (r'touch\s+.*\.(so|a|dylib|dll)', "Creating empty library files"),
    #             (r'touch\s+.*/bin/', "Creating empty executables in bin directory"),
    #             (r'touch\s+.*/sbin/', "Creating empty executables in sbin directory"),
    #         ]
    #         for pattern, desc in touch_patterns:
    #             if re.search(pattern, content, re.IGNORECASE):
    #                 violations.append(desc)

    #         # Pattern 3: Comments explicitly mentioning gaming tests
    #         suspicious_comments = [
    #             (r'#.*fake.*(?:binary|executable|lib)', "Comment mentions 'fake' binary/lib"),
    #             (r'#.*dummy.*(?:binary|executable|lib)', "Comment mentions 'dummy' binary/lib"),
    #             (r'#.*satisfy.*smoke.*test', "Comment about satisfying smoke tests"),
    #             (r'#.*to\s+pass.*test', "Comment about passing tests"),
    #         ]
    #         for pattern, desc in suspicious_comments:
    #             if re.search(pattern, content, re.IGNORECASE):
    #                 violations.append(desc)

    #         # Pattern 4: Creating symbolic links to /dev/null or non-existent files
    #         if re.search(r'ln\s+-s\s+/dev/null', content):
    #             violations.append("Creating symlink to /dev/null")

    #         # Pattern 5: RUN commands that create files in bin/ without actual compilation
    #         # (printf, echo, cat <<EOF to bin paths)
    #         if re.search(r'(cat|printf|echo).*>>?\s*(/usr/local/bin|/usr/bin|/bin)/', content):
    #             violations.append("Writing text directly to bin directory (not from build)")

    #         return {
    #             'violations': violations,
    #             'is_suspicious': len(violations) > 0
    #         }

    #     except Exception as e:
    #         print(f"[WARNING] Could not detect fake binaries: {e}")
    #         return {'violations': [], 'is_suspicious': False}

    def _validate_copy_sources(self, dockerfile_path: Path, repo_path: str) -> Dict[str, list]:
        """
        Validate that COPY sources in Dockerfile actually exist in the repository.

        Common issue: Agent assumes requirements.txt exists but repo uses pyproject.toml.

        Args:
            dockerfile_path: Path to Dockerfile
            repo_path: Path to repository root

        Returns:
            Dict with 'missing' (list of missing files) and 'suggestions' (alternatives found)
        """
        try:
            with open(dockerfile_path, 'r', encoding='utf-8') as f:
                content = f.read()

            repo_path_obj = Path(repo_path)
            missing = []
            suggestions = {}

            # Parse COPY commands (ignore COPY --from=... multi-stage copies)
            # Pattern: COPY <src> <dest> or COPY ["<src>", "<dest>"]
            copy_pattern = r'^COPY\s+(?!--from)([^\s\[]+)'

            for line in content.split('\n'):
                line = line.strip()
                match = re.match(copy_pattern, line, re.IGNORECASE)
                if match:
                    source = match.group(1)
                    # Skip wildcards and context copies
                    if source in ['.', '*', '**'] or '*' in source:
                        continue

                    # Check if source exists
                    source_path = repo_path_obj / source
                    if not source_path.exists():
                        missing.append(source)

                        # Look for alternatives (common substitutions)
                        alternatives = []
                        if 'requirements.txt' in source:
                            for alt in ['pyproject.toml', 'setup.py', 'environment.yml', 'Pipfile']:
                                if (repo_path_obj / alt).exists():
                                    alternatives.append(alt)
                        elif 'package.json' in source:
                            for alt in ['yarn.lock', 'pnpm-lock.yaml']:
                                if (repo_path_obj / alt).exists():
                                    alternatives.append(alt)

                        if alternatives:
                            suggestions[source] = alternatives

            return {'missing': missing, 'suggestions': suggestions}

        except Exception as e:
            print(f"[WARNING] Could not validate COPY sources: {e}")
            return {'missing': [], 'suggestions': {}}


    def _aggressive_cleanup(self, repo_name: str, repo_path: Optional[str], result: Dict):
        """
        Aggressive cleanup to prevent memory issues during parallel execution.
        Cleans up: repository files, Docker images, large variables, and triggers GC.
        """
        self.log(repo_name, "Running cleanup...", to_console=False)

        # 1. Remove cloned repository
        if repo_path and os.path.exists(repo_path):
            try:
                shutil.rmtree(repo_path)
                self.log(repo_name, f"Removed repository at {repo_path}", to_console=False)
            except Exception as e:
                self.log(repo_name, f"Warning: Could not remove repo: {e}", to_console=False)

        # 2. Remove Docker image (even on failure to free disk space)
        if "dockerfile_test" in result:
            # Use slug from result for consistency (match what we create)
            slug = result.get("repo_slug", repo_name.lower())
            image_name = f"learner-{slug}:latest"
            try:
                self.docker_tester.cleanup_image(image_name)
                self.log(repo_name, f"Removed Docker image {image_name}", to_console=False)
            except Exception as e:
                self.log(repo_name, f"Warning: Could not remove Docker image: {e}", to_console=False)

        # 3. Remove local cache directory
        slug = result.get("repo_slug", repo_name.lower())
        cache_dir = self.caches_dir / slug
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                self.log(repo_name, f"Removed local cache at {cache_dir}", to_console=False)
            except Exception as e:
                self.log(repo_name, f"Warning: Could not remove local cache: {e}", to_console=False)

        # 4. Clear large result fields to reduce memory
        # Keep error messages but truncate if too large
        if "dockerfile_test" in result:
            docker_test = result["dockerfile_test"]
            # Handle new iteration-based structure
            if "final_result" in docker_test and docker_test.get("final_result"):
                final_result = docker_test["final_result"]
                if "error_message" in final_result and not docker_test.get("success", False):
                    error_msg = final_result["error_message"]
                    if len(error_msg) > 5000:
                        final_result["error_message"] = error_msg[-5000:]
                        final_result["error_message_truncated"] = True
            # Truncate iteration error messages
            if "iterations" in docker_test:
                for iter_result in docker_test["iterations"]:
                    if "error_snippet" in iter_result and len(iter_result["error_snippet"]) > 500:
                        iter_result["error_snippet"] = iter_result["error_snippet"][:500] + "..."

        # Don't store full Dockerfile content in results (already saved to file)
        if "dockerfile_content" in result:
            # Just store first 500 chars as preview
            content = result["dockerfile_content"]
            if len(content) > 500:
                result["dockerfile_content"] = content[:500] + "\n... [truncated, see Dockerfile file]"
                result["dockerfile_content_truncated"] = True

        # 4. Force garbage collection
        gc.collect()
        self.log(repo_name, "Cleanup complete", to_console=False)

    def _save_incremental_results(self, result: Dict):
        """
        Append single result to JSONL file (O(1) instead of O(n)).
        Already inside results_lock context.

        Args:
            result: The test result to append
        """
        try:
            # Append to JSONL (one line per result, append-only)
            with open(self.results_jsonl_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"[WARNING] Could not append to JSONL results: {e}")

    def run_parallel_tests(self, repo_urls: List[str]) -> Dict:
        """
        Run tests on multiple repositories in parallel.

        Args:
            repo_urls: List of repository URLs to test

        Returns:
            Summary dictionary with overall results
        """
        # Set total count for progress tracking
        self.total_count = len(repo_urls)

        print("="*80)
        print(f"PARALLEL EMPIRICAL TESTING - {len(repo_urls)} repositories")
        print(f"Workers: {self.max_workers}")
        print("="*80)
        print(f"\nResults directory: {self.results_dir.absolute()}")
        print(f"\nLogging Configuration:")
        print(f"  Master Log:        {self.master_log_file}")
        print(f"  Progress Tracker:  {self.progress_file}")
        print(f"  Agent Transcripts: {self.transcripts_dir}/")
        print(f"  Structured Logs:   {self.structured_logs_dir}/")
        print(f"  Artifacts:         {self.artifacts_dir}/")

        # Pre-test infrastructure health check
        print("\n[HEALTH CHECK] Verifying Docker infrastructure...")
        healthy, health_message = self.docker_tester.check_infrastructure_health()

        if not healthy:
            print(f"[HEALTH CHECK] FAILED: {health_message}")
            if "corrupted" in health_message.lower():
                print("[HEALTH CHECK] Attempting to fix corruption with docker builder prune...")
                if self.docker_tester.prune_buildkit_cache():
                    print("[HEALTH CHECK] Prune successful - rechecking infrastructure...")
                    healthy, health_message = self.docker_tester.check_infrastructure_health()
                    if healthy:
                        print(f"[HEALTH CHECK] PASSED: {health_message}")
                    else:
                        print(f"[HEALTH CHECK] Still unhealthy after prune: {health_message}")
                        print("[HEALTH CHECK] WARNING: Proceeding with tests, but failures likely...")
                else:
                    print("[HEALTH CHECK] Prune failed - proceeding anyway...")
        else:
            print(f"[HEALTH CHECK] PASSED: {health_message}")

        print("\nStarting parallel tests...\n")

        overall_start = time.time()

        # Run tests in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_repo = {
                executor.submit(self.test_single_repository, repo_url, worker_id): repo_url
                for worker_id, repo_url in enumerate(repo_urls)
            }

            # Wait for completion
            for future in concurrent.futures.as_completed(future_to_repo):
                repo_url = future_to_repo[future]
                try:
                    future.result()
                except Exception as e:
                    print(f"[ERROR] Future raised exception for {repo_url}: {e}")

        overall_end = time.time()
        overall_duration = overall_end - overall_start

        # Final cleanup
        print("\n[CLEANUP] Running final cleanup...")
        self._final_cleanup()

        # Close master log
        self.master_log.close()

        # Generate summary
        summary = self._generate_summary(overall_duration)

        return summary

    def _final_cleanup(self):
        """Final cleanup after all tests complete."""
        # Remove any remaining repositories
        if self.repos_dir.exists():
            try:
                for item in self.repos_dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item)
                print(f"[CLEANUP] Removed remaining repositories from {self.repos_dir}")
            except Exception as e:
                print(f"[CLEANUP] Warning: Could not clean repos_dir: {e}")

        # Cleanup any dangling Docker images
        try:
            import subprocess
            result = subprocess.run(
                ["docker", "images", "-q", "-f", "reference=learner-*"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.stdout.strip():
                print("[CLEANUP] Removing remaining Docker images...")
                subprocess.run(
                    ["docker", "rmi", "-f"] + result.stdout.strip().split('\n'),
                    capture_output=True,
                    timeout=60
                )
        except Exception as e:
            print(f"[CLEANUP] Warning: Could not cleanup Docker images: {e}")

        # Final garbage collection
        gc.collect()
        print("[CLEANUP] Final cleanup complete")

    def _generate_summary(self, overall_duration: float) -> Dict:
        """Generate and save summary report."""
        print(f"\n{'='*80}")
        print("GENERATING SUMMARY")
        print(f"{'='*80}\n")

        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        failed = total - successful

        # Categorize failures and track iteration statistics
        failure_stages = {}
        iteration_stats = {"succeeded_at_iteration": {}, "refinement_attempts": []}
        
        for result in self.results:
            if "dockerfile_test" in result:
                docker_test = result["dockerfile_test"]
                
                # Track success iterations
                if docker_test.get("success") and "final_iteration" in docker_test:
                    iter_num = docker_test["final_iteration"]
                    iteration_stats["succeeded_at_iteration"][iter_num] = \
                        iteration_stats["succeeded_at_iteration"].get(iter_num, 0) + 1
                
                # Track refinement attempts
                if "iterations" in docker_test:
                    refinement_count = len([i for i in docker_test["iterations"] if i.get("iteration", 0) > 0])
                    if refinement_count > 0:
                        iteration_stats["refinement_attempts"].append(refinement_count)
                
                # Categorize final failure stage
                if not docker_test.get("success"):
                    if "final_result" in docker_test and docker_test.get("final_result"):
                        stage = docker_test["final_result"].get("stage", "UNKNOWN")
                    else:
                        stage = docker_test.get("stage", "UNKNOWN")
                    failure_stages[stage] = failure_stages.get(stage, 0) + 1

        # Calculate statistics
        durations = [r["total_duration_seconds"] for r in self.results]
        avg_duration = sum(durations) / len(durations) if durations else 0
        theoretical_sequential = sum(durations)
        speedup = theoretical_sequential / overall_duration if overall_duration > 0 else 0

        token_totals = []
        cost_totals = []
        for r in self.results:
            # Check root first, then agent_analysis
            if "token_usage" in r:
                 token_totals.append(r["token_usage"].get("total", 0))
            elif "agent_analysis" in r and "token_usage" in r["agent_analysis"]:
                token_totals.append(r["agent_analysis"]["token_usage"].get("total", 0))
            
            # Use combined total_cost_usd (nano + GPT-5) if available
            if "total_cost_usd" in r:
                cost_totals.append(r["total_cost_usd"])
            elif "cost" in r:
                cost_totals.append(r["cost"].get("total_cost_usd", 0.0))
            elif "agent_analysis" in r and "cost_usd" in r["agent_analysis"]:
                cost_totals.append(r["agent_analysis"]["cost_usd"].get("total_cost_usd", 0.0))
        avg_tokens = sum(token_totals) / len(token_totals) if token_totals else 0
        avg_cost = sum(cost_totals) / len(cost_totals) if cost_totals else 0
        total_cost = sum(cost_totals) if cost_totals else 0

        # Build summary dict
        summary = {
            "timestamp": self.timestamp,
            "max_workers": self.max_workers,
            "total_repos": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "overall_duration_seconds": overall_duration,
            "theoretical_sequential_seconds": theoretical_sequential,
            "speedup": speedup,
            "avg_duration_per_repo": avg_duration,
            "avg_tokens_per_repo": avg_tokens,
            "avg_cost_per_repo_usd": avg_cost,
            "total_cost_usd": total_cost,
            "failure_stages": failure_stages,
            "iteration_stats": {
                "succeeded_at_iteration": iteration_stats["succeeded_at_iteration"],
                "avg_refinement_attempts": sum(iteration_stats["refinement_attempts"]) / len(iteration_stats["refinement_attempts"]) if iteration_stats["refinement_attempts"] else 0,
                "total_refinement_attempts": sum(iteration_stats["refinement_attempts"])
            },
            "results_jsonl_file": str(self.results_jsonl_file),
            "results_json_file": str(self.results_json_file),
            "master_log": str(self.master_log_file)
        }

        # Write summary file
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PARALLEL EMPIRICAL TESTING SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write(f"Test Run: {self.timestamp}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Parallel Workers: {self.max_workers}\n\n")

            f.write("OVERALL RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Repositories: {total}\n")
            success_pct = (successful/total*100) if total > 0 else 0
            failed_pct = (failed/total*100) if total > 0 else 0
            f.write(f"Successful: {successful} ({success_pct:.1f}%)\n")
            f.write(f"Failed: {failed} ({failed_pct:.1f}%)\n\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Overall Duration: {overall_duration/60:.2f} minutes\n")
            f.write(f"Theoretical Sequential Time: {theoretical_sequential/60:.2f} minutes\n")
            f.write(f"Speedup: {speedup:.2f}x\n")
            f.write(f"Average Duration per Repo: {avg_duration:.2f} seconds\n")
            f.write(f"Average Token Usage: {avg_tokens:.0f} tokens\n")
            f.write(f"Average Cost per Repo: ${avg_cost:.4f}\n")
            f.write(f"Total Cost: ${total_cost:.4f}\n\n")

            if failure_stages:
                f.write("FAILURE BREAKDOWN BY STAGE\n")
                f.write("-"*80 + "\n")
                for stage, count in sorted(failure_stages.items(), key=lambda x: x[1], reverse=True):
                    # Guard against division by zero if all tests succeed
                    pct = (count / max(failed, 1) * 100)
                    f.write(f"{stage:40s}: {count:3d} ({pct:.1f}%)\n")
                f.write("\n")
            
            if iteration_stats["succeeded_at_iteration"] or iteration_stats["refinement_attempts"]:
                f.write("ITERATION REFINEMENT STATISTICS\n")
                f.write("-"*80 + "\n")
                if iteration_stats["succeeded_at_iteration"]:
                    f.write("Success by iteration:\n")
                    for iter_num in sorted(iteration_stats["succeeded_at_iteration"].keys()):
                        count = iteration_stats["succeeded_at_iteration"][iter_num]
                        iter_label = "Initial" if iter_num == 0 else f"Refinement {iter_num}"
                        f.write(f"  {iter_label:20s}: {count:3d} repositories\n")
                if iteration_stats["refinement_attempts"]:
                    avg_refine = sum(iteration_stats["refinement_attempts"]) / len(iteration_stats["refinement_attempts"])
                    f.write(f"\nAverage refinement attempts: {avg_refine:.1f}\n")
                    f.write(f"Total refinement attempts: {sum(iteration_stats['refinement_attempts'])}\n")
                f.write("\n")

            f.write("DETAILED RESULTS\n")
            f.write("-"*80 + "\n")
            for i, result in enumerate(self.results, 1):
                status = "✓" if result["success"] else "✗"
                f.write(f"\n{i}. {status} {result['repo_name']}\n")
                f.write(f"   URL: {result['repo_url']}\n")
                f.write(f"   Language: {result.get('detected_language', 'Unknown')}\n")
                f.write(f"   Duration: {result['total_duration_seconds']:.2f}s\n")

                if "token_usage" in result:
                    tokens = result["token_usage"]
                    f.write(f"   Tokens: {tokens.get('total', 0):,}\n")
                elif "agent_analysis" in result and "token_usage" in result["agent_analysis"]:
                    tokens = result["agent_analysis"]["token_usage"]
                    f.write(f"   Tokens: {tokens.get('total', 0):,}\n")

                if "cost" in result:
                    cost = result["cost"]
                    f.write(f"   Cost: ${cost.get('total_cost_usd', 0):.4f}\n")
                elif "agent_analysis" in result and "cost_usd" in result["agent_analysis"]:
                    cost = result["agent_analysis"]["cost_usd"]
                    f.write(f"   Cost: ${cost.get('total_cost_usd', 0):.4f}\n")

                if "dockerfile_test" in result:
                    docker = result["dockerfile_test"]
                    if docker.get("success"):
                        final_iter = docker.get("final_iteration", 0)
                        iter_label = "Initial" if final_iter == 0 else f"After {final_iter} refinement(s)"
                        f.write(f"   Succeeded at: {iter_label}\n")
                        if "iterations" in docker:
                            total_iters = len(docker["iterations"])
                            f.write(f"   Total iterations tested: {total_iters}\n")
                    else:
                        if "final_result" in docker and docker.get("final_result"):
                            final_res = docker["final_result"]
                            f.write(f"   Failure Stage: {final_res.get('stage', 'UNKNOWN')}\n")
                            f.write(f"   Failed Command: {final_res.get('failed_command', 'Unknown')}\n")
                        else:
                            f.write(f"   Failure Stage: {docker.get('stage', 'UNKNOWN')}\n")
                            f.write(f"   Failed Command: {docker.get('failed_command', 'Unknown')}\n")
                        if "iterations" in docker:
                            total_iters = len(docker["iterations"])
                            f.write(f"   Refinement attempts: {total_iters - 1}\n")

            f.write("\n" + "="*80 + "\n")

        # Print console summary
        print(f"\n{'='*80}")
        print("PARALLEL EMPIRICAL TESTING SUMMARY")
        print(f"{'='*80}")
        print(f"Total Repositories: {total}")
        success_pct_console = (successful/total*100) if total > 0 else 0
        failed_pct_console = (failed/total*100) if total > 0 else 0
        print(f"Successful: {successful} ({success_pct_console:.1f}%)")
        print(f"Failed: {failed} ({failed_pct_console:.1f}%)")
        print(f"Overall Duration: {overall_duration/60:.2f} minutes")
        print(f"Theoretical Sequential: {theoretical_sequential/60:.2f} minutes")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Average Duration: {avg_duration:.2f} seconds")
        print(f"Average Tokens: {avg_tokens:.0f}")
        print(f"Average Cost: ${avg_cost:.4f} | Total Cost: ${total_cost:.4f}")
        print(f"{'='*80}\n")

        print("Individual Results:")
        print("-"*80)
        for result in self.results:
            status = "✓" if result["success"] else "✗"
            duration = f"{result['total_duration_seconds']:.1f}s"
            iter_info = ""
            
            if "dockerfile_test" in result:
                docker_test = result["dockerfile_test"]
                if docker_test.get("success"):
                    final_iter = docker_test.get("final_iteration", 0)
                    iter_info = f" [iter {final_iter}]" if final_iter > 0 else ""
                elif "final_result" in docker_test and docker_test.get("final_result"):
                    stage = docker_test["final_result"].get("stage", "UNKNOWN")
                    total_iters = len(docker_test.get("iterations", []))
                    iter_info = f" [iter {total_iters-1}] {stage}" if total_iters > 1 else f" {stage}"
                else:
                    stage = docker_test.get("stage", "UNKNOWN")
                    iter_info = f" {stage}"
            
            print(f"{status} {result['repo_name']:30s} | {duration:10s}{iter_info}")
        print("-"*80)
        
        # Print iteration statistics if available
        if iteration_stats["succeeded_at_iteration"]:
            print("\nIteration Success Breakdown:")
            print("-"*80)
            for iter_num in sorted(iteration_stats["succeeded_at_iteration"].keys()):
                count = iteration_stats["succeeded_at_iteration"][iter_num]
                iter_label = "Initial" if iter_num == 0 else f"Refinement {iter_num}"
                print(f"  {iter_label:20s}: {count:3d} repositories")

        # Save final JSON snapshot (pretty-printed for human reading)
        try:
            with open(self.results_json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": self.timestamp,
                    "max_workers": self.max_workers,
                    "total_tested": len(self.results),
                    "summary": summary,
                    "results": self.results
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] Could not save final JSON snapshot: {e}")

        print(f"\nResults saved to: {self.results_dir.absolute()}")
        print(f"Summary: {self.summary_file}")
        print(f"Master log: {self.master_log_file}")
        print(f"Incremental results (JSONL): {self.results_jsonl_file}")
        print(f"Final results (JSON): {self.results_json_file}")

        return summary


def load_repository_list(file_path: str) -> List[str]:
    """Load repository URLs from a file."""
    repos = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                repos.append(line)
    return repos


def main():
    """Main function."""
    print("="*80)
    print("PARALLEL EMPIRICAL TESTING - Planner Agent")
    print("="*80)
    print()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python parallel_empirical_test.py <repo_list_file>")
        print("  python parallel_empirical_test.py <repo_list_file> --workers N")
        print()
        print("Examples:")
        print("  python parallel_empirical_test.py repos.txt")
        print("  python parallel_empirical_test.py repos.txt --workers 8")
        print()
        print("File format (repos.txt):")
        print("  https://github.com/psf/requests")
        print("  https://github.com/pallets/flask")
        print("  # Lines starting with # are ignored")
        sys.exit(1)

    repo_list_file = sys.argv[1]

    # Parse workers argument
    max_workers = 4
    if len(sys.argv) > 2 and sys.argv[2] == "--workers":
        if len(sys.argv) > 3:
            max_workers = int(sys.argv[3])
            # Warn about high concurrency risks
            if max_workers > 8:
                print("\n" + "="*80)
                print("WARNING: High Worker Count Detected")
                print("="*80)
                print(f"You specified {max_workers} workers, which may cause:")
                print("  - Docker BuildKit cache corruption (race conditions)")
                print("  - Overlay2 storage driver issues")
                print("  - 'parent snapshot does not exist' errors")
                print("\nRECOMMENDATION: Use 4-8 workers for stability")
                print("="*80 + "\n")
        else:
            print("ERROR: --workers requires a number")
            sys.exit(1)

    if not os.path.exists(repo_list_file):
        print(f"ERROR: File not found: {repo_list_file}")
        sys.exit(1)

    # Load repositories
    print(f"Loading repository list from: {repo_list_file}")
    repositories = load_repository_list(repo_list_file)
    print(f"Found {len(repositories)} repositories to test")
    print(f"Using {max_workers} parallel workers\n")

    if len(repositories) == 0:
        print("ERROR: No repositories found in list file")
        sys.exit(1)

    # Initialize tester
    tester = ParallelEmpiricalTester(max_workers=max_workers)

    try:
        # Run parallel tests
        summary = tester.run_parallel_tests(repositories)

        # Exit with error code if any repos failed
        if summary["failed"] > 0:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()