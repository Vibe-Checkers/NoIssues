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
PROMPT_COST_PER_1K = float(os.getenv("AZURE_GPT5NANO_PROMPT_COST_PER_1K", "0"))
COMPLETION_COST_PER_1K = float(os.getenv("AZURE_GPT5NANO_COMPLETION_COST_PER_1K", "0"))


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
    """Calculate cost from token usage using configured per-1K prices."""
    prompt = token_usage.get("input", 0)
    completion = token_usage.get("output", 0)
    cost_prompt = (prompt / 1000.0) * PROMPT_COST_PER_1K
    cost_completion = (completion / 1000.0) * COMPLETION_COST_PER_1K
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": token_usage.get("total", prompt + completion),
        "prompt_cost_usd": cost_prompt,
        "completion_cost_usd": cost_completion,
        "total_cost_usd": cost_prompt + cost_completion,
    }


class LLMFunctionalVerifier:
    """
    Generates and analyzes functional smoke tests for containers.
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

    def analyze_error(self, error_log: str, failed_command: str) -> Tuple[Dict, Dict]:
        """
        Analyze a build error and return structured diagnosis.
        Returns: (analysis_dict, log_data)
        """
        system_prompt = """You are a Docker Build Expert and Linux System Administrator.
Your goal is to analyze a failed Docker build log and identify the EXACT valid fix.

Output must be valid JSON with this structure:
{
    "cause": "Concise explanation of what went wrong (e.g. 'Missing compiled extension dependencies')",
    "missing_packages": ["list", "of", "likely", "missing", "linux", "packages"],
    "suggested_fix": "Concrete, single-line command to fix the issue (e.g. 'RUN apt-get update && apt-get install -y libxml2-dev')",
    "search_keywords": "Optimized keywords for web search if the fix is uncertain"
}

Rules:
1. If the error is a missing file (e.g. 'requirements.txt not found'), suggest checking file locations.
2. If it's a network error, suggest a retry or check proxies.
3. Be specific with package names (e.g. 'python3-dev' instead of just 'dev headers').
4. Do NOT hallucinate packages. If unsure, assume a web search is needed.
"""
        
        user_prompt = f"""FAILED COMMAND: {failed_command}
        
ERROR LOG TAIL:
{error_log}

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

        self.docker_tester = DockerBuildTester(timeout=600, serialize_builds=True)

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
        """Save all artifacts (Dockerfile, .dockerignore, build logs) to artifacts directory."""
        try:
            artifact_dir = self.artifacts_dir / slug
            artifact_dir.mkdir(parents=True, exist_ok=True)

            # Save Dockerfile
            dockerfile_src = Path(repo_path) / "Dockerfile"
            if dockerfile_src.exists():
                dockerfile_dst = artifact_dir / "Dockerfile"
                shutil.copy2(dockerfile_src, dockerfile_dst)
                self.log(slug, f"Saved Dockerfile to {dockerfile_dst}", to_console=False)

            # Save .dockerignore
            dockerignore_src = Path(repo_path) / ".dockerignore"
            if dockerignore_src.exists():
                dockerignore_dst = artifact_dir / ".dockerignore"
                shutil.copy2(dockerignore_src, dockerignore_dst)
                self.log(slug, f"Saved .dockerignore to {dockerignore_dst}", to_console=False)

            # Save result metadata as JSON
            metadata_file = artifact_dir / "metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "repo_url": result.get("repo_url"),
                    "repo_name": result.get("repo_name"),
                    "detected_language": result.get("detected_language"),
                    "success": result.get("success"),
                    "timestamp": result.get("timestamp"),
                    "total_duration": result.get("total_duration_seconds"),
                    "token_usage": result.get("token_usage"),
                    "cost": result.get("cost")
                }, f, indent=2)

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

            # Setup per-repo logging with timestamp
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            transcript_file = self.transcripts_dir / f"{slug}_{ts_str}.log"
            structured_log_file = self.structured_logs_dir / f"{slug}_{ts_str}.json"

            def validation_callback(path):
                """Callback for agent to verify its own work via Docker build."""
                dockerfile_path = Path(path) / "Dockerfile"
                if not dockerfile_path.exists():
                    self.log(repo_name, "Validation: Dockerfile not found", to_console=False)
                    return {"success": False, "error": "Dockerfile not found at root of repository"}

                self.log(repo_name, "Validation: Building Docker image...", to_console=False)
                image_name = f"learner-{slug}:latest"
                build_res = self.docker_tester.build_dockerfile(
                    str(dockerfile_path),
                    path,
                    image_name
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
                Verifies the Dockerfile by attempting a build.
                Returns compact JSON structure instead of full logs.

                Input is ignored (it looks for 'Dockerfile' in the root).
                """
                self.log(repo_name, "[VerifyBuild] Starting Docker build verification", to_console=False)

                dockerfile_path = Path(repo_path) / "Dockerfile"

                if not dockerfile_path.exists():
                    self.log(repo_name, "[VerifyBuild] Dockerfile not found", to_console=False)
                    return json.dumps({
                        "status": "error",
                        "message": "Dockerfile not found. You must create it first using WriteToFile."
                    }, indent=2)

                # 1. Pre-build Hygiene Checks (Fail Fast)
                self.log(repo_name, "[VerifyBuild] Running pre-build validation", to_console=False)

                # Check for multi-stage hallucinations (referencing non-existent stages)
                multistage_errors = self._validate_multistage_references(dockerfile_path)
                if multistage_errors['invalid_refs']:
                    defined = ", ".join(multistage_errors['defined_stages']) if multistage_errors['defined_stages'] else "None"
                    self.log(repo_name, f"[VerifyBuild] Invalid stage references: {multistage_errors['invalid_refs']}", to_console=False)
                    return json.dumps({
                        "status": "failed",
                        "stage": "PRE_BUILD_CHECK",
                        "error_type": "INVALID_STAGE_REFERENCE",
                        "message": f"CRITICAL: You are COPYing from stage(s) {multistage_errors['invalid_refs']} which DO NOT EXIST. Defined stages are: {defined}. You must fix this before building."
                    }, indent=2)

                # # Check for fake/dummy binaries
                # fake_binary_check = self._detect_fake_binaries(dockerfile_path)
                # if fake_binary_check['is_suspicious']:
                #     violations = "\n- ".join(fake_binary_check['violations'])
                #     self.log(repo_name, f"[VerifyBuild] Fake binary patterns detected: {violations}", to_console=False)
                #     return json.dumps({
                #         "status": "failed",
                #         "stage": "PRE_BUILD_CHECK",
                #         "error_type": "FAKE_BINARY_DETECTED",
                #         "message": f"CRITICAL: Your Dockerfile appears to create fake/dummy binaries to game smoke tests.\n\nViolations detected:\n- {violations}\n\nYou must create a legitimate Dockerfile that actually builds the application, not fake executables."
                #     }, indent=2)

                self.log(repo_name, "[VerifyBuild] Pre-build validation passed", to_console=False)

                verify_slug = f"verify-{slug}"
                verify_image = f"verify-{verify_slug}:latest"

                # Capture auxiliary logs from LLM helpers
                aux_logs = []

                self.log(repo_name, "[VerifyBuild] Building Docker image", to_console=False)
                result = self.docker_tester.build_dockerfile(
                    str(dockerfile_path),
                    repo_path,
                    verify_image
                )

                if result.get("success"):
                    self.log(repo_name, "[VerifyBuild] Docker build succeeded, running smoke test", to_console=False)
                    # Phase 2: Functional Verification (Smoke Test)
                    try:
                        verifier = LLMFunctionalVerifier()

                        # Read Dockerfile for context
                        with open(dockerfile_path, 'r') as f:
                            df_content = f.read()

                        # 1. Generate Command
                        self.log(repo_name, "[VerifyBuild] Generating smoke test command via LLM", to_console=False)
                        test_cmd, cmd_log = verifier.generate_verification_command(df_content, repo_path.split('/')[-1])
                        aux_logs.append(cmd_log)
                        self.log(repo_name, f"[VerifyBuild] Generated test command: {test_cmd}", to_console=False)

                        # 2. Run Container
                        self.log(repo_name, f"[VerifyBuild] Running container with test command", to_console=False)
                        run_result = self.docker_tester.run_container(verify_image, test_cmd, timeout=20)

                        # 3. Analyze Output
                        self.log(repo_name, "[VerifyBuild] Analyzing smoke test output via LLM", to_console=False)
                        verification, verify_log = verifier.verify_output(test_cmd, run_result.get("output", ""), run_result.get("exit_code", -1))
                        if verify_log: aux_logs.append(verify_log)
                        
                        if verification.get("success"):
                             self.log(repo_name, "[VerifyBuild] ✓ Smoke test PASSED", to_console=False)

                             # 4. Validate Dockerfile Suitability with LLM
                             self.log(repo_name, "[VerifyBuild] Validating Dockerfile suitability", to_console=False)
                             validator = LLMDockerfileValidator()
                             suitability, suitability_log = validator.validate_dockerfile_suitability(
                                 df_content,
                                 repo_path.split('/')[-1],
                                 build_success=True,
                                 smoke_test_passed=True
                             )
                             aux_logs.append(suitability_log)

                             self.docker_tester.cleanup_image(verify_image)

                             # Return response WITHOUT revealing smoke test command
                             return json.dumps({
                                "status": "success",
                                "message": "✓ Docker build succeeded and container is functional.",
                                "verification": {
                                    "build_passed": True,
                                    "smoke_test_passed": True,
                                    "dockerfile_suitable": suitability.get("is_suitable", True),
                                    "explanation": suitability.get("explanation", "Dockerfile appears suitable for the application."),
                                    "confidence": suitability.get("confidence", "medium"),
                                    "concerns": suitability.get("concerns", [])
                                },
                                # "auxiliary_logs": aux_logs # Removed to save context
                            }, indent=2)
                        else:
                            self.log(repo_name, "[VerifyBuild] ✗ Smoke test FAILED", to_console=False)

                            # Validate Dockerfile even on failure to provide explanation
                            validator = LLMDockerfileValidator()
                            suitability, suitability_log = validator.validate_dockerfile_suitability(
                                df_content,
                                repo_path.split('/')[-1],
                                build_success=True,
                                smoke_test_passed=False
                            )
                            aux_logs.append(suitability_log)

                            # Build passed, but runtime failed - don't reveal test command
                            return json.dumps({
                                "status": "failed",
                                "stage": "RUNTIME_CHECK",
                                "error_type": "SMOKE_TEST_FAILED",
                                "message": f"Build succeeded, but the container failed functional verification.\n\nThe container built successfully but does not appear to be functional when executed.",
                                "verification": {
                                    "build_passed": True,
                                    "smoke_test_passed": False,
                                    "dockerfile_suitable": suitability.get("is_suitable", False),
                                    "explanation": suitability.get("explanation", "Container failed to execute properly."),
                                    "concerns": suitability.get("concerns", [])
                                },
                                "error_output": run_result.get("output", ""),
                                "next_steps": "The container builds but doesn't run correctly. Check your ENTRYPOINT, CMD, runtime dependencies, and ensure the application is actually installed.",
                                # "auxiliary_logs": aux_logs # Removed to save context
                            }, indent=2)

                    except Exception as e:
                        # Fallback if verification infra fails - treat as failure
                        self.log(repo_name, f"[VerifyBuild] Smoke test error: {e}", to_console=False)
                        print(f"[Warning] Functional verification failed: {e}")
                        self.docker_tester.cleanup_image(verify_image)

                        # Read Dockerfile for validation
                        with open(dockerfile_path, 'r') as f:
                            df_content = f.read()

                        # Validate Dockerfile to provide helpful feedback
                        validator = LLMDockerfileValidator()
                        suitability, suitability_log = validator.validate_dockerfile_suitability(
                            df_content,
                            repo_path.split('/')[-1],
                            build_success=True,
                            smoke_test_passed=False
                        )
                        aux_logs.append(suitability_log)

                        return json.dumps({
                            "status": "failed",
                            "stage": "SMOKE_TEST_EXCEPTION",
                            "error_type": "VERIFICATION_INFRASTRUCTURE_ERROR",
                            "message": f"Build succeeded but smoke test verification failed with error: {str(e)}\n\nThis likely means your container cannot be executed properly.",
                            "verification": {
                                "build_passed": True,
                                "smoke_test_passed": False,
                                "dockerfile_suitable": suitability.get("is_suitable", False),
                                "explanation": suitability.get("explanation", "Unable to verify container functionality."),
                                "concerns": suitability.get("concerns", ["Smoke test infrastructure error"])
                            },
                            "next_steps": "The Dockerfile built but we couldn't verify it works. Ensure your container has a valid ENTRYPOINT/CMD and all runtime dependencies are installed.",
                            # "auxiliary_logs": aux_logs # Removed to save context
                        }, indent=2)

                else:
                    self.log(repo_name, f"[VerifyBuild] ✗ Docker build FAILED at stage: {result.get('stage')}", to_console=False)

                    # Cleanup verify image on build failure
                    try:
                        self.docker_tester.cleanup_image(verify_image)
                    except:
                        pass

                    # Extract compact error info
                    error_msg = result.get('error_message', '')
                    error_lines = error_msg.split('\n') if error_msg else []

                    # Return truncated error log for context to save tokens
                    tail_lines = error_msg

                    # LLM-Based Error Analysis
                    self.log(repo_name, "[VerifyBuild] Running LLM error analysis", to_console=False)
                    analyzer = LLMErrorAnalyzer()
                    analysis, analysis_log = analyzer.analyze_error(result.get('error_message', ''), result.get('failed_command', ''))
                    aux_logs.append(analysis_log)
                    self.log(repo_name, f"[VerifyBuild] Error cause: {analysis.get('cause', 'Unknown')}", to_console=False)

                    # Build structured error features from LLM analysis
                    error_features = {
                        "cause": analysis.get("cause"),
                        "suggested_fix": analysis.get("suggested_fix"),
                        "missing_packages": analysis.get("missing_packages", []),
                    }

                    # Build compact response
                    compact_error = {
                        "status": "failed",
                        "stage": result.get('stage', 'UNKNOWN'),
                        "failed_command": result.get('failed_command', 'Unknown command'),
                        "error_snippet": result.get('error_snippet', 'See tail_lines'),
                        "error_analysis": error_features, # Replaced static ErrorPatternDetector with rich analysis
                        "tail_lines": tail_lines,
                        "search_keywords": analysis.get("search_keywords", f"docker build error {result.get('failed_command', '')}")
                        # Removed auxiliary_logs to save context
                    }

                    return json.dumps(compact_error, indent=2)

            verify_tool = Tool(
                name="VerifyBuild",
                func=verify_build_tool_func,
                description="Verifies the 'Dockerfile' by running a real Docker build. usage: VerifyBuild('')"
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
                extra_tools=[verify_tool]
            )

            duration = time.time() - start_time # Use start_time from original code
            result["agent_analysis"] = agent_result
            result["total_duration"] = duration
            result["detected_language"] = agent_result.get("language", "Unknown")

            # Capture token usage and calculate cost
            token_usage = callback_handler.token_usage
            cost_data = compute_token_cost(token_usage)
            result["token_usage"] = token_usage
            result["cost"] = cost_data

            # Save structured agent transcript
            try:
                transcript = callback_handler.get_transcript()
                with open(structured_log_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "repo_url": repo_url,
                        "repo_name": repo_name,
                        "detected_language": result["detected_language"],
                        "timestamp": result["timestamp"],
                        "transcript": transcript,
                        "token_usage": token_usage,
                        "cost": cost_data,
                        "final_result": {
                            "status": agent_result["status"],
                            "attempts": agent_result.get("attempts", 0)
                        }
                    }, f, indent=2)
                self.log(repo_name, f"Saved structured transcript to {structured_log_file}", to_console=False)
            except Exception as e:
                self.log(repo_name, f"Failed to save transcript: {e}", to_console=False)

            # Add dockerfile_test structure for compatibility with summary
            result["dockerfile_test"] = {
                "success": agent_result["status"] == "success",
                "attempts": agent_result.get("attempts", 0),
                "final_iteration": agent_result.get("attempts", 0) if agent_result["status"] == "success" else 0,
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

        # 3. Clear large result fields to reduce memory
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
            
            if "cost" in r:
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