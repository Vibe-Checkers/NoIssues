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
# Removed: improved_error_recognition - caused more problems than it solved
# from improved_error_recognition import ImprovedErrorRecognition

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
from run_agent import clone_repository, detect_project_language, analyze_repository
from planner_agent import create_planner_agent
from empirical_test import DockerBuildTester

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


class ErrorPatternDetector:
    """
    Detects build error patterns instead of package names.

    This enables general-purpose error detection that works for ANY package
    exhibiting the same error signature, rather than hardcoding specific package names.
    """

    ERROR_SIGNATURES = {
        'fortran_compiler_needed': [
            'gfortran not found',
            'fortran compiler',
            'f77', 'f90', 'f95',
            'numpy.distutils.fcompiler',
            'cannot find -lgfortran',
            'fortran: command not found'
        ],
        'blas_library_needed': [
            'cannot find -lblas',
            'cannot find -llapack',
            'cblas.h: no such file',
            'lapacke.h: no such file',
            'openblas',
            'libatlas',
            'blas/lapack not found'
        ],
        'python_dev_headers': [
            'python.h: no such file',
            'python3-dev',
            'pyconfig.h',
            'error: command.*gcc.*failed with exit (code|status) 1',
            'building.*extension modules',
            'fatal error: python.h'
        ],
        'c_compiler_needed': [
            'gcc: command not found',
            'cc: command not found',
            'make: command not found',
            'build-essential',
            'compiler not found'
        ],
        'cpp_compiler_needed': [
            'g++: command not found',
            'c++ compiler',
            'cxx compiler',
            'c++: command not found'
        ]
    }

    @staticmethod
    def detect_patterns(error_log: str, failed_cmd: str = "") -> list:
        """
        Returns list of detected pattern types from error logs.

        Args:
            error_log: The error output from build
            failed_cmd: The command that failed (optional)

        Returns:
            List of pattern types detected (e.g., ['fortran_compiler_needed', 'blas_library_needed'])
        """
        combined_text = f"{error_log} {failed_cmd}".lower()
        detected = []

        for pattern_type, indicators in ErrorPatternDetector.ERROR_SIGNATURES.items():
            for indicator in indicators:
                if indicator.lower() in combined_text:
                    detected.append(pattern_type)
                    break

        return list(set(detected))  # Remove duplicates

    @staticmethod
    def detect_base_os(dockerfile_content: str) -> str:
        """
        Detect OS from Dockerfile FROM statement.

        Args:
            dockerfile_content: Contents of the Dockerfile

        Returns:
            OS type: 'alpine', 'debian', or 'redhat'
        """
        import re
        from_match = re.search(r'FROM\s+([^\s:]+)', dockerfile_content, re.IGNORECASE)
        if not from_match:
            return 'debian'  # Safe default

        image_name = from_match.group(1).lower()

        # OS detection map
        if 'alpine' in image_name:
            return 'alpine'
        elif 'ubuntu' in image_name or 'debian' in image_name:
            return 'debian'
        elif 'centos' in image_name or 'rhel' in image_name or 'fedora' in image_name or 'rocky' in image_name:
            return 'redhat'
        elif 'node' in image_name or 'python' in image_name or 'ruby' in image_name:
            return 'debian'  # Most official images are Debian-based
        else:
            return 'debian'  # Conservative default

    @staticmethod
    def get_package_manager(base_os: str) -> str:
        """
        Map OS to its package manager.

        Args:
            base_os: OS type ('alpine', 'debian', 'redhat')

        Returns:
            Package manager command ('apk', 'apt-get', 'yum')
        """
        os_to_pm = {
            'alpine': 'apk',
            'debian': 'apt-get',
            'redhat': 'yum'
        }
        return os_to_pm.get(base_os, 'apt-get')


class PackageManagerDetector:
    """
    Detects package manager and generates appropriate search queries.

    Uses a registry pattern for extensibility - new package managers can be added
    without changing the detection logic.
    """

    PACKAGE_MANAGERS = {
        # Node.js ecosystem
        'npm': {
            'indicators': ['npm err!', 'npm error', 'npm install'],
            'error_patterns': {
                'enoent': 'docker npm ENOENT missing file',
                'gyp': 'docker node-gyp build failed python make gcc',
                'permission': 'docker npm permission denied unsafe-perm',
            },
            'default_query': 'docker npm install error'
        },
        'pnpm': {
            'indicators': ['pnpm err', 'pnpm error', 'pnpm install'],
            'error_patterns': {
                'frozen lockfile': 'docker pnpm frozen-lockfile mismatch',
            },
            'default_query': 'docker pnpm install failed'
        },
        'yarn': {
            'indicators': ['yarn error', 'yarn install'],
            'error_patterns': {},
            'default_query': 'docker yarn install failed'
        },

        # Python ecosystem
        'pip': {
            'indicators': ['pip error', 'pip install', 'pip failed'],
            'error_patterns': {
                'gcc': 'docker pip install gcc compilation failed build-essential',
                'wheel': 'docker pip wheel build failed',
            },
            'default_query': 'docker pip install failed'
        },
        'poetry': {
            'indicators': ['poetry error', 'poetry install'],
            'error_patterns': {},
            'default_query': 'docker poetry install failed'
        },
        'pipenv': {
            'indicators': ['pipenv error', 'pipenv install'],
            'error_patterns': {},
            'default_query': 'docker pipenv install failed'
        },
        'uv': {
            'indicators': ['uv error', 'uv pip', 'uv sync'],
            'error_patterns': {},
            'default_query': 'docker uv install failed'
        },

        # Rust
        'cargo': {
            'indicators': ['cargo error', 'cargo build'],
            'error_patterns': {},
            'default_query': 'docker cargo build failed rust dependencies'
        },

        # Java ecosystem
        'maven': {
            'indicators': ['maven error', 'mvn error', '[error]'],
            'error_patterns': {},
            'default_query': 'docker maven build failed'
        },
        'gradle': {
            'indicators': ['gradle error', './gradlew'],
            'error_patterns': {},
            'default_query': 'docker gradle build failed'
        },
        'sbt': {
            'indicators': ['sbt error', '[error]'],
            'error_patterns': {},
            'default_query': 'docker sbt build failed'
        },

        # Ruby
        'bundler': {
            'indicators': ['bundler error', 'bundle install'],
            'error_patterns': {},
            'default_query': 'docker bundler install failed'
        },

        # PHP
        'composer': {
            'indicators': ['composer error', 'composer install'],
            'error_patterns': {},
            'default_query': 'docker composer install failed'
        },

        # Go
        'go': {
            'indicators': ['go: error', 'go build', 'go get'],
            'error_patterns': {},
            'default_query': 'docker go build failed'
        },

        # System package managers
        'apt': {
            'indicators': ['apt error', 'apt-get', 'unable to locate package'],
            'error_patterns': {
                'unable to locate package': 'docker apt unable to locate package'
            },
            'default_query': 'docker apt-get install failed'
        },
        'apk': {
            'indicators': ['apk error', 'apk add'],
            'error_patterns': {},
            'default_query': 'docker apk add failed alpine'
        },
        'yum': {
            'indicators': ['yum error', 'no package'],
            'error_patterns': {},
            'default_query': 'docker yum install failed'
        }
    }

    @staticmethod
    def detect_package_manager(error_log: str) -> tuple:
        """
        Detect package manager from error log and return appropriate search query.

        Args:
            error_log: The error output from build

        Returns:
            Tuple of (package_manager_name, error_keywords) for search
        """
        error_log_lower = error_log.lower()

        # Try to detect package manager
        for pm_name, pm_config in PackageManagerDetector.PACKAGE_MANAGERS.items():
            # Check if any indicator matches
            if any(indicator.lower() in error_log_lower for indicator in pm_config['indicators']):
                # Found package manager, now check for specific error patterns
                for pattern, query in pm_config['error_patterns'].items():
                    if pattern.lower() in error_log_lower:
                        return (pm_name, query)

                # No specific pattern, use default
                return (pm_name, pm_config['default_query'])

        # No package manager detected
        return (None, "docker dependency install failed")

    @staticmethod
    def extract_package_name(error_log: str, package_manager: str) -> str:
        """
        Extract failing package name from error log.

        Args:
            error_log: The error output
            package_manager: Detected package manager name

        Returns:
            Package name if found, empty string otherwise
        """
        import re

        # Common patterns across package managers
        patterns = [
            r'unable to locate package\s+(\S+)',  # apt
            r'error:\s+package\s+[\'"](\S+)[\'"]',  # generic
            r'failed to (?:install|build)\s+(\S+)',  # generic
            r'npm err!\s+(?:404|enoent).*?(\S+)',  # npm
        ]

        for pattern in patterns:
            match = re.search(pattern, error_log, re.IGNORECASE)
            if match:
                return match.group(1)

        return ""


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

        # Create subdirectories
        self.repos_dir = self.results_dir / "repositories"
        self.reports_dir = self.results_dir / "analysis_reports"
        self.logs_dir = self.results_dir / "logs"
        self.docker_errors_dir = self.results_dir / "docker_errors"

        self.repos_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.docker_errors_dir.mkdir(exist_ok=True)

        self.docker_tester = DockerBuildTester(timeout=600)

        # Thread-safe console output
        self.console_lock = threading.Lock()

        # Initialize results storage
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_jsonl_file = self.results_dir / f"results_{self.timestamp}.jsonl"  # Append-only
        self.results_json_file = self.results_dir / f"results_{self.timestamp}.json"   # Final snapshot
        self.summary_file = self.results_dir / f"summary_{self.timestamp}.txt"
        self.master_log_file = self.results_dir / f"master_log_{self.timestamp}.txt"

        # Open master log
        self.master_log = open(self.master_log_file, 'w', encoding='utf-8')
        self.master_log_lock = threading.Lock()

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

    def test_single_repository(self, repo_url: str, worker_id: int) -> Dict:
        """
        Test a single repository: clone, analyze, generate Dockerfile, test build.

        Args:
            repo_url: GitHub repository URL
            worker_id: Task ID (index in repository list, NOT actual thread ID)

        Returns:
            Dictionary with complete test results
        """
        # Generate unique slug to prevent collisions (e.g., facebook__react vs myuser__react)
        slug = repo_slug(repo_url)
        # Keep human-friendly short name for display
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        test_start_time = time.time()

        result = {
            "repo_url": repo_url,
            "repo_slug": slug,  # Unique identifier for filesystem/docker
            "repo_name": repo_name,  # Human-friendly display name
            "task_id": worker_id,  # Renamed from worker_id for clarity
            "timestamp": datetime.now().isoformat(),
            "agent_analysis": {},
            "dockerfile_test": {},
            "total_duration_seconds": 0,
            "success": False
        }

        repo_path = None

        try:
            # Get actual thread ID for accurate parallel debugging
            thread_id = threading.get_ident()
            self.log(repo_name, f"[Task {worker_id}, Thread {thread_id}] Starting test", to_console=True)

            # Step 1: Clone repository with retry logic
            self.log(repo_name, "Cloning repository...", to_console=True)
            clone_start = time.time()
            
            try:
                repo_path = clone_repository(repo_url, target_dir=str(self.repos_dir), auto_remove=True)
                clone_duration = time.time() - clone_start

                result["clone"] = {
                    "success": True,
                    "duration_seconds": clone_duration,
                    "path": repo_path
                }
                self.log(repo_name, f"Cloned successfully ({clone_duration:.1f}s)", to_console=True)
                
            except SystemExit as e:
                # Clone failed after all retries - don't crash the test suite
                clone_duration = time.time() - clone_start
                result["clone"] = {
                    "success": False,
                    "duration_seconds": clone_duration,
                    "error": "Clone failed after all retry attempts"
                }
                self.log(repo_name, "Clone failed after retries, skipping this repository", to_console=True)
                result["success"] = False
                return result  # Skip to finally block

            # Detect language
            detected_language = detect_project_language(repo_path)
            result["detected_language"] = detected_language
            self.log(repo_name, f"Detected language: {detected_language}", to_console=True)

            # Step 2: Initialize agent
            self.log(repo_name, "Initializing agent...", to_console=True)
            agent_init_start = time.time()
            agent, callback_handler = create_planner_agent(
                max_iterations=25,
                verbose=True,
                repository_path=repo_path,
                repo_name=repo_name,
                detected_language=detected_language
            )
            agent_init_duration = time.time() - agent_init_start

            result["agent_init"] = {
                "success": True,
                "duration_seconds": agent_init_duration
            }
            self.log(repo_name, f"Agent initialized ({agent_init_duration:.1f}s)", to_console=True)

            # Step 3: Run analysis
            self.log(repo_name, "Running agent analysis...", to_console=True)
            analysis_start = time.time()

            # Create report directory using slug to prevent collisions
            report_dir = self.reports_dir / f"{slug}_{self.timestamp}"
            report_dir.mkdir(exist_ok=True)

            # Set thread-local report directory for web search caching (thread-safe)
            import planner_agent
            planner_agent._set_report_directory(str(report_dir))
            # NOTE: No longer setting global REPORT_DIRECTORY - thread-local only for thread safety

            # Run analysis with repo-specific log (use slug to prevent collisions)
            log_file = self.logs_dir / f"{slug}_agent_log.txt"
            report_dir_result = analyze_repository(
                agent, repo_path, repo_name, repo_url,
                callback_handler, log_file_path=log_file, report_dir=report_dir
            )

            analysis_duration = time.time() - analysis_start

            result["agent_analysis"] = {
                "success": True,
                "duration_seconds": analysis_duration,
                "report_directory": str(report_dir_result),
                "log_file": str(log_file)
            }

            # Get token usage
            if callback_handler and callback_handler.token_usage["total"] > 0:
                result["agent_analysis"]["token_usage"] = callback_handler.token_usage
                tokens = callback_handler.token_usage
                cost = compute_token_cost(tokens)
                result["agent_analysis"]["cost_usd"] = cost
                self.log(
                    repo_name,
                    f"Analysis complete ({analysis_duration:.1f}s, {tokens['total']:,} tokens, ${cost['total_cost_usd']:.4f})",
                    to_console=True
                )
            else:
                self.log(repo_name, f"Analysis complete ({analysis_duration:.1f}s)", to_console=True)

            # Step 4: Test Dockerfile with iterative refinement (max 10 iterations)
            self.log(repo_name, "Testing generated Dockerfile...", to_console=True)
            dockerfile_path = report_dir_result / "Dockerfile"

            if not dockerfile_path.exists():
                result["dockerfile_test"] = {
                    "success": False,
                    "stage": "DOCKERFILE_GENERATION",
                    "error_type": "DOCKERFILE_NOT_GENERATED",
                    "error_message": "Agent did not generate a Dockerfile"
                }
                self.log(repo_name, "ERROR: Dockerfile not generated", to_console=True)

            else:
                # Initialize iteration tracking
                result["dockerfile_test"] = {
                    "iterations": [],
                    "final_iteration": 0,
                    "success": False
                }
                max_refinement_iterations = 15  # Increased from 10 to 15 based on empirical data
                iteration_success = False

                # Initialize state for refinement tracking
                previous_from_lines = []
                previous_tools_used = []
                consecutive_validation_errors = 0  # Track repeated validation failures

                # Iterative refinement loop
                for iteration in range(max_refinement_iterations + 1):  # 0 = initial, 1-15 = refinements
                    iteration_start_time = time.time()
                    
                    # Read current Dockerfile content
                    with open(dockerfile_path, 'r', encoding='utf-8') as f:
                        dockerfile_content = f.read()

                    if iteration == 0:
                        # Trust the agent's Dockerfile generation from improved prompts
                        # Removed pre-checks that were breaking agent learning feedback loop
                        result["dockerfile_content"] = dockerfile_content
                        self.log(repo_name, f"Testing initial Dockerfile (iteration {iteration})...", to_console=True)
                    else:
                        self.log(repo_name, f"Testing refined Dockerfile (iteration {iteration}/{max_refinement_iterations})...", to_console=True)

                    # CRITICAL: Strip digest pins to prevent BuildKit corruption
                    # Analysis shows 100% correlation between @sha256 pins and layer corruption
                    digests_removed = self._strip_digest_pins_from_dockerfile(dockerfile_path)
                    if digests_removed:
                        self.log(repo_name, "Stripped digest pins from FROM statements to prevent corruption", to_console=False)

                    # Validate multi-stage COPY --from references
                    multistage_validation = self._validate_multistage_references(dockerfile_path)
                    if multistage_validation['invalid_refs']:
                        self.log(
                            repo_name,
                            f"Warning: Invalid COPY --from references: {', '.join(multistage_validation['invalid_refs'])}",
                            to_console=False
                        )
                        if multistage_validation['defined_stages']:
                            self.log(
                                repo_name,
                                f"  - Defined stages are: {', '.join(multistage_validation['defined_stages'])}",
                                to_console=False
                            )
                        else:
                            self.log(
                                repo_name,
                                f"  - No build stages defined (this will try to pull from registry)",
                                to_console=False
                            )

                    # Validate COPY sources exist in repository
                    copy_validation = self._validate_copy_sources(dockerfile_path, repo_path)
                    if copy_validation['missing']:
                        self.log(
                            repo_name,
                            f"Warning: Missing COPY sources: {', '.join(copy_validation['missing'])}",
                            to_console=False
                        )
                        if copy_validation['suggestions']:
                            for missing, alternatives in copy_validation['suggestions'].items():
                                self.log(
                                    repo_name,
                                    f"  - Instead of {missing}, found: {', '.join(alternatives)}",
                                    to_console=False
                                )

                    # Test Docker build using slug to prevent image name collisions
                    image_name = f"parallel-empirical-{slug}:latest"
                    docker_result = self.docker_tester.build_dockerfile(
                        str(dockerfile_path),
                        repo_path,
                        image_name
                    )

                    iteration_duration = time.time() - iteration_start_time
                    iteration_result = {
                        "iteration": iteration,
                        "duration_seconds": iteration_duration,
                        "success": docker_result.get("success", False),
                        "stage": docker_result.get("stage", "UNKNOWN"),
                        "failed_command": docker_result.get("failed_command", "Unknown"),
                        "exit_code": docker_result.get("exit_code", -1)
                    }

                    if docker_result["success"]:
                        iteration_success = True
                        result["dockerfile_test"]["success"] = True
                        result["dockerfile_test"]["final_iteration"] = iteration
                        iteration_result["message"] = "Build successful"

                        self.log(
                            repo_name,
                            f"Docker build SUCCESS at iteration {iteration} ({docker_result['duration_seconds']:.1f}s)",
                            to_console=True
                        )

                        # Append successful iteration BEFORE breaking
                        result["dockerfile_test"]["iterations"].append(iteration_result)

                        # Cleanup image
                        self.docker_tester.cleanup_image(image_name)
                        break  # Exit refinement loop on success
                        
                    else:
                        # Log failure
                        stage = docker_result.get('stage', 'UNKNOWN')
                        failed_cmd = docker_result.get('failed_command', 'Unknown')
                        
                        iteration_result["error_snippet"] = docker_result.get('error_snippet', '')
                        
                        self.log(
                            repo_name,
                            f"Docker build FAILED at iteration {iteration} - Stage: {stage}, Command: {failed_cmd}",
                            to_console=True
                        )

                        # Save error log for this iteration using slug
                        error_iteration_suffix = f"_iter{iteration}" if iteration > 0 else ""
                        raw_error_file = self.docker_errors_dir / f"{slug}_docker_error{error_iteration_suffix}.log"
                        with open(raw_error_file, 'w', encoding='utf-8') as f:
                            f.write(f"# Docker Build Error Log - Iteration {iteration}\n")
                            f.write(f"# Repository: {repo_name}\n")
                            f.write(f"# URL: {repo_url}\n")
                            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
                            f.write(f"# Stage: {stage}\n")
                            f.write(f"# Failed Command: {failed_cmd}\n")
                            f.write(f"# Exit Code: {docker_result.get('exit_code', -1)}\n")
                            f.write("#" + "="*78 + "\n\n")
                            f.write(docker_result.get('error_message', 'No error message available'))

                        # Check if we should attempt refinement
                        if iteration < max_refinement_iterations:
                            # Check for infrastructure corruption that requires prune
                            if stage == "INFRASTRUCTURE_CORRUPTION":
                                self.log(
                                    repo_name,
                                    f"INFRASTRUCTURE CORRUPTION DETECTED - Docker BuildKit cache corrupted",
                                    to_console=True
                                )
                                self.log(
                                    repo_name,
                                    f"Attempting system-wide docker builder prune to fix corruption...",
                                    to_console=True
                                )

                                # Execute prune to clear corrupted build cache
                                prune_success = self.docker_tester.prune_build_cache()

                                if prune_success:
                                    # Exponential backoff after prune to let Docker stabilize
                                    # Base delay: 2 seconds, doubles each iteration (2s, 4s, 8s, 16s...)
                                    backoff_delay = min(2 ** iteration, 30)  # Cap at 30 seconds
                                    self.log(
                                        repo_name,
                                        f"Docker builder prune completed - waiting {backoff_delay}s for stabilization...",
                                        to_console=True
                                    )
                                    time.sleep(backoff_delay)

                                    self.log(
                                        repo_name,
                                        f"Retrying build after prune (iteration {iteration + 1})...",
                                        to_console=True
                                    )
                                    iteration_result["prune_executed"] = True
                                    iteration_result["prune_success"] = True
                                    iteration_result["backoff_delay_seconds"] = backoff_delay
                                    result["dockerfile_test"]["iterations"].append(iteration_result)
                                    # Continue to retry build after prune
                                    continue
                                else:
                                    self.log(
                                        repo_name,
                                        f"Docker builder prune FAILED - cannot recover from corruption",
                                        to_console=True
                                    )
                                    iteration_result["prune_executed"] = True
                                    iteration_result["prune_success"] = False
                                    iteration_result["refinement_skipped"] = True
                                    iteration_result["reason"] = "Infrastructure corruption - prune failed"
                                    result["dockerfile_test"]["iterations"].append(iteration_result)
                                    break  # Cannot recover if prune fails

                            # NOTE: Transient error retry logic (NETWORK/IMAGE_PULL) has been removed.
                            # User policy: Treat ALL errors as important and requiring agent intervention immediately.
                            # No blind retries.
                            pass

                            # Check for dependency rot (terminal - requires code/lockfile update)
                            if stage == "DEPENDENCY_ROT":
                                self.log(
                                    repo_name,
                                    f"DEPENDENCY ROT DETECTED - Package versions no longer available",
                                    to_console=True
                                )
                                self.log(
                                    repo_name,
                                    f"This requires updating package.json/requirements.txt/lockfiles in source code",
                                    to_console=True
                                )
                                iteration_result["refinement_skipped"] = True
                                iteration_result["reason"] = "Dependency rot - requires code update, not Dockerfile changes"
                                result["dockerfile_test"]["iterations"].append(iteration_result)
                                break  # Cannot fix with Dockerfile iterations

                            # Check for Docker image validation errors (e.g., "" failed validation)
                            error_message = docker_result.get('error_message', '')
                            if 'failed validation' in error_message or 'failed to solve' in error_message:
                                consecutive_validation_errors += 1
                                self.log(
                                    repo_name,
                                    f"Docker image validation error detected (consecutive: {consecutive_validation_errors})",
                                    to_console=True
                                )

                                # After 2-3 consecutive validation errors, force image change
                                if consecutive_validation_errors >= 3:
                                    self.log(
                                        repo_name,
                                        f"FORCING BASE IMAGE CHANGE after {consecutive_validation_errors} validation failures",
                                        to_console=True
                                    )
                                    # Extract current image and suggest fallback
                                    current_from_line = None
                                    try:
                                        for line in dockerfile_content.split('\n'):
                                            if line.strip().upper().startswith('FROM '):
                                                current_from_line = line.strip()
                                                break
                                    except:
                                        pass

                                    if current_from_line:
                                        self.log(repo_name, f"Current problematic image: {current_from_line}", to_console=True)
                                        iteration_result["forced_image_change"] = True
                                        iteration_result["previous_from_line"] = current_from_line
                            else:
                                # Reset counter if different error type
                                consecutive_validation_errors = 0

                            # Check for other non-recoverable errors
                            non_recoverable_stages = ["DOCKER_DAEMON", "BUILD_TIMEOUT", "BUILD_EXCEPTION"]
                            if stage in non_recoverable_stages:
                                self.log(
                                    repo_name,
                                    f"Error stage '{stage}' is non-recoverable, skipping refinement",
                                    to_console=True
                                )
                                iteration_result["refinement_skipped"] = True
                                iteration_result["reason"] = f"Non-recoverable error: {stage}"
                                result["dockerfile_test"]["iterations"].append(iteration_result)
                                break  # Don't attempt refinement for non-recoverable errors

                            # Attempt refinement
                            self.log(repo_name, f"Attempting Dockerfile refinement (iteration {iteration + 1}/{max_refinement_iterations})...", to_console=True)
                            
                            refinement_start = time.time()
                            refinement_success, tools_used = self._refine_dockerfile_with_error_feedback(
                                agent, dockerfile_path, raw_error_file, repo_path, repo_name,
                                iteration + 1, docker_result, previous_from_lines, previous_tools_used,
                                consecutive_validation_errors
                            )
                            refinement_duration = time.time() - refinement_start
                            
                            # Update state for next iteration
                            if tools_used:
                                previous_tools_used = tools_used
                            
                            # Track FROM line changes
                            try:
                                with open(dockerfile_path, 'r', encoding='utf-8') as f:
                                    current_content = f.read()
                                for line in current_content.split('\n'):
                                    if line.strip().upper().startswith('FROM '):
                                        previous_from_lines.append(line.strip())
                                        break
                            except:
                                pass

                            if refinement_success:
                                self.log(repo_name, f"Dockerfile refined successfully ({refinement_duration:.1f}s)", to_console=True)
                                iteration_result["refinement_success"] = True
                                iteration_result["refinement_duration_seconds"] = refinement_duration
                            else:
                                self.log(repo_name, f"Dockerfile refinement failed or produced no changes", to_console=True)
                                iteration_result["refinement_success"] = False
                                iteration_result["refinement_duration_seconds"] = refinement_duration
                                # Continue to next iteration anyway, agent might have made partial changes
                        else:
                            # Max iterations reached
                            iteration_result["max_iterations_reached"] = True
                            
                            # Save final error summary
                            error_summary_file = report_dir_result / "docker_build_error_summary.txt"
                            with open(error_summary_file, 'w', encoding='utf-8') as f:
                                f.write("="*80 + "\n")
                                f.write("DOCKER BUILD ERROR SUMMARY (Final - Max Iterations Reached)\n")
                                f.write("="*80 + "\n\n")
                                f.write(f"Repository: {repo_name}\n")
                                f.write(f"URL: {repo_url}\n")
                                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                                f.write(f"Total Refinement Iterations: {max_refinement_iterations}\n\n")
                                f.write(f"Final Failure Stage: {stage}\n")
                                f.write(f"Failed Docker Command: {failed_cmd}\n")
                                f.write(f"Exit Code: {docker_result.get('exit_code', -1)}\n\n")
                                
                                if docker_result.get('error_snippet'):
                                    f.write(f"Error Snippet:\n{'-'*80}\n")
                                    f.write(f"{docker_result['error_snippet']}\n")
                                    f.write(f"{'-'*80}\n\n")
                                
                                f.write(f"See full error: docker_errors/{slug}_docker_error{error_iteration_suffix}.log\n")

                    # Store iteration result
                    result["dockerfile_test"]["iterations"].append(iteration_result)

                # Store final docker result
                if iteration_success:
                    result["dockerfile_test"]["final_result"] = docker_result
                else:
                    result["dockerfile_test"]["final_result"] = docker_result
                    result["dockerfile_test"]["success"] = False

            # Determine overall success
            result["success"] = (
                result.get("clone", {}).get("success", False) and
                result.get("agent_analysis", {}).get("success", False) and
                result.get("dockerfile_test", {}).get("success", False)
            )

            # Log final status
            status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
            self.log(repo_name, f"Test complete: {status}", to_console=True)

            # NOW cleanup after everything is done
            self._aggressive_cleanup(repo_name, repo_path, result)

        except Exception as e:
            self.log(repo_name, f"EXCEPTION: {e}", to_console=True)
            result["exception"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            result["success"] = False

        finally:
            # Calculate total duration
            result["total_duration_seconds"] = time.time() - test_start_time

            # Emergency cleanup ONLY if exception occurred and cleanup wasn't done
            # Check if cleanup already happened by seeing if result was marked success/failure
            if "success" not in result or result.get("exception"):
                self.log(repo_name, "Emergency cleanup due to exception", to_console=False)
                if repo_path and os.path.exists(repo_path):
                    try:
                        shutil.rmtree(repo_path)
                    except:
                        pass

                # Cleanup Docker image using slug
                try:
                    image_name = f"parallel-empirical-{slug}:latest"
                    self.docker_tester.cleanup_image(image_name)
                except:
                    pass

            # Force garbage collection after each test
            gc.collect()

            # Save result
            with self.results_lock:
                self.results.append(result)
                self._save_incremental_results(result)  # Append to JSONL

        return result

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
            defined_stages = set()
            stage_pattern = r'FROM\s+[^\s]+\s+AS\s+([^\s]+)'
            for match in re.finditer(stage_pattern, content, re.IGNORECASE):
                stage_name = match.group(1).strip()
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

    def _strip_digest_pins_from_dockerfile(self, dockerfile_path: Path) -> bool:
        """
        Remove digest pins (@sha256:...) from FROM statements to prevent BuildKit corruption.

        CRITICAL: Analysis shows 100% correlation between digest-pinned images and
        "parent snapshot does not exist" errors. Digest-pinned layers are treated as
        dangling content and garbage-collected during parallel builds with cleanup.

        Args:
            dockerfile_path: Path to Dockerfile to sanitize

        Returns:
            True if any digests were removed, False otherwise
        """
        try:
            with open(dockerfile_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Remove digest pins: FROM ubuntu:22.04@sha256:... -> FROM ubuntu:22.04
            # Pattern: @sha256:<64 hex chars>
            content = re.sub(
                r'(@sha256:[0-9a-f]{64})',
                '',
                content,
                flags=re.IGNORECASE
            )

            # Also handle @sha512 if present
            content = re.sub(
                r'(@sha512:[0-9a-f]{128})',
                '',
                content,
                flags=re.IGNORECASE
            )

            if content != original_content:
                with open(dockerfile_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True

            return False

        except Exception as e:
            print(f"[WARNING] Could not strip digest pins: {e}")
            return False

    # Removed: _classify_docker_error_improved - was misclassifying errors
    # Caused FILE_COPY_MISSING → DEPENDENCY_BUILD_TOOLS (wrong!)
    # Trust the original Docker build error detection instead

    def _sanitize_error_for_azure(self, error_log: str, max_length: int = 400) -> str:
        """Sanitize error log to avoid Azure content safety filters."""
        # Remove potentially problematic content
        sanitized = error_log
        
        # Remove URLs that might be flagged
        import re
        sanitized = re.sub(r'https?://[^\s]+', '[URL]', sanitized)
        
        # Remove file paths that might contain sensitive info
        sanitized = re.sub(r'/[^\s:]+/', '[PATH]/', sanitized)
        
        # Remove hex addresses and hashes
        sanitized = re.sub(r'0x[0-9a-fA-F]+', '[HEX]', sanitized)
        sanitized = re.sub(r'[0-9a-f]{32,}', '[HASH]', sanitized)
        
        # Keep only error keywords and context
        lines = sanitized.split('\n')
        important_lines = []
        for line in lines:
            # Keep lines with error indicators
            if any(keyword in line.lower() for keyword in [
                'error', 'failed', 'cannot', 'not found', 'missing', 
                'denied', 'manifest', 'exit code', 'returned'
            ]):
                important_lines.append(line.strip())
            if len('\n'.join(important_lines)) > max_length:
                break
        
        result = '\n'.join(important_lines[:10])  # Max 10 lines
        if len(result) > max_length:
            result = result[:max_length] + '...'
        
        return result if result else "Error details sanitized for safety"

    def _refine_dockerfile_with_error_feedback(
        self, agent, dockerfile_path: Path, error_log_path: Path,
        repo_path: str, repo_name: str, iteration: int, docker_result: Dict = None,
        previous_from_lines: List[str] = None, previous_tools_used: List[str] = None,
        consecutive_validation_errors: int = 0
    ) -> Tuple[bool, List[str]]:
        """
        Refine Dockerfile based on build error feedback with enhanced analysis.
        
        Detects error types (especially IMAGE_PULL) and provides targeted guidance.
        Uses increased iteration limit to allow thorough problem-solving.
        
        Args:
            agent: The planner agent instance
            dockerfile_path: Path to current Dockerfile
            error_log_path: Path to error log file
            repo_path: Path to repository
            repo_name: Repository name
            iteration: Current refinement iteration number
            docker_result: Docker build result dict with stage and failed_command info
            
        Returns:
            Tuple of (Success bool, List of tools used)
        """
        try:
            # Read current Dockerfile
            with open(dockerfile_path, 'r', encoding='utf-8') as f:
                current_dockerfile = f.read()
            
            # Read error log
            with open(error_log_path, 'r', encoding='utf-8') as f:
                error_log = f.read()
            
            # Get absolute path for Dockerfile (agent can read from repo_path context)
            dockerfile_absolute = str(dockerfile_path.absolute())

            # Initialize tools_used for return tracking
            result_tools_used = []

            # Extract stage from docker_result - trust the original classification
            if docker_result:
                stage = docker_result.get('stage', 'BUILD')
                failed_cmd = docker_result.get('failed_command', 'unknown')
                exit_code = docker_result.get('exit_code', 1)
                # Removed: improved error classification override - was causing misclassifications
            else:
                stage = 'BUILD'
                failed_cmd = 'unknown'
                exit_code = 1

            # STAGNATION DETECTION: Check if we are stuck on the same base image
            stagnation_prefix = ""
            current_from = ""
            for line in current_dockerfile.split('\n'):
                if line.strip().upper().startswith('FROM '):
                    current_from = line.strip()
                    break
            
            if previous_from_lines and len(previous_from_lines) >= 2 and current_from:
                # Check last 2 iterations
                if previous_from_lines[-1] == current_from and previous_from_lines[-2] == current_from:
                    self.log(repo_name, "STAGNATION DETECTED: Base image hasn't changed for 3 attempts despite failures!", to_console=True)
                    stagnation_prefix = f"""CRITICAL - STAGNATION DETECTED!
You have tried the base image '{current_from}' multiple times and it keeps failing.
STOP TRYING THE SAME IMAGE. YOU MUST CHANGE THE BASE IMAGE TO SOMETHING ELSE.
Use `DockerImageSearch` to find a completely different tag or base image.
DO NOT use the same tag again.
"""

            # Detect error type for targeted guidance
            error_log_lower = error_log.lower()
            
            # NEW: Detect platform/architecture incompatibility FIRST (before image pull)
            # This is when an image exists but was pulled for wrong architecture (amd64 vs arm64)
            is_platform_error = any(x in error_log_lower for x in [
                "invalidbaseimageplatform", "was pulled with platform", 
                "expected \"linux/arm64\"", "expected \"linux/amd64\"",
                "linux/amd64", "linux/arm64"
            ]) and any(x in error_log_lower for x in ["platform", "architecture", "amd64", "arm64"])
            
            is_image_pull_error = any(x in error_log_lower for x in [
                "failed to resolve", "manifest unknown", "pull access denied", 
                "image not found", "not found: manifest unknown", "no such image"
            ]) and not is_platform_error  # Don't double-count platform errors
            
            is_dependency_error = any(x in error_log_lower for x in [
                "npm err!", "pip install failed", "error: failed to compile",
                "cargo build failed", "maven build failed", "no such file or directory"
            ])
            
            is_build_error = any(x in error_log_lower for x in [
                "returned a non-zero code", "exited with code", "build failed"
            ]) and not is_image_pull_error and not is_platform_error
            
            # Detect syntax errors specifically
            is_syntax_error = any(x in error_log_lower for x in [
                "syntax error", "unexpected token", "parse error", "invalid syntax",
                "unknown instruction", "dockerfile parse error", "failed to parse"
            ])

            # Check specific stage values FIRST (most precise error classification from docker_result)
            # These come from error analysis in empirical_test.py and are more specific than pattern matching
            if stage == "DEPENDENCY_FILE_MISSING":
                self.log(repo_name, f"DEPENDENCY_FILE_MISSING error detected", to_console=True)
                safe_error = self._sanitize_error_for_azure(error_log, max_length=400)

                refinement_query = f"""DEPENDENCY FILE NOT FOUND ERROR

The Dockerfile is trying to COPY a dependency file that doesn't exist at the expected location.

**STEP 1:** Use GrepFiles to search for dependency files (package.json, requirements.txt, pom.xml) in the repository
**STEP 2:** Read current Dockerfile at: {dockerfile_absolute}
**STEP 3:** Fix COPY commands to use correct paths
**STEP 4:** Ensure files are copied BEFORE running install commands

COMMON FIXES:
- Files in subdirectory: COPY ./subdir/package.json ./
- Monorepo: COPY entire workspace first
- Wrong file name: Check actual file names in repo

DO NOT CHANGE BASE IMAGE! Final Answer: ONLY Dockerfile content.

Error: {safe_error}"""

            elif stage == "AGENT_OUTPUT_ERROR":
                self.log(repo_name, "AGENT_OUTPUT_ERROR detected - regenerating cleanly", to_console=True)

                refinement_query = f"""CRITICAL: Your previous output had format errors!

Problems detected:
- You included DOCKERFILE_END or DOCKERFILE_START in the actual Dockerfile content
- OR you used placeholder text like [PATH] or [HASH] instead of real values
- OR you used shell syntax (||, 2>/dev/null) in COPY commands

**STEP 1**: Read the project structure again
**STEP 2**: Generate a CLEAN Dockerfile with NO formatting markers
**STEP 3**: Use REAL image names (e.g., gcc:13, NOT docker.io[PATH]/gcc:13@sha256:[HASH])
**STEP 4**: COPY commands must use Docker syntax ONLY (no bash operators)

WRONG:
```dockerfile
COPY package.json ./ 2>/dev/null || true   ❌ NO shell syntax!
FROM docker.io[PATH]/gcc:13@sha256:[HASH]  ❌ NO placeholders!
CMD ["/bin/bash"]
DOCKERFILE_END   ❌ NO markers!
```

CORRECT:
```dockerfile
FROM gcc:13
COPY package*.json ./
CMD ["/bin/bash"]
```

Read current Dockerfile: {dockerfile_absolute}
OUTPUT: Complete, clean Dockerfile starting with FROM"""

            elif stage == "EXTERNAL_DOWNLOAD_FAILED":
                self.log(repo_name, "EXTERNAL_DOWNLOAD_FAILED - use base image instead", to_console=True)

                refinement_query = f"""DOWNLOAD FAILURE - Don't manually install package managers!

You tried to download Maven/Gradle with curl, but it failed (HTTP 404 or similar).

**SOLUTION**: Use the official base image instead!

WRONG:
```dockerfile
FROM eclipse-temurin:17-jdk
RUN curl https://apache.org/.../apache-maven-3.9.6.tar.gz   ❌ Don't do this!
RUN tar -xzf ...
```

CORRECT:
```dockerfile
FROM maven:3.9-eclipse-temurin-17 AS build  ✅ Use official image!
```

**STEP 1**: Read current Dockerfile: {dockerfile_absolute}
**STEP 2**: Replace base image with official Maven/Gradle image
**STEP 3**: Remove all curl/wget/manual installation commands

Available official images:
- Maven: maven:3.9-eclipse-temurin-17
- Gradle: gradle:8-jdk17

OUTPUT: Complete Dockerfile using official base image"""

            elif stage == "BUILD_FAILED":
                self.log(repo_name, "BUILD_FAILED - actual compilation error", to_console=True)
                safe_error = self._sanitize_error_for_azure(error_log, max_length=400)

                refinement_query = f"""BUILD COMMAND FAILED - This is a compilation/build error

The build command itself failed (make, ninja, pnpm run build, etc.)
This is NOT a missing dependency - tools are installed correctly.

This could be:
1. Source code incompatibility
2. Build configuration error
3. TypeScript/compilation error

**STEP 1**: Read the error log carefully
**STEP 2**: Use SearchWeb with the specific error message
**STEP 3**: Try simplifying the build:
   - Add flags: --skip-tests, --no-verify
   - Different target: compile instead of package
   - Ignore errors: || true

Example fixes:
```dockerfile
RUN pnpm run build || pnpm run compile  # Try alternative command
RUN npm run build -- --skip-tests       # Skip tests
RUN make || true                        # Ignore errors
```

Read Dockerfile: {dockerfile_absolute}
Error: {safe_error}

OUTPUT: Modified Dockerfile with adjusted build command"""

            elif stage == "DEPENDENCY_MISSING_LIBRARY":
                self.log(repo_name, "DEPENDENCY_MISSING_LIBRARY - specific library needed", to_console=True)
                safe_error = self._sanitize_error_for_azure(error_log, max_length=400)

                refinement_query = f"""MISSING SPECIFIC LIBRARY - Meson/CMake can't find a dependency

Meson or CMake is looking for a specific library that's not installed.

**STEP 1**: Read the error - what library is it looking for?
**STEP 2**: Use SearchWeb with: "debian install [library_name]-dev package"
**STEP 3**: Add that specific package to RUN apt-get install

Example:
If error says "Could not find libfoo"
→ Search: "debian install libfoo-dev"
→ Add: libfoo-dev to apt-get install command

Read Dockerfile: {dockerfile_absolute}
Error: {safe_error}

OUTPUT: Dockerfile with the specific missing library added"""

            elif stage == "DEPENDENCY_BUILD_TOOLS":
                self.log(repo_name, f"DEPENDENCY_BUILD_TOOLS error detected", to_console=True)
                safe_error = self._sanitize_error_for_azure(error_log, max_length=400)

                # Detect error patterns instead of hardcoded package names
                patterns = ErrorPatternDetector.detect_patterns(error_log, failed_cmd)

                if patterns:
                    # Read current Dockerfile to detect OS
                    try:
                        with open(dockerfile_absolute, 'r') as f:
                            current_dockerfile = f.read()
                        base_os = ErrorPatternDetector.detect_base_os(current_dockerfile)
                    except:
                        base_os = 'debian'

                    pkg_manager = ErrorPatternDetector.get_package_manager(base_os)

                    self.log(repo_name, f"Detected build tool error patterns: {patterns} on {base_os}", to_console=True)

                    # Build search query based on detected patterns
                    pattern_descriptions = {
                        'fortran_compiler_needed': 'fortran compiler gfortran',
                        'blas_library_needed': 'BLAS LAPACK linear algebra libraries',
                        'python_dev_headers': 'python development headers',
                        'c_compiler_needed': 'gcc C compiler build-essential',
                        'cpp_compiler_needed': 'g++ C++ compiler'
                    }

                    search_terms = ' '.join([pattern_descriptions.get(p, p) for p in patterns])

                    refinement_query = f"""MISSING BUILD TOOLS ERROR - Pattern-Based Discovery

Detected missing: {', '.join(patterns)}
Base OS: {base_os}
Package Manager: {pkg_manager}

**DISCOVERY APPROACH** (DO NOT GUESS):

**STEP 1:** Read current Dockerfile: {dockerfile_absolute}
**STEP 2:** Use SearchWeb with query: "docker {base_os} {pkg_manager} install {search_terms}"
**STEP 3:** Based on search results, add RUN command BEFORE dependency install

EXAMPLES of what SearchWeb might find:
- For Alpine (apk): apk add --no-cache build-base gfortran openblas-dev
- For Debian (apt-get): apt-get install build-essential gfortran libopenblas-dev
- For RedHat (yum): yum install gcc-gfortran blas-devel lapack-devel

CRITICAL RULES:
- USE SearchWeb to discover the correct package names for {base_os}
- Add packages BEFORE the failing command
- DO NOT change base image
- DO NOT use hardcoded package names without verifying via SearchWeb

Error context: {safe_error}

Final Answer: ONLY Dockerfile content."""

                else:
                    # No patterns detected - use general discovery approach
                    refinement_query = f"""MISSING BUILD TOOLS ERROR

Native dependencies need build tools (compilers, headers) not in base image.

**STEP 1:** Read current Dockerfile at: {dockerfile_absolute}
**STEP 2:** Extract the base image OS and programming language from Dockerfile
**STEP 3:** Read dependency files (package.json, requirements.txt, Cargo.toml, pom.xml) to identify packages
**STEP 4:** Analyze error log to find specific missing commands or libraries
**STEP 5:** Use SearchWeb with specific query based on what you found:
   - Format: "docker [OS] install build tools for [language] [failing_package_or_command]"
   - Example: "docker alpine install build tools for python when gcc missing"
   - Example: "docker debian install dependencies for node-gyp"
   - Example: "docker ubuntu install libssl-dev for rust openssl"
**STEP 6:** Based on search results, add RUN command BEFORE dependency install

DISCOVERY APPROACH:
1. Be SPECIFIC in your search query - include OS, language, and the failing package/command
2. Look for official documentation in search results
3. If first search doesn't help, try a different query with more context from the error
4. Check error log for library names (libssl, libpq, libffi, etc.) and search for those specifically

DO NOT GUESS OR USE HARDCODED EXAMPLES!
USE SEARCHWEB TO DISCOVER THE CORRECT SOLUTION!
DO NOT CHANGE BASE IMAGE!

Final Answer: ONLY Dockerfile content.

Error: {safe_error}"""

            elif stage == "DEPENDENCY_LOCKFILE_MISMATCH":
                self.log(repo_name, f"DEPENDENCY_LOCKFILE_MISMATCH error detected", to_console=True)
                safe_error = self._sanitize_error_for_azure(error_log, max_length=400)

                refinement_query = f"""LOCKFILE MISMATCH ERROR

Lockfile is out of sync with package.json.

**STEP 1:** Read current Dockerfile at: {dockerfile_absolute}
**STEP 2:** Change install command:

INSTEAD OF: npm ci (requires exact lockfile match)
USE: npm install (generates new lockfile)

INSTEAD OF: pnpm install --frozen-lockfile
USE: pnpm install

DO NOT CHANGE BASE IMAGE! Final Answer: ONLY Dockerfile content.

Error: {safe_error}"""

            elif stage == "FILE_COPY_MISSING":
                self.log(repo_name, f"FILE_COPY_MISSING error detected", to_console=True)
                safe_error = self._sanitize_error_for_azure(error_log, max_length=400)

                refinement_query = f"""FILE COPY ERROR - Source Path Doesn't Exist

Your Dockerfile has a COPY command that references a path which doesn't exist in the repository.

**This is a path/directory issue, not an image or dependency problem.**

**STEP 1 - IDENTIFY THE PROBLEM**: Read Dockerfile at: {dockerfile_absolute}
   Look for the COPY command that's failing (the error will tell you which one)

**STEP 2 - SEE WHAT ACTUALLY EXISTS**: Use DirectoryTree to see the repository structure
   Compare what the Dockerfile expects vs. what's actually there

**STEP 3 - FIX THE COPY COMMAND** (choose the best option):
   Option A: Fix the path if it's just wrong (e.g., COPY src → COPY source)
   Option B: Remove the COPY line if the source truly doesn't exist and isn't needed
   Option C: Use COPY . /app to copy the whole project (if unsure what to copy)
   Option D: Check .dockerignore if the file exists but is being ignored

**HELPFUL HINTS**:
- This is about fixing file paths, not changing the Docker image
- The base image is likely fine - focus on getting the correct source paths
- If you're unsure which directory to copy, DirectoryTree will show you the actual structure

Error details: {safe_error}

OUTPUT: Fixed Dockerfile with correct COPY paths"""

            elif stage == "BUILD_TOOL_MISSING":
                self.log(repo_name, f"BUILD_TOOL_MISSING error detected", to_console=True)
                safe_error = self._sanitize_error_for_azure(error_log, max_length=400)

                refinement_query = f"""BUILD TOOL/SCRIPT MISSING ERROR

Build script or tool not found/executable.

**STEP 1:** Use GrepFiles to find the build script in the repository
**STEP 2:** Read current Dockerfile at: {dockerfile_absolute}
**STEP 3:** Fix path or ensure file is executable (chmod +x)
**STEP 4:** Check if tool needs to be installed first

DO NOT CHANGE BASE IMAGE! Final Answer: ONLY Dockerfile content.

Error: {safe_error}"""

            elif stage == "IMAGE_VALIDATION_FAILED":
                self.log(repo_name, f"IMAGE_VALIDATION_FAILED error detected", to_console=True)
                safe_error = self._sanitize_error_for_azure(error_log, max_length=300)

                # Extract problematic image
                problematic_image = "unknown"
                for line in current_dockerfile.split('\n'):
                    line_stripped = line.strip()
                    if line_stripped.upper().startswith('FROM '):
                        image_part = line_stripped[5:].split()[0]
                        if ' as ' in image_part.lower():
                            image_part = image_part.split()[0]
                        problematic_image = image_part
                        break

                base_image_name = problematic_image.split(':')[0] if problematic_image != "unknown" else "unknown"

                refinement_query = f"""IMAGE VALIDATION FAILED

The base image "{problematic_image}" could not be validated (may be invalid, deprecated, or unavailable).

**STEP 1:** Use SearchDockerError with "docker image validation failed {base_image_name}"
**STEP 2:** Use DockerImageSearch with "tags:{base_image_name}" to find valid tags
**STEP 3:** Pick a modern verified tag
**STEP 4:** Update Dockerfile FROM line

MUST USE DockerImageSearch to verify! Final Answer: ONLY Dockerfile content.

Error: {safe_error}"""

            # Handle pattern-based error detection (fallback for cases without specific stage classification)
            elif is_platform_error:
                # Detect host platform dynamically
                import platform
                host_arch = platform.machine().lower()
                if host_arch in ['arm64', 'aarch64']:
                    host_docker_arch = 'arm64'
                    expected_platform = 'linux/arm64'
                    arch_name = 'ARM64'
                    arch_devices = 'Apple Silicon M1/M2/M3/M4'
                elif host_arch in ['x86_64', 'amd64']:
                    host_docker_arch = 'amd64'
                    expected_platform = 'linux/amd64'
                    arch_name = 'AMD64'
                    arch_devices = 'Intel/AMD x86_64'
                else:
                    host_docker_arch = host_arch
                    expected_platform = f'linux/{host_arch}'
                    arch_name = host_arch.upper()
                    arch_devices = f'{host_arch} system'
                
                self.log(repo_name, f"PLATFORM_INCOMPATIBLE error detected - need {arch_name} compatible image", to_console=True)
                
                # Extract the problematic image from Dockerfile
                problematic_image = "unknown"
                try:
                    for line in current_dockerfile.split('\n'):
                        line_stripped = line.strip()
                        if line_stripped.upper().startswith('FROM '):
                            image_part = line_stripped[5:].split()[0]
                            if ' as ' in image_part.lower():
                                image_part = image_part.split()[0]
                            problematic_image = image_part
                            break
                except Exception:
                    pass
                
                base_image_name = problematic_image.split(':')[0] if problematic_image != "unknown" else "unknown"
                safe_error = self._sanitize_error_for_azure(error_log, max_length=300)
                
                refinement_query = f"""Platform/Architecture Compatibility Issue

The Docker image "{problematic_image}" doesn't support your system's architecture.

Your system: {arch_name} ({arch_devices} - {expected_platform})
The problem: This image tag was built for a different architecture and won't work here

**HOW TO FIX THIS**:

**STEP 1 - LIST AVAILABLE TAGS**: Use DockerImageSearch with "tags:{base_image_name}"
   This will show you all available tags with compatibility markers:
   - [OK] means compatible with {arch_name}
   - [!!] means NOT compatible (skip these)

   Look for:
   - Tags marked [OK] for your architecture
   - Recent versions (2023+) usually have better multi-arch support
   - Prefer: -slim, -bookworm variants (better ARM64 support)
   - Avoid: -alpine variants if you see compatibility issues (sometimes lack ARM64 builds)

**STEP 2 - VERIFY YOUR CHOICE**: Use DockerImageSearch with "{base_image_name}:<your-chosen-tag>"
   Double-check that "Architectures" field includes: {host_docker_arch}
   This confirms it will work on your system

**STEP 3 - UPDATE DOCKERFILE**: Read {dockerfile_absolute}
   Replace the FROM line with your verified {arch_name}-compatible image

**HELPFUL TIPS**:
- Most official images have multi-arch support in recent versions
- If the image name includes "alpine", try the "slim" variant instead for better compatibility
- You can use the same version number, just different variant (e.g., 17-alpine → 17-slim)

Error details: {safe_error}

OUTPUT: Updated Dockerfile with {arch_name}-compatible base image"""

            # Build targeted refinement query based on error type
            elif is_image_pull_error:
                self.log(repo_name, f"IMAGE_PULL error detected - instructing agent to verify images", to_console=True)
                
                # Extract the problematic image from error log - try multiple patterns
                problematic_image = "unknown"
                
                # Pattern 1: Look for "FROM <image>" in error
                for line in error_log.split('\n'):
                    if 'from' in line.lower() and ('manifest' in line.lower() or 'pull' in line.lower() or 'resolve' in line.lower()):
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part.lower() == 'from' and i + 1 < len(parts):
                                problematic_image = parts[i + 1].strip('":')
                                break
                        if problematic_image != "unknown":
                            break
                
                # Pattern 2: If still unknown, try to read from Dockerfile directly
                if problematic_image == "unknown":
                    try:
                        with open(dockerfile_path, 'r', encoding='utf-8') as f:
                            dockerfile_content = f.read()
                        # Find first FROM statement
                        for line in dockerfile_content.split('\n'):
                            line_stripped = line.strip()
                            if line_stripped.startswith('FROM '):
                                # Extract image after FROM
                                image_part = line_stripped[5:].split()[0]  # Get first word after FROM
                                # Remove AS alias if present
                                if ' as ' in image_part.lower():
                                    image_part = image_part.split()[0]
                                problematic_image = image_part
                                self.log(repo_name, f"Extracted image from Dockerfile: {problematic_image}", to_console=True)
                                break
                    except Exception as e:
                        self.log(repo_name, f"Could not read Dockerfile to extract image: {e}", to_console=False)
                
                # Sanitize error for Azure safety
                safe_error = self._sanitize_error_for_azure(error_log, max_length=300)
                
                # Extract base image name (without tag) for searching
                base_image_name = problematic_image.split(':')[0].split('@')[0] if problematic_image != "unknown" else "unknown"

                # Detect repeated validation failures and adjust strategy
                urgency_note = ""
                search_strategy = ""

                if consecutive_validation_errors >= 2:
                    urgency_note = f"\n**CRITICAL**: You've tried {consecutive_validation_errors} times with validation errors. The image is likely DEPRECATED/UNAVAILABLE.\n"
                    search_strategy = f"""
**DISCOVERY STRATEGY** (Find modern alternatives):
1. Use SearchWeb with "docker hub {base_image_name} modern alternatives 2025" to discover recommended replacements
2. Use DockerImageSearch with "tags:{base_image_name}" to see if ANY tags are still available
3. If no tags found or all deprecated, search for alternative images that serve the same purpose
4. Once you identify an alternative, verify it exists with DockerImageSearch
"""
                else:
                    search_strategy = f"""
**VERIFICATION STRATEGY**:
1. Use DockerImageSearch with "tags:{base_image_name}" to list available tags
2. Pick a modern, stable version tag from the list (prefer LTS versions)
3. Use DockerImageSearch with "{base_image_name}:<tag>" to verify it exists
"""

                # General-purpose prompt that discovers solutions dynamically
                refinement_query = f"""CRITICAL: Base image "{problematic_image}" FAILED!
{urgency_note}
{search_strategy}
**THEN**:
4. Read current Dockerfile: {dockerfile_absolute}
5. Update FROM line with your VERIFIED working image
6. Provide Final Answer with ONLY Dockerfile content (no explanations)

RULES:
- DO NOT reuse "{problematic_image}" - it failed validation
- MUST use tools (SearchWeb + DockerImageSearch) to discover and verify alternatives
- If image has no available tags, find a modern replacement that serves the same purpose
- Final Answer: ONLY Dockerfile content starting with FROM

Error: {safe_error}"""

            elif is_dependency_error:
                self.log(repo_name, f"DEPENDENCY error detected - instructing agent to check dependencies", to_console=True)
                
                # Sanitize error for Azure
                safe_error = self._sanitize_error_for_azure(error_log, max_length=400)
                
                # Use PackageManagerDetector for general-purpose detection
                pm_name, error_keywords = PackageManagerDetector.detect_package_manager(error_log)
                pkg_name = PackageManagerDetector.extract_package_name(error_log, pm_name) if pm_name else ""

                if pm_name:
                    self.log(repo_name, f"Detected package manager: {pm_name}", to_console=True)

                    # Enhance query with package name if found
                    if pkg_name:
                        error_keywords = f"{error_keywords} {pkg_name}"
                else:
                    self.log(repo_name, f"No specific package manager detected, using generic approach", to_console=True)
                
                refinement_query = f"""DOCKERFILE DEPENDENCY ERROR - Package installation failed!

IMPORTANT: This is NOT an image problem - DO NOT change the base image!

**STEP 1:** Use SearchDockerError with "{error_keywords}" to find solutions
**STEP 2:** Read the current Dockerfile at: {dockerfile_absolute}
**STEP 3:** Read the project dependency files (package.json, requirements.txt, pom.xml, pyproject.toml, Gemfile, composer.json, go.mod, Cargo.toml, etc.)
**STEP 4:** **CHECK FOR NATIVE DEPENDENCIES**:
   - Look at the dependency file for packages that might need C/C++ compilation
   - Use SearchWeb with: "[package_name] docker build dependencies" to check if it needs native tools
   - Common indicators: packages with "native", "binding", or that interface with databases/crypto/images
   - If uncertain, search: "does [package_name] require gcc or build tools"
**STEP 5:** If native dependencies found, use SearchWeb to find required build tools for that specific package
**STEP 6:** Fix the RUN commands to install missing system packages BEFORE dependency install

CRITICAL RULES:
- DO NOT CHANGE THE BASE IMAGE (FROM line) - it is CORRECT!
- ONLY fix RUN commands to add missing dependencies
- Use SearchWeb to discover what's needed - DO NOT GUESS!
- Add build tools BEFORE the dependency install command
- Your Final Answer MUST start with the SAME FROM line as the current Dockerfile

WRONG: Changing FROM maven:... to FROM node:... (DO NOT DO THIS!)
CORRECT: Keep FROM maven:... and add RUN apt-get install [searched-packages]

Error: {safe_error}

OUTPUT FORMAT - Your answer must be ONLY Dockerfile content starting with FROM:"""

            elif is_syntax_error:
                # SYNTAX ERROR: Use web search with sanitized error
                self.log(repo_name, f"SYNTAX error detected - using web search with sanitized error", to_console=True)
                
                # Sanitize error for web search - extract key syntax error message
                safe_error = self._sanitize_error_for_azure(error_log, max_length=400)
                
                # Extract the specific syntax error for search query
                syntax_search_query = "dockerfile syntax error"
                syntax_error_detail = ""
                for line in error_log.split('\n'):
                    line_lower = line.lower()
                    if any(x in line_lower for x in ['syntax error', 'unexpected', 'unknown instruction', 'parse error']):
                        # Sanitize this line for search
                        syntax_error_detail = self._sanitize_error_for_azure(line, max_length=100)
                        syntax_search_query = f"dockerfile {syntax_error_detail}"
                        break
                
                # Extract problematic lines from Dockerfile if line number is mentioned in error
                problematic_context = ""
                import re
                line_match = re.search(r'(?:line|Dockerfile:)\s*(\d+)', error_log, re.IGNORECASE)
                if line_match:
                    error_line_num = int(line_match.group(1))
                    dockerfile_lines = current_dockerfile.split('\n')
                    # Show 3 lines before and after the error line
                    start_line = max(0, error_line_num - 4)
                    end_line = min(len(dockerfile_lines), error_line_num + 3)
                    context_lines = []
                    for i in range(start_line, end_line):
                        prefix = ">>> " if i == error_line_num - 1 else "    "
                        context_lines.append(f"{prefix}Line {i+1}: {dockerfile_lines[i]}")
                    problematic_context = f"""
PROBLEMATIC DOCKERFILE SECTION (line {error_line_num} marked with >>>):
{chr(10).join(context_lines)}
"""
                
                refinement_query = f"""DOCKERFILE SYNTAX ERROR - Parse error in Dockerfile!

IMPORTANT: This is a SYNTAX problem - DO NOT change the base image!
{problematic_context}
**STEP 1:** Analyze the error message and the problematic line shown above
**STEP 2:** Use SearchWeb with "{syntax_search_query}" to understand the issue
**STEP 3:** Read the full Dockerfile at: {dockerfile_absolute}
**STEP 4:** Fix ONLY the syntax error and related content - keep all other content exactly the same

HINTS for common syntax errors:
- "unknown instruction: <word>" often means multi-line content (heredoc/script) is being parsed as Dockerfile instructions
- "unexpected token" usually means missing quotes, escapes, or line continuation issues
- Each line in a Dockerfile must start with a valid instruction (FROM, RUN, COPY, etc.) or be a continuation

CRITICAL RULES:
- DO NOT CHANGE THE BASE IMAGE (FROM line) - it is CORRECT!
- ONLY fix the syntax error on the specific line mentioned
- Your Final Answer MUST start with FROM and be ONLY Dockerfile content
- NO explanatory text before or after the Dockerfile

WRONG OUTPUT:
"Below is a corrected Dockerfile..."
"Here is the fix..."

CORRECT OUTPUT:
FROM same-image:tag
WORKDIR /app
...

Error: {safe_error}

OUTPUT FORMAT - Your answer must be ONLY Dockerfile content starting with FROM:"""

            else:
                self.log(repo_name, f"BUILD error detected - instructing agent to analyze carefully", to_console=True)

                # Sanitize error
                safe_error = self._sanitize_error_for_azure(error_log, max_length=400)
                
                refinement_query = f"""DOCKERFILE BUILD ERROR at stage: {stage}

IMPORTANT: This is NOT an image problem - DO NOT change the base image!

**STEP 1:** Use SearchDockerError to find solutions for this error type
**STEP 2:** Read the current Dockerfile at: {dockerfile_absolute}
**STEP 3:** Identify what command failed: {failed_cmd}
**STEP 4:** Fix the specific failing command

CRITICAL RULES:
- DO NOT CHANGE THE BASE IMAGE (FROM line) - it is CORRECT!
- ONLY fix the RUN/COPY/WORKDIR commands that are failing
- Your Final Answer MUST start with the SAME FROM line as current Dockerfile
- NO explanatory text - ONLY Dockerfile content

Error: {safe_error}

OUTPUT FORMAT - Your answer must be ONLY Dockerfile content starting with FROM:"""

            self.log(repo_name, f"Requesting enhanced Dockerfile refinement (iteration {iteration})...", to_console=False)
            
            # If this is not the first iteration, check if agent was lazy (didn't use search tools)
            tool_warning = ""
            if iteration > 0 and previous_tools_used:
                # List of search/investigation tools
                search_tools = ['SearchDockerError', 'SearchWeb', 'DockerImageSearch', 'GoogleSearch', 'BingSearch']
                used_search = any(t in search_tools for t in previous_tools_used)
                
                if not used_search:
                    self.log(repo_name, "Agent did not use search tools in previous iteration - injecting warning", to_console=True)
                    tool_warning = """
WARNING: In the previous iteration, you attempted to fix the error WITHOUT using any search tools.
You are blindly guessing!
YOU MUST USE 'SearchDockerError' or 'SearchWeb' to understand the error before proposing a fix.
"""

            # Invoke agent with max_iterations=8 but simpler prompts to force tool usage
            # Shorter prompts = agent focuses more on tool calls vs text generation
            from run_agent import _invoke_agent_with_iteration_limit
            
            try:
                result = _invoke_agent_with_iteration_limit(
                    agent,
                    {
                        "input": stagnation_prefix + tool_warning + refinement_query,
                        "chat_history": f"Repository: {repo_name}. Dockerfile iteration {iteration}. Use tools to investigate."
                    },
                    max_iterations=15  # Increased from 10 to give agent more attempts to recover from parsing errors
                )
            except (ValueError, Exception) as e:
                # Fallback: if agent errors out (early_stopping or iteration limit), try direct LLM call
                if "early_stopping_method" in str(e) or "iteration" in str(e).lower():
                    self.log(repo_name, f"Agent hit iteration limit, using fallback LLM call", to_console=True)
                    try:
                        # Direct LLM call without agent framework
                        from langchain_openai import AzureChatOpenAI
                        import os
                        fallback_llm = AzureChatOpenAI(
                            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
                            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
                        )
                        # Simplify query for direct LLM
                        simple_query = f"Fix this Dockerfile. Current Dockerfile:\\n{current_dockerfile}\\n\\nError: {safe_error if 'safe_error' in locals() else 'Build failed'}\\n\\nProvide ONLY the corrected Dockerfile, no explanations."
                        fallback_result = fallback_llm.invoke(simple_query)
                        result = {"output": fallback_result.content, "intermediate_steps": []}
                    except Exception as fallback_error:
                        self.log(repo_name, f"Fallback LLM also failed: {fallback_error}", to_console=False)
                        return False, []
                else:
                    raise  # Re-raise if not iteration limit error
            
            # Log tool usage during refinement for debugging
            if 'intermediate_steps' in result:
                tools_used = [action.tool for action, _ in result['intermediate_steps']]
                result_tools_used = tools_used  # Store for return
                self.log(repo_name, f"Refinement tools used: {', '.join(tools_used) if tools_used else 'none'}", to_console=True)
                
                if is_image_pull_error and 'DockerImageSearch' not in tools_used:
                    self.log(repo_name, "WARNING: IMAGE_PULL error but agent didn't use DockerImageSearch!", to_console=True)
                    # We could add this warning to the next prompt, but stagnation logic above covers it better now
            
            refined_output = result.get('output', '').strip()
            
            if not refined_output:
                self.log(repo_name, "Agent returned empty output for refinement", to_console=False)
                return False, result_tools_used
            
            # Extract Dockerfile content from output (may be wrapped in code blocks or have prose)
            refined_dockerfile = refined_output
            
            # Remove markdown code blocks if present
            if '```' in refined_dockerfile:
                lines = refined_dockerfile.split('\n')
                # Find content between ``` markers
                in_code_block = False
                code_lines = []
                for line in lines:
                    if line.strip().startswith('```'):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block:
                        code_lines.append(line)
                if code_lines:
                    refined_dockerfile = '\n'.join(code_lines)
            
            # If output starts with prose (not FROM), try to find FROM and extract from there
            lines = refined_dockerfile.split('\n')
            from_index = -1
            for i, line in enumerate(lines):
                stripped = line.strip()
                # Look for FROM line (case insensitive check but preserve original)
                if stripped.upper().startswith('FROM '):
                    from_index = i
                    break
            
            if from_index > 0:
                # There's text before FROM - extract only from FROM onwards
                self.log(repo_name, f"Extracting Dockerfile from line {from_index} (removing {from_index} lines of prose)", to_console=False)
                refined_dockerfile = '\n'.join(lines[from_index:])
            elif from_index == -1:
                # No FROM found at all - this is a failed refinement
                self.log(repo_name, "Refined output does not contain FROM statement, refinement may have failed", to_console=True)
                return False, result_tools_used
            
            # Final cleanup: remove any trailing prose after Dockerfile content
            # Dockerfile ends when we see patterns like "This Dockerfile..." or explanation text
            final_lines = []
            for line in refined_dockerfile.split('\n'):
                stripped = line.strip()
                # Stop if we hit prose/explanation (but keep empty lines and comments)
                if stripped and not stripped.startswith('#') and not any(stripped.upper().startswith(kw) for kw in 
                    ['FROM', 'RUN', 'COPY', 'WORKDIR', 'ENV', 'EXPOSE', 'CMD', 'ENTRYPOINT', 'ARG', 'LABEL', 'ADD', 'VOLUME', 'USER', 'HEALTHCHECK', 'SHELL', 'ONBUILD', 'STOPSIGNAL']):
                    # Check if this looks like prose
                    if any(word in stripped.lower() for word in ['this dockerfile', 'the dockerfile', 'i have', 'note:', 'explanation', 'above', 'below', 'following']):
                        break
                final_lines.append(line)
            
            refined_dockerfile = '\n'.join(final_lines).strip()
            
            # Final validation: must have FROM
            if 'FROM' not in refined_dockerfile.upper():
                self.log(repo_name, "Refined output does not contain FROM statement after cleanup, refinement may have failed", to_console=True)
                return False, result_tools_used
            
            # Check if Dockerfile actually changed
            if refined_dockerfile.strip() == current_dockerfile.strip():
                self.log(repo_name, "Refined Dockerfile is identical to current one", to_console=True)
                return False, result_tools_used
            
            # Log what changed for debugging
            current_from = None
            refined_from = None
            for line in current_dockerfile.split('\n'):
                if line.strip().startswith('FROM '):
                    current_from = line.strip()
                    break
            for line in refined_dockerfile.split('\n'):
                if line.strip().startswith('FROM '):
                    refined_from = line.strip()
                    break
            
            if current_from != refined_from:
                self.log(repo_name, f"Base image changed: {current_from} → {refined_from}", to_console=True)
            else:
                self.log(repo_name, f"Dockerfile modified (base image unchanged)", to_console=True)
            
            # Backup current Dockerfile
            backup_path = dockerfile_path.parent / f"Dockerfile.backup_iter{iteration-1}"
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(current_dockerfile)
            
            # Write refined Dockerfile
            with open(dockerfile_path, 'w', encoding='utf-8') as f:
                f.write(refined_dockerfile)
            
            self.log(repo_name, f"Dockerfile updated (backup: {backup_path.name})", to_console=False)
            
            return True, result_tools_used
            
        except Exception as e:
            self.log(repo_name, f"Exception during Dockerfile refinement: {e}", to_console=False)
            import traceback
            self.log(repo_name, f"Traceback: {traceback.format_exc()}", to_console=False)
            return False, []

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
            # Use slug from result for consistency
            slug = result.get("repo_slug", repo_name.lower())
            image_name = f"parallel-empirical-{slug}:latest"
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
        print("="*80)
        print(f"PARALLEL EMPIRICAL TESTING - {len(repo_urls)} repositories")
        print(f"Workers: {self.max_workers}")
        print("="*80)
        print(f"\nResults directory: {self.results_dir.absolute()}")
        print(f"Master log: {self.master_log_file}")

        # Pre-test infrastructure health check
        print("\n[HEALTH CHECK] Verifying Docker infrastructure...")
        healthy, health_message = self.docker_tester.check_infrastructure_health()

        if not healthy:
            print(f"[HEALTH CHECK] FAILED: {health_message}")
            if "corrupted" in health_message.lower():
                print("[HEALTH CHECK] Attempting to fix corruption with docker builder prune...")
                if self.docker_tester.prune_build_cache():
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
                ["docker", "images", "-q", "-f", "reference=parallel-empirical-*"],
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
            if "agent_analysis" in r and "token_usage" in r["agent_analysis"]:
                token_totals.append(r["agent_analysis"]["token_usage"].get("total", 0))
            if "agent_analysis" in r and "cost_usd" in r["agent_analysis"]:
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

                if "agent_analysis" in result and "token_usage" in result["agent_analysis"]:
                    tokens = result["agent_analysis"]["token_usage"]
                    f.write(f"   Tokens: {tokens.get('total', 0):,}\n")
                if "agent_analysis" in result and "cost_usd" in result["agent_analysis"]:
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