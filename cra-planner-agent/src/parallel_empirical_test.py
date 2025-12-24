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
# Removed legacy imports
# from run_agent import clone_repository, detect_project_language, analyze_repository
# from planner_agent import create_planner_agent
from agent.core import _get_host_platform

from agent.validation import DockerBuildTester

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
        ],
        'npm_lifecycle_hooks_error': [
            'husky install',
            "husky - can't create hook",
            '.husky directory doesn\'t exist',
            'prepare:hooks',
            'husky - git command not found'
        ],
        'apt_package_not_found': [
            'unable to locate package',
            'has no installation candidate',
            'package .* has no installation candidate',
            'e: package',
            'couldn\'t find any package by regex'
        ],
        'npm_invalid_config': [
            'is not a valid npm option',
            'unknown option',
            'invalid config key',
            'npm error unknown',
            'err! config'
        ],
        'vendored_build_system': [
            'vendored-meson',
            'could not find the specified meson',
            'vendored build',
            'custom build backend',
            'meson-python: error: could not find'
        ],
        'git_required_but_missing': [
            'git: command not found',
            'fatal: not a git repository',
            'git repository required',
            'needs git',
            'requires git to be installed'
        ],
        'cmake_needed': [
            'cmake: command not found',
            'cmake not found',
            'could not find cmake',
            'cmake is required'
        ],
        'pkg_config_needed': [
            'pkg-config: command not found',
            'pkg-config not found',
            'no package .*pc found'
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
            try:
                from run_agent import clone_repository # Import strictly if needed, or assume it's moved
                # Wait, I removed clone_repository from run_agent.py? 
                # Yes, I made run_agent.py minimal.
                # Use the local clone helper if I kept it? 
                # I need to restore clone_repository or define it here.
                # Actually, parallel_empirical_test.py imported it from run_agent.
                # I MUST DEFINE IT HERE or import from somewhere.
                # Ideally, I should have put it in tools/utils.
                # I'll implement a simple clone here to be safe.
                pass 
            except ImportError:
                 pass

            # Define simple clone wrapper since I removed it from run_agent
            def _clone(url, target_base):
                import subprocess
                slug = repo_slug(url)
                target = Path(target_base) / slug
                if target.exists():
                    import shutil
                    shutil.rmtree(target)
                target.mkdir(parents=True, exist_ok=True)
                subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True, capture_output=True)
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

            def validation_callback(path):
                """Callback for agent to verify its own work via Docker build."""
                dockerfile_path = Path(path) / "Dockerfile"
                if not dockerfile_path.exists():
                    return {"success": False, "error": "Dockerfile not found at root of repository"}
                
                image_name = f"learner-{slug}:latest"
                build_res = self.docker_tester.build_dockerfile(
                    str(dockerfile_path), 
                    path, 
                    image_name
                )
                
                if build_res["success"]:
                    self.docker_tester.cleanup_image(image_name)
                    return {"success": True}
                else:
                    # Provide detailed feedback for the refinement loop
                    error_snippet = build_res.get('error_snippet', 'Unknown error')
                    error_stage = build_res.get('stage', 'UNKNOWN')
                    full_error = build_res.get('error_message', '')[:500]  # Truncate for prompt
                    return {
                        "success": False, 
                        "error": f"Stage: {error_stage}\nError: {error_snippet}\nDetails: {full_error}"
                    }

            self.log(repo_name, "Running Learner Agent...", to_console=True)
            start_time = time.time()
            
            # Define Verification Tool for the Agent
            from langchain_core.tools import Tool
            
            def verify_build_tool_func(input_str: str) -> str:
                """
                Verifies the Dockerfile by attempting a build.
                Input is ignored (it looks for 'Dockerfile' in the root).
                """
                # The agent writes to repo_path/Dockerfile, so we check there.
                # The `report_dir_result` variable is not defined here, so we assume
                # the Dockerfile is always in repo_path.
                dockerfile_path = Path(repo_path) / "Dockerfile"
                
                if not dockerfile_path.exists():
                    return "Error: Dockerfile not found. You must write the file first."
                
                # Use a specific tag for verification
                verify_slug = f"verify-{slug}"
                verify_image = f"verify-{verify_slug}:latest"
                
                result = self.docker_tester.build_dockerfile(
                    str(dockerfile_path),
                    repo_path,
                    verify_image
                )
                
                if result.get("success"):
                    self.docker_tester.cleanup_image(verify_image)
                    return "SUCCESS: Built successfully."
                else:
                    return f"BUILD FAILED at stage {result.get('stage')}: {result.get('error_snippet')}\nFull Error:\n{result.get('error_message')}"

            verify_tool = Tool(
                name="VerifyBuild",
                func=verify_build_tool_func,
                description="Verifies the 'Dockerfile' by running a real Docker build. usage: VerifyBuild('')"
            )

            # Prepare callback handler
            callback_handler = FormattedOutputHandler()

            # Run Learner Agent with Validation Tool
            print(f"[{repo_name}] Running Learner Agent...")
            agent_result = run_learner_agent(
                repo_path=repo_path,
                repo_name=repo_name,
                repo_url=repo_url,
                max_retries=3,  # 3 attempts with feedback injection
                callback_handler=callback_handler,
                validation_callback=validation_callback,
                extra_tools=[verify_tool]
            )
            
            duration = time.time() - start_time # Use start_time from original code
            result["agent_analysis"] = agent_result
            result["total_duration"] = duration
            
            if agent_result["status"] == "success":
                result["success"] = True
                self.log(repo_name, f"SUCCESS! Dockerfile built and verified in {duration:.1f}s", to_console=True)
            else:
                self.log(repo_name, f"FAILURE: {agent_result.get('error')}", to_console=True)

        except Exception as e:
            self.log(repo_name, f"Test crashed: {e}", to_console=True)
            traceback.print_exc()
        
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