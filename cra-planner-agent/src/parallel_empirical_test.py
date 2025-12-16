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
from typing import Dict, List, Optional
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
                max_refinement_iterations = 10
                iteration_success = False

                # Iterative refinement loop
                for iteration in range(max_refinement_iterations + 1):  # 0 = initial, 1-10 = refinements
                    iteration_start_time = time.time()
                    
                    # Read current Dockerfile content
                    with open(dockerfile_path, 'r', encoding='utf-8') as f:
                        dockerfile_content = f.read()

                    if iteration == 0:
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

                            # Check for transient errors that benefit from retry with backoff
                            transient_stages = ["NETWORK", "IMAGE_PULL"]
                            if stage in transient_stages and iteration < 3:  # Only retry transient errors up to 3 times
                                # Exponential backoff for transient failures
                                backoff_delay = min(2 ** iteration, 30)  # 1s, 2s, 4s... cap at 30s
                                self.log(
                                    repo_name,
                                    f"Transient error '{stage}' detected - retrying after {backoff_delay}s backoff...",
                                    to_console=True
                                )
                                time.sleep(backoff_delay)
                                iteration_result["transient_retry"] = True
                                iteration_result["backoff_delay_seconds"] = backoff_delay
                                result["dockerfile_test"]["iterations"].append(iteration_result)
                                # Retry without refinement (same Dockerfile)
                                continue

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
                            refinement_success = self._refine_dockerfile_with_error_feedback(
                                agent, dockerfile_path, raw_error_file, repo_path, repo_name, iteration + 1, docker_result
                            )
                            refinement_duration = time.time() - refinement_start
                            
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
        repo_path: str, repo_name: str, iteration: int, docker_result: Dict = None
    ) -> bool:
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
            True if refinement appears successful (Dockerfile was modified), False otherwise
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
            
            # NEW: Handle platform incompatibility error FIRST
            if is_platform_error:
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
                
                refinement_query = f"""CRITICAL: Platform/Architecture Mismatch Error!

The base image "{problematic_image}" does NOT support {arch_name} ({arch_devices} / {expected_platform}).
It was built for a different architecture which is incompatible with this system.

**STEP 1 - FIND {arch_name} COMPATIBLE IMAGE**: Use DockerImageSearch with "tags:{base_image_name}"
   Look for tags with multi-arch support, or recent version numbers (2022+)
   
**STEP 2 - VERIFY {arch_name} SUPPORT**: Use DockerImageSearch with "{base_image_name}:<tag>"
   Check the "Architectures" field - must include "{host_docker_arch}" support

COMMON FIXES FOR {arch_name} COMPATIBILITY:
- OLD: maven:3.3.x-jdk-8 -> NEW: maven:3.9-eclipse-temurin-17 (multi-arch)
- OLD: openjdk:8-jdk -> NEW: eclipse-temurin:17-jdk (multi-arch)
- OLD: node:14 -> NEW: node:20-slim (multi-arch)
- OLD: python:3.8-slim -> NEW: python:3.11-slim (multi-arch)

**STEP 3 - UPDATE DOCKERFILE**: Read {dockerfile_absolute}
   Replace the base image with a {arch_name}-compatible version

CRITICAL RULES:
- The image MUST support {expected_platform} architecture
- Use recent versions (2022+) which typically have multi-arch support
- Verify with DockerImageSearch before using
- Final Answer: ONLY Dockerfile content starting with FROM

Error: {safe_error}"""

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
                base_image_name = problematic_image.split(':')[0] if problematic_image != "unknown" else "unknown"
                
                # NEW APPROACH: Web search first to understand what tags exist, then verify
                refinement_query = f"""CRITICAL DOCKERFILE ERROR: Base image "{problematic_image}" DOES NOT EXIST on Docker Hub!

**STEP 1 - WEB SEARCH FIRST**: Use SearchDockerError with "docker hub {base_image_name} available tags versions"
   This will show you what tags are commonly used and recommended.

**STEP 2 - LIST AVAILABLE TAGS**: Use DockerImageSearch with "tags:{base_image_name}"
   This queries Docker Hub API and shows ALL available tags categorized by:
   - Versioned tags (recommended for stability)
   - JDK/OpenJDK tags (for Java projects)
   - Slim/Alpine tags (smaller images)

**STEP 3 - VERIFY YOUR CHOICE**: Use DockerImageSearch with "{base_image_name}:<your-chosen-tag>"
   Confirm the tag exists before using it.

**STEP 4 - UPDATE DOCKERFILE**: Read the current Dockerfile at: {dockerfile_absolute}
   Replace the failing image with your VERIFIED tag.

CRITICAL RULES:
- You MUST use DockerImageSearch with "tags:{base_image_name}" to see available tags
- Pick a SPECIFIC VERSION tag (e.g., "3.9.6" not "latest")
- VERIFY the tag exists before using it
- Your Final Answer MUST start with FROM and contain ONLY Dockerfile content
- NO explanations or prose - ONLY the Dockerfile

Error: {safe_error}"""

            elif is_dependency_error:
                self.log(repo_name, f"DEPENDENCY error detected - instructing agent to check dependencies", to_console=True)
                
                # Sanitize error for Azure
                safe_error = self._sanitize_error_for_azure(error_log, max_length=400)
                
                # Extract SPECIFIC error context for better search - not just generic keywords
                error_keywords = "dependency install failed docker"
                specific_error = ""
                
                # Extract the actual failing package or error message
                for line in error_log.split('\n'):
                    line_lower = line.lower()
                    # npm specific errors
                    if "npm err!" in line_lower or "npm error" in line_lower:
                        # Extract package name if present
                        if "enoent" in line_lower or "no such file" in line_lower:
                            specific_error = "npm ENOENT file not found"
                            error_keywords = f"docker npm ENOENT missing file {specific_error}"
                        elif "gyp" in line_lower or "node-gyp" in line_lower:
                            specific_error = "node-gyp build failed"
                            error_keywords = "docker node-gyp build failed python make gcc"
                        elif "permission" in line_lower:
                            specific_error = "npm permission denied"
                            error_keywords = "docker npm permission denied unsafe-perm"
                        else:
                            error_keywords = f"docker npm install error {line[:50]}"
                        break
                    # pip specific errors
                    elif "pip" in line_lower and ("error" in line_lower or "failed" in line_lower):
                        if "gcc" in line_lower or "compilation" in line_lower:
                            specific_error = "pip compilation failed missing gcc"
                            error_keywords = "docker pip install gcc compilation failed build-essential"
                        elif "wheel" in line_lower:
                            specific_error = "pip wheel build failed"
                            error_keywords = "docker pip wheel build failed"
                        else:
                            error_keywords = f"docker pip install failed {line[:40]}"
                        break
                    # apt/apk specific errors
                    elif "unable to locate package" in line_lower:
                        # Extract package name
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p.lower() == "package" and i + 1 < len(parts):
                                pkg = parts[i + 1]
                                error_keywords = f"docker apt unable to locate package {pkg}"
                                specific_error = f"package {pkg} not found"
                                break
                        break
                    # pnpm specific
                    elif "pnpm" in line_lower and ("err" in line_lower or "error" in line_lower):
                        if "frozen lockfile" in line_lower:
                            error_keywords = "docker pnpm frozen-lockfile mismatch"
                        else:
                            error_keywords = "docker pnpm install failed"
                        break
                
                # If no specific error found, use generic but descriptive search
                if not specific_error:
                    if "npm" in error_log_lower:
                        error_keywords = "docker npm install failed missing dependencies"
                    elif "pip" in error_log_lower:
                        error_keywords = "docker pip install failed build dependencies"
                    elif "pnpm" in error_log_lower:
                        error_keywords = "docker pnpm install failed"
                    elif "yarn" in error_log_lower:
                        error_keywords = "docker yarn install failed"
                    elif "cargo" in error_log_lower:
                        error_keywords = "docker cargo build failed rust dependencies"
                    elif "maven" in error_log_lower or "mvn" in error_log_lower:
                        error_keywords = "docker maven build failed"
                    elif "apt" in error_log_lower or "apt-get" in error_log_lower:
                        error_keywords = "docker apt-get install failed package not found"
                
                refinement_query = f"""DOCKERFILE DEPENDENCY ERROR - Package installation failed!

IMPORTANT: This is NOT an image problem - DO NOT change the base image!

**STEP 1:** Use SearchDockerError with "{error_keywords}" to find solutions
**STEP 2:** Read the current Dockerfile at: {dockerfile_absolute}
**STEP 3:** Read the project dependency files (package.json, requirements.txt, pom.xml, etc.)
**STEP 4:** Fix the RUN commands to install missing system packages

CRITICAL RULES:
- DO NOT CHANGE THE BASE IMAGE (FROM line) - it is CORRECT!
- ONLY fix RUN commands to add missing dependencies
- Common fixes: add build-essential, python3-dev, gcc, make, etc.
- Your Final Answer MUST start with the SAME FROM line as the current Dockerfile

WRONG: Changing FROM maven:... to FROM node:... (DO NOT DO THIS!)
CORRECT: Keep FROM maven:... and add RUN apt-get install build-essential

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
                
                # Get stage and failed command from docker_result if available
                if docker_result:
                    stage = docker_result.get('stage', 'BUILD')
                    failed_cmd = docker_result.get('failed_command', 'unknown')
                else:
                    stage = 'BUILD'
                    failed_cmd = 'unknown'
                
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
            
            # Invoke agent with max_iterations=8 but simpler prompts to force tool usage
            # Shorter prompts = agent focuses more on tool calls vs text generation
            from run_agent import _invoke_agent_with_iteration_limit
            
            try:
                result = _invoke_agent_with_iteration_limit(
                    agent,
                    {
                        "input": refinement_query,
                        "chat_history": f"Repository: {repo_name}. Dockerfile iteration {iteration}. Use tools to investigate."
                    },
                    max_iterations=10  # Enough iterations to use multiple tools and analyze
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
                        return False
                else:
                    raise  # Re-raise if not iteration limit error
            
            # Log tool usage during refinement for debugging
            if 'intermediate_steps' in result:
                tools_used = [action.tool for action, _ in result['intermediate_steps']]
                self.log(repo_name, f"Refinement tools used: {', '.join(tools_used) if tools_used else 'none'}", to_console=True)
                
                # Special check: Did agent use DockerImageSearch for image pull errors?
                if is_image_pull_error and 'DockerImageSearch' not in tools_used:
                    self.log(repo_name, "WARNING: IMAGE_PULL error but agent didn't use DockerImageSearch!", to_console=True)
            
            refined_output = result.get('output', '').strip()
            
            if not refined_output:
                self.log(repo_name, "Agent returned empty output for refinement", to_console=False)
                return False
            
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
                return False
            
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
                return False
            
            # Check if Dockerfile actually changed
            if refined_dockerfile.strip() == current_dockerfile.strip():
                self.log(repo_name, "Refined Dockerfile is identical to current one", to_console=True)
                return False
            
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
            
            return True
            
        except Exception as e:
            self.log(repo_name, f"Exception during Dockerfile refinement: {e}", to_console=False)
            import traceback
            self.log(repo_name, f"Traceback: {traceback.format_exc()}", to_console=False)
            return False

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