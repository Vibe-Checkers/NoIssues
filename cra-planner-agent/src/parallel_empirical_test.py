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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
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
        self.results_file = self.results_dir / f"results_{self.timestamp}.json"
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
            worker_id: Worker thread ID

        Returns:
            Dictionary with complete test results
        """
        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        test_start_time = time.time()

        result = {
            "repo_url": repo_url,
            "repo_name": repo_name,
            "worker_id": worker_id,
            "timestamp": datetime.now().isoformat(),
            "agent_analysis": {},
            "dockerfile_test": {},
            "total_duration_seconds": 0,
            "success": False
        }

        repo_path = None

        try:
            self.log(repo_name, f"[Worker {worker_id}] Starting test", to_console=True)

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

            # Create report directory
            report_dir = self.reports_dir / f"{repo_name}_{self.timestamp}"
            report_dir.mkdir(exist_ok=True)

            # Set global report directory for web search caching
            import planner_agent
            planner_agent.REPORT_DIRECTORY = str(report_dir)

            # Run analysis with repo-specific log
            log_file = self.logs_dir / f"{repo_name}_agent_log.txt"
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

                    # Test Docker build
                    image_name = f"parallel-empirical-{repo_name.lower()}:latest"
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

                        # Save error log for this iteration
                        error_iteration_suffix = f"_iter{iteration}" if iteration > 0 else ""
                        raw_error_file = self.docker_errors_dir / f"{repo_name}_docker_error{error_iteration_suffix}.log"
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
                            # Check for non-recoverable errors
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
                                
                                f.write(f"See full error: docker_errors/{repo_name}_docker_error{error_iteration_suffix}.log\n")

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

                # Cleanup Docker image
                try:
                    image_name = f"parallel-empirical-{repo_name.lower()}:latest"
                    self.docker_tester.cleanup_image(image_name)
                except:
                    pass

            # Force garbage collection after each test
            gc.collect()

            # Save result
            with self.results_lock:
                self.results.append(result)
                self._save_incremental_results()

        return result

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
            is_image_pull_error = any(x in error_log_lower for x in [
                "failed to resolve", "manifest unknown", "pull access denied", 
                "image not found", "not found: manifest unknown", "no such image"
            ])
            
            is_dependency_error = any(x in error_log_lower for x in [
                "npm err!", "pip install failed", "error: failed to compile",
                "cargo build failed", "maven build failed", "no such file or directory"
            ])
            
            is_build_error = any(x in error_log_lower for x in [
                "returned a non-zero code", "exited with code", "build failed"
            ]) and not is_image_pull_error
            
            # Detect syntax errors specifically
            is_syntax_error = any(x in error_log_lower for x in [
                "syntax error", "unexpected token", "parse error", "invalid syntax",
                "unknown instruction", "dockerfile parse error", "failed to parse"
            ])
            
            # Build targeted refinement query based on error type
            if is_image_pull_error:
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

🔍 **STEP 1 - WEB SEARCH FIRST**: Use SearchDockerError with "docker hub {base_image_name} available tags versions"
   This will show you what tags are commonly used and recommended.

📦 **STEP 2 - LIST AVAILABLE TAGS**: Use DockerImageSearch with "tags:{base_image_name}"
   This queries Docker Hub API and shows ALL available tags categorized by:
   - Versioned tags (recommended for stability)
   - JDK/OpenJDK tags (for Java projects)
   - Slim/Alpine tags (smaller images)

✅ **STEP 3 - VERIFY YOUR CHOICE**: Use DockerImageSearch with "{base_image_name}:<your-chosen-tag>"
   Confirm the tag exists before using it.

📝 **STEP 4 - UPDATE DOCKERFILE**: Read the current Dockerfile at: {dockerfile_absolute}
   Replace the failing image with your VERIFIED tag.

⚠️ CRITICAL RULES:
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
                
                # Extract specific error keywords for search
                error_keywords = "dependency install failed docker"
                if "npm" in error_log_lower:
                    error_keywords = "npm install failed docker"
                elif "pip" in error_log_lower:
                    error_keywords = "pip install failed docker"
                elif "cargo" in error_log_lower:
                    error_keywords = "cargo build failed docker"
                elif "maven" in error_log_lower or "mvn" in error_log_lower:
                    error_keywords = "maven build failed docker"
                elif "apt" in error_log_lower or "apt-get" in error_log_lower:
                    error_keywords = "apt-get install failed docker"
                
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
                
                refinement_query = f"""DOCKERFILE SYNTAX ERROR - Parse error in Dockerfile!

IMPORTANT: This is a SYNTAX problem - DO NOT change the base image!

**STEP 1:** Use SearchWeb with "dockerfile syntax error {syntax_error_detail}" to find solutions
**STEP 2:** Read the current Dockerfile at: {dockerfile_absolute}  
**STEP 3:** Identify the exact syntax problem (line number shown in error)
**STEP 4:** Fix ONLY the syntax error - common issues:
   - "unknown instruction" = typo in instruction name or text before FROM
   - "unexpected token" = missing quotes or escape characters
   - Shell heredoc issues = use separate RUN commands instead

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
            image_name = f"parallel-empirical-{repo_name.lower()}:latest"
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

    def _save_incremental_results(self):
        """Save results incrementally (already locked)."""
        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": self.timestamp,
                    "max_workers": self.max_workers,
                    "total_tested": len(self.results),
                    "results": self.results
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] Could not save incremental results: {e}")

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
            "results_file": str(self.results_file),
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
            f.write(f"Successful: {successful} ({successful/total*100:.1f}%)\n")
            f.write(f"Failed: {failed} ({failed/total*100:.1f}%)\n\n")

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
                    f.write(f"{stage:40s}: {count:3d} ({count/failed*100:.1f}%)\n")
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
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
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

        print(f"\nResults saved to: {self.results_dir.absolute()}")
        print(f"Summary: {self.summary_file}")
        print(f"Master log: {self.master_log_file}")
        print(f"Detailed results: {self.results_file}")

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