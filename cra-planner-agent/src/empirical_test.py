#!/usr/bin/env python3
"""
Empirical Testing Script for Planner Agent
Automates testing of Dockerfile generation and validation across multiple repositories.
"""

print("[STARTUP] Empirical test script starting...")
print("[STARTUP] Loading imports (this may take 30-60 seconds)...")

import os
import sys
import json
import subprocess
import shutil
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

print("[STARTUP] Loading agent modules...")
from run_agent import clone_repository, analyze_repository, detect_project_language
from planner_agent import create_planner_agent

print("[STARTUP] All imports loaded successfully!")


class DockerBuildTester:
    """Tests generated Dockerfiles by actually building them with Docker."""

    def __init__(self, timeout: int = 600, platform: str = None):
        """
        Initialize Docker build tester.

        Args:
            timeout: Maximum time in seconds to wait for Docker build (default: 10 minutes)
            platform: Optional platform to build for (e.g., "linux/amd64", "linux/arm64")
                     Use None for native platform. Useful for cross-platform testing.
        """
        self.timeout = timeout
        self.platform = platform
        self.docker_available = self._check_docker()

    def _check_docker(self) -> bool:
        """Check if Docker is installed and accessible."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def build_dockerfile(self, dockerfile_path: str, context_path: str, image_name: str) -> Dict:
        """
        Build a Dockerfile and return detailed results.

        Args:
            dockerfile_path: Path to Dockerfile
            context_path: Path to build context (usually repository root)
            image_name: Name to tag the built image

        Returns:
            Dictionary with build results including success status, stage, error details
        """
        if not self.docker_available:
            return {
                "success": False,
                "stage": "DOCKER_CHECK",
                "error_type": "DOCKER_NOT_AVAILABLE",
                "error_message": "Docker is not installed or not accessible",
                "exit_code": -1,
                "duration_seconds": 0
            }

        if not os.path.exists(dockerfile_path):
            return {
                "success": False,
                "stage": "DOCKERFILE_CHECK",
                "error_type": "DOCKERFILE_NOT_FOUND",
                "error_message": f"Dockerfile not found at {dockerfile_path}",
                "exit_code": -1,
                "duration_seconds": 0
            }

        start_time = time.time()

        # Build Docker image
        try:
            cmd = ["docker", "build"]

            # Add platform flag if specified (for cross-platform builds)
            if self.platform:
                cmd.extend(["--platform", self.platform])

            cmd.extend([
                "-f", dockerfile_path,
                "-t", image_name,
                context_path
            ])

            platform_info = f" (platform: {self.platform})" if self.platform else ""
            print(f"[DOCKER BUILD] Running: {' '.join(cmd)}{platform_info}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                return {
                    "success": True,
                    "stage": "BUILD_COMPLETE",
                    "error_type": None,
                    "error_message": None,
                    "exit_code": 0,
                    "duration_seconds": duration,
                    "stdout": result.stdout[-2000:],  # Last 2000 chars
                    "stderr": result.stderr[-2000:] if result.stderr else None
                }
            else:
                # Parse error to determine stage
                full_error = result.stderr or result.stdout
                stage, failed_docker_command = self._parse_docker_error(full_error)

                # Extract a concise error snippet (last error line)
                error_lines = full_error.strip().split('\n')
                error_snippet = None
                for line in reversed(error_lines):
                    if line.strip() and ('error' in line.lower() or 'failed' in line.lower() or 'fatal' in line.lower()):
                        error_snippet = line.strip()
                        break
                if not error_snippet and error_lines:
                    error_snippet = error_lines[-1].strip()

                return {
                    "success": False,
                    "stage": stage,
                    "failed_command": failed_docker_command,  # The Docker step that failed
                    "error_message": full_error,  # Full error for detailed analysis
                    "error_snippet": error_snippet,  # Concise error line for quick review
                    "exit_code": result.returncode,
                    "duration_seconds": duration,
                    "stdout": result.stdout[-2000:],
                    "stderr": result.stderr[-2000:] if result.stderr else None
                }

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "success": False,
                "stage": "BUILD_TIMEOUT",
                "error_type": "TIMEOUT",
                "error_message": f"Docker build exceeded timeout of {self.timeout} seconds",
                "exit_code": -1,
                "duration_seconds": duration
            }

        except Exception as e:
            duration = time.time() - start_time
            return {
                "success": False,
                "stage": "BUILD_EXCEPTION",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "exit_code": -1,
                "duration_seconds": duration,
                "traceback": traceback.format_exc()
            }

    def _parse_docker_error(self, error_output: str) -> Tuple[str, str]:
        """
        Parse Docker error output to determine which Dockerfile step failed.

        Strategy:
        1. Extract the failed Docker step/command from build output
        2. Categorize into simple high-level stages
        3. Save full error output for manual inspection

        Returns:
            Tuple of (stage, failed_command_or_description)
        """
        error_lower = error_output.lower()

        # Extract the failed Docker step from the build output
        failed_step = self._extract_failed_docker_step(error_output)

        # ===================================================================
        # Simple Stage Detection - High Level Only
        # ===================================================================

        # 1. Docker Daemon Issues
        if any(x in error_lower for x in ["cannot connect to the docker daemon", "is the docker daemon running", "docker: not found", "docker: command not found"]):
            return "DOCKER_DAEMON", failed_step or "Docker daemon not accessible"

        # 2. Docker BuildKit Storage Corruption
        # This occurs when parallel builds + aggressive cleanup corrupt the overlay2 storage driver
        # Symptoms: "parent snapshot sha256:... does not exist: not found"
        # Solution: Requires system-wide docker builder prune --all --force
        if "parent snapshot" in error_lower and "does not exist" in error_lower:
            return "INFRASTRUCTURE_CORRUPTION", failed_step or "Docker BuildKit cache corrupted - requires prune"

        # 3. Platform/Architecture Incompatibility (ARM vs x86)
        # This is a special case of image pull - the image exists but for wrong architecture
        if any(x in error_lower for x in [
            "invalidbaseimageplatform", "platform", "linux/amd64", "linux/arm64",
            "was pulled with platform", "expected \"linux/", "does not match"
        ]) and any(x in error_lower for x in ["amd64", "arm64", "architecture", "platform"]):
            return "PLATFORM_INCOMPATIBLE", failed_step or "Base image platform mismatch (amd64 vs arm64)"

        # 4. Base Image Pull Issues (FROM command)
        # 4a. Image Validation Failures (corrupted/deprecated manifest)
        if any(x in error_lower for x in ["failed validation", "failed to load cache key"]):
            return "IMAGE_VALIDATION_FAILED", failed_step or "Docker image failed validation (likely deprecated/corrupted manifest)"

        # 4b. General image pull failures
        if any(x in error_lower for x in ["failed to resolve", "manifest unknown", "pull access denied", "image not found"]):
            return "IMAGE_PULL", failed_step or "Failed to pull base image"

        # 5. Dockerfile Syntax
        if any(x in error_lower for x in ["dockerfile parse error", "unknown instruction"]):
            return "DOCKERFILE_SYNTAX", failed_step or "Dockerfile syntax error"

        # 6. File Copy/Add (COPY/ADD commands)
        # 6a. File missing from build context
        if "failed to compute cache key" in error_lower and "not found" in error_lower:
            return "FILE_COPY_MISSING", failed_step or "COPY command references file that doesn't exist in build context"

        # 6b. General copy failures
        if any(x in error_lower for x in ["copy failed", "add failed"]) or ("stat" in error_lower and "no such file" in error_lower):
            return "FILE_COPY", failed_step or "File copy/add failed"

        # 7. Dependency Installation (RUN pip/npm/go/cargo install commands)
        # 7a. Missing dependency files (package.json, requirements.txt not at expected location)
        if any(x in error_lower for x in ["enoent", "no such file or directory"]) and \
           any(x in error_lower for x in ["package.json", "requirements.txt", "pom.xml", "cargo.toml", "go.mod"]):
            return "DEPENDENCY_FILE_MISSING", failed_step or "Dependency file not found at expected location"

        # 7b. Missing build tools for native dependencies
        if any(x in error_lower for x in ["node-gyp", "gyp err", "gcc: command not found",
                                            "make: command not found", "build-essential",
                                            "python: command not found", "python3: command not found"]):
            return "DEPENDENCY_BUILD_TOOLS", failed_step or "Missing build tools (gcc, make, python) for native dependencies"

        # 7c. Network issues during dependency install
        if any(x in error_lower for x in ["npm install", "pip install", "pnpm install", "yarn install"]) and \
           any(x in error_lower for x in ["network", "fetch_404", "registry unreachable", "econnrefused", "etimedout"]):
            return "DEPENDENCY_NETWORK", failed_step or "Network error during dependency installation"

        # 7d. Lockfile mismatch issues
        if any(x in error_lower for x in ["frozen-lockfile", "frozen lockfile", "lock file"]) and \
           any(x in error_lower for x in ["outdated", "not in sync", "mismatch", "not up to date"]):
            return "DEPENDENCY_LOCKFILE_MISMATCH", failed_step or "Lockfile out of sync with package.json"

        # 7e. General dependency installation failures
        if any(x in error_lower for x in ["pip install", "pip3 install", "npm install", "yarn install", "go mod download", "go get", "cargo build"]):
            return "DEPENDENCY_INSTALL", failed_step or "Dependency installation failed"

        # 7a. Dependency Rot (missing/removed versions - terminal condition)
        # Symptoms: npm notarget, pip no matching distribution, outdated lockfiles
        if any(x in error_lower for x in [
            "notarget", "no matching version", "no matching distribution",
            "could not find a version that satisfies", "no such file or directory: 'package-lock.json'",
            "error: no matching package named", "version solving failed"
        ]):
            return "DEPENDENCY_ROT", failed_step or "Dependency version no longer exists (requires code update)"

        # 8. Build/Compilation (RUN build commands)
        if any(x in error_lower for x in ["compilation error", "build error", "webpack", "tsc"]):
            return "BUILD_COMPILE", failed_step or "Build/compilation failed"

        # 9. Runtime Execution (CMD/ENTRYPOINT)
        # 9a. Build tool/script not found (exit code 127 = command not found)
        if ("exit code: 127" in error_lower or "command not found" in error_lower) and \
           any(x in error_lower for x in ["gradlew", "./configure", "autogen.sh", "bootstrap", "cmake", "./build"]):
            return "BUILD_TOOL_MISSING", failed_step or "Build tool or script not found/executable"

        # 9b. General runtime execution failures
        if any(x in error_lower for x in ["command not found", "exec format error"]):
            return "RUNTIME_EXEC", failed_step or "Runtime execution failed"

        # 10. Permission/User Issues
        if "permission denied" in error_lower or "useradd" in error_lower:
            return "PERMISSION", failed_step or "Permission/user management error"

        # 11. Network Issues
        if any(x in error_lower for x in ["connection refused", "connection timeout", "network unreachable"]):
            return "NETWORK", failed_step or "Network connection error"

        # 12. Storage Issues
        if any(x in error_lower for x in ["no space left", "disk full", "quota exceeded"]):
            return "STORAGE", failed_step or "Disk space error"

        # Fallback: Return the extracted step or unknown
        return "UNKNOWN", failed_step or "Unknown error - check full log"

    def _extract_failed_docker_step(self, error_output: str) -> str:
        """
        Extract the specific Docker RUN/COPY/FROM command that failed.

        Docker output format examples:
        - "ERROR [stage 3/8] RUN pip install -r requirements.txt"
        - "executor failed running [/bin/sh -c go build]: exit code: 1"
        - "#8 [stage 4/7] RUN npm install"
        """
        import re

        # Pattern 1: ERROR [stage X/Y] COMMAND
        match = re.search(r'ERROR \[.*?\] (RUN|COPY|ADD|FROM|WORKDIR).*', error_output, re.IGNORECASE)
        if match:
            return match.group(0).replace('ERROR ', '').strip()

        # Pattern 2: executor failed running [/bin/sh -c COMMAND]
        match = re.search(r'executor failed running \[/bin/sh -c ([^\]]+)\]', error_output, re.IGNORECASE)
        if match:
            return f"RUN {match.group(1).strip()}"

        # Pattern 3: #N [stage X/Y] COMMAND
        match = re.search(r'#\d+ \[.*?\] (RUN|COPY|ADD|FROM).*', error_output, re.IGNORECASE)
        if match:
            return match.group(0).strip()

        # Return None if we can't extract the step
        return None

    def cleanup_image(self, image_name: str) -> bool:
        """Remove Docker image after testing."""
        try:
            subprocess.run(
                ["docker", "rmi", "-f", image_name],
                capture_output=True,
                timeout=30
            )
            return True
        except Exception:
            return False

    def prune_build_cache(self) -> bool:
        """
        Clear Docker BuildKit cache to fix storage corruption.

        This is required when parallel builds + aggressive cleanup cause overlay2
        storage driver corruption, resulting in "parent snapshot does not exist" errors.

        Returns:
            True if prune succeeded, False otherwise
        """
        try:
            result = subprocess.run(
                ["docker", "builder", "prune", "--all", "--force"],
                capture_output=True,
                text=True,
                timeout=120
            )
            return result.returncode == 0
        except Exception:
            return False

    def check_infrastructure_health(self) -> Tuple[bool, str]:
        """
        Check Docker infrastructure health before starting parallel tests.

        Verifies:
        1. Docker daemon is running
        2. BuildKit is functional
        3. No existing cache corruption

        Returns:
            Tuple of (healthy: bool, message: str)
        """
        try:
            # Check Docker daemon
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return False, "Docker daemon is not running or not accessible"

            # Check BuildKit by attempting a trivial build
            test_dockerfile = "FROM scratch\n"
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.Dockerfile', delete=False) as f:
                f.write(test_dockerfile)
                test_dockerfile_path = f.name

            try:
                result = subprocess.run(
                    ["docker", "build", "-f", test_dockerfile_path, "-t", "infra-health-check:test", "."],
                    capture_output=True,
                    text=True,
                    timeout=30
                )

                # Check for corruption in test build output
                if "parent snapshot" in result.stderr.lower() and "does not exist" in result.stderr.lower():
                    return False, "Docker BuildKit cache is corrupted - prune required"

                # Clean up test image
                subprocess.run(
                    ["docker", "rmi", "-f", "infra-health-check:test"],
                    capture_output=True,
                    timeout=10
                )

                return True, "Infrastructure health check passed"

            finally:
                import os
                try:
                    os.unlink(test_dockerfile_path)
                except:
                    pass

        except subprocess.TimeoutExpired:
            return False, "Docker health check timed out"
        except Exception as e:
            return False, f"Health check failed: {str(e)}"


class EmpiricalTester:
    """Main empirical testing orchestrator."""

    def __init__(self, results_dir: str = "./empirical_results"):
        """
        Initialize empirical tester.

        Args:
            results_dir: Directory to store all results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.repos_dir = self.results_dir / "repositories"
        self.reports_dir = self.results_dir / "analysis_reports"
        self.logs_dir = self.results_dir / "logs"

        self.repos_dir.mkdir(exist_ok=True)
        self.reports_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)

        self.docker_tester = DockerBuildTester(timeout=600)

        # Initialize results storage
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.results_dir / f"empirical_results_{self.timestamp}.json"
        self.summary_file = self.results_dir / f"empirical_summary_{self.timestamp}.txt"

        self.results = []

    def test_repository(self, repo_url: str, repo_index: int, total_repos: int) -> Dict:
        """
        Test a single repository: analyze with agent, generate Dockerfile, test build.

        Args:
            repo_url: GitHub repository URL
            repo_index: Current index (for progress tracking)
            total_repos: Total number of repositories

        Returns:
            Dictionary with complete test results
        """
        print(f"\n{'='*80}")
        print(f"TESTING REPOSITORY [{repo_index}/{total_repos}]")
        print(f"URL: {repo_url}")
        print(f"{'='*80}\n")

        repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
        test_start_time = time.time()
        repo_path = None  # Track repo path for cleanup

        result = {
            "repo_url": repo_url,
            "repo_name": repo_name,
            "index": repo_index,
            "timestamp": datetime.now().isoformat(),
            "agent_analysis": {},
            "dockerfile_test": {},
            "total_duration_seconds": 0,
            "success": False
        }

        try:
            # Step 1: Clone repository
            print(f"\n[STEP 1/{4}] Cloning repository...")
            clone_start = time.time()
            repo_path = clone_repository(repo_url, target_dir=str(self.repos_dir), auto_remove=True)
            clone_duration = time.time() - clone_start

            result["clone"] = {
                "success": True,
                "duration_seconds": clone_duration,
                "path": repo_path
            }

            # Detect language
            detected_language = detect_project_language(repo_path)
            result["detected_language"] = detected_language
            print(f"[INFO] Detected language: {detected_language}")

            # Step 2: Create planner agent
            print(f"\n[STEP 2/{4}] Initializing planner agent...")
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

            # Step 3: Run analysis
            print(f"\n[STEP 3/{4}] Running agent analysis...")
            analysis_start = time.time()

            # Create report directory for this repo
            report_dir = self.reports_dir / f"{repo_name}_{self.timestamp}"
            report_dir.mkdir(exist_ok=True)

            # Set thread-local report directory for web search caching (thread-safe)
            import planner_agent
            planner_agent._set_report_directory(str(report_dir))
            # NOTE: No longer setting global REPORT_DIRECTORY - thread-local only for thread safety

            # Run analysis
            log_file = self.logs_dir / f"{repo_name}_agent_log.txt"
            report_dir_result = analyze_repository(
                agent, repo_path, repo_name, repo_url,
                callback_handler, log_file_path=log_file, report_dir=report_dir
            )

            analysis_duration = time.time() - analysis_start

            # Extract analysis results
            result["agent_analysis"] = {
                "success": True,
                "duration_seconds": analysis_duration,
                "report_directory": str(report_dir_result),
                "log_file": str(log_file)
            }

            # Get token usage from callback handler
            if callback_handler and callback_handler.token_usage["total"] > 0:
                result["agent_analysis"]["token_usage"] = callback_handler.token_usage

            # Read the agent log to extract thought process
            if log_file.exists():
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                result["agent_analysis"]["log_size_chars"] = len(log_content)
                # Save abbreviated log (first and last 5000 chars)
                if len(log_content) > 10000:
                    result["agent_analysis"]["log_excerpt"] = {
                        "beginning": log_content[:5000],
                        "ending": log_content[-5000:]
                    }
                else:
                    result["agent_analysis"]["log_excerpt"] = log_content

            # Step 4: Test Dockerfile
            print(f"\n[STEP 4/{4}] Testing generated Dockerfile...")
            dockerfile_path = report_dir_result / "Dockerfile"

            if not dockerfile_path.exists():
                result["dockerfile_test"] = {
                    "success": False,
                    "stage": "DOCKERFILE_GENERATION",
                    "error_type": "DOCKERFILE_NOT_GENERATED",
                    "error_message": "Agent did not generate a Dockerfile"
                }
            else:
                # Read Dockerfile
                with open(dockerfile_path, 'r', encoding='utf-8') as f:
                    dockerfile_content = f.read()
                result["dockerfile_content"] = dockerfile_content

                # Test build
                image_name = f"empirical-test-{repo_name.lower()}:latest"
                docker_result = self.docker_tester.build_dockerfile(
                    str(dockerfile_path),
                    repo_path,
                    image_name
                )

                result["dockerfile_test"] = docker_result

                # Save Docker error to files if build failed
                if not docker_result["success"]:
                    # Create docker_errors directory for raw error logs
                    docker_errors_dir = self.results_dir / "docker_errors"
                    docker_errors_dir.mkdir(exist_ok=True)

                    # 1. Save summary error file in the report directory
                    error_summary_file = report_dir_result / "docker_build_error_summary.txt"
                    try:
                        with open(error_summary_file, 'w', encoding='utf-8') as f:
                            f.write("="*80 + "\n")
                            f.write("DOCKER BUILD ERROR SUMMARY\n")
                            f.write("="*80 + "\n\n")
                            f.write(f"Repository: {repo_name}\n")
                            f.write(f"URL: {repo_url}\n")
                            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                            f.write(f"Failure Stage: {docker_result.get('stage', 'UNKNOWN')}\n")
                            f.write(f"Failed Docker Command: {docker_result.get('failed_command', 'Unknown')}\n")
                            f.write(f"Exit Code: {docker_result.get('exit_code', -1)}\n\n")

                            if docker_result.get('error_snippet'):
                                f.write(f"Error Snippet (last error line):\n{'-'*80}\n")
                                f.write(f"{docker_result['error_snippet']}\n")
                                f.write(f"{'-'*80}\n\n")

                            f.write(f"See full Docker output in: docker_errors/{repo_name}_docker_error.log\n")

                        print(f"[ERROR SUMMARY] Saved to: {error_summary_file.name}")
                    except Exception as e:
                        print(f"[WARNING] Could not save error summary file: {e}")

                    # 2. Save FULL raw Docker error output in dedicated directory
                    raw_error_file = docker_errors_dir / f"{repo_name}_docker_error.log"
                    try:
                        with open(raw_error_file, 'w', encoding='utf-8') as f:
                            f.write(f"# Docker Build Error Log\n")
                            f.write(f"# Repository: {repo_name}\n")
                            f.write(f"# URL: {repo_url}\n")
                            f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
                            f.write(f"# Stage: {docker_result.get('stage', 'UNKNOWN')}\n")
                            f.write(f"# Failed Command: {docker_result.get('failed_command', 'Unknown')}\n")
                            f.write(f"# Exit Code: {docker_result.get('exit_code', -1)}\n")
                            f.write("#" + "="*78 + "\n\n")

                            # Write the complete raw Docker output
                            f.write(docker_result.get('error_message', 'No error message available'))

                        print(f"[RAW ERROR LOG] Saved to: docker_errors/{raw_error_file.name}")
                    except Exception as e:
                        print(f"[WARNING] Could not save raw error log: {e}")

                # Cleanup image if build succeeded
                if docker_result["success"]:
                    print(f"[CLEANUP] Removing Docker image {image_name}...")
                    self.docker_tester.cleanup_image(image_name)

            # Cleanup: Remove cloned repository after all processing is complete
            # This saves disk space when testing many repositories (100+)
            # All logs, Dockerfiles, and error reports have been saved before this point
            if repo_path and os.path.exists(repo_path):
                try:
                    print(f"\n[CLEANUP] Removing cloned repository: {repo_path}")
                    shutil.rmtree(repo_path)
                    print(f"[OK] Repository removed successfully")
                except Exception as e:
                    print(f"[WARNING] Could not remove repository {repo_path}: {e}")

            # Determine overall success
            result["success"] = (
                result.get("clone", {}).get("success", False) and
                result.get("agent_analysis", {}).get("success", False) and
                result.get("dockerfile_test", {}).get("success", False)
            )

        except Exception as e:
            print(f"\n[ERROR] Exception during testing: {e}")
            traceback.print_exc()
            result["exception"] = {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            }
            result["success"] = False

        finally:
            # Calculate total duration
            result["total_duration_seconds"] = time.time() - test_start_time

            # Print summary for this repo
            self._print_repo_summary(result)

            # Save result
            self.results.append(result)
            self._save_incremental_results()

        return result

    def _print_repo_summary(self, result: Dict):
        """Print summary for a single repository test."""
        print(f"\n{'='*80}")
        print(f"REPOSITORY TEST SUMMARY: {result['repo_name']}")
        print(f"{'='*80}")
        print(f"Overall Success: {'✓ YES' if result['success'] else '✗ NO'}")
        print(f"Total Duration: {result['total_duration_seconds']:.2f} seconds")

        if "clone" in result:
            print(f"\nClone: {'✓' if result['clone'].get('success') else '✗'}")

        if "agent_analysis" in result:
            analysis = result["agent_analysis"]
            print(f"Agent Analysis: {'✓' if analysis.get('success') else '✗'}")
            if "token_usage" in analysis:
                tokens = analysis["token_usage"]
                print(f"  Tokens: {tokens.get('total', 0):,} (in: {tokens.get('input', 0):,}, out: {tokens.get('output', 0):,})")

        if "dockerfile_test" in result:
            docker = result["dockerfile_test"]
            print(f"Dockerfile Build: {'✓' if docker.get('success') else '✗'}")
            if not docker.get('success'):
                print(f"  Failure Stage: {docker.get('stage', 'UNKNOWN')}")
                print(f"  Failed Command: {docker.get('failed_command', 'Unknown')}")
                # Show error snippet if available, otherwise first 200 chars of full error
                error_snippet = docker.get('error_snippet', '')
                if error_snippet:
                    print(f"  Error Snippet: {error_snippet[:200]}")
                # Mention error file
                if "agent_analysis" in result and "report_directory" in result["agent_analysis"]:
                    print(f"  Full error log: docker_build_error.txt")

        print(f"{'='*80}\n")

    def _save_incremental_results(self):
        """Save results incrementally after each repository."""
        try:
            with open(self.results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": self.timestamp,
                    "total_tested": len(self.results),
                    "results": self.results
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARNING] Could not save incremental results: {e}")

    def generate_final_summary(self):
        """Generate final summary report."""
        print(f"\n{'='*80}")
        print("GENERATING FINAL SUMMARY")
        print(f"{'='*80}\n")

        total = len(self.results)
        successful = sum(1 for r in self.results if r["success"])
        failed = total - successful

        # Categorize failures
        failure_stages = {}
        failure_types = {}

        for result in self.results:
            if not result["success"]:
                if "dockerfile_test" in result:
                    stage = result["dockerfile_test"].get("stage", "UNKNOWN")
                    error_type = result["dockerfile_test"].get("error_type", "UNKNOWN")

                    failure_stages[stage] = failure_stages.get(stage, 0) + 1
                    failure_types[error_type] = failure_types.get(error_type, 0) + 1

        # Calculate statistics
        durations = [r["total_duration_seconds"] for r in self.results]
        avg_duration = sum(durations) / len(durations) if durations else 0

        token_totals = []
        for r in self.results:
            if "agent_analysis" in r and "token_usage" in r["agent_analysis"]:
                token_totals.append(r["agent_analysis"]["token_usage"].get("total", 0))
        avg_tokens = sum(token_totals) / len(token_totals) if token_totals else 0

        # Write summary file
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("EMPIRICAL TESTING SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write(f"Test Run: {self.timestamp}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("OVERALL RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Repositories Tested: {total}\n")
            f.write(f"Successful: {successful} ({successful/total*100:.1f}%)\n")
            f.write(f"Failed: {failed} ({failed/total*100:.1f}%)\n\n")

            f.write("PERFORMANCE METRICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Average Duration per Repository: {avg_duration:.2f} seconds\n")
            f.write(f"Average Token Usage per Repository: {avg_tokens:.0f} tokens\n")
            f.write(f"Total Duration: {sum(durations):.2f} seconds ({sum(durations)/60:.2f} minutes)\n\n")

            if failure_stages:
                f.write("FAILURE BREAKDOWN BY STAGE\n")
                f.write("-"*80 + "\n")
                for stage, count in sorted(failure_stages.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{stage:40s}: {count:3d} ({count/failed*100:.1f}%)\n")
                f.write("\n")

            if failure_types:
                f.write("FAILURE BREAKDOWN BY ERROR TYPE\n")
                f.write("-"*80 + "\n")
                for error_type, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{error_type:40s}: {count:3d} ({count/failed*100:.1f}%)\n")
                f.write("\n")

            f.write("DETAILED RESULTS\n")
            f.write("-"*80 + "\n")
            for i, result in enumerate(self.results, 1):
                f.write(f"\n{i}. {result['repo_name']}\n")
                f.write(f"   URL: {result['repo_url']}\n")
                f.write(f"   Success: {'YES' if result['success'] else 'NO'}\n")
                f.write(f"   Language: {result.get('detected_language', 'Unknown')}\n")
                f.write(f"   Duration: {result['total_duration_seconds']:.2f}s\n")

                if not result["success"] and "dockerfile_test" in result:
                    docker = result["dockerfile_test"]
                    f.write(f"   Failure Stage: {docker.get('stage', 'UNKNOWN')}\n")
                    f.write(f"   Error Type: {docker.get('error_type', 'UNKNOWN')}\n")

            f.write("\n" + "="*80 + "\n")

        print(f"[OK] Summary saved to: {self.summary_file}")
        print(f"[OK] Detailed results saved to: {self.results_file}")

        # Print summary to console
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Total Tested: {total}")
        print(f"Successful: {successful} ({successful/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"Average Duration: {avg_duration:.2f} seconds")
        print(f"Average Tokens: {avg_tokens:.0f}")
        print(f"{'='*80}\n")


def load_repository_list(file_path: str) -> List[str]:
    """
    Load repository URLs from a text file.

    Args:
        file_path: Path to file containing one repository URL per line

    Returns:
        List of repository URLs
    """
    repos = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                repos.append(line)
    return repos


def main():
    """Main function."""
    print("="*80)
    print("EMPIRICAL TESTING SCRIPT - Planner Agent")
    print("="*80)
    print()

    if len(sys.argv) < 2:
        print("Usage: python empirical_test.py <repository_list_file>")
        print()
        print("Example:")
        print("  python empirical_test.py repositories.txt")
        print()
        print("Repository list file format (one URL per line):")
        print("  https://github.com/psf/requests")
        print("  https://github.com/expressjs/express")
        print("  https://github.com/gin-gonic/gin")
        print("  # This is a comment")
        sys.exit(1)

    repo_list_file = sys.argv[1]

    if not os.path.exists(repo_list_file):
        print(f"[ERROR] Repository list file not found: {repo_list_file}")
        sys.exit(1)

    # Load repositories
    print(f"[INFO] Loading repository list from: {repo_list_file}")
    repositories = load_repository_list(repo_list_file)
    print(f"[INFO] Found {len(repositories)} repositories to test\n")

    if len(repositories) == 0:
        print("[ERROR] No repositories found in list file")
        sys.exit(1)

    # Initialize tester
    tester = EmpiricalTester()

    print(f"Results will be saved to: {tester.results_dir.absolute()}\n")

    # Test each repository
    total_start_time = time.time()

    for i, repo_url in enumerate(repositories, 1):
        try:
            tester.test_repository(repo_url, i, len(repositories))
        except KeyboardInterrupt:
            print("\n\n[WARNING] Testing interrupted by user")
            break
        except Exception as e:
            print(f"\n[ERROR] Unexpected error testing {repo_url}: {e}")
            traceback.print_exc()
            continue

    total_duration = time.time() - total_start_time

    # Generate final summary
    tester.generate_final_summary()

    print(f"\n{'='*80}")
    print(f"TESTING COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_duration/60:.2f} minutes")
    print(f"Results directory: {tester.results_dir.absolute()}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
