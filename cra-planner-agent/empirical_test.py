#!/usr/bin/env python3
"""
Empirical Testing Script for Planner Agent
Automates testing of Dockerfile generation and validation across multiple repositories.
"""

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

from run_agent import clone_repository, analyze_repository, detect_project_language
from planner_agent import create_planner_agent


class DockerBuildTester:
    """Tests generated Dockerfiles by actually building them with Docker."""

    def __init__(self, timeout: int = 600):
        """
        Initialize Docker build tester.

        Args:
            timeout: Maximum time in seconds to wait for Docker build (default: 10 minutes)
        """
        self.timeout = timeout
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
            cmd = [
                "docker", "build",
                "-f", dockerfile_path,
                "-t", image_name,
                context_path
            ]

            print(f"[DOCKER BUILD] Running: {' '.join(cmd)}")

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
                stage, error_type = self._parse_docker_error(full_error)

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
                    "error_type": error_type,
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
        Parse Docker error output to determine failure stage and type.

        Comprehensive error categorization for research purposes.

        Returns:
            Tuple of (stage, error_type)
        """
        error_lower = error_output.lower()

        # ===================================================================
        # STAGE 1: Base Image Issues
        # ===================================================================
        if "failed to resolve" in error_lower or "manifest unknown" in error_lower:
            return "IMAGE_PULL", "BASE_IMAGE_NOT_FOUND"
        if "pull access denied" in error_lower or "unauthorized" in error_lower:
            return "IMAGE_PULL", "BASE_IMAGE_UNAUTHORIZED"
        if "image" in error_lower and any(x in error_lower for x in ["not found", "does not exist", "no such"]):
            return "IMAGE_PULL", "BASE_IMAGE_NOT_FOUND"
        if "registry" in error_lower and any(x in error_lower for x in ["connection", "timeout", "unreachable"]):
            return "IMAGE_PULL", "REGISTRY_CONNECTION_ERROR"

        # ===================================================================
        # STAGE 2: Dockerfile Syntax Errors
        # ===================================================================
        if "dockerfile parse error" in error_lower or "unknown instruction" in error_lower:
            return "DOCKERFILE_SYNTAX", "SYNTAX_ERROR"
        if "unexpected" in error_lower and "dockerfile" in error_lower:
            return "DOCKERFILE_SYNTAX", "UNEXPECTED_TOKEN"

        # ===================================================================
        # STAGE 3: File Copy/ADD Errors
        # ===================================================================
        if "copy failed" in error_lower or ("copy" in error_lower and "no such file" in error_lower):
            return "FILE_COPY", "SOURCE_FILE_NOT_FOUND"
        if "add failed" in error_lower:
            return "FILE_COPY", "ADD_FAILED"
        if "stat" in error_lower and "no such file" in error_lower:
            return "FILE_COPY", "FILE_STAT_ERROR"

        # ===================================================================
        # STAGE 4: Dependency Management - Python (pip)
        # ===================================================================
        if "pip install" in error_lower or "pip3 install" in error_lower:
            if "could not find a version" in error_lower or "no matching distribution" in error_lower:
                return "DEPENDENCY_INSTALL_PYTHON", "PIP_PACKAGE_NOT_FOUND"
            if "requirement already satisfied" in error_lower and "error" in error_lower:
                return "DEPENDENCY_INSTALL_PYTHON", "PIP_DEPENDENCY_CONFLICT"
            if "requires python" in error_lower:
                return "DEPENDENCY_INSTALL_PYTHON", "PIP_PYTHON_VERSION_MISMATCH"
            if "error: externally-managed-environment" in error_lower:
                return "DEPENDENCY_INSTALL_PYTHON", "PIP_EXTERNALLY_MANAGED_ENV"
            if "permission denied" in error_lower:
                return "DEPENDENCY_INSTALL_PYTHON", "PIP_PERMISSION_DENIED"
            return "DEPENDENCY_INSTALL_PYTHON", "PIP_INSTALL_FAILED"

        # ===================================================================
        # STAGE 5: Dependency Management - Node.js (npm/yarn)
        # ===================================================================
        if "npm install" in error_lower or "npm i " in error_lower:
            if "404" in error_lower or "not found" in error_lower:
                return "DEPENDENCY_INSTALL_NODE", "NPM_PACKAGE_NOT_FOUND"
            if "enoent" in error_lower:
                return "DEPENDENCY_INSTALL_NODE", "NPM_FILE_NOT_FOUND"
            if "peer dep" in error_lower or "peerinvalid" in error_lower:
                return "DEPENDENCY_INSTALL_NODE", "NPM_PEER_DEPENDENCY_ERROR"
            if "engine" in error_lower and "unsupported" in error_lower:
                return "DEPENDENCY_INSTALL_NODE", "NPM_ENGINE_VERSION_MISMATCH"
            return "DEPENDENCY_INSTALL_NODE", "NPM_INSTALL_FAILED"

        if "yarn install" in error_lower or "yarn add" in error_lower:
            if "couldn't find package" in error_lower:
                return "DEPENDENCY_INSTALL_NODE", "YARN_PACKAGE_NOT_FOUND"
            if "network" in error_lower and "error" in error_lower:
                return "DEPENDENCY_INSTALL_NODE", "YARN_NETWORK_ERROR"
            return "DEPENDENCY_INSTALL_NODE", "YARN_INSTALL_FAILED"

        # ===================================================================
        # STAGE 6: Dependency Management - Go
        # ===================================================================
        if "go mod download" in error_lower or "go get" in error_lower or "go build" in error_lower:
            if "missing go.sum entry" in error_lower or "go.sum" in error_lower:
                return "DEPENDENCY_INSTALL_GO", "GO_MISSING_SUM_ENTRY"
            if "no required module" in error_lower or "module not found" in error_lower:
                return "DEPENDENCY_INSTALL_GO", "GO_MODULE_NOT_FOUND"
            if "invalid version" in error_lower:
                return "DEPENDENCY_INSTALL_GO", "GO_INVALID_VERSION"
            if "ambiguous import" in error_lower:
                return "DEPENDENCY_INSTALL_GO", "GO_AMBIGUOUS_IMPORT"
            return "DEPENDENCY_INSTALL_GO", "GO_DEPENDENCY_FAILED"

        # ===================================================================
        # STAGE 7: Dependency Management - Rust (Cargo)
        # ===================================================================
        if "cargo build" in error_lower or "cargo install" in error_lower:
            if "could not find" in error_lower and "crate" in error_lower:
                return "DEPENDENCY_INSTALL_RUST", "CARGO_CRATE_NOT_FOUND"
            if "failed to parse manifest" in error_lower:
                return "DEPENDENCY_INSTALL_RUST", "CARGO_MANIFEST_ERROR"
            return "DEPENDENCY_INSTALL_RUST", "CARGO_BUILD_FAILED"

        # ===================================================================
        # STAGE 8: Dependency Management - Java (Maven/Gradle)
        # ===================================================================
        if "mvn" in error_lower or "maven" in error_lower:
            if "could not resolve dependencies" in error_lower:
                return "DEPENDENCY_INSTALL_JAVA", "MAVEN_DEPENDENCY_NOT_FOUND"
            if "compilation failure" in error_lower:
                return "BUILD_COMPILATION_JAVA", "MAVEN_COMPILATION_FAILED"
            return "DEPENDENCY_INSTALL_JAVA", "MAVEN_BUILD_FAILED"

        if "gradle" in error_lower:
            if "could not resolve all dependencies" in error_lower:
                return "DEPENDENCY_INSTALL_JAVA", "GRADLE_DEPENDENCY_NOT_FOUND"
            return "DEPENDENCY_INSTALL_JAVA", "GRADLE_BUILD_FAILED"

        # ===================================================================
        # STAGE 9: Build/Compilation Errors (Language-specific)
        # ===================================================================
        # Python
        if "syntaxerror" in error_lower or "indentationerror" in error_lower:
            return "BUILD_COMPILATION_PYTHON", "PYTHON_SYNTAX_ERROR"
        if "modulenotfounderror" in error_lower or "importerror" in error_lower:
            return "BUILD_COMPILATION_PYTHON", "PYTHON_IMPORT_ERROR"

        # Node.js/JavaScript
        if "syntaxerror" in error_lower and ("javascript" in error_lower or "typescript" in error_lower):
            return "BUILD_COMPILATION_NODE", "JAVASCRIPT_SYNTAX_ERROR"
        if "webpack" in error_lower and "error" in error_lower:
            return "BUILD_COMPILATION_NODE", "WEBPACK_BUILD_ERROR"
        if "tsc" in error_lower and "error" in error_lower:
            return "BUILD_COMPILATION_NODE", "TYPESCRIPT_COMPILATION_ERROR"

        # Go
        if "go build" in error_lower and "error" in error_lower:
            if "undefined:" in error_lower:
                return "BUILD_COMPILATION_GO", "GO_UNDEFINED_REFERENCE"
            if "cannot use" in error_lower:
                return "BUILD_COMPILATION_GO", "GO_TYPE_ERROR"
            return "BUILD_COMPILATION_GO", "GO_BUILD_ERROR"

        # Rust
        if "error[e" in error_lower and "cargo" in error_lower:
            return "BUILD_COMPILATION_RUST", "RUST_COMPILER_ERROR"

        # C/C++
        if any(x in error_lower for x in ["gcc", "g++", "clang", "make:", "cmake"]):
            if "undefined reference" in error_lower:
                return "BUILD_COMPILATION_C", "C_UNDEFINED_REFERENCE"
            if "fatal error:" in error_lower and ".h:" in error_lower:
                return "BUILD_COMPILATION_C", "C_HEADER_NOT_FOUND"
            return "BUILD_COMPILATION_C", "C_COMPILATION_FAILED"

        # Generic compilation
        if any(x in error_lower for x in ["compilation error", "compilation failed", "build error"]):
            return "BUILD_COMPILATION", "GENERIC_COMPILATION_ERROR"

        # ===================================================================
        # STAGE 10: Runtime/Execution Errors
        # ===================================================================
        if "command not found" in error_lower or "no such file or directory" in error_lower:
            return "RUNTIME_EXECUTION", "COMMAND_NOT_FOUND"
        if "permission denied" in error_lower and "exec" in error_lower:
            return "RUNTIME_EXECUTION", "EXEC_PERMISSION_DENIED"

        # ===================================================================
        # STAGE 11: User/Permission Errors
        # ===================================================================
        if "useradd" in error_lower and "error" in error_lower:
            return "USER_MANAGEMENT", "USER_CREATION_FAILED"
        if "chown" in error_lower and "error" in error_lower:
            return "USER_MANAGEMENT", "CHOWN_FAILED"
        if "permission denied" in error_lower:
            return "PERMISSION_ERROR", "PERMISSION_DENIED"

        # ===================================================================
        # STAGE 12: Network/Download Errors
        # ===================================================================
        if any(x in error_lower for x in ["connection refused", "connection timeout", "network unreachable"]):
            return "NETWORK_ERROR", "CONNECTION_FAILED"
        if "certificate" in error_lower and "error" in error_lower:
            return "NETWORK_ERROR", "SSL_CERTIFICATE_ERROR"
        if "503" in error_lower or "service unavailable" in error_lower:
            return "NETWORK_ERROR", "SERVICE_UNAVAILABLE"

        # ===================================================================
        # STAGE 13: Disk/Storage Errors
        # ===================================================================
        if "no space left" in error_lower or "disk full" in error_lower:
            return "STORAGE_ERROR", "DISK_FULL"
        if "quota exceeded" in error_lower:
            return "STORAGE_ERROR", "QUOTA_EXCEEDED"

        # ===================================================================
        # STAGE 14: Docker-specific Errors
        # ===================================================================
        if "executor failed running" in error_lower:
            return "DOCKER_EXECUTOR", "RUN_COMMAND_FAILED"
        if "returned a non-zero code" in error_lower:
            # Try to extract which command failed
            if "step" in error_lower:
                return "DOCKER_EXECUTOR", "STEP_FAILED_NONZERO_EXIT"
            return "DOCKER_EXECUTOR", "COMMAND_NONZERO_EXIT"

        # ===================================================================
        # FALLBACK: Try to extract more specific info from error message
        # ===================================================================
        # Look for specific error patterns in the actual error text
        if "error" in error_lower:
            # Check for language-specific errors
            if any(lang in error_lower for lang in ["python", "pip"]):
                return "DEPENDENCY_INSTALL_PYTHON", "PYTHON_UNSPECIFIED_ERROR"
            if any(lang in error_lower for lang in ["node", "npm", "yarn", "javascript"]):
                return "DEPENDENCY_INSTALL_NODE", "NODE_UNSPECIFIED_ERROR"
            if "go" in error_lower and any(x in error_lower for x in ["module", "package"]):
                return "DEPENDENCY_INSTALL_GO", "GO_UNSPECIFIED_ERROR"
            if "cargo" in error_lower or "rust" in error_lower:
                return "DEPENDENCY_INSTALL_RUST", "RUST_UNSPECIFIED_ERROR"

        # Absolute fallback
        return "UNKNOWN_STAGE", "UNKNOWN_ERROR"

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
            repo_path = clone_repository(repo_url, target_dir=str(self.repos_dir))
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

            # Set global report directory for web search caching
            import planner_agent
            planner_agent.REPORT_DIRECTORY = str(report_dir)

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

                # Save Docker error to file if build failed
                if not docker_result["success"]:
                    error_file = report_dir_result / "docker_build_error.txt"
                    try:
                        with open(error_file, 'w', encoding='utf-8') as f:
                            f.write("="*80 + "\n")
                            f.write("DOCKER BUILD ERROR DETAILS\n")
                            f.write("="*80 + "\n\n")
                            f.write(f"Repository: {repo_name}\n")
                            f.write(f"URL: {repo_url}\n")
                            f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                            f.write(f"Stage: {docker_result.get('stage', 'UNKNOWN')}\n")
                            f.write(f"Error Type: {docker_result.get('error_type', 'UNKNOWN')}\n")
                            f.write(f"Exit Code: {docker_result.get('exit_code', -1)}\n")
                            if docker_result.get('error_snippet'):
                                f.write(f"\nError Snippet:\n{'-'*80}\n{docker_result['error_snippet']}\n{'-'*80}\n")
                            f.write(f"\nFull Error Output:\n{'-'*80}\n")
                            f.write(docker_result.get('error_message', 'No error message available'))
                            f.write(f"\n{'-'*80}\n")
                        print(f"[ERROR FILE] Docker error saved to: {error_file.name}")
                    except Exception as e:
                        print(f"[WARNING] Could not save error file: {e}")

                # Cleanup image if build succeeded
                if docker_result["success"]:
                    print(f"[CLEANUP] Removing Docker image {image_name}...")
                    self.docker_tester.cleanup_image(image_name)

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
                print(f"  Error Type: {docker.get('error_type', 'UNKNOWN')}")
                # Show error snippet if available, otherwise first 200 chars of full error
                error_snippet = docker.get('error_snippet', '')
                if error_snippet:
                    print(f"  Error Snippet: {error_snippet}")
                else:
                    error_msg = docker.get('error_message', '')
                    if error_msg:
                        print(f"  Error: {error_msg[:200]}...")
                # Mention error file
                if "agent_analysis" in result and "report_directory" in result["agent_analysis"]:
                    print(f"  Full error saved to: docker_build_error.txt")

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
