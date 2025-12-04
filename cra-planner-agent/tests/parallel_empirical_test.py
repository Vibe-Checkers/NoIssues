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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

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

            # Step 1: Clone repository
            self.log(repo_name, "Cloning repository...", to_console=True)
            clone_start = time.time()
            repo_path = clone_repository(repo_url, target_dir=str(self.repos_dir), auto_remove=True)
            clone_duration = time.time() - clone_start

            result["clone"] = {
                "success": True,
                "duration_seconds": clone_duration,
                "path": repo_path
            }
            self.log(repo_name, f"Cloned successfully ({clone_duration:.1f}s)", to_console=True)

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

            # Step 4: Test Dockerfile
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
                # Read Dockerfile content
                with open(dockerfile_path, 'r', encoding='utf-8') as f:
                    dockerfile_content = f.read()
                result["dockerfile_content"] = dockerfile_content

                # Test Docker build
                image_name = f"parallel-empirical-{repo_name.lower()}:latest"
                docker_result = self.docker_tester.build_dockerfile(
                    str(dockerfile_path),
                    repo_path,
                    image_name
                )

                result["dockerfile_test"] = docker_result

                if docker_result["success"]:
                    self.log(
                        repo_name,
                        f"Docker build SUCCESS ({docker_result['duration_seconds']:.1f}s)",
                        to_console=True
                    )
                    # Cleanup image
                    self.docker_tester.cleanup_image(image_name)
                else:
                    # Log failure
                    stage = docker_result.get('stage', 'UNKNOWN')
                    failed_cmd = docker_result.get('failed_command', 'Unknown')
                    self.log(
                        repo_name,
                        f"Docker build FAILED at {stage}: {failed_cmd}",
                        to_console=True
                    )

                    # Save error summary in report directory
                    error_summary_file = report_dir_result / "docker_build_error_summary.txt"
                    with open(error_summary_file, 'w', encoding='utf-8') as f:
                        f.write("="*80 + "\n")
                        f.write("DOCKER BUILD ERROR SUMMARY\n")
                        f.write("="*80 + "\n\n")
                        f.write(f"Repository: {repo_name}\n")
                        f.write(f"URL: {repo_url}\n")
                        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
                        f.write(f"Failure Stage: {stage}\n")
                        f.write(f"Failed Docker Command: {failed_cmd}\n")
                        f.write(f"Exit Code: {docker_result.get('exit_code', -1)}\n\n")

                        if docker_result.get('error_snippet'):
                            f.write(f"Error Snippet:\n{'-'*80}\n")
                            f.write(f"{docker_result['error_snippet']}\n")
                            f.write(f"{'-'*80}\n\n")

                        f.write(f"See full error: docker_errors/{repo_name}_docker_error.log\n")

                    # Save full error log
                    raw_error_file = self.docker_errors_dir / f"{repo_name}_docker_error.log"
                    with open(raw_error_file, 'w', encoding='utf-8') as f:
                        f.write(f"# Docker Build Error Log\n")
                        f.write(f"# Repository: {repo_name}\n")
                        f.write(f"# URL: {repo_url}\n")
                        f.write(f"# Timestamp: {datetime.now().isoformat()}\n")
                        f.write(f"# Stage: {stage}\n")
                        f.write(f"# Failed Command: {failed_cmd}\n")
                        f.write(f"# Exit Code: {docker_result.get('exit_code', -1)}\n")
                        f.write("#" + "="*78 + "\n\n")
                        f.write(docker_result.get('error_message', 'No error message available'))

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
        if "dockerfile_test" in result and not result["dockerfile_test"].get("success", False):
            if "error_message" in result["dockerfile_test"]:
                # Keep only last 5000 chars of error message in memory
                error_msg = result["dockerfile_test"]["error_message"]
                if len(error_msg) > 5000:
                    result["dockerfile_test"]["error_message"] = error_msg[-5000:]
                    result["dockerfile_test"]["error_message_truncated"] = True

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

        # Categorize failures
        failure_stages = {}
        for result in self.results:
            if not result["success"] and "dockerfile_test" in result:
                stage = result["dockerfile_test"].get("stage", "UNKNOWN")
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

                if not result["success"] and "dockerfile_test" in result:
                    docker = result["dockerfile_test"]
                    f.write(f"   Failure Stage: {docker.get('stage', 'UNKNOWN')}\n")
                    f.write(f"   Failed Command: {docker.get('failed_command', 'Unknown')}\n")

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
            print(f"{status} {result['repo_name']:30s} | {duration:10s}")
            if not result["success"] and "dockerfile_test" in result:
                stage = result["dockerfile_test"].get("stage", "UNKNOWN")
                print(f"  └─ Failed at: {stage}")
        print("-"*80)

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
