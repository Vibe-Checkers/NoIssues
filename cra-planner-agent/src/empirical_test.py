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
import logging
import re
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

print("[STARTUP] Loading agent modules...")
from run_agent import clone_repository, analyze_repository, detect_project_language
from planner_agent import create_planner_agent

print("[STARTUP] All imports loaded successfully!")


# Configure logging if not already configured
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global lock for serializing Docker builds when needed
_docker_build_lock = threading.Lock()
_buildkit_prune_lock = threading.Lock()


class DockerfileQualityChecker:
    """
    Validates Dockerfile quality and detects gaming patterns.

    Prevents false positives where Dockerfiles pass Docker build but don't actually
    build the project properly (e.g., skipping build steps, using placeholders).
    """

    # Patterns that indicate the agent is gaming the system
    GAMING_PATTERNS = [
        r'echo\s+["\'].*placeholder.*["\']',  # Placeholder echo statements
        r'echo\s+["\'].*fix\s+applied.*["\']',  # Fake fix messages
        r'echo\s+["\'].*skipp(ed|ing).*["\']',  # Admitted skipping
        r'echo\s+["\'].*todo.*["\']',  # TODO placeholders
        r'#\s*\(Dockerfile content continues here',  # Incomplete marker
        r'RUN\s+echo\s+.*>\s*/dev/null',  # Meaningless echo to /dev/null
        r'RUN\s+true\s*$',  # No-op RUN true
        r'RUN\s+:\s*$',  # No-op RUN :
        r'\|\|\s*true\s*$',  # Suppressing errors with || true
        r'\|\|\s*exit\s*0\s*$',  # Suppressing errors with || exit 0
    ]

    # Patterns that indicate skipped build steps
    SKIP_PATTERNS = [
        r'["\'].*build\s+skipped.*["\']',
        r'["\'].*skipping\s+(build|compilation|test).*["\']',
        r'if\s+\[.*\];\s*then\s+echo\s+.*skipp',
        r'#\s*RUN\s+(mvn|gradle|npm\s+install|pip\s+install)',  # Commented out critical steps
    ]

    @staticmethod
    def check_dockerfile_quality(dockerfile_path: str, project_language: Optional[str] = None) -> Tuple[bool, List[str], int]:
        """
        Check Dockerfile for quality issues and gaming patterns.

        Args:
            dockerfile_path: Path to Dockerfile
            project_language: Detected project language (Java, Python, etc.)

        Returns:
            Tuple of (is_valid, issues_list, quality_score)
            - is_valid: False if critical issues found
            - issues_list: List of detected issues
            - quality_score: 0-100, higher is better
        """
        issues = []
        quality_score = 100

        try:
            with open(dockerfile_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return False, [f"Cannot read Dockerfile: {e}"], 0

        # Check for gaming patterns
        for pattern in DockerfileQualityChecker.GAMING_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                issues.append(f"Gaming pattern detected: {pattern}")
                quality_score -= 30

        # Check for skip patterns
        for pattern in DockerfileQualityChecker.SKIP_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                issues.append(f"Build step appears to be skipped: {pattern}")
                quality_score -= 25

        # Check for incomplete/placeholder content
        if '# (' in content and 'continues here' in content.lower():
            issues.append("Incomplete Dockerfile with placeholder comments")
            quality_score -= 50

        # Check for minimal content (just FROM + COPY)
        run_commands = len(re.findall(r'^RUN\s+', content, re.MULTILINE))
        if run_commands == 0:
            issues.append("No RUN commands - Dockerfile doesn't build anything")
            quality_score -= 40
        elif run_commands == 1:
            # Check if the single RUN is just echo or no-op
            if re.search(r'^RUN\s+(echo|true|:)\s', content, re.MULTILINE):
                issues.append("Only RUN command is a no-op (echo/true)")
                quality_score -= 40

        # Language-specific validation
        if project_language:
            lang_issues, lang_score = DockerfileQualityChecker._check_language_specific(content, project_language)
            issues.extend(lang_issues)
            quality_score += lang_score  # Can be negative

        # Ensure score is in valid range
        quality_score = max(0, min(100, quality_score))

        # Critical failure: quality score below 40
        is_valid = quality_score >= 40 and len(issues) == 0

        return is_valid, issues, quality_score

    @staticmethod
    def _check_language_specific(content: str, language: str) -> Tuple[List[str], int]:
        """Check for language-specific build requirements."""
        issues = []
        score_adjustment = 0

        lang_lower = language.lower()

        # Java projects
        if lang_lower in ['java', 'kotlin', 'scala']:
            has_maven = 'mvn' in content
            has_gradle = 'gradle' in content or './gradlew' in content

            if not has_maven and not has_gradle:
                issues.append("Java project missing Maven or Gradle build command")
                score_adjustment -= 30
            elif 'mvn' in content and '-DskipTests' not in content:
                # Bonus for proper testing
                score_adjustment += 5

        # Python projects
        elif lang_lower == 'python':
            has_pip = 'pip install' in content
            has_requirements = 'requirements.txt' in content

            if has_requirements and not has_pip:
                issues.append("requirements.txt referenced but pip install is commented out or missing")
                score_adjustment -= 25
            elif has_pip and '-r requirements' in content:
                # Proper dependency installation
                score_adjustment += 5

        # Node.js/TypeScript projects
        elif lang_lower in ['javascript', 'typescript']:
            has_npm = 'npm install' in content or 'npm ci' in content
            has_yarn = 'yarn install' in content
            has_package_json = 'package.json' in content

            if has_package_json and not (has_npm or has_yarn):
                issues.append("package.json exists but npm/yarn install is missing")
                score_adjustment -= 25
            elif has_npm or has_yarn:
                score_adjustment += 5

        # Rust projects
        elif lang_lower == 'rust':
            if 'cargo build' not in content:
                issues.append("Rust project missing 'cargo build' command")
                score_adjustment -= 30
            else:
                score_adjustment += 5

        # Go projects
        elif lang_lower in ['go', 'golang']:
            if 'go build' not in content and 'go install' not in content:
                issues.append("Go project missing 'go build' or 'go install' command")
                score_adjustment -= 30
            else:
                score_adjustment += 5

        # C/C++ projects
        elif lang_lower in ['c', 'c++', 'cpp']:
            has_make = 'make' in content
            has_cmake = 'cmake' in content

            if not has_make and not has_cmake:
                issues.append("C/C++ project missing make or cmake build command")
                score_adjustment -= 25

        return issues, score_adjustment


class BuildArtifactValidator:
    """
    Validates that expected build artifacts were created during Docker build.

    This prevents false positives where Docker build succeeds but no actual
    compilation or dependency installation occurred.
    """

    @staticmethod
    def validate_artifacts(image_name: str, project_language: Optional[str] = None,
                          context_path: str = None) -> Tuple[bool, List[str]]:
        """
        Check if expected build artifacts exist in the Docker image.

        Args:
            image_name: Name of built Docker image
            project_language: Detected project language
            context_path: Path to repository (to check for package manifests)

        Returns:
            Tuple of (artifacts_valid, issues_list)
        """
        issues = []

        if not project_language:
            # Can't validate without knowing language
            return True, []

        lang_lower = project_language.lower()

        try:
            # Java: Check for JAR/WAR files
            if lang_lower in ['java', 'kotlin', 'scala']:
                # Check target/ or build/ directory for artifacts
                result = subprocess.run(
                    ["docker", "run", "--rm", image_name, "sh", "-c",
                     "find target build -name '*.jar' -o -name '*.war' 2>/dev/null | head -5"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0 and result.stdout.strip():
                    # Found JAR/WAR files
                    print(f"[ARTIFACT CHECK] Found Java artifacts: {result.stdout.strip()[:100]}")
                else:
                    # Check if pom.xml or build.gradle exists
                    if context_path:
                        has_pom = (Path(context_path) / "pom.xml").exists()
                        has_gradle = (Path(context_path) / "build.gradle").exists() or \
                                   (Path(context_path) / "build.gradle.kts").exists()

                        if has_pom or has_gradle:
                            issues.append("Java project has build files but no JAR/WAR artifacts found in image")
                            return False, issues

            # Python: Check for installed packages or successful imports
            elif lang_lower == 'python':
                # Check if requirements.txt exists in context
                if context_path and (Path(context_path) / "requirements.txt").exists():
                    # Verify pip packages were installed
                    result = subprocess.run(
                        ["docker", "run", "--rm", image_name, "pip", "list"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if result.returncode == 0:
                        installed_packages = result.stdout.lower()
                        # Should have more than just pip and setuptools
                        package_count = len([l for l in installed_packages.split('\n') if l.strip() and not l.startswith('-')])
                        if package_count <= 3:  # Just pip, setuptools, wheel
                            issues.append("requirements.txt exists but no packages appear to be installed")
                            return False, issues
                    else:
                        issues.append("Cannot verify pip packages in image (pip missing or failed)")
                        return False, issues

            # Node.js: Check for node_modules
            elif lang_lower in ['javascript', 'typescript']:
                if context_path and (Path(context_path) / "package.json").exists():
                    result = subprocess.run(
                        ["docker", "run", "--rm", image_name, "sh", "-c",
                         "[ -d node_modules ] && echo 'EXISTS' || echo 'MISSING'"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if result.returncode == 0 and 'MISSING' in result.stdout:
                        issues.append("package.json exists but node_modules directory not found in image")
                        return False, issues

            # Rust: Check for target/ directory with binaries
            elif lang_lower == 'rust':
                if context_path and (Path(context_path) / "Cargo.toml").exists():
                    result = subprocess.run(
                        ["docker", "run", "--rm", image_name, "sh", "-c",
                         "find target -type f -perm -111 2>/dev/null | head -5"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if result.returncode == 0 and not result.stdout.strip():
                        # Try finding ANY interesting file in release/debug in case permissions masked it
                        result_fallback = subprocess.run(
                             ["docker", "run", "--rm", image_name, "sh", "-c",
                              "ls -R target | grep -E '\.(exe|a|so|rlib|dylib)' | head -5"],
                             capture_output=True, text=True, timeout=10
                        )
                        if not result_fallback.stdout.strip():
                            issues.append("Cargo.toml exists but no compiled artifacts found in target/")
                            return False, issues

            # Go: Check for compiled binary
            elif lang_lower in ['go', 'golang']:
                if context_path and (Path(context_path) / "go.mod").exists():
                    result = subprocess.run(
                        ["docker", "run", "--rm", image_name, "sh", "-c",
                         "find . -maxdepth 2 -type f -perm -111 ! -path './.*' 2>/dev/null | head -5"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )

                    if result.returncode == 0 and not result.stdout.strip():
                         issues.append("go.mod exists but no compiled binary found")
                         return False, issues

        except subprocess.TimeoutExpired:
            print("[ARTIFACT CHECK] Validation timed out")
            return True, []  # Don't fail on timeout
        except Exception as e:
            print(f"[ARTIFACT CHECK] Validation error: {e}")
            return True, []  # Don't fail on validation errors

        return True, []


class DockerBuildTester:
    """Tests generated Dockerfiles by actually building them with Docker.
    
    Includes automatic detection and recovery from BuildKit cache corruption
    that can occur with high parallel worker counts.
    """

    def __init__(self, timeout: int = 1200, platform: str = None, serialize_builds: bool = False):
        """
        Initialize Docker build tester.

        Args:
            timeout: Maximum time in seconds to wait for Docker build (default: 20 minutes)
            platform: Optional platform to build for (e.g., "linux/amd64", "linux/arm64")
            serialize_builds: If True, use a global lock to prevent parallel builds.
                             Slower but prevents cache corruption entirely.
        """
        self.timeout = timeout
        self.platform = platform
        self.serialize_builds = serialize_builds
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

    def check_infrastructure_health(self) -> Tuple[bool, str]:
        """Verify Docker infrastructure is healthy."""
        if self._check_docker():
            return True, "Infrastructure health check passed"
        return False, "Docker daemon is not running or not accessible"

    def prune_buildkit_cache(self) -> bool:
        """
        Prune Docker BuildKit cache to recover from corruption.
        
        Uses a lock to prevent multiple workers from pruning simultaneously.
        
        Returns:
            True if prune succeeded, False otherwise
        """
        with _buildkit_prune_lock:
            try:
                print("[DOCKER] Pruning BuildKit cache to recover from corruption...")
                result = subprocess.run(
                    ["docker", "builder", "prune", "--all", "--force"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    print("[DOCKER] BuildKit cache pruned successfully")
                    return True
                else:
                    print(f"[DOCKER] Prune failed: {result.stderr}")
                    return False
            except Exception as e:
                print(f"[DOCKER] Prune exception: {e}")
                return False

    def _is_cache_corruption_error(self, error_output: str) -> bool:
        """Check if the error is a BuildKit cache corruption error."""
        error_lower = error_output.lower()
        return (
            ("parent snapshot" in error_lower and "does not exist" in error_lower) or
            ("failed to load cache key" in error_lower and "not found" in error_lower) or
            ("failed to get state for index" in error_lower)
        )

    def build_dockerfile(self, dockerfile_path: str, context_path: str, image_name: str,
                         retry_on_cache_error: bool = True, project_language: Optional[str] = None,
                         validate_artifacts: bool = True, validate_quality: bool = True) -> Dict:
        """
        Build a Dockerfile and return detailed results with comprehensive validation.

        Automatically detects BuildKit cache corruption and recovers by pruning
        the cache and retrying the build.

        Includes quality checks and artifact validation to prevent false positives.

        Args:
            dockerfile_path: Path to Dockerfile
            context_path: Path to build context (usually repository root)
            image_name: Name to tag the built image
            retry_on_cache_error: If True, auto-recover from cache corruption
            project_language: Detected project language for semantic validation
            validate_artifacts: If True, check for expected build artifacts after successful build
            validate_quality: If True, check Dockerfile quality before building

        Returns:
            Dictionary with build results including success status, stage, error details,
            quality score, and artifact validation results
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

        # Quality check before building (prevents wasting time on bad Dockerfiles)
        if validate_quality:
            is_valid, quality_issues, quality_score = DockerfileQualityChecker.check_dockerfile_quality(
                dockerfile_path, project_language
            )

            if not is_valid:
                print(f"[QUALITY CHECK] Dockerfile quality check failed (score: {quality_score}/100)")
                print(f"[QUALITY CHECK] Issues: {quality_issues}")
                return {
                    "success": False,
                    "stage": "QUALITY_CHECK",
                    "error_type": "QUALITY_FAILED",
                    "error_message": f"Dockerfile quality check failed. Issues: {'; '.join(quality_issues)}",
                    "quality_score": quality_score,
                    "quality_issues": quality_issues,
                    "exit_code": -1,
                    "duration_seconds": 0
                }
            else:
                # print(f"[QUALITY CHECK] Passed (score: {quality_score}/100)")
                pass

        # Optionally serialize builds to prevent cache corruption
        if self.serialize_builds:
            with _docker_build_lock:
                return self._do_build(dockerfile_path, context_path, image_name, retry_on_cache_error,
                                     project_language, validate_artifacts)
        else:
            return self._do_build(dockerfile_path, context_path, image_name, retry_on_cache_error,
                                 project_language, validate_artifacts)

    def _do_build(self, dockerfile_path: str, context_path: str, image_name: str,
                  retry_on_cache_error: bool = True, project_language: Optional[str] = None,
                  validate_artifacts: bool = True, _is_retry: bool = False) -> Dict:
        """Internal build method with retry logic for cache corruption and artifact validation."""
        start_time = time.time()

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

            # Enable BuildKit for modern Dockerfile features
            env = os.environ.copy()
            env["DOCKER_BUILDKIT"] = "1"

            platform_info = f" (platform: {self.platform})" if self.platform else ""
            retry_info = " [RETRY after cache prune]" if _is_retry else ""
            print(f"[DOCKER BUILD]{retry_info} Running: {' '.join(cmd)}{platform_info}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env
            )

            duration = time.time() - start_time

            if result.returncode == 0:
                # Docker build succeeded - now validate artifacts
                build_result = {
                    "success": True,
                    "stage": "BUILD_COMPLETE",
                    "error_type": None,
                    "error_message": None,
                    "exit_code": 0,
                    "duration_seconds": duration,
                    "stdout": result.stdout[-2000:] if result.stdout else "",
                    "stderr": result.stderr[-2000:] if result.stderr else None,
                    "was_retry": _is_retry
                }

                # Validate build artifacts if enabled
                if validate_artifacts and project_language:
                    # print("[ARTIFACT VALIDATION] Checking for expected build artifacts...")
                    artifacts_valid, artifact_issues = BuildArtifactValidator.validate_artifacts(
                        image_name, project_language, context_path
                    )

                    build_result["artifacts_validated"] = True
                    build_result["artifacts_valid"] = artifacts_valid
                    build_result["artifact_issues"] = artifact_issues

                    if not artifacts_valid:
                        # Log warning
                        print(f"[ARTIFACT VALIDATION] Failed: {artifact_issues}")
                        build_result["success"] = False
                        build_result["stage"] = "ARTIFACT_VALIDATION"
                        build_result["error_type"] = "MISSING_ARTIFACTS"
                        build_result["error_message"] = f"Build artifacts validation failed: {'; '.join(artifact_issues)}"
                    else:
                        # print("[ARTIFACT VALIDATION] Passed - expected artifacts found")
                        pass
                else:
                    build_result["artifacts_validated"] = False

                return build_result
            else:
                # Parse error to determine stage
                full_error = result.stderr or result.stdout or "Unknown error"
                
                # Check for cache corruption and auto-recover
                if retry_on_cache_error and not _is_retry and self._is_cache_corruption_error(full_error):
                    print("[DOCKER] Detected BuildKit cache corruption, attempting recovery...")
                    if self.prune_buildkit_cache():
                        # Wait a moment for Docker to stabilize
                        time.sleep(2)
                        # Retry the build
                        return self._do_build(dockerfile_path, context_path, image_name,
                                             retry_on_cache_error=False, project_language=project_language,
                                             validate_artifacts=validate_artifacts, _is_retry=True)
                    else:
                        print("[DOCKER] Cache prune failed, returning original error")
                
                stage, failed_docker_command = self._parse_docker_error(full_error)

                # Extract a concise error snippet
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
                    "failed_command": failed_docker_command,
                    "error_message": full_error,
                    "error_snippet": error_snippet,
                    "exit_code": result.returncode,
                    "duration_seconds": duration,
                    "stdout": result.stdout[-2000:] if result.stdout else "",
                    "stderr": result.stderr[-2000:] if result.stderr else None,
                    "was_retry": _is_retry
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
        # STRICT detection to avoid false positives (e.g., matching "platform" in dependency names)
        # Must match specific Docker error phrases or "exec format error"
        is_platform_mismatch = (
            "exec format error" in error_lower or
            ("image" in error_lower and "platform" in error_lower and "does not match" in error_lower) or
            ("invalidbaseimageplatform" in error_lower) or
            ("requested image's platform" in error_lower)
        )
        if is_platform_mismatch:
            return "PLATFORM_INCOMPATIBLE", failed_step or "Base image platform mismatch (amd64 vs arm64)"

        # 4. Base Image Issues
        # 4a. Deprecated V1 Manifest (Ancient Images)
        if "docker image format v1" in error_lower or "manifest version 2, schema 1" in error_lower:
            return "IMAGE_DEPRECATED", failed_step or "Image uses deprecated Schema V1 (unsupported by modern Docker)"

        # 4b. EOL Distribution (Apt Repositories Gone)
        if (("release check failed" in error_lower or "release file" in error_lower) and "expired" in error_lower) or \
           (("archive.ubuntu.com" in error_lower or "security.debian.org" in error_lower) and "404" in error_lower and "apt-get" in error_lower):
            return "EOL_DISTRO", failed_step or "OS distribution is End-of-Life (package repositories removed)"

        # 4c. Image Validation Failures
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

        self.docker_tester = DockerBuildTester(timeout=1200)  # 20 minutes for large projects

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
                    image_name,
                    project_language=detected_language  # Passed for validation
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
