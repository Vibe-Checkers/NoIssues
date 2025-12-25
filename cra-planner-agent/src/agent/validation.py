import os
import time
import subprocess
import traceback
import threading
import logging
import re
from typing import Dict, Tuple, List, Optional
from pathlib import Path

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
            with open(dockerfile_path, 'r') as f:
                content = f.read()
        except Exception as e:
            return False, [f"Cannot read Dockerfile: {e}"], 0

        # Check for gaming patterns
        for pattern in DockerfileQualityChecker.GAMING_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
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
                    logger.info(f"[ARTIFACT CHECK] Found Java artifacts: {result.stdout.strip()[:100]}")
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
                        issues.append("Cannot verify pip packages in image")
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
                        issues.append("Cargo.toml exists but no compiled binaries found in target/")
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
            logger.warning("[ARTIFACT CHECK] Validation timed out")
            return True, []  # Don't fail on timeout
        except Exception as e:
            logger.warning(f"[ARTIFACT CHECK] Validation error: {e}")
            return True, []  # Don't fail on validation errors

        return True, []


class DockerBuildTester:
    """Tests generated Dockerfiles by actually building them with Docker.
    
    Includes automatic detection and recovery from BuildKit cache corruption
    that can occur with high parallel worker counts.
    """

    def __init__(self, timeout: int = 600, platform: str = None, serialize_builds: bool = False):
        """
        Initialize Docker build tester.

        Args:
            timeout: Maximum time in seconds to wait for Docker build (default: 10 minutes)
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
                logger.warning("[DOCKER] Pruning BuildKit cache to recover from corruption...")
                result = subprocess.run(
                    ["docker", "builder", "prune", "--all", "--force"],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode == 0:
                    logger.info("[DOCKER] BuildKit cache pruned successfully")
                    return True
                else:
                    logger.error(f"[DOCKER] Prune failed: {result.stderr}")
                    return False
            except Exception as e:
                logger.error(f"[DOCKER] Prune exception: {e}")
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
                logger.warning(f"[QUALITY CHECK] Dockerfile quality check failed (score: {quality_score}/100)")
                logger.warning(f"[QUALITY CHECK] Issues: {quality_issues}")
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
                logger.info(f"[QUALITY CHECK] Passed (score: {quality_score}/100)")

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
                    logger.info("[ARTIFACT VALIDATION] Checking for expected build artifacts...")
                    artifacts_valid, artifact_issues = BuildArtifactValidator.validate_artifacts(
                        image_name, project_language, context_path
                    )

                    build_result["artifacts_validated"] = True
                    build_result["artifacts_valid"] = artifacts_valid
                    build_result["artifact_issues"] = artifact_issues

                    if not artifacts_valid:
                        logger.warning(f"[ARTIFACT VALIDATION] Failed: {artifact_issues}")
                        build_result["success"] = False
                        build_result["stage"] = "ARTIFACT_VALIDATION"
                        build_result["error_type"] = "MISSING_ARTIFACTS"
                        build_result["error_message"] = f"Build artifacts validation failed: {'; '.join(artifact_issues)}"
                    else:
                        logger.info("[ARTIFACT VALIDATION] Passed - expected artifacts found")
                else:
                    build_result["artifacts_validated"] = False

                return build_result
            else:
                # Parse error to determine stage
                full_error = result.stderr or result.stdout or "Unknown error"
                
                # Check for cache corruption and auto-recover
                if retry_on_cache_error and not _is_retry and self._is_cache_corruption_error(full_error):
                    logger.warning("[DOCKER] Detected BuildKit cache corruption, attempting recovery...")
                    if self.prune_buildkit_cache():
                        # Wait a moment for Docker to stabilize
                        time.sleep(2)
                        # Retry the build
                        return self._do_build(dockerfile_path, context_path, image_name,
                                             retry_on_cache_error=False, project_language=project_language,
                                             validate_artifacts=validate_artifacts, _is_retry=True)
                    else:
                        logger.error("[DOCKER] Cache prune failed, returning original error")
                
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
        """Parse Docker error output to determine which Dockerfile step failed."""
        error_lower = error_output.lower()

        # Extract the failed Docker step from the build output
        failed_step = self._extract_failed_docker_step(error_output)

        # 1. Docker Daemon Issues
        if any(x in error_lower for x in ["cannot connect to the docker daemon", "is the docker daemon running", "docker: not found"]):
            return "DOCKER_DAEMON", failed_step or "Docker daemon not accessible"

        # 2. Docker BuildKit Storage Corruption
        if "parent snapshot" in error_lower and "does not exist" in error_lower:
            return "INFRASTRUCTURE_CORRUPTION", failed_step or "Docker BuildKit cache corrupted"

        # 3. Platform/Architecture Incompatibility
        is_platform_mismatch = (
            "exec format error" in error_lower or
            ("image" in error_lower and "platform" in error_lower and "does not match" in error_lower) or
            "invalidbaseimageplatform" in error_lower or
            "requested image's platform" in error_lower
        )
        if is_platform_mismatch:
            return "PLATFORM_INCOMPATIBLE", failed_step or "Base image platform mismatch"

        # 4. Base Image Issues
        if "docker image format v1" in error_lower or "manifest version 2, schema 1" in error_lower:
            return "IMAGE_DEPRECATED", failed_step or "Image uses deprecated Schema V1"
        
        if any(x in error_lower for x in ["failed to resolve", "manifest unknown", "pull access denied", "image not found"]):
            return "IMAGE_PULL", failed_step or "Failed to pull base image"

        # 5. Dockerfile Syntax
        if any(x in error_lower for x in ["dockerfile parse error", "unknown instruction"]):
            return "DOCKERFILE_SYNTAX", failed_step or "Dockerfile syntax error"

        # 6. File Copy/Add
        if "failed to compute cache key" in error_lower and "not found" in error_lower:
            return "FILE_COPY_MISSING", failed_step or "COPY command references missing file"
        
        if any(x in error_lower for x in ["copy failed", "add failed"]) or ("stat" in error_lower and "no such file" in error_lower):
            return "FILE_COPY", failed_step or "File copy/add failed"

        # 7. Dependency Installation
        if any(x in error_lower for x in ["node-gyp", "gyp err", "gcc: command not found", "make: command not found"]):
            return "DEPENDENCY_BUILD_TOOLS", failed_step or "Missing build tools"

        # 8. Build/Compilation
        if any(x in error_lower for x in ["compilation error", "build error", "webpack", "tsc"]):
            return "BUILD_COMPILE", failed_step or "Build/compilation failed"

        # 9. Runtime Execution
        if "exit code: 127" in error_lower or "command not found" in error_lower:
            return "BUILD_TOOL_MISSING", failed_step or "Build tool not found"

        return "UNKNOWN", failed_step or "Unknown error - check full log"

    def _extract_failed_docker_step(self, error_output: str) -> str:
        """Extract the specific Docker RUN/COPY/FROM command that failed."""
        import re
        # Pattern 1: ERROR [stage X/Y] COMMAND
        match = re.search(r'ERROR \[.*?\] (RUN|COPY|ADD|FROM|WORKDIR).*', error_output, re.IGNORECASE)
        if match:
            return match.group(0).replace('ERROR ', '').strip()
            
        # Pattern 2: executor failed running [/bin/sh -c COMMAND]
        match = re.search(r'executor failed running \[/bin/sh -c ([^\]]+)\]', error_output, re.IGNORECASE)
        if match:
            return f"RUN {match.group(1).strip()}"

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
