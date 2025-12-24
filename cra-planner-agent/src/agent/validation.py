import os
import time
import subprocess
import traceback
from typing import Dict, Tuple

class DockerBuildTester:
    """Tests generated Dockerfiles by actually building them with Docker."""

    def __init__(self, timeout: int = 1200, platform: str = None):
        """
        Initialize Docker build tester.

        Args:
            timeout: Maximum time in seconds to wait for Docker build (default: 20 minutes)
                     Increased from 10 to 20 minutes to support large C++/Java projects (guava, opencv)
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

    def check_infrastructure_health(self) -> Tuple[bool, str]:
        """Verify Docker infrastructure is healthy."""
        if self._check_docker():
            return True, "Infrastructure health check passed"
        return False, "Docker daemon is not running or not accessible"

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

        # Build Docker image with BuildKit enabled
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

            # Enable BuildKit for modern Dockerfile features (--mount, --secret, etc.)
            env = os.environ.copy()
            env["DOCKER_BUILDKIT"] = "1"

            platform_info = f" (platform: {self.platform})" if self.platform else ""
            print(f"[DOCKER BUILD] Running with BuildKit: {' '.join(cmd)}{platform_info}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env
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
                    "stdout": result.stdout[-2000:] if result.stdout else "",
                    "stderr": result.stderr[-2000:] if result.stderr else None
                }
            else:
                # Parse error to determine stage
                full_error = result.stderr or result.stdout or "Unknown error"
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
                    "stdout": result.stdout[-2000:] if result.stdout else "",
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
        """
        error_lower = error_output.lower()

        # Extract the failed Docker step from the build output
        failed_step = self._extract_failed_docker_step(error_output)

        # 1. Docker Daemon Issues
        if any(x in error_lower for x in ["cannot connect to the docker daemon", "is the docker daemon running", "docker: not found", "docker: command not found"]):
            return "DOCKER_DAEMON", failed_step or "Docker daemon not accessible"

        # 2. Docker BuildKit Storage Corruption
        if "parent snapshot" in error_lower and "does not exist" in error_lower:
            return "INFRASTRUCTURE_CORRUPTION", failed_step or "Docker BuildKit cache corrupted - requires prune"

        # 3. Platform/Architecture Incompatibility
        is_platform_mismatch = (
            "exec format error" in error_lower or
            ("image" in error_lower and "platform" in error_lower and "does not match" in error_lower) or
            ("invalidbaseimageplatform" in error_lower) or
            ("requested image's platform" in error_lower)
        )
        if is_platform_mismatch:
            return "PLATFORM_INCOMPATIBLE", failed_step or "Base image platform mismatch (amd64 vs arm64)"

        # 4. Base Image Issues
        if "docker image format v1" in error_lower or "manifest version 2, schema 1" in error_lower:
            return "IMAGE_DEPRECATED", failed_step or "Image uses deprecated Schema V1"
        
        if (("release check failed" in error_lower or "release file" in error_lower) and "expired" in error_lower) or \
           (("archive.ubuntu.com" in error_lower or "security.debian.org" in error_lower) and "404" in error_lower and "apt-get" in error_lower):
            return "EOL_DISTRO", failed_step or "OS distribution is End-of-Life"

        if "failed validation" in error_lower or "failed to load cache key" in error_lower:
            return "IMAGE_VALIDATION_FAILED", failed_step or "Docker image failed validation"

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
        if any(x in error_lower for x in ["enoent", "no such file or directory"]) and \
           any(x in error_lower for x in ["package.json", "requirements.txt", "pom.xml", "cargo.toml", "go.mod"]):
            return "DEPENDENCY_FILE_MISSING", failed_step or "Dependency file not found at expected location"

        if any(x in error_lower for x in ["node-gyp", "gyp err", "gcc: command not found", "make: command not found", "build-essential", "python: command not found"]):
            return "DEPENDENCY_BUILD_TOOLS", failed_step or "Missing build tools for native dependencies"

        # 8. Build/Compilation
        if any(x in error_lower for x in ["compilation error", "build error", "webpack", "tsc"]):
            return "BUILD_COMPILE", failed_step or "Build/compilation failed"

        # 9. Runtime Execution
        if ("exit code: 127" in error_lower or "command not found" in error_lower):
            return "BUILD_TOOL_MISSING", failed_step or "Build tool or script not found"

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
