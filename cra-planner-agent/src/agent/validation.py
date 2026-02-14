import os
import time
import subprocess
import traceback
import threading
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

# Global lock for serializing Docker builds when needed
_docker_build_lock = threading.Lock()
_buildkit_prune_lock = threading.Lock()


class DockerBuildTester:
    """Tests generated Dockerfiles by actually building them with Docker.
    
    Includes automatic detection and recovery from BuildKit cache corruption
    that can occur with high parallel worker counts.
    """

    def __init__(self, timeout: int = 1200, platform: str = None, serialize_builds: bool = False):
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
        if self.docker_available:
            self._ensure_compatible_builder()

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
            try:
                self._ensure_compatible_builder()
                return True, "Infrastructure health check passed"
            except Exception as e:
                return False, f"Infrastructure health check failed: {e}"
        return False, "Docker daemon is not running or not accessible"

    def _ensure_compatible_builder(self):
        """
        Ensure that the current Docker Buildx builder supports local cache exports.
        The default 'docker' driver does NOT support this. We need a 'docker-container' driver.
        """
        try:
            # 1. Check current builder driver
            result = subprocess.run(
                ["docker", "buildx", "inspect"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Look for Driver: docker (the limited one)
            is_limited_driver = False
            for line in result.stdout.splitlines():
                if "Driver:" in line and "docker" in line.lower() and "container" not in line.lower():
                    is_limited_driver = True
                    break
            
            if not is_limited_driver:
                return

            logger.warning("[DOCKER] Default 'docker' driver does not support local cache isolation. Setting up compatible builder...")

            # 2. Try to use an existing managed builder
            builder_name = "noissues-managed-builder"
            use_result = subprocess.run(
                ["docker", "buildx", "use", builder_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if use_result.returncode == 0:
                logger.info(f"[DOCKER] Switched to existing builder: {builder_name}")
                return

            # 3. Create it if it doesn't exist
            logger.info(f"[DOCKER] Creating new 'docker-container' builder: {builder_name}")
            create_result = subprocess.run(
                ["docker", "buildx", "create", "--name", builder_name, "--driver", "docker-container", "--use"],
                capture_output=True,
                text=True,
                timeout=20
            )
            
            if create_result.returncode == 0:
                logger.info(f"[DOCKER] Successfully created and switched to {builder_name}")
                # Bootstrap to ensure it's ready
                subprocess.run(["docker", "buildx", "inspect", "--bootstrap"], timeout=60, capture_output=True)
                return
            else:
                error_msg = f"Failed to create compatible Docker builder: {create_result.stderr}"
                logger.error(f"[DOCKER] {error_msg}")
                raise RuntimeError(error_msg)

        except subprocess.TimeoutExpired:
            error_msg = "Timeout while configuring Docker Buildx builder"
            logger.error(f"[DOCKER] {error_msg}")
            raise RuntimeError(error_msg)
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            error_msg = f"Unexpected error during Docker builder setup: {e}"
            logger.error(f"[DOCKER] {error_msg}")
            raise RuntimeError(error_msg)

    def prune_buildkit_cache(self) -> bool:
        """
        Prune Docker BuildKit cache to recover from corruption.

        Acquires BOTH build lock and prune lock to pause all builds during prune.
        This prevents race conditions where builds run during cache cleanup.

        Returns:
            True if prune succeeded, False otherwise
        """
        # Acquire build lock first to pause all builds
        with _docker_build_lock:
            with _buildkit_prune_lock:
                try:
                    logger.warning("[DOCKER] Pruning BuildKit cache (all builds paused)...")
                    result = subprocess.run(
                        ["docker", "builder", "prune", "--all", "--force"],
                        capture_output=True,
                        text=True,
                        timeout=120
                    )
                    if result.returncode == 0:
                        logger.info("[DOCKER] BuildKit cache pruned successfully")
                        # Wait for Docker daemon to stabilize
                        time.sleep(2)
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
                         retry_on_cache_error: bool = True, cache_dir: Optional[str] = None) -> Dict:
        """
        Build a Dockerfile and return detailed results.
        
        Automatically detects BuildKit cache corruption and recovers by pruning
        the cache and retrying the build.

        Args:
            dockerfile_path: Path to Dockerfile
            context_path: Path to build context (usually repository root)
            image_name: Name to tag the built image
            retry_on_cache_error: If True, auto-recover from cache corruption
            cache_dir: Optional directory for local Buildx cache isolation

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

        if self.serialize_builds:
            with _docker_build_lock:
                return self._do_build(dockerfile_path, context_path, image_name, retry_on_cache_error, cache_dir=cache_dir)
        else:
            return self._do_build(dockerfile_path, context_path, image_name, retry_on_cache_error, cache_dir=cache_dir)

    def _do_build(self, dockerfile_path: str, context_path: str, image_name: str,
                  retry_on_cache_error: bool = True, _is_retry: bool = False,
                  cache_dir: Optional[str] = None) -> Dict:
        """Internal build method with retry logic for cache corruption."""
        start_time = time.time()

        try:
            if cache_dir:
                # Use buildx for local cache isolation
                cmd = ["docker", "buildx", "build", "--load"] # --load ensures image is available in regular docker images list
                
                # Create cache directory if it doesn't exist
                os.makedirs(cache_dir, exist_ok=True)
                
                # Always export cache to this directory
                cmd.extend(["--cache-to", f"type=local,dest={cache_dir},mode=max"])
                
                # ONLY import cache if it actually exists to avoid warnings/errors on first run
                # index.json is the indicator of a valid local cache export
                if os.path.exists(os.path.join(cache_dir, "index.json")):
                    cmd.extend(["--cache-from", f"type=local,src={cache_dir}"])
            else:
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

            # Execute build
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env
            )
            
            # Transient Network Error Retry Logic
            # Detects: apt/pip timeouts, temporary DNS issues, registry 503s
            if result.returncode != 0 and not _is_retry:
                err_lower = (result.stderr or "").lower() + (result.stdout or "").lower()
                transient_keywords = [
                    "temporary failure resolving", "name resolution temporarily failed",
                    "connection timed out", "readpoolerror", "net/http: request canceled",
                    "failed to fetch", "503 service unavailable", "504 gateway time-out",
                    "eai_again", "unable to connect"
                ]
                
                is_transient = any(k in err_lower for k in transient_keywords)
                
                if is_transient:
                    logger.warning("[DOCKER] Detected transient network error. Retrying build in 5s...")
                    time.sleep(5)
                    # Retry cleanly
                    return self._do_build(dockerfile_path, context_path, image_name, 
                                        retry_on_cache_error=retry_on_cache_error, _is_retry=True)

            duration = time.time() - start_time

            if result.returncode == 0:
                print(f"[DOCKER BUILD] Success! Duration: {duration:.2f}s")
                return {
                    "success": True,
                    "image_name": image_name,
                    "duration_seconds": duration
                }
            else:
                # Same error analysis as before...
                full_error = result.stderr or result.stdout or "Unknown error"
                
                # Check for cache corruption and auto-recover
                if retry_on_cache_error and not _is_retry and self._is_cache_corruption_error(full_error):
                    logger.warning("[DOCKER] Detected BuildKit cache corruption, attempting recovery...")
                    
                    # Also remove local cache directory if it exists as it might be the source
                    if cache_dir and os.path.exists(cache_dir):
                        try:
                            logger.warning(f"[DOCKER] Removing local cache directory: {cache_dir}")
                            shutil.rmtree(cache_dir)
                        except Exception as e:
                            logger.error(f"[DOCKER] Warning: Could not remove local cache: {e}")

                    if self.prune_buildkit_cache():
                        # Wait a moment for Docker to stabilize
                        time.sleep(2)
                        # Retry the build
                        return self._do_build(dockerfile_path, context_path, image_name, 
                                             retry_on_cache_error=False, _is_retry=True,
                                             cache_dir=cache_dir)
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

    def run_container(self, image_name: str, command: str, timeout: int = 20) -> Dict:
        """
        Run a command inside the container to verify functionality (Smoke Test).
        
        Args:
            image_name: Name of the Docker image to run
            command: Shell command to execute
            timeout: Max execution time in seconds (default 20s to prevent hangs)
            
        Returns:
            Dict w/ success, output, etc.
        """
        import subprocess
        
        # Verify image exists first
        inspect = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True
        )
        if inspect.returncode != 0:
            return {
                "success": False,
                "error": f"Image {image_name} not found locally.",
                "output": ""
            }

        print(f"[DOCKER RUN] Executing: {command} (timeout={timeout}s)")
        
        # Run container ephemerally (--rm)
        # We override entrypoint to ensure we can run our specific command
        cmd = [
            "docker", "run", "--rm", 
            "--entrypoint", "", # Reset entrypoint
            image_name, 
            "sh", "-c", command 
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            output = (result.stdout + "\n" + result.stderr).strip()
            return {
                "success": result.returncode == 0,
                "exit_code": result.returncode,
                "output": output,
                "timeout": False
            }
            
        except subprocess.TimeoutExpired as e:
            output = (e.stdout or "") + "\n" + (e.stderr or "")
            return {
                "success": False, 
                "error": "Execution timed out",
                "output": output.strip() + "\n[KILLED DUE TO TIMEOUT]",
                "timeout": True
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "timeout": False
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
