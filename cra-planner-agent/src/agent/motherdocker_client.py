import os
import time
import tarfile
import tempfile
import logging
import requests
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MotherDockerClient:
    """
    Client for motherdocker remote build service.

    Implements the same interface as DockerBuildTester for compatibility,
    but submits builds to a remote Kubernetes cluster running Kaniko.
    """

    def __init__(self, api_url: str = None, timeout: int = 600, **kwargs):
        """
        Initialize motherdocker client.

        Args:
            api_url: Base URL of motherdocker API (default: from env MOTHERDOCKER_API_URL or localhost)
            timeout: Maximum time in seconds to wait for build completion (default: 10 minutes)
            **kwargs: Ignored (for compatibility with DockerBuildTester interface)
        """
        self.api_url = (api_url or os.getenv("MOTHERDOCKER_API_URL", "http://localhost:8000")).rstrip("/")
        self.timeout = timeout
        self.docker_available = True  # Always True for remote builds

    def _check_api_health(self) -> bool:
        """Check if motherdocker API is accessible."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"[MOTHERDOCKER] API health check failed: {e}")
            return False

    def check_infrastructure_health(self) -> tuple[bool, str]:
        """Verify motherdocker API is healthy."""
        if self._check_api_health():
            return True, "Infrastructure health check passed"
        return False, f"Motherdocker API at {self.api_url} is not accessible"

    def _load_dockerignore(self, context_path: str) -> set:
        """
        Load .dockerignore patterns from context directory.

        Args:
            context_path: Build context directory

        Returns:
            Set of patterns to exclude
        """
        dockerignore_path = os.path.join(context_path, ".dockerignore")
        patterns = set()

        if os.path.exists(dockerignore_path):
            try:
                with open(dockerignore_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        # Skip comments and empty lines
                        if line and not line.startswith('#'):
                            patterns.add(line)
            except Exception as e:
                logger.warning(f"[MOTHERDOCKER] Failed to read .dockerignore: {e}")

        # Always exclude common patterns
        patterns.update(['.git', '__pycache__', '*.pyc', '.DS_Store'])

        return patterns

    def _should_exclude(self, path: str, patterns: set) -> bool:
        """
        Check if a path should be excluded based on .dockerignore patterns.

        Args:
            path: Relative path to check
            patterns: Set of patterns from .dockerignore

        Returns:
            True if path should be excluded
        """
        path = path.lstrip('./')

        for pattern in patterns:
            # Simple pattern matching (support * wildcard)
            if pattern.endswith('/'):
                # Directory pattern
                if path.startswith(pattern) or path == pattern.rstrip('/'):
                    return True
            elif '*' in pattern:
                # Wildcard pattern (simple implementation)
                import fnmatch
                if fnmatch.fnmatch(path, pattern):
                    return True
            else:
                # Exact match
                if path == pattern or path.startswith(pattern + '/'):
                    return True

        return False

    def _create_tarball(self, dockerfile_path: str, context_path: str) -> str:
        """
        Create tarball from Dockerfile and build context.

        Respects .dockerignore patterns to avoid uploading unnecessary files.

        Args:
            dockerfile_path: Path to Dockerfile
            context_path: Path to build context directory

        Returns:
            Path to created tarball (in temp directory)
        """
        # Load .dockerignore patterns
        exclude_patterns = self._load_dockerignore(context_path)

        # Create temporary tarball
        temp_tar = tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False)
        temp_tar_path = temp_tar.name
        temp_tar.close()

        try:
            with tarfile.open(temp_tar_path, 'w:gz') as tar:
                # Add Dockerfile (always at root of tarball)
                dockerfile_arcname = os.path.basename(dockerfile_path)
                tar.add(dockerfile_path, arcname=dockerfile_arcname)
                logger.info(f"[MOTHERDOCKER] Added {dockerfile_arcname} to tarball")

                # Add context files (respecting .dockerignore)
                context_root = Path(context_path).resolve()
                added_count = 0
                excluded_count = 0

                for root, dirs, files in os.walk(context_path):
                    # Calculate relative path
                    rel_root = os.path.relpath(root, context_path)

                    # Filter directories based on .dockerignore
                    dirs_to_remove = []
                    for d in dirs:
                        rel_dir = os.path.normpath(os.path.join(rel_root, d))
                        if self._should_exclude(rel_dir, exclude_patterns):
                            dirs_to_remove.append(d)
                            excluded_count += 1

                    for d in dirs_to_remove:
                        dirs.remove(d)

                    # Add files
                    for file in files:
                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, context_path)

                        # Skip if matches .dockerignore or if it's the Dockerfile (already added)
                        if self._should_exclude(rel_path, exclude_patterns):
                            excluded_count += 1
                            continue

                        if os.path.samefile(file_path, dockerfile_path):
                            continue

                        tar.add(file_path, arcname=rel_path)
                        added_count += 1

                logger.info(f"[MOTHERDOCKER] Created tarball: {added_count} files added, {excluded_count} excluded")

        except Exception as e:
            # Cleanup on failure
            if os.path.exists(temp_tar_path):
                os.unlink(temp_tar_path)
            raise RuntimeError(f"Failed to create tarball: {e}")

        return temp_tar_path

    def _poll_build(self, build_id: str, start_time: float) -> Dict:
        """
        Poll motherdocker API for build status.

        Args:
            build_id: Build identifier from submission
            start_time: Timestamp when build was submitted

        Returns:
            Build result dict compatible with DockerBuildTester format
        """
        poll_interval = 5  # seconds

        while (time.time() - start_time) < self.timeout:
            try:
                response = requests.get(f"{self.api_url}/build/{build_id}", timeout=10)

                if response.status_code == 404:
                    return {
                        "success": False,
                        "stage": "BUILD_NOT_FOUND",
                        "error_message": f"Build {build_id} not found on server",
                        "exit_code": -1,
                        "duration_seconds": time.time() - start_time
                    }

                if response.status_code != 200:
                    logger.error(f"[MOTHERDOCKER] Status check failed: {response.status_code}")
                    time.sleep(poll_interval)
                    continue

                data = response.json()
                status = data.get("status", "").upper()  # Normalize to uppercase

                # Terminal states
                if status == "SUCCESS":
                    duration = time.time() - start_time
                    logger.info(f"[MOTHERDOCKER] Build {build_id} succeeded! Duration: {duration:.2f}s")
                    return {
                        "success": True,
                        "image_name": data.get("original_image_tag", "unknown"),
                        "duration_seconds": duration,
                        "build_id": build_id
                    }

                elif status == "BUILD_FAILED":
                    duration = time.time() - start_time
                    error_msg = data.get("error", data.get("error_message", "Build failed"))
                    logger.error(f"[MOTHERDOCKER] Build {build_id} failed: {error_msg}")
                    return {
                        "success": False,
                        "stage": "BUILD_FAILED",
                        "error_message": error_msg,
                        "exit_code": 1,
                        "duration_seconds": duration
                    }

                elif status == "TEST_FAILED":
                    duration = time.time() - start_time
                    smoke_result = data.get("smoke_test_result", {})
                    error_msg = smoke_result.get("error", "Smoke test failed")
                    logger.error(f"[MOTHERDOCKER] Build {build_id} test failed: {error_msg}")
                    return {
                        "success": False,
                        "stage": "TEST_FAILED",
                        "error_message": f"Build succeeded but smoke test failed: {error_msg}",
                        "smoke_test_output": smoke_result.get("output"),
                        "exit_code": smoke_result.get("exit_code", 1),
                        "duration_seconds": duration
                    }

                elif status == "FAILED":
                    duration = time.time() - start_time
                    error_msg = data.get("error", data.get("error_message", "Unknown failure"))
                    logger.error(f"[MOTHERDOCKER] Build {build_id} failed: {error_msg}")
                    return {
                        "success": False,
                        "stage": "FAILED",
                        "error_message": error_msg,
                        "exit_code": 1,
                        "duration_seconds": duration
                    }

                elif status == "TIMEOUT":
                    duration = time.time() - start_time
                    error_msg = data.get("error", data.get("error_message", "Build timed out on server"))
                    logger.error(f"[MOTHERDOCKER] Build {build_id} timed out: {error_msg}")
                    return {
                        "success": False,
                        "stage": "BUILD_TIMEOUT",
                        "error_type": "TIMEOUT",
                        "error_message": error_msg,
                        "exit_code": -1,
                        "duration_seconds": duration
                    }

                # Non-terminal states: keep polling
                elif status in ["QUEUED", "BUILDING", "TESTING"]:
                    logger.info(f"[MOTHERDOCKER] Build {build_id} status: {status}")
                    time.sleep(poll_interval)
                    continue

                else:
                    logger.warning(f"[MOTHERDOCKER] Unknown status: {status}")
                    time.sleep(poll_interval)
                    continue

            except requests.RequestException as e:
                logger.error(f"[MOTHERDOCKER] Poll error: {e}")
                time.sleep(poll_interval)
                continue
            except Exception as e:
                logger.exception(f"[MOTHERDOCKER] Unexpected poll error: {e}")
                time.sleep(poll_interval)
                continue

        # Timeout
        duration = time.time() - start_time
        return {
            "success": False,
            "stage": "BUILD_TIMEOUT",
            "error_type": "TIMEOUT",
            "error_message": f"Build polling exceeded timeout of {self.timeout} seconds",
            "exit_code": -1,
            "duration_seconds": duration
        }

    def build_dockerfile(self, dockerfile_path: str, context_path: str, image_name: str, **kwargs) -> Dict:
        """
        Build a Dockerfile using motherdocker remote build service.

        Args:
            dockerfile_path: Path to Dockerfile
            context_path: Path to build context (repository root)
            image_name: Name to tag the built image
            **kwargs: Ignored (for compatibility)

        Returns:
            Dictionary with build results (compatible with DockerBuildTester format)
        """
        # Check if Dockerfile exists
        if not os.path.exists(dockerfile_path):
            return {
                "success": False,
                "stage": "DOCKERFILE_CHECK",
                "error_type": "DOCKERFILE_NOT_FOUND",
                "error_message": f"Dockerfile not found at {dockerfile_path}",
                "exit_code": -1,
                "duration_seconds": 0
            }

        # Check API health
        if not self._check_api_health():
            return {
                "success": False,
                "stage": "API_CHECK",
                "error_type": "API_NOT_AVAILABLE",
                "error_message": f"Motherdocker API at {self.api_url} is not accessible",
                "exit_code": -1,
                "duration_seconds": 0
            }

        start_time = time.time()
        tarball_path = None

        try:
            # Create tarball
            logger.info(f"[MOTHERDOCKER] Creating tarball from {context_path}")
            tarball_path = self._create_tarball(dockerfile_path, context_path)
            tarball_size_mb = os.path.getsize(tarball_path) / (1024 * 1024)
            logger.info(f"[MOTHERDOCKER] Tarball size: {tarball_size_mb:.2f} MB")

            # Calculate relative Dockerfile path
            dockerfile_in_tarball = os.path.basename(dockerfile_path)

            # Submit build
            logger.info(f"[MOTHERDOCKER] Submitting build to {self.api_url}/build/tarball")

            with open(tarball_path, 'rb') as f:
                files = {'tarball': (os.path.basename(tarball_path), f, 'application/gzip')}
                data = {
                    'target_image_tag': image_name,
                    'dockerfile_path': dockerfile_in_tarball
                }

                response = requests.post(
                    f"{self.api_url}/build/tarball",
                    files=files,
                    data=data,
                    timeout=60  # Upload timeout
                )

            if response.status_code != 201:
                error_msg = f"Build submission failed: {response.status_code}"
                try:
                    error_detail = response.json().get("detail", response.text)
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text[:200]}"

                return {
                    "success": False,
                    "stage": "SUBMIT_BUILD",
                    "error_message": error_msg,
                    "exit_code": response.status_code,
                    "duration_seconds": time.time() - start_time
                }

            # Extract build ID
            result = response.json()
            build_id = result.get("build_id")
            logger.info(f"[MOTHERDOCKER] Build submitted successfully. Build ID: {build_id}")

            # Poll for completion
            return self._poll_build(build_id, start_time)

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"[MOTHERDOCKER] Build error: {e}")
            return {
                "success": False,
                "stage": "BUILD_EXCEPTION",
                "error_type": type(e).__name__,
                "error_message": str(e),
                "exit_code": -1,
                "duration_seconds": duration
            }
        finally:
            # Cleanup tarball
            if tarball_path and os.path.exists(tarball_path):
                try:
                    os.unlink(tarball_path)
                    logger.info(f"[MOTHERDOCKER] Cleaned up tarball")
                except Exception as e:
                    logger.warning(f"[MOTHERDOCKER] Failed to cleanup tarball: {e}")

    def run_container(self, image_name: str, command: str, timeout: int = 20) -> Dict:
        """
        Stub method for compatibility with DockerBuildTester interface.

        Motherdocker handles smoke tests remotely, so this is not used.

        Returns:
            Dict indicating the method is not supported
        """
        return {
            "success": False,
            "error": "run_container not supported in motherdocker mode (use smoke_test_command in build request)",
            "output": ""
        }

    def cleanup_image(self, image_name: str) -> bool:
        """
        Stub method for compatibility with DockerBuildTester interface.

        Motherdocker handles image cleanup automatically for ephemeral images.

        Returns:
            True (no-op)
        """
        return True

    def prune_buildkit_cache(self) -> bool:
        """
        Stub method for compatibility with DockerBuildTester interface.

        Not applicable for remote builds.

        Returns:
            True (no-op)
        """
        return True
