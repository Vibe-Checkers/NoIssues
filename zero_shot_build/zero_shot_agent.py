#!/usr/bin/env python3
"""
Zero-Shot Build Agent
Generated Dockerfiles based ONLY on README.md and a System Prompt.
Then attempts to build the image immediately.
"""

import os
import sys
import subprocess
import shutil
import time
import stat
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
root_dir = Path(__file__).parent.parent
env_path = root_dir / "cra-planner-agent" / ".env"

if not env_path.exists():
    # Fallback to current directory or parent
    env_path = root_dir / ".env"

if env_path.exists():
    load_dotenv(env_path)
else:
    print(f"[WARNING] .env not found at {env_path}")

def setup_llm():
    """Initialize Azure OpenAI LLM."""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not all([api_key, endpoint, deployment]):
        print("Error: Missing Azure OpenAI credentials in .env file")
        sys.exit(1)

    return AzureChatOpenAI(
        azure_deployment=deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version
    )


import traceback
from typing import Dict, Tuple, Optional

# Global log file path
LOG_FILE_PATH = None

def logger(message, title=None):
    """Log to console and file."""
    if title:
        formatted_msg = f"\n{'='*20} {title} {'='*20}\n{message}\n{'='*50}"
    else:
        formatted_msg = message

    print(formatted_msg)
    
    if LOG_FILE_PATH:
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(formatted_msg + "\n")

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

            logger(f"[DOCKER BUILD] Running: {' '.join(cmd)}")

            # Run without text=True to handle decoding manually
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=False,  # Capture raw bytes
                timeout=self.timeout
            )

            # Manual decoding with error replacement
            stdout_str = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
            stderr_str = result.stderr.decode('utf-8', errors='replace') if result.stderr else None

            duration = time.time() - start_time
            
            # Log the full build output to file
            if LOG_FILE_PATH:
                 with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                    f.write(f"\n{'='*20} DOCKER BUILD OUTPUT (Combined) {'='*20}\n")
                    f.write(stdout_str)
                    if stderr_str:
                         f.write("\n--- STDERR ---\n")
                         f.write(stderr_str)
                    f.write(f"\n{'='*50}\n")

            if result.returncode == 0:
                return {
                    "success": True,
                    "stage": "BUILD_COMPLETE",
                    "error_type": None,
                    "error_message": None,
                    "exit_code": 0,
                    "duration_seconds": duration,
                    "stdout": stdout_str[-2000:],  # Last 2000 chars
                    "stderr": stderr_str[-2000:] if stderr_str else None
                }
            else:
                # Parse error to determine stage
                full_error = (stderr_str or "") + (stdout_str or "")
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
                    "stdout": stdout_str[-2000:] if stdout_str else "",
                    "stderr": stderr_str[-2000:] if stderr_str else None
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

        # ===================================================================
        # Simple Stage Detection - High Level Only
        # ===================================================================

        # 1. Docker Daemon Issues
        if any(x in error_lower for x in ["cannot connect to the docker daemon", "is the docker daemon running", "docker: not found", "docker: command not found"]):
            return "DOCKER_DAEMON", failed_step or "Docker daemon not accessible"

        # 2. Base Image Pull Issues (FROM command)
        if any(x in error_lower for x in ["failed to resolve", "manifest unknown", "pull access denied", "image not found"]):
            return "IMAGE_PULL", failed_step or "Failed to pull base image"

        # 3. Dockerfile Syntax
        if any(x in error_lower for x in ["dockerfile parse error", "unknown instruction"]):
            return "DOCKERFILE_SYNTAX", failed_step or "Dockerfile syntax error"

        # 4. File Copy/Add (COPY/ADD commands)
        if any(x in error_lower for x in ["copy failed", "add failed"]) or ("stat" in error_lower and "no such file" in error_lower):
            return "FILE_COPY", failed_step or "File copy/add failed"

        # 5. Dependency Installation (RUN pip/npm/go/cargo install commands)
        if any(x in error_lower for x in ["pip install", "pip3 install", "npm install", "yarn install", "go mod download", "go get", "cargo build"]):
            return "DEPENDENCY_INSTALL", failed_step or "Dependency installation failed"

        # 6. Build/Compilation (RUN build commands)
        if any(x in error_lower for x in ["compilation error", "build error", "webpack", "tsc"]):
            return "BUILD_COMPILE", failed_step or "Build/compilation failed"

        # 7. Runtime Execution (CMD/ENTRYPOINT)
        if any(x in error_lower for x in ["command not found", "exec format error"]):
            return "RUNTIME_EXEC", failed_step or "Runtime execution failed"

        # 8. Permission/User Issues
        if "permission denied" in error_lower or "useradd" in error_lower:
            return "PERMISSION", failed_step or "Permission/user management error"

        # 9. Network Issues
        if any(x in error_lower for x in ["connection refused", "connection timeout", "network unreachable"]):
            return "NETWORK", failed_step or "Network connection error"

        # 10. Storage Issues
        if any(x in error_lower for x in ["no space left", "disk full", "quota exceeded"]):
            return "STORAGE", failed_step or "Disk space error"

        # Fallback: Return the extracted step or unknown
        return "UNKNOWN", failed_step or "Unknown error - check full log"

    def _extract_failed_docker_step(self, error_output: str) -> str:
        """
        Extract the specific Docker RUN/COPY/FROM command that failed.
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
            print(f"[CLEANUP] Removing Docker image {image_name}...")
            subprocess.run(
                ["docker", "rmi", "-f", image_name],
                capture_output=True,
                timeout=30
            )
            return True
        except Exception:
            return False

def clone_repository(repo_url, target_dir="./temp_repo"):
    """Clone repository to a temporary directory."""
    logger(f"\n[CLONE] Cloning {repo_url}...")
    
    if os.path.exists(target_dir):
        def remove_readonly(func, path, excinfo):
            os.chmod(path, stat.S_IWRITE)
            func(path)
            
        shutil.rmtree(target_dir, onexc=remove_readonly)
        
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target_dir],
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        logger(f"[OK] Cloned to {target_dir}")
        return Path(target_dir)
    except subprocess.CalledProcessError as e:
        logger(f"[ERROR] Failed to clone repository: {e.stderr}")
        sys.exit(1)

def get_readme_content(repo_path):
    """Find and read the README file."""
    for file in os.listdir(repo_path):
        if file.lower().startswith("readme"):
            logger(f"[READ] Found README: {file}")
            try:
                # Use errors='replace' to avoid crashing on special characters
                with open(repo_path / file, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            except Exception as e:
                logger(f"[ERROR] Could not read README: {e}")
                sys.exit(1)
    
    logger("[WARNING] No README found in repository.")
    return "No README file found in this repository."

def generate_dockerfile(llm, readme_content):
    """Generate Dockerfile using LLM."""
    logger("\n[GENERATE] Generating Dockerfile using Zero-Shot prompting...")
    
    system_prompt = """You are an expert DevOps engineer. Your task is to create a production-ready Dockerfile for a project based ONLY on its README content.

RULES:
1. USE BEST PRACTICES: Multi-stage builds, specific versions, non-root users if possible.
2. INFER DEPENDENCIES: Look for language (Python, Node, Go, etc.) and package managers (pip, npm, go mod, etc.) in the text.
3. OUTPUT FORMAT: Return ONLY the Dockerfile content. No markdown code blocks, no explanations, no "Here is the file". Just the raw Dockerfile content.
4. If the README is missing or empty, try to create a generic Dockerfile for the most likely language if detectable, or a safe default.
"""

    user_message = f"Here is the README content of the project:\n\n{readme_content}\n\nGenerate the Dockerfile now."
    
    if LOG_FILE_PATH:
        # Log purely to file, no print for full prompts
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*20} SYSTEM PROMPT {'='*20}\n{system_prompt}\n{'='*50}\n")
            f.write(f"\n{'='*20} USER MESSAGE (Prompt) {'='*20}\n{user_message}\n{'='*50}\n")

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ])
        
        content = response.content.strip()
        
        if LOG_FILE_PATH:
             with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*20} LLM RESPONSE {'='*20}\n{content}\n{'='*50}\n")

        # Strip markdown code blocks if the LLM ignores the rule
        if content.startswith("```"):
            lines = content.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
            
        logger(f"[OK] Dockerfile generated ({len(content)} chars)")
        return content
    except Exception as e:
        logger(f"[ERROR] LLM generation failed: {e}")
        sys.exit(1)

def main():
    global LOG_FILE_PATH

    if len(sys.argv) < 2:
        print("Usage: python zero_shot_agent.py <repo_url>")
        sys.exit(1)
        
    repo_url = sys.argv[1]
    repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    
    # Create working directory
    script_dir = Path(__file__).parent
    work_dir = script_dir / "temp" / repo_name
    work_dir.mkdir(parents=True, exist_ok=True)
    
    # Init log file
    LOG_FILE_PATH = script_dir / f"{repo_name}_agent.log"
    # Clear previous log
    with open(LOG_FILE_PATH, 'w', encoding='utf-8') as f:
        f.write(f"Zero-Shot Build Agent Log - {repo_name}\n")
        f.write(f"Date: {time.ctime()}\n")

    logger(f"[INFO] Logging to {LOG_FILE_PATH}")
    
    # Execution Flow
    repo_path = clone_repository(repo_url, target_dir=work_dir)
    readme_content = get_readme_content(repo_path)
    
    llm = setup_llm()
    dockerfile_content = generate_dockerfile(llm, readme_content)
    
    # Save Dockerfile BEFORE build
    with open(script_dir / f"{repo_name}_Dockerfile", 'w', encoding='utf-8') as f:
        f.write(dockerfile_content)
    logger(f"\n[INFO] Saved Dockerfile to {repo_name}_Dockerfile")

    # TEST DOCKER BUILD using ported logic
    image_tag = f"zero-shot-{repo_name.lower()}:latest"
    tester = DockerBuildTester(timeout=600)
    
    dockerfile_path = script_dir / f"{repo_name}_Dockerfile"
    # Note: Using repo_path as context, but referencing the saved Dockerfile
    # Ideally we should move Dockerfile to repo_path or point to it
    # The build_dockerfile argument expects a path to the file
    
    logger("\n[BUILD] Attempting to build Docker image...")
    build_result = tester.build_dockerfile(str(dockerfile_path), str(repo_path), image_tag)
    
    if build_result["success"]:
        logger(f"[SUCCESS] Docker build completed in {build_result['duration_seconds']:.2f}s")
        logger(f"[INFO] Image tag: {image_tag}")
        tester.cleanup_image(image_tag)
    else:
        logger(f"[FAILURE] Docker build failed (Exit Code: {build_result['exit_code']})")
        logger(f"  Stage: {build_result.get('stage', 'UNKNOWN')}")
        logger(f"  Command: {build_result.get('failed_command', 'Unknown')}")
        logger(f"  Error Snippet: {build_result.get('error_snippet', 'No snippet')}")
        
        # Save error summary
        error_summary_file = script_dir / f"{repo_name}_docker_build_error_summary.txt"
        with open(error_summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DOCKER BUILD ERROR SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Repository: {repo_name}\n")
            f.write(f"URL: {repo_url}\n")
            f.write(f"Timestamp: {time.ctime()}\n\n")
            f.write(f"Failure Stage: {build_result.get('stage', 'UNKNOWN')}\n")
            f.write(f"Failed Docker Command: {build_result.get('failed_command', 'Unknown')}\n")
            f.write(f"Exit Code: {build_result.get('exit_code', -1)}\n\n")
            if build_result.get('error_snippet'):
                f.write(f"Error Snippet (last error line):\n{'-'*80}\n")
                f.write(f"{build_result['error_snippet']}\n")
                f.write(f"{'-'*80}\n\n")
            f.write(f"See full Docker output in: {repo_name}_agent.log\n")
        logger(f"[INFO] Saved error summary to {error_summary_file.name}")

    # Cleanup
    if os.getenv("KEEP_TEMP", "false").lower() != "true":
         try:
            def remove_readonly(func, path, excinfo):
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(work_dir, onexc=remove_readonly)
            logger("[CLEANUP] Temporary directory removed")
         except Exception as e:
            logger(f"[WARNING] Cleanup failed: {e}")

if __name__ == "__main__":
    main()
