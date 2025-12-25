#!/usr/bin/env python3
"""
Run Agent Script
Full workflow script that clones GitHub repositories and performs analysis.
Use this for complete repository analysis tasks.
"""

import os
import re
from collections import Counter
import sys
import subprocess
import shutil
import time
import json
import tempfile
import io
from pathlib import Path
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
import threading
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in the project root (parent of src/)
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

from planner_agent import create_planner_agent, REPORT_DIRECTORY


def clone_repository(repo_url: str, target_dir: str = "./temp", auto_remove: bool = True, max_retries: int = 3) -> str:
    """
    Clone a GitHub repository to a local directory with retry logic and fallback strategies.

    Args:
        repo_url: GitHub repository URL
        target_dir: Directory to clone repositories into
        auto_remove: If True, automatically remove existing repo and re-clone (default: True for automation)
        max_retries: Maximum number of retry attempts (default: 3)

    Returns:
        Path to the cloned repository
        
    Raises:
        SystemExit: If cloning fails after all retries
    """
    print(f"\n[CLONE] Cloning repository: {repo_url}")

    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Extract repo name from URL
    repo_name = repo_url.rstrip('/').split('/')[-1].replace('.git', '')
    clone_path = os.path.join(target_dir, repo_name)

    # Remove existing clone if it exists
    if os.path.exists(clone_path):
        if auto_remove:
            print(f"[AUTO-REMOVE] Repository already exists at {clone_path}, removing...")
            shutil.rmtree(clone_path)
            print(f"[DELETED] Existing repository removed")
        else:
            print(f"[WARNING] Repository already exists at {clone_path}")
            response = input("Remove and re-clone? (y/N): ").strip().lower()
            if response == 'y':
                print(f"[DELETE] Removing existing repository...")
                shutil.rmtree(clone_path)
            else:
                print(f"[OK] Using existing repository at {clone_path}")
                return clone_path

    # Clone strategies to try in order
    strategies = [
        {
            "name": "Standard clone",
            "cmd": ["git", "clone", repo_url, clone_path],
            "timeout": 1800  # 30 minutes for full clone
        },
        {
            "name": "Shallow clone (depth=1)",
            "cmd": ["git", "clone", "--depth", "1", repo_url, clone_path],
            "timeout": 900  # 15 minutes
        },
        {
            "name": "HTTP/1.1 clone",
            "cmd": ["git", "-c", "http.version=HTTP/1.1", "clone", "--depth", "1", repo_url, clone_path],
            "timeout": 900  # 15 minutes
        },
        {
            "name": "Single-threaded clone",
            "cmd": ["git", "clone", "--depth", "1", "--single-branch", repo_url, clone_path],
            "timeout": 900  # 15 minutes
        }
    ]

    last_error = None
    
    for attempt in range(max_retries):
        # Try each strategy
        for strategy_idx, strategy in enumerate(strategies):
            # Clean up partial clone if exists
            if os.path.exists(clone_path):
                shutil.rmtree(clone_path)
            
            strategy_name = strategy["name"]
            if attempt > 0:
                print(f"[RETRY {attempt + 1}/{max_retries}] Trying {strategy_name}...")
            else:
                print(f"[CLONING] Trying {strategy_name} to {clone_path}...")
            
            try:
                result = subprocess.run(
                    strategy["cmd"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=strategy["timeout"]
                )
                print(f"[SUCCESS] Repository cloned successfully using {strategy_name}!")
                return clone_path
                
            except subprocess.TimeoutExpired:
                last_error = f"Timeout after {strategy['timeout']}s using {strategy_name}"
                print(f"[WARNING] {last_error}")
                continue
                
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.strip() if e.stderr else str(e)
                last_error = f"{strategy_name} failed: {error_msg}"
                
                # Check for specific errors
                if "HTTP/2 stream" in error_msg or "curl 92" in error_msg:
                    print(f"[WARNING] HTTP/2 stream error detected, will try HTTP/1.1...")
                elif "early EOF" in error_msg or "fetch-pack" in error_msg:
                    print(f"[WARNING] Network interruption detected, will retry...")
                else:
                    print(f"[WARNING] {last_error}")
                
                continue
        
        # Wait before next retry attempt
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            print(f"[WAIT] Waiting {wait_time}s before next retry attempt...")
            time.sleep(wait_time)
    
    # All attempts failed
    print(f"[ERROR] Failed to clone repository after {max_retries} attempts with all strategies")
    print(f"[ERROR] Last error: {last_error}")
    sys.exit(1)


def save_analysis_reports(
    repo_name: str,
    repo_url: str,
    repo_path: str,
    final_dockerfile: str,
    tool_usage: dict,
    total_tokens: dict,
    duration_seconds: float,
    callback_handler,
    log_file_path: Path = None,
    report_dir: Path = None,
    final_dockerignore: str = None
):
    """
    Save all analysis reports, metrics, and performance data to a structured folder.
    
    Args:
        repo_name: Name of the repository
        repo_url: URL of the repository
        repo_path: Path to the cloned repository
        final_dockerfile: Final Dockerfile content
        tool_usage: Dictionary of tool usage statistics
        total_tokens: Dictionary of token usage
        duration_seconds: Analysis duration in seconds
        callback_handler: Callback handler with token usage
        log_file_path: Optional path to log file
        report_dir: Optional existing report directory (if None, creates new one)
        final_dockerignore: Optional .dockerignore content
    """
    # Use provided report_dir or create new one
    if report_dir is None:
        # Create analysis_reports directory in the root (parent of cra-planner-agent)
        script_dir = Path(__file__).parent
        root_dir = script_dir.parent
        reports_base_dir = root_dir / "analysis_reports"
        reports_base_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this analysis run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = reports_base_dir / f"{repo_name}_{timestamp}"
        report_dir.mkdir(exist_ok=True)
    else:
        # Ensure it's a Path object
        report_dir = Path(report_dir)
        report_dir.mkdir(exist_ok=True)
        # Extract timestamp from report_dir name if possible, otherwise create new one
        # Report dir format: {repo_name}_{timestamp}
        dir_name = report_dir.name
        if '_' in dir_name:
            # Try to extract timestamp (format: YYYYMMDD_HHMMSS)
            parts = dir_name.rsplit('_', 1)
            if len(parts) == 2 and len(parts[1]) == 15:  # timestamp format check
                timestamp = parts[1]
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*70}")
    print("Saving Analysis Reports")
    print('='*70)
    print(f"Report directory: {report_dir.absolute()}")
    
    # 1. Save Dockerfile
    if final_dockerfile:
        dockerfile_path = report_dir / "Dockerfile"
        try:
            # Clean up the Dockerfile content - remove any markdown formatting or explanations
            dockerfile_content = final_dockerfile.strip()
            
            # Remove markdown code blocks if present
            if dockerfile_content.startswith("```"):
                # Find the end of the code block
                lines = dockerfile_content.split('\n')
                start_idx = 0
                end_idx = len(lines)
                
                # Skip opening ```dockerfile or ```
                for i, line in enumerate(lines):
                    if line.strip().startswith("```"):
                        start_idx = i + 1
                        break
                
                # Find closing ```
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() == "```":
                        end_idx = i
                        break
                
                dockerfile_content = '\n'.join(lines[start_idx:end_idx]).strip()
            
            # Write the Dockerfile
            with open(dockerfile_path, 'w', encoding='utf-8') as f:
                f.write(f"# Dockerfile for {repo_name}\n")
                f.write(f"# Generated by Planner Agent\n")
                f.write(f"# Repository: {repo_url}\n")
                f.write(f"# Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(dockerfile_content)
            print(f"[OK] Dockerfile saved: {dockerfile_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to save Dockerfile: {e}")

    # 1.5 Save .dockerignore
    if final_dockerignore:
        dockerignore_path = report_dir / ".dockerignore"
        try:
            # Clean up content
            dockerignore_content = final_dockerignore.strip()
            
            # Remove markdown code blocks if present
            if dockerignore_content.startswith("```"):
                lines = dockerignore_content.split('\n')
                start_idx = 0
                end_idx = len(lines)
                for i, line in enumerate(lines):
                    if line.strip().startswith("```"):
                        start_idx = i + 1
                        break
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip() == "```":
                        end_idx = i
                        break
                dockerignore_content = '\n'.join(lines[start_idx:end_idx]).strip()
            
            with open(dockerignore_path, 'w', encoding='utf-8') as f:
                f.write(dockerignore_content)
            print(f"[OK] .dockerignore saved: {dockerignore_path.name}")
        except Exception as e:
            print(f"[ERROR] Failed to save .dockerignore: {e}")
    
    # 2. Collect all metrics
    metrics = {
        "repository": {
            "name": repo_name,
            "url": repo_url,
            "local_path": str(repo_path)
        },
        "analysis": {
            "timestamp": timestamp,
            "date": datetime.now().isoformat(),
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_seconds / 60,
            "duration_formatted": f"{int(duration_seconds // 60)} min {int(duration_seconds % 60)} sec" if duration_seconds >= 60 else f"{duration_seconds:.2f} seconds"
        },
        "token_usage": {},
        "tool_usage": {}
    }
    
    # Get token usage from callback handler or fallback
    if callback_handler and callback_handler.token_usage["total"] > 0:
        metrics["token_usage"] = callback_handler.token_usage.copy()
    elif total_tokens["total"] > 0:
        metrics["token_usage"] = total_tokens.copy()
    
    # Add tool usage statistics
    if tool_usage:
        total_tool_calls = sum(tool_usage.values())
        metrics["tool_usage"] = {
            "tools": {tool: {"calls": count, "percentage": (count / total_tool_calls * 100) if total_tool_calls > 0 else 0} 
                     for tool, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)},
            "total_calls": total_tool_calls,
            "unique_tools": len(tool_usage)
        }
    
    # 3. Save performance report as JSON
    performance_json_file = report_dir / "performance_report.json"
    try:
        with open(performance_json_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"[OK] Performance report (JSON) saved: {performance_json_file.name}")
    except Exception as e:
        print(f"[ERROR] Failed to save performance report (JSON): {e}")
    
    # 4. Save logs file if provided
    if log_file_path and log_file_path.exists():
        logs_file = report_dir / "logs.txt"
        try:
            shutil.copy2(log_file_path, logs_file)
            print(f"[OK] Agent logs saved: {logs_file.name}")
        except Exception as e:
            print(f"[ERROR] Failed to save logs: {e}")
    
    # 5. Save human-readable performance report
    performance_txt_file = report_dir / "performance_report.txt"
    try:
        with open(performance_txt_file, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("ANALYSIS PERFORMANCE REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("Repository Information\n")
            f.write("-"*70 + "\n")
            f.write(f"Name: {repo_name}\n")
            f.write(f"URL: {repo_url}\n")
            f.write(f"Local Path: {repo_path}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Analysis Duration\n")
            f.write("-"*70 + "\n")
            if duration_seconds >= 60:
                f.write(f"Total time: {int(duration_seconds // 60)} min {int(duration_seconds % 60)} sec ({duration_seconds:.2f} seconds)\n")
            else:
                f.write(f"Total time: {duration_seconds:.2f} seconds\n")
            f.write("\n")
            
            f.write("Token Usage Summary\n")
            f.write("-"*70 + "\n")
            if metrics["token_usage"]:
                usage = metrics["token_usage"]
                f.write(f"Input tokens:  {usage.get('input', 0):,}\n")
                f.write(f"Output tokens: {usage.get('output', 0):,}\n")
                f.write(f"Total tokens:  {usage.get('total', 0):,}\n")
            else:
                f.write("Token usage information not available\n")
            f.write("\n")
            
            f.write("Tool Usage Report\n")
            f.write("-"*70 + "\n")
            if metrics["tool_usage"] and metrics["tool_usage"].get("tools"):
                for tool_name, tool_data in metrics["tool_usage"]["tools"].items():
                    count = tool_data["calls"]
                    percentage = tool_data["percentage"]
                    f.write(f"{tool_name:30s} : {count:3d} calls ({percentage:5.1f}%)\n")
                f.write("-"*70 + "\n")
                f.write(f"{'Total tool calls':30s} : {metrics['tool_usage']['total_calls']:3d}\n")
                f.write(f"{'Unique tools used':30s} : {metrics['tool_usage']['unique_tools']:3d}\n")
            else:
                f.write("Tool usage information not available\n")
            f.write("\n")
            
            f.write("="*70 + "\n")
        print(f"[OK] Performance report (TXT) saved: {performance_txt_file.name}")
    except Exception as e:
        print(f"[ERROR] Failed to save performance report (TXT): {e}")
    
    print('='*70)
    print(f"\n[SUCCESS] All reports saved to: {report_dir.absolute()}")
    print('='*70)
    
    return report_dir


class ThreadAwareStdout:
    """
    Thread-safe stdout wrapper that writes to original stdout 
    and a thread-local log file if registered.
    """
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.thread_files = {}
        self.lock = threading.Lock()
    
    def register(self, f):
        with self.lock:
            self.thread_files[threading.get_ident()] = f
    
    def unregister(self):
        with self.lock:
            self.thread_files.pop(threading.get_ident(), None)
    
    def write(self, text):
        # Write to original stream
        try:
            self.original_stream.write(text)
            self.original_stream.flush()
        except Exception:
            pass
            
        # Write to thread-local file
        f = self.thread_files.get(threading.get_ident())
        if f:
            try:
                f.write(text)
                f.flush()
            except Exception:
                pass
    
    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
            pass
        f = self.thread_files.get(threading.get_ident())
        if f:
            try:
                f.flush()
            except Exception:
                pass
                
    def __getattr__(self, name):
        return getattr(self.original_stream, name)


_stdout_patched = False
_stderr_patched = False
_patch_lock = threading.Lock()

def _ensure_patched():
    global _stdout_patched, _stderr_patched
    with _patch_lock:
        if not _stdout_patched:
            if not isinstance(sys.stdout, ThreadAwareStdout):
                sys.stdout = ThreadAwareStdout(sys.stdout)
            _stdout_patched = True
        if not _stderr_patched:
            if not isinstance(sys.stderr, ThreadAwareStdout):
                sys.stderr = ThreadAwareStdout(sys.stderr)
            _stderr_patched = True


def get_dockerfile_format_example(language: str) -> str:
    """
    Return concrete Dockerfile example for the detected language.
    NO PLACEHOLDERS - only real, working examples.
    """
    examples = {
        "Java": """
Final Answer:
DOCKERFILE_START
FROM maven:3.9-eclipse-temurin-17
WORKDIR /app
COPY pom.xml ./
RUN mvn dependency:go-offline
COPY src ./src
RUN mvn package -DskipTests
EXPOSE 8080
CMD ["java", "-jar", "target/app.jar"]
DOCKERFILE_END

DOCKERIGNORE_START
.git
target
*.class
.mvn
.idea
*.iml
DOCKERIGNORE_END
""",
        "Python": """
Final Answer:
DOCKERFILE_START
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
DOCKERFILE_END

DOCKERIGNORE_START
.git
__pycache__
*.pyc
.venv
.env
*.egg-info
dist
DOCKERIGNORE_END
""",
        "JavaScript": """
Final Answer:
DOCKERFILE_START
FROM node:20-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["node", "index.js"]
DOCKERFILE_END

DOCKERIGNORE_START
.git
node_modules
npm-debug.log
.env
.next
dist
DOCKERIGNORE_END
""",
        "TypeScript": """
Final Answer:
DOCKERFILE_START
FROM node:20-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["node", "dist/index.js"]
DOCKERFILE_END

DOCKERIGNORE_START
.git
node_modules
npm-debug.log
.env
dist
DOCKERIGNORE_END
""",
        "Go": """
Final Answer:
DOCKERFILE_START
FROM golang:1.22-alpine AS build
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o app .
FROM alpine:latest
WORKDIR /app
COPY --from=build /app/app .
EXPOSE 8080
CMD ["./app"]
DOCKERFILE_END

DOCKERIGNORE_START
.git
vendor
*.log
.env
DOCKERIGNORE_END
""",
        "Rust": """
Final Answer:
DOCKERFILE_START
FROM rust:1.83-slim AS build
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release && rm -rf src
COPY . .
RUN cargo build --release
FROM debian:bookworm-slim
WORKDIR /app
COPY --from=build /app/target/release/app .
EXPOSE 8080
CMD ["./app"]
DOCKERFILE_END

DOCKERIGNORE_START
.git
target
Cargo.lock
.env
DOCKERIGNORE_END
""",
        "C++": """
Final Answer:
DOCKERFILE_START
FROM gcc:13
WORKDIR /app
COPY . .
RUN g++ -o app main.cpp
CMD ["./app"]
DOCKERFILE_END

DOCKERIGNORE_START
.git
*.o
*.out
build
.vscode
DOCKERIGNORE_END
""",
        "C": """
Final Answer:
DOCKERFILE_START
FROM gcc:13
WORKDIR /app
COPY . .
RUN gcc -o app main.c
CMD ["./app"]
DOCKERFILE_END

DOCKERIGNORE_START
.git
*.o
*.out
build
DOCKERIGNORE_END
"""
    }

    return examples.get(language, examples["Python"])


def validate_dockerfile_output(output: str) -> tuple[bool, str]:
    """
    Validate Dockerfile output for common placeholder and syntax errors.
    Returns (is_valid, error_message)
    """
    issues = []

    # Check for placeholder syntax
    heredoc_sanitized = '\n'.join(
        line for line in output.splitlines()
        if '<<' not in line  # Ignore heredoc markers like <<EOF that are valid Dockerfile syntax
    )

    angle_brackets = re.findall(r'<([^>]+)>', heredoc_sanitized)
    if angle_brackets:
        issues.append(f"Found angle bracket placeholders: {', '.join(set(angle_brackets))}")

    square_brackets = re.findall(r'\[([A-Z_]+)\]', heredoc_sanitized)
    if square_brackets:
        issues.append(f"Found square bracket placeholders: {', '.join(set(square_brackets))}")

    # Check for shell syntax in COPY commands (not in RUN)
    copy_lines = re.findall(r'COPY[^\n]*', heredoc_sanitized, re.IGNORECASE)
    for line in copy_lines:
        if '||' in line:
            issues.append(f"Found shell || operator in COPY: {line.strip()}")
        if '2>/dev/null' in line or '2>&1' in line:
            issues.append(f"Found stderr redirect in COPY: {line.strip()}")
        if '&&' in line:
            issues.append(f"Found && operator in COPY: {line.strip()}")

    # Check for real image tags in FROM
    from_matches = re.findall(r'FROM\s+(\S+)', output, re.IGNORECASE)
    for image in from_matches:
        if '<' in image or '[' in image or '{' in image:
            issues.append(f"FROM line has placeholder: {image}")
        if image.endswith(':latest'):
            issues.append(f"Warning: Using :latest tag is not recommended: {image}")

    # Check for curl/wget installing package managers
    if re.search(r'curl.*maven|wget.*maven|curl.*gradle|wget.*gradle', output, re.IGNORECASE):
        issues.append("Found manual Maven/Gradle installation via curl/wget - use official base image instead")

    if issues:
        return False, "Dockerfile validation failed:\n" + '\n'.join(f"  - {i}" for i in issues)

    return True, "Valid"


def _find_missing_copy_sources(dockerfile_content: str, repo_path: str) -> list[str]:
    """
    Detect COPY/ADD statements that reference sources not present in the repo.
    Returns list of human-readable missing paths to surface before build.
    """
    import shlex
    missing = []

    for line in dockerfile_content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if not stripped.lower().startswith(('copy ', 'add ')):
            continue

        try:
            parts = shlex.split(stripped, posix=True)
        except Exception:
            # If parsing fails, skip — we don't want to false-positive
            continue

        # Remove the instruction token
        if parts:
            parts = parts[1:]

        # Drop flags like --chown
        parts = [p for p in parts if not p.startswith('--')]
        if len(parts) < 2:
            continue  # Need at least one source and a destination

        dest = parts[-1]
        sources = parts[:-1]

        for src in sources:
            # Skip remote/absolute paths where existence check isn't meaningful
            if src.startswith('http://') or src.startswith('https://') or src.startswith('/'):
                continue

            src_path = Path(repo_path) / src
            if '*' in src or '?' in src or '[' in src:
                # Glob pattern: treat as missing only if glob returns nothing
                if not list(src_path.parent.glob(src_path.name)):
                    missing.append(src)
            else:
                if not src_path.exists():
                    missing.append(src)

    return missing


def detect_project_language(repo_path: str) -> str:
    """
    Detect the primary programming language of a repository.
    PRIORITIZES build system files over source files to avoid confusion from test/example code.

    Args:
        repo_path: Path to the repository

    Returns:
        Detected language name (e.g., "Python", "JavaScript", "Go", "Rust")
    """
    import os

    # PRIORITY 1: Build system files (highest confidence - check root only)
    # These are definitive indicators of the primary language
    build_system_files = {
        "Java": ["pom.xml", "build.gradle", "build.gradle.kts"],
        "JavaScript": ["package.json"],
        "TypeScript": ["tsconfig.json"],  # Check before package.json
        "Python": ["setup.py", "pyproject.toml"],
        "Go": ["go.mod", "go.sum"],
        "Rust": ["Cargo.toml", "Cargo.lock"],
        "C": ["configure", "configure.ac", "autogen.sh"],
        "C++": ["CMakeLists.txt"],
        "Ruby": ["Gemfile"],
        "PHP": ["composer.json"],
    }

    # Check root directory only for build files (avoids test/example pollution)
    try:
        root_files = os.listdir(repo_path)

        # Priority order: TypeScript > JavaScript (TypeScript projects have package.json too)
        for lang in ["TypeScript", "Java", "Python", "Go", "Rust", "C++", "C", "JavaScript", "Ruby", "PHP"]:
            if lang in build_system_files:
                for build_file in build_system_files[lang]:
                    if build_file in root_files:
                        print(f"[LANGUAGE DETECTION] Found {build_file} in root → {lang}")
                        return lang
    except Exception as e:
        print(f"[LANGUAGE DETECTION] Error checking root: {e}")

    # PRIORITY 2: Secondary config files (moderate confidence)
    secondary_indicators = {
        "Python": ["requirements.txt", "Pipfile"],
        "JavaScript": ["yarn.lock", "package-lock.json", ".npmrc"],
        "C": ["Makefile"],  # Makefiles are common but not definitive
    }

    try:
        for lang, files in secondary_indicators.items():
            for file in files:
                if file in root_files:
                    print(f"[LANGUAGE DETECTION] Found {file} in root → {lang}")
                    return lang
    except:
        pass

    # PRIORITY 3: Count source file LINES (not just files) - fixes scipy false detection
    # Only used as fallback when no build system detected
    # Using line count instead of file count handles mixed-language repos correctly
    # (e.g., scipy has Python code + some Java test files)
    source_patterns = {
        "Python": ["*.py"],
        "JavaScript": ["*.js", "*.jsx"],
        "TypeScript": ["*.ts", "*.tsx"],
        "Go": ["*.go"],
        "Rust": ["*.rs"],
        "Java": ["*.java"],
        "C++": ["*.cpp", "*.hpp", "*.cc", "*.hh", "*.cxx"],
        "C": ["*.c", "*.h"],
        "Ruby": ["*.rb"],
        "PHP": ["*.php"],
        "C#": ["*.cs"],
        "Swift": ["*.swift"],
        "Kotlin": ["*.kt"],
    }

    language_line_counts = {}  # Changed from language_scores (file counts)

    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories and common test/example dirs
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in [
            'test', 'tests', 'examples', 'example', 'docs', 'doc', 'node_modules', '__pycache__', '.git'
        ]]

        for file in files:
            # Determine language from extension
            for lang, patterns in source_patterns.items():
                for pattern in patterns:
                    ext = pattern.replace('*', '')  # e.g., "*.py" → ".py"
                    if file.endswith(ext):
                        # Count lines in this file
                        file_path = Path(root) / file
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                line_count = len(f.readlines())
                                language_line_counts[lang] = language_line_counts.get(lang, 0) + line_count
                        except:
                            # If can't read file, count it as 100 lines (rough estimate)
                            language_line_counts[lang] = language_line_counts.get(lang, 0) + 100
                        break

    # Return the language with most lines of code
    if language_line_counts:
        detected = max(language_line_counts.items(), key=lambda x: x[1])[0]
        total_lines = sum(language_line_counts.values())
        percentage = (language_line_counts[detected] / total_lines * 100) if total_lines > 0 else 0

        print(f"[LANGUAGE DETECTION] Line count → {detected} ({language_line_counts[detected]:,} lines, {percentage:.1f}%)")

        # Show runner-up if close (helps diagnose mixed repos)
        sorted_langs = sorted(language_line_counts.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_langs) > 1:
            runner_up_lang, runner_up_lines = sorted_langs[1]
            runner_up_pct = (runner_up_lines / total_lines * 100) if total_lines > 0 else 0
            print(f"[LANGUAGE DETECTION] Runner-up: {runner_up_lang} ({runner_up_lines:,} lines, {runner_up_pct:.1f}%)")

        # Special case: TypeScript often comes with JavaScript
        if detected == "TypeScript" and "JavaScript" in language_line_counts:
            return "TypeScript"
        return detected

    print("[LANGUAGE DETECTION] No language detected → Unknown")
    return "Unknown"


def _has_dockerfile(output: str) -> bool:
    """Check if output contains a Dockerfile (has FROM statement)."""
    if not output or not output.strip():
        return False
    return "FROM" in output.upper()


def _has_repeated_tool_calls(intermediate_steps, threshold: int = 3) -> bool:
    """
    Detect thrashing where the agent repeatedly calls the same tool with the same input.
    Returns True if any (tool, input) pair occurs at least `threshold` times.
    """
    if not intermediate_steps:
        return False

    counts = Counter()
    for step in intermediate_steps:
        try:
            action = step[0]
            tool_name = getattr(action, "tool", None) or getattr(action, "tool_name", None)
            tool_input = getattr(action, "tool_input", None) or getattr(action, "input", None)
            key = (tool_name, str(tool_input))
            counts[key] += 1
        except Exception:
            # If we can't parse this step, skip it
            continue

    return any(count >= threshold for count in counts.values())


def _validate_dockerfile(dockerfile_content: str, repo_path: str = None) -> tuple[bool, list[str]]:
    """
    Validate Dockerfile quality BEFORE proceeding to Docker build (language-agnostic).

    Checks:
    1. Has FROM statement
    2. No placeholder text ([PATH], [HASH], DOCKERFILE_END markers)
    3. No shell operators in COPY commands
    4. Basic syntax validity

    This saves 100+ seconds by catching broken Dockerfiles early instead of during build.

    Args:
        dockerfile_content: The generated Dockerfile
        repo_path: Path to repository (optional, for COPY validation)

    Returns:
        (is_valid, list_of_errors)
    """
    import os
    import re

    errors = []

    if not dockerfile_content:
        errors.append("Dockerfile is empty")
        return False, errors

    # Check 1: Must have FROM statement
    if not _has_dockerfile(dockerfile_content):
        errors.append("Missing FROM statement")
        return False, errors

    # Check 2: No placeholder text (agent didn't finish substituting values)
    placeholders = ['[PATH]', '[HASH]', '[IMAGE]', '[TAG]', '[VERSION]', '[PORT]',
                   'DOCKERFILE_END', 'DOCKERFILE_START', 'DOCKERIGNORE_START', 'DOCKERIGNORE_END',
                   '<image>', '<tag>', '<hash>', '<version>', '<port>', '<command>',
                   'TODO', 'FIXME', 'PLACEHOLDER']

    found_placeholders = []
    for p in placeholders:
        if p in dockerfile_content:
            found_placeholders.append(p)

    if found_placeholders:
        errors.append(f"Contains unsubstituted placeholders: {', '.join(found_placeholders[:5])}")

    # Check 3: No shell operators in COPY/ADD commands (common mistake)
    copy_add_lines = [line.strip() for line in dockerfile_content.split('\n')
                     if line.strip().upper().startswith(('COPY ', 'ADD '))]

    for line in copy_add_lines:
        # Check for shell operators that don't belong in Dockerfile
        if any(op in line for op in ['||', '&&', '2>/dev/null', '2>&1', '>/dev/null', ' | ']):
            errors.append(f"COPY/ADD contains shell operators (not valid): {line[:70]}")
            break  # One example is enough

    # Check 4: Basic syntax - lines should start with valid commands or continuations
    valid_commands = ['FROM', 'RUN', 'COPY', 'ADD', 'WORKDIR', 'EXPOSE', 'CMD', 'ENTRYPOINT',
                     'ENV', 'ARG', 'LABEL', 'USER', 'VOLUME', 'HEALTHCHECK', 'SHELL',
                     'ONBUILD', 'STOPSIGNAL', 'MAINTAINER']

    lines = dockerfile_content.split('\n')
    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            continue

        # Check if it's a continuation line
        if i > 1 and lines[i-2].rstrip().endswith('\\'):
            continue  # Continuation of previous command

        # Check if starts with valid command
        starts_with_valid_cmd = any(stripped.upper().startswith(cmd) for cmd in valid_commands)

        if not starts_with_valid_cmd:
            # Could be malformed or non-Dockerfile content leaked in
            errors.append(f"Line {i} doesn't start with valid Dockerfile instruction: '{stripped[:60]}'")
            break  # One error is enough to indicate problems

    # Check 5: FROM line should reference a real image (not localhost paths or weird formats)
    from_lines = [line.strip() for line in lines if line.strip().upper().startswith('FROM ')]
    if from_lines:
        first_from = from_lines[0]
        # Extract image name (everything after FROM and before AS)
        from_parts = first_from.split()
        if len(from_parts) >= 2:
            image = from_parts[1]

            # Check for obviously wrong formats
            if image.startswith(('/', '\\', './', '../')):
                errors.append(f"FROM references local path instead of Docker image: {image}")
            elif image.startswith('docker.io[PATH]') or '[PATH]' in image:
                errors.append(f"FROM contains placeholder: {image}")

    # Check 6: Multi-stage build validation (fixes imgui, xgboost failures)
    # Extract all FROM lines with optional AS clauses
    # Pattern: FROM image:tag [AS name]
    from_pattern = r'FROM\s+(\S+)(?:\s+AS\s+(\S+))?'
    from_matches = re.findall(from_pattern, dockerfile_content, re.IGNORECASE)

    # Collect all defined stage names
    defined_stages = set()
    for image, stage_name in from_matches:
        if stage_name:  # If AS clause exists
            defined_stages.add(stage_name.lower())

    # Find all COPY --from=<stage> references
    copy_from_pattern = r'COPY\s+--from=(\S+)'
    copy_from_refs = re.findall(copy_from_pattern, dockerfile_content, re.IGNORECASE)

    # Validate each reference
    for ref in copy_from_refs:
        ref_lower = ref.lower()

        # Skip numeric references (COPY --from=0 is valid)
        if ref.isdigit():
            continue

        # Check if stage is defined
        if ref_lower not in defined_stages:
            errors.append(
                f"COPY --from={ref} references undefined stage. "
                f"Available stages: {sorted(defined_stages) if defined_stages else 'none'}"
            )
            errors.append(
                f"FIX: Add 'AS {ref}' to the FROM line, or use stage index (--from=0)"
            )

    # Return validation result
    is_valid = len(errors) == 0
    return is_valid, errors


def _invoke_agent_with_iteration_limit(agent, inputs: dict, max_iterations: int = None):
    """
    Thread-safe wrapper to invoke agent with optional iteration limit.
    Does not mutate the agent object permanently, safe for parallel execution.
    
    THREAD SAFETY:
    - Each parallel test creates its own agent instance (see parallel_empirical_test.py)
    - This function temporarily modifies max_iterations for ONE invocation only
    - Uses try-finally to ALWAYS restore original value, even on exceptions
    - No race conditions: each agent instance is isolated to its test thread
    
    Args:
        agent: AgentExecutor instance (unique per test)
        inputs: Input dictionary for agent.invoke()
        max_iterations: Optional iteration limit (None = use agent default)
    
    Returns:
        Agent result dictionary
    """
    if max_iterations is None:
        # Use default behavior
        return agent.invoke(inputs)
    
    # Thread-safe approach: temporarily override max_iterations
    # Save original value
    original_max_iterations = agent.max_iterations
    
    try:
        # Set temporary limit (only affects this agent instance)
        agent.max_iterations = max_iterations
        # Invoke agent
        result = agent.invoke(inputs)
        return result
    finally:
        # Always restore original value, even if exception occurs
        # This ensures the agent is back to its original state
        agent.max_iterations = original_max_iterations


def analyze_repository(agent, repo_path: str, repo_name: str, repo_url: str, callback_handler, log_file_path=None, report_dir=None):
    """
    Run analysis queries on a cloned repository.

    Args:
        agent: The planner agent instance
        repo_path: Path to the cloned repository
        repo_name: Name of the repository
        repo_url: URL of the repository
        callback_handler: FormattedOutputHandler instance for token tracking
        log_file_path: Optional path to log file for saving agent steps
    """
    # Create temporary log file to capture all output
    temp_log_file = None
    if log_file_path is None:
        temp_log_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8')
        log_file_path = Path(temp_log_file.name)
        temp_log_file.close()
    
    # Open log file for writing
    log_file_handle = open(log_file_path, 'w', encoding='utf-8')
    
    # Create a tee that writes to both stdout and log file
    # Ensure stdout/stderr are patched
    _ensure_patched()
    
    # Register log file for this thread
    sys.stdout.register(log_file_handle)
    sys.stderr.register(log_file_handle)
    
    try:
        # Configure logging to use sys.stderr (which is now patched)
        import logging
        
        # Check if we have a handler for sys.stderr already
        has_stderr_handler = False
        for h in logging.root.handlers:
            if isinstance(h, logging.StreamHandler) and h.stream == sys.stderr:
                has_stderr_handler = True
                break
        
        if not has_stderr_handler:
            # Create a new StreamHandler that writes to sys.stderr
            handler = logging.StreamHandler(sys.stderr)
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logging.root.addHandler(handler)
            logging.root.setLevel(logging.INFO)
        
        print(f"\n{'='*70}")
        print(f"Analyzing Repository: {repo_name}")
        print('='*70)

        # Detect project language before analysis
        detected_language = detect_project_language(repo_path)
        print(f"[INFO] Detected primary language: {detected_language}")
        
        # Set thread-local report directory for web search to save files (thread-safe)
        if report_dir:
            import planner_agent
            planner_agent._set_report_directory(str(report_dir))
            # NOTE: No longer setting global REPORT_DIRECTORY - thread-local only for thread safety
            print(f"[INFO] Report directory: {report_dir}")

        # Define analysis queries using discovery-based approach with relative paths
        # Use simple string concatenation to avoid f-string issues with JSON braces
        first_query = "Analyze the repository. **FIRST**: Search the web for official documentation using the repository name and language. Then show me the directory tree structure (depth 2), and identify what type of project this is by finding and examining configuration files."

        queries = [
        first_query,

        "Based on what you discovered in the previous step, find all build-related configuration files and extract the key information like dependencies, build scripts, runtime requirements, and environment variables. Also cross-reference with the official documentation you found earlier. Pay special attention to: version requirements, system dependencies, build tools needed, and any special environment setup.",

        "Based on everything you've learned so far, read the README file and extract installation/build instructions. Also search for any 'install', 'build', 'run', or 'start' commands mentioned in configuration files or scripts. Identify: entry points, default ports, required environment variables, volume mounts, and any runtime configuration. Compare with official documentation.",

        # NEW STEP: Mandatory Modern Base Image Selection with Discovery-Based Approach
        # NEW STEP: Mandatory Modern Base Image Selection (Concise Version)
        """**MANDATORY: SELECT MODERN BASE IMAGE**

You MUST use the `DockerImageSearch` tool to find a modern, maintained base image.

RULES:
1.  **Search Tags:** Call `DockerImageSearch` with `tags:<image>` (e.g., `tags:python`).
2.  **Select Modern Tag:** Pick a tag updated in 2024/2025. AVOID tags older than 1 year.
3.  **Check Architecture:** Ensure the tag has `[OK]` for the host architecture.
4.  **Verify:** Call `DockerImageSearch` with `<image>:<tag>` to confirm existence.

CRITICAL: Do NOT use EOL versions (Python 2.7, Node 14, Java 8). Use LTS/Stable versions.
""",


            """Based on the information you gathered in Query 1, 2, 3, and the VERIFIED base image from Query 4, create a Dockerfile AND a .dockerignore file.

**IMPORTANT**: You already have all the information needed:
- Query 1-3: Project structure, dependencies, build instructions
- Query 4: The VERIFIED base image (use this EXACTLY)

**YOUR TASK**: Generate the Dockerfile and .dockerignore.

**BANNED SYNTAX** (These will cause build failures):
❌ **NO angle bracket placeholders: <image>, <tag>, <path>, <command>**
❌ **NO square bracket placeholders: [PATH], [HASH], [VERSION]**
❌ **NO curly brace placeholders: {variable}, {command}**
❌ **NO shell operators in COPY: ||, 2>/dev/null, &&**
❌ **NO manual package manager installation: curl maven, wget gradle**
❌ **NO delimiters inside file content (they're only format markers)**

**CORRECT OUTPUT FORMAT** (adapt this example for your project's language):

Final Answer:
DOCKERFILE_START
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
DOCKERFILE_END

DOCKERIGNORE_START
.git
__pycache__
*.pyc
.venv
.env
DOCKERIGNORE_END

**CRITICAL RULES**:
✅ Use REAL image tags: python:3.12-slim, node:20-slim, maven:3.9-eclipse-temurin-17
✅ Use REAL paths: ./src, /app, target/app.jar
✅ Use REAL commands: npm install, mvn package, cargo build
✅ Must start with "Final Answer:" on its own line
✅ Use verified image from Query 4 (if provided)

**ADAPTATION GUIDE**:
- For Java/Maven: FROM maven:3.9-eclipse-temurin-17, RUN mvn package, CMD java -jar target/app.jar
- For Python: FROM python:3.12-slim, RUN pip install -r requirements.txt, CMD python app.py
- For Node.js: FROM node:20-slim, RUN npm ci, CMD node index.js
- For Rust: FROM rust:1.83-slim, RUN cargo build --release, CMD ./target/release/app
- For Go: FROM golang:1.22-alpine, RUN go build, CMD ./app

**DOCKERIGNORE MUST INCLUDE**:
- .git (CRITICAL - always exclude)
- Language-specific: node_modules, __pycache__, *.pyc, target, .venv, vendor
- Build artifacts: build, dist, .next, out
- Environment: .env, .env.local
- IDE: .vscode, .idea, *.iml

Now generate the Final Answer with both files using REAL values for your specific project.
"""
        ]

        # Initialize conversation history to maintain context across queries
        # Initialize conversation history to maintain context across queries
        chat_history = []
        final_instructions = None
        dockerignore_content = None
        total_tokens = {"input": 0, "output": 0, "total": 0}
        tool_usage = {}  # Track how many times each tool is used
        start_time = time.time()  # Track analysis duration

        for i, query in enumerate(queries, 1):
            print(f"\n{'-'*70}")
            print(f"Analysis Step {i}/{len(queries)}")
            print(f"{'-'*70}")
            print(f"Query: {query}\n")

            try:
                # Format chat history as readable text
                formatted_history = ""
                if chat_history:
                    formatted_history = "\n".join([
                        f"Previous Query: {msg['content']}" if msg['role'] == 'user'
                        else f"Previous Answer: {msg['content'][:500]}..." if len(msg['content']) > 500
                        else f"Previous Answer: {msg['content']}"
                        for msg in chat_history
                    ])

                # Special handling for final query (Query 4 - Dockerfile generation)
                # Use smart retry with max_iterations control to prevent endless tool calls
                max_iterations_for_final = 15  # Allow more room for image verification while still capping thrash
                
                if i == len(queries):
                    # For final query, limit iterations to prevent endless searches
                    # Thread-safe: uses wrapper function with try-finally
                    print(f"[INFO] Final query - limiting to {max_iterations_for_final} tool calls max")
                    
                    result = _invoke_agent_with_iteration_limit(
                        agent,
                        {
                            "input": query,
                            "chat_history": formatted_history or "No previous context."
                        },
                        max_iterations=max_iterations_for_final
                    )
                else:
                    # Normal invocation for queries 1-3
                    result = _invoke_agent_with_iteration_limit(
                        agent,
                        {
                            "input": query,
                            "chat_history": formatted_history or "No previous context."
                        },
                        max_iterations=None  # Use default
                    )

                output = result.get('output', '')
                print(f"\n[RESULT]\n{output}\n")

                # Track tool usage if intermediate steps are available
                if 'intermediate_steps' in result:
                    for action, _ in result['intermediate_steps']:
                        tool_name = action.tool
                        tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1


                # Track token usage if available in result - try multiple possible keys
                usage_found = False

                # Try different possible locations for usage data
                for key in ['usage_metadata', 'usage', 'token_usage', 'llm_output']:
                    if key in result and result[key]:
                        usage = result[key]
                        if isinstance(usage, dict):
                            # Try different token key names
                            input_tokens = usage.get('input_tokens') or usage.get('prompt_tokens') or usage.get('total_input_tokens', 0)
                            output_tokens = usage.get('output_tokens') or usage.get('completion_tokens') or usage.get('total_output_tokens', 0)
                            total = usage.get('total_tokens', input_tokens + output_tokens)

                            if input_tokens or output_tokens:
                                total_tokens["input"] += input_tokens
                                total_tokens["output"] += output_tokens
                                total_tokens["total"] += total
                                usage_found = True
                                print(f"[TOKEN USAGE] Step {i}: +{input_tokens} input, +{output_tokens} output, +{total} total")
                                break

                # Debug: print available keys on first iteration if no usage found
                if i == 1 and not usage_found:
                    print(f"[DEBUG] Result keys available: {list(result.keys())}")
                    print(f"[DEBUG] Will rely on callback handler for token tracking")

                # Save the final step output (Dockerfile and .dockerignore)
                if i == len(queries):
                    # Parse the output for Dockerfile and .dockerignore
                    dockerfile_content = None
                    # dockerignore_content is already initialized outside loop

                    # Clean output first - remove backticks that may wrap delimiters
                    cleaned_output = output.replace('`', '')

                    # Try to parse with new format
                    if "DOCKERFILE_START" in cleaned_output and "DOCKERFILE_END" in cleaned_output:
                        try:
                            dockerfile_content = cleaned_output.split("DOCKERFILE_START")[1].split("DOCKERFILE_END")[0].strip()
                        except IndexError:
                            pass

                    if "DOCKERIGNORE_START" in cleaned_output and "DOCKERIGNORE_END" in cleaned_output:
                        try:
                            dockerignore_content = cleaned_output.split("DOCKERIGNORE_START")[1].split("DOCKERIGNORE_END")[0].strip()
                        except IndexError:
                            pass

                    # Validate Dockerfile for placeholder and syntax errors
                    if dockerfile_content:
                        is_valid, validation_error = validate_dockerfile_output(dockerfile_content)
                        if not is_valid:
                            print(f"\n[VALIDATION ERROR] {validation_error}")
                            print(f"[INFO] Will attempt to retry with corrected instructions...")
                            # Clear dockerfile_content to trigger retry with validation feedback
                            invalid_dockerfile = dockerfile_content
                            dockerfile_content = None

                    # Fallback: Check if output contains Dockerfile directly (old behavior)
                    if not dockerfile_content and _has_dockerfile(output):
                        dockerfile_content = output
                        # Remove delimiters if present (safety check for validation)
                        # This handles cases where agent included delimiters in the content
                        if "DOCKERFILE_START" in dockerfile_content:
                            dockerfile_content = re.sub(r'DOCKERFILE_START|DOCKERFILE_END|DOCKERIGNORE_START|DOCKERIGNORE_END', '', dockerfile_content).strip()

                        # Validate again
                        is_valid, validation_error = validate_dockerfile_output(dockerfile_content)
                        if not is_valid:
                            print(f"\n[VALIDATION ERROR] {validation_error}")
                            dockerfile_content = None
                    
                    # Smart retry logic if Dockerfile is missing
                    max_retries = 2  # Keep two retries: one format reminder, one forced answer
                    for retry_attempt in range(max_retries):
                        if dockerfile_content and _has_dockerfile(dockerfile_content):
                            break
                        
                        # Detect if agent is stuck doing tool calls instead of providing answer
                        steps = result.get('intermediate_steps', [])
                        tool_calls_in_result = len(steps)
                        repeat_thrash = _has_repeated_tool_calls(steps, threshold=3)
                        is_stuck_searching = tool_calls_in_result >= max_iterations_for_final or repeat_thrash
                        
                        if retry_attempt == 0:
                            if is_stuck_searching:
                                reason = "iteration cap" if tool_calls_in_result >= max_iterations_for_final else "repeated tool calls"
                                print(f"\n[WARNING] Agent is stuck ({reason}: {tool_calls_in_result} calls). Forcing direct answer...")
                                print(f"[INFO] Forcing direct answer with NO tool calls allowed...")
                            else:
                                print(f"\n[WARNING] Output doesn't contain valid Dockerfile. Retrying with strict format prompt...")
                        else:
                            print(f"\n[WARNING] Retry {retry_attempt + 1}/{max_retries} still didn't produce valid Dockerfile. Retrying again...")
                        
                        # Build retry query with progressively stricter instructions
                        # Check if we have validation errors to include
                        validation_feedback = ""
                        if 'invalid_dockerfile' in locals() and invalid_dockerfile:
                            validation_feedback = f"""
YOUR PREVIOUS OUTPUT HAD THESE ERRORS:
{validation_error}

FIX THESE ISSUES:
- Use REAL values: python:3.12-slim NOT <image>:<tag>
- Use REAL paths: ./src NOT <path>
- Use REAL commands: npm install NOT <command>
- NO angle brackets, square brackets, or curly braces
"""

                        if is_stuck_searching or retry_attempt > 0:
                            # Force immediate answer - NO tool calls
                            retry_query = f"""CRITICAL: You MUST provide the Final Answer NOW using the information already available from previous queries (Query 1, 2, 3).

DO NOT use any tools. DO NOT search. DO NOT read files. Just generate the Dockerfile and .dockerignore using the context you already have.
{validation_feedback}
PREVIOUS CONTEXT SUMMARY:
{formatted_history[-1000:] if formatted_history else "See chat history"}

OUTPUT FORMAT (REQUIRED - adapt for your language this is an real example):
Final Answer:
DOCKERFILE_START
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "app.py"]
DOCKERFILE_END

DOCKERIGNORE_START
.git
node_modules
__pycache__
*.pyc
.env
DOCKERIGNORE_END

Use REAL image tags (python:3.12-slim, node:20-slim, maven:3.9-eclipse-temurin-17).
Use REAL paths (./src, /app, requirements.txt).
Use REAL commands (pip install, npm ci, mvn package).
NO placeholders of any kind.

Provide ONLY the Final Answer in this format. No tool calls."""
                            
                            # Use thread-safe wrapper with max_iterations=1 to force immediate answer
                            retry_result = _invoke_agent_with_iteration_limit(
                                agent,
                                {
                                    "input": retry_query,
                                    "chat_history": formatted_history or "No previous context."
                                },
                                max_iterations=1
                            )
                        else:
                            # Normal retry with format reminder
                            retry_result = _invoke_agent_with_iteration_limit(
                                agent,
                                {
                                    "input": queries[-1] + "\n\nREMINDER: You MUST use the exact format with DOCKERFILE_START and DOCKERIGNORE_START tags. Use information from previous queries (Query 1, 2, 3).",
                                    "chat_history": formatted_history or "No previous context."
                                },
                                max_iterations=None  # Use default
                            )
                        
                        output = retry_result.get('output', output)
                        
                        # Re-parse after retry
                        if "DOCKERFILE_START" in output and "DOCKERFILE_END" in output:
                            try:
                                dockerfile_content = output.split("DOCKERFILE_START")[1].split("DOCKERFILE_END")[0].strip()
                            except IndexError:
                                pass
                        
                        if "DOCKERIGNORE_START" in output and "DOCKERIGNORE_END" in output:
                            try:
                                dockerignore_content = output.split("DOCKERIGNORE_START")[1].split("DOCKERIGNORE_END")[0].strip()
                            except IndexError:
                                pass
                                
                        if not dockerfile_content and _has_dockerfile(output):
                            dockerfile_content = output

                    # CRITICAL FALLBACK: Aggressive extraction to prevent "Dockerfile not generated" errors
                    # Even if format is imperfect, we extract SOMETHING that refinement can fix
                    if not dockerfile_content or not _has_dockerfile(dockerfile_content):
                        print(f"\n[WARNING] Standard extraction failed. Attempting aggressive fallback extraction...")

                        # Strategy 1: Extract from markdown code blocks
                        if "```dockerfile" in output.lower() or "```docker" in output.lower():
                            try:
                                # Find dockerfile code block
                                pattern = r'```(?:dockerfile|docker)\s*\n(.*?)```'
                                match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
                                if match:
                                    dockerfile_content = match.group(1).strip()
                                    print(f"[FALLBACK] Extracted Dockerfile from markdown code block")
                            except Exception as e:
                                print(f"[FALLBACK] Markdown extraction failed: {e}")

                        # Strategy 2: Extract anything that looks like a Dockerfile (starts with FROM)
                        if not dockerfile_content or not _has_dockerfile(dockerfile_content):
                            try:
                                lines = output.split('\n')
                                dockerfile_lines = []
                                in_dockerfile = False

                                for line in lines:
                                    # Start capturing when we see FROM
                                    if line.strip().upper().startswith('FROM '):
                                        in_dockerfile = True
                                        dockerfile_lines = [line]
                                    elif in_dockerfile:
                                        # Stop if we hit obvious delimiters or text
                                        if any(marker in line for marker in ['```', 'DOCKERIGNORE_START', 'Thought:', 'Action:', 'Observation:']):
                                            break
                                        # Stop if line looks like natural language (not Dockerfile command)
                                        stripped = line.strip().upper()
                                        if stripped and not any(stripped.startswith(cmd) for cmd in [
                                            'FROM', 'RUN', 'COPY', 'ADD', 'WORKDIR', 'EXPOSE', 'CMD', 'ENTRYPOINT',
                                            'ENV', 'ARG', 'LABEL', 'USER', 'VOLUME', 'HEALTHCHECK', 'SHELL', '#'
                                        ]):
                                            # Might be end of Dockerfile
                                            break
                                        dockerfile_lines.append(line)

                                if dockerfile_lines:
                                    dockerfile_content = '\n'.join(dockerfile_lines).strip()
                                    print(f"[FALLBACK] Extracted Dockerfile from FROM keyword ({len(dockerfile_lines)} lines)")
                            except Exception as e:
                                print(f"[FALLBACK] FROM-based extraction failed: {e}")

                        # Strategy 3: Last resort - extract entire output if it contains FROM
                        if not dockerfile_content or not _has_dockerfile(dockerfile_content):
                            if 'FROM ' in output:
                                # Clean up obvious non-Dockerfile content
                                cleaned = output
                                # Remove delimiters if present
                                for delimiter in ['DOCKERFILE_START', 'DOCKERFILE_END', 'DOCKERIGNORE_START', 'DOCKERIGNORE_END']:
                                    cleaned = cleaned.replace(delimiter, '')
                                # Remove markdown
                                cleaned = cleaned.replace('```dockerfile', '').replace('```docker', '').replace('```', '')
                                # Remove obvious agent formatting
                                for marker in ['Thought:', 'Action:', 'Action Input:', 'Observation:', 'Final Answer:']:
                                    if marker in cleaned:
                                        # Take content after Final Answer if present
                                        if marker == 'Final Answer:':
                                            cleaned = cleaned.split(marker, 1)[-1]
                                        else:
                                            # Remove these lines
                                            lines = cleaned.split('\n')
                                            cleaned = '\n'.join(line for line in lines if not line.strip().startswith(marker))

                                dockerfile_content = cleaned.strip()
                                print(f"[FALLBACK] Extracted cleaned output containing FROM keyword")

                    # Final validation and messaging
                    if not dockerfile_content or not _has_dockerfile(dockerfile_content):
                        print(f"\n[ERROR] Failed to generate Dockerfile after {max_retries + 1} attempts + fallback extraction")
                        print(f"[ERROR] No placeholder will be created - this is a genuine failure")
                        # DO NOT create placeholder - let the failure be properly recorded
                        dockerfile_content = None
                    else:
                        print(f"\n[SUCCESS] Dockerfile generated successfully!")

                        # CRITICAL: Validate Dockerfile quality BEFORE proceeding to Docker build
                        # This catches broken Dockerfiles early (saves 100+ seconds per iteration)
                        is_valid, validation_errors = _validate_dockerfile(dockerfile_content, repo_path)

                        # Early check for missing COPY/ADD sources to avoid FILE_COPY_MISSING cycles
                        missing_copy = _find_missing_copy_sources(dockerfile_content, repo_path) if repo_path else []
                        if missing_copy:
                            validation_errors.append(
                                f"COPY/ADD sources not found in repo: {', '.join(sorted(set(missing_copy)))[:300]}"
                            )
                            is_valid = False

                        if not is_valid:
                            print(f"\n[VALIDATION FAILED] Dockerfile has {len(validation_errors)} quality issues:")
                            for err in validation_errors:
                                print(f"  - {err}")

                            # Give agent ONE chance to fix validation errors immediately
                            print(f"\n[VALIDATION] Requesting immediate fix from agent...")

                            validation_fix_query = f"""CRITICAL VALIDATION ERRORS in your Dockerfile!

Your Dockerfile has the following issues that will cause build failures:
{chr(10).join(f"- {err}" for err in validation_errors)}

**CURRENT DOCKERFILE** (with problems):
```dockerfile
{dockerfile_content}
```

**REQUIRED FIX**:
1. Read the errors above carefully
2. Generate a CORRECTED Dockerfile that fixes ALL issues
3. DO NOT use placeholder text like [PATH], [HASH], TODO, etc.
4. DO NOT use shell operators in COPY commands (no ||, &&, 2>/dev/null)
5. Ensure all lines start with valid Dockerfile instructions

**OUTPUT FORMAT** (REQUIRED):
DOCKERFILE_START
FROM <actual-image-name-and-tag>
... corrected Dockerfile ...
DOCKERFILE_END

Provide ONLY the corrected Dockerfile. No tool calls needed - just fix the issues listed above."""

                            # Force immediate answer (no tool calls)
                            validation_retry = _invoke_agent_with_iteration_limit(
                                agent,
                                {
                                    "input": validation_fix_query,
                                    "chat_history": formatted_history or "No previous context."
                                },
                                max_iterations=1  # Force immediate answer
                            )

                            retry_output = validation_retry.get('output', '')

                            # Try to extract fixed Dockerfile
                            fixed_dockerfile = None
                            if "DOCKERFILE_START" in retry_output and "DOCKERFILE_END" in retry_output:
                                try:
                                    fixed_dockerfile = retry_output.split("DOCKERFILE_START")[1].split("DOCKERFILE_END")[0].strip()
                                except IndexError:
                                    pass

                            if not fixed_dockerfile and _has_dockerfile(retry_output):
                                fixed_dockerfile = retry_output

                            # Re-validate
                            if fixed_dockerfile:
                                is_valid_retry, retry_errors = _validate_dockerfile(fixed_dockerfile, repo_path)
                                if is_valid_retry:
                                    print(f"[VALIDATION] ✓ Agent fixed all issues! Using corrected Dockerfile.")
                                    dockerfile_content = fixed_dockerfile
                                else:
                                    print(f"[VALIDATION] ✗ Agent's fix still has {len(retry_errors)} errors. Using original (refinement will fix later).")
                                    # Keep original - let refinement loop handle it
                            else:
                                print(f"[VALIDATION] Agent didn't provide valid fix. Using original (refinement will fix later).")
                        else:
                            print(f"[VALIDATION] ✓ Dockerfile passed all quality checks!")

                    final_instructions = dockerfile_content
                    
                    # Save .dockerignore to repo immediately if found
                    if dockerignore_content:
                        try:
                            dockerignore_path = Path(repo_path) / ".dockerignore"
                            with open(dockerignore_path, 'w', encoding='utf-8') as f:
                                f.write(dockerignore_content)
                            print(f"[OK] .dockerignore written to repository: {dockerignore_path}")
                        except Exception as e:
                            print(f"[ERROR] Failed to write .dockerignore to repo: {e}")

                # Add this interaction to chat history for next query
                chat_history.extend([
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": output}
                ])

            except Exception as e:
                print(f"\n[ERROR] {e}\n")
                import traceback
                traceback.print_exc()

        # Calculate duration
        end_time = time.time()
        duration_seconds = end_time - start_time

    finally:
        # Unregister log file
        if hasattr(sys.stdout, 'unregister'):
            sys.stdout.unregister()
        if hasattr(sys.stderr, 'unregister'):
            sys.stderr.unregister()
        
        # Close log file
        log_file_handle.close()
    
    # Print summary to console (for immediate feedback)
    print(f"\n{'='*70}")
    print("Analysis Summary")
    print('='*70)
    
    duration_minutes = duration_seconds / 60
    if duration_minutes >= 1:
        print(f"Duration: {int(duration_minutes)} min {int(duration_seconds % 60)} sec")
    else:
        print(f"Duration: {duration_seconds:.2f} seconds")
    
    if tool_usage:
        total_tool_calls = sum(tool_usage.values())
        print(f"Tool calls: {total_tool_calls} (across {len(tool_usage)} unique tools)")
    
    if callback_handler and callback_handler.token_usage["total"] > 0:
        usage = callback_handler.token_usage
        print(f"Tokens: {usage['total']:,} (input: {usage['input']:,}, output: {usage['output']:,})")
    elif total_tokens["total"] > 0:
        print(f"Tokens: {total_tokens['total']:,} (input: {total_tokens['input']:,}, output: {total_tokens['output']:,})")
    
    print('='*70)
    
    # Save all reports and metrics to structured folder
    # Use provided report_dir if available, otherwise create new one
    if report_dir is None:
        report_dir = save_analysis_reports(
            repo_name=repo_name,
            repo_url=repo_url,
            repo_path=repo_path,
            final_dockerfile=final_instructions,
            tool_usage=tool_usage,
            total_tokens=total_tokens,
            duration_seconds=duration_seconds,
            callback_handler=callback_handler,
            log_file_path=log_file_path,
            final_dockerignore=dockerignore_content
        )
    else:
        # Use existing report directory
        save_analysis_reports(
            repo_name=repo_name,
            repo_url=repo_url,
            repo_path=repo_path,
            final_dockerfile=final_instructions,
            tool_usage=tool_usage,
            total_tokens=total_tokens,
            duration_seconds=duration_seconds,
            callback_handler=callback_handler,
            log_file_path=log_file_path,
            report_dir=report_dir,
            final_dockerignore=dockerignore_content
        )
    
    # Clean up temporary log file
    if temp_log_file and log_file_path and log_file_path.exists():
        try:
            log_file_path.unlink()
        except:
            pass
    
    return report_dir


def main():
    """Main function to run repository analysis workflow."""
    print("="*70)
    print("Planner Agent - Repository Analysis Workflow")
    print("="*70)

    # Check if repository URL is provided
    if len(sys.argv) < 2:
        print("\nUsage: python run_agent.py <github_repo_url>")
        print("\nExample:")
        print("  python run_agent.py https://github.com/psf/requests")
        print("  python run_agent.py https://github.com/microsoft/playwright")
        sys.exit(1)

    repo_url = sys.argv[1]

    # Validate GitHub URL
    if "github.com" not in repo_url:
        print(f"\n[ERROR] Invalid GitHub URL: {repo_url}")
        print("Please provide a valid GitHub repository URL")
        sys.exit(1)

    try:
        # Step 1: Clone the repository
        print("\n" + "="*70)
        print("Step 1: Cloning Repository")
        print("="*70)
        repo_path = clone_repository(repo_url)
        repo_name = os.path.basename(repo_path)

        # Detect language
        detected_language = detect_project_language(repo_path)
        print(f"[INFO] Detected primary language: {detected_language}")

        # Step 2: Initialize the agent
        print("\n" + "="*70)
        print("Step 2: Initializing Agent")
        print("="*70)
        agent, callback_handler = create_planner_agent(
            max_iterations=25, 
            verbose=True, 
            repository_path=repo_path,
            repo_name=repo_name,
            detected_language=detected_language
        )
        print("\n[OK] Agent initialized successfully!")

        # Step 3: Analyze the repository
        print("\n" + "="*70)
        print("Step 3: Running Analysis")
        print("="*70)
        # Create report directory early
        script_dir = Path(__file__).parent
        root_dir = script_dir.parent
        reports_base_dir = root_dir / "analysis_reports"
        reports_base_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = reports_base_dir / f"{repo_name}_{timestamp}"
        report_dir.mkdir(exist_ok=True)
        
        # Set thread-local report directory for web search to save files (thread-safe)
        import planner_agent
        planner_agent._set_report_directory(str(report_dir))
        # NOTE: No longer setting global REPORT_DIRECTORY - thread-local only for thread safety
        
        report_dir_result = analyze_repository(agent, repo_path, repo_name, repo_url, callback_handler, report_dir=report_dir)
        # Use the returned report_dir (should be same as what we passed)
        if report_dir_result:
            report_dir = report_dir_result

        # Summary
        print("\n" + "="*70)
        print("Analysis Complete!")
        print("="*70)
        print(f"\n[OK] Repository: {repo_name}")
        print(f"[OK] Repository Path: {repo_path}")
        print(f"[OK] Reports Saved: {report_dir}")
        print(f"\nAll analysis reports and metrics have been saved to:")
        print(f"  {report_dir.absolute()}")
        print(f"\nYou can now manually explore the repository at: {repo_path}")
        print("="*70)

    except KeyboardInterrupt:
        print("\n\n[WARNING] Analysis interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease ensure:")
        print("1. Git is installed and accessible")
        print("2. You have a .env file with AZURE_OPENAI_* variables")
        print("3. Your Azure OpenAI API key is valid")
        print("4. You have internet connectivity")
        sys.exit(1)


if __name__ == "__main__":
    main()
