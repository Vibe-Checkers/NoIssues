#!/usr/bin/env python3
"""
Run Agent Script
Full workflow script that clones GitHub repositories and performs analysis.
Use this for complete repository analysis tasks.
"""

import os
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


def detect_project_language(repo_path: str) -> str:
    """
    Detect the primary programming language of a repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Detected language name (e.g., "Python", "JavaScript", "Go", "Rust")
    """
    import os
    
    # Common file patterns that indicate language
    language_indicators = {
        "Python": ["*.py", "requirements.txt", "setup.py", "pyproject.toml", "Pipfile"],
        "JavaScript": ["package.json", "*.js", "*.ts", "yarn.lock", "package-lock.json"],
        "TypeScript": ["tsconfig.json", "*.ts", "*.tsx"],
        "Go": ["go.mod", "go.sum", "*.go"],
        "Rust": ["Cargo.toml", "Cargo.lock", "*.rs"],
        "Java": ["pom.xml", "build.gradle", "*.java"],
        "C++": ["CMakeLists.txt", "Makefile", "*.cpp", "*.hpp"],
        "C": ["Makefile", "*.c", "*.h"],
        "Ruby": ["Gemfile", "*.rb", "Rakefile"],
        "PHP": ["composer.json", "*.php"],
        "C#": ["*.csproj", "*.sln", "*.cs"],
        "Swift": ["Package.swift", "*.swift"],
        "Kotlin": ["build.gradle.kts", "*.kt"],
    }
    
    # Count matches for each language
    language_scores = {}
    
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            for lang, patterns in language_indicators.items():
                for pattern in patterns:
                    if file == pattern or file.endswith(pattern.replace('*', '')):
                        language_scores[lang] = language_scores.get(lang, 0) + 1
                        break
    
    # Return the language with highest score, or "Unknown" if none found
    if language_scores:
        detected = max(language_scores.items(), key=lambda x: x[1])[0]
        # Special case: TypeScript often comes with JavaScript
        if detected == "TypeScript" and "JavaScript" in language_scores:
            return "TypeScript"
        return detected
    
    return "Unknown"


def _has_dockerfile(output: str) -> bool:
    """Check if output contains a Dockerfile (has FROM statement)."""
    if not output or not output.strip():
        return False
    return "FROM" in output.upper()


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
        
        # Set global report directory for web search to save files (if provided)
        if report_dir:
            import planner_agent
            planner_agent.REPORT_DIRECTORY = str(report_dir)
            print(f"[INFO] Report directory: {report_dir}")

        # Define analysis queries using discovery-based approach with relative paths
        queries = [
        "Analyze the repository. **FIRST**: Search the web for official documentation using the repository name and language. Then show me the directory tree structure (depth 2), and identify what type of project this is by finding and examining configuration files.",

        "Based on what you discovered in the previous step, find all build-related configuration files and extract the key information like dependencies, build scripts, runtime requirements, and environment variables. Also cross-reference with the official documentation you found earlier. Pay special attention to: version requirements, system dependencies, build tools needed, and any special environment setup.",

        "Based on everything you've learned so far, read the README file and extract installation/build instructions. Also search for any 'install', 'build', 'run', or 'start' commands mentioned in configuration files or scripts. Identify: entry points, default ports, required environment variables, volume mounts, and any runtime configuration. Compare with official documentation.",

            """Based on the information you gathered in Query 1, 2, and 3, create a Dockerfile AND a .dockerignore file.

**IMPORTANT**: You already have all the information needed from the previous 3 queries:
- Query 1: Project structure, type, and configuration files
- Query 2: Dependencies, build scripts, and requirements
- Query 3: Installation instructions, entry points, and environment setup

**YOUR TASK**: Generate the Dockerfile and .dockerignore using the information from previous queries.

**OPTIONAL**: You MAY use DockerImageSearch ONLY if you need to verify a base image tag exists. Otherwise, provide the Final Answer immediately.

**OUTPUT FORMAT** (Your Final Answer MUST use this EXACT format):

DOCKERFILE_START
<Dockerfile content>
DOCKERFILE_END

DOCKERIGNORE_START
<.dockerignore content>
DOCKERIGNORE_END

**DOCKERFILE REQUIREMENTS**:
- Use appropriate base image for the detected language/framework
- Install dependencies (npm install, pip install, cargo build, etc.)
- Set WORKDIR /app
- COPY application files
- EXPOSE the correct port (from previous queries)
- Define CMD or ENTRYPOINT to start the application

**DOCKERIGNORE REQUIREMENTS** (MUST include):
- .git (CRITICAL - always exclude)
- Language-specific: node_modules, __pycache__, *.pyc, target, .venv
- Build artifacts: build, dist, .next, out
- Environment: .env, .env.local
- Version control: .git, .gitignore, .github

Now generate the Final Answer with both files in the specified format.
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
                max_iterations_for_final = 5  # Allow up to 5 tool calls for Docker image verification
                
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
                    
                    # Try to parse with new format
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
                    
                    # Fallback: Check if output contains Dockerfile directly (old behavior)
                    if not dockerfile_content and _has_dockerfile(output):
                        dockerfile_content = output
                    
                    # Smart retry logic if Dockerfile is missing
                    max_retries = 2  # Reduced from 3 to 2 for efficiency
                    for retry_attempt in range(max_retries):
                        if dockerfile_content and _has_dockerfile(dockerfile_content):
                            break
                        
                        # Detect if agent is stuck doing tool calls instead of providing answer
                        tool_calls_in_result = len(result.get('intermediate_steps', []))
                        is_stuck_searching = tool_calls_in_result >= max_iterations_for_final
                        
                        if retry_attempt == 0:
                            if is_stuck_searching:
                                print(f"\n[WARNING] Agent hit iteration limit ({tool_calls_in_result} tool calls) without providing Dockerfile.")
                                print(f"[INFO] Forcing direct answer with NO tool calls allowed...")
                            else:
                                print(f"\n[WARNING] Output doesn't contain valid Dockerfile. Retrying with strict format prompt...")
                        else:
                            print(f"\n[WARNING] Retry {retry_attempt + 1}/{max_retries} still didn't produce valid Dockerfile. Retrying again...")
                        
                        # Build retry query with progressively stricter instructions
                        if is_stuck_searching or retry_attempt > 0:
                            # Force immediate answer - NO tool calls
                            retry_query = f"""CRITICAL: You MUST provide the Final Answer NOW using the information already available from previous queries (Query 1, 2, 3).

DO NOT use any tools. DO NOT search. DO NOT read files. Just generate the Dockerfile and .dockerignore using the context you already have.

PREVIOUS CONTEXT SUMMARY:
{formatted_history[-1000:] if formatted_history else "See chat history"}

OUTPUT FORMAT (REQUIRED):
DOCKERFILE_START
FROM <base-image>
WORKDIR /app
COPY . .
RUN <install-commands>
EXPOSE <port>
CMD <start-command>
DOCKERFILE_END

DOCKERIGNORE_START
.git
node_modules
__pycache__
*.pyc
.env
DOCKERIGNORE_END

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
                    
                    if not dockerfile_content or not _has_dockerfile(dockerfile_content):
                        print(f"\n[ERROR] Failed to generate Dockerfile after {max_retries + 1} attempts")
                    else:
                        print(f"\n[SUCCESS] Dockerfile generated successfully!")
                    
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
        
        # Set global report directory for web search to save files
        import planner_agent
        planner_agent.REPORT_DIRECTORY = str(report_dir)
        
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
