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

from planner_agent import create_planner_agent, REPORT_DIRECTORY


def clone_repository(repo_url: str, target_dir: str = "./temp", auto_remove: bool = True) -> str:
    """
    Clone a GitHub repository to a local directory.

    Args:
        repo_url: GitHub repository URL
        target_dir: Directory to clone repositories into
        auto_remove: If True, automatically remove existing repo and re-clone (default: True for automation)

    Returns:
        Path to the cloned repository
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

    # Clone the repository
    try:
        print(f"[CLONING] Cloning to {clone_path}...")
        result = subprocess.run(
            ["git", "clone", repo_url, clone_path],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[SUCCESS] Repository cloned successfully!")
        return clone_path

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error cloning repository: {e.stderr}")
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
    report_dir: Path = None
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


class TeeOutput:
    """Helper class to tee output to both terminal and log file"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, text):
        for f in self.files:
            f.write(text)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


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
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        # Redirect stdout and stderr to both terminal and log file
        sys.stdout = TeeOutput(original_stdout, log_file_handle)
        sys.stderr = TeeOutput(original_stderr, log_file_handle)
        
        # Reconfigure logging to use redirected stderr so INFO logs are captured
        import logging
        # Remove all existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Create a new StreamHandler that writes to current stderr (which is now redirected)
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

            """Now create a comprehensive, production-ready Dockerfile that includes ALL of the following elements (DO NOT SKIP ANY):

1. **Base Image**: Choose the appropriate official base image (e.g., python:3.11-slim, node:18-alpine, etc.) based on the project's language and version requirements. Use specific version tags, not 'latest'.

2. **Metadata**: Add LABEL instructions for maintainer and description.

3. **Working Directory**: Set WORKDIR to an appropriate directory (typically /app or /usr/src/app).

4. **System Dependencies**: Install ALL required system packages, build tools, and libraries needed for the project. Include packages for: compiling native extensions, SSL/certificates, database clients, image processing libraries, or any other system-level dependencies mentioned in documentation.

5. **Environment Variables**: Set any required environment variables (e.g., PYTHONUNBUFFERED=1, NODE_ENV=production, etc.).

6. **Dependency Files**: Copy dependency files (requirements.txt, package.json, Pipfile, Cargo.toml, go.mod, etc.) BEFORE copying source code to leverage Docker layer caching.

7. **Install Dependencies**: Run the appropriate install commands (pip install, npm install, cargo build, go mod download, etc.) with production flags where applicable. For Python: use --no-cache-dir. For Node.js: use --production or --omit=dev if appropriate.

8. **Source Code**: Copy the application source code to the container. Use .dockerignore patterns if needed, but copy all necessary files.

9. **Build Steps**: If the project requires compilation or building, include ALL build commands (npm run build, python setup.py build, cargo build --release, etc.). Ensure build artifacts are properly placed.

10. **User Permissions**: Create a non-root user and switch to it for security (if applicable). Set proper ownership of files.

11. **Expose Ports**: EXPOSE the port(s) the application uses. Check documentation, configuration files, or default ports for the framework.

12. **Health Check**: Add HEALTHCHECK instruction if applicable (especially for web services).

13. **Entry Point**: Set CMD or ENTRYPOINT to run the application. Use the exact command from documentation or configuration files. Include all required arguments.

14. **Multi-stage Build** (if beneficial): Consider using multi-stage builds to reduce final image size for compiled languages.

15. **Security Best Practices**: 
    - Use specific version tags for base images
    - Run as non-root user when possible
    - Minimize layers and clean up temporary files
    - Use .dockerignore to exclude unnecessary files

IMPORTANT: 
- Include ALL dependencies mentioned in requirements files, package.json, or documentation
- Include ALL build steps - do not skip any compilation or build processes
- Include ALL environment variables that are required
- Use the exact versions specified in the project's dependency files
- If the project has multiple entry points or services, create a Dockerfile for the main service
- If there are database migrations, include them in the appropriate step
- If there are static files to collect or assets to build, include those steps
- Reference the official documentation you found earlier to ensure best practices

Output ONLY the Dockerfile content, starting with FROM and ending with CMD/ENTRYPOINT. Do not include markdown formatting, explanations, or additional text - just the raw Dockerfile content that can be saved directly to a file."""
        ]

        # Initialize conversation history to maintain context across queries
        chat_history = []
        final_instructions = None
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

                # Invoke agent with accumulated chat history for context continuity
                result = agent.invoke({
                    "input": query,
                    "chat_history": formatted_history or "No previous context."
                })

                output = result['output']
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

                # Save the final step output (Dockerfile)
                if i == len(queries):
                    final_instructions = output

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
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        
        # Restore logging handlers to original stderr
        import logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        # Reconfigure with original stderr
        handler = logging.StreamHandler(original_stderr)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
        
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
            log_file_path=log_file_path
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
            report_dir=report_dir
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
