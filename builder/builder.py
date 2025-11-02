#!/usr/bin/env python3
"""
Builder Agent Module
Executes build instructions by running terminal commands and managing the build process.
Works in conjunction with the planner agent to build projects.
"""

import os
import json
import logging
import subprocess
import shutil
from typing import Any, List, Optional, Dict
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ============================================================================
# Custom Callback Handler for Better Formatting
# ============================================================================

class FormattedOutputHandler(BaseCallbackHandler):
    """Custom callback handler to format agent output with proper spacing and track token usage."""

    def __init__(self):
        super().__init__()
        self.token_usage = {"input": 0, "output": 0, "total": 0}
        self.commands_executed = []

    def on_agent_action(self, action, **kwargs):
        """Called when agent takes an action."""
        print(f"\n{'─'*70}")
        print(f"[THOUGHT] {action.log.split('Action:')[0].strip() if 'Action:' in action.log else action.log.strip()}")
        print(f"\n[ACTION] {action.tool}")
        print(f"[INPUT] {action.tool_input}")
        print(f"{'─'*70}")

    def on_tool_end(self, output, **kwargs):
        """Called when tool finishes."""
        output_str = str(output)
        
        # For command execution, show more output (up to 3000 chars)
        # For other tools, keep it shorter
        if '"command":' in output_str or '"success":' in output_str:
            # This is likely a command execution result
            max_length = 3000
        else:
            max_length = 1000
        
        if len(output_str) > max_length:
            # Show first and last parts
            half = max_length // 2
            output_preview = output_str[:half] + f"\n\n... [OUTPUT TRUNCATED - {len(output_str)} total chars] ...\n\n" + output_str[-half:]
        else:
            output_preview = output_str
        
        print(f"\n[OBSERVATION] {output_preview}\n")

    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes - capture token usage."""
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            if usage:
                self.token_usage["input"] += usage.get('prompt_tokens', 0)
                self.token_usage["output"] += usage.get('completion_tokens', 0)
                self.token_usage["total"] += usage.get('total_tokens', 0)


# ============================================================================
# Path Resolution Infrastructure
# ============================================================================

# Global variable to track the current working directory for build operations
BUILD_WORKING_DIRECTORY = None


def _resolve_path(user_path: str) -> str:
    """
    Convert user-provided paths to absolute paths using build working directory.

    Args:
        user_path: Path provided by the user (can be relative or absolute)

    Returns:
        Absolute path resolved against build working directory if set
    """
    if BUILD_WORKING_DIRECTORY is None:
        return user_path

    # If user provides absolute path, use it directly
    if os.path.isabs(user_path):
        return user_path

    # Special case: "." refers to the build working directory itself
    if user_path == ".":
        return BUILD_WORKING_DIRECTORY

    # Otherwise, resolve relative to build working directory
    return os.path.join(BUILD_WORKING_DIRECTORY, user_path)


# ============================================================================
# Build Execution State
# ============================================================================

class BuildState:
    """Track the state of the build process."""
    
    def __init__(self, working_dir: str):
        self.working_dir = working_dir
        self.commands_executed = []
        self.commands_failed = []
        self.start_time = datetime.now()
        # Use full environment including PATH
        self.env_vars = os.environ.copy()
        # Ensure common paths are included
        if 'PATH' in self.env_vars:
            # Add common binary locations if not already present
            common_paths = ['/usr/local/bin', '/usr/bin', '/bin', '/usr/local/sbin', '/usr/sbin', '/sbin']
            current_paths = self.env_vars['PATH'].split(':')
            for path in common_paths:
                if path not in current_paths:
                    current_paths.append(path)
            self.env_vars['PATH'] = ':'.join(current_paths)
        
    def add_command(self, command: str, success: bool, output: str, return_code: int):
        """Record a command execution."""
        record = {
            "command": command,
            "success": success,
            "output": output,
            "return_code": return_code,
            "timestamp": datetime.now().isoformat()
        }
        
        if success:
            self.commands_executed.append(record)
        else:
            self.commands_failed.append(record)
            
    def get_summary(self) -> str:
        """Get a summary of the build process."""
        duration = (datetime.now() - self.start_time).total_seconds()
        return json.dumps({
            "working_directory": self.working_dir,
            "duration_seconds": round(duration, 2),
            "commands_succeeded": len(self.commands_executed),
            "commands_failed": len(self.commands_failed),
            "total_commands": len(self.commands_executed) + len(self.commands_failed)
        }, indent=2)


# Global build state
BUILD_STATE = None


# ============================================================================
# Tool Functions
# ============================================================================

def execute_command(command: str) -> str:
    """
    Execute a shell command in the build working directory.
    
    Args:
        command: Shell command to execute (e.g., "npm install", "cargo build")
        
    Returns:
        JSON with command output, return code, and success status
    """
    global BUILD_STATE
    
    try:
        working_dir = BUILD_WORKING_DIRECTORY or os.getcwd()
        logger.info(f"Executing command in {working_dir}: {command}")
        
        # Debug: Log PATH being used (only for first few commands)
        if BUILD_STATE and len(BUILD_STATE.commands_executed) < 2:
            logger.info(f"PATH being used: {BUILD_STATE.env_vars.get('PATH', 'NOT SET')}")
        
        # Security: Basic validation to prevent dangerous commands and sudo
        dangerous_patterns = ['rm -rf /', 'dd if=', 'mkfs', 'format', ':(){:|:&};:']
        if any(pattern in command.lower() for pattern in dangerous_patterns):
            error_msg = f"Command rejected for safety: {command}"
            logger.warning(error_msg)
            if BUILD_STATE:
                BUILD_STATE.add_command(command, False, error_msg, -1)
            return json.dumps({
                "success": False,
                "command": command,
                "error": "Command rejected for safety reasons",
                "return_code": -1
            }, indent=2)
        
        # Block sudo commands - they require password and will hang
        if command.strip().startswith('sudo '):
            error_msg = f"sudo commands are not allowed - they require password input. If a tool is missing, report the error instead of trying to install it."
            logger.warning(error_msg)
            if BUILD_STATE:
                BUILD_STATE.add_command(command, False, error_msg, -1)
            return json.dumps({
                "success": False,
                "command": command,
                "error": "sudo commands are blocked - cannot run interactive commands that require password",
                "return_code": -1
            }, indent=2)
        
        # Execute command
        result = subprocess.run(
            command,
            shell=True,
            executable='/bin/bash',  # Force bash instead of default /bin/sh
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            env=BUILD_STATE.env_vars if BUILD_STATE else None
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        # Record in build state
        if BUILD_STATE:
            BUILD_STATE.add_command(command, success, output, result.returncode)
        
        # Log detailed error for debugging
        if not success:
            logger.error(f"Command failed: {command}")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"Stdout: {result.stdout}")
            logger.error(f"Stderr: {result.stderr}")
        
        # Truncate very long output (only if extremely large)
        if len(output) > 50000:
            output = output[:25000] + "\n\n... [OUTPUT TRUNCATED] ...\n\n" + output[-25000:]
        
        response = {
            "success": success,
            "command": command,
            "return_code": result.returncode,
            "output": output,
            "working_directory": working_dir
        }
        
        logger.info(f"Command {'succeeded' if success else 'failed'} with return code {result.returncode}")
        return json.dumps(response, indent=2)
        
    except subprocess.TimeoutExpired:
        error_msg = f"Command timed out after 600 seconds: {command}"
        logger.error(error_msg)
        if BUILD_STATE:
            BUILD_STATE.add_command(command, False, error_msg, -1)
        return json.dumps({
            "success": False,
            "command": command,
            "error": "Command timed out after 600 seconds",
            "return_code": -1
        }, indent=2)
    except Exception as e:
        error_msg = f"Error executing command: {str(e)}"
        logger.error(error_msg)
        if BUILD_STATE:
            BUILD_STATE.add_command(command, False, error_msg, -1)
        return json.dumps({
            "success": False,
            "command": command,
            "error": str(e),
            "return_code": -1
        }, indent=2)


def set_environment_variable(input_str: str) -> str:
    """
    Set an environment variable for subsequent commands.
    
    Args:
        input_str: Format "KEY=VALUE" or "KEY=VALUE,KEY2=VALUE2"
                   Example: "NODE_ENV=production" or "CC=gcc,CXX=g++"
        
    Returns:
        JSON with success status
    """
    global BUILD_STATE
    
    try:
        if not BUILD_STATE:
            return json.dumps({
                "success": False,
                "error": "Build state not initialized"
            }, indent=2)
        
        # Parse environment variables
        env_pairs = []
        for pair in input_str.split(','):
            if '=' not in pair:
                return json.dumps({
                    "success": False,
                    "error": f"Invalid format: {pair}. Use KEY=VALUE"
                }, indent=2)
            
            key, value = pair.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            BUILD_STATE.env_vars[key] = value
            env_pairs.append({"key": key, "value": value})
            logger.info(f"Set environment variable: {key}={value}")
        
        return json.dumps({
            "success": True,
            "variables_set": env_pairs
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error setting environment variable: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


def get_build_status() -> str:
    """
    Get current build status and summary.
    
    Returns:
        JSON with build statistics and command history
    """
    global BUILD_STATE
    
    if not BUILD_STATE:
        return json.dumps({
            "error": "Build state not initialized"
        }, indent=2)
    
    return BUILD_STATE.get_summary()


# ============================================================================
# Agent Creation
# ============================================================================

def create_builder_agent(
    max_iterations: int = 20,
    verbose: bool = True,
    working_directory: str = None
):
    """
    Create and return a builder agent configured with Azure OpenAI.
    
    Args:
        max_iterations: Maximum number of agent iterations
        verbose: Whether to print agent steps
        working_directory: Working directory for build operations
        
    Returns:
        Tuple of (AgentExecutor instance, FormattedOutputHandler for accessing token usage)
    """
    global BUILD_WORKING_DIRECTORY, BUILD_STATE
    
    # Set working directory
    if working_directory:
        BUILD_WORKING_DIRECTORY = os.path.abspath(working_directory)
        BUILD_STATE = BuildState(BUILD_WORKING_DIRECTORY)
        logger.info(f"Build working directory set to: {BUILD_WORKING_DIRECTORY}")
    else:
        BUILD_WORKING_DIRECTORY = os.getcwd()
        BUILD_STATE = BuildState(BUILD_WORKING_DIRECTORY)
        logger.info(f"Using current directory: {BUILD_WORKING_DIRECTORY}")
    
    logger.info("Creating builder agent...")
    
    # Get configuration from environment
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY environment variable. Please check your .env file.")
    
    # Initialize OpenAI
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0
    )
    
    # Define tools - minimal set for build execution
    tools = [
        Tool(
            name="ExecuteCommand",
            func=execute_command,
            description="Execute a shell command in the working directory. Input: command string (e.g., 'npm install', 'cargo build --release', 'make', 'python setup.py install'). Returns output, return code, and success status. Use this to run all build commands, install dependencies, compile code, run tests, etc."
        ),
        Tool(
            name="SetEnvironmentVariable",
            func=set_environment_variable,
            description="Set environment variables for subsequent commands. Input: 'KEY=VALUE' or 'KEY=VALUE,KEY2=VALUE2' (comma-separated for multiple). Example: 'NODE_ENV=production' or 'CC=gcc,CXX=g++'. Useful for configuring build environment."
        ),
        Tool(
            name="GetBuildStatus",
            func=get_build_status,
            description="Get current build status with command history and statistics. No input required. Shows commands executed, successes, failures, and timing information."
        )
    ]
    
    # Custom ReAct prompt for builder agent
    template = """You are a builder agent that executes build instructions to compile and build software projects.

IMPORTANT CONTEXT:
- The repository is ALREADY CLONED at: {working_directory}
- You are working inside the repository - DO NOT clone it again
- Skip any "clone repository" or "git clone" steps in the instructions
- Start directly with installation, dependency setup, or build commands

Available tools: {tool_names}

{tools}

BUILD EXECUTION APPROACH:
1. SKIP CLONING: The repository is already present, start with setup/install steps
2. SET ENVIRONMENT: Use SetEnvironmentVariable if build requires specific env vars
3. RUN COMMANDS: Use ExecuteCommand to run build steps (install, compile, test, etc.)
4. HANDLE ERRORS: If a command fails, analyze the error output and try to resolve
5. CHECK STATUS: Use GetBuildStatus to review progress when needed
6. REPORT: Provide clear feedback on success or failure of the build

CRITICAL FORMAT RULES (YOU MUST FOLLOW EXACTLY):
1. After "Action:", you MUST write "Action Input:" on the next line
2. After "Action Input:", write the input WITHOUT quotes
3. After "Action Input:", STOP IMMEDIATELY - do not write anything else
4. Do NOT write "Observation:" - the system provides it
5. Each response: EITHER (Thought + Action + Action Input) OR (Thought + Final Answer), NEVER BOTH

COMMAND EXECUTION GUIDELINES:
- The repository is already at {working_directory} - you're working inside it
- Run commands one at a time and check their output
- If a command fails with "command not found" (return code 127), try using 'which <command>' to locate it
- If 'which' finds the command, use the full path (e.g., '/usr/bin/node' instead of 'node')
- Common issues: missing dependencies, wrong environment variables, incorrect permissions
- You can use shell commands to check prerequisites (e.g., 'which npm', 'python --version')

CORRECT FORMAT EXAMPLES:

Example 1 - Checking prerequisites:
Thought: I should verify that Node.js is installed before proceeding.
Action: ExecuteCommand
Action Input: node --version

Example 2 - Installing dependencies:
Thought: Now I will install the dependencies using npm install.
Action: ExecuteCommand
Action Input: npm install

Example 3 - Setting environment variable:
Thought: The build requires NODE_ENV to be set to production.
Action: SetEnvironmentVariable
Action Input: NODE_ENV=production

Example 4 - Checking build status:
Thought: Let me check the build status to see how many commands have been executed.
Action: GetBuildStatus
Action Input: 

Example 5 - Providing final answer:
Thought: All build steps completed successfully. The project has been built.
Final Answer: Build completed successfully. Executed 5 commands: npm install, npm run build, npm test. All tests passed. Build artifacts are in the dist/ directory.

SAFETY NOTES:
- Dangerous commands (rm -rf /, format, etc.) are automatically blocked
- Commands have a 10-minute timeout
- All commands run in the specified working directory: {working_directory}

Now begin!

Build Instructions (NOTE: Repository is already cloned at {working_directory}, skip any clone/checkout steps):
{input}

Thought:{agent_scratchpad}"""
    
    prompt = PromptTemplate.from_template(template)
    
    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create agent executor with custom callback
    callback_handler = FormattedOutputHandler() if verbose else None
    callbacks = [callback_handler] if callback_handler else []
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # Disable default verbose output
        callbacks=callbacks,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
        max_execution_time=None,
        early_stopping_method="generate",
        return_intermediate_steps=True
    )
    
    logger.info("Builder agent created successfully")
    return agent_executor, callback_handler


# ============================================================================
# Main Execution Function
# ============================================================================

def execute_build(
    repository_path: str,
    instructions_file: str,
    github_url: str = None,
    max_iterations: int = 20,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Execute build instructions using the builder agent.
    
    Args:
        repository_path: Path where the repository should be/is located
        instructions_file: Path to markdown file containing build instructions
        github_url: Optional GitHub URL to clone. If provided, clones repo to repository_path
        max_iterations: Maximum number of agent iterations
        verbose: Whether to print detailed output
        
    Returns:
        Dictionary with build results, including success status, output, and statistics
    """
    try:
        # If GitHub URL provided, clone the repository first
        if github_url:
            logger.info(f"Cloning repository from {github_url} to {repository_path}")
            
            # Check if directory already exists
            if os.path.exists(repository_path):
                if os.path.isdir(repository_path) and os.listdir(repository_path):
                    # Directory exists and is not empty
                    logger.warning(f"Directory {repository_path} already exists and is not empty. Removing...")
                    shutil.rmtree(repository_path)
                elif os.path.isfile(repository_path):
                    raise ValueError(f"Path exists but is a file, not a directory: {repository_path}")
            
            # Create parent directory if it doesn't exist
            parent_dir = os.path.dirname(repository_path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            # Clone the repository
            clone_result = subprocess.run(
                ['git', 'clone', github_url, repository_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for cloning
            )
            
            if clone_result.returncode != 0:
                raise ValueError(f"Failed to clone repository: {clone_result.stderr}")
            
            logger.info(f"Successfully cloned repository to {repository_path}")
        
        # Validate paths
        if not os.path.exists(repository_path):
            raise ValueError(f"Repository path does not exist: {repository_path}")
        
        if not os.path.isdir(repository_path):
            raise ValueError(f"Repository path is not a directory: {repository_path}")
        
        if not os.path.exists(instructions_file):
            raise ValueError(f"Instructions file does not exist: {instructions_file}")
        
        # Read instructions from markdown file
        logger.info(f"Reading instructions from: {instructions_file}")
        with open(instructions_file, 'r', encoding='utf-8') as f:
            instructions = f.read()
        
        if not instructions.strip():
            raise ValueError(f"Instructions file is empty: {instructions_file}")
        
        logger.info(f"Starting build execution in {repository_path}")
        logger.info(f"Instructions length: {len(instructions)} characters")
        
        # Create agent
        agent_executor, callback_handler = create_builder_agent(
            max_iterations=max_iterations,
            verbose=verbose,
            working_directory=repository_path
        )
        
        # Execute build
        print("\n" + "="*70)
        print("BUILDER AGENT - Starting Execution")
        if github_url:
            print(f"Cloned from: {github_url}")
        print(f"Repository: {repository_path}")
        print(f"Instructions: {instructions_file}")
        print("="*70 + "\n")
        
        result = agent_executor.invoke({
            "input": instructions,
            "working_directory": repository_path
        })
        
        print("\n" + "="*70)
        print("BUILDER AGENT - Execution Complete")
        print("="*70 + "\n")
        
        # Get build summary
        build_summary = json.loads(get_build_status())
        
        # Compile results
        execution_result = {
            "success": True,
            "output": result.get("output", ""),
            "build_summary": build_summary,
            "token_usage": callback_handler.token_usage if callback_handler else {},
            "intermediate_steps": len(result.get("intermediate_steps", [])),
            "repository_path": repository_path,
            "instructions_file": instructions_file,
            "github_url": github_url
        }
        
        # Print summary
        print("\n[BUILD SUMMARY]")
        print(json.dumps(build_summary, indent=2))
        
        if callback_handler:
            print(f"\n[TOKEN USAGE]")
            print(f"Input tokens: {callback_handler.token_usage['input']}")
            print(f"Output tokens: {callback_handler.token_usage['output']}")
            print(f"Total tokens: {callback_handler.token_usage['total']}")
        
        logger.info("Build execution completed successfully")
        return execution_result
        
    except Exception as e:
        logger.error(f"Build execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "build_summary": json.loads(get_build_status()) if BUILD_STATE else {},
            "repository_path": repository_path if 'repository_path' in locals() else None,
            "instructions_file": instructions_file if 'instructions_file' in locals() else None
        }


if __name__ == "__main__":
    """
    Example usage when run directly.
    """
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python builder.py <repository_path> <instructions_file> [github_url]")
        print("\nArguments:")
        print("  repository_path    - Path where repository should be/is located")
        print("  instructions_file  - Path to markdown file with build instructions")
        print("  github_url         - (Optional) GitHub URL to clone before building")
        print("\nExamples:")
        print('  # Build existing repository:')
        print('  python builder.py /path/to/repo /path/to/instructions.md')
        print()
        print('  # Clone and build:')
        print('  python builder.py /path/to/repo /path/to/instructions.md https://github.com/user/repo.git')
        sys.exit(1)
    
    repo_path = sys.argv[1]
    instructions_path = sys.argv[2]
    github_url = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Execute build
    result = execute_build(
        repository_path=repo_path,
        instructions_file=instructions_path,
        github_url=github_url,
        verbose=True
    )
    
    # Exit with appropriate code
    if result["success"]:
        print("\n✓ Build completed successfully!")
        sys.exit(0)
    else:
        print(f"\n✗ Build failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)

