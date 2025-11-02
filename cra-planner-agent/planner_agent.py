#!/usr/bin/env python3
"""
Planner Agent Module
Core agent functionality without direct execution or GitHub API dependencies.
Import this module to use the agent in other scripts.
"""

import os
import json
import logging
from typing import Any, List, Optional, Dict
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs from httpx and OpenAI client
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

    def on_agent_action(self, action, **kwargs):
        """Called when agent takes an action."""
        print(f"\n{'─'*70}")
        print(f"[THOUGHT] {action.log.split('Action:')[0].strip() if 'Action:' in action.log else action.log.strip()}")
        print(f"\n[ACTION] {action.tool}")
        print(f"[INPUT] {action.tool_input}")
        print(f"{'─'*70}")

    def on_tool_end(self, output, **kwargs):
        """Called when tool finishes."""
        output_preview = str(output)[:200] + "..." if len(str(output)) > 200 else str(output)
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
# LLM Wrapper for gpt-5-nano
# ============================================================================

class GPT5NanoWrapper(BaseChatModel):
    """Wrapper for gpt-5-nano that strips unsupported parameters."""

    llm: AzureChatOpenAI

    class Config:
        arbitrary_types_allowed = True

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        # Remove unsupported parameters
        kwargs.pop('stop', None)
        # Call the underlying LLM without stop parameter
        return self.llm._generate(messages, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "gpt-5-nano-wrapper"

    def bind_tools(self, tools, **kwargs):
        return self.llm.bind_tools(tools, **kwargs)


# ============================================================================
# Path Resolution Infrastructure
# ============================================================================

# Global variable to track the current repository base path
REPOSITORY_BASE_PATH = None


def _resolve_path(user_path: str) -> str:
    """
    Convert user-provided paths to absolute paths using repository base.

    Args:
        user_path: Path provided by the user (can be relative or absolute)

    Returns:
        Absolute path resolved against repository base if set
    """
    if REPOSITORY_BASE_PATH is None:
        # No repository context, use path as-is
        return user_path

    # If user provides absolute path, use it directly
    if os.path.isabs(user_path):
        return user_path

    # Special case: "." refers to the repository base itself
    if user_path == ".":
        return REPOSITORY_BASE_PATH

    # Otherwise, resolve relative to repository base
    return os.path.join(REPOSITORY_BASE_PATH, user_path)


# ============================================================================
# Tool Functions
# ============================================================================

def search_web(query: str) -> str:
    """
    Search the web using DuckDuckGo.

    Args:
        query: Search query string

    Returns:
        Search results as a string
    """
    try:
        logger.info(f"Searching the web for: {query}")
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        logger.info(f"Search completed, {len(results)} chars returned")
        return results
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search error: {str(e)}"


def read_local_file(filepath: str) -> str:
    """
    Reads a file from the local filesystem.

    Args:
        filepath: Path to the file to read

    Returns:
        File contents or error message
    """
    try:
        resolved_path = _resolve_path(filepath)
        logger.info(f"Reading local file: {resolved_path}")
        with open(resolved_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"Successfully read {resolved_path} ({len(content)} chars)")
        return content
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}")
        return f"Error reading file: {str(e)}"


def grep_files(input_str: str) -> str:
    """
    Search for patterns in files using regex (like grep command).

    Args:
        input_str: Format "directory,pattern,file_glob,context_lines"
                   - directory: Directory to search in
                   - pattern: Regex pattern to search for
                   - file_glob: File pattern to match (e.g., *.py, *.json, *)
                   - context_lines: Number of context lines before/after match (default: 0)

                   Examples:
                   - ".,dependencies,*.json,2" - Find "dependencies" in JSON files with 2 lines context
                   - ".,^import|^from,*.py,0" - Find Python imports
                   - ".,build,*,1" - Find "build" in all files with 1 line context

    Returns:
        Matching lines with file paths, line numbers, and context
    """
    try:
        import re
        import fnmatch

        # Parse input
        parts = input_str.split(",")
        directory = parts[0].strip() if len(parts) > 0 else "."
        pattern = parts[1].strip() if len(parts) > 1 else ""
        file_glob = parts[2].strip() if len(parts) > 2 else "*"
        context_lines = int(parts[3].strip()) if len(parts) > 3 else 0

        if not pattern:
            return "Error: Pattern is required"

        # Resolve directory path
        directory = _resolve_path(directory)
        logger.info(f"Grepping for '{pattern}' in {directory}/{file_glob}")

        if not os.path.exists(directory):
            return f"Error: Directory not found: {directory}"

        # Compile regex
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        matches = []
        total_files_searched = 0

        # Walk directory
        for root, dirs, files in os.walk(directory):
            # Filter hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            for filename in files:
                # Skip hidden files
                if filename.startswith('.'):
                    continue

                # Check if filename matches glob
                if not fnmatch.fnmatch(filename, file_glob):
                    continue

                filepath = os.path.join(root, filename)
                total_files_searched += 1

                # Try to read file
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()

                    # Search for pattern
                    for i, line in enumerate(lines, 1):
                        if regex.search(line):
                            match_info = {
                                "file": filepath,
                                "line_number": i,
                                "line": line.rstrip(),
                                "context_before": [],
                                "context_after": []
                            }

                            # Add context lines if requested
                            if context_lines > 0:
                                start = max(0, i - context_lines - 1)
                                end = min(len(lines), i + context_lines)

                                match_info["context_before"] = [
                                    lines[j].rstrip() for j in range(start, i-1)
                                ]
                                match_info["context_after"] = [
                                    lines[j].rstrip() for j in range(i, end)
                                ]

                            matches.append(match_info)

                            # Limit results
                            if len(matches) >= 100:
                                break

                    if len(matches) >= 100:
                        break

                except (UnicodeDecodeError, PermissionError, IsADirectoryError):
                    continue

            if len(matches) >= 100:
                break

        result = {
            "pattern": pattern,
            "file_glob": file_glob,
            "directory": directory,
            "total_matches": len(matches),
            "files_searched": total_files_searched,
            "matches": matches[:50],  # Return max 50 matches
            "truncated": len(matches) > 50
        }

        logger.info(f"Found {len(matches)} matches in {total_files_searched} files")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error in grep_files: {e}")
        return f"Error: {str(e)}"


def find_files(input_str: str) -> str:
    """
    Find files matching name patterns (like Unix find command).

    Args:
        input_str: Format "directory,name_pattern,max_depth"
                   - directory: Directory to search in
                   - name_pattern: Glob pattern for filenames (e.g., *.py, package*.json, Makefile)
                   - max_depth: Maximum directory depth to search (default: 5)

                   Examples:
                   - ".,*requirements*.txt,5" - Find any requirements files
                   - ".,package*.json,3" - Find package.json files
                   - ".,Makefile,2" - Find Makefiles
                   - ".,*.py,10" - Find all Python files

    Returns:
        JSON list of matching file paths with metadata
    """
    try:
        import fnmatch

        # Parse input
        parts = input_str.split(",")
        directory = parts[0].strip() if len(parts) > 0 else "."
        name_pattern = parts[1].strip() if len(parts) > 1 else "*"
        max_depth = int(parts[2].strip()) if len(parts) > 2 else 5

        # Resolve directory path
        directory = _resolve_path(directory)
        logger.info(f"Finding files matching '{name_pattern}' in {directory} (depth {max_depth})")

        if not os.path.exists(directory):
            return json.dumps({"error": f"Directory not found: {directory}"})

        if not os.path.isdir(directory):
            return json.dumps({"error": f"Not a directory: {directory}"})

        matches = []
        base_depth = directory.rstrip(os.sep).count(os.sep)

        # Walk directory
        for root, dirs, files in os.walk(directory):
            # Calculate current depth
            current_depth = root.count(os.sep) - base_depth

            # Stop if max depth reached
            if current_depth >= max_depth:
                del dirs[:]
                continue

            # Filter hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]

            # Check files
            for filename in files:
                # Skip hidden files
                if filename.startswith('.'):
                    continue

                # Check if filename matches pattern
                if fnmatch.fnmatch(filename, name_pattern):
                    filepath = os.path.join(root, filename)
                    try:
                        stat = os.stat(filepath)
                        matches.append({
                            "path": filepath,
                            "name": filename,
                            "directory": root,
                            "size_bytes": stat.st_size,
                            "depth": current_depth
                        })
                    except OSError:
                        continue

                # Limit results
                if len(matches) >= 100:
                    break

            if len(matches) >= 100:
                break

        result = {
            "pattern": name_pattern,
            "directory": directory,
            "max_depth": max_depth,
            "total_found": len(matches),
            "files": matches[:50],  # Return max 50 files
            "truncated": len(matches) > 50
        }

        logger.info(f"Found {len(matches)} files matching pattern")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error in find_files: {e}")
        return json.dumps({"error": str(e)})


def extract_json_field(input_str: str) -> str:
    """
    Extract specific fields from JSON files using dot notation.

    Args:
        input_str: Format "filepath,json_path"
                   - filepath: Path to JSON file
                   - json_path: Dot-separated path to field (e.g., "scripts.build", "dependencies")

                   Examples:
                   - "package.json,scripts.build" - Get build script
                   - "package.json,dependencies" - Get all dependencies
                   - "composer.json,require.php" - Get PHP version requirement

    Returns:
        Extracted value as JSON string or error message
    """
    try:
        # Parse input
        parts = input_str.split(",", 1)
        if len(parts) < 2:
            return json.dumps({"error": "Input format must be 'filepath,json_path'"})

        filepath = parts[0].strip()
        json_path = parts[1].strip()

        # Resolve filepath
        filepath = _resolve_path(filepath)
        logger.info(f"Extracting {json_path} from {filepath}")

        if not os.path.exists(filepath):
            return json.dumps({"error": f"File not found: {filepath}"})

        # Read JSON file
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Navigate path
        current = data
        path_parts = json_path.split('.')

        for part in path_parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return json.dumps({"error": f"Path '{json_path}' not found in file"})

        result = {
            "file": filepath,
            "path": json_path,
            "value": current
        }

        logger.info(f"Successfully extracted field")
        return json.dumps(result, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON file: {str(e)}"})
    except Exception as e:
        logger.error(f"Error in extract_json_field: {e}")
        return json.dumps({"error": str(e)})


def get_file_metadata(filepath: str) -> str:
    """
    Get metadata and information about a file.

    Args:
        filepath: Path to the file

    Returns:
        JSON with file metadata: size, permissions, type info, encoding, line count
    """
    try:
        # Resolve filepath
        filepath = _resolve_path(filepath)
        logger.info(f"Getting metadata for {filepath}")

        if not os.path.exists(filepath):
            return json.dumps({"error": f"File not found: {filepath}"})

        if not os.path.isfile(filepath):
            return json.dumps({"error": f"Not a file: {filepath}"})

        stat = os.stat(filepath)

        # Determine if file is binary
        is_binary = False
        encoding = "unknown"
        line_count = 0

        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(1024)
                is_binary = b'\x00' in chunk

            if not is_binary:
                # Try to determine encoding and count lines
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    line_count = sum(1 for _ in f)
                encoding = "utf-8"
        except Exception:
            pass

        result = {
            "path": filepath,
            "name": os.path.basename(filepath),
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 2),
            "is_executable": os.access(filepath, os.X_OK),
            "is_binary": is_binary,
            "encoding": encoding,
            "line_count": line_count if not is_binary else None,
            "extension": os.path.splitext(filepath)[1],
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }

        logger.info(f"Metadata retrieved successfully")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error in get_file_metadata: {e}")
        return json.dumps({"error": str(e)})


def list_directory(directory: str) -> str:
    """
    Lists contents of a directory.

    Args:
        directory: Path to the directory

    Returns:
        JSON string with directory contents
    """
    try:
        import os
        # Resolve directory path
        directory = _resolve_path(directory)
        logger.info(f"Listing directory: {directory}")

        if not os.path.exists(directory):
            return json.dumps({"error": f"Directory not found: {directory}"})

        if not os.path.isdir(directory):
            return json.dumps({"error": f"Not a directory: {directory}"})

        contents = []
        for item in os.listdir(directory):
            full_path = os.path.join(directory, item)
            item_type = "dir" if os.path.isdir(full_path) else "file"
            size = os.path.getsize(full_path) if os.path.isfile(full_path) else None
            contents.append({
                "name": item,
                "type": item_type,
                "size_bytes": size
            })

        # Sort: directories first, then files
        contents.sort(key=lambda x: (x["type"] == "file", x["name"]))

        result = {
            "directory": directory,
            "contents": contents,
            "total_items": len(contents),
            "files": len([c for c in contents if c["type"] == "file"]),
            "directories": len([c for c in contents if c["type"] == "dir"])
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Error listing directory: {e}")
        return json.dumps({"error": str(e)})


def create_directory_tree(input_str: str) -> str:
    """
    Creates a visual directory tree structure.

    Args:
        input_str: Format "directory_path,max_depth,show_hidden,show_files"
                   Example: "./myproject,3,false,true"
                   - directory_path: Path to the directory
                   - max_depth: Maximum depth to traverse (default: 3)
                   - show_hidden: Show hidden files/dirs starting with . (default: false)
                   - show_files: Show files in addition to directories (default: true)

    Returns:
        Tree structure as string
    """
    try:
        import os

        # Parse input
        parts = input_str.split(",")
        directory = parts[0].strip() if len(parts) > 0 else "."
        max_depth = int(parts[1].strip()) if len(parts) > 1 else 3
        show_hidden = parts[2].strip().lower() == "true" if len(parts) > 2 else False
        show_files = parts[3].strip().lower() == "true" if len(parts) > 3 else True

        # Resolve directory path
        directory = _resolve_path(directory)
        logger.info(f"Creating tree for {directory} (depth={max_depth}, hidden={show_hidden}, files={show_files})")

        if not os.path.exists(directory):
            return f"Error: Directory not found: {directory}"

        if not os.path.isdir(directory):
            return f"Error: Not a directory: {directory}"

        def build_tree(dir_path: str, prefix: str = "", current_depth: int = 0) -> List[str]:
            """Recursively build tree structure."""
            if current_depth >= max_depth:
                return []

            lines = []
            try:
                # Get directory contents
                items = os.listdir(dir_path)

                # Filter hidden files if needed
                if not show_hidden:
                    items = [item for item in items if not item.startswith('.')]

                # Separate and sort directories and files
                dirs = sorted([item for item in items if os.path.isdir(os.path.join(dir_path, item))])
                files = sorted([item for item in items if os.path.isfile(os.path.join(dir_path, item))])

                # Combine based on show_files setting
                all_items = dirs + (files if show_files else [])

                for i, item in enumerate(all_items):
                    is_last = (i == len(all_items) - 1)
                    full_path = os.path.join(dir_path, item)
                    is_dir = os.path.isdir(full_path)

                    # Tree symbols
                    connector = "└── " if is_last else "├── "
                    extension = "    " if is_last else "│   "

                    # Add item
                    item_marker = "[DIR] " if is_dir else ""
                    lines.append(f"{prefix}{connector}{item_marker}{item}")

                    # Recurse into directories
                    if is_dir:
                        lines.extend(build_tree(full_path, prefix + extension, current_depth + 1))

            except PermissionError:
                lines.append(f"{prefix}└── [Permission Denied]")

            return lines

        # Build the tree
        tree_lines = [f"{os.path.basename(os.path.abspath(directory)) or directory}/"]
        tree_lines.extend(build_tree(directory))

        result = "\n".join(tree_lines)
        logger.info(f"Tree created with {len(tree_lines)} lines")

        return result

    except Exception as e:
        logger.error(f"Error creating directory tree: {e}")
        return f"Error creating directory tree: {str(e)}"


# ============================================================================
# Agent Creation
# ============================================================================

def create_planner_agent(
    max_iterations: int = 10,
    verbose: bool = True,
    repository_path: str = None
):
    """
    Create and return a planner agent configured with Azure OpenAI.

    Args:
        max_iterations: Maximum number of agent iterations
        verbose: Whether to print agent steps
        repository_path: Base path for repository operations (tools will resolve relative paths against this)

    Returns:
        Tuple of (AgentExecutor instance, FormattedOutputHandler for accessing token usage)
    """
    global REPOSITORY_BASE_PATH
    REPOSITORY_BASE_PATH = repository_path

    logger.info("Creating planner agent...")
    if repository_path:
        logger.info(f"Repository base path set to: {repository_path}")

    # Get configuration from environment
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

    if not all([api_key, endpoint, deployment]):
        raise ValueError("Missing required environment variables. Please check your .env file.")

    # Initialize Azure OpenAI
    base_llm = AzureChatOpenAI(
        azure_deployment=deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version
    )

    # Wrap the model to strip unsupported parameters
    llm = GPT5NanoWrapper(llm=base_llm)

    # Define tools
    tools = [
        Tool(
            name="SearchWeb",
            func=search_web,
            description="Search the web using DuckDuckGo for current information, documentation, or build tool guides. Use sparingly - prefer local analysis first. Input: search query string."
        ),
        Tool(
            name="ReadFile",
            func=read_local_file,
            description="Reads any file from the filesystem. Input: file path (e.g., './README.md', './package.json', './Makefile')."
        ),
        Tool(
            name="ListDirectory",
            func=list_directory,
            description="Lists files and directories with types and sizes. Input: directory path (e.g., '.', './src')."
        ),
        Tool(
            name="DirectoryTree",
            func=create_directory_tree,
            description="Creates visual tree of directory structure. Input: 'path,depth,show_hidden,show_files' (e.g., '.,3,false,true'). Good for understanding project layout."
        ),
        Tool(
            name="FindFiles",
            func=find_files,
            description="Find files matching name patterns (like Unix find). Input: 'directory,pattern,max_depth' (e.g., '.,package*.json,5', '.,*requirements*.txt,3', '.,Makefile,2'). Use to locate config files."
        ),
        Tool(
            name="GrepFiles",
            func=grep_files,
            description="Search for regex patterns in files (like grep). Input: 'directory,pattern,file_glob,context' (e.g., '.,dependencies,*.json,2', '.,^import,*.py,0', '.,build,*,1'). Find build commands, imports, etc."
        ),
        Tool(
            name="ExtractJsonField",
            func=extract_json_field,
            description="Extract fields from JSON files using dot notation. Input: 'filepath,json_path' (e.g., 'package.json,scripts.build', 'package.json,dependencies'). Quick way to get specific JSON values."
        ),
        Tool(
            name="GetFileMetadata",
            func=get_file_metadata,
            description="Get file metadata: size, type, encoding, line count, if executable. Input: file path. Helps determine if file is readable or binary."
        )
    ]

    # Custom ReAct prompt for reasoning models
    template = """You are analyzing a repository to understand its structure and create build instructions.

Available tools: {tool_names}

{tools}

ANALYSIS APPROACH - Discovery over Assumptions:
1. START BROAD: Use DirectoryTree to see overall structure
2. LOCATE FILES: Use FindFiles to locate config files (don't assume locations)
3. READ & EXTRACT: Use ReadFile and ExtractJsonField to examine configs
4. SEARCH PATTERNS: Use GrepFiles to find build commands, imports, requirements
5. CROSS-REFERENCE: Verify findings from multiple sources
6. WEB SEARCH: Only if critical info is missing from repository

CRITICAL FORMAT RULES (YOU MUST FOLLOW EXACTLY):
1. After "Action:", you MUST write "Action Input:" on the next line
2. After "Action Input:", STOP IMMEDIATELY - do not write anything else
3. Do NOT write "Observation:" - the system provides it
4. Do NOT write "Final Answer" until you have all observations
5. Each response: EITHER (Thought + Action + Action Input) OR (Thought + Final Answer), NEVER BOTH

CORRECT FORMAT EXAMPLES:

Example 1 - Using a tool:
Question: Show me the directory tree
Thought: I need to see the directory structure to understand the project layout.
Action: DirectoryTree
Action Input: .,2,false,true
[STOP HERE - wait for system to provide Observation]

Example 2 - Using another tool:
Question: Find package.json files
Thought: I should locate all package.json files in the repository.
Action: FindFiles
Action Input: .,package.json,5
[STOP HERE - wait for system to provide Observation]

Example 3 - Providing final answer:
Thought: I have gathered all the necessary information from the files I examined.
Final Answer: This is a Node.js project with TypeScript. Prerequisites: Node.js 18+. Installation: npm install. Build: npm run build.
[STOP HERE - task complete]

WRONG FORMAT (DO NOT DO THIS):
Thought: I will check the directory
Action: DirectoryTree
Action Input: .,2,false,true
Observation: [some result]  ← WRONG! Do not write Observation!
Final Answer: ...           ← WRONG! Do not put Final Answer after an Action!

Now begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # Create agent executor with custom callback for better formatting
    callback_handler = FormattedOutputHandler() if verbose else None
    callbacks = [callback_handler] if callback_handler else []

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # Disable default verbose output
        callbacks=callbacks,
        handle_parsing_errors=True,
        max_iterations=max_iterations,
        return_intermediate_steps=True  # Enable to track tool usage
    )

    logger.info("Planner agent created successfully")
    return agent_executor, callback_handler
