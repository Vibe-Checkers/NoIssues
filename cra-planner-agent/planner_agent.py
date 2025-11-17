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


def _make_relative_path(absolute_path: str) -> str:
    """
    Convert an absolute path to a relative path from the repository base.
    This is used in tool outputs so the agent sees relative paths.

    Args:
        absolute_path: Absolute path to convert

    Returns:
        Relative path from repository base, or original path if no base set
    """
    if REPOSITORY_BASE_PATH is None:
        return absolute_path

    try:
        # Get relative path from repository base
        rel_path = os.path.relpath(absolute_path, REPOSITORY_BASE_PATH)
        # If it doesn't go up directories, use it; otherwise return as-is
        if not rel_path.startswith('..'):
            return rel_path
        return absolute_path
    except (ValueError, TypeError):
        return absolute_path


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
    # Tools now return relative paths, so this works correctly
    return os.path.join(REPOSITORY_BASE_PATH, user_path)


# ============================================================================
# Tool Functions
# ============================================================================

def search_web(query: str) -> str:
    """
    Search the web using DuckDuckGo for official product documentation.
    This tool searches for real product documentation, not general web pages.

    Args:
        query: Search query string (should include "documentation" for best results)

    Returns:
        Search results as a formatted string with relevant documentation links
    """
    try:
        logger.info(f"Searching the web for: {query}")
        
        # Use duckduckgo_search directly for better reliability
        try:
            from ddgs import DDGS
            
            formatted_results = f"Documentation Search Results for: {query}\n"
            formatted_results += "=" * 70 + "\n\n"
            
            with DDGS() as ddgs:
                results_list = []
                # Search for documentation, prioritizing official sources
                for r in ddgs.text(query, max_results=8):
                    title = r.get('title', 'No title')
                    body = r.get('body', 'No description')
                    href = r.get('href', 'No URL')
                    
                    # Format each result
                    result_entry = f"Title: {title}\n"
                    result_entry += f"Description: {body[:200]}...\n" if len(body) > 200 else f"Description: {body}\n"
                    result_entry += f"URL: {href}\n"
                    result_entry += "-" * 70 + "\n"
                    results_list.append(result_entry)
                
                if results_list:
                    formatted_results += "\n".join(results_list)
                    formatted_results += f"\n[Found {len(results_list)} results. Focus on official documentation sources (official websites, GitHub, documentation sites).]"
                else:
                    formatted_results += "No results found. Try a more specific query."
            
            logger.info(f"Search completed, {len(formatted_results)} chars returned")
            return formatted_results
            
        except ImportError as e:
            error_msg = "Search dependencies not available. Please install: pip install -U ddgs"
            logger.error(f"{error_msg}. Error: {e}")
            return f"Search error: {error_msg}"
        except Exception as e:
            error_msg = f"Search failed: {str(e)}"
            logger.error(error_msg)
            return f"Search error: {error_msg}. Please try again or check your internet connection."
            
    except Exception as e:
        logger.error(f"Unexpected search error: {e}")
        return f"Search error: {str(e)}. Please ensure ddgs package is installed: pip install -U ddgs"


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
                                "file": _make_relative_path(filepath),
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
                            "path": _make_relative_path(filepath),
                            "name": filename,
                            "directory": _make_relative_path(root),
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
            "file": _make_relative_path(filepath),
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
            "path": _make_relative_path(filepath),
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


def fetch_web_page(url: str) -> str:
    """
    Fetch and extract text content from a web page URL.
    This tool actually visits the URL and retrieves the page content.

    Args:
        url: Full URL of the web page to fetch (e.g., "https://docs.python-requests.org/")

    Returns:
        Extracted text content from the web page, or error message
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        logger.info(f"Fetching web page: {url}")
        
        # Set headers to appear as a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Fetch the page
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Parse HTML and extract text
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Limit to first 5000 characters to avoid token limits
        if len(text) > 5000:
            text = text[:5000] + "\n\n[Content truncated - page is very long. Focus on the most relevant sections.]"
        
        logger.info(f"Successfully fetched {len(text)} chars from {url}")
        return f"Content from {url}:\n{'='*70}\n{text}"
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL {url}: {e}")
        return f"Error fetching web page: {str(e)}. The URL may be invalid or inaccessible."
    except ImportError:
        logger.error("BeautifulSoup4 not installed. Please install: pip install beautifulsoup4")
        return "Error: beautifulsoup4 package required. Install with: pip install beautifulsoup4"
    except Exception as e:
        logger.error(f"Unexpected error fetching web page: {e}")
        return f"Error fetching web page: {str(e)}"


# ============================================================================
# Agent Creation
# ============================================================================

def create_planner_agent(
    max_iterations: int = 10,
    verbose: bool = True,
    repository_path: str = None,
    repo_name: str = None,
    detected_language: str = None
):
    """
    Create and return a planner agent configured with Azure OpenAI.

    Args:
        max_iterations: Maximum number of agent iterations
        verbose: Whether to print agent steps
        repository_path: Base path for repository operations (tools will resolve relative paths against this)
        repo_name: Name of the repository (for documentation search context)
        detected_language: Detected programming language (for documentation search context)

    Returns:
        Tuple of (AgentExecutor instance, FormattedOutputHandler for accessing token usage)
    """
    global REPOSITORY_BASE_PATH
    REPOSITORY_BASE_PATH = repository_path

    logger.info("Creating planner agent...")
    if repository_path:
        logger.info(f"Repository base path set to: {repository_path}")
    if repo_name:
        logger.info(f"Repository name: {repo_name}")
    if detected_language:
        logger.info(f"Detected language: {detected_language}")

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
            description="**MANDATORY FIRST STEP**: Search the web for official product documentation, API references, and build guides. You MUST use this tool at the beginning of your analysis. Search format: '<repo_name> <language> documentation' or '<library_name> official documentation'. This tool searches for real product documentation from official sources. Input: search query string (e.g., 'requests Python documentation' or 'requests official documentation'). Returns search results with URLs - use FetchWebPage to get actual content."
        ),
        Tool(
            name="FetchWebPage",
            func=fetch_web_page,
            description="Fetch and extract text content from a web page URL. Use this after SearchWeb to get the actual page content. Input: full URL (e.g., 'https://docs.python-requests.org/')."
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

    # Build documentation search context
    doc_search_context = ""
    if repo_name and detected_language:
        doc_search_context = f"\n\nDOCUMENTATION SEARCH GUIDANCE:\n"
        doc_search_context += f"- Repository: {repo_name}\n"
        doc_search_context += f"- Detected Language: {detected_language}\n"
        doc_search_context += f"- Recommended search: '{repo_name} {detected_language} documentation' or '{repo_name} official documentation'\n"
        doc_search_context += f"- Search for official documentation early in your analysis to understand build requirements, dependencies, and setup procedures.\n"
    elif repo_name:
        doc_search_context = f"\n\nDOCUMENTATION SEARCH GUIDANCE:\n"
        doc_search_context += f"- Repository: {repo_name}\n"
        doc_search_context += f"- Recommended search: '{repo_name} documentation' or '{repo_name} official documentation'\n"
        doc_search_context += f"- Search for official documentation early in your analysis.\n"

    # Custom ReAct prompt for reasoning models
    template = """You are analyzing a repository to understand its structure and create build instructions.

Available tools: {tool_names}

{tools}

ANALYSIS APPROACH - Discovery over Assumptions:
1. **START WITH WEB SEARCH**: **MANDATORY FIRST STEP** - Use SearchWeb to find official documentation. Search for "{repo_name} {language} documentation" or "{repo_name} official documentation". This gives you authoritative build instructions, prerequisites, and setup guides from official sources. Do this BEFORE exploring local files.
2. START BROAD: Use DirectoryTree to see overall structure
3. LOCATE FILES: Use FindFiles to locate config files (don't assume locations)
4. READ & EXTRACT: Use ReadFile and ExtractJsonField to examine configs
5. SEARCH PATTERNS: Use GrepFiles to find build commands, imports, requirements
6. CROSS-REFERENCE: Verify findings from web documentation with local files{doc_search_context}

CRITICAL FORMAT RULES (YOU MUST FOLLOW EXACTLY):
1. **NEVER write free text or explanations outside the format below**
2. After "Thought:", you MUST write "Action:" followed by the tool name on the SAME line
3. After "Action: <tool_name>", you MUST write "Action Input:" on the next line
4. After "Action Input:", write the input WITHOUT quotes on the SAME line
5. After "Action Input: <input>", STOP IMMEDIATELY - do not write anything else
6. Do NOT write "Observation:" - the system provides it
7. **If you want to provide a final answer, write "Final Answer:" after Thought, not free text**
8. WHEN TO STOP AND PROVIDE FINAL ANSWER:
   - You have searched for official documentation using SearchWeb
   - You have examined the main configuration files (package.json, Cargo.toml, Makefile, etc.)
   - You have read key documentation (README, INSTALL, BUILD files)
   - You have found the core information requested in the question
   - Typically 5-10 tool calls is sufficient for most questions
   - Maximum 15 tool calls per question - after that, provide Final Answer with what you have
9. Each response: EITHER (Thought + Action + Action Input) OR (Thought + Final Answer), NEVER BOTH
10. AVOID OVER-EXPLORATION: Do not read every file or search exhaustively. Focus on the specific question and answer it.
11. **DO NOT REPEAT THE SAME ACTION** - if you already searched, move on to the next step

CORRECT FORMAT EXAMPLES:

Example 1 - Using a tool:
Thought: I need to see the directory structure.
Action: DirectoryTree
Action Input: .,2,false,true

Example 2 - Using web search:
Thought: I should search for official documentation.
Action: SearchWeb
Action Input: requests Python documentation

Example 3 - Providing final answer:
Thought: I have gathered all necessary information.
Final Answer: This is a Node.js project with TypeScript. Prerequisites: Node.js 18+. Installation: npm install. Build: npm run build.

WRONG FORMAT (DO NOT DO THIS):
Thought: I cannot provide a final answer yet. I will first search...
[WRONG - This is free text, not the required format!]

Thought: I need to search
Action:
Action Input: requests Python documentation
[WRONG - Missing tool name after Action:]

Thought: I need to search
Action: SearchWeb
[WRONG - Missing Action Input:]

IMPORTANT: 
- If you already performed SearchWeb in this conversation, do NOT search again - use the results you already have
- If you already read a file, do NOT read it again - use the information you already have
- Move forward with your analysis, don't repeat actions

Now begin!

Previous conversation context (if any):
{chat_history}

Current Question: {input}
Thought:{agent_scratchpad}"""

    # Format the template with repo_name and language if available
    if repo_name and detected_language:
        template = template.replace("{repo_name}", repo_name)
        template = template.replace("{language}", detected_language)
    elif repo_name:
        template = template.replace("{repo_name}", repo_name)
        template = template.replace("{language}", "programming language")
    else:
        template = template.replace("{repo_name}", "the repository")
        template = template.replace("{language}", "programming language")
    
    # Insert doc_search_context
    template = template.replace("{doc_search_context}", doc_search_context)

    prompt = PromptTemplate.from_template(template)

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)

    # Create agent executor with custom callback for better formatting
    callback_handler = FormattedOutputHandler() if verbose else None
    callbacks = [callback_handler] if callback_handler else []

    # Custom parsing error handler function
    def handle_parsing_error(error: Exception) -> str:
        """Handle parsing errors by providing a clear message to the agent."""
        error_str = str(error)
        logger.warning(f"Parsing error: {error_str}")
        if "Could not parse" in error_str or "Missing 'Action:'" in error_str or "Invalid Format" in error_str:
            return "I made a format error. I MUST write 'Action: <tool_name>' on one line, then 'Action Input: <input>' on the next line. I cannot write free text. Example:\nThought: I need to search for documentation.\nAction: SearchWeb\nAction Input: requests Python documentation"
        return f"Format error: {error_str}. I must write 'Action: <tool_name>' then 'Action Input: <input>' on separate lines. I cannot write free text."

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # Disable default verbose output
        callbacks=callbacks,
        handle_parsing_errors=handle_parsing_error,  # Use the function, not True
        max_iterations=max_iterations,
        max_execution_time=None,  # No time limit
        early_stopping_method="generate",  # Generate final answer if max iterations reached
        return_intermediate_steps=True  # Enable to track tool usage
    )

    logger.info("Planner agent created successfully")
    return agent_executor, callback_handler
