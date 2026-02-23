
import os
import re
import sys
import json
import logging
import threading
import httpx
from typing import Optional, List, Dict, Any, Type
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

# ============================================================================
# Shared Resources (HTTP Client)
# ============================================================================

_http_client = None
_http_client_lock = threading.Lock()

def _get_http_client() -> httpx.Client:
    """Get or create the singleton HTTP client (thread-safe)."""
    global _http_client
    if _http_client is None:
        with _http_client_lock:
            if _http_client is None:
                _http_client = httpx.Client(
                    timeout=httpx.Timeout(120.0, connect=30.0),
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                )
    return _http_client

# ============================================================================
# Logging / Output Utilities
# ============================================================================

class ThreadAwareStdout:
    """Thread-safe stdout wrapper that writes to original stdout and a thread-local log file."""
    def __init__(self, original_stream):
        self.original_stream = original_stream
        self.thread_files = {}
        self.lock = threading.Lock()

    def register(self, f):
        thread_id = threading.get_ident()
        with self.lock:
            self.thread_files[thread_id] = f

    def unregister(self):
        thread_id = threading.get_ident()
        with self.lock:
            if thread_id in self.thread_files:
                del self.thread_files[thread_id]

    def write(self, text):
        self.original_stream.write(text)
        thread_id = threading.get_ident()
        if thread_id in self.thread_files:
            try:
                self.thread_files[thread_id].write(text)
                self.thread_files[thread_id].flush()
            except Exception:
                pass

    def flush(self):
        self.original_stream.flush()
        thread_id = threading.get_ident()
        if thread_id in self.thread_files:
            try:
                self.thread_files[thread_id].flush()
            except Exception:
                pass

    def __getattr__(self, name):
        return getattr(self.original_stream, name)

class FormattedOutputHandler(BaseCallbackHandler):
    """Custom callback handler to format agent output with proper spacing and track token usage."""
    def __init__(self, log_file=None):
        super().__init__()
        self.token_usage = {"input": 0, "output": 0, "total": 0}
        self.log_file = log_file
        self.step_counter = 0
        self.transcript = []  # Store full transcript for later analysis

    def _write_log(self, message: str):
        """Write to both console and log file if available."""
        print(message)
        if self.log_file:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
                    f.flush()
            except Exception as e:
                print(f"[WARNING] Failed to write to log file: {e}")

    def on_agent_action(self, action, **kwargs):
        self.step_counter += 1

        # Extract thought from log
        thought = action.log.split('Action:')[0].strip() if 'Action:' in action.log else action.log.strip()

        # Format step
        separator = f"\n{'═'*80}"
        step_header = f"STEP {self.step_counter}: {action.tool}"

        msg = f"{separator}\n{step_header}\n{'═'*80}"
        self._write_log(msg)

        msg = f"\n[THOUGHT]\n{thought}\n"
        self._write_log(msg)

        msg = f"[ACTION] {action.tool}"
        self._write_log(msg)

        # Format input nicely
        tool_input = action.tool_input
        if isinstance(tool_input, dict):
            import json
            input_str = json.dumps(tool_input, indent=2)
        else:
            input_str = str(tool_input)

        msg = f"[INPUT]\n{input_str}\n"
        self._write_log(msg)

        # Store in transcript
        self.transcript.append({
            "step": self.step_counter,
            "type": "action",
            "tool": action.tool,
            "thought": thought,
            "input": tool_input
        })

    def on_tool_end(self, output, **kwargs):
        # Determine output length
        output_str = str(output)
        
        # We always want full output in the log file and transcript
        # For console, we might want to truncate, but for now we'll follow the request 
        # to "write the complete output of the tools to transcripts" which usually implies 
        # the log file. The _write_log method writes to both. 
        # To be safe and meet the user requirement of "complete output", I will remove truncation entirely.
        
        msg = f"[OBSERVATION]\n{output_str}\n{'─'*80}\n"
        self._write_log(msg)

        # Store in transcript
        if self.transcript:
            self.transcript[-1]["observation"] = output_str
            self.transcript[-1]["observation_length"] = len(output_str)

    def on_llm_end(self, response, **kwargs):
        usage = None
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage')

        if usage and isinstance(usage, dict):
            input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
            total = usage.get('total_tokens', 0) or (input_tokens + output_tokens)

            self.token_usage["input"] += input_tokens
            self.token_usage["output"] += output_tokens
            self.token_usage["total"] += total

            # Log token usage for this call
            msg = f"[TOKENS] Input: {input_tokens}, Output: {output_tokens}, Total: {total}"
            self._write_log(msg)

    def on_agent_finish(self, finish, **kwargs):
        """Called when agent completes successfully."""
        msg = f"\n{'═'*80}\n[AGENT FINISHED]\n{finish.return_values.get('output', '')}\n{'═'*80}\n"
        self._write_log(msg)

        # Log final token summary
        msg = f"\n[FINAL TOKEN USAGE] Input: {self.token_usage['input']}, Output: {self.token_usage['output']}, Total: {self.token_usage['total']}\n"
        self._write_log(msg)

    def get_transcript(self):
        """Return the full transcript for saving."""
        return self.transcript

# ============================================================================
# Core Tool Helpers
# ============================================================================

def _resolve_path(user_path: str, repo_base: str) -> str:
    """Resolve path with hard sandbox to repository root."""
    if repo_base is None:
        raise ValueError("Repository base path not set. Tools cannot operate without context.")

    # Resolve to absolute path
    if os.path.isabs(user_path):
        resolved = os.path.abspath(user_path)
    elif user_path == ".":
        resolved = os.path.abspath(repo_base)
    else:
        resolved = os.path.abspath(os.path.join(repo_base, user_path))

    # Ensure it's within repo (security sandbox)
    repo_base_abs = os.path.abspath(repo_base)

    # Allow exact match or paths starting with repo_base/
    if resolved != repo_base_abs and not resolved.startswith(repo_base_abs + os.sep):
        # Allow checking immediate parent if needed? No, strict sandbox.
        raise ValueError(
            f"Security: Path '{user_path}' resolves to '{resolved}' which is outside "
            f"repository '{repo_base_abs}'. Access denied."
        )

    return resolved

def _make_relative_path(absolute_path: str, repo_base: str) -> str:
    if repo_base is None:
        return absolute_path
    try:
        rel_path = os.path.relpath(absolute_path, repo_base)
        if not rel_path.startswith('..'):
            return rel_path
        return absolute_path
    except (ValueError, TypeError):
        return absolute_path

def extract_relevant_sections(soup: BeautifulSoup, max_chars_per_section=2000, max_total_chars=2500):
    """
    Extract relevant sections (installation, build, setup) from HTML with command-bias.
    """
    # aggressive cleanup
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'iframe', 'svg', 'button', 'form']):
        tag.decompose()
        
    build_keywords = ['build', 'compil', 'make', 'cmake', 'cargo build', 'npm run build', 'setup.py', 'dist']
    install_keywords = ['install', 'installation', 'setup', 'getting started', 'quick start', 'prerequisites']
    command_indicators = ['$', 'run ', 'apt-get', 'apk add', 'npm install', 'pip install', 'docker build']

    # 1. Extract High-Value Code Blocks
    code_blocks = []
    for code in soup.find_all(['pre', 'code']):
        text = code.get_text(strip=True)
        if len(text) > 10 and any(x in text.lower() for x in ['install', 'build', 'run', '$', 'npm', 'pip', 'apt']):
            code_blocks.append(f"```\n{text[:500]}\n```") # truncate individual blocks
    
    # 2. Extract Text with Keywords
    body = soup.find('body')
    if not body: return "[No content]"
        
    text = body.get_text(separator='\n\n', strip=True)
    paragraphs = text.split('\n\n')
    
    relevant_text = []
    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean: continue
        
        p_lower = p_clean.lower()
        if any(ci in p_lower for ci in command_indicators) or any(k in p_lower for k in build_keywords + install_keywords):
            relevant_text.append(p_clean)

    combined = "\n".join(code_blocks[:5]) + "\n\n" + "\n\n".join(relevant_text)
    if len(combined) > max_total_chars:
        return combined[:max_total_chars] + "\n... [truncated]"
    if not combined.strip():
        return text[:1000]
    return combined

def _get_expanded_platform_info():
    import platform
    host_arch = platform.machine().lower()
    if host_arch in ['arm64', 'aarch64']:
        return {'docker_arch': 'arm64', 'variants': ['arm64', 'aarch64', 'arm64/v8'], 'display_name': 'ARM64', 'description': 'Apple Silicon / Pi'}
    elif host_arch in ['x86_64', 'amd64']:
        return {'docker_arch': 'amd64', 'variants': ['amd64', 'x86_64'], 'display_name': 'AMD64', 'description': 'Intel/AMD x86_64'}
    else:
        return {'docker_arch': host_arch, 'variants': [host_arch], 'display_name': host_arch.upper(), 'description': f'{host_arch}'}

def _docker_hub_list_tags(image_name: str, platform_info: dict) -> str:
    import requests
    api_image_name = f"library/{image_name}" if "/" not in image_name else image_name
    url = f"https://hub.docker.com/v2/repositories/{api_image_name}/tags"
    params = {"page_size": 20, "ordering": "-last_updated"}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 404: return f"Image '{image_name}' not found."
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results: return "No tags found."
        
        output = [f"Tags for '{image_name}' (Host: {platform_info['display_name']}):"]
        for t in results:
            name = t.get("name")
            archs = [i.get("architecture") for i in t.get("images", [])]
            compatible = any(a in platform_info['variants'] for a in archs)
            mark = "[OK]" if compatible else "[!!]"
            output.append(f"{name} {mark}")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing tags: {e}"

def _docker_hub_verify_tag(image_name: str, tag: str, platform_info: dict) -> str:
    import requests
    api_image_name = f"library/{image_name}" if "/" not in image_name else image_name
    url = f"https://hub.docker.com/v2/repositories/{api_image_name}/tags/{tag}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            archs = [i.get("architecture") for i in data.get("images", [])]
            compatible = any(a in platform_info['variants'] for a in archs)
            return f"VERIFIED: {image_name}:{tag} EXISTS. Architecture check: {'COMPATIBLE' if compatible else 'INCOMPATIBLE'} ({platform_info['display_name']})"
        return f"Tag {image_name}:{tag} NOT FOUND."
    except Exception as e:
        return f"Error verifying: {e}"

def _validate_dockerfile_from_lines(content: str) -> Optional[str]:
    """
    Validate all FROM lines in a Dockerfile against Docker Hub.

    Returns None if every image:tag is verified (or unverifiable due to variable
    substitution / scratch / network error).  Returns a human-readable error
    string — suitable for returning directly to the ReAct agent — if any tag
    is confirmed NOT to exist on Docker Hub.
    """
    platform_info = _get_expanded_platform_info()
    errors = []

    for raw_line in content.splitlines():
        line = raw_line.strip()
        # Match: FROM [--flag=val …] <image_ref> [AS <alias>]
        m = re.match(
            r'^FROM\s+(?:--\w+=\S+\s+)*(\S+)(?:\s+AS\s+\S+)?\s*$',
            line,
            re.IGNORECASE,
        )
        if not m:
            continue

        image_ref = m.group(1)

        # Can't verify variable-substitution refs at write time
        if '$' in image_ref:
            continue

        # Docker's special sentinel — always valid
        if image_ref.lower() == 'scratch':
            continue

        # Split image:tag — use the last colon in the final path segment so
        # registry hosts with ports (registry.example.com:5000/img:tag) work.
        last_segment = image_ref.split('/')[-1]
        if ':' in last_segment:
            idx = image_ref.rfind(':')
            image_name, tag = image_ref[:idx], image_ref[idx + 1:]
        else:
            image_name, tag = image_ref, 'latest'

        try:
            result = _docker_hub_verify_tag(image_name, tag, platform_info)
        except Exception:
            # Network / timeout — don't block the write on infra failures
            continue

        if 'NOT FOUND' in result:
            errors.append(f"  • {image_ref}  →  {result}")

    if not errors:
        return None

    return (
        "WRITE BLOCKED — the following base image(s) do NOT exist on Docker Hub:\n"
        + "\n".join(errors)
        + "\n\n"
        "You MUST verify base images before writing the Dockerfile.\n"
        "REQUIRED STEPS:\n"
        "  1. Call DockerImageSearch(query=\"tags:<image_name>\") to list real tags.\n"
        "  2. Choose a tag marked [OK] for your platform.\n"
        "  3. Rewrite the FROM line with the verified tag, then call WriteToFile again."
    )


def _docker_hub_search(term: str) -> str:
    import requests
    url = "https://hub.docker.com/v2/search/repositories"
    try:
        response = requests.get(url, params={"query": term}, timeout=10)
        results = response.json().get("results", [])
        output = [f"Search '{term}':"]
        for r in results[:5]:
            output.append(f"- {r.get('repo_name')} (Stars: {r.get('star_count')})")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching: {e}"

# ============================================================================
# Pydantic Schemas
# ============================================================================

class WriteToFileInput(BaseModel):
    """Schema for WriteToFile - accepts various input formats including nested JSON."""
    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility
    file_path: str = Field(default="")  # Make optional to handle malformed input
    content: str = Field(default="")  # Make optional to handle malformed input

class ReadLocalFileInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    file_path: str = Field(default="")

class ListDirectoryInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    directory: str = Field(default="")

class CreateDirectoryTreeInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    directory: str = Field(default=".")
    depth: int = Field(default=2)

class FindFilesInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    directory: str = Field(default="")
    pattern: str = Field(default="")
    depth: int = Field(default=5)

class GrepFilesInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    directory: str = Field(default="")
    pattern: str = Field(default="")
    glob: str = Field(default="*")

class ExtractJsonFieldInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    file_path: str = Field(default="")
    field_path: str = Field(default="")

class GetFileMetadataInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    file_path: str = Field(default="")

class DockerImageSearchInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    query: str = Field(default="")

class FetchWebPageInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    url: str = Field(default="")

class SearchWebInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    query: str = Field(default="")

class SearchDockerErrorInput(BaseModel):
    model_config = ConfigDict(extra="allow")
    error_keywords: str = Field(default="", description="Short error keywords for web search")
    full_error_log: str = Field(default="", description="Complete Docker build error output for detailed analysis")
    dockerfile_content: str = Field(default="", description="Full Dockerfile content for context")
    agent_context: str = Field(default="", description="Additional context, observations, or specific questions from the agent about the error")

def _parse_input(input_data: Any, schema: Type[BaseModel], key: str) -> BaseModel:
    """Parse and validate input data against a Pydantic schema.

    Args:
        input_data: The input to parse (can be dict, str, or schema instance)
        schema: The Pydantic BaseModel class to validate against
        key: Fallback field name if input_data is a plain string

    Returns:
        Validated instance of the schema
    """
    if isinstance(input_data, schema): return input_data
    if isinstance(input_data, dict): return schema(**input_data)
    if isinstance(input_data, str):
        try:
            data = json.loads(input_data)
            if isinstance(data, dict): return schema(**data)
        except: pass
        return schema(**{key: input_data})
    raise ValueError(f"Unexpected input type: {type(input_data)}")

# ============================================================================
# Repository Tools (Instance Isolation)
# ============================================================================

class RepositoryTools:
    def __init__(self, repo_root: str):
        self.repo_root = os.path.abspath(repo_root)

    def write_to_file_structured(self, input_data: Any) -> str:
        try:
            # Handle direct dict/object input
            if isinstance(input_data, WriteToFileInput):
                return self._write_impl(input_data.file_path, input_data.content)

            # Handle dict input
            if isinstance(input_data, dict):
                # Check if the dict has file_path as a nested JSON string
                if 'file_path' in input_data and isinstance(input_data['file_path'], str):
                    # Try to parse file_path as JSON if it looks like JSON
                    if input_data['file_path'].strip().startswith('{'):
                        try:
                            nested = json.loads(input_data['file_path'])
                            if isinstance(nested, dict) and 'file_path' in nested and 'content' in nested:
                                return self._write_impl(nested['file_path'], nested['content'])
                        except json.JSONDecodeError:
                            pass

                # Normal dict with file_path and content keys
                if 'file_path' in input_data and 'content' in input_data:
                    return self._write_impl(input_data['file_path'], input_data['content'])

                return "ERROR: Dict must contain 'file_path' and 'content' keys"

            # Handle string input (should be JSON)
            if isinstance(input_data, str):
                try:
                    parsed = json.loads(input_data)
                    if isinstance(parsed, dict) and 'file_path' in parsed and 'content' in parsed:
                        return self._write_impl(parsed['file_path'], parsed['content'])
                    return "ERROR: JSON must contain 'file_path' and 'content' keys"
                except json.JSONDecodeError:
                    return "ERROR: Invalid JSON string for WriteToFile"

            return f"ERROR: Unexpected input type {type(input_data)}. Expected dict with file_path and content."
        except Exception as e:
            return f"Error in WriteToFile: {e}"

    def _write_impl(self, file_path: str, content: str) -> str:
        try:
            path = Path(_resolve_path(file_path, self.repo_root))

            # Hard gate: reject Dockerfiles that reference non-existent base images.
            # This forces the agent to run DockerImageSearch *before* writing FROM lines.
            filename = path.name
            if filename == 'Dockerfile' or filename.startswith('Dockerfile.'):
                validation_error = _validate_dockerfile_from_lines(content)
                if validation_error:
                    logger.warning("WriteToFile blocked: invalid FROM line(s) in %s", file_path)
                    return validation_error

            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    def read_local_file_structured(self, input_data: Any) -> str:
        try:
            data = _parse_input(input_data, ReadLocalFileInput, 'file_path')
            path = _resolve_path(data.file_path, self.repo_root)
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

    def list_directory_structured(self, input_data: Any) -> str:
        try:
            data = _parse_input(input_data, ListDirectoryInput, 'directory')
            path = _resolve_path(data.directory, self.repo_root)
            if not os.path.exists(path):
                return f"Directory not found. (Resolved: '{path}', Exists: False)"
            items = []
            for entry in os.scandir(path):
                kind = "DIR" if entry.is_dir() else "FILE"
                size = entry.stat().st_size if entry.is_file() else 0
                items.append(f"{kind:4} {entry.name} ({size} bytes)")
            return "\n".join(items)
        except Exception as e:
             return f"Error listing directory: {e}"

    def create_directory_tree_structured(self, input_data: Any) -> str:
        try:
            data = _parse_input(input_data, CreateDirectoryTreeInput, 'directory')
            path = _resolve_path(data.directory, self.repo_root)
            tree = []
            start_level = path.rstrip(os.sep).count(os.sep)
            for root, dirs, files in os.walk(path):
                level = root.count(os.sep) - start_level
                if level > data.depth: continue
                indent = "  " * level
                tree.append(f"{indent}{os.path.basename(root)}/")
                for f in files[:20]:
                     tree.append(f"{indent}  {f}")
                if len(files) > 20: 
                    tree.append(f"{indent}  ... ({len(files)-20} more)")
            return "\n".join(tree)
        except Exception as e:
            return f"Error creating tree: {e}"

    def find_files_structured(self, input_data: Any) -> str:
        import fnmatch
        try:
            if isinstance(input_data, FindFilesInput): data = input_data
            elif isinstance(input_data, dict): data = FindFilesInput(**input_data)
            else: data = FindFilesInput(**json.loads(input_data))
            
            path = _resolve_path(data.directory, self.repo_root)
            matches = []
            start_level = path.rstrip(os.sep).count(os.sep)
            for root, dirs, files in os.walk(path):
                level = root.count(os.sep) - start_level
                if level > data.depth: continue
                for f in files:
                    if fnmatch.fnmatch(f, data.pattern):
                        matches.append(os.path.join(root, f))
            return "\n".join([_make_relative_path(p, self.repo_root) for p in matches])
        except Exception as e:
            return f"Error finding files: {e}"

    def grep_files_structured(self, input_data: Any) -> str:
        import fnmatch
        try:
            if isinstance(input_data, GrepFilesInput): data = input_data
            elif isinstance(input_data, dict): data = GrepFilesInput(**input_data)
            else: data = GrepFilesInput(**json.loads(input_data))
            
            path = _resolve_path(data.directory, self.repo_root)
            results = []
            regex = re.compile(data.pattern)
            for root, _, files in os.walk(path):
                for f in files:
                    if fnmatch.fnmatch(f, data.glob):
                        fp = os.path.join(root, f)
                        try:
                            with open(fp, 'r', errors='ignore') as f_obj:
                                for i, line in enumerate(f_obj, 1):
                                    if regex.search(line):
                                        results.append(f"{_make_relative_path(fp, self.repo_root)}:{i}: {line.strip()}")
                                        if len(results) > 50: break
                        except: pass
                if len(results) > 50: break
            return "\n".join(results)
        except Exception as e:
            return f"Error grepping: {e}"

    def extract_json_field_structured(self, input_data: Any) -> str:
        try:
            if isinstance(input_data, ExtractJsonFieldInput): data = input_data
            elif isinstance(input_data, dict): data = ExtractJsonFieldInput(**input_data)
            else: data = ExtractJsonFieldInput(**json.loads(input_data))
            
            filepath = _resolve_path(data.file_path, self.repo_root)
            key_path = data.field_path.split(".")
            with open(filepath, 'r') as f:
                json_data = json.load(f)
            curr = json_data
            for k in key_path:
                curr = curr.get(k, {})
            return str(curr)
        except Exception as e:
            return f"Error extracting json: {e}"

    def get_file_metadata_structured(self, input_data: Any) -> str:
        try:
            data = _parse_input(input_data, GetFileMetadataInput, 'file_path')
            path = _resolve_path(data.file_path, self.repo_root)
            stat = os.stat(path)
            return f"Size: {stat.st_size} bytes, Mode: {oct(stat.st_mode)}"
        except Exception as e:
            return f"Error metadata: {e}"

# ============================================================================
# Global (Statsless) Tools
# ============================================================================

def docker_image_search_structured(input_data: Any) -> str:
    try:
        data = _parse_input(input_data, DockerImageSearchInput, 'query')
        platform_info = _get_expanded_platform_info()
        if data.query.startswith("tags:"):
            return _docker_hub_list_tags(data.query[5:].strip(), platform_info)
        elif ":" in data.query:
            parts = data.query.split(":")
            return _docker_hub_verify_tag(parts[0], parts[1], platform_info)
        else:
            return _docker_hub_search(data.query)
    except Exception as e:
        return f"Input error: {e}"

def fetch_web_page_structured(input_data: Any) -> str:
    try:
        if isinstance(input_data, FetchWebPageInput): data = input_data
        elif isinstance(input_data, dict): data = FetchWebPageInput(**input_data)
        else: data = FetchWebPageInput(**json.loads(input_data))
        
        import requests
        resp = requests.get(data.url, timeout=10)
        soup = BeautifulSoup(resp.content, "html.parser")
        return extract_relevant_sections(soup)
    except Exception as e:
        return f"Error fetching: {e}"

def search_web_structured(input_data: Any) -> str:
    try:
        data = _parse_input(input_data, SearchWebInput, 'query')
        from ddgs import DDGS
        import requests
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(data.query, max_results=3):
                    results.append(r)
        except: return "Search failed."

        if not results: return "No results."
        out = []
        for r in results:
            url = r.get("href")
            try:
                resp = requests.get(url, timeout=10)
                soup = BeautifulSoup(resp.content, "html.parser")
                content = extract_relevant_sections(soup)
                out.append(f"Title: {r.get('title')}\nURL: {url}\nContent: {content}\n")
            except:
                out.append(f"Title: {r.get('title')}\nURL: {url}\n(Content fetch failed)\n")
        return "\n".join(out)
    except Exception as e:
        return f"Input error: {e}"

def search_docker_error_structured(input_data: Any) -> str:
    """
    Search for Docker error solutions and use LLM to analyze and suggest fixes.

    This enhanced version:
    1. Searches the web for relevant solutions
    2. Uses LLM to analyze full error log + Dockerfile content + search results
    3. Returns a specific, actionable fix suggestion with precise line numbers
    """
    try:
        from langchain_openai import AzureChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        data = _parse_input(input_data, SearchDockerErrorInput, 'error_keywords')

        # Validate input
        if not data.error_keywords or not data.error_keywords.strip():
            return "ERROR: Please provide error keywords to search for solutions."

        # Step 1: Search for solutions (best-effort — analysis still runs if search fails)
        query = f"docker {data.error_keywords} solution site:stackoverflow.com OR site:github.com OR site:docs.docker.com"
        search_results = search_web_structured({"query": query})
        search_available = bool(
            search_results
            and "Search failed" not in search_results
            and "No results" not in search_results
            and "Input error" not in search_results
        )

        # Step 2: Use LLM to analyze and suggest fix.
        # Prefer the large deployment for richer analysis; fall back to the default deployment.
        _analysis_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_LARGE", os.getenv("AZURE_OPENAI_DEPLOYMENT"))
        llm = AzureChatOpenAI(
            azure_deployment=_analysis_deployment,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION")
        )

        system_prompt = """You are a Docker expert debugging build errors.
Analyze the complete error log, Dockerfile content, and search results to provide a SPECIFIC, ACTIONABLE fix.

When you have the Dockerfile, reference exact line numbers in your fix.
Consider the full error trace to identify cascading failures and root causes.

CRITICAL: You MUST follow this exact format:

**Root Cause:** [1-2 sentences explaining what went wrong and which Dockerfile line/step caused it]
**Fix:** [Concrete Dockerfile changes needed with line numbers if available]
**Example:** [Show the exact Dockerfile code to add/change]

Be concise and practical. Focus on the most common solution first."""

        # Build comprehensive user prompt with all available context
        user_prompt_parts = [f"ERROR KEYWORDS: {data.error_keywords}"]

        # Add full error log if provided
        if data.full_error_log and data.full_error_log.strip():
            # Truncate if extremely long (keep first and last portions)
            error_log = data.full_error_log
            if len(error_log) > 15000:
                error_log = error_log[:7500] + "\n\n... [middle truncated for brevity] ...\n\n" + error_log[-7500:]
            user_prompt_parts.append(f"\nFULL ERROR LOG:\n{error_log}")

        # Add Dockerfile content if provided
        if data.dockerfile_content and data.dockerfile_content.strip():
            user_prompt_parts.append(f"\nDOCKERFILE CONTENT:\n{data.dockerfile_content}")

        # Add search results only when the search actually succeeded
        if search_available:
            user_prompt_parts.append(f"\nSEARCH RESULTS FROM WEB:\n{search_results[:3000]}")
        else:
            user_prompt_parts.append("\n(Web search unavailable — analyse from error log and Dockerfile only.)")

        # Add agent context if provided
        if hasattr(data, 'agent_context') and data.agent_context and data.agent_context.strip():
            user_prompt_parts.append(f"\nADDITIONAL AGENT CONTEXT/QUESTIONS:\n{data.agent_context}")

        user_prompt_parts.append("\nAnalyze this Docker build error and provide a fix in the required format.")

        user_prompt = "\n".join(user_prompt_parts)

        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # Validate response has expected sections
            content = response.content
            if "**Root Cause:**" not in content or "**Fix:**" not in content:
                logger.warning("LLM response missing required sections, using raw response")
                content = f"**Root Cause:** Analysis incomplete\n**Fix:** {content}\n**Example:** See fix above"

            sources_section = (
                f"=== SEARCH SOURCES (for reference) ===\n{search_results[:1000]}"
                if search_available
                else "(Web search was unavailable; analysis is based on error log and Dockerfile only.)"
            )
            return f"=== AI ANALYSIS ===\n{content}\n\n{sources_section}\n"

        except Exception as llm_error:
            # Fallback to search results (if any) when LLM fails
            logger.warning(f"LLM analysis failed: {llm_error}")
            fallback = search_results if search_available else "(Neither web search nor LLM analysis available.)"
            return f"[AI analysis unavailable: {str(llm_error)}]\n\n{fallback}"

    except Exception as e:
        logger.error(f"SearchDockerError failed: {e}")
        return f"Error searching for Docker error solution: {e}\n\nPlease try searching manually or checking Docker documentation."

# ============================================================================
# Factory
# ============================================================================

def create_structured_tools(repo_root: str) -> list:
    """Create all structured tools for the agent, bound to the specific repo root."""
    repo_tools = RepositoryTools(repo_root)
    
    return [
        StructuredTool(
            name="WriteToFile",
            func=repo_tools.write_to_file_structured,
            description="Write content to a file at the specified path. Overwrites if exists.",
            args_schema=WriteToFileInput
        ),
        StructuredTool(
            name="ReadLocalFile",
            func=repo_tools.read_local_file_structured,
            description="Read the content of a local file.",
            args_schema=ReadLocalFileInput
        ),
        StructuredTool(
            name="ListDirectory",
            func=repo_tools.list_directory_structured,
            description="List files and directories in a directory.",
            args_schema=ListDirectoryInput
        ),
        StructuredTool(
            name="CreateDirectoryTree",
            func=repo_tools.create_directory_tree_structured,
            description="Create a tree view of directory structure.",
            args_schema=CreateDirectoryTreeInput
        ),
        StructuredTool(
            name="FindFiles",
            func=repo_tools.find_files_structured,
            description="Find files matching a pattern in a directory.",
            args_schema=FindFilesInput
        ),
        StructuredTool(
            name="GrepFiles",
            func=repo_tools.grep_files_structured,
            description="Search for text pattern in files.",
            args_schema=GrepFilesInput
        ),
        StructuredTool(
            name="ExtractJsonField",
            func=repo_tools.extract_json_field_structured,
            description="Extract a specific field from a JSON file.",
            args_schema=ExtractJsonFieldInput
        ),
        StructuredTool(
            name="GetFileMetadata",
            func=repo_tools.get_file_metadata_structured,
            description="Get metadata (size, permissions) for a file.",
            args_schema=GetFileMetadataInput
        ),
        # Stateless Tools (safe to share logic, but we create fresh instances)
        StructuredTool(
            name="SearchWeb",
            func=search_web_structured,
            description="Search the web for documentation and solutions.",
            args_schema=SearchWebInput
        ),
        StructuredTool(
            name="SearchDockerError",
            func=search_docker_error_structured,
            description="""Search for Docker build error solutions with AI-powered analysis.

            WHEN TO USE: After VerifyBuild fails with an error.

            INPUT: Provide as much context as possible:
            - error_keywords (REQUIRED): Key error message for web search (e.g., "unable to locate package")
            - full_error_log (RECOMMENDED): Complete Docker build output for detailed analysis
            - dockerfile_content (RECOMMENDED): Full Dockerfile content for precise line-level fixes

            OUTPUT: You will receive:
            - AI ANALYSIS with Root Cause, Fix, and Example code (with line numbers if Dockerfile provided)
            - SEARCH SOURCES with relevant documentation links

            WHAT TO DO NEXT:
            1. Read the AI ANALYSIS to understand the root cause
            2. Apply the suggested fix to your Dockerfile
            3. Run VerifyBuild again to test the fix

            BEST PRACTICE EXAMPLE:
            SearchDockerError(
                error_keywords="unable to
                 locate package build-essential",
                full_error_log="<entire output from docker build>",
                dockerfile_content="<read from ReadLocalFile>"
            )

            MINIMAL EXAMPLE (less accurate):
            SearchDockerError(error_keywords="unable to locate package build-essential")
            """,
            args_schema=SearchDockerErrorInput
        ),
        StructuredTool(
            name="FetchWebPage",
            func=fetch_web_page_structured,
            description="Fetch and extract relevant content from a web page.",
            args_schema=FetchWebPageInput
        ),
        StructuredTool(
            name="DockerImageSearch",
            func=docker_image_search_structured,
            description="Search Docker Hub for images or verify tag existence.",
            args_schema=DockerImageSearchInput
        ),
    ]
