
import os
import sys
import json
import logging
import threading
import httpx
from typing import Optional, List, Dict, Any
from pathlib import Path
from bs4 import BeautifulSoup
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

# ============================================================================
# Path Resolution Infrastructure (Thread-Safe)
# ============================================================================

_thread_local = threading.local()
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

def _get_repository_base_path() -> Optional[str]:
    return getattr(_thread_local, 'repository_base_path', None)

def _set_repository_base_path(path: str):
    _thread_local.repository_base_path = path

def _get_report_directory() -> Optional[str]:
    return getattr(_thread_local, 'report_directory', None)

def _set_report_directory(path: str):
    _thread_local.report_directory = path

def _make_relative_path(absolute_path: str) -> str:
    repo_base = _get_repository_base_path()
    if repo_base is None:
        return absolute_path
    try:
        rel_path = os.path.relpath(absolute_path, repo_base)
        if not rel_path.startswith('..'):
            return rel_path
        return absolute_path
    except (ValueError, TypeError):
        return absolute_path

def _resolve_path(user_path: str) -> str:
    repo_base = _get_repository_base_path()
    if repo_base is None:
        return user_path
    if os.path.isabs(user_path):
        return user_path
    if user_path == ".":
        return repo_base
    return os.path.join(repo_base, user_path)

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
    def __init__(self):
        super().__init__()
        self.token_usage = {"input": 0, "output": 0, "total": 0}

    def on_agent_action(self, action, **kwargs):
        print(f"\n{'─'*70}")
        print(f"[THOUGHT] {action.log.split('Action:')[0].strip() if 'Action:' in action.log else action.log.strip()}")
        print(f"\n[ACTION] {action.tool}")
        print(f"[INPUT] {action.tool_input}")
        print(f"{'─'*70}")

    def on_tool_end(self, output, **kwargs):
        output_preview = str(output)[:200] + "..." if len(str(output)) > 200 else str(output)
        print(f"\n[OBSERVATION] {output_preview}\n")

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

# ============================================================================
# Core Tool Helpers
# ============================================================================

def extract_relevant_sections(soup: BeautifulSoup, max_chars_per_section=3000, max_total_chars=5000):
    """Extract relevant sections (installation, build, setup) from HTML using heuristics."""
    import re
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'iframe', 'svg']):
        tag.decompose()
        
    build_keywords = ['build', 'compil', 'make', 'cmake', 'cargo build', 'npm run build', 'setup.py', 'dist']
    install_keywords = ['install', 'installation', 'setup', 'getting started', 'quick start', 'prerequisites']
    
    body = soup.find('body')
    if not body: return "[No content]"
        
    text = body.get_text(separator='\n', strip=True)
    paragraphs = text.split('\n\n')
    
    relevant = []
    for p in paragraphs:
        if any(k in p.lower() for k in build_keywords + install_keywords):
            relevant.append(p)
    
    final_text = "\n\n".join(relevant)
    if not final_text:
        # Fallback to first 5000 chars of body if keywords fail
        return text[:max_total_chars]
        
    return final_text[:max_total_chars]

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
# Tool Implementations
# ============================================================================

def write_to_file(file_path: str, content: str) -> str:
    """
    Write content to a file at the specified path. Overwrites if exists.
    
    Args:
        file_path: The relative or absolute path to the file.
        content: The content to write.
    """
    try:
        path = Path(_resolve_path(file_path))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {e}"

def read_local_file(file_path: str) -> str:
    try:
        resolved_path = _resolve_path(file_path)
        with open(resolved_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def list_directory(directory: str) -> str:
    try:
        path = _resolve_path(directory)
        if not os.path.exists(path): return "Directory not found."
        items = []
        for entry in os.scandir(path):
            kind = "DIR" if entry.is_dir() else "FILE"
            size = entry.stat().st_size if entry.is_file() else 0
            items.append(f"{kind:4} {entry.name} ({size} bytes)")
        return "\n".join(items)
    except Exception as e:
        return f"Error listing directory: {e}"

def create_directory_tree(input_str: str) -> str:
    # "d,depth,hidden,files"
    try:
        parts = input_str.split(",")
        path = _resolve_path(parts[0]) if len(parts) > 0 else "."
        depth = int(parts[1]) if len(parts) > 1 else 2
        
        tree = []
        start_level = path.rstrip(os.sep).count(os.sep)
        for root, dirs, files in os.walk(path):
            level = root.count(os.sep) - start_level
            if level > depth: continue
            indent = "  " * level
            tree.append(f"{indent}{os.path.basename(root)}/")
            for f in files[:20]: # Limit files
                 tree.append(f"{indent}  {f}")
            if len(files) > 20: 
                tree.append(f"{indent}  ... ({len(files)-20} more)")
        return "\n".join(tree)
    except Exception as e:
        return f"Error creating tree: {e}"

def find_files(input_str: str) -> str:
    # "dir,pattern,depth"
    import fnmatch
    try:
        parts = input_str.split(",")
        path = _resolve_path(parts[0])
        pattern = parts[1]
        depth = int(parts[2]) if len(parts) > 2 else 5
        
        matches = []
        start_level = path.rstrip(os.sep).count(os.sep)
        for root, dirs, files in os.walk(path):
            level = root.count(os.sep) - start_level
            if level > depth: continue
            for f in files:
                if fnmatch.fnmatch(f, pattern):
                    matches.append(os.path.join(root, f))
        return "\n".join([_make_relative_path(p) for p in matches])
    except Exception as e:
        return f"Error finding files: {e}"

def grep_files(input_str: str) -> str:
    # "dir,pattern,glob,context"
    import re, fnmatch
    try:
        parts = input_str.split(",")
        path = _resolve_path(parts[0])
        pattern = parts[1]
        glob = parts[2] if len(parts) > 2 else "*"
        
        results = []
        regex = re.compile(pattern)
        
        for root, _, files in os.walk(path):
            for f in files:
                if fnmatch.fnmatch(f, glob):
                    fp = os.path.join(root, f)
                    try:
                        with open(fp, 'r', errors='ignore') as f_obj:
                            for i, line in enumerate(f_obj, 1):
                                if regex.search(line):
                                    results.append(f"{_make_relative_path(fp)}:{i}: {line.strip()}")
                                    if len(results) > 50: break
                    except: pass
            if len(results) > 50: break
        return "\n".join(results)
    except Exception as e:
        return f"Error grepping: {e}"

def extract_json_field(input_str: str) -> str:
    # "file,path.to.key"
    try:
        parts = input_str.split(",")
        filepath = _resolve_path(parts[0])
        key_path = parts[1].split(".")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        curr = data
        for k in key_path:
            curr = curr.get(k, {})
            
        return str(curr)
    except Exception as e:
        return f"Error extracting json: {e}"

def get_file_metadata(filepath: str) -> str:
    try:
        path = _resolve_path(filepath)
        stat = os.stat(path)
        return f"Size: {stat.st_size} bytes, Mode: {oct(stat.st_mode)}"
    except Exception as e:
        return f"Error metadata: {e}"

def docker_image_search(query: str) -> str:
    platform_info = _get_expanded_platform_info()
    if query.startswith("tags:"):
        return _docker_hub_list_tags(query[5:].strip(), platform_info)
    elif ":" in query:
        parts = query.split(":")
        return _docker_hub_verify_tag(parts[0], parts[1], platform_info)
    else:
        return _docker_hub_search(query)

def search_docker_error(error_keywords: str) -> str:
    return search_web(f"docker {error_keywords} solution")

def search_web(query: str) -> str:
    try:
        from ddgs import DDGS
        import requests
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(r)
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
        return f"Search failed: {e}"

def fetch_web_page(url: str) -> str:
    try:
        import requests
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.content, "html.parser")
        return extract_relevant_sections(soup)
    except Exception as e:
        return f"Fetch failed: {e}"
