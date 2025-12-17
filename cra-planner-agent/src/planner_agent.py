#!/usr/bin/env python3
"""
Planner Agent Module
Core agent functionality without direct execution or GitHub API dependencies.
Import this module to use the agent in other scripts.
"""

import os
import sys
import json
import logging
import ssl
import certifi
import httpx
import threading
from typing import Any, List, Optional, Dict
from datetime import datetime

# Fix SSL certificate issue on macOS with Anaconda Python
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

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
        # DEBUG: Print response structure on first call
        if self.token_usage["total"] == 0:
            print(f"\n[DEBUG TOKEN] Response type: {type(response)}")
            print(f"[DEBUG TOKEN] Response attributes: {[a for a in dir(response) if not a.startswith('_')]}")
            if hasattr(response, 'llm_output'):
                print(f"[DEBUG TOKEN] llm_output: {response.llm_output}")
            if hasattr(response, 'generations'):
                print(f"[DEBUG TOKEN] generations length: {len(response.generations) if response.generations else 0}")
                if response.generations and len(response.generations) > 0 and len(response.generations[0]) > 0:
                    gen = response.generations[0][0]
                    print(f"[DEBUG TOKEN] Generation info: {gen.generation_info if hasattr(gen, 'generation_info') else 'No generation_info'}")
            print(f"[DEBUG TOKEN] kwargs keys: {list(kwargs.keys())}")

        # Try to get token usage from multiple possible locations
        usage = None

        # Method 1: response.llm_output
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            if usage:
                print(f"[DEBUG TOKEN] Found usage in llm_output: {usage}")

        # Method 2: response.generations (for Azure OpenAI)
        if not usage and hasattr(response, 'generations') and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, 'generation_info') and gen.generation_info:
                        usage = gen.generation_info.get('token_usage', {})
                        if usage:
                            print(f"[DEBUG TOKEN] Found usage in generation_info: {usage}")
                            break
                if usage:
                    break

        # Method 3: kwargs (sometimes passed in callbacks)
        if not usage and 'usage' in kwargs:
            usage = kwargs['usage']
            print(f"[DEBUG TOKEN] Found usage in kwargs: {usage}")

        # Method 4: Check response_metadata (Azure specific)
        if not usage and hasattr(response, 'response_metadata'):
            usage = response.response_metadata.get('token_usage', {})
            if usage:
                print(f"[DEBUG TOKEN] Found usage in response_metadata: {usage}")

        # Extract tokens
        if usage and isinstance(usage, dict):
            input_tokens = usage.get('prompt_tokens', 0) or usage.get('input_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0) or usage.get('output_tokens', 0)
            total_tokens = usage.get('total_tokens', 0) or (input_tokens + output_tokens)

            if input_tokens or output_tokens:
                self.token_usage["input"] += input_tokens
                self.token_usage["output"] += output_tokens
                self.token_usage["total"] += total_tokens
                print(f"[TOKEN UPDATE] +{input_tokens} input, +{output_tokens} output (total so far: {self.token_usage['total']})")
        else:
            if self.token_usage["total"] == 0:
                print(f"[DEBUG TOKEN] No usage found in response!")


# ============================================================================
# LLM Wrapper for gpt-5-nano
# ============================================================================

class GPT5NanoWrapper(BaseChatModel):
    """Wrapper for gpt-5-nano that strips unsupported parameters and rate-limits API calls."""

    llm: AzureChatOpenAI

    class Config:
        arbitrary_types_allowed = True

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        # Remove unsupported parameters
        kwargs.pop('stop', None)
        # Use semaphore to limit concurrent API calls under parallel load
        with _api_semaphore:
            return self.llm._generate(messages, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "gpt-5-nano-wrapper"

    def bind_tools(self, tools, **kwargs):
        """Bind tools while preserving the wrapper (keeps semaphore and param stripping)."""
        bound = self.llm.bind_tools(tools, **kwargs)
        # Return a new wrapper instance with the bound LLM
        return GPT5NanoWrapper(llm=bound)


# ============================================================================
# Path Resolution Infrastructure (Thread-Safe)
# ============================================================================

# Thread-local storage for repository paths - each thread gets its own copy
_thread_local = threading.local()

# Global semaphore to limit concurrent API calls
# Allow 2 concurrent calls for better parallelism while avoiding DNS issues
# (Previously 1, but with connection pooling + keepalive, 2 is safe)
_api_semaphore = threading.Semaphore(2)

# ============================================================================
# Singleton HTTP Client (Thread-Safe)
# ============================================================================
# httpx.Client is thread-safe and supports connection pooling
# Using a singleton ensures all threads share the same connection pool
_http_client = None
_http_client_lock = threading.Lock()


def _get_http_client() -> httpx.Client:
    """Get or create the singleton HTTP client (thread-safe)."""
    global _http_client
    if _http_client is None:
        with _http_client_lock:
            # Double-check locking pattern
            if _http_client is None:
                _http_client = httpx.Client(
                    timeout=httpx.Timeout(120.0, connect=30.0),  # 120s total, 30s connect
                    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                )
    return _http_client

# Legacy global variables (for backwards compatibility with single-threaded code)
# These are ONLY used when thread-local is not set
REPOSITORY_BASE_PATH = None
REPORT_DIRECTORY = None


def _get_repository_base_path() -> str:
    """Get the repository base path for the current thread."""
    # Try thread-local first, fall back to global
    return getattr(_thread_local, 'repository_base_path', None) or REPOSITORY_BASE_PATH


def _set_repository_base_path(path: str):
    """Set the repository base path for the current thread."""
    _thread_local.repository_base_path = path


def _get_report_directory() -> str:
    """Get the report directory for the current thread."""
    # Try thread-local first, fall back to global
    return getattr(_thread_local, 'report_directory', None) or REPORT_DIRECTORY


def _set_report_directory(path: str):
    """Set the report directory for the current thread."""
    _thread_local.report_directory = path


def _make_relative_path(absolute_path: str) -> str:
    """
    Convert an absolute path to a relative path from the repository base.
    This is used in tool outputs so the agent sees relative paths.

    Args:
        absolute_path: Absolute path to convert

    Returns:
        Relative path from repository base, or original path if no base set
    """
    repo_base = _get_repository_base_path()
    if repo_base is None:
        return absolute_path

    try:
        # Get relative path from repository base
        rel_path = os.path.relpath(absolute_path, repo_base)
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
    repo_base = _get_repository_base_path()
    if repo_base is None:
        # No repository context, use path as-is
        return user_path

    # If user provides absolute path, use it directly
    if os.path.isabs(user_path):
        return user_path

    # Special case: "." refers to the repository base itself
    if user_path == ".":
        return repo_base

    # Otherwise, resolve relative to repository base
    # Tools now return relative paths, so this works correctly
    return os.path.join(repo_base, user_path)


# ============================================================================
# Tool Functions
# ============================================================================

def extract_relevant_sections(soup, max_chars_per_section=3000, max_total_chars=5000):
    """
    Extract relevant sections (installation, build, setup) from HTML using heuristics.
    
    Improvements:
    - Link Density Filter: Ignores blocks where (>50%) text is links (menus/sidebars).
    - Semantic Targeting: Prioritizes <main>, <article>, and specific IDs.
    - Code Block Boost: Prioritizes sections with <pre>/<code> tags.
    
    Args:
        soup: BeautifulSoup parsed HTML
        max_chars_per_section: Maximum characters per extracted section
        max_total_chars: Maximum total characters to return
        
    Returns:
        Extracted text content focusing on build/installation sections
    """
    import re
    
    # 1. Clean up obviously bad elements first
    for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript', 'iframe', 'svg']):
        tag.decompose()
        
    # Keywords
    build_keywords = ['build', 'compil', 'make', 'cmake', 'cargo build', 'npm run build', 'setup.py', 'dist']
    install_keywords = ['install', 'installation', 'setup', 'getting started', 'quick start', 'prerequisites', 'requirements', 'dependencies']
    
    found_sections = []
    
    def get_link_density(element):
        """Calculate ratio of link text to total text."""
        total_text_len = len(element.get_text(strip=True))
        if total_text_len == 0:
            return 0
        link_text_len = sum(len(a.get_text(strip=True)) for a in element.find_all('a'))
        return link_text_len / total_text_len

    def is_menu_or_garbage(element):
        """Heuristic to detect menus/sidebars."""
        # High link density = likely a menu
        if get_link_density(element) > 0.5:
            return True
        return False

    # 2. Identify potential content containers
    # Prioritize semantic tags or divs with specific classes
    content_roots = []
    
    # Strict semantic search first
    main_tag = soup.find('main')
    if main_tag: content_roots.append(main_tag)
    
    article_tags = soup.find_all('article')
    if article_tags: content_roots.extend(article_tags)
    
    # Common content wrappers (if no semantic tags found or to supplement)
    if not content_roots:
        for class_name in ['content', 'main', 'documentation', 'wiki-content', 'markdown-body', 'guide']:
            found = soup.find_all('div', class_=re.compile(class_name, re.I))
            content_roots.extend(found)
            
    # Fallback to body if nothing specific found
    if not content_roots:
        body = soup.find('body')
        if body: content_roots.append(body)

    # 3. Extract from identified roots
    processed_headings = set()
    
    for root in content_roots:
        # Check if root itself is garbage
        if is_menu_or_garbage(root) and root.name != 'body': # Be clearer with body
            continue
            
        headings = root.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        for heading in headings:
            if heading in processed_headings:
                continue
            processed_headings.add(heading)
            
            heading_text = heading.get_text().strip().lower()
            
            # Relevance check
            is_build = any(k in heading_text for k in build_keywords)
            is_install = any(k in heading_text for k in install_keywords)
            
            # Boost score if "code" or "pre" follows immediately
            has_code = False
            
            if is_build or is_install:
                section_content = []
                current = heading.next_sibling
                chars_collected = 0
                
                # Determine hierarchy level
                try:
                    heading_level = int(heading.name[1])
                except:
                    heading_level = 6
                
                while current and chars_collected < max_chars_per_section:
                    if hasattr(current, 'name'):
                        # Stop at same/higher level heading
                        if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            try:
                                next_level = int(current.name[1])
                                if next_level <= heading_level:
                                    break
                            except:
                                pass
                        
                        # Filtering
                        if current.name in ['div', 'section', 'ul', 'ol', 'p']:
                            if is_menu_or_garbage(current):
                                current = current.next_sibling
                                continue
                                
                        # Check for code
                        if current.name in ['pre', 'code'] or (hasattr(current, 'find_all') and current.find_all(['pre', 'code'])):
                            has_code = True
                            
                        # Add text
                        text = current.get_text(separator=' ', strip=True)
                        if text:
                            section_content.append(text)
                            chars_collected += len(text)
                            
                    elif isinstance(current, str):
                        text = current.strip()
                        if text:
                            section_content.append(text)
                            chars_collected += len(text)
                            
                    current = current.next_sibling
                
                if section_content:
                    full_text = '\n'.join(section_content)
                    
                    # Heuristic Scoring for Priority
                    # Base: 1 (Build), 2 (Install)
                    # Modifiers: -5 if has_code (makes it effectively Top Priority 0 or -4)
                    priority = 1 if is_build else 2
                    if has_code:
                        priority -= 5  # Boost significantly
                        
                    found_sections.append({
                        'heading': heading.get_text().strip(),
                        'content': full_text[:max_chars_per_section],
                        'priority': priority,
                        'order': len(found_sections)
                    })

    # 4. Fallback: If no relevant sections found, return Main Content text (cleaned)
    if not found_sections:
        # Try to find the single largest block of text that isn't a menu
        best_block = None
        max_text_len = 0
        
        candidates = soup.find_all(['div', 'section', 'article'])
        for block in candidates:
            # Skip if it contains other candidates (get leaf-ish nodes)
            # Actually, getting largest distinct block is safer
            if is_menu_or_garbage(block):
                continue
                
            text_len = len(block.get_text(strip=True))
            if text_len > max_text_len:
                max_text_len = text_len
                best_block = block
        
        if best_block:
             return f"[Auto-extracted Main Content]\n{best_block.get_text(separator=' ', strip=True)[:max_total_chars]}"
        
        # Absolute fallback
        if soup.body:
             return soup.body.get_text(separator=' ', strip=True)[:max_total_chars]
        return "[No content found]"

    # Sort and Format
    found_sections.sort(key=lambda x: (x['priority'], x['order']))
    
    extracted_text = []
    total_chars = 0
    
    for sec in found_sections:
        text = f"\n{'='*40}\nSECTION: {sec['heading']}\n{'='*40}\n{sec['content']}\n"
        if total_chars + len(text) <= max_total_chars:
            extracted_text.append(text)
            total_chars += len(text)
        else:
            break
            
    return '\n'.join(extracted_text)


def search_docker_error(error_keywords: str) -> str:
    """
    Search web for Docker error solutions and debugging guidance.
    Focuses on Stack Overflow, Docker docs, and troubleshooting guides.
    
    BEST USAGE:
    - For image errors: "docker hub maven available tags" or "what java docker images exist"
    - For build errors: "npm install failed docker alpine" or "pip install gcc docker"
    - For syntax errors: "dockerfile syntax error unknown instruction"
    
    Args:
        error_keywords: Key error terms - be DESCRIPTIVE (e.g., "docker hub maven available tags", "pip install failed docker")
        
    Returns:
        Summary of solutions from top results (max 4000 chars)
    """
    try:
        logger.info(f"Searching Docker error solutions for: {error_keywords}")
        
        results = []
        
        # Azure Safety Sanitization: Remove code-like symbols to prevent content filter triggers
        # We strip characters that look like injection attacks: {} [] <> ; $ \ ` ' "
        import re
        
        # 1. Remove obvious code block markers and special chars
        clean_keywords = re.sub(r'[{}<>\[\]$;`\\\'"]', '', error_keywords)
        
        # 2. Remove extremely long file paths (keep filenames)
        # Matches /very/long/path/to/file.ext -> file.ext
        clean_keywords = re.sub(r'(?:/[a-zA-Z0-9_\-\.]+)+/([a-zA-Z0-9_\-\.]+\.[a-z]+)', r'\1', clean_keywords)
        
        # 3. Collapse whitespace and truncate
        clean_keywords = re.sub(r'\s+', ' ', clean_keywords).strip()
        
        if len(clean_keywords) > 200:
            logger.warning(f"Sanitizing search query: '{error_keywords[:50]}...' -> '{clean_keywords[:200]}...'")
            error_keywords = clean_keywords[:200]
        else:
            error_keywords = clean_keywords

        try:
            from ddgs import DDGS
            import requests
            from bs4 import BeautifulSoup
            
            # Build search queries - use the input directly plus variations
            search_queries = [
                error_keywords,  # Use exact input first
                f"docker {error_keywords} solution",
            ]
            
            # Clean queries of newlines/tabs which confuse search engines
            search_queries = [q.replace('\n', ' ').replace('\r', ' ').strip() for q in search_queries]
            
            results = []
            seen_urls = set()
            
            with DDGS() as ddgs:
                for query in search_queries:
                    try:
                        # Fetch more results to filter duplicates
                        search_results = list(ddgs.text(query, max_results=5))
                        for result in search_results:
                            url = result.get('href', '')
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                results.append({
                                    'title': result.get('title', 'No title'),
                                    'url': url,
                                    'snippet': result.get('body', '')
                                })
                                if len(results) >= 6:
                                    break
                    except Exception as e:
                        logger.warning(f"DDGS query failed: {query} - {e}")
                        continue
                    
                    if len(results) >= 6:
                        break
                        
        except ImportError:
            logger.warning("ddgs not installed, falling back to SearchWeb")
        except Exception as e:
            logger.warning(f"DDGS search failed completely: {e}")
            
        # FALLBACK METHOD: Standard SearchWeb (using requests/BeautifulSoup loop if available)
        if not results:
            logger.info("Using fallback SearchWeb implementation for error search...")
            try:
                # Reuse the existing search_web tool logic if imported or accessible
                # Since we are inside the same module, we can call search_web directly
                fallback_result = search_web(error_keywords)
                if "No results found" not in fallback_result:
                    return f"FALLBACK SEARCH RESULTS:\n{fallback_result}"
            except Exception as e:
                logger.error(f"Fallback search also failed: {e}")

        if not results:
            return f"No solutions found for: {error_keywords}. Try:\n1) Use DockerImageSearch with 'tags:<image>' to list available tags\n2) Check Dockerfile syntax\n3) Verify dependency versions"
        
        summary_parts = [f"Web Search Results for '{error_keywords}':\n{'='*60}\n"]
        
        for idx, result in enumerate(results[:5], 1):
            summary_parts.append(f"\n{idx}. {result['title']}")
            summary_parts.append(f"   URL: {result['url']}")
            if result['snippet']:
                snippet = result['snippet'][:300]
                summary_parts.append(f"   {snippet}...")
            summary_parts.append("")
            
        # FETCH TOP RESULT CONTENT (Depth-1)
        # This is critical for actually solving the error
        if results:
            top_url = results[0]['url']
            try:
                logger.info(f"Fetching top error solution: {top_url}")
                # Use a short timeout to keep it fast
                resp = requests.get(top_url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.content, 'html.parser')
                    # Use our smart extraction
                    content = extract_relevant_sections(soup, max_chars_per_section=2000, max_total_chars=3000)
                    
                    # Sanitize content relative to "Azure Error" (remove non-printable chars)
                    # This prevents binary data or weird encodings from breaking the LLM context
                    cleaned_content = "".join(c for c in content if c.isprintable() or c in ['\n', '\r', '\t'])
                    
                    summary_parts.append(f"\n{'='*60}\nTOP SOLUTION DETAILED CONTENT ({top_url})\n{'='*60}\n{cleaned_content}")
            except Exception as e:
                logger.warning(f"Could not fetch top error solution {top_url}: {e}")
        
        summary_parts.append(f"\nTIP: Use FetchWebPage tool to read full content from other URLs")
        
        summary = '\n'.join(summary_parts)
        
        if len(summary) > 6000: # Increased limit since we have real content
            summary = summary[:6000] + "\n\n[Truncated]"
        
        return summary
        
    except Exception as e:
        logger.error(f"Docker error search failed: {e}")
        return f"Search failed: {str(e)}. Manual debugging:\n1) Use DockerImageSearch with 'tags:<image>' to list available tags\n2) Check Dockerfile syntax\n3) Review error logs for missing dependencies"


def search_web(query: str) -> str:
    """
    Search the web for official product documentation and build guides.
    Performs multiple searches (documentation + build guide), fetches top 3 unique pages,
    extracts relevant build/installation sections, and saves full content to files.
    
    Args:
        query: Base search query (e.g., "requests Python")
        
    Returns:
        Shortened summary (max 5000 chars) with key build/installation sections from top 3 pages
    """
    try:
        logger.info(f"Searching the web for: {query}")
        
        try:
            from ddgs import DDGS
            import requests
            from bs4 import BeautifulSoup
            from datetime import datetime
            from pathlib import Path
            import re
            
            # Check for cached web search files first
            cached_pages = []
            report_dir = _get_report_directory()
            if report_dir:
                report_path = Path(report_dir)
                for idx in range(1, 4):
                    cache_file = report_path / f"web_search_{idx}.txt"
                    if cache_file.exists():
                        try:
                            with open(cache_file, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Parse cached file to extract metadata and content
                            title_match = re.search(r'Title: (.+)', content)
                            url_match = re.search(r'URL: (.+)', content)
                            
                            if title_match and url_match:
                                # Extract content after the header (after "FULL PAGE CONTENT:")
                                content_start = content.find("FULL PAGE CONTENT:")
                                if content_start != -1:
                                    full_content = content[content_start + len("FULL PAGE CONTENT:"):].strip()
                                    # Remove the separator line
                                    if full_content.startswith("="*70):
                                        parts = full_content.split("="*70, 2)
                                        if len(parts) > 2:
                                            full_content = parts[2].strip()
                                    
                                    cached_pages.append({
                                        'title': title_match.group(1).strip(),
                                        'url': url_match.group(1).strip(),
                                        'full_content': full_content,
                                        'cached': True
                                    })
                                    logger.info(f"Found cached web search result {idx}: {url_match.group(1).strip()}")
                        except Exception as cache_error:
                            logger.warning(f"Error reading cached file {cache_file}: {cache_error}")
                            continue
            
            # If we have cached pages, use them instead of searching
            if cached_pages:
                logger.info(f"Using {len(cached_pages)} cached web search results")
                fetched_pages = []
                total_output_chars = 0
                max_output_chars = 5000
                
                for idx, cached_page in enumerate(cached_pages, 1):
                    # Parse the cached content to extract relevant sections
                    try:
                        # Create a simple HTML-like structure from text for section extraction
                        # We'll search for headings in the text content
                        full_text = cached_page['full_content']
                        
                        # Extract relevant sections using text patterns
                        build_keywords = ['build', 'compil', 'make', 'cmake', 'cargo build', 'npm run build', 'setup.py', 'dist']
                        install_keywords = ['install', 'installation', 'setup', 'getting started', 'quick start', 'prerequisites', 'requirements', 'dependencies']
                        
                        # Find sections by looking for lines that look like headings (all caps, or followed by colons)
                        lines = full_text.split('\n')
                        relevant_sections = []
                        current_section = []
                        in_relevant_section = False
                        
                        for i, line in enumerate(lines):
                            line_lower = line.lower().strip()
                            # Check if line looks like a heading
                            is_heading = (line.isupper() and len(line) < 100) or \
                                        (line.endswith(':') and len(line) < 100) or \
                                        (line.startswith('#') and len(line) < 100) or \
                                        (len(line) > 0 and len(line) < 80 and not line[0].islower())
                            
                            if is_heading:
                                # Check if heading is relevant
                                is_build = any(kw in line_lower for kw in build_keywords)
                                is_install = any(kw in line_lower for kw in install_keywords)
                                
                                if is_build or is_install:
                                    # Save previous section if it was relevant
                                    if current_section and in_relevant_section:
                                        relevant_sections.append('\n'.join(current_section))
                                    # Start new section
                                    current_section = [line]
                                    in_relevant_section = True
                                else:
                                    # Save previous section
                                    if current_section and in_relevant_section:
                                        relevant_sections.append('\n'.join(current_section))
                                    current_section = []
                                    in_relevant_section = False
                            elif in_relevant_section:
                                current_section.append(line)
                                # Limit section size
                                if len('\n'.join(current_section)) > 2000:
                                    relevant_sections.append('\n'.join(current_section))
                                    current_section = []
                                    in_relevant_section = False
                        
                        # Add final section if relevant
                        if current_section and in_relevant_section:
                            relevant_sections.append('\n'.join(current_section))
                        
                        # Combine relevant sections
                        if relevant_sections:
                            relevant_content = '\n\n'.join(relevant_sections)
                            if len(relevant_content) > 2000:
                                relevant_content = relevant_content[:2000] + "..."
                        else:
                            # Fallback: use first part of content
                            relevant_content = full_text[:2000] + ("..." if len(full_text) > 2000 else "")
                        
                        # Prepare summary for agent
                        page_summary = f"\n{'='*70}\nPAGE {idx}: {cached_page['title']}\nURL: {cached_page['url']}\n[FROM CACHE]\n{'='*70}\n"
                        page_summary += relevant_content
                        
                        # Check if we have room in output
                        if total_output_chars + len(page_summary) <= max_output_chars:
                            fetched_pages.append(page_summary)
                            total_output_chars += len(page_summary)
                        elif total_output_chars < max_output_chars - 500:
                            remaining = max_output_chars - total_output_chars - 200
                            if remaining > 500:
                                page_summary_short = page_summary[:remaining] + "\n[Content truncated...]"
                                fetched_pages.append(page_summary_short)
                                total_output_chars = max_output_chars
                        
                    except Exception as parse_error:
                        logger.warning(f"Error parsing cached content: {parse_error}")
                        # Fallback: use first part of content
                        page_summary = f"\n{'='*70}\nPAGE {idx}: {cached_page['title']}\nURL: {cached_page['url']}\n[FROM CACHE]\n{'='*70}\n"
                        page_summary += cached_page['full_content'][:1500] + ("..." if len(cached_page['full_content']) > 1500 else "")
                        if total_output_chars + len(page_summary) <= max_output_chars:
                            fetched_pages.append(page_summary)
                            total_output_chars += len(page_summary)
                
                # Build final output from cache
                output = f"WEB SEARCH RESULTS - Build/Installation Guides (FROM CACHE)\n"
                output += f"{'='*70}\n"
                output += f"Using {len(cached_pages)} cached results from previous search\n"
                output += f"Focus: Build and installation sections extracted\n"
                output += f"{'='*70}\n\n"
                
                if fetched_pages:
                    output += '\n'.join(fetched_pages)
                else:
                    output += "No relevant content found in cache."
                
                if len(output) > max_output_chars:
                    output = output[:max_output_chars] + "\n\n[Output truncated - see web_search_X.txt files for full content]"
                
                logger.info(f"Returned cached search results, {len(output)} chars")
                return output
            
            # No cache found, perform new search
            logger.info("No cached results found, performing new web search")
            
            # Extract project name from query (first word, before language)
            # e.g., "requests Python" -> "requests", "react JavaScript" -> "react"
            base_query = query.strip()
            query_parts = base_query.split()
            project_name = query_parts[0].lower() if query_parts else base_query.lower()
            logger.info(f"Extracted project name: {project_name}")
            
            # Generate multiple search queries - focus on actionable installation/build guides
            # Skip general documentation, target specific guides that contain step-by-step instructions
            queries = [
                f"{base_query} how to install",
                f"{base_query} installation instructions",
                f"{base_query} build from source",
                f"{base_query} building guide",
                f"{base_query} setup guide",
                f"{base_query} getting started install"
            ]
            
            # Collect unique results from all searches
            all_results = []
            seen_urls = set()
            
            with DDGS() as ddgs:
                for search_query in queries:
                    try:
                        for r in ddgs.text(search_query, max_results=5):
                            href = r.get('href', '')
                            if href and href.startswith('http') and href not in seen_urls:
                                # Filter out unwanted sources
                                skip_indicators = ['wikipedia.org', 'stackoverflow.com', 'reddit.com', 'youtube.com']
                                if not any(skip in href.lower() for skip in skip_indicators):
                                    all_results.append({
                                        'title': r.get('title', 'No title'),
                                        'body': r.get('body', 'No description'),
                                        'href': href,
                                        'query': search_query
                                    })
                                    seen_urls.add(href)
                    except Exception as e:
                        logger.warning(f"Search query '{search_query}' failed: {e}")
                        continue
            
            if not all_results:
                return "No relevant documentation found. Try a more specific query."
            
            # Prioritize results: MUST contain project name in URL or title (not just body)
            def prioritize_result(result):
                href_lower = result['href'].lower()
                title_lower = result.get('title', '').lower()
                body_lower = result.get('body', '').lower()
                combined_text = f"{href_lower} {title_lower} {body_lower}"
                score = 0
                
                # CRITICAL: Must contain project name in URL or title (not just body text)
                # This prevents false matches like "HTTP requests" in unrelated docs
                project_in_url = project_name in href_lower
                project_in_title = project_name in title_lower
                
                if project_in_url or project_in_title:
                    score += 100  # Massive boost for project-specific pages (URL or title)
                    if project_in_url and project_in_title:
                        score += 20  # Extra boost if in both
                else:
                    # Heavily penalize - these are likely false matches
                    score -= 100  # Very negative score to filter out
                
                # Highest priority: pages with installation/build in URL or title
                if any(x in href_lower for x in ['/install', '/installation', '/build', '/building', '/setup', '/getting-started']):
                    score += 15
                if any(x in title_lower for x in ['install', 'installation', 'build', 'building', 'setup', 'getting started']):
                    score += 12
                
                # Official documentation sites with install/build content
                if any(x in href_lower for x in ['docs.', 'documentation', 'readthedocs.io']):
                    if any(x in combined_text for x in ['install', 'build', 'setup']):
                        score += 10
                    else:
                        score += 5  # Lower score if no install/build content
                
                # GitHub pages with install/build in path
                if 'github.com' in href_lower:
                    if any(x in href_lower for x in ['/install', '/build', '/setup', 'readme']):
                        score += 10
                    elif any(x in title_lower for x in ['install', 'build', 'setup']):
                        score += 8
                
                # General indicators
                if any(x in combined_text for x in ['build from source', 'how to install', 'installation guide', 'build guide']):
                    score += 8
                if any(x in href_lower for x in ['build', 'install', 'setup', 'guide']):
                    score += 5
                if 'official' in combined_text:
                    score += 3
                
                return score
            
            # Sort and filter: prioritize project-specific results
            all_results.sort(key=prioritize_result, reverse=True)
            
            # STRICT FILTER: Only include results with project name in URL or title (not just body)
            project_specific_results = [
                r for r in all_results 
                if project_name in r['href'].lower() or project_name in r.get('title', '').lower()
            ]
            
            if project_specific_results:
                logger.info(f"Found {len(project_specific_results)} project-specific results (project name in URL/title), filtering out {len(all_results) - len(project_specific_results)} irrelevant results")
                all_results = project_specific_results[:10]  # Keep top 10 project-specific results
            else:
                logger.warning(f"No project-specific results found for '{project_name}' (in URL or title). This might indicate:")
                logger.warning(f"  1. The project name '{project_name}' might be incorrect")
                logger.warning(f"  2. Search results don't contain the project name in URLs/titles")
                logger.warning(f"  3. Using top results anyway, but they may not be relevant")
                # Still use top results, but warn that they might not be relevant
                all_results = all_results[:10]
            
            # Fetch top 3 unique pages
            pages_to_fetch = all_results[:3]
            fetched_pages = []
            total_output_chars = 0
            max_output_chars = 5000
            successful_fetches = 0
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            for idx, result in enumerate(pages_to_fetch, 1):
                url = result['href']
                title = result['title']
                
                try:
                    logger.info(f"Fetching page {idx}/3: {url}")
                    
                    # Try fetching with retries and better error handling
                    response = None
                    fetch_success = False
                    error_details = None
                    
                    try:
                        # Increase timeout and add SSL verification options
                        response = requests.get(
                            url, 
                            headers=headers, 
                            timeout=30,  # Increased from 10 to 30 seconds
                            verify=True,  # SSL verification
                            allow_redirects=True,
                            stream=False
                        )
                        response.raise_for_status()
                        fetch_success = True
                        logger.info(f"Successfully fetched {url} (status: {response.status_code})")
                    except requests.exceptions.Timeout as e:
                        error_details = f"Timeout after 30 seconds: {str(e)}"
                        logger.error(f"Timeout fetching {url}: {e}")
                    except requests.exceptions.HTTPError as e:
                        error_details = f"HTTP Error {response.status_code if response else 'unknown'}: {str(e)}"
                        logger.error(f"HTTP error fetching {url}: {e}")
                    except requests.exceptions.ConnectionError as e:
                        error_details = f"Connection error: {str(e)}"
                        logger.error(f"Connection error fetching {url}: {e}")
                    except requests.exceptions.RequestException as e:
                        error_details = f"Request error: {str(e)}"
                        logger.error(f"Request error fetching {url}: {e}")
                    except Exception as e:
                        error_details = f"Unexpected error: {type(e).__name__}: {str(e)}"
                        logger.error(f"Unexpected error fetching {url}: {e}")
                    
                    if not fetch_success or response is None:
                        # Save error details
                        report_dir = _get_report_directory()
                        if report_dir:
                            try:
                                save_path = Path(report_dir) / f"web_search_{idx}.txt"
                                with open(save_path, 'w', encoding='utf-8') as f:
                                    f.write(f"Web Search Result #{idx}\n")
                                    f.write(f"{'='*70}\n")
                                    f.write(f"Title: {title}\n")
                                    f.write(f"URL: {url}\n")
                                    f.write(f"Search Query: {result['query']}\n")
                                    f.write(f"Project Name: {project_name}\n")
                                    f.write(f"Fetched: {datetime.now().isoformat()}\n")
                                    f.write(f"Status: FAILED\n")
                                    f.write(f"Error: {error_details}\n")
                                    f.write(f"{'='*70}\n\n")
                                    f.write("ERROR: Could not fetch page content.\n")
                                    f.write(f"\nError Details:\n{error_details}\n")
                                logger.info(f"Saved failed fetch info to {save_path}")
                            except Exception as save_error:
                                logger.error(f"Could not save failed fetch info: {save_error}")
                        continue
                    
                    # Parse HTML
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Remove unwanted elements
                    for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        script.decompose()
                    
                    # Extract relevant sections (build/installation focused)
                    relevant_content = extract_relevant_sections(soup, max_chars_per_section=2000, max_total_chars=2000)
                    
                    # Get full text for saving to file
                    full_text = soup.get_text()
                    lines = (line.strip() for line in full_text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    full_text_clean = '\n'.join(chunk for chunk in chunks if chunk)
                    
                    # Save to file if report directory is set
                    report_dir = _get_report_directory()
                    if report_dir:
                        try:
                            save_path = Path(report_dir) / f"web_search_{idx}.txt"
                            with open(save_path, 'w', encoding='utf-8') as f:
                                f.write(f"Web Search Result #{idx}\n")
                                f.write(f"{'='*70}\n")
                                f.write(f"Title: {title}\n")
                                f.write(f"URL: {url}\n")
                                f.write(f"Search Query: {result['query']}\n")
                                f.write(f"Project Name: {project_name}\n")
                                f.write(f"Fetched: {datetime.now().isoformat()}\n")
                                f.write(f"Status: SUCCESS (HTTP {response.status_code})\n")
                                f.write(f"{'='*70}\n\n")
                                f.write("FULL PAGE CONTENT:\n")
                                f.write(f"{'='*70}\n\n")
                                f.write(full_text_clean)
                            logger.info(f"Saved web search content to {save_path} ({len(full_text_clean)} chars)")
                            successful_fetches += 1
                        except Exception as save_error:
                            logger.error(f"Could not save web search file {idx}: {save_error}")
                            # Continue anyway - don't break the loop
                    
                    # NOTE: Don't append to fetched_pages yet - wait until after depth-1 crawling
                    # so we only append once with all content included

                except Exception as unexpected_error:
                    # Catch any other unexpected errors
                    error_msg = f"Unexpected error: {type(unexpected_error).__name__}: {str(unexpected_error)}"
                    logger.error(f"Unexpected error processing {url}: {error_msg}")
                    continue

                # DEPTH-1 CRAWLING LOGIC
                # If content seems generic (no strong "Build"/"Install" sections extracted), looks for links
                is_generic = len(relevant_content) < 500 or "SECTION:" not in relevant_content
                
                if is_generic and idx <= 2: # Only for top 2 results to save time
                    logger.info(f"Result {idx} seems generic. Scanning for guide links...")
                    
                    # Find best "guide" link
                    best_link = None
                    best_score = 0
                    
                    # Quick scan of links in the soup
                    for a in soup.find_all('a', href=True):
                        text = a.get_text().strip().lower()
                        href = a['href']
                        
                        # skip anchors and same page
                        if href.startswith('#') or href == url:
                            continue
                            
                        score = 0
                        if 'install' in text: score += 5
                        if 'build' in text: score += 5
                        if 'get started' in text: score += 4
                        if 'guide' in text: score += 3
                        if 'quick' in text: score += 2
                        
                        # Bias towards relative links or same domain
                        if not href.startswith('http') or project_name in href.lower():
                            score += 1
                            
                        if score > best_score:
                            best_score = score
                            best_link = href
                            
                    if best_link and best_score >= 4:
                        # Resolve relative URL
                        if not best_link.startswith('http'):
                            from urllib.parse import urljoin
                            best_link = urljoin(url, best_link)
                            
                        logger.info(f"Following depth-1 link: {best_link} (score: {best_score})")
                        
                        try:
                            # Fetch the sub-page
                            sub_resp = requests.get(best_link, headers=headers, timeout=20, verify=True)
                            if sub_resp.status_code == 200:
                                sub_soup = BeautifulSoup(sub_resp.content, 'html.parser')
                                
                                # Extract from sub-page
                                sub_content = extract_relevant_sections(sub_soup, max_chars_per_section=2500, max_total_chars=4000)
                                
                                # Append to relevant content explicitly
                                relevant_content += f"\n\n{'='*70}\n[AUTO-CRAWLED SUB-PAGE]: {best_link}\n{'='*70}\n{sub_content}"
                                logger.info(f"Added content from sub-page {best_link}")
                        except Exception as e:
                            logger.warning(f"Failed to crawl sub-page {best_link}: {e}")

                # NOW append to fetched_pages (after optional depth-1 crawling is complete)
                # Prepare summary for agent (limited chars)
                page_summary = f"\n{'='*70}\nPAGE {idx}: {title}\nURL: {url}\n{'='*70}\n"
                page_summary += relevant_content

                # Check if we have room in output
                if total_output_chars + len(page_summary) <= max_output_chars:
                    fetched_pages.append(page_summary)
                    total_output_chars += len(page_summary)
                elif total_output_chars < max_output_chars - 500:  # Add partial if there's significant room
                    remaining = max_output_chars - total_output_chars - 200
                    if remaining > 500:
                        page_summary_short = page_summary[:remaining] + "\n[Content truncated...]"
                        fetched_pages.append(page_summary_short)
                        total_output_chars = max_output_chars

                logger.info(f"Successfully processed page {idx}: {title} ({len(relevant_content)} chars extracted, appended to results)")
            output = f"WEB SEARCH RESULTS - Build/Installation Guides\n"
            output += f"{'='*70}\n"
            output += f"Found {len(all_results)} total results, fetched top {len(fetched_pages)} pages\n"
            output += f"Focus: Build and installation sections extracted\n"
            report_dir = _get_report_directory()
            if report_dir:
                output += f"Full content saved to: web_search_1.txt, web_search_2.txt, web_search_3.txt\n"
            output += f"{'='*70}\n\n"
            
            if fetched_pages:
                output += '\n'.join(fetched_pages)
            else:
                output += "No pages could be fetched. Check logs for details."
            
            # Ensure we don't exceed limit
            if len(output) > max_output_chars:
                output = output[:max_output_chars] + "\n\n[Output truncated - see web_search_X.txt files for full content]"
            
            logger.info(f"Search completed, {len(output)} chars returned, {len(fetched_pages)} pages processed")
            
            # Save the exact summary the agent sees to a file for user inspection
            report_dir = _get_report_directory()
            if report_dir:
                try:
                    summary_path = Path(report_dir) / "web_search_agent_summary.txt"
                    with open(summary_path, 'w', encoding='utf-8') as f:
                        f.write(output)
                    logger.info(f"Saved agent search view to {summary_path}")
                except Exception as save_err:
                    logger.error(f"Could not save agent search view: {save_err}")

            return output
            
        except ImportError as e:
            error_msg = "Search dependencies not available. Please install: pip install -U ddgs beautifulsoup4 requests"
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


def _get_host_platform() -> tuple:
    """
    Detect the host platform for Docker compatibility checking.
    Returns (platform_name, docker_arch) e.g., ('linux/arm64', 'arm64') or ('linux/amd64', 'amd64')
    
    Supported architectures:
    - arm64/aarch64: Apple Silicon, AWS Graviton, modern ARM servers
    - x86_64/amd64: Intel/AMD 64-bit processors
    - i386/i686: 32-bit x86 processors
    - armv7l/armv6l: 32-bit ARM (Raspberry Pi, embedded)
    - ppc64le: IBM PowerPC 64-bit Little Endian
    - s390x: IBM Z mainframes
    - riscv64: RISC-V 64-bit
    - mips64le: MIPS 64-bit Little Endian
    """
    import platform
    machine = platform.machine().lower()
    
    # Map machine types to Docker architecture names
    if machine in ['arm64', 'aarch64']:
        return ('linux/arm64', 'arm64')
    elif machine in ['x86_64', 'amd64']:
        return ('linux/amd64', 'amd64')
    elif machine in ['i386', 'i686', 'i586']:
        return ('linux/386', '386')
    elif machine in ['armv7l', 'armv7', 'armhf']:
        return ('linux/arm/v7', 'arm')
    elif machine in ['armv6l', 'armv6']:
        return ('linux/arm/v6', 'arm')
    elif machine == 'ppc64le':
        return ('linux/ppc64le', 'ppc64le')
    elif machine == 's390x':
        return ('linux/s390x', 's390x')
    elif machine == 'riscv64':
        return ('linux/riscv64', 'riscv64')
    elif machine in ['mips64le', 'mips64el']:
        return ('linux/mips64le', 'mips64le')
    else:
        return (f'linux/{machine}', machine)


def _get_expanded_platform_info():
    """Helper to get detailed platform info for Docker compatibility checks."""
    import platform
    host_arch = platform.machine().lower()
    
    if host_arch in ['arm64', 'aarch64']:
        return {
            'docker_arch': 'arm64',
            'variants': ['arm64', 'aarch64', 'arm64/v8'],
            'display_name': 'ARM64',
            'description': 'Apple Silicon / Raspberry Pi'
        }
    elif host_arch in ['x86_64', 'amd64']:
        return {
            'docker_arch': 'amd64',
            'variants': ['amd64', 'x86_64'],
            'display_name': 'AMD64',
            'description': 'Intel/AMD x86_64'
        }
    else:
        return {
            'docker_arch': host_arch,
            'variants': [host_arch],
            'display_name': host_arch.upper(),
            'description': f'{host_arch} processor'
        }

def _docker_hub_list_tags(image_name: str, platform_info: dict) -> str:
    """Helper to list and filter tags for an image."""
    import requests
    
    # Handle official images
    api_image_name = f"library/{image_name}" if "/" not in image_name else image_name
    url = f"https://hub.docker.com/v2/repositories/{api_image_name}/tags"
    params = {"page_size": 100, "ordering": "-last_updated"}  # Increased to 100
    
    try:
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 404:
            return f"ERROR: Image '{image_name}' not found on Docker Hub. Check the image name."
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            return f"No tags found for image: {image_name}"

        # Categorization buckets
        categories = {
            "versioned": [],
            "slim": [],
            "jdk": [], 
            "latest": [],
            "compatible": [],
            "other": []
        }
        
        for tag_info in results:
            tag = tag_info.get("name", "")
            last_updated = tag_info.get("last_updated", "")[:10]
            
            # Check architecture support
            images = tag_info.get("images", [])
            archs = [img.get("architecture", "") for img in images]
            is_compatible = any(arch in platform_info['variants'] for arch in archs)
            
            compat_marker = "[OK]" if is_compatible else "[!!]"
            entry = f"{tag} {compat_marker} (updated: {last_updated})"
            
            if is_compatible:
                categories["compatible"].append(tag)
                
            tag_lower = tag.lower()
            if tag_lower in ["latest", "stable"]:
                categories["latest"].append(entry)
            elif any(x in tag_lower for x in ["slim", "alpine", "minimal", "distroless"]):
                categories["slim"].append(entry)
            elif any(x in tag_lower for x in ["jdk", "jre", "openjdk", "temurin"]):
                categories["jdk"].append(entry)
            elif any(c.isdigit() for c in tag) and "." in tag:
                categories["versioned"].append(entry)
            else:
                categories["other"].append(entry)

        # Build Output
        output = [
            f"Available tags for '{image_name}':",
            f"{'='*60}",
            f"HOST: {platform_info['display_name']} ({platform_info['description']})",
            f"[OK] = Compatible  |  [!!] = INCOMPATIBLE",
            ""
        ]
        
        if categories["versioned"]:
            output.append("VERSIONED TAGS (Recommended):")
            output.extend([f"  - {image_name}:{t.split(' ')[0]}" for t in categories["versioned"][:15]])
            output.append("")
            
        if categories["jdk"]:
            output.append("JAVA/JDK TAGS:")
            output.extend([f"  - {image_name}:{t.split(' ')[0]}" for t in categories["jdk"][:10]])
            output.append("")
            
        if categories["slim"]:
            output.append("SLIM/ALPINE TAGS:")
            output.extend([f"  - {image_name}:{t.split(' ')[0]}" for t in categories["slim"][:10]])
            output.append("")
            
        if categories["latest"]:
            output.append("LATEST (Use with caution):")
            output.extend([f"  - {image_name}:{t.split(' ')[0]}" for t in categories["latest"]])
            output.append("")
            
        # Smart Recommendation
        output.append("RECOMMENDATION:")
        if categories["compatible"]:
            rec = categories["compatible"][0]
            # Try to find a better recommendation (versioned > latest)
            for t in categories["compatible"]:
                if any(c.isdigit() for c in t) and "latest" not in t.lower():
                    rec = t
                    break
            output.append(f"Use: {image_name}:{rec} (Verified {platform_info['display_name']} compatible)")
        else:
            output.append(f"WARNING: No {platform_info['display_name']} compatible tags found!")
            
        return "\n".join(output)

    except Exception as e:
        return f"Error listing tags: {str(e)}"

def _docker_hub_verify_tag(image_name: str, tag: str, platform_info: dict) -> str:
    """Helper to verify a specific tag exists and check architecture."""
    import requests
    
    api_image_name = f"library/{image_name}" if "/" not in image_name else image_name
    url = f"https://hub.docker.com/v2/repositories/{api_image_name}/tags/{tag}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            images = data.get("images", [])
            archs = list(set([img.get("architecture", "unknown") for img in images]))
            
            is_compatible = any(arch in platform_info['variants'] for arch in archs)
            
            status = "COMPATIBLE" if is_compatible else "NOT COMPATIBLE"
            
            output = [
                f"VERIFIED: {image_name}:{tag} EXISTS",
                f"   Architectures: {', '.join(sorted(archs))}",
                f"   Host Platform: {platform_info['display_name']} -> {status}",
                ""
            ]
            
            if is_compatible:
                output.append(f"   Safe to use: FROM {image_name}:{tag}")
            else:
                output.append("   WARNING: Image verified but incompatible with host architecture!")
                output.append(f"   The build will fail on {platform_info['display_name']}.")
                
            return "\n".join(output)
            
        elif response.status_code == 404:
            return f"NOT FOUND: {image_name}:{tag} does not exist.\nUse 'tags:{image_name}' to see valid tags."
        else:
            return f"Error verifying verify: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error verifying tag: {str(e)}"

def _docker_hub_search(term: str) -> str:
    """Helper to search Docker Hub with smart sorting (Official first)."""
    import requests
    
    url = "https://hub.docker.com/v2/search/repositories"
    params = {"query": term, "page_size": 10}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])
        
        if not results:
            return f"No results found for '{term}'"
            
        # IMPROVEMENT: Sort Official images to top
        results.sort(key=lambda x: (not x.get("is_official", False), -x.get("star_count", 0)))
        
        output = [f"Docker Hub Search Results for '{term}':", f"{'='*60}"]
        
        for res in results[:5]:  # Top 5
            name = res.get("repo_name")
            is_official = " [OFFICIAL]" if res.get("is_official") else ""
            stars = res.get("star_count", 0)
            desc = res.get("short_description", "")[:80] + "..." if len(res.get("short_description", "")) > 80 else res.get("short_description", "")
            
            simple_name = name.split("/")[-1] if "/" in name else name
            
            output.append(f"- {name}{is_official}")
            output.append(f"  Stars: {stars} | {desc}")
            output.append(f"  List tags: 'tags:{simple_name}'")
            output.append(f"{'-'*60}")
            
        return "\n".join(output)

    except Exception as e:
        return f"Error searching: {str(e)}"

def docker_image_search(query: str) -> str:
    """
    Search Docker Hub for images, list available tags, or verify a specific tag exists.
    
    Args:
        query: 
            - "tags:<image>" to list available tags (e.g., "tags:python")
            - "<image>:<tag>" to verify a specific tag (e.g., "python:3.9")
            - "<search term>" to search for images (e.g., "python")
    """
    try:
        import requests
        
        # Get host info once
        platform_info = _get_expanded_platform_info()
        logger.info(f"Docker Hub query: {query} (Host: {platform_info['display_name']})")
        
        query = query.strip()
        
        # Dispatch based on query pattern
        if query.lower().startswith("tags:"):
            image_name = query[5:].strip()
            return _docker_hub_list_tags(image_name, platform_info)
            
        elif ":" in query and " " not in query:
            parts = query.split(":")
            return _docker_hub_verify_tag(parts[0], parts[1], platform_info)
            
        else:
            return _docker_hub_search(query)
            
    except ImportError:
        return "Error: requests package required. Install with: pip install requests"
    except Exception as e:
        logger.error(f"Error in docker_image_search: {e}")
        return f"Error: {str(e)}"


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
    # Set thread-local repository path (thread-safe for parallel execution)
    _set_repository_base_path(repository_path)
    
    # Also set global for backwards compatibility with single-threaded code
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

    # Initialize Azure OpenAI with robust retry and timeout settings
    # Use singleton http_client for connection pooling across all threads
    base_llm = AzureChatOpenAI(
        azure_deployment=deployment,
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
        max_retries=5,  # Retry up to 5 times on connection errors
        timeout=120,  # 120 second timeout
        http_client=_get_http_client(),
    )

    # Wrap the model to strip unsupported parameters
    llm = GPT5NanoWrapper(llm=base_llm)

    # Define tools
    tools = [
        Tool(
            name="SearchWeb",
            func=search_web,
            description="**MANDATORY FIRST STEP**: Search the web for installation instructions and build guides. You MUST use this tool at the beginning of your analysis. This tool performs targeted searches for 'how to install', 'installation instructions', 'build from source', 'building guide', 'setup guide', and 'getting started' pages. It fetches the top 3 unique pages with actual step-by-step instructions and extracts relevant build/installation sections. Input: base search query (e.g., 'requests Python' or 'react JavaScript'). Returns focused build/installation content (max 5000 chars). Full page content is saved to web_search_X.txt files in the report directory."
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
        ),
        Tool(
            name="DockerImageSearch",
            func=docker_image_search,
            description="""Docker Hub tool with THREE modes:
1. LIST TAGS: Input 'tags:<image>' (e.g., 'tags:maven', 'tags:python') - Shows ALL available tags categorized by version, JDK, slim/alpine
2. VERIFY TAG: Input '<image>:<tag>' (e.g., 'python:3.9') - Checks if specific tag exists
3. SEARCH: Input '<term>' (e.g., 'python') - Searches for images

RECOMMENDED WORKFLOW for fixing image errors:
1. First use 'tags:maven' to list ALL available tags
2. Pick a MODERN, VERSIONED tag (e.g., '3.9.6', '21-jdk'). AVOID ancient tags (e.g., '0.10', '3.3.1').
3. Verify with 'maven:3.9.6' before using in Dockerfile"""
        ),
        Tool(
            name="SearchDockerError",
            func=search_docker_error,
            description="""Search web for Docker solutions and available image tags. Input should be DESCRIPTIVE:
- For image errors: "docker hub maven available tags versions" or "what tags exist for python docker image"
- For build errors: "npm install failed docker alpine missing dependencies"
- For syntax errors: "dockerfile syntax error unknown instruction"
Returns top web results with snippets. Use FetchWebPage to read full content from URLs."""
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
                    doc_search_context += f"- Search for official documentation early in your analysis.\n"

    # Detect host architecture for prompt context
    _, host_arch_name = _get_host_platform()
    
    # Custom ReAct prompt for reasoning models
    template = """You are analyzing a repository to understand its structure and create build instructions.

HOST ARCHITECTURE: {host_arch_name} (You MUST ensure Docker images are compatible with this architecture)

Available tools: {tool_names}

{tools}

ANALYSIS APPROACH - Discovery over Assumptions:

**CONTEXT AWARENESS**:
- You may receive "PRE-LOADED CONFIGURATION FILES" (package.json, requirements.txt, etc.) in your input.
- **ALWAYS** check these first! They contain critical dependency and build info.
- Do not re-read these files using `ReadFile` unless you need to see lines beyond the preview.

1. **START WITH WEB SEARCH**: **MANDATORY FIRST STEP** - Use SearchWeb to find official documentation. Search for "{repo_name} {language} documentation" or "{repo_name} official documentation". This gives you authoritative build instructions, prerequisites, and setup guides from official sources. Do this BEFORE exploring local files.
2. START BROAD: Use DirectoryTree to see overall structure
3. LOCATE FILES: Use FindFiles to locate config files (don't assume locations)
4. READ & EXTRACT: Use ReadFile and ExtractJsonField to examine configs
5. SEARCH PATTERNS: Use GrepFiles to find build commands, imports, requirements
6. CROSS-REFERENCE: Verify findings from web documentation with local files
7. **VERIFY DOCKER IMAGES**: If you plan to use a Docker base image (e.g., in a FROM instruction), you MUST verify it exists using `DockerImageSearch`.
   - **CRITICAL**: Do NOT use ancient/deprecated images (e.g., `python:2.7`, `node:0.10`, `ubuntu:trusty`). These cause "Manifest V1" errors and will fail.
   - **ALWAYS** prefer modern LTS versions (e.g., `node:18`, `python:3.10`, `ubuntu:22.04`) unless the project *strictly* requires legacy versions.
   - **ARCH CHECK**: Ensure the image supports {host_arch_name}.
8. **BUILD DEPENDENCIES**: If the project uses compiled languages (Python/Node with native modules, C++, etc.), ALWAYS install `build-essential`, `gcc`, `make`, or `python3-dev` in the Dockerfile.{doc_search_context}

CRITICAL FORMAT RULES (YOU MUST FOLLOW EXACTLY):
1. **NEVER write free text or explanations outside the format below**
2. After "Thought:", you MUST write "Action:" followed by the tool name on the SAME line
3. After "Action: <tool_name>", you MUST write "Action Input:" on the next line
4. After "Action Input:", write the input WITHOUT quotes on the SAME line
5. After "Action Input: <input>", STOP IMMEDIATELY - do not write anything else
6. Do NOT write "Observation:" - the system provides it
7. **If you want to provide a final answer, write "Final Answer:" after Thought, not free text**
8. **DO NOT wrap Action or Action Input in markdown code blocks (like ```)**
9. WHEN TO STOP AND PROVIDE FINAL ANSWER:
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
- **CRITICAL FOR FINAL QUERY**: If the question says "Based on information from Query 1, 2, 3" or mentions previous queries, you MUST use the information from chat_history and provide Final Answer immediately WITHOUT any tool calls (except optional DockerImageSearch for verification)
- When generating final deliverables (Dockerfile, etc.), use the accumulated context - do NOT start searching again

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
    
    # Insert host architecture
    template = template.replace("{host_arch_name}", host_arch_name)

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
            return "I made a format error. I MUST write 'Action: <tool_name>' on one line, then 'Action Input: <input>' on the next line. I cannot write free text or markdown code blocks for the Action.\n\nCorrect Example:\nThought: I need to search.\nAction: SearchWeb\nAction Input: query\n\nExample Final Answer:\nThought: I am done.\nFinal Answer: The result is..."
        return f"Format error: {error_str}. I must write 'Action: <tool_name>' then 'Action Input: <input>' on separate lines. If I am done, I must write 'Final Answer:'. Do not use markdown blocks."

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # Disable default verbose output
        callbacks=callbacks,
        handle_parsing_errors=handle_parsing_error,  # Use the function, not True
        max_iterations=max_iterations,
        max_execution_time=None,  # No time limit
        early_stopping_method="force",  # Force stop if max iterations reached (generate not supported)
        return_intermediate_steps=True  # Enable to track tool usage
    )

    logger.info("Planner agent created successfully")
    return agent_executor, callback_handler
