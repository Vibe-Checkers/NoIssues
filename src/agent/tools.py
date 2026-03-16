"""Agent tool definitions for BuildAgent v2.0.

Each tool has: name, description, Pydantic InputSchema, execute() → str.
All file operations are sandboxed within repo_root.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import requests
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

FILE_SIZE_LIMIT = 512 * 1024  # 512KB


# ─── Sandbox ─────────────────────────────────────────

class PathTraversalError(Exception):
    pass


def resolve_path(repo_root: Path, user_path: str) -> Path:
    """Resolve user_path within repo_root, blocking traversal."""
    resolved = (repo_root / user_path).resolve()
    if not resolved.is_relative_to(repo_root.resolve()):
        raise PathTraversalError(f"Path traversal blocked: {user_path}")
    return resolved


# ═══════════════════════════════════════════════════════
# Tool 1: ReadFile
# ═══════════════════════════════════════════════════════

class ReadFileInput(BaseModel):
    path: str = Field(description="Relative path to the file")


class ReadFileTool:
    name = "ReadFile"
    description = "Read a file from the repository. Input: {\"path\": \"relative/path/to/file\"}. Returns file content. Max 512KB."
    args_schema = ReadFileInput

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def execute(self, path: str) -> str:
        try:
            resolved = resolve_path(self.repo_root, path)
            if not resolved.is_file():
                return f"Error: file not found: {path}"
            if resolved.stat().st_size > FILE_SIZE_LIMIT:
                return f"Error: file exceeds 512KB limit ({resolved.stat().st_size} bytes)"
            return resolved.read_text(errors="replace")
        except PathTraversalError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {e}"


# ═══════════════════════════════════════════════════════
# Tool 2: ListDirectory
# ═══════════════════════════════════════════════════════

class ListDirectoryInput(BaseModel):
    path: str = Field(default=".", description="Relative path to directory")


class ListDirectoryTool:
    name = "ListDirectory"
    description = "List contents of a directory. Input: {\"path\": \"relative/path\"}. Defaults to repo root if path is \".\"."
    args_schema = ListDirectoryInput

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def execute(self, path: str = ".") -> str:
        try:
            resolved = resolve_path(self.repo_root, path)
            if not resolved.is_dir():
                return f"Error: not a directory: {path}"
            entries = []
            for item in sorted(resolved.iterdir()):
                rel = item.relative_to(self.repo_root)
                suffix = "/" if item.is_dir() else ""
                entries.append(f"{rel}{suffix}")
            return "\n".join(entries) if entries else "(empty directory)"
        except PathTraversalError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error listing directory: {e}"


# ═══════════════════════════════════════════════════════
# Tool 3: FindFiles
# ═══════════════════════════════════════════════════════

class FindFilesInput(BaseModel):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py'")


class FindFilesTool:
    name = "FindFiles"
    description = "Search for files matching a glob pattern. Input: {\"pattern\": \"**/*.py\"}. Returns matching file paths."
    args_schema = FindFilesInput

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def execute(self, pattern: str) -> str:
        try:
            # Strip leading **/ for glob — rglob already recurses
            glob_pattern = pattern.lstrip("*").lstrip("/") or "*"
            matches = []
            for item in self.repo_root.rglob(glob_pattern):
                if item.is_file():
                    rel = str(item.relative_to(self.repo_root))
                    matches.append(rel)
                    if len(matches) >= 200:
                        matches.append("... (truncated at 200 results)")
                        break
            return "\n".join(matches) if matches else "No files found matching pattern."
        except Exception as e:
            return f"Error finding files: {e}"


# ═══════════════════════════════════════════════════════
# Tool 4: GrepFiles
# ═══════════════════════════════════════════════════════

class GrepFilesInput(BaseModel):
    pattern: str = Field(description="Regex pattern to search for")
    path: str = Field(default=".", description="Optional directory to search in")


class GrepFilesTool:
    name = "GrepFiles"
    description = "Search file contents for a pattern. Input: {\"pattern\": \"regex pattern\", \"path\": \"optional/dir\"}. Returns matching lines."
    args_schema = GrepFilesInput

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def execute(self, pattern: str, path: str = ".") -> str:
        try:
            search_dir = resolve_path(self.repo_root, path)
            regex = re.compile(pattern)
            results = []
            for item in search_dir.rglob("*"):
                if not item.is_file() or item.stat().st_size > FILE_SIZE_LIMIT:
                    continue
                try:
                    text = item.read_text(errors="replace")
                    rel = item.relative_to(self.repo_root)
                    for i, line in enumerate(text.splitlines(), 1):
                        if regex.search(line):
                            results.append(f"{rel}:{i}: {line.rstrip()}")
                            if len(results) >= 100:
                                results.append("... (truncated at 100 matches)")
                                return "\n".join(results)
                except Exception:
                    continue
            return "\n".join(results) if results else "No matches found."
        except PathTraversalError as e:
            return f"Error: {e}"
        except re.error as e:
            return f"Error: invalid regex pattern: {e}"
        except Exception as e:
            return f"Error searching files: {e}"


# ═══════════════════════════════════════════════════════
# Tool 5: WriteFile
# ═══════════════════════════════════════════════════════

class WriteFileInput(BaseModel):
    path: str = Field(description="Relative path for the file")
    content: str = Field(description="File content to write")


class WriteFileTool:
    name = "WriteFile"
    description = "Write content to a file in the repository. Input: {\"path\": \"relative/path\", \"content\": \"file content\"}. For Dockerfiles, validates that FROM references a real image."
    args_schema = WriteFileInput

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def execute(self, path: str, content: str) -> str:
        try:
            resolved = resolve_path(self.repo_root, path)

            # FROM validation for Dockerfiles
            if resolved.name == "Dockerfile":
                validation_error = self._validate_from_lines(content)
                if validation_error:
                    return validation_error

            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content)
            return f"Written {len(content)} bytes to {path}"
        except PathTraversalError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {e}"

    def _validate_from_lines(self, content: str) -> str | None:
        """Validate FROM image references in Dockerfile. Returns error string or None."""
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("FROM "):
                parts = stripped.split()
                if len(parts) < 2:
                    continue
                image_ref = parts[1]
                # Skip build stage references (FROM ... AS ...)
                if image_ref.startswith("$") or image_ref == "scratch":
                    continue
                if not self._image_exists(image_ref):
                    return f"Error: base image not found on Docker Hub: {image_ref}"
        return None

    @staticmethod
    def _image_exists(image_ref: str) -> bool:
        """Check if a Docker image:tag exists on Docker Hub."""
        if ":" in image_ref:
            name, tag = image_ref.split(":", 1)
        else:
            name, tag = image_ref, "latest"

        # Handle library images (no namespace)
        if "/" not in name:
            name = f"library/{name}"

        try:
            url = f"https://hub.docker.com/v2/repositories/{name}/tags/{tag}"
            resp = requests.get(url, timeout=10)
            return resp.status_code == 200
        except Exception:
            # On network failure, allow the image (don't block the agent)
            return True


# ═══════════════════════════════════════════════════════
# Tool 6: DockerImageSearch
# ═══════════════════════════════════════════════════════

class DockerImageSearchInput(BaseModel):
    query: str = Field(description="Image name or image:tag to search for")


class DockerImageSearchTool:
    name = "DockerImageSearch"
    description = "Search Docker Hub for images or verify a tag exists. Input: {\"query\": \"image name or image:tag\"}. Returns available images and tags."
    args_schema = DockerImageSearchInput

    def execute(self, query: str) -> str:
        try:
            # If query looks like "name:tag", verify that specific tag
            if ":" in query and "/" not in query.split(":")[1]:
                name, tag = query.split(":", 1)
                return self._check_tag(name, tag)

            # Otherwise search
            return self._search(query)
        except Exception as e:
            return f"Error searching Docker Hub: {e}"

    def _check_tag(self, name: str, tag: str) -> str:
        lib_name = f"library/{name}" if "/" not in name else name
        url = f"https://hub.docker.com/v2/repositories/{lib_name}/tags/{tag}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return f"Image {name}:{tag} exists on Docker Hub."
        return f"Image {name}:{tag} NOT found on Docker Hub."

    def _search(self, query: str) -> str:
        url = f"https://hub.docker.com/v2/search/repositories/?query={query}&page_size=10"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for repo in data.get("results", []):
            name = repo.get("repo_name", "")
            desc = repo.get("short_description", "")[:80]
            stars = repo.get("star_count", 0)
            results.append(f"{name} ({stars} stars): {desc}")
        return "\n".join(results) if results else "No images found."


# ═══════════════════════════════════════════════════════
# Tool 7: SearchWeb
# ═══════════════════════════════════════════════════════

class SearchWebInput(BaseModel):
    query: str = Field(description="Search terms")


class SearchWebTool:
    name = "SearchWeb"
    description = "Search the web for Docker build solutions. Input: {\"query\": \"search terms\"}. Use when you encounter an error you don't know how to fix."
    args_schema = SearchWebInput

    def execute(self, query: str) -> str:
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=5):
                    results.append(f"{r['title']}\n{r['href']}\n{r['body'][:200]}\n")
            return "\n".join(results) if results else "No results found."
        except ImportError:
            return "Error: duckduckgo_search package not installed."
        except Exception as e:
            return f"Error searching web: {e}"


# ═══════════════════════════════════════════════════════
# Tool Registry
# ═══════════════════════════════════════════════════════

def create_tools(repo_root: Path) -> list:
    """Create all agent tools for the given repo root.

    Note: VerifyBuildTool is created separately in verify_build.py
    and appended to this list by the caller.
    """
    return [
        ReadFileTool(repo_root),
        ListDirectoryTool(repo_root),
        FindFilesTool(repo_root),
        GrepFilesTool(repo_root),
        WriteFileTool(repo_root),
        DockerImageSearchTool(),
        SearchWebTool(),
    ]
