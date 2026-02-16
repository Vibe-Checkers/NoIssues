
import os
import re
import json
import logging
import yaml
import platform
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)


# =============================================================================
# TOKEN TRACKING FOR PREPARATION LLM CALLS
# =============================================================================

class PreparationTokenTracker:
    """Track token usage across all preparation-phase LLM calls."""

    def __init__(self):
        self.total_input = 0
        self.total_output = 0
        self.calls = []

    def track(self, response, call_name: str = ""):
        """Extract and accumulate token usage from an LLM response."""
        try:
            usage = getattr(response, 'usage_metadata', None)
            if usage:
                inp = usage.get('input_tokens', 0)
                out = usage.get('output_tokens', 0)
            else:
                # Fallback: try response_metadata
                meta = getattr(response, 'response_metadata', {})
                token_usage = meta.get('token_usage', {})
                inp = token_usage.get('prompt_tokens', 0)
                out = token_usage.get('completion_tokens', 0)

            self.total_input += inp
            self.total_output += out
            self.calls.append({
                "name": call_name,
                "input_tokens": inp,
                "output_tokens": out,
            })
            logger.info(
                f"[PrepTokens] {call_name}: "
                f"{inp} in / {out} out"
            )
        except Exception as e:
            logger.debug(f"Could not track tokens for {call_name}: {e}")

    def summary(self) -> Dict[str, Any]:
        return {
            "input": self.total_input,
            "output": self.total_output,
            "total": self.total_input + self.total_output,
            "calls": self.calls,
        }


# =============================================================================
# COMPREHENSIVE LLM PRE-ANALYSIS
# =============================================================================

def _get_analysis_model() -> Optional[BaseChatModel]:
    """
    Get the model for pre-analysis. Configurable via environment variables.
    Falls back to the default agent model if not specified.

    Supports a separate Azure resource for the analysis model:
      ANALYSIS_MODEL_DEPLOYMENT - deployment name (e.g. gpt-5-chat)
      ANALYSIS_MODEL_ENDPOINT   - Azure endpoint (if different from agent)
      ANALYSIS_MODEL_API_KEY    - API key (if different from agent)
    """
    deployment = os.getenv("ANALYSIS_MODEL_DEPLOYMENT")
    if not deployment:
        return None

    # Use separate endpoint/key if provided, otherwise fall back to agent's
    endpoint = (
        os.getenv("ANALYSIS_MODEL_ENDPOINT")
        or os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    api_key = (
        os.getenv("ANALYSIS_MODEL_API_KEY")
        or os.getenv("AZURE_OPENAI_API_KEY")
    )
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

    try:
        return AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            temperature=0.1,
        )
    except Exception as e:
        logger.warning(f"Could not create analysis model: {e}")
        return None


def gather_build_context(repo_path: str, max_file_size: int = 8000) -> Dict[str, Any]:
    """
    Gather all build-relevant files and metadata from the repository.

    This collects the raw context that will be analyzed by the LLM.
    No interpretation here - just data gathering.

    Args:
        repo_path: Path to the repository
        max_file_size: Maximum characters to read per file (truncate if larger)

    Returns:
        Dictionary with all gathered context
    """
    repo = Path(repo_path)
    context = {
        "files": {},
        "directory_structure": [],
        "meta": {}
    }

    # Files to look for (prioritized by importance for Docker builds)
    build_files = [
        # Package managers / dependencies
        "package.json",
        "package-lock.json",  # Just check existence, don't read content
        "pnpm-lock.yaml",     # Just check existence
        "yarn.lock",          # Just check existence
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "setup.cfg",
        "Pipfile",
        "Cargo.toml",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "Gemfile",
        "composer.json",
        # Build configuration
        "Makefile",
        "Dockerfile",
        ".dockerignore",
        "docker-compose.yml",
        "docker-compose.yaml",
        # Test configuration
        "karma.conf.js",
        "karma.conf.cjs",
        "jest.config.js",
        "jest.config.ts",
        "vitest.config.ts",
        "pytest.ini",
        "tox.ini",
        ".nycrc",
        # Runtime version specs
        ".nvmrc",
        ".node-version",
        ".python-version",
        ".ruby-version",
        ".tool-versions",
        # Documentation (often has build instructions)
        "README.md",
        "CONTRIBUTING.md",
    ]

    # Lockfiles - just note existence, don't read content
    lockfiles = ["package-lock.json", "pnpm-lock.yaml", "yarn.lock", "Pipfile.lock", "poetry.lock", "Cargo.lock"]

    for filename in build_files:
        filepath = repo / filename
        if filepath.exists():
            if filename in lockfiles:
                # Just note that it exists
                context["files"][filename] = "[EXISTS - content not included]"
            else:
                try:
                    content = filepath.read_text(encoding='utf-8', errors='ignore')
                    if len(content) > max_file_size:
                        content = content[:max_file_size] + f"\n... [TRUNCATED - {len(content)} total chars]"
                    context["files"][filename] = content
                except Exception as e:
                    context["files"][filename] = f"[ERROR reading file: {e}]"

    # Gather CI workflow files
    workflows_dir = repo / ".github" / "workflows"
    if workflows_dir.exists():
        for wf_file in list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml")):
            rel_path = f".github/workflows/{wf_file.name}"
            try:
                content = wf_file.read_text(encoding='utf-8', errors='ignore')
                if len(content) > max_file_size:
                    content = content[:max_file_size] + "\n... [TRUNCATED]"
                context["files"][rel_path] = content
            except Exception as e:
                context["files"][rel_path] = f"[ERROR: {e}]"

    # Get directory structure (top 2 levels)
    try:
        for item in sorted(repo.iterdir()):
            if item.name.startswith('.') and item.name not in ['.github', '.dockerignore']:
                continue
            if item.is_dir():
                context["directory_structure"].append(f"📁 {item.name}/")
                # One level deeper
                try:
                    for subitem in sorted(item.iterdir())[:10]:  # Limit subdirectory items
                        if not subitem.name.startswith('.'):
                            prefix = "📁" if subitem.is_dir() else "📄"
                            context["directory_structure"].append(f"   {prefix} {item.name}/{subitem.name}")
                except PermissionError:
                    pass
            else:
                context["directory_structure"].append(f"📄 {item.name}")
    except Exception as e:
        context["directory_structure"].append(f"[Error listing directory: {e}]")

    # Meta information
    context["meta"]["platform"] = platform.machine()
    context["meta"]["is_arm64"] = platform.machine().lower() in ['arm64', 'aarch64']
    context["meta"]["files_found"] = list(context["files"].keys())

    return context


def analyze_repository_for_docker(
    llm: BaseChatModel,
    repo_path: str,
    language: str,
    gathered_context: Optional[Dict[str, Any]] = None,
    tracker: Optional['PreparationTokenTracker'] = None
) -> str:
    """
    Use LLM to comprehensively analyze the repository for Docker build challenges.

    This is the core pre-analysis function. The LLM examines all build-relevant
    files and produces actionable guidance for Dockerfile creation.

    Args:
        llm: The language model to use for analysis
        repo_path: Path to the repository
        language: Detected primary language
        gathered_context: Pre-gathered context (if None, will gather)

    Returns:
        LLM-generated analysis as a string
    """
    if gathered_context is None:
        gathered_context = gather_build_context(repo_path)

    # Build the file contents section
    files_section = ""
    for filename, content in gathered_context["files"].items():
        files_section += f"\n{'='*60}\nFILE: {filename}\n{'='*60}\n{content}\n"

    # Directory structure
    dir_structure = "\n".join(gathered_context["directory_structure"][:50])  # Limit lines

    is_arm = gathered_context["meta"].get("is_arm64", False)

    system_prompt = """You are a senior DevOps engineer. Your analysis will be fed directly
to an autonomous agent that creates Dockerfiles. Be SPECIFIC and ACTIONABLE.
Every recommendation must include the exact command or Dockerfile line.
Do NOT give generic advice. Analyze the ACTUAL files provided.

IMPORTANT — The agent has NO direct shell access. It can ONLY:
  1. Write Dockerfile / run_tests.sh and test them via VerifyBuild
  2. Use RunInContainer for read-only diagnostics inside the built image
  3. Use SearchDockerError for AI-powered fix suggestions after build failures
  4. Use DiagnoseTestFailure for AI analysis of test failures
It CANNOT run apt-get, npm, or any command directly on the host.

CRITICAL ARCHITECTURE — TWO-FILE APPROACH:
  - Dockerfile: builds the image ONLY (install deps, compile). NO test execution.
  - run_tests.sh: a separate shell script that runs the tests INSIDE the container.
  - VerifyBuild mounts and runs run_tests.sh after a successful image build.
  NEVER suggest putting test execution (e.g. RUN npm test, RUN pytest) inside the Dockerfile.
  ALL test commands belong in run_tests.sh, not in any Dockerfile RUN or CMD instruction.

Design recommendations as Dockerfile RUN instructions, not host commands.
When you warn about a pitfall, suggest the SearchDockerError keywords the agent
should use if it hits that error (e.g., 'EACCES permission denied', 'husky not found')."""

    user_prompt = f"""Analyze this {language} repository for Docker containerization.

PLATFORM: {"ARM64 (Apple Silicon)" if is_arm else "x86_64"}

DIRECTORY STRUCTURE:
{dir_structure}

{files_section}

Produce a structured analysis with these EXACT sections:

## BASE IMAGE
Recommend the exact FROM line. Justify version choice based on files you see.

## SYSTEM DEPENDENCIES
List every apt-get package needed. Check for:
- Native compilation needs (gcc, g++, make, cmake)
- Library headers (-dev packages) referenced in build configs
- Browser/GUI needs for testing (chromium, xvfb)
- Git (if submodules or git-based deps exist)

## BUILD STEPS (in exact order)
1. What to COPY first (lock files for caching)
2. Dependency install command (exact command with flags)
3. Build/compile command (if needed)
List the exact RUN lines.
⚠️ DO NOT include any test execution here. The Dockerfile only builds the image.
Tests will be run via a separate run_tests.sh script AFTER the image is built.

## TEST ENVIRONMENT
The Dockerfile must be fully test-compatible. Specify:
- Test framework detected (from config files)
- Test command that will go in run_tests.sh
- All ENV vars the test suite needs → add as ENV instructions IN THE DOCKERFILE
  (e.g. ENV CI=true, ENV CHROME_BIN=/usr/bin/chromium, ENV DISPLAY=:99)
- Any extra system packages needed ONLY for tests (browsers, xvfb, etc.) → add to apt-get in Dockerfile
- Does testing need: browsers? display server (xvfb-run)? network? services?
- Provide a ready-to-use run_tests.sh template using #!/bin/sh with set -e
  (run_tests.sh only contains the test execution command — all env setup is in the Dockerfile)

## CRITICAL WARNINGS
Things that WILL break if not handled:
- Interactive prompts (tzdata, etc.)
- Post-install scripts that need git/network (husky, prepare hooks)
- Platform-specific packages unavailable on this architecture
- Monorepo/workspace setup requiring special install commands
- Files that must NOT be in .dockerignore"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        if tracker:
            tracker.track(response, "repo_analysis")
        return response.content
    except Exception as e:
        logger.error(f"LLM analysis failed: {e}")
        return f"[Analysis failed: {e}]"

def detect_project_language(repo_path: str) -> str:
    """
    Detect the primary programming language of a repository.
    PRIORITIZES build system files over source files.
    Uses ordered checks: framework-specific > package manager > build system > file extensions.
    """
    try:
        if not os.path.exists(repo_path):
            return "Unknown"

        def _exists(filename):
            return os.path.exists(os.path.join(repo_path, filename))

        # Tier 1: Framework-specific files (highest confidence)
        framework_indicators = [
            ('angular.json', 'Angular'),
            ('next.config.js', 'Next.js'),
            ('next.config.mjs', 'Next.js'),
            ('next.config.ts', 'Next.js'),
            ('nuxt.config.js', 'Node.js'),
            ('nuxt.config.ts', 'Node.js'),
            ('svelte.config.js', 'Node.js'),
        ]
        for filename, language in framework_indicators:
            if _exists(filename):
                logger.info(f"Detected language {language} from framework file {filename}")
                return language

        # Tier 2: Package manager / build system files (ordered by specificity)
        # More specific files first, ambiguous ones (Makefile) last
        build_indicators = [
            # Node.js ecosystem
            ('package.json', 'Node.js'),
            # Python ecosystem
            ('pyproject.toml', 'Python'),
            ('requirements.txt', 'Python'),
            ('setup.py', 'Python'),
            ('setup.cfg', 'Python'),
            ('Pipfile', 'Python'),
            # JVM ecosystem
            ('pom.xml', 'Java'),
            ('build.gradle', 'Java'),
            ('build.gradle.kts', 'Kotlin'),
            # Other languages
            ('go.mod', 'Go'),
            ('Cargo.toml', 'Rust'),
            ('Gemfile', 'Ruby'),
            ('composer.json', 'PHP'),
            ('mix.exs', 'Elixir'),
            ('cabal.project', 'Haskell'),
            ('stack.yaml', 'Haskell'),
            ('pubspec.yaml', 'Dart'),
            ('CMakeLists.txt', 'C++'),
            ('meson.build', 'C/C++'),
            # Makefile is ambiguous — only use as last resort in this tier
        ]

        for filename, language in build_indicators:
            if _exists(filename):
                logger.info(f"Detected language {language} based on {filename}")
                return language

        # Tier 3: Fallback to file extension counting
        extension_map = {
            '.py': 'Python',
            '.js': 'Node.js', '.jsx': 'Node.js',
            '.ts': 'Node.js', '.tsx': 'Node.js',
            '.mjs': 'Node.js', '.cjs': 'Node.js',
            '.java': 'Java',
            '.kt': 'Kotlin', '.kts': 'Kotlin',
            '.go': 'Go',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.rs': 'Rust',
            '.c': 'C/C++', '.h': 'C/C++',
            '.cpp': 'C++', '.cc': 'C++', '.cxx': 'C++', '.hpp': 'C++',
            '.cs': 'C#',
            '.swift': 'Swift',
            '.scala': 'Scala',
            '.ex': 'Elixir', '.exs': 'Elixir',
        }

        extension_counts = {}
        repo_path_abs = os.path.abspath(repo_path)
        for root, dirs, files in os.walk(repo_path_abs):
            # Skip hidden directories and common non-source dirs
            dirs[:] = [
                d for d in dirs
                if not d.startswith('.') and d not in (
                    'node_modules', 'vendor', '__pycache__',
                    'venv', '.venv', 'dist', 'build', 'target'
                )
            ]
            # Don't go deeper than 3 levels
            rel_depth = root[len(repo_path_abs):].count(os.sep)
            if rel_depth > 3:
                continue

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                lang = extension_map.get(ext)
                if lang:
                    extension_counts[lang] = extension_counts.get(lang, 0) + 1

        if extension_counts:
            most_common = max(extension_counts.items(), key=lambda x: x[1])[0]
            logger.info(
                f"Detected language {most_common} based on "
                f"file extensions {extension_counts}"
            )
            return most_common

        # Tier 4: Makefile-only projects (very last resort)
        if _exists('Makefile'):
            logger.info("Detected C/C++ based on Makefile (last resort)")
            return "C/C++"

        return "Unknown"
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return "Unknown"

def generate_language_guidelines(
    llm: BaseChatModel,
    language: str,
    tracker: Optional['PreparationTokenTracker'] = None
) -> str:
    """
    Meta-prompting: generate language-specific guidelines for building
    and testing projects in Docker containers.

    Args:
        llm: The base chat model (NOT the agent executor)
        language: Detected programming language
        tracker: Optional token tracker

    Returns:
        Guidelines as formatted string
    """
    if not language or language == "Unknown":
        return ""

    prompt = f"""I need to build a {language} project from source inside a Docker container
and run its full test suite. Give me exactly 10 guidelines I should follow, covering:

1. How to identify the correct base Docker image and runtime version for {language}
2. Common package managers and how to install ALL dependencies (including dev/test deps)
3. How to detect and handle build systems (e.g., the standard ones for {language})
4. How to identify the correct test command from project config files
5. Common native/system dependencies that {language} projects need (compilers, libs, headers)
6. How to handle version pinning and lock files
7. Common pitfalls when building {language} projects in a minimal Docker container
8. How to handle projects that use multiple languages alongside {language}
9. Environment variables commonly needed for {language} builds and test suites
10. How to parse test results (pass/fail/skip counts) from common {language} test frameworks

For each guideline, include:
- WHY it matters
- WHAT TO DO (specific commands or config file locations to check)
- COMMON MISTAKES to avoid

Be specific to {language} — reference actual tools, file names, and commands."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        if tracker:
            tracker.track(response, "language_guidelines")
        return response.content
    except Exception as e:
        logger.error(f"Failed to generate language guidelines: {e}")
        return ""


def generate_container_guidelines(
    llm: BaseChatModel,
    tracker: Optional['PreparationTokenTracker'] = None
) -> str:
    """
    Meta-prompting: generate containerization best practices.
    These are language-independent Docker guidelines.
    """
    prompt = """I need to containerize arbitrary software projects for building and testing.
Give me 8 actionable guidelines about modern Docker containerization, covering:

1. Choosing between ubuntu, debian-slim, alpine, and language-specific base images — when to use which
2. Handling interactive prompts (DEBIAN_FRONTEND, -y flags, tzdata, etc.)
3. Layer caching strategy: what to COPY first for optimal rebuild speed
4. Handling platform/architecture issues (ARM64 vs AMD64, emulation, --platform flags)
5. Managing apt-get correctly (update + install in one RUN, cleaning cache, pinning versions)
6. Setting up non-root users vs running as root for test execution
7. Handling git submodules and .git directory inside containers
8. Timeout and resource management for long-running builds

For each, give the concrete Dockerfile snippet or command. Focus on 2024/2025 best practices."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        if tracker:
            tracker.track(response, "container_guidelines")
        return response.content
    except Exception as e:
        logger.error(f"Failed to generate container guidelines: {e}")
        return ""


def generate_universal_build_guidelines(
    llm: BaseChatModel,
    tracker: Optional['PreparationTokenTracker'] = None
) -> str:
    """
    Meta-prompting: language-independent guidelines for building
    any project from source and running its tests in Docker.
    """
    prompt = """Give me 8 universal guidelines for building ANY software project from source
and running its tests inside a Docker container, regardless of programming language:

1. How to identify the build system from project files (what files to look for, in what order)
2. How to find the test command when documentation is missing or outdated
3. How to handle projects with git submodules
4. How to deal with projects that need network access during build vs test
5. How to handle projects that need databases, message queues, or external services for tests
6. Common reasons why tests pass in CI but fail in a fresh Docker container
7. How to handle flaky or environment-dependent tests (skip vs fix)
8. How to extract test results (pass/fail/skip counts) from arbitrary test output

Be concrete — give actual commands and patterns to look for."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        if tracker:
            tracker.track(response, "universal_guidelines")
        return response.content
    except Exception as e:
        logger.error(f"Failed to generate universal guidelines: {e}")
        return ""


def summarize_ci_scripts_with_llm(
    llm: BaseChatModel,
    repo_path: str,
    max_files: int = 3,
    tracker: Optional['PreparationTokenTracker'] = None
) -> str:
    """
    LLM-powered CI/CD script summarization.
    Instead of just dumping run: lines, the LLM extracts structured
    dependencies, build/test commands, and environment variables.

    Args:
        llm: The language model to use
        repo_path: Path to the repository
        max_files: Maximum CI files to process

    Returns:
        LLM-generated CI/CD summary
    """
    ci_files = {}

    # Collect CI files from multiple possible locations
    search_paths = [
        (Path(repo_path) / ".github" / "workflows", "*.yml"),
        (Path(repo_path) / ".github" / "workflows", "*.yaml"),
        (Path(repo_path) / ".circleci", "config.yml"),
        (Path(repo_path), "Jenkinsfile"),
        (Path(repo_path), ".travis.yml"),
    ]

    for search_dir, pattern in search_paths:
        if not search_dir.exists():
            continue
        if '*' in pattern:
            files = list(search_dir.glob(pattern))
        else:
            f = search_dir / pattern if search_dir.is_dir() else search_dir
            files = [f] if f.exists() else []

        for f in files[:max_files]:
            try:
                content = f.read_text(encoding='utf-8', errors='ignore')
                if len(content) > 6000:
                    content = content[:6000] + "\n... [TRUNCATED]"
                rel_path = str(f.relative_to(repo_path))
                ci_files[rel_path] = content
            except Exception:
                pass

    if not ci_files:
        return "No CI/CD configuration files found."

    # LLM summarization
    files_text = ""
    for path, content in ci_files.items():
        files_text += f"\n--- FILE: {path} ---\n{content}\n"

    prompt = f"""Analyze these CI/CD configuration files and extract information relevant
to building and testing this project locally in a Docker container.

{files_text}

For each file, provide a JSON-like summary:
{{
  "summary": "<what this CI workflow does in 1-2 sentences>",
  "extracted_dependencies": ["<system packages and tools it installs>"],
  "build_commands": ["<commands used to build the project>"],
  "test_commands": ["<commands used to run tests>"],
  "environment_variables": ["<env vars set, especially for testing>"],
  "important_notes": ["<anything unusual — custom scripts, services needed, special flags>"]
}}

Focus on what I need to REPRODUCE this locally in Docker.
Ignore deployment steps, artifact publishing, and notification steps.
If a file is not relevant to build/test, say so briefly."""

    try:
        response = llm.invoke([
            SystemMessage(content="You are a CI/CD expert. Extract actionable build/test information."),
            HumanMessage(content=prompt)
        ])
        if tracker:
            tracker.track(response, "ci_summary")
        return response.content
    except Exception as e:
        logger.error(f"CI/CD LLM summarization failed: {e}")
        return _fallback_workflow_summary(repo_path)


def _fallback_workflow_summary(repo_path: str) -> str:
    """Basic non-LLM fallback if the LLM CI/CD summarization fails."""
    workflows_dir = Path(repo_path) / ".github" / "workflows"
    if not workflows_dir.exists():
        return "No .github/workflows found."

    lines = ["CI/CD files found (raw):"]
    try:
        files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
        for wf in files:
            lines.append(f"\n- {wf.name}")
            try:
                content = wf.read_text(encoding='utf-8', errors='ignore')
                for line in content.split('\n'):
                    stripped = line.strip()
                    if stripped.startswith('run:') or stripped.startswith('- run:'):
                        cmd = stripped.split(':', 1)[1].strip()
                        lines.append(f"    run: {cmd}")
            except Exception:
                pass
        return "\n".join(lines)
    except Exception as e:
        return f"Error reading workflows: {e}"

def _gather_test_relevant_files(repo_path: str, max_file_size: int = 6000) -> Dict[str, str]:
    """
    Gather all files that could contain test configuration or test command hints.
    Returns a dict of {relative_path: content}.
    """
    repo = Path(repo_path)
    test_files = {}

    # Direct test config files
    test_config_files = [
        "package.json", "Makefile", "pyproject.toml", "tox.ini", "setup.cfg",
        "setup.py", "pytest.ini", "conftest.py", ".nycrc", ".nycrc.json",
        "jest.config.js", "jest.config.ts", "jest.config.mjs", "jest.config.cjs",
        "vitest.config.ts", "vitest.config.js", "vitest.config.mts",
        "karma.conf.js", "karma.conf.cjs",
        ".mocharc.yml", ".mocharc.json", ".mocharc.js",
        "phpunit.xml", "phpunit.xml.dist",
        "build.gradle", "build.gradle.kts", "pom.xml",
        "Cargo.toml", "go.mod",
        "Gemfile", "Rakefile", ".rspec",
        "mix.exs", "CMakeLists.txt",
        "docker-compose.yml", "docker-compose.yaml",
        "docker-compose.test.yml", "docker-compose.ci.yml",
        "README.md", "CONTRIBUTING.md",
        ".github/CONTRIBUTING.md",
    ]

    for filename in test_config_files:
        filepath = repo / filename
        if filepath.exists():
            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
                if len(content) > max_file_size:
                    content = content[:max_file_size] + "\n... [TRUNCATED]"
                test_files[filename] = content
            except Exception:
                pass

    # CI workflow files (critical for test discovery)
    ci_dirs = [
        (repo / ".github" / "workflows", "*.yml"),
        (repo / ".github" / "workflows", "*.yaml"),
    ]
    for ci_dir, pattern in ci_dirs:
        if ci_dir.exists():
            for wf_file in list(ci_dir.glob(pattern))[:5]:
                rel_path = f".github/workflows/{wf_file.name}"
                try:
                    content = wf_file.read_text(encoding='utf-8', errors='ignore')
                    if len(content) > max_file_size:
                        content = content[:max_file_size] + "\n... [TRUNCATED]"
                    test_files[rel_path] = content
                except Exception:
                    pass

    # Also check for .travis.yml, .circleci/config.yml, Jenkinsfile
    for ci_file in [".travis.yml", ".circleci/config.yml", "Jenkinsfile"]:
        filepath = repo / ci_file
        if filepath.exists():
            try:
                content = filepath.read_text(encoding='utf-8', errors='ignore')
                if len(content) > max_file_size:
                    content = content[:max_file_size] + "\n... [TRUNCATED]"
                test_files[ci_file] = content
            except Exception:
                pass

    # Scan for test directories to understand test structure
    test_dir_indicators = []
    for d in ["tests", "test", "spec", "__tests__", "test_suite", "testing"]:
        test_dir = repo / d
        if test_dir.is_dir():
            try:
                items = list(test_dir.iterdir())[:20]
                listing = [f"  {item.name}" for item in items]
                test_dir_indicators.append(f"{d}/:\n" + "\n".join(listing))
            except Exception:
                pass
    if test_dir_indicators:
        test_files["__TEST_DIRECTORIES__"] = "\n".join(test_dir_indicators)

    return test_files


def discover_test_command(
    repo_path: str,
    language: str,
    llm: Optional[BaseChatModel] = None,
    tracker: Optional['PreparationTokenTracker'] = None
) -> Optional[Dict[str, str]]:
    """
    Discover the test command using GPT-5 deep analysis of ALL test-relevant files.

    This replaces the old heuristic-only approach. GPT-5 analyzes:
      - CI workflows (GitHub Actions, Travis, CircleCI, Jenkins)
      - Build system files (Makefile, package.json, pyproject.toml, pom.xml, etc.)
      - Test config files (jest.config, pytest.ini, karma.conf, etc.)
      - README/CONTRIBUTING for test instructions
      - Test directory structure

    Falls back to heuristics only if LLM is not available.

    Returns:
        Dict with 'command', 'source', 'confidence', 'setup_commands',
        'env_vars', 'test_framework', 'skip_patterns' or None
    """
    repo = Path(repo_path)

    # Gather all test-relevant files
    test_files = _gather_test_relevant_files(repo_path)

    if not test_files:
        logger.warning("No test-relevant files found in repository")
        return _heuristic_test_command_fallback(repo_path, language)

    # If LLM available, use GPT-5 for deep analysis
    if llm is not None:
        try:
            result = _llm_discover_test_command(llm, test_files, language, repo_path, tracker)
            if result:
                return result
            logger.warning("LLM test discovery returned no result, falling back to heuristics")
        except Exception as e:
            logger.warning(f"LLM test discovery failed: {e}, falling back to heuristics")

    return _heuristic_test_command_fallback(repo_path, language)


def _llm_discover_test_command(
    llm: BaseChatModel,
    test_files: Dict[str, str],
    language: str,
    repo_path: str,
    tracker: Optional['PreparationTokenTracker'] = None
) -> Optional[Dict[str, str]]:
    """
    Use GPT-5 to deeply analyze test configuration and discover the correct test command.
    Returns enriched test command info including setup steps and environment variables.
    """
    # Build file contents section
    files_section = ""
    for filename, content in test_files.items():
        files_section += f"\n{'='*50}\nFILE: {filename}\n{'='*50}\n{content}\n"

    system_prompt = """You are an expert at analyzing software repositories to discover how to run their test suites.

You must analyze ALL provided files carefully and determine:
1. The EXACT test command the project uses
2. What test framework is being used
3. Any setup steps needed BEFORE running tests
4. Environment variables needed for tests
5. Any tests that should be skipped in a Docker container (network, GUI, integration tests)

CRITICAL RULES:
- Extract the ACTUAL test command from CI workflows or config files — don't guess
- If CI uses multiple test stages, pick the PRIMARY unit test command (not lint, not e2e)
- If package.json scripts.test exists and is not "no test specified" or "echo \\"Error: no test\\"", use "npm test"
- For Python: check if the project uses pytest, unittest, nose, or tox
- For Java: check Maven (mvn test), Gradle (./gradlew test), or custom
- For Go: typically "go test ./..." but check CI for flags like -race, -v, -count=1
- For Rust: typically "cargo test" but check for features or workspace flags
- If the project needs a build step before tests, include it in setup_commands
- Prefer the command from CI over README, prefer README over pure heuristics

You MUST respond in valid JSON only — no markdown, no explanation outside the JSON."""

    user_prompt = f"""Analyze this {language} repository's test configuration and discover the test command.

{files_section}

Respond with this exact JSON structure:
{{
    "test_command": "<the exact command to run tests>",
    "test_framework": "<framework name: pytest/jest/mocha/junit/go-test/cargo-test/rspec/etc>",
    "source": "<which file you found this in and why>",
    "confidence": "<high/medium/low>",
    "setup_commands": ["<commands to run BEFORE the test command, e.g. 'pip install -e .[test]'>"],
    "env_vars": {{"KEY": "VALUE"}},
    "skip_patterns": ["<patterns for tests to skip in Docker, e.g. 'not integration'>"],
    "needs_build_first": <true/false>,
    "build_command": "<build command if needs_build_first is true, else null>",
    "notes": "<any important observations about the test setup>"
}}

If you cannot determine the test command with any confidence, respond:
{{"test_command": null, "reason": "<why>"}}"""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        if tracker:
            tracker.track(response, "test_command_discovery")

        # Parse response
        content = response.content.strip()
        # Strip markdown fences if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        result = json.loads(content)

        if not result.get("test_command"):
            logger.info(f"LLM could not determine test command: {result.get('reason', 'unknown')}")
            return None

        logger.info(
            f"LLM discovered test command: {result['test_command']} "
            f"(framework: {result.get('test_framework', 'unknown')}, "
            f"confidence: {result.get('confidence', 'unknown')}, "
            f"source: {result.get('source', 'unknown')})"
        )

        return {
            "command": result["test_command"],
            "source": result.get("source", "LLM analysis"),
            "confidence": result.get("confidence", "medium"),
            "test_framework": result.get("test_framework", "unknown"),
            "setup_commands": result.get("setup_commands", []),
            "env_vars": result.get("env_vars", {}),
            "skip_patterns": result.get("skip_patterns", []),
            "needs_build_first": result.get("needs_build_first", False),
            "build_command": result.get("build_command"),
            "notes": result.get("notes", ""),
        }

    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM test discovery response: {e}")
        return None
    except Exception as e:
        logger.warning(f"LLM test discovery failed: {e}")
        return None


def _heuristic_test_command_fallback(repo_path: str, language: str) -> Optional[Dict[str, str]]:
    """
    Fallback heuristic test command discovery (used when LLM is unavailable).
    """
    repo = Path(repo_path)

    # Known test command patterns to look for in CI
    test_patterns = re.compile(
        r'\b(pytest|py\.test|python\s+-m\s+pytest|npm\s+test|npx\s+jest|'
        r'mvn\s+test|gradle\s+test|go\s+test|cargo\s+test|make\s+(?:test|check)|'
        r'ctest|phpunit|rspec|bundle\s+exec\s+rspec|mix\s+test)\b',
        re.IGNORECASE
    )

    # 1. CI Workflows
    workflows_dir = repo / ".github" / "workflows"
    if workflows_dir.exists():
        try:
            for wf_file in list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml")):
                with open(wf_file, 'r') as f:
                    content = yaml.safe_load(f)
                if not isinstance(content, dict):
                    continue
                jobs = content.get('jobs', {})
                for job_name, job_data in jobs.items():
                    if not isinstance(job_data, dict):
                        continue
                    for step in job_data.get('steps', []):
                        run_cmd = step.get('run', '')
                        if isinstance(run_cmd, str):
                            match = test_patterns.search(run_cmd)
                            if match:
                                for line in run_cmd.strip().split('\n'):
                                    if test_patterns.search(line):
                                        cmd = line.strip()
                                        return {
                                            "command": cmd,
                                            "source": f".github/workflows/{wf_file.name} (job: {job_name})",
                                            "confidence": "high"
                                        }
        except Exception as e:
            logger.debug(f"Error parsing CI workflows: {e}")

    # 2. Makefile
    makefile = repo / "Makefile"
    if makefile.exists():
        try:
            with open(makefile, 'r') as f:
                content = f.read()
            for target in ['test', 'check', 'tests']:
                pattern = re.compile(rf'^{target}\s*:', re.MULTILINE)
                if pattern.search(content):
                    return {"command": f"make {target}", "source": "Makefile", "confidence": "high"}
        except Exception:
            pass

    # 3. package.json
    pkg_json = repo / "package.json"
    if pkg_json.exists():
        try:
            with open(pkg_json, 'r') as f:
                pkg = json.load(f)
            test_script = pkg.get('scripts', {}).get('test', '')
            if test_script and 'no test specified' not in test_script.lower():
                return {"command": "npm test", "source": f"package.json (scripts.test = {test_script})", "confidence": "high"}
        except Exception:
            pass

    # 4. pyproject.toml
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        try:
            with open(pyproject, 'r') as f:
                content = f.read()
            if '[tool.pytest' in content or 'pytest' in content.lower():
                return {"command": "python -m pytest", "source": "pyproject.toml", "confidence": "high"}
        except Exception:
            pass

    # 5. Language-based fallback
    fallbacks = {
        'Python': 'python -m pytest', 'Node.js': 'npm test', 'Java': 'mvn test',
        'Kotlin': './gradlew test', 'Go': 'go test ./...', 'Rust': 'cargo test',
        'Ruby': 'bundle exec rspec', 'PHP': './vendor/bin/phpunit',
        'Elixir': 'mix test', 'C/C++': 'make check', 'C++': 'ctest',
    }
    if language in fallbacks:
        return {"command": fallbacks[language], "source": f"language fallback ({language})", "confidence": "low"}

    return None


def analyze_test_environment(
    llm: BaseChatModel,
    repo_path: str,
    language: str,
    test_command: Optional[Dict[str, str]] = None,
    gathered_context: Optional[Dict[str, Any]] = None,
    tracker: Optional['PreparationTokenTracker'] = None
) -> str:
    """
    GPT-5 powered deep analysis of the test environment requirements.

    This is a NEW preparation step that analyzes what the test suite needs to run
    successfully inside Docker — system packages, env vars, services, file permissions,
    build artifacts, etc.

    Returns a structured analysis string to be included in agent context.
    """
    test_files = _gather_test_relevant_files(repo_path)

    # Build file contents section
    files_section = ""
    for filename, content in test_files.items():
        files_section += f"\n--- {filename} ---\n{content}\n"

    test_cmd_info = ""
    if test_command:
        test_cmd_info = f"""
DISCOVERED TEST COMMAND: {test_command.get('command', 'N/A')}
TEST FRAMEWORK: {test_command.get('test_framework', 'unknown')}
SETUP COMMANDS: {test_command.get('setup_commands', [])}
ENV VARS NEEDED: {test_command.get('env_vars', {})}
SKIP PATTERNS: {test_command.get('skip_patterns', [])}
NEEDS BUILD FIRST: {test_command.get('needs_build_first', False)}
BUILD COMMAND: {test_command.get('build_command', 'N/A')}
NOTES: {test_command.get('notes', '')}
"""

    system_prompt = """You are a senior DevOps engineer specialized in running test suites inside Docker containers.
Analyze the repository files and produce a COMPLETE test environment specification.

Your analysis will be fed directly to an autonomous agent that creates:
1. A Dockerfile (must include ALL test dependencies as system packages and ENV vars)
2. A run_tests.sh script (runs inside the container after build)

Be EXTREMELY specific and actionable. Every recommendation must be copy-paste ready."""

    user_prompt = f"""Analyze this {language} repository and determine EVERYTHING needed to run its test suite inside Docker.

{test_cmd_info}

PROJECT FILES:
{files_section}

Produce a structured analysis with these EXACT sections:

## TEST FRAMEWORK & COMMAND
- Exact test framework and version (if detectable)
- The exact command to run in run_tests.sh
- Any flags needed (--no-coverage, --forceExit, -x, etc.)

## PRE-TEST SETUP (for run_tests.sh)
Steps that must happen BEFORE running the test command:
- Dependency installation that can't go in Dockerfile (e.g., pip install -e .)
- Database migrations or fixtures
- Git submodule initialization
- File permission fixes
- Creating temp directories

## DOCKERFILE REQUIREMENTS (for test compatibility)
System packages, ENV vars, and config the Dockerfile MUST include:
- List every apt-get package (e.g., chromium, xvfb, sqlite3, libpq-dev)
- List every ENV var (e.g., CI=true, NODE_ENV=test, DISPLAY=:99)
- Any config files that must exist (e.g., .env.test)
- WORKDIR considerations

## TESTS TO SKIP IN DOCKER
Tests that will ALWAYS fail in a Docker container:
- Browser/GUI tests without headless support
- Tests requiring external services (databases, Redis, etc.)
- Network-dependent tests
- Tests requiring specific OS features (systemd, etc.)
Provide the exact skip flags/patterns for the test framework.

## COMMON FAILURE PATTERNS
Known issues for this specific tech stack in Docker:
- Permission errors (node_modules, .cache)
- Missing native dependencies
- Timezone issues
- Locale issues
- Memory limits

## READY-TO-USE run_tests.sh
Provide a complete, ready-to-use run_tests.sh script."""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        if tracker:
            tracker.track(response, "test_environment_analysis")
        return response.content
    except Exception as e:
        logger.error(f"Test environment analysis failed: {e}")
        return f"[Test environment analysis failed: {e}]"


def build_initial_context(llm: BaseChatModel, repo_path: str, skip_analysis: bool = False) -> dict:
    """
    Build the complete preparation phase context.
    Aggregates language detection, LLM pre-analysis, meta-prompted guidelines,
    LLM-powered CI/CD summarization, and test command discovery.

    Args:
        llm: The base chat model for generating guidelines
        repo_path: Path to repository
        skip_analysis: If True, skip the comprehensive LLM analysis

    Returns:
        Dictionary with language, context string, and optional test_command
    """
    language = detect_project_language(repo_path)
    logger.info(f"Detected language: {language}")

    # Gather build context first (needed for analysis)
    gathered_context = gather_build_context(repo_path)

    # Use dedicated analysis model (GPT-5) if available, fall back to agent model
    analysis_model = _get_analysis_model() or llm
    logger.info(f"Analysis model: {getattr(analysis_model, 'deployment_name', 'default')}")

    # Token tracker for all preparation-phase LLM calls (GPT-5)
    tracker = PreparationTokenTracker()

    # --- Meta-prompting: generate guidelines ---

    # 1. Language-specific guidelines
    logger.info("Generating language-specific guidelines...")
    lang_guidelines = generate_language_guidelines(analysis_model, language, tracker)

    # 2. Containerization guidelines
    logger.info("Generating containerization guidelines...")
    container_guidelines = generate_container_guidelines(analysis_model, tracker)

    # 3. Universal build guidelines
    logger.info("Generating universal build guidelines...")
    universal_guidelines = generate_universal_build_guidelines(analysis_model, tracker)

    # --- Information retrieval ---

    # 4. CI/CD script summarization via LLM
    logger.info("Summarizing CI/CD scripts with LLM...")
    ci_summary = summarize_ci_scripts_with_llm(analysis_model, repo_path, tracker=tracker)

    # 5. Comprehensive repo analysis
    repo_analysis = ""
    if not skip_analysis:
        logger.info("Running comprehensive repository analysis...")
        repo_analysis = analyze_repository_for_docker(
            analysis_model, repo_path, language, gathered_context, tracker
        )

    # 6. Test command discovery (GPT-5 powered deep analysis)
    logger.info("Discovering test command with LLM deep analysis...")
    test_command = discover_test_command(
        repo_path, language, llm=analysis_model, tracker=tracker
    )

    # 7. Test environment analysis (GPT-5 powered)
    logger.info("Analyzing test environment requirements...")
    test_env_analysis = analyze_test_environment(
        analysis_model, repo_path, language,
        test_command=test_command,
        gathered_context=gathered_context,
        tracker=tracker
    )

    # --- Assemble context string ---

    test_cmd_section = ""
    if test_command:
        setup_cmds = test_command.get('setup_commands', [])
        env_vars = test_command.get('env_vars', {})
        skip_pats = test_command.get('skip_patterns', [])
        test_cmd_section = f"""
AUTO-DISCOVERED TEST COMMAND (GPT-5 analyzed):
  Command: {test_command['command']}
  Framework: {test_command.get('test_framework', 'unknown')}
  Source:  {test_command.get('source', 'unknown')}
  Confidence: {test_command.get('confidence', 'unknown')}
  Setup Commands: {setup_cmds if setup_cmds else 'None'}
  Required ENV vars: {env_vars if env_vars else 'None'}
  Skip Patterns (for Docker): {skip_pats if skip_pats else 'None'}
  Needs Build First: {test_command.get('needs_build_first', False)}
  Build Command: {test_command.get('build_command', 'N/A')}
  Notes: {test_command.get('notes', '')}
"""
        logger.info(
            f"Auto-discovered test command: {test_command['command']} "
            f"(from {test_command.get('source', 'unknown')})"
        )

    sep = '=' * 60

    agent_capabilities = f"""
{sep}
AGENT CAPABILITIES & CONSTRAINTS
{sep}
The agent executing these instructions has access to ONLY these tools:

FILE OPERATIONS:
  - WriteToFile       : Create/overwrite a file (Dockerfile, run_tests.sh, etc.)
  - ReadLocalFile     : Read a file from the repo
  - ListDirectory     : List files in a directory
  - FindFiles         : Find files by name/pattern
  - GrepFiles         : Search file contents by regex

BUILD & VERIFICATION:
  - VerifyBuild       : Builds the Docker image (from Dockerfile) AND runs run_tests.sh
                        inside the container if it exists.
                        Returns status, full build log, and test_output.
                        This is the ONLY way to test if a Dockerfile works.
                        ⚠️ The Dockerfile must NOT contain any test execution (no RUN npm test,
                        no RUN pytest, etc.). Tests belong exclusively in run_tests.sh.

ERROR DIAGNOSIS (MUST USE AFTER FAILURES):
  - SearchDockerError : AI-powered Docker build error analysis.
                        Call this IMMEDIATELY after every VerifyBuild failure.
                        Provide: error_keywords + full_error_log + dockerfile_content
  - DiagnoseTestFailure: AI diagnosis for test failures inside Docker.
                        Call this when VerifyBuild fails at stage=TEST_SUITE.
                        Provide: test_output + dockerfile_content + run_tests_content
  - RunInContainer    : Run a diagnostic command inside the already-built container.
                        Use to inspect the environment, check paths, verify installs.

SEARCH & DOCUMENTATION:
  - DockerImageSearch : Search Docker Hub for valid base images and tags.
  - SearchWeb         : Search the web for documentation.
  - FetchWebPage      : Fetch a specific URL for documentation.

⚠️ CRITICAL CONSTRAINT — NO DIRECT SHELL ACCESS:
The agent CANNOT run commands on the host machine. It cannot directly execute
apt-get, npm install, node, python, or any shell command.
ALL commands must go inside:
  1. Dockerfile RUN instructions → verified via VerifyBuild
  2. run_tests.sh script → executed inside Docker via VerifyBuild
  3. RunInContainer → for diagnostics only (does not modify the image)

Design all recommendations assuming the agent works through Dockerfile iteration only.
"""

    context_str = f"""
{sep}
DETECTED LANGUAGE: {language}
{sep}
{agent_capabilities}
{sep}
REPOSITORY ANALYSIS
{sep}
{repo_analysis}

{sep}
CI/CD ANALYSIS (LLM-summarized)
{sep}
{ci_summary}

{sep}
LANGUAGE-SPECIFIC GUIDELINES ({language})
{sep}
{lang_guidelines}

{sep}
CONTAINERIZATION GUIDELINES
{sep}
{container_guidelines}

{sep}
UNIVERSAL BUILD GUIDELINES
{sep}
{universal_guidelines}

{sep}
TEST ENVIRONMENT ANALYSIS (GPT-5 deep analysis)
{sep}
{test_env_analysis}
{test_cmd_section}"""

    # Log preparation token summary
    prep_tokens = tracker.summary()
    logger.info(
        f"[Preparation tokens] Total: {prep_tokens['total']} "
        f"({prep_tokens['input']} in / {prep_tokens['output']} out) "
        f"across {len(prep_tokens['calls'])} calls"
    )

    return {
        "language": language,
        "context_str": context_str,
        "test_command": test_command,
        "gathered_context": gathered_context,
        "repo_analysis": repo_analysis,
        "ci_summary": ci_summary,
        "preparation_token_usage": prep_tokens,
    }
