
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

def discover_test_command(repo_path: str, language: str) -> Optional[Dict[str, str]]:
    """
    Auto-discover the test command by probing project files.
    
    Checks (in priority order):
      1. CI workflow files (.github/workflows/*.yml)
      2. Makefile test/check targets
      3. package.json scripts.test
      4. pyproject.toml / tox.ini / setup.cfg
      5. Language-based fallback heuristics
    
    Returns:
        Dict with 'command', 'source', 'confidence' or None
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
                                # Extract the full line containing the match
                                for line in run_cmd.strip().split('\n'):
                                    if test_patterns.search(line):
                                        cmd = line.strip()
                                        logger.info(f"Discovered test command from CI: {cmd}")
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
            # Look for test: or check: targets
            for target in ['test', 'check', 'tests']:
                pattern = re.compile(rf'^{target}\s*:', re.MULTILINE)
                if pattern.search(content):
                    cmd = f"make {target}"
                    logger.info(f"Discovered test command from Makefile: {cmd}")
                    return {
                        "command": cmd,
                        "source": "Makefile",
                        "confidence": "high"
                    }
        except Exception as e:
            logger.debug(f"Error reading Makefile: {e}")
    
    # 3. package.json (Node.js)
    pkg_json = repo / "package.json"
    if pkg_json.exists():
        try:
            with open(pkg_json, 'r') as f:
                pkg = json.load(f)
            test_script = pkg.get('scripts', {}).get('test', '')
            if test_script and 'no test specified' not in test_script.lower():
                logger.info(f"Discovered test command from package.json: npm test")
                return {
                    "command": "npm test",
                    "source": f"package.json (scripts.test = {test_script})",
                    "confidence": "high"
                }
        except Exception as e:
            logger.debug(f"Error reading package.json: {e}")
    
    # 4. pyproject.toml
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        try:
            with open(pyproject, 'r') as f:
                content = f.read()
            if '[tool.pytest' in content or 'pytest' in content.lower():
                logger.info("Discovered test command from pyproject.toml: pytest")
                return {
                    "command": "python -m pytest",
                    "source": "pyproject.toml ([tool.pytest] section found)",
                    "confidence": "high"
                }
        except Exception as e:
            logger.debug(f"Error reading pyproject.toml: {e}")
    
    # 5. tox.ini
    tox_ini = repo / "tox.ini"
    if tox_ini.exists():
        try:
            with open(tox_ini, 'r') as f:
                content = f.read()
            # Look for commands = pytest or similar
            for line in content.split('\n'):
                if 'commands' in line.lower() and '=' in line:
                    cmd_part = line.split('=', 1)[1].strip()
                    if cmd_part:
                        logger.info(f"Discovered test command from tox.ini: {cmd_part}")
                        return {
                            "command": cmd_part,
                            "source": "tox.ini",
                            "confidence": "medium"
                        }
        except Exception as e:
            logger.debug(f"Error reading tox.ini: {e}")
    
    # 6. setup.cfg
    setup_cfg = repo / "setup.cfg"
    if setup_cfg.exists():
        try:
            with open(setup_cfg, 'r') as f:
                content = f.read()
            if '[tool:pytest]' in content:
                logger.info("Discovered test command from setup.cfg: pytest")
                return {
                    "command": "python -m pytest",
                    "source": "setup.cfg ([tool:pytest] section)",
                    "confidence": "medium"
                }
        except Exception as e:
            logger.debug(f"Error reading setup.cfg: {e}")
    
    # 7. Language-based fallback
    fallbacks = {
        'Python': ('python -m pytest', 'low'),
        'Node.js': ('npm test', 'low'),
        'Java': ('mvn test', 'low'),
        'Kotlin': ('./gradlew test', 'low'),
        'Go': ('go test ./...', 'low'),
        'Rust': ('cargo test', 'low'),
        'Ruby': ('bundle exec rspec', 'low'),
        'PHP': ('./vendor/bin/phpunit', 'low'),
        'Elixir': ('mix test', 'low'),
        'C/C++': ('make check', 'low'),
        'C++': ('ctest', 'low'),
    }
    
    if language in fallbacks:
        cmd, confidence = fallbacks[language]
        logger.info(f"Using fallback test command for {language}: {cmd}")
        return {
            "command": cmd,
            "source": f"language fallback ({language})",
            "confidence": confidence
        }
    
    return None


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

    # 6. Test command discovery (heuristic + structured)
    test_command = discover_test_command(repo_path, language)

    # --- Assemble context string ---

    test_cmd_section = ""
    if test_command:
        test_cmd_section = f"""
AUTO-DISCOVERED TEST COMMAND:
  Command: {test_command['command']}
  Source:  {test_command['source']}
  Confidence: {test_command['confidence']}
"""
        logger.info(
            f"Auto-discovered test command: {test_command['command']} "
            f"(from {test_command['source']})"
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
