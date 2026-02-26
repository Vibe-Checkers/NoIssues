"""
Repository classification using LLM analysis.
Runs BEFORE the agent to provide structured context.
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import Dict, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage


logger = logging.getLogger(__name__)

# Canonical repo types — agent and verifier both use these
REPO_TYPES = [
    "cli_tool",            # Produces executable binary/script with CLI
    "web_service",         # HTTP server, API, web application
    "library",             # Importable package (Python, JS, Ruby, Go)
    "native_library",      # C/C++/Rust shared/static library (.so, .a, .dylib)
    "framework",           # Application framework (Spring, Rails, Django)
    "data_pipeline",       # ETL, batch jobs, ML training scripts
    "desktop_app",         # GUI app (Electron, Qt, GTK)
    "documentation_only",  # No buildable code
    "monorepo",            # Multiple packages/services in one repo
]

# Verification strategies — determines how VerifyBuild tests the container
VERIFICATION_STRATEGIES = [
    "binary_run",    # Execute binary with --version/--help
    "import_test",   # Import library in language runtime
    "link_test",     # Verify .so/.a with ldd/nm/ctypes
    "server_probe",  # Start process, check it stays alive for N seconds
    "build_only",    # Build success IS the test (monorepos, complex projects)
    "multi_target",  # Multiple artifacts to verify
]


class RepoClassifier:
    """
    Classify repository type and determine verification strategy.

    Uses GPT-5 (or configured large model) to analyze repo signals.
    Designed to run ONCE before the agent starts, adding ~500-1000 tokens cost.
    """

    def __init__(self):
        # Use the large/expensive model for classification accuracy.
        # This is a one-shot call so cost is negligible vs agent iterations.
        _deployment = os.getenv(
            "AZURE_OPENAI_DEPLOYMENT_LARGE",
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        )
        self.llm = AzureChatOpenAI(
            azure_deployment=_deployment,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )

    def gather_signals(self, repo_path: str) -> Dict:
        """
        Collect lightweight repo signals for classification.
        Reads only small portions of key files — no full tree scan.
        """
        repo = Path(repo_path)
        signals: Dict = {}

        # 1. README (first 3000 chars — enough for project description)
        for readme_name in ["README.md", "README.rst", "README.txt", "README"]:
            readme_path = repo / readme_name
            if readme_path.exists():
                signals["readme_excerpt"] = readme_path.read_text(errors="ignore")[:3000]
                break

        # 2. Build/config file detection + first 500 chars of each
        config_candidates = [
            # Python
            "setup.py", "setup.cfg", "pyproject.toml", "Pipfile",
            # JavaScript/TypeScript
            "package.json", "tsconfig.json",
            # Rust
            "Cargo.toml",
            # C/C++
            "CMakeLists.txt", "Makefile", "configure.ac", "meson.build",
            # Java/Kotlin
            "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
            # Go
            "go.mod",
            # Ruby
            "Gemfile",
            # .NET
            "*.csproj", "*.sln",
        ]
        found_configs = {}
        for cf in config_candidates:
            if "*" in cf:
                matches = list(repo.glob(cf))
                if matches:
                    found_configs[matches[0].name] = matches[0].read_text(errors="ignore")[:500]
            else:
                p = repo / cf
                if p.exists():
                    found_configs[cf] = p.read_text(errors="ignore")[:500]
        signals["config_files"] = found_configs

        # 3. Top-level directory listing (capped at 60 entries)
        try:
            signals["top_level_entries"] = sorted([
                f.name + ("/" if f.is_dir() else "")
                for f in repo.iterdir()
                if not f.name.startswith(".")
            ])[:60]
        except Exception:
            signals["top_level_entries"] = []

        # 4. Entrypoint heuristics
        signals["entrypoint_hints"] = {
            "has_main_py": (repo / "main.py").exists() or (repo / "app.py").exists(),
            "has_cmd_dir": (repo / "cmd").is_dir(),
            "has_src_main_rs": (repo / "src" / "main.rs").exists(),
            "has_src_main_java": (repo / "src" / "main").is_dir(),
            "has_manage_py": (repo / "manage.py").exists(),
            "has_dockerfile": (repo / "Dockerfile").exists(),
        }

        # 5. Count Java modules (critical for monorepo detection)
        signals["java_module_count"] = (
            len(list(repo.rglob("pom.xml"))) + len(list(repo.rglob("build.gradle")))
        )

        # 6. src/ directory structure (1-level peek)
        src_dir = repo / "src"
        if src_dir.is_dir():
            signals["src_listing"] = sorted([
                f.name + ("/" if f.is_dir() else "")
                for f in src_dir.iterdir()
                if not f.name.startswith(".")
            ])[:30]

        return signals

    def classify(self, repo_path: str, repo_name: str) -> Dict:
        """
        Classify the repository and determine verification strategy.

        Returns:
            {
                "repo_type": "library",
                "confidence": "high",
                "primary_language": "Python",
                "build_system": "setuptools",
                "verification_strategy": "import_test",
                "package_name": "requests",       # for import_test
                "binary_name": null,               # for binary_run
                "library_name": null,              # for link_test
                "is_monorepo": false,
                "module_count": 1,
                "rationale": "...",
                "dockerfile_hints": "..."          # guidance for the agent
            }
        """
        signals = self.gather_signals(repo_path)

        prompt = f"""You are classifying a software repository to determine:
1. What TYPE of project this is
2. How to VERIFY a Docker container built from it
3. What HINTS the Dockerfile author needs

REPOSITORY: {repo_name}
TOP-LEVEL FILES: {json.dumps(signals.get('top_level_entries', []))}
CONFIG FILES: {json.dumps(list(signals.get('config_files', {}).keys()))}
ENTRYPOINT HINTS: {json.dumps(signals.get('entrypoint_hints', {}))}
JAVA MODULES: {signals.get('java_module_count', 0)}
SRC LISTING: {json.dumps(signals.get('src_listing', []))}

README (excerpt):
{signals.get('readme_excerpt', 'N/A')[:2000]}

CONFIG FILE CONTENTS:
{json.dumps(signals.get('config_files', {}), indent=2)[:2500]}

CLASSIFICATION RULES:
- "library": produces an importable package, NOT a standalone executable
- "native_library": C/C++/Rust library producing .so/.a/.dylib files
- "cli_tool": produces a binary or script meant to be run directly
- "web_service": starts an HTTP/gRPC server
- "framework": provides scaffolding for OTHER apps (like Spring Boot, Django)
- "monorepo": contains 3+ independent modules/packages with separate build configs
- "documentation_only": no compilable source code, just docs/configs
- "data_pipeline": ML training, ETL, batch processing

VERIFICATION STRATEGY RULES:
- "import_test": for libraries — import the package in the language runtime and print version/symbols
- "binary_run": for CLI tools — execute the binary with --version or --help
- "link_test": for native libraries — verify with ldd, nm, or ctypes
- "server_probe": for web services — start process, sleep 3s, check alive, kill
- "build_only": for monorepos, frameworks, or projects where build success IS the test
- "multi_target": for monorepos with multiple independent artifacts

For multi-module Maven projects: do NOT build the aggregator POM alone.
Either build a specific module that produces a JAR (mvn -pl <module> -am package)
or use 'mvn package -DskipTests' at root to build all modules.
SNAPSHOT dependencies are reactor-internal — they resolve during a full build,
not when building isolated modules.

Return ONLY valid JSON:
{{
    "repo_type": "<one of: {json.dumps(REPO_TYPES)}>",
    "confidence": "high|medium|low",
    "primary_language": "<e.g. Python, Java, C++, Go, Rust, JavaScript>",
    "build_system": "<e.g. cmake, setuptools, cargo, maven, gradle, npm>",
    "verification_strategy": "<one of: {json.dumps(VERIFICATION_STRATEGIES)}>",
    "package_name": "<importable name for import_test, or null>",
    "binary_name": "<executable name for binary_run, or null>",
    "library_name": "<.so/.a name for link_test, or null>",
    "is_monorepo": true/false,
    "module_count": <int>,
    "rationale": "<1-2 sentences explaining classification>",
    "dockerfile_hints": "<2-3 sentences of critical guidance for Dockerfile creation. Mention: which build commands to use, what the output artifact is, any known gotchas for this project type.>"
}}"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip()
            # Strip markdown fences
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            # Validate required fields
            if result.get("repo_type") not in REPO_TYPES:
                result["repo_type"] = "cli_tool"
                result["confidence"] = "low"
            if result.get("verification_strategy") not in VERIFICATION_STRATEGIES:
                result["verification_strategy"] = "build_only"

            return result

        except Exception as e:
            logger.warning(f"LLM classification failed for {repo_name}: {e}")
            # Fallback: conservative defaults
            return {
                "repo_type": "cli_tool",
                "confidence": "low",
                "primary_language": "Unknown",
                "build_system": "unknown",
                "verification_strategy": "build_only",
                "package_name": None,
                "binary_name": None,
                "library_name": None,
                "is_monorepo": False,
                "module_count": 1,
                "rationale": f"LLM classification failed: {e}",
                "dockerfile_hints": "Examine the repo structure manually.",
            }
