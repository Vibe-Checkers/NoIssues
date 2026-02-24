
import os
import re
import json
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language Detection
# ---------------------------------------------------------------------------

# Dirs to skip at every pass to avoid false positives
_SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "env", "dist", "build", "target", "out", ".gradle",
    "docs", "doc", "examples", "sample", "test", "tests",
    "vendor", "third_party", "thirdparty", "external",
}

# Priority-ordered build indicators. Higher entries win over lower ones when
# found at the same depth. This list is intentionally ordered from most
# specific to least specific.
_BUILD_INDICATORS: List[tuple] = [
    ("pom.xml",           "Java"),
    ("build.gradle",      "Java"),
    ("build.gradle.kts",  "Kotlin"),
    ("Cargo.toml",        "Rust"),
    ("go.mod",            "Go"),
    ("pyproject.toml",    "Python"),
    ("setup.py",          "Python"),
    ("requirements.txt",  "Python"),
    ("Pipfile",           "Python"),
    ("package.json",      "Node.js"),
    ("Gemfile",           "Ruby"),
    ("composer.json",     "PHP"),
    ("CMakeLists.txt",    "C++"),
    ("configure.ac",      "C"),   # autoconf (git, distcc, openvpn)
    ("meson.build",       "C"),   # meson (mpv)
    ("Makefile",          "C"),
]

_EXTENSION_MAP: Dict[str, str] = {
    ".py":  "Python",
    ".java": "Java",
    ".kt":  "Kotlin",
    ".rs":  "Rust",
    ".go":  "Go",
    ".rb":  "Ruby",
    ".php": "PHP",
    ".cpp": "C++", ".cxx": "C++", ".cc": "C++", ".hpp": "C++",
    ".c":   "C",   ".h":   "C/C++",
    ".js":  "Node.js", ".jsx": "Node.js",
    ".ts":  "Node.js", ".tsx": "Node.js",
}


def _skip_path(path: Path) -> bool:
    """Return True if any part of this path is in the skip-dirs set."""
    return any(
        part.lower() in _SKIP_DIRS or part.startswith(".")
        for part in path.parts
    )


def detect_project_language(repo_path: str) -> str:
    """
    Detect the primary programming language of a repository.

    Four-pass strategy (returns on first confident match):
      1. Root-directory build files  — fastest, highest confidence
      2. Immediate subdirectories    — catches multi-module roots
      3. Recursive glob depth 2–3    — catches nested build systems
      4. File-extension counting     — fallback, least reliable

    For polyglot repos where multiple indicators fire at the same depth,
    priority is determined by position in _BUILD_INDICATORS (top = wins).
    """
    try:
        if not os.path.exists(repo_path):
            return "Unknown"

        repo = Path(repo_path)

        # Pass 1: root directory only
        for filename, language in _BUILD_INDICATORS:
            if (repo / filename).exists():
                logger.info(f"[LangDetect] Pass 1 — {language} via {filename}")
                return language

        # Pass 2: immediate subdirectories (depth 1)
        for subdir in sorted(repo.iterdir()):  # sorted for determinism
            if not subdir.is_dir():
                continue
            if subdir.name.startswith(".") or subdir.name.lower() in _SKIP_DIRS:
                continue
            for filename, language in _BUILD_INDICATORS:
                if (subdir / filename).exists():
                    logger.info(f"[LangDetect] Pass 2 — {language} via {subdir.name}/{filename}")
                    return language

        # Pass 3: shallow recursive glob (depth 2–3), priority order preserved
        for depth in range(2, 4):
            pattern = "/".join(["*"] * depth)
            for filename, language in _BUILD_INDICATORS:
                matches = [
                    m for m in repo.glob(f"{pattern}/{filename}")
                    if not _skip_path(m.relative_to(repo))
                ]
                if matches:
                    rel = matches[0].relative_to(repo)
                    logger.info(f"[LangDetect] Pass 3 (depth {depth}) — {language} via {rel}")
                    return language

        # Pass 4: extension counting across repo (depth ≤ 4)
        ext_counts: Dict[str, int] = {}
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d.lower() not in _SKIP_DIRS
            ]
            depth = root.replace(repo_path, "").count(os.sep)
            if depth > 4:
                dirs.clear()
                continue
            for fname in files:
                lang = _EXTENSION_MAP.get(os.path.splitext(fname)[1].lower())
                if lang:
                    ext_counts[lang] = ext_counts.get(lang, 0) + 1

        if ext_counts:
            detected = max(ext_counts.items(), key=lambda x: x[1])[0]
            logger.info(f"[LangDetect] Pass 4 (extensions) — {detected}: {ext_counts}")
            return detected

        return "Unknown"

    except Exception as e:
        logger.error(f"[LangDetect] Error: {e}")
        return "Unknown"


# ---------------------------------------------------------------------------
# Language Guidelines
# ---------------------------------------------------------------------------

def generate_language_guidelines(llm: BaseChatModel, language: str) -> str:
    """Generate up-to-date Dockerfile best practices for the detected language."""
    if not language or language == "Unknown":
        return ""

    prompt = f"""You are an expert DevOps engineer.
Generate 10 concise, UP-TO-DATE guidelines for Dockerizing a modern {language} project in 2024/2025.
Focus on: Base images, Package managers, Security, and Common Pitfalls.
Format as bullet list.
Be specific and actionable."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        logger.error(f"Failed to generate guidelines: {e}")
        return ""


# ---------------------------------------------------------------------------
# CI/CD Workflow Extraction (structured)
# ---------------------------------------------------------------------------

def summarize_github_workflows(repo_path: str) -> str:
    """
    Scan .github/workflows for CI/CD hints.
    Returns human-readable summary plus a KEY FACTS block with extracted
    versions and commands the agent can use directly.
    """
    workflows_dir = Path(repo_path) / ".github" / "workflows"
    if not workflows_dir.exists():
        return "No .github/workflows found."

    extracted: Dict[str, list] = {
        "build_commands":   [],
        "install_commands": [],
        "node_versions":    [],
        "python_versions":  [],
        "java_versions":    [],
        "timeout_minutes":  [],
    }
    summary_lines = ["Found CI/CD Workflows:"]

    try:
        for wf in list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml")):
            try:
                with open(wf, "r") as f:
                    content = yaml.safe_load(f)
                if not content:
                    continue
                name = content.get("name", wf.name)
                summary_lines.append(f"\n  Workflow: {name}")

                for job_name, job_data in (content.get("jobs") or {}).items():
                    if not isinstance(job_data, dict):
                        continue

                    timeout = job_data.get("timeout-minutes")
                    if timeout:
                        try:
                            extracted["timeout_minutes"].append(int(timeout))
                        except (ValueError, TypeError):
                            pass

                    # Matrix version extraction
                    matrix = {}
                    strategy = job_data.get("strategy")
                    if isinstance(strategy, dict):
                        matrix = strategy.get("matrix") or {}
                    if isinstance(matrix, dict):
                        for key in ("node-version", "node_version"):
                            v = matrix.get(key)
                            if v:
                                extracted["node_versions"].extend(v if isinstance(v, list) else [v])
                        for key in ("python-version", "python_version"):
                            v = matrix.get(key)
                            if v:
                                extracted["python_versions"].extend(v if isinstance(v, list) else [v])
                        for key in ("java-version", "java_version"):
                            v = matrix.get(key)
                            if v:
                                extracted["java_versions"].extend(v if isinstance(v, list) else [v])

                    for step in (job_data.get("steps") or []):
                        if not isinstance(step, dict):
                            continue
                        uses = step.get("uses", "")
                        with_data = step.get("with") or {}
                        if "setup-node" in uses and "node-version" in with_data:
                            extracted["node_versions"].append(str(with_data["node-version"]))
                        if "setup-python" in uses and "python-version" in with_data:
                            extracted["python_versions"].append(str(with_data["python-version"]))
                        if "setup-java" in uses and "java-version" in with_data:
                            extracted["java_versions"].append(str(with_data["java-version"]))

                        run = step.get("run", "")
                        if not run:
                            continue
                        run_clean = run.strip().replace("\n", "; ")[:200]
                        summary_lines.append(f"    run: {run_clean}")
                        run_lower = run.lower()

                        if any(k in run_lower for k in ("npm install", "npm ci", "yarn install", "yarn add")):
                            extracted["install_commands"].append(run_clean)
                        if any(k in run_lower for k in ("npm run build", "yarn build", "mvn package",
                                                         "gradle build", "pip install", "python setup.py",
                                                         "cargo build")):
                            extracted["build_commands"].append(run_clean)
            except Exception:
                pass

    except Exception as e:
        return f"Error reading workflows: {e}"

    # Build KEY FACTS block
    facts: List[str] = []
    if extracted["node_versions"]:
        facts.append(f"  Node.js versions in CI: {sorted(set(str(v) for v in extracted['node_versions']))}")
    if extracted["python_versions"]:
        facts.append(f"  Python versions in CI: {sorted(set(str(v) for v in extracted['python_versions']))}")
    if extracted["java_versions"]:
        facts.append(f"  Java versions in CI: {sorted(set(str(v) for v in extracted['java_versions']))}")
    if extracted["timeout_minutes"]:
        facts.append(f"  Max CI job timeout: {max(extracted['timeout_minutes'])} min")
    if extracted["install_commands"]:
        facts.append(f"  Install command from CI: {extracted['install_commands'][0]}")
    if extracted["build_commands"]:
        facts.append(f"  Build command from CI: {extracted['build_commands'][0]}")

    result = "\n".join(summary_lines)
    if facts:
        result += "\n\n  KEY FACTS EXTRACTED FROM CI:\n" + "\n".join(facts)
    return result


# ---------------------------------------------------------------------------
# Build Manifest Analysis
# ---------------------------------------------------------------------------

def analyze_build_manifests(repo_path: str) -> Dict[str, Any]:
    """
    Parse manifest files (package.json, pyproject.toml, build.gradle, pom.xml,
    CMakeLists.txt, gradle-wrapper.properties) to extract version pins, build
    tool details, and generate specific warnings for the agent.

    Returns a dict with extracted metadata and a 'warnings' list of strings.
    """
    repo = Path(repo_path)
    result: Dict[str, Any] = {
        "node_version":    None,
        "python_version":  None,
        "java_version":    None,
        "package_manager": None,
        "has_postinstall": False,
        "has_preinstall":  False,
        "has_husky":       False,
        "has_c_extensions": False,
        "requires_fortran": False,
        "gradle_version":  None,
        "cmake_minimum":   None,
        "warnings":        [],
    }

    # --- Node.js ---
    pkg_json = repo / "package.json"
    if pkg_json.exists():
        try:
            pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
            result["package_manager"] = "yarn" if (repo / "yarn.lock").exists() else (
                "pnpm" if (repo / "pnpm-lock.yaml").exists() else "npm"
            )
            engines = pkg.get("engines") or {}
            if "node" in engines:
                result["node_version"] = engines["node"]

            scripts = pkg.get("scripts") or {}
            if "postinstall" in scripts:
                result["has_postinstall"] = True
                result["warnings"].append(
                    f"CAUTION: package.json has a 'postinstall' script: "
                    f"'{scripts['postinstall'][:120]}'. "
                    f"This runs automatically during npm/yarn install. "
                    f"COPY the FULL repository before running install, not just package.json."
                )
            if "preinstall" in scripts:
                result["has_preinstall"] = True
                sv = scripts["preinstall"]
                result["warnings"].append(
                    f"CAUTION: package.json has a 'preinstall' script: '{sv[:120]}'. "
                    f"Copy the scripts/ directory into the image BEFORE running install."
                )
            dev_deps = pkg.get("devDependencies") or {}
            if "husky" in dev_deps or "husky" in (pkg.get("dependencies") or {}):
                result["has_husky"] = True
                result["warnings"].append(
                    "CAUTION: Husky is a devDependency that requires a .git directory for git hooks. "
                    "Docker builds typically lack .git, so hook execution will fail during install. "
                    "If the install step fails due to Husky, use SearchDockerError to find the "
                    "correct flag to skip lifecycle scripts for your package manager."
                )
        except Exception as e:
            logger.warning(f"[Manifests] package.json parse error: {e}")

    # --- Python version pins ---
    for py_file in (".python-version", "runtime.txt", ".tool-versions"):
        candidate = repo / py_file
        if candidate.exists():
            try:
                text = candidate.read_text().strip()
                if py_file == ".tool-versions":
                    for line in text.splitlines():
                        if line.startswith("python"):
                            result["python_version"] = line.split()[-1]
                            break
                else:
                    result["python_version"] = text.splitlines()[0].strip()
                break
            except Exception:
                pass

    # --- Python C extensions / Fortran ---
    for ext_file in ("setup.py", "pyproject.toml"):
        candidate = repo / ext_file
        if candidate.exists():
            try:
                text = candidate.read_text()
                tl = text.lower()
                if "ext_modules" in text or "cffi" in tl or "cython" in tl:
                    result["has_c_extensions"] = True
                    result["warnings"].append(
                        "Python project has C extensions. "
                        "Install build-essential, gcc, and python3-dev before pip install."
                    )
                if "gfortran" in tl or "fortran" in tl or "blas" in tl or "lapack" in tl:
                    result["requires_fortran"] = True
                    result["warnings"].append(
                        "Python project requires native numerical libraries (Fortran/BLAS/LAPACK). "
                        "Install the appropriate Fortran compiler and math library packages "
                        "for the base image's OS before building."
                    )
            except Exception:
                pass

    # --- Java / Gradle wrapper ---
    gradle_props = repo / "gradle" / "wrapper" / "gradle-wrapper.properties"
    if gradle_props.exists():
        try:
            text = gradle_props.read_text()
            m = re.search(r"distributionUrl=.*gradle-([\d.]+)-", text)
            if m:
                result["gradle_version"] = m.group(1)
                result["warnings"].append(
                    f"Gradle wrapper version: {m.group(1)}. Use this EXACT version in your "
                    f"base image. Confirm the distribution exists at "
                    f"https://services.gradle.org/distributions/ before building."
                )
        except Exception:
            pass

    # --- Java toolchain version from build.gradle ---
    for build_file in ("build.gradle", "build.gradle.kts"):
        candidate = repo / build_file
        if candidate.exists():
            try:
                text = candidate.read_text()
                m = re.search(r"languageVersion\s*=\s*JavaLanguageVersion\.of\((\d+)\)", text)
                if m:
                    result["java_version"] = m.group(1)
                    result["warnings"].append(
                        f"Project requires Java {m.group(1)} (build.gradle toolchain). "
                        f"Use DockerImageSearch to find a JDK {m.group(1)} base image."
                    )
                break
            except Exception:
                pass

    # --- QA / compliance plugin detection ---
    # Detect any build plugin that runs validation checks outside compilation.
    # These checks (license headers, style, static analysis) can fail in Docker
    # context for reasons unrelated to the Dockerfile (no git history, generated
    # files, missing headers, etc.).
    _COMPLIANCE_PLUGINS = [
        "rat",           # Apache RAT (license header check)
        "checkstyle",    # Java style checker
        "spotbugs",      # Static analysis (successor to FindBugs)
        "findbugs",      # Static analysis
        "pmd",           # Static analysis / code quality
        "jacoco",        # Code coverage (can block on minimum threshold)
        "license-maven-plugin",  # Generic license header plugin
    ]
    for build_file in ("pom.xml", "build.gradle", "build.gradle.kts"):
        candidate = repo / build_file
        if candidate.exists():
            try:
                text = candidate.read_text().lower()
                detected = [p for p in _COMPLIANCE_PLUGINS if re.search(rf'\b{p}\b', text)]
                if detected:
                    result["warnings"].append(
                        f"Build configuration includes QA/compliance plugins "
                        f"({', '.join(detected)}). These checks may fail in Docker context "
                        f"(e.g. no git history, generated files without required headers, "
                        f"coverage thresholds not met). If the build fails due to one of these "
                        f"checks, search for the plugin's skip flag using SearchDockerError."
                    )
                    break
            except Exception:
                pass

    # --- C/C++: CMake minimum version ---
    cmake_file = repo / "CMakeLists.txt"
    if cmake_file.exists():
        try:
            text = cmake_file.read_text()
            m = re.search(r"cmake_minimum_required\s*\(\s*VERSION\s+([\d.]+)", text)
            if m:
                result["cmake_minimum"] = m.group(1)
        except Exception:
            pass

    return result


def format_manifest_warnings(manifest: Dict[str, Any]) -> str:
    """Format manifest analysis into a string for injection into the goal prompt."""
    lines: List[str] = []
    if manifest.get("node_version"):
        lines.append(f"- Node.js version required: {manifest['node_version']}")
    if manifest.get("python_version"):
        lines.append(f"- Python version required: {manifest['python_version']}")
    if manifest.get("java_version"):
        lines.append(f"- Java version required: {manifest['java_version']}")
    if manifest.get("gradle_version"):
        lines.append(f"- Gradle wrapper version: {manifest['gradle_version']}")
    if manifest.get("package_manager"):
        lines.append(f"- Package manager: {manifest['package_manager']}")
    if manifest.get("cmake_minimum"):
        lines.append(f"- CMake minimum version: {manifest['cmake_minimum']}")
    for warning in manifest.get("warnings", []):
        lines.append(f"⚠️  {warning}")
    return "\n".join(lines) if lines else ""




# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------

def build_initial_context(llm: BaseChatModel, repo_path: str) -> dict:
    """
    Build the full preparation context for the Learner Agent.

    Runs:
      1. Language detection (4-pass)
      2. Language guidelines (LLM call)
      3. CI/CD workflow extraction (structured)
      4. Build manifest analysis (version pins, build tool warnings)
    """
    language = detect_project_language(repo_path)
    logger.info(f"[Prep] Building context for language: {language}")

    guidelines   = generate_language_guidelines(llm, language)
    workflows    = summarize_github_workflows(repo_path)
    manifest     = analyze_build_manifests(repo_path)
    manifest_str = format_manifest_warnings(manifest)

    context_str = f"""
DETECTED LANGUAGE: {language}

LANGUAGE GUIDELINES (Latest Best Practices):
{guidelines}

CI/CD WORKFLOWS (Hints from .github):
{workflows}
"""

    if manifest_str:
        context_str += f"""
BUILD MANIFEST ANALYSIS (Version pins and build tool warnings — READ CAREFULLY):
{manifest_str}
"""

    return {
        "language":     language,
        "context_str":  context_str,
        "manifest":     manifest,
    }
