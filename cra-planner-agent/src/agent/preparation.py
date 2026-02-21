
import os
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from .tools import _docker_hub_list_tags, _get_expanded_platform_info

logger = logging.getLogger(__name__)

# Files to read per language for meta-analysis
_LANGUAGE_KEY_FILES = {
    'Python':  ['requirements.txt', 'pyproject.toml', 'setup.py', 'Pipfile'],
    'Node.js': ['package.json'],
    'Java':    ['pom.xml', 'build.gradle'],
    'Kotlin':  ['build.gradle.kts'],
    'Go':      ['go.mod'],
    'Rust':    ['Cargo.toml'],
    'Ruby':    ['Gemfile'],
    'PHP':     ['composer.json'],
    'Elixir':  ['mix.exs'],
    'Dart':    ['pubspec.yaml'],
    'C++':     ['CMakeLists.txt'],
}

# Docker Hub image names to query for each language
_LANGUAGE_BASE_IMAGES = {
    'Python':  ['python'],
    'Node.js': ['node'],
    'Java':    ['eclipse-temurin', 'openjdk'],
    'Kotlin':  ['eclipse-temurin'],
    'Go':      ['golang'],
    'Rust':    ['rust'],
    'Ruby':    ['ruby'],
    'PHP':     ['php'],
    'Elixir':  ['elixir'],
    'Dart':    ['dart'],
    'C++':     ['gcc'],
    'Next.js': ['node'],
    'Angular': ['node'],
}

def _fetch_verified_docker_tags(language: str) -> str:
    """
    Query Docker Hub for real available tags for the language's base image(s).
    Returns a formatted string of tags with platform-compatibility markers,
    or an empty string if the lookup fails.
    """
    image_names = _LANGUAGE_BASE_IMAGES.get(language, [])
    if not image_names:
        return ""

    platform_info = _get_expanded_platform_info()
    sections = []
    for image_name in image_names:
        try:
            result = _docker_hub_list_tags(image_name, platform_info)
            if result and "not found" not in result.lower():
                sections.append(f"Docker Hub tags for '{image_name}':\n{result}")
        except Exception:
            pass

    return "\n\n".join(sections)

def _create_analysis_llm() -> AzureChatOpenAI:
    """Create the GPT-5 analysis LLM, falling back to the main model if unconfigured."""
    load_dotenv()
    return AzureChatOpenAI(
        azure_deployment=os.getenv(
            "ANALYSIS_MODEL_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT")
        ),
        azure_endpoint=os.getenv(
            "ANALYSIS_MODEL_ENDPOINT", os.getenv("AZURE_OPENAI_ENDPOINT")
        ),
        api_key=os.getenv(
            "ANALYSIS_MODEL_API_KEY", os.getenv("AZURE_OPENAI_API_KEY")
        ),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        timeout=60,
    )

def generate_meta_strategy(repo_path: str, language: str) -> str:
    """
    Use GPT-5 to pre-analyze the repo and produce a concise Dockerfile strategy.

    Reads the directory listing and primary dependency file, then asks GPT-5 for
    a targeted strategy memo. This is injected into the agent's initial context so
    it can skip most of its exploration phase and go straight to writing.

    Returns an empty string on any failure (non-blocking).
    """
    intel_parts = []

    # 1. Root directory listing
    try:
        items = []
        for entry in sorted(os.scandir(repo_path), key=lambda e: (e.is_file(), e.name)):
            kind = "DIR" if entry.is_dir() else "FILE"
            items.append(f"  {kind}: {entry.name}")
        intel_parts.append("ROOT STRUCTURE:\n" + "\n".join(items[:50]))
    except Exception:
        pass

    # 2. Primary dependency file for the detected language (first one found wins)
    for fname in _LANGUAGE_KEY_FILES.get(language, []):
        fpath = os.path.join(repo_path, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(3000)
                intel_parts.append(f"{fname}:\n{content}")
            except Exception:
                pass
            break  # one file is enough

    if not intel_parts:
        return ""

    intel_str = "\n\n".join(intel_parts)

    # Fetch real Docker Hub tags so GPT-5 can only pick from verified, live images
    verified_tags = _fetch_verified_docker_tags(language)
    tags_section = (
        f"\nVERIFIED DOCKER HUB TAGS (live, [OK] = compatible with host platform):\n"
        f"{verified_tags}\n"
        f"IMPORTANT: You MUST pick your base image FROM THIS LIST ONLY. "
        f"Do not suggest any image or tag not shown above.\n"
        if verified_tags else ""
    )

    try:
        analysis_llm = _create_analysis_llm()
        user_prompt = (
            f"Analyze this {language} repository and produce a Dockerfile strategy memo.\n\n"
            f"REPOSITORY INTEL:\n{intel_str}\n"
            f"{tags_section}\n"
            "Produce a CONCISE strategy (max 200 words) covering:\n"
            "1. Recommended base image (exact tag — must be from the verified list above)\n"
            "2. System + language dependencies to install\n"
            "3. Build steps in order\n"
            "4. Entry point / start command\n"
            "5. One key pitfall to avoid for this specific project\n\n"
            "Base your answers strictly on the files and verified tags above."
        )
        response = analysis_llm.invoke([
            SystemMessage(content="You are a senior DevOps engineer. Be concise and specific."),
            HumanMessage(content=user_prompt),
        ])
        return response.content.strip()
    except Exception as e:
        logger.warning("GPT-5 meta-strategy generation failed: %s", e)
        return ""

def detect_project_language(repo_path: str) -> str:
    """
    Detect the primary programming language of a repository.
    PRIORITIZES build system files over source files.
    """
    try:
        if not os.path.exists(repo_path):
            return "Unknown"

        # Check for build files first (strongest signal)
        build_indicators = {
            'package.json': 'Node.js',
            'requirements.txt': 'Python',
            'setup.py': 'Python',
            'pyproject.toml': 'Python',
            'Pipfile': 'Python',
            'pom.xml': 'Java',
            'build.gradle': 'Java',
            'build.gradle.kts': 'Kotlin',
            'go.mod': 'Go',
            'Cargo.toml': 'Rust',
            'Gemfile': 'Ruby',
            'composer.json': 'PHP',
            'mix.exs': 'Elixir',
            'cabal.project': 'Haskell',
            'pubspec.yaml': 'Dart',
            'CMakeLists.txt': 'C++',
            'Makefile': 'C/C++', # Ambiguous but often C/C++
            'angular.json': 'Angular', # Specific framework
            'next.config.js': 'Next.js', # Specific framework
        }

        # Check root directory for build files
        for filename, language in build_indicators.items():
            if os.path.exists(os.path.join(repo_path, filename)):
                logger.info(f"Detected language {language} based on {filename}")
                return language

        # Fallback to file extensions in top-level directories
        extension_counts = {}
        for root, dirs, files in os.walk(repo_path):
            # Don't go deep
            if root.count(os.sep) - repo_path.count(os.sep) > 2:
                continue
                
            # Skip hidden dirs
            if any(part.startswith('.') for part in root.split(os.sep)):
                continue

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in ['.py']: extension_counts['Python'] = extension_counts.get('Python', 0) + 1
                elif ext in ['.js', '.jsx', '.ts', '.tsx']: extension_counts['Node.js'] = extension_counts.get('Node.js', 0) + 1
                elif ext in ['.java']: extension_counts['Java'] = extension_counts.get('Java', 0) + 1
                elif ext in ['.go']: extension_counts['Go'] = extension_counts.get('Go', 0) + 1
                elif ext in ['.rb']: extension_counts['Ruby'] = extension_counts.get('Ruby', 0) + 1
                elif ext in ['.php']: extension_counts['PHP'] = extension_counts.get('PHP', 0) + 1
                elif ext in ['.rs']: extension_counts['Rust'] = extension_counts.get('Rust', 0) + 1

        if extension_counts:
            # Return most frequent
            most_common = max(extension_counts.items(), key=lambda x: x[1])[0]
            logger.info(f"Detected language {most_common} based on file extensions {extension_counts}")
            return most_common

        return "Unknown"
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return "Unknown"

def generate_language_guidelines(llm: BaseChatModel, language: str) -> str:
    """
    Generate recent best practices using bare LLM (no tools).

    Args:
        llm: The base chat model (NOT the agent executor)
        language: Detected programming language

    Returns:
        Guidelines as formatted string
    """
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

def summarize_github_workflows(repo_path: str) -> str:
    """Scan .github/workflows for CI/CD hints."""
    workflows_dir = Path(repo_path) / ".github" / "workflows"
    if not workflows_dir.exists(): return "No .github/workflows found."
    
    summary = ["Found CI/CD Workflows:"]
    found = False
    
    try:
        files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
        for wf in files:
            found = True
            try:
                with open(wf, 'r') as f:
                    content = yaml.safe_load(f)
                name = content.get('name', wf.name)
                summary.append(f"\n- Workflow: {name}")
                
                jobs = content.get('jobs', {})
                for job_name, job_data in jobs.items():
                    steps = job_data.get('steps', [])
                    summary.append(f"  Job: {job_name}")
                    for step in steps:
                        run = step.get('run')
                        if run:
                            cleaned_run = run.strip().replace('\n', '; ')
                            summary.append(f"    Run: {cleaned_run}")
            except: pass
            
        return "\n".join(summary) if found else "No workflows found."
    except Exception as e:
        return f"Error reading workflows: {e}"

def build_initial_context(llm: BaseChatModel, repo_path: str) -> dict:
    """
    Build the initial context for the Learner Agent.
    Aggregates language detection, guidelines, and CI/CD summary.

    Args:
        llm: The base chat model for generating guidelines
        repo_path: Path to repository

    Returns:
        Dictionary with language and context string
    """
    language = detect_project_language(repo_path)
    logger.info(f"Building context for language: {language}")

    # Use bare LLM for guidelines (not agent executor)
    guidelines = generate_language_guidelines(llm, language)
    workflows = summarize_github_workflows(repo_path)

    # GPT-5 pre-analysis: gives the agent a targeted strategy before it starts,
    # reducing wasted exploration iterations.
    logger.info("Running GPT-5 meta-strategy pre-analysis...")
    meta_strategy = generate_meta_strategy(repo_path, language)
    meta_section = (
        f"\nDOCKERFILE STRATEGY (Pre-analyzed by GPT-5 — follow this closely):\n{meta_strategy}\n"
        if meta_strategy else ""
    )

    context_str = f"""
DETECTED LANGUAGE: {language}
{meta_section}
LANGUAGE GUIDELINES (Latest Best Practices):
{guidelines}

CI/CD WORKFLOWS (Hints from .github):
{workflows}
"""
    return {
        "language": language,
        "context_str": context_str,
        "meta_strategy": meta_strategy,  # empty string if generation failed
    }
