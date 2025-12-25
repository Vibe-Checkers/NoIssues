
import os
import logging
import yaml
from pathlib import Path
from typing import List, Dict, Optional, Any
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

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

    context_str = f"""
DETECTED LANGUAGE: {language}

LANGUAGE GUIDELINES (Latest Best Practices):
{guidelines}

CI/CD WORKFLOWS (Hints from .github):
{workflows}
"""
    return {
        "language": language,
        "context_str": context_str
    }
