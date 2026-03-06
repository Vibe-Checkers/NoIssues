"""
Taxonomy-to-Context Metaprompt Generator.

Takes the 7-dimension repository taxonomy and generates tailored,
actionable Dockerfile creation context via a single LLM call.
Replaces the previous build_initial_context() approach.
"""

import logging
from typing import Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


METAPROMPT_SYSTEM = """You are an expert DevOps architect specializing in Docker containerization.

Given a repository's 7-dimension taxonomy classification, generate PRECISE, ACTIONABLE context
that a Dockerfile-writing agent will use. Your output must be structured and directly applicable.

For each dimension, derive the specific Dockerfile implications:

1. DOMAIN -> Base image family, runtime requirements, whether an ENTRYPOINT makes sense
2. BUILD_TOOL -> Exact install/build commands, multi-stage build patterns, cache optimization
3. AUTOMATION_LEVEL -> How much the agent can rely on documented commands vs needing to reverse-engineer
4. ENVIRONMENT_SPECIFICITY -> Platform flags, OS-specific packages, version pinning requirements
5. DEPENDENCY_TRANSPARENCY -> Whether to use lockfiles, how to handle implicit deps, pip freeze strategies
6. TOOLING_COMPLEXITY -> Number of build stages needed, tool installation ordering, inter-tool coordination
7. REPRODUCIBILITY_SUPPORT -> Whether CI config can be mined for build commands/versions, confidence level

You MUST follow this exact output format:
---
RECOMMENDED BASE IMAGE: <specific image:tag suggestion based on domain + build_tool>
BUILD STRATEGY: <1-2 sentences: single-stage vs multi-stage, why>
INSTALL COMMANDS: <exact commands the Dockerfile should use, based on build_tool>
ENVIRONMENT SETUP: <OS packages, env vars, platform flags needed based on environment_specificity>
DEPENDENCY HANDLING: <strategy based on dependency_transparency: lockfile approach, pinning, etc.>
BUILD COMPLEXITY NOTES: <warnings based on tooling_complexity: multi-tool coordination, ordering>
CI CONFIDENCE: <based on reproducibility_support: can we trust CI config? what to mine from it?>
CRITICAL WARNINGS: <anything the agent must be careful about given this combination of dimensions>
---"""


METAPROMPT_USER_TEMPLATE = """Repository: {repo_name}

7-DIMENSION TAXONOMY CLASSIFICATION:
- Domain: {domain}
- Build Tool: {build_tool}
- Automation Level: {automation_level}
- Environment Specificity: {environment_specificity}
- Dependency Transparency: {dependency_transparency}
- Tooling Complexity: {tooling_complexity}
- Reproducibility Support: {reproducibility_support}

Generate the Dockerfile creation context for this specific repository."""


def generate_metaprompt_context(
    llm: BaseChatModel,
    repo_name: str,
    taxonomy: Dict[str, str],
) -> str:
    """Generate tailored agent context from the 7-dimension taxonomy via a metaprompt LLM call.

    This REPLACES build_initial_context() from preparation.py.

    Args:
        llm: The LLM to use for the metaprompt call.
        repo_name: Repository name (owner/repo or just repo).
        taxonomy: 7-dimension classification dict.

    Returns:
        A string of structured, actionable context to inject into the agent's goal prompt.
    """
    user_msg = METAPROMPT_USER_TEMPLATE.format(
        repo_name=repo_name,
        domain=taxonomy.get("domain", "unknown"),
        build_tool=taxonomy.get("build_tool", "unknown"),
        automation_level=taxonomy.get("automation_level", "unknown"),
        environment_specificity=taxonomy.get("environment_specificity", "unknown"),
        dependency_transparency=taxonomy.get("dependency_transparency", "unknown"),
        tooling_complexity=taxonomy.get("tooling_complexity", "unknown"),
        reproducibility_support=taxonomy.get("reproducibility_support", "unknown"),
    )

    try:
        response = llm.invoke([
            SystemMessage(content=METAPROMPT_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        return response.content.strip()
    except Exception as e:
        logger.error(f"[Metaprompt] Generation failed: {e}")
        return _fallback_context(taxonomy)


def _fallback_context(taxonomy: Dict[str, str]) -> str:
    """Generate basic context without LLM if the metaprompt call fails."""
    lines = ["TAXONOMY-BASED CONTEXT (fallback — metaprompt LLM call failed):"]
    for dim, val in taxonomy.items():
        lines.append(f"  {dim}: {val}")
    return "\n".join(lines)
