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

FAILURE-AWARE DIRECTIVE:
Before generating outputs, assess risk level for this domain and build_tool combination.
High-risk combinations (historically >60% failure rate):
  - mobile-development + gradle/cmake (native SDK layers, emulator deps)
  - systems-programming + cmake/make (glibc vs musl incompatibility risk)
  - web-development + npm (npm ci lockfile issues, node_modules COPY patterns)
  - data-science + pip (CUDA/cuDNN version mismatches, heavy deps)
  - monorepo + maven/gradle (SNAPSHOT deps, multi-module build order)
For HIGH-RISK combos: expand reasoning depth — provide explicit command examples,
fallback logic, and cross-checks. For LOW-RISK: keep concise.

Given a repository's 7-dimension taxonomy classification, generate PRECISE, ACTIONABLE context
that a Dockerfile-writing agent will use. Your output must be structured and directly applicable.

For each dimension, derive the specific Dockerfile implications:

1. DOMAIN -> Distinguish runtime vs build-time image families, identify language ecosystem,
   and specify whether ENTRYPOINT should invoke a runtime, CLI, or test harness.
   Domain heuristics:
   - web-development: node-based multi-stage with static asset copy; watch for node_modules COPY
   - mobile-development: Android SDK layers + emulator deps; prefer ubuntu:22.04 over Alpine
   - data-science/ml: CUDA/cuDNN base; numpy/scipy need libgomp, libffi-dev
   - systems-programming: check glibc vs musl; Alpine incompatible with many native binaries
   - devops/infra: include CLI tools (kubectl, helm, terraform) with pinned versions
2. BUILD_TOOL -> Exact install/build commands, required tool version and installation source,
   multi-stage build patterns, cache optimization, and artifact handoff between stages.
3. AUTOMATION_LEVEL -> How much the agent can rely on documented commands vs needing to reverse-engineer.
4. ENVIRONMENT_SPECIFICITY -> Platform flags, OS-specific packages, version pinning requirements,
   and justification for chosen OS family (Debian vs Alpine vs Ubuntu).
   Rule: prefer Debian-based if tooling_complexity > single_layer or domain is mobile/systems.
5. DEPENDENCY_TRANSPARENCY -> Whether to use lockfiles, how to handle implicit deps, pip freeze strategies,
   and fallback method if no lockfile is found (e.g., pip freeze, npm ls, cargo metadata).
6. TOOLING_COMPLEXITY -> Number of build stages needed, tool installation ordering, inter-tool coordination,
   and explicit stage mapping (what runs where, what artifacts are copied forward).
7. REPRODUCIBILITY_SUPPORT -> Whether CI config can be mined for build commands/versions, confidence level,
   and specific CI files to inspect (e.g., .github/workflows, Jenkinsfile).

You MUST follow this exact output format:
---
RISK LEVEL: <HIGH / MEDIUM / LOW — based on domain+build_tool failure history>
RECOMMENDED BASE IMAGE: <specific image:tag suggestion based on domain + build_tool>
TOOLCHAIN VERSIONING: <exact versions of language runtimes and build tools inferred from taxonomy or CI config; explain fallback if version unknown>
BUILD STRATEGY: <1-2 sentences: single-stage vs multi-stage, why>
BUILD STAGE MAP: <enumerate each stage (builder, runtime, test), what tools run in each, and what artifacts are copied forward>
INSTALL COMMANDS: <exact commands the Dockerfile should use, based on build_tool>
ENVIRONMENT SETUP: <OS packages, env vars, platform flags needed based on environment_specificity; include justification for chosen OS family>
DEPENDENCY HANDLING: <strategy based on dependency_transparency: lockfile approach, pinning; if no lockfile found, describe fallback method>
BUILD COMPLEXITY NOTES: <warnings based on tooling_complexity: multi-tool coordination, ordering>
CI CONFIDENCE: <based on reproducibility_support: can we trust CI config? what to mine from it? list specific CI files and extractable build commands/versions>
CROSS-DIMENSION CONSISTENCY CHECK: <verify that base image, build tool, and environment setup are mutually compatible; flag glibc/musl conflicts, SDK version mismatches>
CRITICAL WARNINGS: <domain/build_tool-specific pitfalls to avoid (e.g., Gradle daemon memory, npm cache path, cargo target dir)>
VALIDATION CHECKPOINTS: <list 2-3 concrete checks the agent should perform to verify the Dockerfile before calling VerifyBuild: confirm tool versions, validate COPY paths exist, test ENTRYPOINT resolves>
FAILURE-WEIGHTED PRIORITY: <for HIGH-RISK combos, list explicit fallback commands and compatibility matrix; for LOW-RISK, one sentence>
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
