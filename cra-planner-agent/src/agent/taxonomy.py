"""
7-Dimension Repository Taxonomy Classifier.

Dual-mode: CSV lookup from pre-computed majority-vote data (instant),
with live LLM classification as fallback for out-of-sample repos.

Dimensions:
  1. domain
  2. build_tool
  3. automation_level
  4. environment_specificity
  5. dependency_transparency
  6. tooling_complexity
  7. reproducibility_support
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

TAXONOMY_DIMENSIONS = [
    "domain",
    "build_tool",
    "automation_level",
    "environment_specificity",
    "dependency_transparency",
    "tooling_complexity",
    "reproducibility_support",
]

# Default CSV path — override with TAXONOMY_CSV_PATH env var
_DEFAULT_CSV_PATH = os.getenv(
    "TAXONOMY_CSV_PATH",
    str(
        Path(__file__).resolve().parents[4]
        / "repo-data-collection"
        / "stratified_repos_2000_majority_vote.csv"
    ),
)

_csv_cache: Optional[Dict[str, Dict]] = None


# ---------------------------------------------------------------------------
# CSV Lookup
# ---------------------------------------------------------------------------

def _load_csv(csv_path: str = None) -> Dict[str, Dict]:
    """Load majority-vote CSV into a dict keyed by normalized repo_link URL."""
    global _csv_cache
    if _csv_cache is not None:
        return _csv_cache

    path = csv_path or _DEFAULT_CSV_PATH
    if not os.path.exists(path):
        logger.warning(f"[Taxonomy] CSV not found at {path}")
        _csv_cache = {}
        return _csv_cache

    lookup: Dict[str, Dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("repo_link", "").rstrip("/")
            if not url:
                continue
            lookup[url] = {
                "domain": row.get("domain", "unknown"),
                "build_tool": row.get("build_type", "unknown"),  # CSV uses build_type
                "automation_level": row.get("automation_level", "unknown"),
                "environment_specificity": row.get("environment_specificity", "unknown"),
                "dependency_transparency": row.get("dependency_transparency", "unknown"),
                "tooling_complexity": row.get("tooling_complexity", "unknown"),
                "reproducibility_support": row.get("reproducibility_support", "unknown"),
            }

    _csv_cache = lookup
    logger.info(f"[Taxonomy] Loaded {len(lookup)} repos from CSV")
    return _csv_cache


def lookup_taxonomy(repo_url: str, csv_path: str = None) -> Optional[Dict[str, str]]:
    """Look up pre-computed taxonomy from the majority-vote CSV.

    Returns a dict with the 7 dimension values, or None if not found.
    """
    lookup = _load_csv(csv_path)
    normalized = repo_url.rstrip("/").replace(".git", "")
    return lookup.get(normalized)


# ---------------------------------------------------------------------------
# Live LLM Classification (fallback)
# ---------------------------------------------------------------------------

# Exact prompt from repo-data-collection/src/experiments/runner.py build_base_prompt()
TAXONOMY_PROMPT = """Analyze this GitHub repository and classify it according to these criteria:

1. DOMAIN: The primary domain/purpose of the repository.
   Examples: web-development, mobile-development, machine-learning, data-science,
   systems-programming, devops, game-development, security, scientific-computing,
   documentation, education, utilities, etc.
   Note: Choose "documentation" for curated lists/awesome-lists/guides. Choose "devops"
   only for infrastructure/deployment tooling, NOT for projects that merely use Docker.

2. BUILD_TOOL: The primary build tool or package manager used.
   Examples: npm, cargo, maven, gradle, bazel, cmake, make, pip, poetry, go-mod, etc.
   For Python: Use "pip" if only requirements.txt/pyproject.toml exists. Use "poetry" if
   poetry.lock is present. Use "setuptools" ONLY if setup.py with custom build logic exists.
   Use "none" if no clear build tool is present.

3. AUTOMATION_LEVEL: How much of the build/setup is automated. Pick ONE:
   - "fully_automated": Single documented command runs the project end-to-end (e.g., `npm start`,
     `docker-compose up`, `cargo run`). No manual env vars, config edits, or pre-install steps needed.
   - "semi_automated": Build/run works but requires 2-5 prerequisite steps: setting env vars,
     editing config files, running multiple commands in sequence, or installing system dependencies.
   - "manual": Build is described as imperative steps without cohesive script
   - "reverse_engineering": No meaningful build/install instructions

4. ENVIRONMENT_SPECIFICITY: How constrained the environment is. Pick ONE:
   - "cross_platform_generic": Works on generic POSIX or mainstream platforms
   - "specific_os": Targets specific OS (Ubuntu, Windows, macOS only)
   - "specific_stack_versions": Requires particular distro, kernel, CUDA version, etc.
   - "custom_hardware_or_drivers": Requires GPUs/TPUs or special hardware/drivers

5. DEPENDENCY_TRANSPARENCY: How clearly dependencies are specified. Pick ONE:
   - "explicit_machine_readable": Lockfile present (package-lock.json, yarn.lock, Cargo.lock,
     poetry.lock, go.sum) OR all versions explicitly pinned with exact versions (==, =).
   - "explicit_loose": Dependencies listed in manifest but with ranges (>=, ~, ^) or unpinned.
   - "implicit": Only mentioned in prose, no formal manifest
   - "opaque": Must be inferred from imports or build errors

6. TOOLING_COMPLEXITY: How many build tools are involved. Pick ONE:
   - "single_layer_tool": One ecosystem handles deps AND build (pip, npm, cargo). Docker for
     deployment only (not build orchestration) still counts as single layer.
   - "multi_layer_toolchains": Build requires executing multiple distinct tools in sequence
     (e.g., Make calling npm AND webpack, shell scripts coordinating cmake + pip).
   - "mixed_languages_and_bindings": Multiple languages and bindings coordinated
   - "requires_external_services_to_build": Depends on external services/private infrastructure

7. REPRODUCIBILITY_SUPPORT: How well the project supports reproducible builds. Pick ONE:
   - "repro_ready": GitHub Actions/CI config files present (/.github/workflows/, .travis.yml, etc.)
   - "partial_repro": Some CI/scripts exist but flaky or incomplete
   - "no_ci_manual_only": No CI, only manual steps described
   - "broken_or_outdated": Instructions reference deprecated tools or are clearly broken

Respond ONLY with valid JSON in this exact format:
{
  "domain": "...",
  "build_tool": "...",
  "automation_level": "...",
  "environment_specificity": "...",
  "dependency_transparency": "...",
  "tooling_complexity": "...",
  "reproducibility_support": "..."
}
"""


def classify_taxonomy_live(
    llm: BaseChatModel,
    repo_path: str,
    signals: Dict,
) -> Dict[str, str]:
    """Classify repo using the 7-dimension taxonomy via a live LLM call.

    Uses signals already gathered by RepoClassifier.gather_signals().
    """
    repo_input = f"""README (excerpt):
{signals.get('readme_excerpt', 'N/A')[:2000]}

TOP-LEVEL FILES: {json.dumps(signals.get('top_level_entries', []))}
CONFIG FILES: {json.dumps(list(signals.get('config_files', {}).keys()))}
CONFIG FILE CONTENTS:
{json.dumps(signals.get('config_files', {}), indent=2)[:2500]}
"""

    prompt = f"{TAXONOMY_PROMPT}\n\nRepository information:\n{repo_input}"

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        # Strip markdown fences
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed = json.loads(content)
        return {dim: parsed.get(dim, "unknown") for dim in TAXONOMY_DIMENSIONS}

    except Exception as e:
        logger.warning(f"[Taxonomy] Live LLM classification failed: {e}")
        return {dim: "unknown" for dim in TAXONOMY_DIMENSIONS}


# ---------------------------------------------------------------------------
# Unified Entry Point
# ---------------------------------------------------------------------------

def get_taxonomy(
    repo_url: str,
    llm: BaseChatModel = None,
    repo_path: str = None,
    signals: Dict = None,
    csv_path: str = None,
) -> Optional[Dict[str, str]]:
    """Get 7-dimension taxonomy for a repository.

    Strategy: CSV lookup first (free, instant), live LLM fallback.

    Args:
        repo_url: GitHub URL of the repository.
        llm: LLM instance for live classification fallback.
        repo_path: Path to cloned repo (needed for live classification).
        signals: Pre-gathered repo signals (from RepoClassifier.gather_signals()).
        csv_path: Override path to the majority-vote CSV.

    Returns:
        Dict with 7 taxonomy dimensions, or None if unavailable.
    """
    # Try CSV lookup first
    result = lookup_taxonomy(repo_url, csv_path)
    if result:
        logger.info(f"[Taxonomy] CSV hit for {repo_url}")
        return result

    # Fallback: live LLM classification
    if llm and (signals or repo_path):
        logger.info(f"[Taxonomy] CSV miss for {repo_url} — using live LLM")
        if not signals and repo_path:
            from .classification import RepoClassifier
            signals = RepoClassifier().gather_signals(repo_path)
        return classify_taxonomy_live(llm, repo_path, signals)

    logger.warning(f"[Taxonomy] No CSV data and no LLM available for {repo_url}")
    return None
