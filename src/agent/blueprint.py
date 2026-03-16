"""Phase 0 + Phase 1: Docker Image Catalog and Context Blueprint generation.

- ImageCatalog: fetches and caches Docker Hub official images (once per batch).
- select_build_files: LLM-based file selection with heuristic fallback.
- generate_blueprint: metaprompting to produce a structured build blueprint.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────

HEURISTIC_FILES = [
    "Dockerfile", "docker-compose.yml",
    "pom.xml", "build.gradle", "build.gradle.kts",
    "package.json", "pyproject.toml", "setup.py", "requirements.txt",
    "Cargo.toml", "go.mod", "CMakeLists.txt", "Makefile",
    ".github/workflows/ci.yml", ".github/workflows/build.yml",
]

EXTENSION_LANGUAGE_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".java": "java", ".go": "go", ".rs": "rust",
    ".c": "c", ".cpp": "cpp", ".cs": "csharp",
    ".rb": "ruby", ".php": "php", ".swift": "swift",
    ".kt": "kotlin", ".scala": "scala",
}

FILE_TREE_MAX_ENTRIES = 500
README_MAX_CHARS = 5000
FILE_CONTENT_MAX_CHARS = 20_000
MAX_TAGS_PER_IMAGE = 6


# ═══════════════════════════════════════════════════════
# Phase 0: Docker Official Image Catalog
# ═══════════════════════════════════════════════════════

class ImageCatalog:
    """Fetches and caches Docker Hub official image list with tags.

    Called once per batch. The resulting string is passed to all workers (read-only).
    """

    def __init__(self):
        self._catalog: str | None = None
        self._fetched_at: datetime | None = None

    def get(self, db=None) -> str:
        """Return cached catalog string, loading from DB or fetching from Docker Hub."""
        if self._catalog is None:
            if db is not None:
                cached = db.load_image_catalog()
                if cached:
                    logger.info("Image catalog loaded from DB cache")
                    self._catalog = cached
                    return self._catalog
            self._catalog = self._fetch_from_docker_hub()
            self._fetched_at = datetime.now(timezone.utc)
            if db is not None:
                db.save_image_catalog(self._catalog, self._catalog.count("\n"))
        return self._catalog

    def _fetch_from_docker_hub(self) -> str:
        """Paginate Docker Hub API to build compact catalog string."""
        logger.info("Fetching Docker Official Image catalog from Docker Hub...")
        images = self._fetch_image_list()
        lines = ["=== DOCKER OFFICIAL IMAGES ==="]

        for name in images:
            try:
                tags = self._fetch_tags(name)
                if tags:
                    lines.append(f"{name}: {', '.join(tags[:MAX_TAGS_PER_IMAGE])}")
            except Exception:
                logger.debug("Failed to fetch tags for %s, skipping", name)

        catalog = "\n".join(lines)
        logger.info("Image catalog: %d images, %d bytes", len(images), len(catalog))
        return catalog

    def _fetch_image_list(self) -> list[str]:
        """Paginate /v2/repositories/library/ to get all official image names."""
        images = []
        url = "https://hub.docker.com/v2/repositories/library/?page_size=100"

        while url:
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                for repo in data.get("results", []):
                    images.append(repo["name"])
                url = data.get("next")
            except Exception:
                logger.warning("Docker Hub pagination failed at %s", url)
                break

        return images

    def _fetch_tags(self, image_name: str) -> list[str]:
        """Fetch top tags for a single official image."""
        url = (
            f"https://hub.docker.com/v2/repositories/library/{image_name}"
            f"/tags/?page_size=25&ordering=last_updated"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        return [t["name"] for t in data.get("results", []) if t.get("name")]


# ═══════════════════════════════════════════════════════
# Phase 1: Context Blueprint
# ═══════════════════════════════════════════════════════

# ─── File Selector ───────────────────────────────────

FILE_SELECTOR_SYSTEM = (
    "You select files from a repository that are most relevant "
    "for building the project in a Docker container."
)

FILE_SELECTOR_USER = """\
FILE TREE:
{file_tree}

README (first 5000 chars):
{readme}

Select 3 to 5 files from the file tree that are most useful for understanding \
how to build and containerize this project. Prioritize:
- Build configuration (pom.xml, build.gradle, CMakeLists.txt, Makefile, Cargo.toml, etc.)
- Dependency manifests (package.json, requirements.txt, pyproject.toml, go.mod, etc.)
- CI/CD configs (.github/workflows/*.yml, .gitlab-ci.yml, Jenkinsfile)
- Existing Dockerfiles or docker-compose files
- Project entry points (main.py, src/main.rs, cmd/main.go, etc.) ONLY if no build config exists

Return ONLY a JSON array of file paths exactly as they appear in the tree. Example:
["package.json", "tsconfig.json", ".github/workflows/ci.yml"]"""


def select_build_files(repo_root: str | Path, llm) -> tuple[list[str], int, int]:
    """Select 3-5 build-relevant files from the repo.

    Returns (paths, prompt_tokens, completion_tokens).
    """
    repo_root = Path(repo_root)
    file_tree = generate_file_tree(repo_root)
    readme = read_readme(repo_root)

    try:
        response = llm.call_nano([
            {"role": "system", "content": FILE_SELECTOR_SYSTEM},
            {"role": "user", "content": FILE_SELECTOR_USER.format(
                file_tree=file_tree, readme=readme,
            )},
        ])
        paths = json.loads(response.content)
        paths = [p for p in paths if (repo_root / p).is_file()]

        if not paths:
            paths = _heuristic_file_selection(repo_root)

        return paths[:5], response.prompt_tokens, response.completion_tokens
    except Exception:
        logger.warning("File selector LLM call failed, using heuristic", exc_info=True)
        return _heuristic_file_selection(repo_root), 0, 0


def _heuristic_file_selection(repo_root: Path) -> list[str]:
    """Scan for known build/config filenames as fallback."""
    found = []
    for name in HEURISTIC_FILES:
        if (repo_root / name).is_file():
            found.append(name)
            if len(found) >= 5:
                break
    return found


# ─── Blueprint Generator ────────────────────────────

BLUEPRINT_SYSTEM = (
    "You are a Docker expert. You analyze repository files to produce "
    "a structured build blueprint that will guide an AI agent in generating a Dockerfile."
)

BLUEPRINT_USER = """\
SELECTED FILES:
{file_contents}

DOCKER OFFICIAL IMAGES (image: available tags):
{image_catalog}

Analyze the files above and produce a JSON build blueprint.

Return ONLY valid JSON in this exact format:
{{
  "language": "<primary language>",
  "build_system": "<build tool: npm, maven, gradle, cargo, go, cmake, make, pip, etc.>",
  "package_manager": "<npm, yarn, pnpm, pip, poetry, etc.>",
  "build_commands": ["<command1>", "<command2>"],
  "install_commands": ["<command1>", "<command2>"],
  "runtime_requirements": {{
    "language_version": "<e.g., 3.12, 22, 17>",
    "system_packages": ["<e.g., libssl-dev, gcc>"]
  }},
  "repo_type": "<library|cli_tool|web_service|data_pipeline|native_library|framework|desktop_app|monorepo>",
  "base_image": "<image:tag from the official images list>",
  "base_image_rationale": "<1 sentence why this image was chosen>",
  "pitfalls": ["<known issue 1>", "<known issue 2>"],
  "notes": "<any other relevant context for Dockerfile generation>"
}}

Rules:
- base_image MUST be selected from the DOCKER OFFICIAL IMAGES list above. Use a specific tag, not "latest".
- Prefer -slim variants unless build tools (gcc, make) are needed during build.
- Match the language version from the config files (e.g., if package.json engines says node 20, use node:20-slim).
- If the project needs a non-official image (nvidia/cuda, mcr.microsoft.com/dotnet/sdk), \
pick the closest official image and note the real requirement in pitfalls.
- build_commands should be the commands to build the project (e.g., ["npm ci", "npm run build"]).
- install_commands should be system-level installs needed before build (e.g., ["apt-get install -y libpq-dev"]).
- Be specific about versions. Don't guess — only include what the files explicitly state."""


def generate_blueprint(
    repo_root: str | Path,
    image_catalog: str,
    llm,
) -> tuple[dict, int, int]:
    """Generate context blueprint for a repository.

    Returns (blueprint_dict, total_prompt_tokens, total_completion_tokens).
    """
    repo_root = Path(repo_root)

    # Step 1: select files
    paths, sel_pt, sel_ct = select_build_files(repo_root, llm)

    # Step 2: read selected files
    file_contents = _read_selected_files(repo_root, paths)

    # Step 3: call metaprompt
    try:
        response = llm.call_nano([
            {"role": "system", "content": BLUEPRINT_SYSTEM},
            {"role": "user", "content": BLUEPRINT_USER.format(
                file_contents=file_contents,
                image_catalog=image_catalog,
            )},
        ])

        blueprint = json.loads(response.content)

        # Validate required fields
        for field in ("language", "build_system", "repo_type", "base_image"):
            if field not in blueprint:
                blueprint[field] = "unknown"

        total_pt = sel_pt + response.prompt_tokens
        total_ct = sel_ct + response.completion_tokens
        return blueprint, total_pt, total_ct

    except Exception:
        logger.warning("Blueprint metaprompt failed, using fallback", exc_info=True)
        lang = detect_language_by_extensions(repo_root)
        fallback = {
            "language": lang,
            "build_system": "unknown",
            "repo_type": "unknown",
            "base_image": None,
            "pitfalls": [],
            "notes": "Blueprint generation failed. Agent must analyze the repository from scratch.",
        }
        return fallback, sel_pt, sel_ct


# ─── Helpers ─────────────────────────────────────────

def _read_selected_files(repo_root: Path, paths: list[str]) -> str:
    """Read selected files and format for the metaprompt."""
    sections = []
    for p in paths[:5]:
        try:
            text = (repo_root / p).read_text(errors="replace")[:FILE_CONTENT_MAX_CHARS]
            sections.append(f"--- {p} ---\n{text}")
        except Exception:
            continue
    return "\n\n".join(sections)


def generate_file_tree(repo_root: Path) -> str:
    """Generate a compact file tree string for the repo."""
    lines = []
    count = 0
    for item in sorted(repo_root.rglob("*")):
        if count >= FILE_TREE_MAX_ENTRIES:
            lines.append(f"... ({count}+ files, truncated)")
            break

        rel = item.relative_to(repo_root)
        # Skip hidden dirs and common noise
        parts = rel.parts
        if any(p.startswith(".") and p not in (".github",) for p in parts):
            continue
        if any(p in ("node_modules", "__pycache__", "venv", ".venv", "target", "build", "dist") for p in parts):
            continue

        lines.append(str(rel))
        count += 1

    return "\n".join(lines)


def read_readme(repo_root: Path) -> str:
    """Read README file, returning first 5000 chars or empty string."""
    for name in ("README.md", "README.rst", "README.txt", "README"):
        path = repo_root / name
        if path.is_file():
            try:
                return path.read_text(errors="replace")[:README_MAX_CHARS]
            except Exception:
                continue
    return "(no README found)"


def detect_language_by_extensions(repo_root: Path) -> str:
    """Detect primary language by counting file extensions."""
    counts: dict[str, int] = {}
    for item in repo_root.rglob("*"):
        if item.is_file():
            ext = item.suffix.lower()
            if ext in EXTENSION_LANGUAGE_MAP:
                lang = EXTENSION_LANGUAGE_MAP[ext]
                counts[lang] = counts.get(lang, 0) + 1

    if not counts:
        return "unknown"
    return max(counts, key=counts.get)
