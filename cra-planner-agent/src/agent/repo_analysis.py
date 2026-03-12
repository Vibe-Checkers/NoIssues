"""
Rich repository analysis for taxonomy classification.

Ports input creation techniques from repo-data-collection/src/experiments/generator.py
and provides verification hint derivation (replacing classification.py in the pipeline).

Four input variants are generated per repository:
  1. input_readme_only        — README text only
  2. input_readme_tree        — README + repository structure visualization
  3. input_readme_digest_hybrid — README + hybrid-sampled file contents
  4. input_readme_llm_select  — README + LLM-selected key file contents
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# README candidates in priority order
README_CANDIDATES = ("README.md", "README.rst", "README.txt", "README", "readme.md", "Readme.md")

# Max bytes to read per sampled file
MAX_FILE_BYTES = 2000


# ---------------------------------------------------------------------------
# Tree structure analysis (ported from generator.py)
# ---------------------------------------------------------------------------

def build_tree_structure(paths: List[str], max_depth: int = 3) -> dict:
    """Build a nested tree structure up to max_depth levels."""
    tree = {}
    for path in paths:
        parts = path.split("/")
        parts_to_process = parts[:max_depth]
        current = tree
        for i, part in enumerate(parts_to_process):
            if part not in current:
                current[part] = {"_type": "dir", "_children": {}}
            if i == len(parts_to_process) - 1:
                if i == len(parts) - 1:
                    current[part] = {"_type": "file"}
                else:
                    if "_children" not in current[part]:
                        current[part] = {"_type": "dir", "_children": {}, "_truncated": True}
                    else:
                        current[part]["_truncated"] = True
            else:
                current = current[part]["_children"]
    return tree


def tree_to_list(tree: dict, prefix: str = "", max_items: int = 200) -> List[str]:
    """Convert nested tree dict to a list of visual string representations."""
    lines = []
    items = sorted(tree.items(), key=lambda x: (x[1].get("_type", "dir") == "file", x[0]))
    for i, (name, node) in enumerate(items):
        if len(lines) >= max_items:
            lines.append(f"{prefix}... ({len(items) - i} more items)")
            break
        is_last = (i == len(items) - 1)
        current_prefix = "\u2514\u2500\u2500 " if is_last else "\u251c\u2500\u2500 "
        if node.get("_type") == "file":
            lines.append(f"{prefix}{current_prefix}{name}")
        else:
            truncated_marker = " (...)" if node.get("_truncated") else ""
            lines.append(f"{prefix}{current_prefix}{name}/{truncated_marker}")
            if "_children" in node and node["_children"]:
                child_prefix = prefix + ("    " if is_last else "\u2502   ")
                child_lines = tree_to_list(node["_children"], child_prefix, max_items - len(lines))
                lines.extend(child_lines)
    return lines


def summarize_tree(paths: List[str]) -> dict:
    """Compute tree statistics: file counts, ratios, depth, directory patterns."""
    exts_docs = {".md", ".rst", ".org", ".adoc", ".txt", ".ipynb"}
    exts_code = {
        ".c", ".cc", ".cpp", ".h", ".hpp", ".rs", ".go", ".java", ".kt", ".kts",
        ".py", ".rb", ".php", ".swift", ".m", ".mm", ".cs", ".ts", ".tsx", ".js",
        ".jsx", ".scala", ".lua", ".dart", ".zig", ".nim",
    }
    doc = code = other = 0
    topdirs: Dict[str, int] = {}
    depths: List[int] = []
    dir_patterns = {
        "src": 0, "lib": 0, "pkg": 0, "app": 0, "test": 0, "tests": 0,
        "__tests__": 0, "examples": 0, "example": 0, "algorithms": 0,
        "solutions": 0, "exercises": 0,
    }
    for p in paths:
        parts = p.split("/")
        depths.append(len(parts))
        if parts:
            topdirs[parts[0]] = topdirs.get(parts[0], 0) + 1
        p_lower = p.lower()
        for k in dir_patterns:
            if f"/{k}/" in f"/{p_lower}/" or p_lower.startswith(f"{k}/"):
                dir_patterns[k] += 1
        _, dot, ext = p.rpartition(".")
        ext = ("." + ext) if dot else ""
        if ext.lower() in exts_docs:
            doc += 1
        elif ext.lower() in exts_code:
            code += 1
        else:
            other += 1

    total = max(1, doc + code + other)
    avg_depth = sum(depths) / len(depths) if depths else 0.0
    max_depth = max(depths) if depths else 0
    depth_dist: Dict[int, int] = {}
    for d in depths:
        depth_dist[d] = depth_dist.get(d, 0) + 1

    tree_structure = build_tree_structure(paths, max_depth=3)
    tree_lines = tree_to_list(tree_structure, max_items=200)

    return {
        "total_files": doc + code + other,
        "doc_count": doc,
        "code_count": code,
        "other_count": other,
        "doc_ratio": round(doc / total, 3),
        "code_ratio": round(code / total, 3),
        "avg_depth": round(avg_depth, 2),
        "max_depth": max_depth,
        "depth_distribution": dict(sorted(depth_dist.items())[:10]),
        "topdirs": sorted(topdirs.items(), key=lambda x: x[1], reverse=True)[:12],
        "dir_patterns": {k: v for k, v in dir_patterns.items() if v > 0},
        "tree_structure": tree_lines,
    }


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def collect_repo_paths(repo_path: str) -> List[str]:
    """Walk repo filesystem and collect all relative file paths (skip .git)."""
    paths = []
    for root, dirs, files in os.walk(repo_path):
        if ".git" in dirs:
            dirs.remove(".git")
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), repo_path)
            paths.append(rel)
    return paths


def get_file_metadata(repo_path: str, paths: List[str]) -> List[dict]:
    """Extract metadata (size, extension, depth) for files on disk."""
    files = []
    for rel_path in paths:
        full_path = os.path.join(repo_path, rel_path)
        if os.path.isfile(full_path):
            size = os.path.getsize(full_path)
            ext = ""
            if "." in rel_path.split("/")[-1]:
                ext = "." + rel_path.rsplit(".", 1)[1].lower()
            depth = len(rel_path.split("/"))
            files.append({
                "path": rel_path,
                "size": size,
                "extension": ext,
                "depth": depth,
            })
    return files


def read_file_contents(repo_path: str, file_list: List[dict],
                       max_bytes: int = MAX_FILE_BYTES) -> Dict[str, str]:
    """Read contents of sampled files, capped at max_bytes each."""
    contents = {}
    for f in file_list:
        path = f["path"]
        if path in contents:
            continue
        try:
            with open(os.path.join(repo_path, path), "r", errors="replace") as fh:
                contents[path] = fh.read(max_bytes)
        except Exception:
            contents[path] = ""
    return contents


# ---------------------------------------------------------------------------
# File sampling strategies (ported from generator.py)
# ---------------------------------------------------------------------------

def sample_hybrid(file_metadata: List[dict]) -> List[dict]:
    """Hybrid sampling: combines extension, location, and size strategies."""
    sampled = []

    # By extension (top 2 extensions, 1 sample each)
    ext_counts: Dict[str, int] = {}
    for f in file_metadata:
        ext = f["extension"]
        if ext:
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
    top_exts = sorted(ext_counts.items(), key=lambda x: x[1], reverse=True)[:2]
    for ext, _ in top_exts:
        ext_files = [f for f in file_metadata if f["extension"] == ext]
        sampled.extend(ext_files[:1])

    # By location (root, top dirs, depths)
    root_files = [f for f in file_metadata if f["depth"] == 1]
    sampled.extend(root_files[:1])
    topdir_files: Dict[str, List[dict]] = {}
    for f in file_metadata:
        if f["depth"] > 1:
            topdir = f["path"].split("/")[0]
            if topdir not in topdir_files:
                topdir_files[topdir] = []
            topdir_files[topdir].append(f)
    for topdir, files in sorted(topdir_files.items())[:5]:
        sampled.extend(files[:1])
    for depth in range(2, 6):
        depth_files = [f for f in file_metadata if f["depth"] == depth]
        if depth_files:
            sampled.append(depth_files[0])

    # By size (top 3 largest)
    largest = sorted(file_metadata, key=lambda x: x["size"], reverse=True)[:3]
    sampled.extend(largest)

    # Deduplicate
    seen = set()
    unique = []
    for f in sampled:
        if f["path"] not in seen:
            seen.add(f["path"])
            unique.append(f)
    return unique


# ---------------------------------------------------------------------------
# LLM-based file selection (adapted from generator.py for langchain)
# ---------------------------------------------------------------------------

def build_file_selection_prompt(tree_structure: str, total_files: int) -> str:
    """Create prompt for LLM to select 5-7 representative files."""
    return f"""You are analyzing a GitHub repository to select the most representative files for classification.

Given the repository structure below, select 5-7 files that would be most helpful for understanding:
- The repository's domain and purpose
- The build tool and dependency management approach
- The automation and deployment setup
- The project architecture and complexity

SELECTION CRITERIA (in priority order):
1. Package/dependency manifests (package.json, requirements.txt, Cargo.toml, pom.xml, go.mod, etc.)
2. Build/deployment config (Dockerfile, Makefile, CMakeLists.txt, setup.py, etc.)
3. CI/CD configs (.github/workflows/*, .travis.yml, .gitlab-ci.yml, etc.)
4. Main entry points (main.py, index.js, App.tsx, main.go, Program.cs, etc.)
5. Core architecture files (routes/, models/, controllers/, src/lib/, etc.)

AVOID: Test files, examples, docs (unless repo is docs-focused), generated files, assets

REPOSITORY STRUCTURE:
{tree_structure}

Total files: {total_files}

Respond with ONLY a JSON array of 5-7 file paths:
["path/to/file1", "path/to/file2", ...]

Selected files:"""


def select_files_with_llm(llm: BaseChatModel, tree_lines: List[str],
                          file_metadata: List[dict],
                          max_files: int = 7) -> List[dict]:
    """Use LLM to select 5-7 most important files. Falls back to hybrid sampling."""
    try:
        tree_str = "\n".join(tree_lines[:500])
        prompt = build_file_selection_prompt(tree_str, len(file_metadata))

        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()

        json_match = re.search(r'\[.*?\]', content, re.DOTALL)
        if json_match:
            selected_paths = json.loads(json_match.group())
            path_set = {f["path"] for f in file_metadata}
            valid_paths = [p for p in selected_paths if p in path_set]
            selected = [f for f in file_metadata if f["path"] in valid_paths[:max_files]]

            if 5 <= len(selected) <= max_files:
                return selected
            elif len(selected) < 5:
                hybrid = sample_hybrid(file_metadata)
                combined = selected + [f for f in hybrid if f not in selected]
                return combined[:max_files]
            else:
                return selected[:max_files]

    except Exception as e:
        logger.warning(f"[RepoAnalysis] LLM file selection failed, using hybrid fallback: {e}")

    return sample_hybrid(file_metadata)


# ---------------------------------------------------------------------------
# Formatting functions (ported from generator.py)
# ---------------------------------------------------------------------------

def format_tree_structure(tree_lines: List[str]) -> str:
    return "\n".join(tree_lines)


def format_tree_stats(stats: dict) -> str:
    lines = [
        f"Total files: {stats.get('total_files', 0)}",
        f"Code files: {stats.get('code_count', 0)} ({stats.get('code_ratio', 0):.1%})",
        f"Doc files: {stats.get('doc_count', 0)} ({stats.get('doc_ratio', 0):.1%})",
        f"Average depth: {stats.get('avg_depth', 0):.1f}",
        f"Max depth: {stats.get('max_depth', 0)}",
    ]
    if stats.get('dir_patterns'):
        lines.append("\nDirectory patterns:")
        for k, v in stats['dir_patterns'].items():
            lines.append(f"  {k}: {v} files")
    if stats.get('topdirs'):
        lines.append("\nTop-level directories:")
        for name, count in stats['topdirs'][:5]:
            lines.append(f"  {name}/: {count} files")
    return "\n".join(lines)


def format_sampled_files(sampled: List[dict], file_contents: Dict[str, str]) -> str:
    lines = []
    for f in sampled:
        path = f["path"]
        content = file_contents.get(path, "")
        if content:
            lines.append(f"\n--- {path} ({f.get('size', 0)} bytes) ---")
            lines.append(content)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Input variant creation (ported from generator.py)
# ---------------------------------------------------------------------------

def create_input_variants(readme: Optional[str], tree_lines: List[str],
                          hybrid_sampled: List[dict],
                          llm_selected: List[dict],
                          file_contents: Dict[str, str]) -> dict:
    """Create the 4 input variants for taxonomy classification."""
    tree_str = format_tree_structure(tree_lines)
    structure_section = f"## Repository Structure\n\n{tree_str}"
    readme_text = readme or "(No README found)"

    # 1. README only
    input_readme_only = readme_text

    # 2. README + tree
    input_readme_tree = f"{readme_text}\n\n{structure_section}"

    # 3. README + hybrid-sampled files
    files_hybrid = format_sampled_files(hybrid_sampled, file_contents)
    input_readme_digest_hybrid = f"{readme_text}\n\n## Sample Files (hybrid)\n{files_hybrid}"

    # 4. README + LLM-selected files
    files_llm = format_sampled_files(llm_selected, file_contents)
    input_readme_llm_select = f"{readme_text}\n\n## Key Files (LLM-selected)\n{files_llm}"

    return {
        "input_readme_only": input_readme_only,
        "input_readme_tree": input_readme_tree,
        "input_readme_digest_hybrid": input_readme_digest_hybrid,
        "input_readme_llm_select": input_readme_llm_select,
    }


# ---------------------------------------------------------------------------
# Main entry point: gather rich inputs for a repository
# ---------------------------------------------------------------------------

def create_rich_inputs(repo_path: str, llm: BaseChatModel = None) -> dict:
    """
    Gather rich repository signals using 4 input techniques.

    Args:
        repo_path: Path to the cloned repository on disk.
        llm: Optional LLM for file selection (input_readme_llm_select).
             If None, falls back to hybrid sampling for that variant.

    Returns:
        Dict with keys: readme_text, tree_stats, input_readme_only,
        input_readme_tree, input_readme_digest_hybrid, input_readme_llm_select.
    """
    logger.info(f"[RepoAnalysis] Gathering rich inputs for {repo_path}")

    # Collect all file paths
    paths = collect_repo_paths(repo_path)

    # README
    readme = None
    for cand in README_CANDIDATES:
        readme_path = os.path.join(repo_path, cand)
        if os.path.exists(readme_path):
            try:
                with open(readme_path, "r", errors="replace") as f:
                    readme = f.read(MAX_FILE_BYTES)
            except Exception:
                pass
            break

    # Tree stats
    tree_stats = summarize_tree(paths)

    # File metadata + sampling
    file_metadata = get_file_metadata(repo_path, paths)
    hybrid_sampled = sample_hybrid(file_metadata)

    # LLM-based file selection
    if llm is not None:
        llm_selected = select_files_with_llm(
            llm, tree_stats["tree_structure"], file_metadata
        )
    else:
        llm_selected = hybrid_sampled  # fallback

    # Read file contents for all sampled files
    all_sampled = list({f["path"]: f for f in hybrid_sampled + llm_selected}.values())
    file_contents = read_file_contents(repo_path, all_sampled)

    # Create input variants
    variants = create_input_variants(
        readme=readme,
        tree_lines=tree_stats["tree_structure"],
        hybrid_sampled=hybrid_sampled,
        llm_selected=llm_selected,
        file_contents=file_contents,
    )

    logger.info(
        f"[RepoAnalysis] Rich inputs ready: {tree_stats['total_files']} files, "
        f"{len(hybrid_sampled)} hybrid samples, {len(llm_selected)} LLM-selected"
    )

    return {
        "readme_text": readme,
        "tree_stats": tree_stats,
        **variants,
    }


# ---------------------------------------------------------------------------
# Verification hint derivation (replaces classification.py for the pipeline)
# ---------------------------------------------------------------------------

# Canonical types and strategies (kept in sync with classification.py definitions)
REPO_TYPES = [
    "cli_tool", "web_service", "library", "native_library",
    "framework", "data_pipeline", "desktop_app", "documentation_only", "monorepo",
]

VERIFICATION_STRATEGIES = [
    "binary_run", "import_test", "link_test", "server_probe",
    "build_only", "multi_target",
]


def _detect_monorepo(repo_path: str) -> Tuple[bool, int]:
    """Detect monorepo by counting Maven/Gradle module descriptors."""
    repo = Path(repo_path)
    pom_count = len(list(repo.rglob("pom.xml")))
    gradle_count = len(list(repo.rglob("build.gradle")))
    gradle_kts_count = len(list(repo.rglob("build.gradle.kts")))
    module_count = pom_count + gradle_count + gradle_kts_count
    # 3+ module descriptors indicates monorepo
    is_monorepo = module_count >= 3
    return is_monorepo, module_count


def _extract_package_name(repo_path: str, language: str) -> Optional[str]:
    """Try to extract the importable package name from config files."""
    repo = Path(repo_path)

    if language in ("JavaScript", "TypeScript"):
        pkg_json = repo / "package.json"
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text(errors="ignore")[:2000])
                if data.get("name"):
                    return data["name"]
            except Exception:
                pass

    if language == "Python":
        for config_name in ("pyproject.toml", "setup.cfg"):
            config_path = repo / config_name
            if config_path.exists():
                try:
                    content = config_path.read_text(errors="ignore")[:2000]
                    m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', content)
                    if m:
                        return m.group(1)
                except Exception:
                    pass

    return None


def derive_verification_hints(repo_path: str, language: str,
                              taxonomy: dict) -> dict:
    """
    Derive repo_type, verification_strategy, and related fields from
    taxonomy dimensions + filesystem signals.

    This replaces classification.py's role in the pipeline.

    Args:
        repo_path: Path to the cloned repository.
        language: Primary language (from detect_project_language).
        taxonomy: 7-dimension taxonomy dict.

    Returns:
        Dict with: repo_type, verification_strategy, primary_language,
        build_system, package_name, binary_name, library_name,
        is_monorepo, module_count.
    """
    repo = Path(repo_path)
    domain = taxonomy.get("domain", "unknown") if taxonomy else "unknown"
    build_tool = taxonomy.get("build_tool", "unknown") if taxonomy else "unknown"

    # Monorepo detection
    is_monorepo, module_count = _detect_monorepo(repo_path)

    # Entrypoint heuristics
    has_cmd_dir = (repo / "cmd").is_dir()
    has_src_main_rs = (repo / "src" / "main.rs").exists()
    has_main_py = (repo / "main.py").exists() or (repo / "app.py").exists()
    has_manage_py = (repo / "manage.py").exists()

    # Check for code files
    code_extensions = {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".rb"}
    has_code = False
    try:
        for entry in repo.iterdir():
            if not entry.name.startswith(".") and entry.is_file():
                if any(entry.name.endswith(ext) for ext in code_extensions):
                    has_code = True
                    break
        if not has_code:
            # Check config files as proxy for code presence
            config_files = {"package.json", "Cargo.toml", "go.mod", "pom.xml",
                            "build.gradle", "setup.py", "pyproject.toml"}
            has_code = any((repo / cf).exists() for cf in config_files)
    except Exception:
        has_code = True  # assume code exists on error

    # --- Determine repo_type ---
    repo_type = "cli_tool"  # conservative default

    if not has_code:
        repo_type = "documentation_only"
    elif is_monorepo and module_count >= 3:
        repo_type = "monorepo"
    elif domain in ("web-development",) and (has_manage_py or has_main_py):
        repo_type = "web_service"
    elif domain in ("web-development",) and language in ("JavaScript", "TypeScript"):
        # Could be library or web_service — check for server indicators
        pkg_json = repo / "package.json"
        if pkg_json.exists():
            try:
                content = pkg_json.read_text(errors="ignore")[:2000]
                if '"start"' in content and ("express" in content or "next" in content
                                              or "react-scripts" in content):
                    repo_type = "web_service"
                else:
                    repo_type = "library"
            except Exception:
                repo_type = "library"
        else:
            repo_type = "library"
    elif language == "Go" and has_cmd_dir:
        repo_type = "cli_tool"
    elif language == "Rust" and has_src_main_rs:
        repo_type = "cli_tool"
    elif language in ("C", "C++"):
        # Check for library indicators
        if (repo / "CMakeLists.txt").exists():
            try:
                cmake_content = (repo / "CMakeLists.txt").read_text(errors="ignore")[:2000]
                if "add_library" in cmake_content:
                    repo_type = "native_library"
                else:
                    repo_type = "cli_tool"
            except Exception:
                repo_type = "cli_tool"
        else:
            repo_type = "cli_tool"
    elif domain in ("libraries", "frameworks"):
        repo_type = "library"
    elif domain in ("machine-learning", "data-science", "scientific-computing"):
        repo_type = "data_pipeline"
    elif domain in ("devops",):
        repo_type = "cli_tool"
    else:
        # Default based on language ecosystem norms
        if language in ("Python", "JavaScript", "TypeScript", "Ruby"):
            pkg_configs = {"package.json", "setup.py", "pyproject.toml",
                           "setup.cfg", "Gemfile"}
            if any((repo / cf).exists() for cf in pkg_configs):
                repo_type = "library"
        elif language == "Java":
            repo_type = "library"

    # --- Determine verification_strategy ---
    strategy = "build_only"  # safe default

    if repo_type == "documentation_only":
        strategy = "build_only"
    elif repo_type == "monorepo":
        strategy = "build_only" if module_count <= 10 else "multi_target"
    elif repo_type == "cli_tool":
        strategy = "binary_run"
    elif repo_type == "web_service":
        strategy = "server_probe"
    elif repo_type == "library":
        strategy = "import_test"
    elif repo_type == "native_library":
        strategy = "link_test"
    elif repo_type in ("data_pipeline", "desktop_app", "framework"):
        strategy = "build_only"

    # --- Extract package/binary/library names ---
    package_name = None
    binary_name = None
    library_name = None

    if strategy == "import_test":
        package_name = _extract_package_name(repo_path, language)
        if not package_name:
            # Fallback to repo name
            repo_name = os.path.basename(repo_path)
            package_name = repo_name

    elif strategy == "binary_run":
        binary_name = os.path.basename(repo_path)

    elif strategy == "link_test":
        library_name = os.path.basename(repo_path)

    return {
        "repo_type": repo_type,
        "verification_strategy": strategy,
        "primary_language": language,
        "build_system": build_tool,
        "package_name": package_name,
        "binary_name": binary_name,
        "library_name": library_name,
        "is_monorepo": is_monorepo,
        "module_count": module_count,
    }
