"""
Repository characterization using a 7-dimension taxonomy.

This module is intentionally lightweight and runs at startup (after clone)
to produce the manuscript-aligned characterization artifact before agent
generation starts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any


logger = logging.getLogger(__name__)


class RepositoryTaxonomyAnalyzer:
    """
    Analyze repository signals and emit a 7-dimension taxonomy.

    Dimensions:
      - domain
      - build_tool
      - automation_level
      - environment_specificity
      - dependency_transparency
      - tooling_complexity
      - repro_support
    """

    def _read_readme_excerpt(self, repo: Path, max_chars: int = 3500) -> str:
        for readme_name in ("README.md", "README.rst", "README.txt", "README"):
            p = repo / readme_name
            if p.exists():
                try:
                    return p.read_text(errors="ignore")[:max_chars]
                except Exception:
                    return ""
        return ""

    def _collect_signals(self, repo_path: str) -> Dict[str, Any]:
        repo = Path(repo_path)
        signals: Dict[str, Any] = {
            "readme_excerpt": self._read_readme_excerpt(repo),
            "top_level_entries": [],
            "has_dockerfile": (repo / "Dockerfile").exists(),
            "config_files": {},
            "language_markers": set(),
        }

        try:
            signals["top_level_entries"] = sorted(
                [
                    f.name + ("/" if f.is_dir() else "")
                    for f in repo.iterdir()
                    if not f.name.startswith(".")
                ]
            )[:80]
        except Exception:
            pass

        config_candidates = [
            "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile",
            "package.json", "pnpm-lock.yaml", "yarn.lock",
            "Cargo.toml",
            "go.mod",
            "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
            "CMakeLists.txt", "Makefile", "configure.ac", "meson.build",
            "Gemfile", "composer.json",
            ".github/workflows",
            ".dockerignore",
        ]

        for cf in config_candidates:
            p = repo / cf
            if p.exists():
                if p.is_dir():
                    signals["config_files"][cf] = "<dir>"
                else:
                    try:
                        signals["config_files"][cf] = p.read_text(errors="ignore")[:1000]
                    except Exception:
                        signals["config_files"][cf] = "<unreadable>"

        # Minimal language marker hints for domain/tooling decisions
        markers = signals["language_markers"]
        cfg = signals["config_files"]
        if any(k in cfg for k in ("pyproject.toml", "setup.py", "requirements.txt", "Pipfile")):
            markers.add("python")
        if "package.json" in cfg:
            markers.add("node")
        if "Cargo.toml" in cfg:
            markers.add("rust")
        if "go.mod" in cfg:
            markers.add("go")
        if any(k in cfg for k in ("pom.xml", "build.gradle", "build.gradle.kts")):
            markers.add("jvm")
        if any(k in cfg for k in ("CMakeLists.txt", "Makefile", "configure.ac", "meson.build")):
            markers.add("native")

        return signals

    def _infer_build_tool(self, signals: Dict[str, Any]) -> str:
        cfg = signals.get("config_files", {})
        if "pom.xml" in cfg:
            return "maven"
        if "build.gradle" in cfg or "build.gradle.kts" in cfg:
            return "gradle"
        if "Cargo.toml" in cfg:
            return "cargo"
        if "go.mod" in cfg:
            return "go"
        if "package.json" in cfg:
            if "pnpm-lock.yaml" in cfg:
                return "pnpm"
            if "yarn.lock" in cfg:
                return "yarn"
            return "npm"
        if "pyproject.toml" in cfg or "setup.py" in cfg or "requirements.txt" in cfg:
            return "pip"
        if "CMakeLists.txt" in cfg:
            return "cmake"
        if "Meson.build" in cfg or "meson.build" in cfg:
            return "meson"
        if "Makefile" in cfg or "configure.ac" in cfg:
            return "make"
        if "Gemfile" in cfg:
            return "bundler"
        if "composer.json" in cfg:
            return "composer"
        return "none"

    def _infer_domain(self, signals: Dict[str, Any], build_tool: str) -> str:
        readme = (signals.get("readme_excerpt") or "").lower()
        entries = [e.lower() for e in signals.get("top_level_entries", [])]
        cfg = signals.get("config_files", {})

        has_code_markers = build_tool != "none" or any(
            e.endswith(ext) for e in entries for ext in (".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp")
        )
        if not has_code_markers:
            return "documentation"

        if any(k in readme for k in ("web", "http", "api", "frontend", "backend", "server")):
            return "web-development"
        if any(k in readme for k in ("machine learning", "deep learning", "neural", "model training", "pytorch", "tensorflow")):
            return "machine-learning"
        if build_tool in ("cmake", "meson", "make") or "native" in signals.get("language_markers", set()):
            return "systems-programming"
        if any(k in readme for k in ("cli", "command line", "terminal")):
            return "developer-tooling"
        return "general-software"

    def _infer_automation_level(self, signals: Dict[str, Any]) -> str:
        cfg = signals.get("config_files", {})
        readme = (signals.get("readme_excerpt") or "").lower()
        has_ci = ".github/workflows" in cfg

        if has_ci and any(x in readme for x in ("install", "build", "test")):
            return "fully_automated"
        if any(x in cfg for x in ("Makefile", "package.json", "pyproject.toml", "pom.xml", "build.gradle")):
            return "semi_automated"
        if readme:
            return "manual"
        return "reverse_engineering"

    def _infer_environment_specificity(self, signals: Dict[str, Any]) -> str:
        text = (
            (signals.get("readme_excerpt") or "")
            + "\n"
            + "\n".join(str(v) for v in (signals.get("config_files") or {}).values())
        ).lower()

        if any(k in text for k in ("cuda", "gpu", "tpu", "nvidia", "rocm")):
            return "custom_hardware_or_drivers"
        if any(k in text for k in ("ubuntu", "debian", "alpine", "macos", "windows only", "linux only")):
            return "specific_os"
        if any(k in text for k in ("requires", "version", "jdk", "python", "node", "gcc")):
            return "specific_stack_versions"
        return "cross_platform_generic"

    def _infer_dependency_transparency(self, signals: Dict[str, Any]) -> str:
        cfg = signals.get("config_files", {})
        readme = (signals.get("readme_excerpt") or "").lower()

        if any(k in cfg for k in ("poetry.lock", "Pipfile.lock", "package-lock.json", "pnpm-lock.yaml", "yarn.lock")):
            return "explicit_machine_readable"
        if any(k in cfg for k in ("pyproject.toml", "setup.py", "requirements.txt", "package.json", "pom.xml", "build.gradle", "Cargo.toml", "go.mod")):
            return "explicit_loose"
        if any(k in readme for k in ("dependency", "requirements", "install")):
            return "implicit"
        return "opaque"

    def _infer_tooling_complexity(self, signals: Dict[str, Any], build_tool: str) -> str:
        markers = signals.get("language_markers", set())
        cfg = signals.get("config_files", {})

        if any(k in cfg for k in ("docker-compose.yml", "docker-compose.yaml", "helm", "k8s")):
            return "requires_external_services"
        if len(markers) >= 2:
            return "mixed_languages_and_bindings"
        if build_tool in ("maven", "gradle") and "package.json" in cfg:
            return "multi_layer_toolchains"
        return "single_layer_tool"

    def _infer_repro_support(self, signals: Dict[str, Any]) -> str:
        cfg = signals.get("config_files", {})
        readme = (signals.get("readme_excerpt") or "").lower()

        if ".github/workflows" in cfg and any(k in readme for k in ("test", "ci", "build status")):
            return "repro_ready"
        if ".github/workflows" in cfg:
            return "partial_repro"
        if readme:
            return "no_ci_manual_only"
        return "broken_or_outdated"

    def classify(self, repo_path: str, repo_name: str) -> Dict[str, Any]:
        """Return a normalized 7-dimension taxonomy artifact."""
        signals = self._collect_signals(repo_path)
        build_tool = self._infer_build_tool(signals)

        taxonomy = {
            "domain": self._infer_domain(signals, build_tool),
            "build_tool": build_tool,
            "automation_level": self._infer_automation_level(signals),
            "environment_specificity": self._infer_environment_specificity(signals),
            "dependency_transparency": self._infer_dependency_transparency(signals),
            "tooling_complexity": self._infer_tooling_complexity(signals, build_tool),
            "repro_support": self._infer_repro_support(signals),
            "has_existing_dockerfile": bool(signals.get("has_dockerfile", False)),
            "_meta": {
                "repo": repo_name,
                "signals": {
                    "top_level_entries": signals.get("top_level_entries", []),
                    "config_files": list((signals.get("config_files") or {}).keys()),
                },
            },
        }
        logger.info("[Taxonomy] %s => %s", repo_name, json.dumps(taxonomy, ensure_ascii=False))
        return taxonomy

