"""Tests for Phase C: Blueprint Pipeline (needs LLM mocks, no Docker)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.blueprint import (
    ImageCatalog,
    select_build_files,
    generate_blueprint,
    generate_file_tree,
    read_readme,
    detect_language_by_extensions,
    _heuristic_file_selection,
)

FIXTURES = Path(__file__).parent / "fixtures" / "toy_python_project"


# ═══════════════════════════════════════════════════════
# ImageCatalog
# ═══════════════════════════════════════════════════════

class TestImageCatalog:
    def test_fetch_and_cache(self):
        """Mock Docker Hub responses, verify pagination and format."""
        catalog = ImageCatalog()

        page1 = {
            "results": [{"name": "python"}, {"name": "node"}],
            "next": "https://hub.docker.com/v2/repositories/library/?page=2",
        }
        page2 = {
            "results": [{"name": "golang"}],
            "next": None,
        }
        python_tags = {
            "results": [{"name": "3.12"}, {"name": "3.12-slim"}, {"name": "3.11"}],
        }
        node_tags = {
            "results": [{"name": "22"}, {"name": "22-slim"}],
        }
        golang_tags = {
            "results": [{"name": "1.22"}, {"name": "1.22-alpine"}],
        }

        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()

            if "page=2" in url:
                resp.json.return_value = page2
            elif "/library/?" in url or url.endswith("/library/"):
                resp.json.return_value = page1
            elif "/python/tags" in url:
                resp.json.return_value = python_tags
            elif "/node/tags" in url:
                resp.json.return_value = node_tags
            elif "/golang/tags" in url:
                resp.json.return_value = golang_tags
            else:
                resp.json.return_value = {"results": []}
            return resp

        with patch("agent.blueprint.requests.get", side_effect=mock_get):
            result = catalog.get()

        assert "=== DOCKER OFFICIAL IMAGES ===" in result
        assert "python: 3.12, 3.12-slim, 3.11" in result
        assert "node: 22, 22-slim" in result
        assert "golang: 1.22, 1.22-alpine" in result

        # Verify caching — second call doesn't fetch again
        with patch("agent.blueprint.requests.get", side_effect=Exception("should not be called")):
            result2 = catalog.get()
        assert result2 == result

    def test_fetch_skips_failed_tags(self):
        """Images with failed tag fetches are skipped."""
        catalog = ImageCatalog()

        def mock_get(url, **kwargs):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "/library/?" in url:
                resp.json.return_value = {
                    "results": [{"name": "python"}, {"name": "broken"}],
                    "next": None,
                }
            elif "/python/tags" in url:
                resp.json.return_value = {"results": [{"name": "3.12"}]}
            elif "/broken/tags" in url:
                raise Exception("connection failed")
            return resp

        with patch("agent.blueprint.requests.get", side_effect=mock_get):
            result = catalog.get()

        assert "python: 3.12" in result
        assert "broken" not in result


# ═══════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════

class TestHelpers:
    def test_generate_file_tree(self):
        tree = generate_file_tree(FIXTURES)
        assert "pyproject.toml" in tree
        assert "main.py" in tree
        assert "requirements.txt" in tree

    def test_read_readme(self):
        readme = read_readme(FIXTURES)
        assert "Toy Python Project" in readme

    def test_read_readme_missing(self, tmp_path):
        result = read_readme(tmp_path)
        assert "no README" in result

    def test_detect_language(self):
        lang = detect_language_by_extensions(FIXTURES)
        assert lang == "python"

    def test_detect_language_empty(self, tmp_path):
        lang = detect_language_by_extensions(tmp_path)
        assert lang == "unknown"

    def test_heuristic_file_selection(self):
        found = _heuristic_file_selection(FIXTURES)
        assert "pyproject.toml" in found or "requirements.txt" in found
        assert len(found) <= 5


# ═══════════════════════════════════════════════════════
# File Selector (with mock LLM)
# ═══════════════════════════════════════════════════════

class TestFileSelector:
    def test_select_with_valid_llm_response(self):
        """LLM returns valid paths — they are validated and returned."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '["pyproject.toml", "requirements.txt", "main.py"]'
        mock_response.prompt_tokens = 200
        mock_response.completion_tokens = 30
        mock_llm.call_nano.return_value = mock_response

        paths, pt, ct = select_build_files(FIXTURES, mock_llm)
        assert "pyproject.toml" in paths
        assert "requirements.txt" in paths
        assert pt == 200
        assert ct == 30

    def test_select_filters_invalid_paths(self):
        """Invalid paths from LLM are filtered out."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '["pyproject.toml", "nonexistent.txt", "also_fake.yml"]'
        mock_response.prompt_tokens = 100
        mock_response.completion_tokens = 20
        mock_llm.call_nano.return_value = mock_response

        paths, _, _ = select_build_files(FIXTURES, mock_llm)
        assert "pyproject.toml" in paths
        assert "nonexistent.txt" not in paths

    def test_select_falls_back_on_empty_llm_result(self):
        """If LLM returns all invalid paths, falls back to heuristic."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '["fake1.txt", "fake2.txt"]'
        mock_response.prompt_tokens = 50
        mock_response.completion_tokens = 10
        mock_llm.call_nano.return_value = mock_response

        paths, _, _ = select_build_files(FIXTURES, mock_llm)
        # Should have heuristic results
        assert len(paths) > 0
        assert any(p in HEURISTIC_FILES_SET for p in paths)

    def test_select_falls_back_on_llm_error(self):
        """LLM call failure triggers heuristic fallback."""
        mock_llm = MagicMock()
        mock_llm.call_nano.side_effect = Exception("API error")

        paths, pt, ct = select_build_files(FIXTURES, mock_llm)
        assert len(paths) > 0
        assert pt == 0
        assert ct == 0


# Need this for the test above
HEURISTIC_FILES_SET = set([
    "Dockerfile", "docker-compose.yml",
    "pom.xml", "build.gradle", "build.gradle.kts",
    "package.json", "pyproject.toml", "setup.py", "requirements.txt",
    "Cargo.toml", "go.mod", "CMakeLists.txt", "Makefile",
    ".github/workflows/ci.yml", ".github/workflows/build.yml",
])


# ═══════════════════════════════════════════════════════
# Blueprint Generator (with mock LLM)
# ═══════════════════════════════════════════════════════

class TestBlueprintGenerator:
    def _make_llm(self, selector_response: str, blueprint_response: str):
        """Create a mock LLM that returns different responses for selector vs blueprint."""
        mock_llm = MagicMock()
        call_count = [0]

        def call_nano(messages, **kwargs):
            call_count[0] += 1
            resp = MagicMock()
            resp.prompt_tokens = 100
            resp.completion_tokens = 50
            if call_count[0] == 1:
                resp.content = selector_response
            else:
                resp.content = blueprint_response
            return resp

        mock_llm.call_nano.side_effect = call_nano
        return mock_llm

    def test_successful_blueprint(self):
        """Full pipeline: file selection + metaprompt → blueprint dict."""
        blueprint_json = json.dumps({
            "language": "python",
            "build_system": "pip",
            "package_manager": "pip",
            "build_commands": ["pip install -e ."],
            "install_commands": [],
            "runtime_requirements": {"language_version": "3.12", "system_packages": []},
            "repo_type": "library",
            "base_image": "python:3.12-slim",
            "base_image_rationale": "Python 3.12 specified in pyproject.toml",
            "pitfalls": [],
            "notes": "",
        })

        llm = self._make_llm(
            '["pyproject.toml", "requirements.txt"]',
            blueprint_json,
        )

        bp, pt, ct = generate_blueprint(FIXTURES, "fake catalog", llm)
        assert bp["language"] == "python"
        assert bp["base_image"] == "python:3.12-slim"
        assert bp["repo_type"] == "library"
        assert pt > 0

    def test_missing_fields_filled_with_unknown(self):
        """Blueprint with missing required fields gets 'unknown' defaults."""
        llm = self._make_llm(
            '["pyproject.toml"]',
            '{"language": "python"}',  # missing build_system, repo_type, base_image
        )

        bp, _, _ = generate_blueprint(FIXTURES, "catalog", llm)
        assert bp["language"] == "python"
        assert bp["build_system"] == "unknown"
        assert bp["repo_type"] == "unknown"
        assert bp["base_image"] == "unknown"

    def test_metaprompt_failure_returns_fallback(self):
        """If metaprompt fails, returns extension-based fallback."""
        mock_llm = MagicMock()
        call_count = [0]

        def call_nano(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # File selector succeeds
                resp = MagicMock()
                resp.content = '["pyproject.toml"]'
                resp.prompt_tokens = 50
                resp.completion_tokens = 10
                return resp
            else:
                # Metaprompt fails
                raise Exception("LLM error")

        mock_llm.call_nano.side_effect = call_nano

        bp, pt, ct = generate_blueprint(FIXTURES, "catalog", mock_llm)
        assert bp["language"] == "python"
        assert bp["base_image"] is None
        assert "failed" in bp["notes"].lower()

    def test_unparseable_json_returns_fallback(self):
        """If metaprompt returns garbage, falls back."""
        llm = self._make_llm(
            '["pyproject.toml"]',
            "This is not JSON at all!",
        )

        bp, _, _ = generate_blueprint(FIXTURES, "catalog", llm)
        assert bp["language"] == "python"
        assert bp["base_image"] is None
