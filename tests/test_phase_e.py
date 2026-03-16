"""Tests for Phase E: Tools + VerifyBuild."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.tools import (
    ReadFileTool, ListDirectoryTool, FindFilesTool, GrepFilesTool,
    WriteFileTool, PathTraversalError, resolve_path, create_tools,
)
from agent.verify_build import VerifyBuildTool
from db.models import VerifyBuildResult


# ═══════════════════════════════════════════════════════
# Sandbox
# ═══════════════════════════════════════════════════════

class TestSandbox:
    def test_resolve_valid_path(self, tmp_path):
        (tmp_path / "file.txt").write_text("ok")
        resolved = resolve_path(tmp_path, "file.txt")
        assert resolved == (tmp_path / "file.txt").resolve()

    def test_resolve_traversal_blocked(self, tmp_path):
        with pytest.raises(PathTraversalError):
            resolve_path(tmp_path, "../../../etc/passwd")

    def test_resolve_symlink_traversal(self, tmp_path):
        target = tmp_path.parent / "outside.txt"
        target.write_text("secret")
        link = tmp_path / "evil_link"
        link.symlink_to(target)
        with pytest.raises(PathTraversalError):
            resolve_path(tmp_path, "evil_link")
        target.unlink()


# ═══════════════════════════════════════════════════════
# ReadFile
# ═══════════════════════════════════════════════════════

class TestReadFile:
    def test_read_existing(self, tmp_path):
        (tmp_path / "hello.txt").write_text("hello world")
        tool = ReadFileTool(tmp_path)
        result = tool.execute("hello.txt")
        assert result == "hello world"

    def test_read_nonexistent(self, tmp_path):
        tool = ReadFileTool(tmp_path)
        result = tool.execute("nope.txt")
        assert "Error" in result

    def test_read_oversized(self, tmp_path):
        big = tmp_path / "big.bin"
        big.write_bytes(b"x" * (513 * 1024))
        tool = ReadFileTool(tmp_path)
        result = tool.execute("big.bin")
        assert "512KB" in result

    def test_read_traversal(self, tmp_path):
        tool = ReadFileTool(tmp_path)
        result = tool.execute("../../etc/passwd")
        assert "Error" in result


# ═══════════════════════════════════════════════════════
# ListDirectory
# ═══════════════════════════════════════════════════════

class TestListDirectory:
    def test_list_root(self, tmp_path):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "subdir").mkdir()
        tool = ListDirectoryTool(tmp_path)
        result = tool.execute(".")
        assert "a.txt" in result
        assert "subdir/" in result

    def test_list_empty(self, tmp_path):
        tool = ListDirectoryTool(tmp_path)
        result = tool.execute(".")
        assert "empty" in result.lower()


# ═══════════════════════════════════════════════════════
# FindFiles
# ═══════════════════════════════════════════════════════

class TestFindFiles:
    def test_find_python(self, tmp_path):
        (tmp_path / "main.py").write_text("pass")
        (tmp_path / "lib").mkdir()
        (tmp_path / "lib" / "util.py").write_text("pass")
        (tmp_path / "readme.md").write_text("hi")
        tool = FindFilesTool(tmp_path)
        result = tool.execute("**/*.py")
        assert "main.py" in result
        assert "util.py" in result
        assert "readme.md" not in result

    def test_find_no_match(self, tmp_path):
        tool = FindFilesTool(tmp_path)
        result = tool.execute("**/*.rs")
        assert "No files" in result


# ═══════════════════════════════════════════════════════
# GrepFiles
# ═══════════════════════════════════════════════════════

class TestGrepFiles:
    def test_grep_match(self, tmp_path):
        (tmp_path / "app.py").write_text("import flask\napp = flask.Flask(__name__)")
        tool = GrepFilesTool(tmp_path)
        result = tool.execute("flask")
        assert "app.py:1:" in result
        assert "flask" in result

    def test_grep_no_match(self, tmp_path):
        (tmp_path / "app.py").write_text("import os")
        tool = GrepFilesTool(tmp_path)
        result = tool.execute("flask")
        assert "No matches" in result

    def test_grep_invalid_regex(self, tmp_path):
        tool = GrepFilesTool(tmp_path)
        result = tool.execute("[invalid")
        assert "Error" in result


# ═══════════════════════════════════════════════════════
# WriteFile
# ═══════════════════════════════════════════════════════

class TestWriteFile:
    def test_write_normal(self, tmp_path):
        tool = WriteFileTool(tmp_path)
        result = tool.execute("output.txt", "hello")
        assert "Written" in result
        assert (tmp_path / "output.txt").read_text() == "hello"

    def test_write_creates_subdirs(self, tmp_path):
        tool = WriteFileTool(tmp_path)
        result = tool.execute("sub/dir/file.txt", "deep")
        assert "Written" in result
        assert (tmp_path / "sub" / "dir" / "file.txt").read_text() == "deep"

    def test_write_traversal_blocked(self, tmp_path):
        tool = WriteFileTool(tmp_path)
        result = tool.execute("../../evil.txt", "hack")
        assert "Error" in result

    def test_write_dockerfile_validates_from(self, tmp_path):
        """FROM validation checks Docker Hub (mocked to reject)."""
        tool = WriteFileTool(tmp_path)

        # Mock _image_exists to reject
        original = WriteFileTool._image_exists
        WriteFileTool._image_exists = staticmethod(lambda ref: False)
        try:
            result = tool.execute("Dockerfile", "FROM nonexistent:tag\nRUN echo ok")
            assert "Error" in result
            assert "not found" in result
        finally:
            WriteFileTool._image_exists = original

    def test_write_dockerfile_allows_valid_from(self, tmp_path):
        """FROM validation passes for valid images."""
        tool = WriteFileTool(tmp_path)
        WriteFileTool._image_exists = staticmethod(lambda ref: True)
        try:
            result = tool.execute("Dockerfile", "FROM python:3.12\nRUN echo ok")
            assert "Written" in result
        finally:
            # Restore (doesn't matter much in tests)
            pass

    def test_write_dockerfile_allows_scratch(self, tmp_path):
        tool = WriteFileTool(tmp_path)
        WriteFileTool._image_exists = staticmethod(lambda ref: False)
        try:
            result = tool.execute("Dockerfile", "FROM scratch\nCOPY app /app")
            assert "Written" in result
        finally:
            pass


# ═══════════════════════════════════════════════════════
# VerifyBuild
# ═══════════════════════════════════════════════════════

class TestVerifyBuild:
    def _make_tool(self, tmp_path, llm_review: dict, build_ok=True,
                   smoke_results=None):
        """Create VerifyBuildTool with mocked dependencies."""
        # Write a Dockerfile
        (tmp_path / "Dockerfile").write_text("FROM alpine:3.19\nRUN echo ok\n")

        # Mock LLM
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = json.dumps(llm_review)
        mock_response.prompt_tokens = 50
        mock_response.completion_tokens = 20
        mock_llm.call_nano.return_value = mock_response

        # Mock docker_ops
        mock_ops = MagicMock()
        mock_ops.build.return_value = (build_ok, "" if build_ok else "build error", 1000)
        if smoke_results is None:
            mock_ops.run_container.return_value = (0, "ok", False)
        else:
            mock_ops.run_container.side_effect = smoke_results

        blueprint = {"repo_type": "library", "language": "python"}

        tool = VerifyBuildTool(
            repo_root=tmp_path,
            image_name="test-image",
            docker_ops=mock_ops,
            llm=mock_llm,
            blueprint=blueprint,
        )
        return tool

    def test_accepted(self, tmp_path):
        review = {"approved": True, "concerns": [], "smoke_test_commands": ["echo ok"]}
        tool = self._make_tool(tmp_path, review)
        result = tool.execute()
        assert "accepted" in result

    def test_rejected(self, tmp_path):
        review = {"approved": False, "concerns": ["empty Dockerfile"],
                  "smoke_test_commands": ["echo test"]}
        tool = self._make_tool(tmp_path, review)
        result = tool.execute()
        assert "rejected" in result

    def test_build_failed(self, tmp_path):
        review = {"approved": True, "concerns": [], "smoke_test_commands": ["echo ok"]}
        tool = self._make_tool(tmp_path, review, build_ok=False)
        result = tool.execute()
        assert "build_failed" in result

    def test_smoke_failed(self, tmp_path):
        review = {"approved": True, "concerns": [], "smoke_test_commands": ["test -f /app"]}
        smoke = [(1, "not found", False)]
        tool = self._make_tool(tmp_path, review, smoke_results=smoke)
        result = tool.execute()
        assert "smoke_failed" in result

    def test_no_dockerfile(self, tmp_path):
        mock_llm = MagicMock()
        mock_ops = MagicMock()
        tool = VerifyBuildTool(tmp_path, "img", mock_ops, mock_llm)
        result = tool.execute()
        assert "rejected" in result
        assert "No Dockerfile" in result

    def test_llm_review_failure_fallback(self, tmp_path):
        """If LLM review fails, approve by default and build."""
        (tmp_path / "Dockerfile").write_text("FROM alpine:3.19\nRUN echo ok\n")

        mock_llm = MagicMock()
        mock_llm.call_nano.side_effect = Exception("API error")

        mock_ops = MagicMock()
        mock_ops.build.return_value = (True, "", 500)
        mock_ops.run_container.return_value = (0, "ok", False)

        tool = VerifyBuildTool(tmp_path, "img", mock_ops, mock_llm)
        result = tool.execute()
        # Should still attempt build (fallback approves)
        assert "accepted" in result or "smoke" in result.lower()

    def test_get_last_result_returns_dataclass(self, tmp_path):
        review = {"approved": True, "concerns": [], "smoke_test_commands": ["echo ok"]}
        tool = self._make_tool(tmp_path, review)
        result = tool.get_last_result()
        assert isinstance(result, VerifyBuildResult)
        assert result.status == "accepted"
        assert result.review_approved is True


# ═══════════════════════════════════════════════════════
# Tool Registry
# ═══════════════════════════════════════════════════════

class TestToolRegistry:
    def test_create_tools(self, tmp_path):
        tools = create_tools(tmp_path)
        names = [t.name for t in tools]
        assert "ReadFile" in names
        assert "WriteFile" in names
        assert "ListDirectory" in names
        assert "FindFiles" in names
        assert "GrepFiles" in names
        assert "DockerImageSearch" in names
        assert "SearchWeb" in names
        assert len(tools) == 7  # VerifyBuild added separately
