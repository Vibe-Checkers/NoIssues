"""Tests for Phase F: Agent Loop (mocked LLM, no real Docker)."""

import json
import os
from unittest.mock import MagicMock

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from db.models import StepRecord, IterationRecord, RunRecord
from db.writer import DBWriter
from agent.react_loop import (
    extract_lessons,
    _format_step_history,
    _fallback_lessons,
    _build_prompt,
)


# ═══════════════════════════════════════════════════════
# Lesson Extraction
# ═══════════════════════════════════════════════════════

class TestLessonExtraction:
    def _make_steps(self) -> list[StepRecord]:
        return [
            StepRecord(step_number=1, thought="Check structure", tool_name="ListDirectory",
                       tool_input={"path": "."}, tool_output="README.md\nsrc/\n"),
            StepRecord(step_number=2, thought="Read config", tool_name="ReadFile",
                       tool_input={"path": "package.json"},
                       tool_output='{"name": "app", "scripts": {"build": "tsc"}}'),
            StepRecord(step_number=3, thought="Write Dockerfile", tool_name="WriteFile",
                       tool_input={"path": "Dockerfile", "content": "FROM node:22\n..."},
                       tool_output="Written 100 bytes to Dockerfile"),
            StepRecord(step_number=4, thought="Verify", tool_name="VerifyBuild",
                       tool_input={},
                       tool_output="VerifyBuild status: build_failed\nBuild error: tsc not found"),
        ]

    def test_format_step_history(self):
        steps = self._make_steps()
        history = _format_step_history(steps)
        assert "Step 1 [ListDirectory]" in history
        assert "Step 4 [VerifyBuild]" in history
        assert "tsc not found" in history

    def test_extract_lessons_with_llm(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "1. tsc was not installed.\n2. Add RUN npm ci before build."
        mock_response.prompt_tokens = 300
        mock_response.completion_tokens = 80
        mock_llm.call_chat.return_value = mock_response

        lessons, pt, ct = extract_lessons(self._make_steps(), mock_llm)
        assert "tsc" in lessons
        assert pt == 300
        assert ct == 80
        mock_llm.call_chat.assert_called_once()

    def test_extract_lessons_fallback_on_error(self):
        mock_llm = MagicMock()
        mock_llm.call_chat.side_effect = Exception("API down")

        lessons, pt, ct = extract_lessons(self._make_steps(), mock_llm)
        assert "VerifyBuild error" in lessons
        assert "tsc not found" in lessons
        assert pt == 0

    def test_fallback_no_verify(self):
        steps = [
            StepRecord(step_number=1, tool_name="ListDirectory",
                       tool_input={}, tool_output="files"),
        ]
        lessons = _fallback_lessons(steps)
        assert "without calling VerifyBuild" in lessons

    def test_fallback_with_verify(self):
        steps = [
            StepRecord(step_number=1, tool_name="VerifyBuild",
                       tool_input={}, tool_output="build_failed: missing package"),
        ]
        lessons = _fallback_lessons(steps)
        assert "missing package" in lessons


# ═══════════════════════════════════════════════════════
# Prompt Building
# ═══════════════════════════════════════════════════════

class TestPromptBuilding:
    def test_build_prompt_no_lessons(self):
        blueprint = {"language": "python", "base_image": "python:3.12-slim"}
        prompt = _build_prompt(blueprint, None)
        assert "BuildAgent" in prompt
        assert "python:3.12-slim" in prompt
        assert "LESSONS" not in prompt

    def test_build_prompt_with_lessons(self):
        blueprint = {"language": "go", "base_image": "golang:1.22"}
        lessons = "1. Use CGO_ENABLED=0\n2. Build with go build -o /app"
        prompt = _build_prompt(blueprint, lessons)
        assert "LESSONS FROM PREVIOUS ATTEMPTS" in prompt
        assert "CGO_ENABLED=0" in prompt
        assert "golang:1.22" in prompt

    def test_build_prompt_missing_base_image(self):
        blueprint = {"language": "rust"}
        prompt = _build_prompt(blueprint, None)
        assert "auto-detect" in prompt


# ═══════════════════════════════════════════════════════
# DB Integration
# ═══════════════════════════════════════════════════════

class TestDBIntegration:
    """Verify step records can be written during iteration."""

    @pytest.fixture
    def db(self):
        writer = DBWriter("sqlite:///:memory:")
        yield writer
        writer.close()

    def test_write_steps_from_iteration(self, db):
        from db.models import BatchRun

        batch = BatchRun(worker_count=1, repo_count=1)
        db.write_batch_start(batch)

        run = RunRecord(batch_id=batch.id, repo_url="u", repo_slug="s")
        db.write_run_start(run)

        it = IterationRecord(run_id=run.id, iteration_number=1, status="running")
        db.write_iteration_start(it)

        # Simulate writing steps like the agent loop would
        for i in range(3):
            step = StepRecord(
                step_number=i + 1,
                thought=f"Step {i + 1} thought",
                tool_name="ReadFile",
                tool_input={"path": f"file_{i}.txt"},
                tool_output_raw=f"content_{i}",
                tool_output=f"content_{i}",
            )
            it.steps.append(step)
            db.write_step(it.id, step)

        it.status = "failure"
        db.write_iteration_finish(it)

        rows = db._query("SELECT COUNT(*) FROM step WHERE iteration_id=?", (it.id,))
        assert rows[0][0] == 3

        rows = db._query("SELECT status FROM iteration WHERE id=?", (it.id,))
        assert rows[0][0] == "failure"
