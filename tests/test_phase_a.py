"""Tests for Phase A: Foundation (no LLM calls, no Docker)."""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from db.models import StepRecord, IterationRecord, RunRecord, VerifyBuildResult, BatchRun
from db.schema import create_tables, TABLES_SQL, INDEXES_SQL
from db.writer import DBWriter
from agent.summarizer import summarize_output, SUMMARY_THRESHOLD


# ═══════════════════════════════════════════════════════
# A1: Models
# ═══════════════════════════════════════════════════════

class TestModels:
    def test_step_record_defaults(self):
        s = StepRecord()
        assert s.id  # UUID generated
        assert isinstance(s.started_at, datetime)
        assert s.step_number == 0
        assert s.tool_input == {}
        assert s.was_summarized is False

    def test_step_record_to_dict(self):
        s = StepRecord(tool_input={"path": "Dockerfile"})
        d = s.to_dict()
        assert d["tool_input"] == '{"path": "Dockerfile"}'
        assert "id" in d
        assert "thought" in d

    def test_verify_build_result_to_dict(self):
        v = VerifyBuildResult(
            status="accepted",
            review_approved=True,
            review_concerns=["minor issue"],
            smoke_test_commands=["echo ok"],
            smoke_results=[{"command": "echo ok", "exit_code": 0, "output": "ok", "timed_out": False}],
        )
        d = v.to_dict()
        assert d["review_concerns"] == '["minor issue"]'
        assert d["smoke_test_commands"] == '["echo ok"]'
        assert json.loads(d["smoke_results"])[0]["exit_code"] == 0

    def test_iteration_record_step_count(self):
        it = IterationRecord()
        assert it.step_count == 0
        it.steps.append(StepRecord())
        it.steps.append(StepRecord())
        assert it.step_count == 2

    def test_iteration_record_to_dict(self):
        it = IterationRecord(lesson_extraction_tokens=(100, 50))
        d = it.to_dict()
        assert "steps" not in d
        assert d["step_count"] == 0
        assert d["lesson_extraction_tokens_prompt"] == 100
        assert d["lesson_extraction_tokens_completion"] == 50
        assert "lesson_extraction_tokens" not in d

    def test_run_record_to_dict(self):
        r = RunRecord(repo_url="https://github.com/foo/bar", repo_slug="foo__bar")
        d = r.to_dict()
        assert d["repo_slug"] == "foo__bar"
        assert "iterations" not in d

    def test_batch_run_to_dict(self):
        b = BatchRun(worker_count=4, repo_count=100)
        d = b.to_dict()
        assert d["worker_count"] == 4
        assert d["repo_count"] == 100
        assert d["success_count"] == 0

    def test_all_models_have_unique_ids(self):
        ids = {StepRecord().id, IterationRecord().id, RunRecord().id, BatchRun().id}
        assert len(ids) == 4  # all unique


# ═══════════════════════════════════════════════════════
# A2: Schema
# ═══════════════════════════════════════════════════════

class TestSchema:
    def test_create_tables_in_memory(self):
        conn = sqlite3.connect(":memory:")
        create_tables(conn)

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = {row[0] for row in cursor.fetchall()}
        expected = {"batch_run", "run", "iteration", "step", "verify_build_detail", "run_artifact"}
        assert expected == tables

    def test_tables_have_correct_columns(self):
        conn = sqlite3.connect(":memory:")
        create_tables(conn)

        expected_columns = {
            "batch_run": {"id", "started_at", "finished_at", "worker_count", "repo_count",
                          "success_count", "failure_count", "total_prompt_tokens",
                          "total_completion_tokens", "config_json"},
            "step": {"id", "iteration_id", "step_number", "started_at", "finished_at",
                     "duration_ms", "thought", "tool_name", "tool_input",
                     "tool_output_raw", "tool_output", "was_summarized",
                     "prompt_tokens", "completion_tokens",
                     "summary_prompt_tokens", "summary_completion_tokens"},
        }

        for table, cols in expected_columns.items():
            cursor = conn.execute(f"PRAGMA table_info({table})")
            actual_cols = {row[1] for row in cursor.fetchall()}
            assert cols == actual_cols, f"Column mismatch in {table}: missing={cols - actual_cols}, extra={actual_cols - cols}"

    def test_indexes_created(self):
        conn = sqlite3.connect(":memory:")
        create_tables(conn)

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'")
        indexes = {row[0] for row in cursor.fetchall()}
        expected = {"idx_run_batch", "idx_run_status", "idx_run_repo",
                    "idx_iteration_run", "idx_step_iteration",
                    "idx_verify_step", "idx_artifact_run"}
        assert expected == indexes

    def test_idempotent_create(self):
        """create_tables can be called multiple times without error."""
        conn = sqlite3.connect(":memory:")
        create_tables(conn)
        create_tables(conn)  # should not raise


# ═══════════════════════════════════════════════════════
# A3: Writer
# ═══════════════════════════════════════════════════════

class TestWriter:
    @pytest.fixture
    def db(self):
        writer = DBWriter("sqlite:///:memory:")
        yield writer
        writer.close()

    def test_full_chain(self, db):
        """Write batch -> run -> iteration -> step, query back."""
        # Batch
        batch = BatchRun(worker_count=2, repo_count=10)
        db.write_batch_start(batch)

        # Run
        run = RunRecord(batch_id=batch.id, repo_url="https://github.com/foo/bar",
                        repo_slug="foo__bar", worker_id=1)
        db.write_run_start(run)

        # Update blueprint
        run.detected_language = "python"
        run.repo_type = "library"
        run.context_blueprint = '{"language": "python"}'
        run.blueprint_tokens_prompt = 500
        run.blueprint_tokens_completion = 200
        run.blueprint_duration_ms = 1200
        db.update_run_blueprint(run)

        # Iteration
        it = IterationRecord(run_id=run.id, iteration_number=1, status="pending")
        db.write_iteration_start(it)

        # Step
        step = StepRecord(
            step_number=1,
            thought="I need to read the Dockerfile",
            tool_name="ReadFile",
            tool_input={"path": "Dockerfile"},
            tool_output_raw="FROM python:3.12\nRUN pip install flask",
            tool_output="FROM python:3.12\nRUN pip install flask",
            prompt_tokens=100,
            completion_tokens=50,
        )
        db.write_step(it.id, step)

        # Finish iteration
        it.status = "success"
        it.finished_at = datetime.now(timezone.utc)
        it.duration_ms = 5000
        it.steps.append(step)
        it.verify_result = "accepted"
        db.write_iteration_finish(it)

        # Finish run
        run.status = "success"
        run.finished_at = datetime.now(timezone.utc)
        run.duration_ms = 8000
        run.iteration_count = 1
        run.total_prompt_tokens = 600
        run.total_completion_tokens = 250
        run.total_steps = 1
        db.write_run_finish(run)

        # Finish batch
        batch.finished_at = datetime.now(timezone.utc)
        batch.success_count = 1
        db.write_batch_finish(batch)

        # Query back and verify
        rows = db._query("SELECT status, repo_slug FROM run WHERE id=?", (run.id,))
        assert rows[0] == ("success", "foo__bar")

        rows = db._query("SELECT thought, tool_name FROM step WHERE iteration_id=?", (it.id,))
        assert rows[0] == ("I need to read the Dockerfile", "ReadFile")

        rows = db._query("SELECT detected_language FROM run WHERE id=?", (run.id,))
        assert rows[0][0] == "python"

        rows = db._query("SELECT success_count FROM batch_run WHERE id=?", (batch.id,))
        assert rows[0][0] == 1

    def test_verify_detail(self, db):
        """Write verify_build_detail row."""
        batch = BatchRun(worker_count=1, repo_count=1)
        db.write_batch_start(batch)

        run = RunRecord(batch_id=batch.id, repo_url="u", repo_slug="s")
        db.write_run_start(run)

        it = IterationRecord(run_id=run.id, iteration_number=1, status="pending")
        db.write_iteration_start(it)

        step = StepRecord(step_number=1, thought="verify", tool_name="VerifyBuild",
                          tool_input={}, tool_output="accepted")
        db.write_step(it.id, step)

        detail = VerifyBuildResult(
            status="accepted",
            review_approved=True,
            review_concerns=["none"],
            smoke_test_commands=["echo ok"],
            build_success=True,
            build_duration_ms=3000,
            smoke_results=[{"command": "echo ok", "exit_code": 0, "output": "ok", "timed_out": False}],
            review_tokens=(80, 40),
        )
        db.write_verify_detail(step.id, detail)

        rows = db._query("SELECT review_approved, build_success FROM verify_build_detail WHERE step_id=?", (step.id,))
        assert rows[0][0] == 1  # True in SQLite
        assert rows[0][1] == 1

    def test_artifact(self, db):
        batch = BatchRun(worker_count=1, repo_count=1)
        db.write_batch_start(batch)
        run = RunRecord(batch_id=batch.id, repo_url="u", repo_slug="s")
        db.write_run_start(run)

        db.write_artifact(run.id, "dockerfile", "Dockerfile", content="FROM alpine\nRUN echo ok")

        rows = db._query("SELECT artifact_type, content FROM run_artifact WHERE run_id=?", (run.id,))
        assert rows[0][0] == "dockerfile"
        assert "FROM alpine" in rows[0][1]

    def test_successful_slugs(self, db):
        batch = BatchRun(worker_count=1, repo_count=2)
        db.write_batch_start(batch)

        for slug, status in [("foo__bar", "success"), ("baz__qux", "failure")]:
            run = RunRecord(batch_id=batch.id, repo_url=f"u/{slug}", repo_slug=slug, status=status)
            db.write_run_start(run)
            run.status = status
            run.finished_at = datetime.now(timezone.utc)
            db.write_run_finish(run)

        slugs = db.get_successful_slugs(batch.id)
        assert slugs == {"foo__bar"}

    def test_concurrent_writes(self, db):
        """4 threads writing steps concurrently."""
        batch = BatchRun(worker_count=4, repo_count=4)
        db.write_batch_start(batch)

        run = RunRecord(batch_id=batch.id, repo_url="u", repo_slug="s")
        db.write_run_start(run)

        it = IterationRecord(run_id=run.id, iteration_number=1, status="pending")
        db.write_iteration_start(it)

        errors = []

        def writer(thread_id):
            try:
                for i in range(5):
                    step = StepRecord(
                        step_number=thread_id * 5 + i,
                        thought=f"thread {thread_id} step {i}",
                        tool_name="ReadFile",
                        tool_input={"path": f"file_{thread_id}_{i}.py"},
                        tool_output_raw="content",
                        tool_output="content",
                    )
                    db.write_step(it.id, step)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent writes raised errors: {errors}"

        rows = db._query("SELECT COUNT(*) FROM step WHERE iteration_id=?", (it.id,))
        assert rows[0][0] == 20  # 4 threads × 5 steps


# ═══════════════════════════════════════════════════════
# A4: Summarizer
# ═══════════════════════════════════════════════════════

class TestSummarizer:
    def test_short_content_passthrough(self):
        text = "short output"
        result, pt, ct = summarize_output(text)
        assert result == text
        assert pt == 0
        assert ct == 0

    def test_exact_threshold_passthrough(self):
        text = "x" * SUMMARY_THRESHOLD
        result, pt, ct = summarize_output(text)
        assert result == text
        assert pt == 0

    def test_long_content_truncated_no_llm(self):
        text = "x" * 3000
        result, pt, ct = summarize_output(text)
        assert len(result) < 3000
        assert result.endswith("[truncated]")
        assert pt == 0

    def test_long_content_summarized_with_llm(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "summarized output"
        mock_response.prompt_tokens = 500
        mock_response.completion_tokens = 100
        mock_llm.call_nano.return_value = mock_response

        text = "x" * 3000
        result, pt, ct = summarize_output(text, llm=mock_llm)
        assert result == "summarized output"
        assert pt == 500
        assert ct == 100
        mock_llm.call_nano.assert_called_once()

    def test_llm_failure_falls_back_to_truncation(self):
        mock_llm = MagicMock()
        mock_llm.call_nano.side_effect = Exception("API error")

        text = "x" * 3000
        result, pt, ct = summarize_output(text, llm=mock_llm)
        assert result.endswith("[truncated]")
        assert pt == 0

    def test_build_error_context_type(self):
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "build error summary"
        mock_response.prompt_tokens = 400
        mock_response.completion_tokens = 80
        mock_llm.call_nano.return_value = mock_response

        text = "ERROR: " + "x" * 3000
        result, pt, ct = summarize_output(text, context_type="build_error", llm=mock_llm)
        assert result == "build error summary"

        # Verify the build_error prompt was used
        call_args = mock_llm.call_nano.call_args[0][0]
        assert "Docker build error" in call_args[1]["content"]
