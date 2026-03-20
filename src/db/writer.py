"""Thread-safe database writer for BuildAgent v2.0.

SQLite path: single connection + threading.Lock + WAL mode.
PostgreSQL path: ThreadedConnectionPool.
Backend selected by DATABASE_URL env var prefix.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

# Register adapters for timezone-aware datetimes in SQLite
sqlite3.register_adapter(datetime, lambda dt: dt.isoformat())
sqlite3.register_converter("TIMESTAMP", lambda b: datetime.fromisoformat(b.decode()))

from .schema import create_tables
from .models import BatchRun, RunRecord, IterationRecord, StepRecord, VerifyBuildResult

logger = logging.getLogger(__name__)


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class DBWriter:
    """Thread-safe writer for both SQLite and PostgreSQL backends."""

    def __init__(self, db_url: str | None = None):
        self.db_url = db_url or os.environ.get("DATABASE_URL", "sqlite:///results.db")

        if self.db_url.startswith("sqlite"):
            path = self.db_url.replace("sqlite:///", "")
            # :memory: for testing
            self.conn = sqlite3.connect(path, check_same_thread=False,
                                            detect_types=sqlite3.PARSE_DECLTYPES)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=5000")
            self._lock = threading.Lock()
            self._backend = "sqlite"
        else:
            try:
                from psycopg2.pool import ThreadedConnectionPool
            except ImportError:
                raise ImportError("psycopg2 is required for PostgreSQL. Install with: pip install psycopg2-binary")
            self._pool = ThreadedConnectionPool(2, 10, self.db_url)
            self._lock = None
            self._backend = "pg"

        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        if self._backend == "sqlite":
            create_tables(self.conn)
        else:
            conn = self._pool.getconn()
            try:
                create_tables(conn)
            finally:
                self._pool.putconn(conn)

    def _execute(self, sql: str, params: tuple = ()) -> None:
        if self._backend == "sqlite":
            with self._lock:
                self.conn.execute(sql, params)
                self.conn.commit()
        else:
            sql = sql.replace("?", "%s")
            conn = self._pool.getconn()
            try:
                conn.cursor().execute(sql, params)
                conn.commit()
            finally:
                self._pool.putconn(conn)

    def _query(self, sql: str, params: tuple = ()) -> list:
        if self._backend == "sqlite":
            with self._lock:
                cursor = self.conn.execute(sql, params)
                return cursor.fetchall()
        else:
            sql = sql.replace("?", "%s")
            conn = self._pool.getconn()
            try:
                cur = conn.cursor()
                cur.execute(sql, params)
                return cur.fetchall()
            finally:
                self._pool.putconn(conn)

    # ── Batch operations ──

    def write_batch_start(self, batch: BatchRun) -> None:
        self._execute(
            """INSERT INTO batch_run (id, started_at, finished_at, worker_count, repo_count,
                success_count, failure_count, total_prompt_tokens, total_completion_tokens, config_json, ablation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (batch.id, batch.started_at, batch.finished_at, batch.worker_count,
             batch.repo_count, batch.success_count, batch.failure_count,
             batch.total_prompt_tokens, batch.total_completion_tokens, batch.config_json, batch.ablation),
        )

    def write_batch_finish(self, batch: BatchRun) -> None:
        self._execute(
            """UPDATE batch_run SET finished_at=?, success_count=?, failure_count=?,
                total_prompt_tokens=?, total_completion_tokens=? WHERE id=?""",
            (batch.finished_at, batch.success_count, batch.failure_count,
             batch.total_prompt_tokens, batch.total_completion_tokens, batch.id),
        )

    def update_batch_progress(self, batch_id: str) -> None:
        """Recalculate and persist batch_run counters from completed runs."""
        rows = self._query(
            """SELECT
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END),
                SUM(CASE WHEN status IN ('failure', 'error') THEN 1 ELSE 0 END),
                COALESCE(SUM(total_prompt_tokens), 0),
                COALESCE(SUM(total_completion_tokens), 0)
            FROM run WHERE batch_id=?""",
            (batch_id,),
        )
        if not rows:
            return
        sc, fc, pt, ct = rows[0]
        self._execute(
            """UPDATE batch_run SET success_count=?, failure_count=?,
                total_prompt_tokens=?, total_completion_tokens=? WHERE id=?""",
            (sc or 0, fc or 0, pt, ct, batch_id),
        )

    def run_exists(self, batch_id: str, repo_slug: str) -> bool:
        rows = self._query(
            "SELECT 1 FROM run WHERE batch_id=? AND repo_slug=? LIMIT 1",
            (batch_id, repo_slug),
        )
        return len(rows) > 0

    def pre_insert_run(self, batch_id: str, repo_url: str, repo_slug: str) -> None:
        self._execute(
            """INSERT INTO run (id, batch_id, repo_url, repo_slug, status)
                VALUES (?, ?, ?, ?, ?)""",
            (_new_id(), batch_id, repo_url, repo_slug, "waiting"),
        )

    def get_run_id_for_repo(self, batch_id: str, repo_slug: str) -> str | None:
        rows = self._query(
            "SELECT id FROM run WHERE batch_id=? AND repo_slug=?",
            (batch_id, repo_slug),
        )
        return rows[0][0] if rows else None

    def write_run_start(self, run: RunRecord) -> None:
        """Insert a new run (legacy/fallback)."""
        self._execute(
            """INSERT INTO run (id, batch_id, repo_url, repo_slug, status, started_at,
                worker_id) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (run.id, run.batch_id, run.repo_url, run.repo_slug, run.status,
             run.started_at, run.worker_id),
        )

    def update_run_start(self, run: RunRecord) -> None:
        """Update an existing 'waiting' run to 'running' state."""
        self._execute(
            """UPDATE run SET status=?, started_at=?, worker_id=? WHERE id=?""",
            (run.status, run.started_at, run.worker_id, run.id),
        )

    def update_run_blueprint(self, run: RunRecord) -> None:
        self._execute(
            """UPDATE run SET detected_language=?, repo_type=?, context_blueprint=?,
                blueprint_tokens_prompt=?, blueprint_tokens_completion=?,
                blueprint_duration_ms=? WHERE id=?""",
            (run.detected_language, run.repo_type, run.context_blueprint,
             run.blueprint_tokens_prompt, run.blueprint_tokens_completion,
             run.blueprint_duration_ms, run.id),
        )

    def write_run_finish(self, run: RunRecord) -> None:
        self._execute(
            """UPDATE run SET status=?, finished_at=?, duration_ms=?, iteration_count=?,
                final_dockerfile=?, smoke_test_passed=?,
                total_prompt_tokens=?, total_completion_tokens=?, total_steps=?,
                error_message=? WHERE id=?""",
            (run.status, run.finished_at, run.duration_ms, run.iteration_count,
             run.final_dockerfile, run.smoke_test_passed,
             run.total_prompt_tokens, run.total_completion_tokens, run.total_steps,
             run.error_message, run.id),
        )

    # ── Iteration operations ──

    def write_iteration_start(self, iteration: IterationRecord) -> None:
        self._execute(
            """INSERT INTO iteration (id, run_id, iteration_number, status, started_at,
                injected_lessons) VALUES (?, ?, ?, ?, ?, ?)""",
            (iteration.id, iteration.run_id, iteration.iteration_number,
             iteration.status, iteration.started_at, iteration.injected_lessons),
        )

    def write_iteration_finish(self, iteration: IterationRecord) -> None:
        self._execute(
            """UPDATE iteration SET status=?, finished_at=?, duration_ms=?, step_count=?,
                prompt_tokens=?, completion_tokens=?,
                lesson_extraction_tokens_prompt=?, lesson_extraction_tokens_completion=?,
                dockerfile_generated=?, verify_attempted=?, verify_result=?,
                error_message=? WHERE id=?""",
            (iteration.status, iteration.finished_at, iteration.duration_ms,
             iteration.step_count, iteration.prompt_tokens, iteration.completion_tokens,
             iteration.lesson_extraction_tokens[0], iteration.lesson_extraction_tokens[1],
             iteration.dockerfile_generated, iteration.verify_attempted,
             iteration.verify_result, iteration.error_message, iteration.id),
        )

    # ── Step operations ──

    def write_step(self, iteration_id: str, step: StepRecord) -> None:
        self._execute(
            """INSERT INTO step (id, iteration_id, step_number, started_at, finished_at,
                duration_ms, thought, tool_name, tool_input, tool_output_raw, tool_output,
                was_summarized, prompt_tokens, completion_tokens,
                summary_prompt_tokens, summary_completion_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (step.id, iteration_id, step.step_number, step.started_at, step.finished_at,
             step.duration_ms, step.thought, step.tool_name,
             json.dumps(step.tool_input), step.tool_output_raw, step.tool_output,
             step.was_summarized, step.prompt_tokens, step.completion_tokens,
             step.summary_prompt_tokens, step.summary_completion_tokens),
        )

    # ── VerifyBuild detail ──

    def write_verify_detail(self, step_id: str, detail: VerifyBuildResult) -> None:
        detail_id = _new_id()
        self._execute(
            """INSERT INTO verify_build_detail (id, step_id,
                review_approved, review_concerns, smoke_test_commands,
                review_prompt_tokens, review_completion_tokens, review_duration_ms,
                build_attempted, build_success, build_duration_ms,
                build_error_raw, build_error, build_error_summarized,
                build_error_summary_tokens_prompt, build_error_summary_tokens_completion,
                smoke_attempted, smoke_passed, smoke_results, smoke_duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (detail_id, step_id,
             detail.review_approved,
             json.dumps(detail.review_concerns),
             json.dumps(detail.smoke_test_commands),
             detail.review_tokens[0], detail.review_tokens[1],
             detail.review_duration_ms,
             detail.build_success is not None,  # build_attempted
             detail.build_success, detail.build_duration_ms,
             detail.build_error_raw, detail.build_error,
             detail.build_error is not None and detail.build_error != detail.build_error_raw,
             detail.error_summary_tokens[0], detail.error_summary_tokens[1],
             detail.smoke_results is not None,  # smoke_attempted
             all(r.get("exit_code") == 0 for r in (detail.smoke_results or [])) if detail.smoke_results else None,
             json.dumps(detail.smoke_results) if detail.smoke_results else None,
             detail.smoke_duration_ms),
        )

    # ── Artifact operations ──

    def write_artifact(self, run_id: str, artifact_type: str, file_name: str,
                       content: str | None = None, file_path: str | None = None) -> None:
        self._execute(
            """INSERT INTO run_artifact (id, run_id, artifact_type, file_name, content,
                file_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (_new_id(), run_id, artifact_type, file_name, content, file_path, _now()),
        )

    # ── Image catalog cache ──

    def save_image_catalog(self, content: str, image_count: int) -> None:
        self._execute(
            "INSERT INTO image_catalog (id, fetched_at, image_count, content) VALUES (?, ?, ?, ?)",
            (_new_id(), _now(), image_count, content),
        )

    def load_image_catalog(self, max_age_hours: int = 24) -> str | None:
        """Return cached catalog content if fresh enough, else None."""
        rows = self._query(
            "SELECT content, fetched_at FROM image_catalog ORDER BY fetched_at DESC LIMIT 1",
        )
        if not rows:
            return None
        content, fetched_at = rows[0]
        if isinstance(fetched_at, str):
            fetched_at = datetime.fromisoformat(fetched_at)
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - fetched_at
        if age.total_seconds() < max_age_hours * 3600:
            return content
        return None

    # ── Crash recovery ──

    def get_successful_slugs(self, batch_id: str) -> set[str]:
        """Return repo slugs that already succeeded in this batch (for crash recovery)."""
        rows = self._query(
            "SELECT repo_slug FROM run WHERE batch_id=? AND status='success'",
            (batch_id,),
        )
        return {row[0] for row in rows}

    # ── Cleanup ──

    def close(self) -> None:
        if self._backend == "sqlite":
            self.conn.close()
        else:
            self._pool.closeall()
