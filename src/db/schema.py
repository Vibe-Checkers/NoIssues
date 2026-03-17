"""Database schema creation for BuildAgent v2.0.

Runs all CREATE TABLE / CREATE INDEX statements.
Compatible with both SQLite and PostgreSQL.
"""

from __future__ import annotations


TABLES_SQL = [
    # ── batch_run ──
    """
    CREATE TABLE IF NOT EXISTS batch_run (
        id              TEXT PRIMARY KEY,
        started_at      TIMESTAMP NOT NULL,
        finished_at     TIMESTAMP,
        worker_count    INTEGER NOT NULL,
        repo_count      INTEGER NOT NULL,
        success_count   INTEGER DEFAULT 0,
        failure_count   INTEGER DEFAULT 0,
        total_prompt_tokens     INTEGER DEFAULT 0,
        total_completion_tokens INTEGER DEFAULT 0,
        config_json     TEXT,
        tag             TEXT
    )
    """,

    # ── run ──
    """
    CREATE TABLE IF NOT EXISTS run (
        id              TEXT PRIMARY KEY,
        batch_id        TEXT REFERENCES batch_run(id),
        repo_url        TEXT NOT NULL,
        repo_slug       TEXT NOT NULL,
        status          TEXT NOT NULL,
        started_at      TIMESTAMP NOT NULL,
        finished_at     TIMESTAMP,
        duration_ms     INTEGER,
        iteration_count INTEGER DEFAULT 0,
        detected_language   TEXT,
        repo_type           TEXT,
        context_blueprint   TEXT,
        blueprint_tokens_prompt     INTEGER DEFAULT 0,
        blueprint_tokens_completion INTEGER DEFAULT 0,
        blueprint_duration_ms       INTEGER,
        final_dockerfile    TEXT,
        verify_score        INTEGER,
        smoke_test_passed   BOOLEAN,
        total_prompt_tokens     INTEGER DEFAULT 0,
        total_completion_tokens INTEGER DEFAULT 0,
        total_steps             INTEGER DEFAULT 0,
        error_message       TEXT,
        worker_id           INTEGER
    )
    """,

    # ── iteration ──
    """
    CREATE TABLE IF NOT EXISTS iteration (
        id              TEXT PRIMARY KEY,
        run_id          TEXT NOT NULL REFERENCES run(id),
        iteration_number INTEGER NOT NULL,
        status          TEXT NOT NULL,
        started_at      TIMESTAMP NOT NULL,
        finished_at     TIMESTAMP,
        duration_ms     INTEGER,
        step_count      INTEGER DEFAULT 0,
        injected_lessons TEXT,
        prompt_tokens       INTEGER DEFAULT 0,
        completion_tokens   INTEGER DEFAULT 0,
        lesson_extraction_tokens_prompt     INTEGER DEFAULT 0,
        lesson_extraction_tokens_completion INTEGER DEFAULT 0,
        dockerfile_generated BOOLEAN DEFAULT FALSE,
        verify_attempted     BOOLEAN DEFAULT FALSE,
        verify_result        TEXT,
        error_message        TEXT
    )
    """,

    # ── step ──
    """
    CREATE TABLE IF NOT EXISTS step (
        id              TEXT PRIMARY KEY,
        iteration_id    TEXT NOT NULL REFERENCES iteration(id),
        step_number     INTEGER NOT NULL,
        started_at      TIMESTAMP NOT NULL,
        finished_at     TIMESTAMP,
        duration_ms     INTEGER,
        thought         TEXT NOT NULL,
        tool_name       TEXT NOT NULL,
        tool_input      TEXT,
        tool_output_raw TEXT,
        tool_output     TEXT,
        was_summarized  BOOLEAN DEFAULT FALSE,
        prompt_tokens       INTEGER DEFAULT 0,
        completion_tokens   INTEGER DEFAULT 0,
        summary_prompt_tokens       INTEGER DEFAULT 0,
        summary_completion_tokens   INTEGER DEFAULT 0
    )
    """,

    # ── verify_build_detail ──
    """
    CREATE TABLE IF NOT EXISTS verify_build_detail (
        id              TEXT PRIMARY KEY,
        step_id         TEXT NOT NULL REFERENCES step(id),
        review_approved     BOOLEAN,
        review_score        INTEGER,
        review_concerns     TEXT,
        smoke_test_commands TEXT,
        review_prompt_tokens     INTEGER DEFAULT 0,
        review_completion_tokens INTEGER DEFAULT 0,
        review_duration_ms       INTEGER,
        build_attempted     BOOLEAN DEFAULT FALSE,
        build_success       BOOLEAN,
        build_duration_ms   INTEGER,
        build_error_raw     TEXT,
        build_error         TEXT,
        build_error_summarized BOOLEAN DEFAULT FALSE,
        build_error_summary_tokens_prompt     INTEGER DEFAULT 0,
        build_error_summary_tokens_completion INTEGER DEFAULT 0,
        build_retried       BOOLEAN DEFAULT FALSE,
        build_retry_reason  TEXT,
        smoke_attempted     BOOLEAN DEFAULT FALSE,
        smoke_passed        BOOLEAN,
        smoke_results       TEXT,
        smoke_duration_ms   INTEGER
    )
    """,

    # ── run_artifact ──
    """
    CREATE TABLE IF NOT EXISTS run_artifact (
        id          TEXT PRIMARY KEY,
        run_id      TEXT NOT NULL REFERENCES run(id),
        artifact_type TEXT NOT NULL,
        file_name   TEXT NOT NULL,
        content     TEXT,
        file_path   TEXT,
        created_at  TIMESTAMP NOT NULL
    )
    """,

    # ── image_catalog ──
    """
    CREATE TABLE IF NOT EXISTS image_catalog (
        id          TEXT PRIMARY KEY,
        fetched_at  TIMESTAMP NOT NULL,
        image_count INTEGER NOT NULL,
        content     TEXT NOT NULL
    )
    """,
]

INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_run_batch ON run(batch_id)",
    "CREATE INDEX IF NOT EXISTS idx_run_status ON run(status)",
    "CREATE INDEX IF NOT EXISTS idx_run_repo ON run(repo_slug)",
    "CREATE INDEX IF NOT EXISTS idx_iteration_run ON iteration(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_step_iteration ON step(iteration_id)",
    "CREATE INDEX IF NOT EXISTS idx_verify_step ON verify_build_detail(step_id)",
    "CREATE INDEX IF NOT EXISTS idx_artifact_run ON run_artifact(run_id)",
]


def create_tables(conn) -> None:
    """Execute all CREATE TABLE and CREATE INDEX statements on the given connection.

    Args:
        conn: A DB-API 2.0 connection (sqlite3.Connection or psycopg2 connection).
    """
    cursor = conn.cursor()
    for sql in TABLES_SQL:
        cursor.execute(sql)
    for sql in INDEXES_SQL:
        cursor.execute(sql)
    conn.commit()
