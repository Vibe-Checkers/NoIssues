# Database Schema

## BuildAgent v2.0

SQLite for local/development, PostgreSQL for production. The schema uses standard SQL types compatible with both.

---

## Entity Relationship

```
batch_run 1──N run 1──N iteration 1──N step
                │                       │
                └── run_artifact         └── step has llm_call(s)
```

---

## Tables

### `batch_run`

Top-level grouping for a parallel batch execution.

```sql
CREATE TABLE batch_run (
    id              TEXT PRIMARY KEY,           -- UUID
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    worker_count    INTEGER NOT NULL,
    repo_count      INTEGER NOT NULL,           -- total repos in input
    success_count   INTEGER DEFAULT 0,
    failure_count   INTEGER DEFAULT 0,
    total_prompt_tokens     INTEGER DEFAULT 0,
    total_completion_tokens INTEGER DEFAULT 0,
    config_json     TEXT                        -- serialized run config (timeouts, model names, etc.)
);
```

### `run`

One row per repository processed.

```sql
CREATE TABLE run (
    id              TEXT PRIMARY KEY,           -- UUID
    batch_id        TEXT REFERENCES batch_run(id),
    repo_url        TEXT NOT NULL,
    repo_slug       TEXT NOT NULL,              -- owner__repo
    status          TEXT NOT NULL,              -- 'success', 'failure', 'error', 'skipped'
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    duration_ms     INTEGER,
    iteration_count INTEGER DEFAULT 0,
    -- context blueprint results
    detected_language   TEXT,
    repo_type           TEXT,                   -- cli_tool, library, web_service, etc.
    context_blueprint   TEXT,                   -- full blueprint text
    blueprint_tokens_prompt     INTEGER DEFAULT 0,
    blueprint_tokens_completion INTEGER DEFAULT 0,
    blueprint_duration_ms       INTEGER,
    -- final result
    final_dockerfile    TEXT,                   -- content of accepted Dockerfile
    verify_score        INTEGER,               -- 1-10 from VerifyBuild review
    smoke_test_passed   BOOLEAN,
    -- totals
    total_prompt_tokens     INTEGER DEFAULT 0,
    total_completion_tokens INTEGER DEFAULT 0,
    total_steps             INTEGER DEFAULT 0,
    error_message       TEXT,                   -- if status='error'
    worker_id           INTEGER                 -- which parallel worker handled this
);

CREATE INDEX idx_run_batch ON run(batch_id);
CREATE INDEX idx_run_status ON run(status);
CREATE INDEX idx_run_repo ON run(repo_slug);
```

### `iteration`

One row per iteration within a run (up to 3).

```sql
CREATE TABLE iteration (
    id              TEXT PRIMARY KEY,           -- UUID
    run_id          TEXT NOT NULL REFERENCES run(id),
    iteration_number INTEGER NOT NULL,          -- 1, 2, or 3
    status          TEXT NOT NULL,              -- 'success', 'failure', 'error'
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    duration_ms     INTEGER,
    step_count      INTEGER DEFAULT 0,
    -- lesson from previous iteration (NULL for iteration 1)
    injected_lessons TEXT,
    -- tokens consumed in this iteration
    prompt_tokens       INTEGER DEFAULT 0,
    completion_tokens   INTEGER DEFAULT 0,
    -- summarization overhead
    lesson_extraction_tokens_prompt     INTEGER DEFAULT 0,
    lesson_extraction_tokens_completion INTEGER DEFAULT 0,
    -- outcome
    dockerfile_generated BOOLEAN DEFAULT FALSE,
    verify_attempted     BOOLEAN DEFAULT FALSE,
    verify_result        TEXT,                  -- 'accepted', 'rejected', 'build_failed', 'smoke_failed', NULL
    error_message        TEXT
);

CREATE INDEX idx_iteration_run ON iteration(run_id);
```

### `step`

One row per agent step (LLM call + tool action + result).

```sql
CREATE TABLE step (
    id              TEXT PRIMARY KEY,           -- UUID
    iteration_id    TEXT NOT NULL REFERENCES iteration(id),
    step_number     INTEGER NOT NULL,           -- 1..15
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    duration_ms     INTEGER,
    -- agent reasoning
    thought         TEXT NOT NULL,              -- agent's reasoning text
    -- tool call
    tool_name       TEXT NOT NULL,              -- ReadFile, WriteFile, VerifyBuild, etc.
    tool_input      TEXT,                       -- JSON-serialized input parameters
    -- tool result
    tool_output_raw TEXT,                       -- full original output (before summarization)
    tool_output     TEXT,                       -- output fed to agent (summarized if >2000 chars)
    was_summarized  BOOLEAN DEFAULT FALSE,      -- TRUE if output was summarized
    -- LLM tokens for the agent call
    prompt_tokens       INTEGER DEFAULT 0,
    completion_tokens   INTEGER DEFAULT 0,
    -- summarization tokens (if applicable)
    summary_prompt_tokens       INTEGER DEFAULT 0,
    summary_completion_tokens   INTEGER DEFAULT 0
);

CREATE INDEX idx_step_iteration ON step(iteration_id);
```

### `verify_build_detail`

Extended detail for steps where `tool_name = 'VerifyBuild'`.

```sql
CREATE TABLE verify_build_detail (
    id              TEXT PRIMARY KEY,           -- UUID
    step_id         TEXT NOT NULL REFERENCES step(id),
    -- LLM review phase
    review_approved     BOOLEAN,
    review_score        INTEGER,               -- 1-10
    review_concerns     TEXT,                  -- JSON array of strings
    smoke_test_commands TEXT,                  -- JSON array of commands
    review_prompt_tokens     INTEGER DEFAULT 0,
    review_completion_tokens INTEGER DEFAULT 0,
    review_duration_ms       INTEGER,
    -- Docker build phase
    build_attempted     BOOLEAN DEFAULT FALSE,
    build_success       BOOLEAN,
    build_duration_ms   INTEGER,
    build_error_raw     TEXT,                  -- full error output
    build_error         TEXT,                  -- error fed to agent (summarized if needed)
    build_error_summarized BOOLEAN DEFAULT FALSE,
    build_error_summary_tokens_prompt     INTEGER DEFAULT 0,
    build_error_summary_tokens_completion INTEGER DEFAULT 0,
    -- retry info
    build_retried       BOOLEAN DEFAULT FALSE, -- transient error / cache corruption retry
    build_retry_reason  TEXT,
    -- Smoke test phase
    smoke_attempted     BOOLEAN DEFAULT FALSE,
    smoke_passed        BOOLEAN,
    smoke_results       TEXT,                  -- JSON: [{command, exit_code, output, timed_out}]
    smoke_duration_ms   INTEGER
);

CREATE INDEX idx_verify_step ON verify_build_detail(step_id);
```

### `run_artifact`

Files produced during a run.

```sql
CREATE TABLE run_artifact (
    id          TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES run(id),
    artifact_type TEXT NOT NULL,                -- 'dockerfile', 'dockerignore', 'context_blueprint', 'log'
    file_name   TEXT NOT NULL,
    content     TEXT,                           -- file content (NULL for large files stored on disk)
    file_path   TEXT,                           -- disk path if content is NULL
    created_at  TIMESTAMP NOT NULL
);

CREATE INDEX idx_artifact_run ON run_artifact(run_id);
```

---

## Query Examples

**Success rate per batch:**
```sql
SELECT
    batch_id,
    COUNT(*) AS total,
    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS successes,
    ROUND(100.0 * SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) / COUNT(*), 1) AS success_pct
FROM run
GROUP BY batch_id;
```

**Token cost per run:**
```sql
SELECT
    repo_slug,
    total_prompt_tokens,
    total_completion_tokens,
    ROUND((total_prompt_tokens * 0.0001 + total_completion_tokens * 0.0003) / 1000, 4) AS est_cost_usd
FROM run
WHERE batch_id = ?
ORDER BY total_prompt_tokens + total_completion_tokens DESC;
```

**Steps that required summarization:**
```sql
SELECT s.tool_name, COUNT(*) AS summarized_count,
       AVG(LENGTH(s.tool_output_raw)) AS avg_raw_len
FROM step s
WHERE s.was_summarized = TRUE
GROUP BY s.tool_name
ORDER BY summarized_count DESC;
```

**Average iterations to success:**
```sql
SELECT
    ROUND(AVG(iteration_count), 2) AS avg_iterations
FROM run
WHERE status = 'success';
```

**VerifyBuild outcomes breakdown:**
```sql
SELECT
    vbd.review_approved,
    vbd.build_success,
    vbd.smoke_passed,
    COUNT(*) AS count
FROM verify_build_detail vbd
GROUP BY vbd.review_approved, vbd.build_success, vbd.smoke_passed;
```
