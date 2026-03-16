# Implementation Plan

## BuildAgent v2.0

Build order designed so each module can be tested independently before integration.

---

## Phase A: Foundation (no LLM calls, no Docker)

### A1. `src/db/models.py`

Dataclasses: `StepRecord`, `IterationRecord`, `RunRecord`, `VerifyBuildResult`, `BatchRun`.

**Test:** Instantiate each, serialize to dict, verify fields.

### A2. `src/db/schema.py`

`create_tables()` function. Runs all CREATE TABLE / CREATE INDEX statements from DATABASE_SCHEMA.md.

**Test:** Create in-memory SQLite, verify all tables exist with correct columns.

### A3. `src/db/writer.py`

Thread-safe `DBWriter` class. Methods:
- `write_batch_start(batch)`
- `write_run_start(run)`, `update_run_blueprint(run)`, `write_run_finish(run)`
- `write_iteration_start(iteration)`, `write_iteration_finish(iteration)`
- `write_step(step)`
- `write_verify_detail(detail)`
- `write_artifact(artifact)`
- `get_successful_slugs() → set[str]` (for crash recovery)

SQLite path: single connection + `threading.Lock` + WAL mode.
PostgreSQL path: `ThreadedConnectionPool`.
Pick backend based on `DATABASE_URL` env var prefix.

**Test:** Spin up in-memory SQLite, write a full batch→run→iteration→step chain, query back, verify data integrity. Test concurrent writes from 4 threads.

### A4. `src/agent/summarizer.py`

`summarize_output(content, context_type, llm) → (summary, prompt_tokens, completion_tokens)`

Handles the >2000 char gate. No LLM call if under threshold.
Fallback: truncate + append `[truncated]`.

**Test (unit):** Verify short strings pass through. Verify long strings are truncated in fallback mode (mock the LLM).

---

## Phase B: LLM Client + Rate Limiter (no Docker)

### B1. `src/parallel/rate_limiter.py`

`GlobalRateLimiter` class:
- `acquire(estimated_tokens)` — blocks if RPM or TPM budget exhausted
- `release(actual_tokens)` — adjusts TPM counter
- `backoff(retry_after)` — called on 429
- Background thread refills TPM budget every 60s

**Test:** Spawn 10 threads, each calling acquire/release. Verify RPM never exceeded. Verify TPM accounting is correct. No real API calls needed.

### B2. `src/agent/llm.py`

`LLMClient` class:
- `__init__(rate_limiter)` — creates two `AzureChatOpenAI` instances (nano + chat) from env vars
- `call_nano(messages) → LLMResponse`
- `call_chat(messages) → LLMResponse`
- Both go through rate limiter
- Retry decorator: 3 attempts, exponential backoff on 429/5xx

`LLMResponse` dataclass: `content: str`, `prompt_tokens: int`, `completion_tokens: int`.

**Test:** Mock Azure endpoint. Verify retry on 429. Verify rate limiter integration. Verify both deployments are configurable independently.

---

## Phase C: Blueprint Pipeline (needs LLM, no Docker)

### C1. `src/agent/blueprint.py`

Three functions + the image catalog:

**`ImageCatalog.fetch() → str`**
- Paginate Docker Hub `/v2/repositories/library/`
- For each image, fetch top tags from `/tags/`
- Format as compact catalog string
- Cache result

**`select_build_files(repo_root, llm) → list[str]`**
- Generate file tree string
- Read README
- Call file selector prompt (from PROMPT_SPECIFICATIONS.md §1)
- Validate paths, fallback to heuristic

**`generate_blueprint(repo_root, image_catalog, llm) → dict`**
- Read selected files (20K cap each)
- Call metaprompt (from PROMPT_SPECIFICATIONS.md §2)
- Parse JSON, validate required fields
- Fallback to extension-based detection

**Test:**
- `ImageCatalog`: mock Docker Hub responses, verify pagination and format.
- `select_build_files`: clone a small real repo (e.g., a 5-file toy project you create in `tests/fixtures/`), run with mocked LLM, verify path validation.
- `generate_blueprint`: mock LLM, verify JSON parsing and fallback.
- **Integration test:** run full blueprint on a real repo with real LLM. Verify output has all required fields and base_image is from the catalog.

---

## Phase D: Docker Operations (needs Docker, no LLM)

### D1. `src/agent/docker_ops.py`

`DockerOps` class:
- `build(context_dir, image_name) → (success, error, duration_ms)`
  - Semaphore for concurrency control
  - Transient error detection + 1 retry
  - Cache corruption detection + prune + 1 retry
  - Timeout enforcement
- `run_container(image_name, command, timeout=30) → (exit_code, output, timed_out)`
- `cleanup(image_name)`
- `prune_cache(keep_storage_gb)`

**Test:** Write a trivial Dockerfile (`FROM alpine\nRUN echo ok`), build it, run `echo test` in it, clean up. Test timeout with a `RUN sleep 999` Dockerfile. Test cleanup removes image.

### D2. `src/parallel/disk_monitor.py`

`DiskSpaceMonitor`:
- `check_or_wait()` — blocks if free space < threshold
- Calls `docker image prune -f` and `docker container prune -f` while waiting

**Test:** Mock `shutil.disk_usage` to return low space, verify it blocks. Verify it unblocks when space is restored.

---

## Phase E: Tools (needs LLM + Docker)

### E1. `src/agent/tools.py`

Tool classes, each with:
- `name`, `description` (from PROMPT_SPECIFICATIONS.md §3)
- Pydantic `InputSchema`
- `execute(input) → str`

Implement in order:
1. `ReadFileTool` — path resolution + sandbox + 512KB limit
2. `ListDirectoryTool`
3. `FindFilesTool`
4. `GrepFilesTool`
5. `WriteFileTool` — sandbox + FROM validation (uses DockerImageSearch internally)
6. `DockerImageSearchTool` — Docker Hub API calls
7. `SearchWebTool` — DuckDuckGo via `ddgs`
8. `VerifyBuildTool` — see E2

**Test per tool:** Create a temp directory with known files, run each tool, verify output. For WriteFile, verify FROM validation rejects bad images (mock Docker Hub). For sandbox, verify path traversal raises error.

### E2. `src/agent/verify_build.py`

`VerifyBuildTool.execute()`:
1. Read Dockerfile from repo root
2. Call reviewer prompt (PROMPT_SPECIFICATIONS.md §6) → approved/concerns/smoke_commands
3. If not approved → return rejected
4. Call `docker_ops.build()` → if fails, summarize error if >2000 chars
5. Run smoke tests via `docker_ops.run_container()`
6. Return VerifyBuildResult
7. Write `verify_build_detail` row to DB

**Test:** Mock the LLM reviewer. Build a known-good Dockerfile, verify accepted. Build a known-bad one, verify error is returned (and summarized if long). Test smoke test pass/fail paths.

---

## Phase F: Agent Loop (full integration)

### F1. `src/agent/react_loop.py`

**`run_iteration(prompt, tools, llm, db, max_steps=15) → IterationRecord`**
- Create LangChain `AgentExecutor` with tools and system prompt
- Set `max_iterations=max_steps`, `return_intermediate_steps=True`
- Run executor
- Extract steps from `intermediate_steps`
- For each step: check output length → summarize if >2000 → write to DB
- Detect if VerifyBuild accepted

**`extract_lessons(steps, llm) → str`**
- Format step history
- Call lesson extractor prompt (PROMPT_SPECIFICATIONS.md §5) via gpt5-chat
- Fallback to raw error extraction

**`run_agent(repo_root, blueprint, llm, docker_ops, image_name, db, max_iterations=3) → RunRecord`**
- Outer loop: up to 3 iterations
- Delete Dockerfile between iterations
- Build prompt with blueprint + lessons
- Call `run_iteration`
- If accepted → return success
- If failed → extract lessons → next iteration

**Test:**
- Mock LLM to return a predefined sequence of steps (tool calls). Verify step extraction, DB writes, summarization triggers.
- Integration test with real LLM + Docker on a toy repo. Verify end-to-end: blueprint → agent → VerifyBuild → accepted.

---

## Phase G: CLI + Batch Runner

### G1. `src/build_agent.py`

Single-repo CLI:
```
python src/build_agent.py <repo_url> [--db results.db]
```

Wires together: clone → blueprint → agent → print result.

### G2. `src/batch_runner.py`

Parallel batch runner:
```
python src/batch_runner.py repos.txt [--workers 4] [--db results.db]
```

- Phase 0: fetch image catalog
- Init shared resources (rate limiter, build semaphore, disk monitor, DB)
- Create batch_run row
- Skip already-successful slugs (crash recovery)
- ThreadPoolExecutor → worker_loop per repo
- Print progress + summary

### G3. `src/parallel/worker.py`

`worker_loop(worker_id, repo_url, ...)`:
- Clone → blueprint → agent → cleanup
- try/finally for cleanup
- All exceptions caught and logged to DB

**Test:**
- `build_agent.py`: run on a real repo, verify Dockerfile produced and DB populated.
- `batch_runner.py`: run on 3 repos with 2 workers. Verify all 3 processed, DB has correct batch→run→iteration→step hierarchy. Test crash recovery by killing mid-batch and restarting.

---

## Phase H: Hardening

### H1. End-to-end test suite

- 5 repos of different types (Python lib, Node CLI, Java Maven, Go binary, C++ CMake)
- Run batch, assert success rate, verify DB completeness
- Verify token counts are non-zero at every level

### H2. Rate limiter tuning

- Run with artificially low RPM (5) and verify no 429s
- Verify backoff works by injecting fake 429 responses

### H3. Disk monitor test

- Run with low disk threshold, verify backpressure kicks in

---

## Build Order Summary

```
A1 models.py ──┐
A2 schema.py ──┼── A3 writer.py ──┐
A4 summarizer  ──────────────────┤
                                  │
B1 rate_limiter ── B2 llm.py ────┤
                                  │
C1 blueprint.py ─────────────────┤
                                  │
D1 docker_ops.py ─ D2 disk_mon ─┤
                                  │
E1 tools.py ── E2 verify_build ──┤
                                  │
F1 react_loop.py ────────────────┤
                                  │
G1 build_agent.py ── G2 batch_runner.py ── G3 worker.py
                                  │
                          H1–H3 hardening
```

**Dependencies flow downward.** Each phase only depends on completed phases above it.

---

## File Count

| Phase | New Files | Description |
|-------|-----------|-------------|
| A     | 4         | db/models, db/schema, db/writer, agent/summarizer |
| B     | 2         | parallel/rate_limiter, agent/llm |
| C     | 1         | agent/blueprint |
| D     | 2         | agent/docker_ops, parallel/disk_monitor |
| E     | 2         | agent/tools, agent/verify_build |
| F     | 1         | agent/react_loop |
| G     | 3         | build_agent, batch_runner, parallel/worker |
| **Total** | **15** | + `__init__.py` files and `.env` |
