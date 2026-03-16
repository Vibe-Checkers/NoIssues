# Non-Functional Requirements

## BuildAgent v2.0

---

## NFR-1: Performance

**NFR-1.1 Step Budget**
- 15 steps per iteration, 3 iterations per run = max 45 LLM agent calls per repository.
- Each step shall complete within 60s (LLM call + tool execution).

**NFR-1.2 Docker Build Timeout**
- Default: 600s. Configurable per run.
- Builds exceeding timeout are killed and the error returned to the agent.

**NFR-1.3 Smoke Test Timeout**
- Default: 30s per command. Timed-out commands count as failures.

**NFR-1.4 Output Summarization Overhead**
- Summarization calls (for outputs >2000 chars) shall complete within 10s.
- Summarization shall preserve error messages, paths, versions, and commands.

---

## NFR-2: Scalability

**NFR-2.1 Worker Concurrency**
- Configurable via `--workers N` (default: 4).
- Each worker uses an isolated clone directory and Docker image namespace.

**NFR-2.2 API Rate Limit Compliance**
- A global token-bucket or semaphore shall throttle Azure OpenAI calls across all workers.
- The system shall respect both TPM (tokens-per-minute) and RPM (requests-per-minute) limits.
- See [Parallel Execution Strategy](./PARALLEL_EXECUTION.md) for detailed design.

**NFR-2.3 Disk Space Management**
- Cloned repos deleted immediately after run completes.
- Docker images removed after verification.
- A disk-space guard shall pause new clones if available space drops below a configurable threshold (default: 5GB).

---

## NFR-3: Reliability

**NFR-3.1 Transient Error Recovery**
- Docker build: retry once after 5s on network errors (DNS, timeout, 503).
- BuildKit cache corruption: prune cache + retry once.
- Azure OpenAI: retry with exponential backoff (3 attempts, 1s/2s/4s).

**NFR-3.2 LLM Fallbacks**
- If context blueprint generation fails → proceed with file-extension language detection only.
- If output summarization fails → truncate to first 2000 chars.
- If VerifyBuild review fails → skip review, attempt build directly.
- If lesson extraction fails → inject raw last-error text as lesson.
- All fallbacks shall be logged.

**NFR-3.3 Crash Recovery**
- All completed steps are persisted to the database incrementally.
- A crashed run can be identified by checking for runs without a `finished_at` timestamp.
- Batch runner shall skip repos that already have a successful run in the database.

---

## NFR-4: Security

**NFR-4.1 File Sandbox**
- All agent file operations resolve within `repo_root`. Path traversal rejected.

**NFR-4.2 Dockerfile Validation**
- `FROM` lines validated against Docker Hub before write. Invalid images blocked.

**NFR-4.3 Container Isolation**
- Smoke test containers run with `--rm`, overridden entrypoint, and timeout enforcement.

---

## NFR-5: Observability

**NFR-5.1 Token Tracking**
- Every LLM call (agent steps, summarizations, reviews, lesson extraction) shall record prompt_tokens and completion_tokens.
- Tokens aggregated per step, iteration, run, and batch.

**NFR-5.2 Duration Tracking**
- Wall-clock duration recorded for: each step, each iteration, each run, Docker builds, smoke tests.

**NFR-5.3 Database Persistence**
- All metrics persisted to SQL. See [Database Schema](./DATABASE_SCHEMA.md).

**NFR-5.4 Structured Logs**
- JSON logs per repo for backward compatibility with analysis scripts.

---

## NFR-6: Portability

**NFR-6.1** Supported on Linux (production) and macOS (development).

**NFR-6.2** Host architecture auto-detected (amd64/arm64) and passed to Docker builds.
