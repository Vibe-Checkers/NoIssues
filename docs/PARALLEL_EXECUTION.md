# Parallel Execution Strategy

## BuildAgent v2.0

---

## 1. Architecture Overview

```
                    ┌──────────────────────────┐
                    │       Batch Runner        │
                    │  (main thread)            │
                    │                           │
                    │  • Reads repo list        │
                    │  • Spawns worker pool     │
                    │  • Collects results       │
                    │  • Writes batch_run to DB │
                    └────────┬─────────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Worker 1 │  │ Worker 2 │  │ Worker N │
        │          │  │          │  │          │
        │ • Clone  │  │ • Clone  │  │ • Clone  │
        │ • Bluepr.│  │ • Bluepr.│  │ • Bluepr.│
        │ • Agent  │  │ • Agent  │  │ • Agent  │
        │ • Build  │  │ • Build  │  │ • Build  │
        │ • Cleanup│  │ • Cleanup│  │ • Cleanup│
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │              │              │
             ▼              ▼              ▼
    ┌────────────────────────────────────────────┐
    │           Shared Resources                  │
    │                                             │
    │  ┌─────────────┐  ┌───────────────────────┐ │
    │  │ API Rate    │  │ Docker Daemon         │ │
    │  │ Limiter     │  │ (build serialization) │ │
    │  │ (semaphore) │  │                       │ │
    │  └─────────────┘  └───────────────────────┘ │
    │  ┌─────────────┐  ┌───────────────────────┐ │
    │  │ Disk Space  │  │ SQL Database          │ │
    │  │ Monitor     │  │ (thread-safe writes)  │ │
    │  └─────────────┘  └───────────────────────┘ │
    └─────────────────────────────────────────────┘
```

---

## 2. API Rate Limiting

### The Problem

Azure OpenAI enforces per-deployment limits:
- **RPM** (requests per minute) — e.g., 250 RPM per deployment
- **TPM** (tokens per minute) — e.g., 250K TPM per deployment

With N workers each making LLM calls for agent steps, summarizations, reviews, and lesson extraction, a single deployment is easily saturated.

### The Solution: Multi-Deployment Round-Robin + Global Rate Limiter

**Multi-deployment round-robin**: The nano model is deployed across multiple Azure OpenAI deployments (e.g., 4 deployments x 250 RPM = 1000 RPM aggregate). Each worker is assigned to a deployment via `worker_id % num_deployments`:

```
AZURE_OPENAI_DEPLOYMENT_NANO=gpt-5-nano,gpt-5-nano-2,gpt-5-nano-3,gpt-5-nano-4
```

All deployments share the same Azure resource, endpoint, and API key — they are named slots within a single resource.

**Global token-bucket rate limiter** tracks aggregate RPM + TPM across all deployments:

```
┌─────────────────────────────────────────┐
│  GlobalRateLimiter (singleton)          │
│                                         │
│  rpm_semaphore:  threading.Semaphore    │
│  tpm_budget:     AtomicCounter          │
│  refill_task:    runs every 60s         │
│                                         │
│  acquire(estimated_tokens) → blocks     │
│  release(actual_tokens)   → adjusts    │
│  backoff(retry_after)     → on 429     │
└─────────────────────────────────────────┘
```

**Implementation:**
1. Before each LLM call, the worker calls `rate_limiter.acquire(estimated_tokens)`.
2. If the RPM semaphore is full or TPM budget is exhausted, the call blocks.
3. A background thread refills the TPM budget every 60 seconds.
4. After the call completes, `release(actual_tokens)` adjusts the running count.

**Configuration:**
```
AZURE_OPENAI_RPM=1000         # aggregate across all nano deployments
AZURE_OPENAI_TPM=1000000      # aggregate across all nano deployments
```

**Backoff on 429:**
- If a 429 is received despite the limiter, wait for `Retry-After` header value (or 10s default).
- Retry decorator: 3 attempts with exponential backoff on 429/5xx.

---

## 3. Disk Space Management

### The Problem

Each concurrent worker can consume:
- Clone: 10MB–2GB per repo
- Docker build context + layers: 500MB–10GB per repo
- Total for 4 workers: up to 48GB simultaneously

### The Solution: Disk Space Monitor + Backpressure

```python
class DiskSpaceMonitor:
    threshold_bytes: int   # default: 5GB
    check_interval: int    # default: every repo, before clone

    def check_or_wait(self):
        """Block until disk space is above threshold."""
        while get_free_space() < self.threshold_bytes:
            log.warning(f"Disk below {self.threshold_bytes/1e9:.1f}GB, waiting...")
            # Force cleanup of any orphaned images/containers
            run("docker image prune -f")
            run("docker container prune -f")
            sleep(30)
```

**Cleanup protocol per worker:**
1. After run completes (success or failure):
   - `docker rmi -f <image_name>` — remove built image
   - `shutil.rmtree(clone_dir)` — remove cloned repo
2. On worker crash: the batch runner's finally-block scans for orphaned clone dirs and images.
3. Periodic Docker prune: every 10 completed repos, run `docker builder prune --keep-storage=10GB -f`.

**Configuration:**
```
DISK_SPACE_THRESHOLD_GB=5
DOCKER_PRUNE_INTERVAL=10      # repos between prunes
DOCKER_KEEP_STORAGE_GB=10     # BuildKit cache cap
```

---

## 4. Docker Build Serialization

### The Problem

Concurrent `docker build` commands sharing the same BuildKit daemon can cause:
- Cache corruption ("parent snapshot does not exist")
- Resource contention slowing all builds
- OOM when multiple large builds run simultaneously

### The Solution: Build Semaphore

```python
build_semaphore = threading.Semaphore(2)  # max 2 concurrent builds

def build_dockerfile(dockerfile_path, ...):
    with build_semaphore:
        return _do_build(dockerfile_path, ...)
```

**Tuning:**
- Default concurrency: `min(2, workers // 2)` — at most half the workers build simultaneously.
- Configurable via `DOCKER_BUILD_CONCURRENCY`.
- Independent of API rate limiting (different resource).

---

## 5. Worker Isolation

Each worker must have fully isolated state to prevent cross-contamination:

| Resource             | Isolation Method                                     |
|----------------------|------------------------------------------------------|
| Clone directory      | Unique path: `workdir/<batch_id>/<worker_id>/<slug>` |
| Docker image name    | Unique tag: `buildagent-<slug>-<worker_id>`          |
| Logging context      | Thread-local logger with `[worker-N]` prefix         |
| Error dedup set      | Thread-local set per worker                          |
| DB writes            | Thread-safe (SQLite WAL mode or PG connection pool)  |

---

## 6. Database Concurrency

### SQLite (local/dev)

- Enable WAL mode: `PRAGMA journal_mode=WAL;`
- Set busy timeout: `PRAGMA busy_timeout=5000;`
- Use a single `sqlite3.Connection` with a `threading.Lock` for writes.
- Reads can proceed concurrently (WAL allows this).

### PostgreSQL (production)

- Use a connection pool (e.g., `psycopg2.pool.ThreadedConnectionPool`).
- Pool size = worker count + 2 (for batch runner + monitoring).
- Each worker gets its own connection from the pool.
- Commits after each step to ensure crash recovery data is persisted.

---

## 7. Failure Handling in Parallel

| Failure Type          | Handling                                                     |
|-----------------------|--------------------------------------------------------------|
| Worker exception      | Catch per-repo, log to DB as `status='error'`, continue batch|
| API 429 (rate limit)  | Global backoff via rate limiter, retry after `Retry-After`   |
| API 5xx               | Exponential backoff (3 attempts), then fail the step         |
| Docker daemon crash   | Detect via build error, skip remaining repos, alert          |
| Disk full             | Backpressure monitor blocks new clones, force prune          |
| OOM (Python process)  | Not recoverable; batch runner logs partial results to DB     |
| Network outage        | Git clone retry (3 attempts); API retry (3 attempts)         |

---

## 8. Monitoring and Observability

**During a batch run, the batch runner shall print:**
```
[14:32:01] [batch] 12/100 complete | 9 success | 2 fail | 1 error | 3 active
[14:32:01] [worker-1] flask: iteration 2/3, step 8/15
[14:32:01] [worker-2] react: VerifyBuild (building...)
[14:32:01] [worker-3] guava: clone
[14:32:01] [rate] API: 42/60 RPM, 61K/80K TPM | Disk: 28GB free
```

**Post-run summary:**
```
Batch complete: 100 repos in 4h 12m
  Success: 78 (78%)
  Failure: 18 (18%)
  Error:    4 (4%)
  Total tokens: 12.4M prompt, 3.1M completion
  Est. cost: $1.86
  Avg iterations/success: 1.4
  Avg duration/repo: 2m 31s
```

---

## 9. Recommended Configurations

### Small batch (≤20 repos, local machine)

```
WORKERS=2
DOCKER_BUILD_CONCURRENCY=1
AZURE_OPENAI_RPM=60
AZURE_OPENAI_TPM=80000
DISK_SPACE_THRESHOLD_GB=5
DATABASE=sqlite:///results.db
```

### Medium batch (20–100 repos, single VM)

```
WORKERS=4
DOCKER_BUILD_CONCURRENCY=2
AZURE_OPENAI_RPM=120
AZURE_OPENAI_TPM=150000
DISK_SPACE_THRESHOLD_GB=10
DATABASE=sqlite:///results.db
DOCKER_KEEP_STORAGE_GB=20
```

### Large batch (100+ repos, dedicated VM)

```
WORKERS=8
DOCKER_BUILD_CONCURRENCY=3
AZURE_OPENAI_RPM=300
AZURE_OPENAI_TPM=400000
DISK_SPACE_THRESHOLD_GB=20
DATABASE=postgresql://user:pass@host/buildagent
DOCKER_KEEP_STORAGE_GB=50
```

---

## 10. Improvements Over Current Implementation

| Area                  | Current                              | Improved                                                |
|-----------------------|--------------------------------------|---------------------------------------------------------|
| API rate limiting     | Simple semaphore (2 concurrent)      | Token-bucket with RPM+TPM tracking, 429 backoff         |
| Disk management       | Manual cleanup, no monitoring        | Automated monitor with backpressure + periodic prune    |
| Docker concurrency    | Optional global lock                 | Tunable semaphore based on worker count                 |
| Crash recovery        | Lost on crash                        | Incremental DB writes; skips completed repos on restart |
| Result storage        | Flat files (JSONL)                   | SQL database with queryable schema                      |
| Build serialization   | All-or-nothing lock                  | Configurable concurrent build slots                     |
| Observability         | Scattered log files                  | Live progress line + post-run summary + DB queries      |
| Worker isolation      | Partially shared state               | Fully isolated (dirs, image names, logging, dedup)      |
