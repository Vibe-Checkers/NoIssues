# BuildAgent v2.0

An autonomous agent that generates production-ready Dockerfiles for open-source GitHub repositories. Given a repository URL, BuildAgent analyzes the codebase, selects a base image, writes a Dockerfile, builds it, and validates it with smoke tests — all without human intervention.

---

## How It Works

BuildAgent runs a three-phase pipeline per repository:

**Phase 0 — Image Catalog**
Fetches Docker Hub's official image catalog once per batch run and caches it for all workers.

**Phase 1 — Context Blueprint**
- Uses an LLM to select the most relevant build files (package manifests, CI configs, etc.)
- Calls a metaprompt to produce a structured blueprint: detected language, repo type, suggested base image, build commands, and notes.

**Phase 2 — ReAct Agent Loop**
- Runs up to 3 iterations. Each iteration runs a langgraph ReAct agent with up to 15 tool-call steps.
- Tools: `ReadFile`, `ListDirectory`, `FindFiles`, `GrepFiles`, `WriteFile`, `DockerImageSearch`, `SearchWeb`, `VerifyBuild`
- `VerifyBuild`: LLM reviews the Dockerfile → builds it with Docker → runs smoke tests.
- On failure, an LLM extracts lessons from the previous attempt and injects them into the next iteration's prompt.

---

## Project Structure

```
src/
├── build_agent.py          # Single-repo CLI entrypoint
├── batch_runner.py         # Parallel batch runner CLI
├── agent/
│   ├── llm.py              # Azure OpenAI client (nano + chat) with rate limiting + retry
│   ├── blueprint.py        # Phase 0+1: image catalog, file selection, blueprint generation
│   ├── react_loop.py       # Phase 2: ReAct agent loop, lesson extraction
│   ├── tools.py            # Sandboxed file tools + DockerImageSearch + SearchWeb
│   ├── verify_build.py     # VerifyBuild: LLM review → Docker build → smoke tests
│   ├── summarizer.py       # Truncates/summarizes long tool outputs via LLM
│   └── docker_ops.py       # Docker build, run, cleanup with semaphore + retry
├── db/
│   ├── models.py           # Dataclasses: StepRecord, IterationRecord, RunRecord, BatchRun
│   ├── schema.py           # SQLite/PostgreSQL schema (6 tables)
│   └── writer.py           # Thread-safe DB writer (SQLite WAL / PG connection pool)
└── parallel/
    ├── rate_limiter.py     # Global RPM+TPM token bucket with background refill
    ├── disk_monitor.py     # Blocks workers when disk space is low
    └── worker.py           # Worker loop: clone → blueprint → agent → cleanup

tests/
├── test_phase_a.py         # DB models, schema, writer, summarizer
├── test_phase_b.py         # Rate limiter, LLM client
├── test_phase_c.py         # Image catalog, file selector, blueprint generator
├── test_phase_d.py         # Docker ops (requires Docker), disk monitor
├── test_phase_e.py         # All tools, VerifyBuild, sandbox
├── test_phase_f.py         # Agent loop: lesson extraction, prompt building
└── test_phase_g.py         # Worker loop, CLI, batch runner

docs/
├── TECHNICAL_DESIGN.md
├── IMPLEMENTATION_PLAN.md
├── PROMPT_SPECIFICATIONS.md
├── DATABASE_SCHEMA.md
├── PARALLEL_EXECUTION.md
├── FUNCTIONAL_REQUIREMENTS.md
├── NON_FUNCTIONAL_REQUIREMENTS.md
└── SRS.md
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# Fill in your Azure OpenAI credentials and other settings
```

**3. Run tests**
```bash
python -m pytest tests/ -v
# Docker daemon must be running for test_phase_d and test_phase_e
```

---

## Usage

### Single Repository
```bash
python src/build_agent.py https://github.com/owner/repo
python src/build_agent.py https://github.com/owner/repo --db sqlite:///results.db --max-iterations 3
```

### Batch Mode
```bash
# repos.txt: one GitHub URL per line, # for comments
python src/batch_runner.py sampled_repos_urls.txt --workers 4 --db results.db
```

### Docker Cleanup Cron Management
```bash
# Install/update a tagged cron entry (default every 30 minutes)
python scripts/manage_docker_prune_cron.py install

# Install with a custom cron schedule
python scripts/manage_docker_prune_cron.py install --schedule "*/15 * * * *"

# Check whether the tagged entry is installed
python scripts/manage_docker_prune_cron.py status

# Remove the tagged entry (safe if already absent)
python scripts/manage_docker_prune_cron.py remove
```

Safety note: this cron job runs `docker system prune --force --filter "until=1h"`, which removes **unused** Docker resources older than one hour. Active containers/images in use are not pruned, but old dangling/unused artifacts and caches will be cleaned.

---

## Database

Results are stored in SQLite (default) or PostgreSQL. The schema has 6 tables:

| Table | Description |
|-------|-------------|
| `batch_run` | One row per batch execution |
| `run` | One row per repository |
| `iteration` | Up to 3 iterations per run |
| `step` | Up to 15 steps per iteration |
| `verify_build_detail` | Full VerifyBuild output per attempt |
| `run_artifact` | Stores the final Dockerfile |

---

## Configuration

See `.env.example` for all available environment variables. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_RPM` | 60 | Requests per minute limit (`AZURE_OPENAI_RPM` fallback supported) |
| `LLM_TPM` | 80000 | Tokens per minute limit (`AZURE_OPENAI_TPM` fallback supported) |
| `OPENROUTER_TIMEOUT_SECONDS` | 120 | Total request timeout for LLM calls |
| `OPENROUTER_CONNECT_TIMEOUT_SECONDS` | 10 | Connection timeout for LLM calls |
| `OPENROUTER_READ_TIMEOUT_SECONDS` | 90 | Read timeout for LLM calls |
| `OPENROUTER_WRITE_TIMEOUT_SECONDS` | 30 | Write timeout for LLM calls |
| `OPENROUTER_MAX_RETRIES` | 3 | Transport retry count passed to LLM client |
| `DOCKER_BUILD_CONCURRENCY` | 2 | Parallel Docker builds |
| `MAX_ITERATIONS` | 3 | Agent retry iterations |
| `WORKERS` | 4 | Parallel batch workers |
| `WORKER_HEARTBEAT_LOG_INTERVAL_SECONDS` | 60 | Watchdog heartbeat log interval |
| `WORKER_HEARTBEAT_WARN_SECONDS` | 300 | Stale-worker warning threshold |
| `DATABASE_URL` | `sqlite:///results.db` | DB connection string |

Batch execution now includes worker phase heartbeat logging and stale-worker watchdog warnings from [`worker_loop()`](src/parallel/worker.py) and [`main()`](src/batch_runner.py), plus clone retry hardening in [`clone_repo()`](src/parallel/worker.py) and explicit OpenRouter timeout settings in [`LLMClient`](src/agent/llm.py).
