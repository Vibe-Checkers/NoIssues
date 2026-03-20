# BuildAgent v2.0 — Project Guide

A complete reference for anyone new to this repository. BuildAgent is an autonomous AI agent that generates production-ready Dockerfiles for open-source GitHub repositories — the subject of an academic study evaluating LLM-driven containerization.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Architecture Overview](#architecture-overview)
3. [Source Code Structure](#source-code-structure)
4. [Database Schema](#database-schema)
5. [Pipeline Phases in Detail](#pipeline-phases-in-detail)
6. [Configuration & Environment](#configuration--environment)
7. [Running the Agent](#running-the-agent)
8. [Test Suite](#test-suite)
9. [Result Analysis](#result-analysis)
10. [Key Findings Summary](#key-findings-summary)

---

## What This Project Does

Given a GitHub repository URL, BuildAgent:

1. Clones the repository
2. Uses an LLM to analyze the codebase and generate a build blueprint
3. Runs a ReAct agent loop that writes a Dockerfile, builds it with Docker, and validates it with smoke tests
4. If the build or tests fail, the agent extracts lessons and retries (up to 3 iterations)
5. Records every step, token count, and timing to a database for analysis

The study evaluated this pipeline on **282 repositories** spanning 13 domains and 46 build systems.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                      batch_runner.py                         │
│              ThreadPoolExecutor (N workers)                   │
├──────────────┬───────────────┬───────────────┬───────────────┤
│   Worker 1   │   Worker 2    │   Worker 3    │   Worker N    │
│  ┌────────┐  │  ┌────────┐   │  ┌────────┐   │  ┌────────┐  │
│  │ Clone  │  │  │ Clone  │   │  │ Clone  │   │  │ Clone  │  │
│  │ Blue-  │  │  │ Blue-  │   │  │ Blue-  │   │  │ Blue-  │  │
│  │ print  │  │  │ print  │   │  │ print  │   │  │ print  │  │
│  │ Agent  │  │  │ Agent  │   │  │ Agent  │   │  │ Agent  │  │
│  │ Loop   │  │  │ Loop   │   │  │ Loop   │   │  │ Loop   │  │
│  └────────┘  │  └────────┘   │  └────────┘   │  └────────┘  │
├──────────────┴───────────────┴───────────────┴───────────────┤
│       Shared: Rate Limiter │ Disk Monitor │ DB Writer        │
└──────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  PostgreSQL / SQLite│
                    └───────────────────┘
```

**LLM Models Used (via OpenRouter):**
- **google/gemini-2.0-flash-001** — agent steps, blueprint, verify review, summarization
- **anthropic/claude-sonnet-4** — lesson extraction between iterations

---

## Source Code Structure

```
src/
├── build_agent.py              # CLI: process a single repository
├── batch_runner.py             # CLI: process repos in parallel from a file
├── agent/
│   ├── llm.py                  # OpenRouter client with rate limiting, retry, timeouts
│   ├── blueprint.py            # Phase 0+1: Docker image catalog, file selection, blueprint
│   ├── react_loop.py           # Phase 2: ReAct agent loop, iteration management, lessons
│   ├── tools.py                # 8 sandboxed tools the agent can call
│   ├── verify_build.py         # VerifyBuild: LLM review → docker build → smoke tests
│   ├── summarizer.py           # Truncates/summarizes long tool outputs via LLM
│   └── docker_ops.py           # Docker build, run, cleanup with concurrency control
├── db/
│   ├── models.py               # Dataclasses: StepRecord, IterationRecord, RunRecord, BatchRun
│   ├── schema.py               # CREATE TABLE statements (SQLite + PostgreSQL compatible)
│   └── writer.py               # Thread-safe DB writer (SQLite WAL / PG connection pool)
└── parallel/
    ├── rate_limiter.py          # Global RPM + TPM token bucket with background refill
    ├── disk_monitor.py          # Blocks workers when disk < threshold, runs docker prune
    └── worker.py                # Per-worker loop: clone → blueprint → agent → cleanup

scripts/
├── manage_docker_prune_cron.py  # Install/remove Docker cleanup cron job
├── sample_repos.py              # Utility to sample repos from a list
└── show_steps.py                # Print step history from the database

docs/                            # Design documents (SRS, technical design, prompts, etc.)
tests/                           # 8-phase test suite (A through H)
```

---

## Database Schema

The database stores the complete execution trace. Every LLM call, tool invocation, Docker build, and smoke test is recorded.

### Entity Relationship

```
batch_run  1──N  run  1──N  iteration  1──N  step
                  │                           │
                  └── run_artifact             └── verify_build_detail
```

### Tables

#### `batch_run` — One row per batch execution
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `started_at` / `finished_at` | TIMESTAMP | Batch timing |
| `worker_count` | INTEGER | Number of parallel workers |
| `repo_count` | INTEGER | Total repos in input file |
| `success_count` / `failure_count` | INTEGER | Final tallies |
| `total_prompt_tokens` / `total_completion_tokens` | INTEGER | Aggregated LLM usage |
| `config_json` | TEXT | Serialized run configuration |
| `ablation` | TEXT | Experiment variant (`"default"` or `"no-metaprompt"`) |

#### `run` — One row per repository processed
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `batch_id` | TEXT FK | References `batch_run` |
| `repo_url` / `repo_slug` | TEXT | GitHub URL and `owner/repo` slug |
| `status` | TEXT | `success`, `failure`, `error`, or `skipped` |
| `started_at` / `finished_at` / `duration_ms` | — | Timing |
| `iteration_count` | INTEGER | How many iterations were run (1-3) |
| `detected_language` / `repo_type` | TEXT | From blueprint phase |
| `context_blueprint` | TEXT | Full blueprint JSON |
| `blueprint_tokens_prompt` / `blueprint_tokens_completion` | INTEGER | Blueprint LLM cost |
| `blueprint_duration_ms` | INTEGER | Blueprint wall-clock time |
| `final_dockerfile` | TEXT | The accepted Dockerfile (NULL if failed) |
| `smoke_test_passed` | BOOLEAN | Whether smoke tests passed |
| `total_prompt_tokens` / `total_completion_tokens` / `total_steps` | INTEGER | Run totals |
| `error_message` | TEXT | Error details if status = `error` |
| `worker_id` | INTEGER | Which parallel worker handled this repo |

#### `iteration` — Up to 3 per run
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `run_id` | TEXT FK | References `run` |
| `iteration_number` | INTEGER | 1, 2, or 3 |
| `status` | TEXT | `success`, `failure`, `error` |
| `step_count` | INTEGER | Number of agent steps taken |
| `injected_lessons` | TEXT | Lessons from previous iteration's failure |
| `dockerfile_generated` | BOOLEAN | Whether the agent wrote a Dockerfile |
| `verify_attempted` | BOOLEAN | Whether VerifyBuild was called |
| `verify_result` | TEXT | `accepted`, `rejected`, `build_failed`, `smoke_failed` |
| `prompt_tokens` / `completion_tokens` | INTEGER | Agent step tokens |
| `lesson_extraction_tokens_prompt` / `_completion` | INTEGER | Sonnet lesson extraction cost |

#### `step` — Up to 25 per iteration (one per tool call)
| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `iteration_id` | TEXT FK | References `iteration` |
| `step_number` | INTEGER | Sequential within iteration |
| `thought` | TEXT | LLM's reasoning before tool call |
| `tool_name` | TEXT | Which tool was called |
| `tool_input` | TEXT | JSON of tool arguments |
| `tool_output_raw` / `tool_output` | TEXT | Raw and (possibly summarized) output |
| `was_summarized` | BOOLEAN | Whether output was LLM-summarized |
| `prompt_tokens` / `completion_tokens` | INTEGER | This step's LLM tokens |
| `summary_prompt_tokens` / `summary_completion_tokens` | INTEGER | Summarization tokens |

#### `verify_build_detail` — One per VerifyBuild call
| Column | Type | Description |
|--------|------|-------------|
| `step_id` | TEXT FK | References `step` |
| `review_approved` | BOOLEAN | LLM reviewer's verdict |
| `review_score` | INTEGER | Quality score from reviewer |
| `review_concerns` | TEXT | JSON list of concerns |
| `smoke_test_commands` | TEXT | JSON list of smoke test commands designed by reviewer |
| `build_attempted` / `build_success` | BOOLEAN | Docker build outcome |
| `build_error_raw` / `build_error` | TEXT | Raw and summarized build errors |
| `build_duration_ms` | INTEGER | Docker build wall-clock time |
| `smoke_attempted` / `smoke_passed` | BOOLEAN | Smoke test outcome |
| `smoke_results` | TEXT | JSON with per-command results |
| `smoke_duration_ms` | INTEGER | Smoke test wall-clock time |

#### `run_artifact` — Generated files
| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT FK | References `run` |
| `artifact_type` | TEXT | e.g., `dockerfile`, `dockerignore` |
| `file_name` | TEXT | e.g., `Dockerfile` |
| `content` | TEXT | File content |

#### `image_catalog` — Cached Docker Hub image list
| Column | Type | Description |
|--------|------|-------------|
| `fetched_at` | TIMESTAMP | When the catalog was fetched |
| `image_count` | INTEGER | Number of official images (~178) |
| `content` | TEXT | JSON list of images with tags |

---

## Pipeline Phases in Detail

### Phase 0 — Docker Image Catalog
- Fetches all official Docker Hub images via the registry API
- Extracts the top 6 tags per image
- Cached in `image_catalog` table (fetched once per batch)

### Phase 1 — Context Blueprint
1. **File Selection**: LLM sees the repo's file tree and selects the most relevant build files (package.json, requirements.txt, Makefile, CI configs, etc.)
2. **Context Collection**: Reads selected files + README (budget: ~60K chars)
3. **Metaprompt**: LLM generates a structured blueprint:
   ```json
   {
     "language": "python",
     "repo_type": "web_service",
     "base_image": "python:3.11-slim",
     "build_commands": ["pip install -r requirements.txt"],
     "notes": "Uses FastAPI with uvicorn"
   }
   ```
4. **Ablation variant**: `--ablation no-metaprompt` skips this phase entirely

### Phase 2 — ReAct Agent Loop
- **Outer loop**: up to 3 iterations
- **Inner loop**: up to 25 steps per iteration (LangGraph ReAct agent)
- **Available tools**:

| Tool | Purpose |
|------|---------|
| `ReadFile` | Read a file from the cloned repo (max 512KB) |
| `ListDirectory` | List directory contents |
| `FindFiles` | Glob pattern search |
| `GrepFiles` | Regex search across files |
| `WriteFile` | Write files (Dockerfile, .dockerignore) — sandboxed to repo root, validates FROM |
| `DockerImageSearch` | Search Docker Hub for images |
| `SearchWeb` | Web search via DuckDuckGo |
| `VerifyBuild` | LLM review → Docker build → smoke tests |

- **VerifyBuild pipeline**: LLM reviews the Dockerfile and designs smoke tests. If approved, Docker builds the image. If the build succeeds, smoke tests run in the container. Result is one of: `accepted`, `rejected`, `build_failed`, `smoke_failed`.
- **Lesson extraction**: On failure, an LLM (Claude Sonnet) analyzes the step history and produces concise lessons injected into the next iteration's system prompt.

---

## Configuration & Environment

Copy `.env.example` to `.env` and fill in credentials. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | — | **Required.** OpenRouter API key |
| `OPENROUTER_MODEL_NANO` | `google/gemini-2.0-flash-001` | Fast model for agent steps |
| `OPENROUTER_MODEL_CHAT` | `anthropic/claude-sonnet-4` | Model for lesson extraction |
| `DATABASE_URL` | `sqlite:///results.db` | PostgreSQL or SQLite connection string |
| `MAX_ITERATIONS` | 3 | Agent retry iterations per repo |
| `WORKERS` | 4 | Parallel workers in batch mode |
| `DOCKER_BUILD_CONCURRENCY` | 2 | Max parallel Docker builds |
| `DOCKER_BUILD_TIMEOUT` | 600 | Docker build timeout (seconds) |
| `LLM_RPM` / `LLM_TPM` | 60 / 200000 | Rate limits (requests/tokens per minute) |
| `DISK_SPACE_THRESHOLD_GB` | 5 | Minimum free disk before blocking workers |

---

## Running the Agent

### Single repository
```bash
python src/build_agent.py https://github.com/owner/repo
```

### Batch mode (parallel)
```bash
python src/batch_runner.py our_282.txt --workers 4 --db postgresql://... --ablation default
```

### Ablation study
```bash
# Full pipeline (default)
python src/batch_runner.py repos.txt --ablation default

# Skip blueprint phase
python src/batch_runner.py repos.txt --ablation no-metaprompt
```

---

## Test Suite

Eight test phases (A through H) with increasing integration scope:

| Phase | File | What it tests | Requirements |
|-------|------|---------------|--------------|
| A | `test_phase_a.py` | DB models, schema, writer, summarizer | None |
| B | `test_phase_b.py` | Rate limiter, LLM client | Network (OpenRouter) |
| C | `test_phase_c.py` | Image catalog, file selector, blueprint | Network + LLM |
| D | `test_phase_d.py` | Docker ops, disk monitor | Docker daemon |
| E | `test_phase_e.py` | All tools, VerifyBuild, sandbox | Docker + LLM |
| F | `test_phase_f.py` | Agent loop, lesson extraction | LLM |
| G | `test_phase_g.py` | Worker loop, CLI, batch runner | Docker + LLM + Network |
| H | `test_phase_h.py` | End-to-end integration | Docker + LLM + Network |

```bash
python -m pytest tests/ -v                # Run all
python -m pytest tests/test_phase_a.py -v # Run just phase A
```

---

## Result Analysis

The `result_analysis/` directory contains the complete analysis pipeline for the 282-repo experiment. Each script is self-contained and produces CSV, JSON, and LaTeX outputs.

### Directory Layout

```
result_analysis/
├── our_282.txt                              # Input: 282 GitHub repo URLs
├── stratified_repos_2000_majority_vote.csv  # Repo characterization (from 2000-repo pool)
│
├── fetch_all_results.py                     # Step 1: Fetch all data from PostgreSQL
├── summary.csv                              # One row per repo: result, language, tokens
├── aggregate_stats.json                     # Overall success rate, language distribution
├── per_repo/                                # 282 JSON files, full hierarchy per repo
│   ├── fastapi--fastapi.json
│   ├── facebook--react.json
│   └── ...
│
├── analyze_results.py                       # Step 2: Category-level analysis
├── analysis_output/
│   ├── merged_dataset.csv                   # 282 rows: results + characterization joined
│   ├── statistical_tests.csv                # Chi-square + Cramer's V for all 7 dimensions
│   ├── fisher_pairwise.csv                  # Best-vs-worst Fisher exact tests
│   ├── breakdown_domain.csv                 # Success/fail by project domain
│   ├── breakdown_build_type.csv             # Success/fail by build system
│   ├── breakdown_automation_level.csv       # ... and 5 more category breakdowns
│   ├── build_type_families.csv              # Grouped build systems (JS, Python, etc.)
│   ├── cross_domain_buildtype.csv           # Domain x build_type cross-tabulation
│   ├── cross_automation_tooling.csv         # Automation x tooling cross-tabulation
│   ├── usage_*.csv                          # Avg iterations/steps/tokens by category
│   └── latex_tables.tex                     # Paper-ready LaTeX tables
│
├── analyze_failures.py                      # Step 3: Deep failure analysis
├── failure_analysis/
│   ├── root_cause_distribution.csv          # Why repos failed (build error, smoke, etc.)
│   ├── build_error_taxonomy.csv             # Error categories (deps, version, syntax, etc.)
│   ├── smoke_error_taxonomy.csv             # Smoke test failure categories
│   ├── iteration_progression.json           # Did retries help?
│   ├── per_repo_failure_report.csv          # Detailed report card per failed repo
│   ├── tool_usage_comparison.csv            # Tool usage: failed vs successful repos
│   ├── review_concerns.csv                  # Most common LLM reviewer concerns
│   ├── failed_base_images.csv               # Base images used in failed Dockerfiles
│   ├── failing_docker_commands.csv          # Which Docker commands fail most
│   ├── cross_rootcause_domain.csv           # Root cause x domain
│   ├── cross_rootcause_build_type.csv       # Root cause x build type
│   ├── failure_summary.json                 # Full aggregate + error examples
│   └── failure_latex_tables.tex             # Paper-ready LaTeX tables
│
├── cost_time_analysis/
│   ├── analyze_cost_time.py                 # Step 4: Cost and time analysis
│   ├── per_repo_cost_time.csv               # Full cost/time breakdown per repo
│   ├── per_iteration_cost_time.csv          # Cost/time per iteration
│   ├── per_iteration_number_stats.csv       # Averages by iteration number (1st, 2nd, 3rd)
│   ├── cost_by_domain.csv                   # Total/avg cost by project domain
│   ├── cost_by_build_type.csv               # Total/avg cost by build system
│   ├── cost_time_summary.json               # Aggregate stats, success vs fail comparison
│   └── cost_time_latex_tables.tex           # Paper-ready LaTeX tables
```

### How to Run the Analysis

The scripts are designed to run in sequence. Each depends on the output of the previous one.

```bash
# Step 1: Fetch all run data from PostgreSQL into per-repo JSONs
python3 result_analysis/fetch_all_results.py

# Step 2: Analyze success/fail rates by repo characteristics
python3 result_analysis/analyze_results.py

# Step 3: Deep-dive into the 86 failure cases
python3 result_analysis/analyze_failures.py

# Step 4: Cost and time breakdown
python3 result_analysis/cost_time_analysis/analyze_cost_time.py
```

### Repo Characterization Dimensions

Each of the 282 repos was characterized along 7 dimensions (from `stratified_repos_2000_majority_vote.csv`):

| Dimension | Values | Description |
|-----------|--------|-------------|
| `domain` | web-development, machine-learning, utilities, systems-programming, mobile-development, security, devops, education, data-science, game-development, scientific-computing, documentation | Project's application domain |
| `build_type` | npm, pip, go-mod, cargo, cmake, gradle, maven, etc. (46 types) | Primary build system |
| `automation_level` | manual, semi_automated, fully_automated, reverse_engineering | How automated is the build? |
| `environment_specificity` | cross_platform_generic, specific_os, specific_hardware_or_drivers, custom_hardware_or_drivers | Platform requirements |
| `dependency_transparency` | explicit_machine_readable, explicit_loose, implicit, opaque | How clear are dependencies? |
| `tooling_complexity` | single_layer_tool, multi_layer_toolchains, mixed_languages_and_bindings | Build toolchain complexity |
| `reproducibility_support` | repro_ready, no_ci_manual_only, partial_repro, broken_or_outdated | CI/reproducibility maturity |

---

## Key Findings Summary

### Overall Results
- **282 repositories** evaluated, **196 successful** (69.5%), **86 failed** (30.5%)
- Total cost: **$34.80** for all 282 repos ($0.12 avg per repo)
- Total wall-clock time: **193.5 hours** (41 min avg per repo)

### Statistically Significant Factors

| Dimension | Chi-square | p-value | Cramer's V | Effect |
|-----------|-----------|---------|------------|--------|
| Build type | 80.27 | 0.001 | 0.534 | Strong |
| Domain | 27.22 | 0.002 | 0.311 | Medium |
| Dependency transparency | 10.02 | 0.018 | 0.189 | Small-medium |

### Best and Worst Performers
- **Go (go-mod): 100% success** (16/16) — the best build system
- **DevOps repos: 88.9% success** — the best domain
- **setuptools: 31.2% success** — builds succeed but smoke tests fail
- **Data science: 40% success** — GPU/data dependencies cause runtime failures

### Failure Root Causes
- 51.2% — Build always failed (could never produce a working image)
- 44.2% — Smoke test failure (image built but runtime tests failed)
- 4.6% — Review rejection or no attempt

### Cost Comparison
- Successful repos: **$0.08 avg**, 24 min avg
- Failed repos: **$0.23 avg**, 81 min avg (2.9x more expensive)

---

## Documentation

Additional design documents are in `docs/`:

| Document | Description |
|----------|-------------|
| `TECHNICAL_DESIGN.md` | Module architecture, dataclasses, phase design |
| `DATABASE_SCHEMA.md` | Full ER diagram and table definitions |
| `PROMPT_SPECIFICATIONS.md` | All LLM prompts (system, metaprompt, lessons) |
| `PARALLEL_EXECUTION.md` | Worker coordination, rate limiting, error recovery |
| `FUNCTIONAL_REQUIREMENTS.md` | User stories, acceptance criteria |
| `NON_FUNCTIONAL_REQUIREMENTS.md` | Performance, scalability, security |
| `IMPLEMENTATION_PLAN.md` | Development timeline and milestones |
| `SRS.md` | Software requirements specification |
