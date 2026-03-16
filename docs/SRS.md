# Software Requirements Specification

## BuildAgent — Automated Dockerfile Generation System

| Field   | Value              |
|---------|--------------------|
| Version | 2.0                |
| Date    | 2026-03-16         |

---

## 1. Introduction

BuildAgent is a Python-based system that autonomously generates production-ready Dockerfiles for arbitrary GitHub repositories. It uses a two-phase LLM pipeline: a **context blueprint** phase that analyzes the repository structure, followed by a **ReAct agent** phase that generates, builds, and verifies Dockerfiles.

### 1.1 Glossary

| Term              | Definition                                                       |
|-------------------|------------------------------------------------------------------|
| Context Blueprint | Structured analysis of a repo produced by the metaprompting LLM before the agent starts |
| Step              | A single LLM call that produces a thought + tool action + result |
| Iteration         | A sequence of up to 15 steps; 3 iterations per run              |
| Lesson            | Summary of a failed iteration injected into the next iteration   |
| Smoke Test        | Command run inside a built container to verify functionality     |
| Run               | The full processing of one repository (up to 3 iterations)      |

### 1.2 References

- [Functional Requirements](./FUNCTIONAL_REQUIREMENTS.md)
- [Non-Functional Requirements](./NON_FUNCTIONAL_REQUIREMENTS.md)
- [Database Schema](./DATABASE_SCHEMA.md)
- [Parallel Execution Strategy](./PARALLEL_EXECUTION.md)

---

## 2. System Overview

### 2.1 Pipeline

```
Repository URL
    │
    ▼
┌─────────────────────────────────────────────┐
│  Phase 0: Docker Image Catalog (once/batch) │
│  • Fetch all ~178 official images + tags    │
│  • Cached for the entire batch run          │
├─────────────────────────────────────────────┤
│  Phase 1: Context Blueprint (per repo)      │
│                                             │
│  1. Clone repo (git clone --depth 1)        │
│  2. Collect file tree + README              │
│  3. gpt5-nano selects 3–5 build files       │
│  4. Read selected files (≤20K chars each)   │
│  5. Metaprompting LLM + image catalog       │
│     → context blueprint (incl. base image)  │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Phase 2: ReAct Agent (up to 3 iterations)  │
│                                             │
│  Per iteration (15 steps max):              │
│    Agent reasons → picks tool → gets result │
│    (tool outputs >2000 chars summarized)    │
│    Available tools:                         │
│      ReadFile, WriteFile, ListDir,          │
│      FindFiles, GrepFiles, SearchWeb,       │
│      DockerImageSearch, VerifyBuild         │
│                                             │
│  On iteration failure:                      │
│    gpt5-chat summarizes context → lessons   │
│    Lessons injected into next iteration     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  VerifyBuild (called by agent)              │
│                                             │
│  1. gpt5-nano reviews Dockerfile →          │
│     score + smoke test commands             │
│  2. If approved → docker build              │
│     • Error >2000 chars → summarize         │
│     • Error ≤2000 chars → return directly   │
│  3. If build succeeds → run smoke tests     │
│  4. If smoke tests pass → accept            │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Output                                     │
│  • Dockerfile + .dockerignore               │
│  • All steps persisted to SQL database      │
│  • Structured logs (JSON)                   │
└─────────────────────────────────────────────┘
```

### 2.2 LLM Roles

| Role                  | Model      | Purpose                                            |
|-----------------------|------------|-----------------------------------------------------|
| File selector         | gpt5-nano  | Pick 3–5 build-relevant files from tree + README    |
| Context blueprint     | gpt5-nano (metaprompt) | Produce structured repo analysis + select base image from official catalog |
| ReAct agent           | gpt5-nano  | Generate Dockerfile via tool-use loop               |
| Output summarizer     | gpt5-nano  | Compress tool outputs >2000 chars                   |
| Lesson extractor      | gpt5-chat  | Summarize failed iteration into lessons             |
| VerifyBuild reviewer  | gpt5-nano  | Score Dockerfile + design smoke tests               |
| Error summarizer      | gpt5-nano  | Compress Docker build errors >2000 chars            |

### 2.3 Operating Environment

| Component        | Requirement                                    |
|------------------|------------------------------------------------|
| Python           | 3.11+                                          |
| Docker           | Engine with BuildKit                           |
| OS               | Linux (production), macOS (development)        |
| API              | Azure OpenAI (gpt5-nano, gpt5-chat deployments)|
| Database         | SQLite (local) or PostgreSQL (production)      |
| Network          | Internet for Git, Docker Hub, Azure OpenAI     |

---

## 3. System Interfaces

### 3.1 CLI

```bash
# Single repo
python src/build_agent.py <repo_url>

# Batch (parallel)
python src/batch_runner.py <repo_list.txt> [--workers N]
```

### 3.2 External APIs

| Interface        | Purpose                                  |
|------------------|------------------------------------------|
| Azure OpenAI     | All LLM calls (agent, summarizer, etc.)  |
| GitHub (Git)     | Clone repos (depth=1)                    |
| Docker Hub       | Official image catalog fetch, image search, tag verification |
| Docker Daemon    | Build images, run containers             |
| DuckDuckGo       | Web search for error solutions           |

### 3.3 Database

All run, iteration, and step data is persisted to a SQL database. See [Database Schema](./DATABASE_SCHEMA.md).

---

## 4. Data Requirements

### 4.1 Inputs

| Data               | Format       | Source  |
|--------------------|--------------|---------|
| Repository URLs    | Text file    | User    |
| API credentials    | `.env` file  | User    |

### 4.2 Outputs

| Data                   | Format     | Destination           |
|------------------------|------------|-----------------------|
| Dockerfile             | Text       | Per-repo report dir   |
| .dockerignore          | Text       | Per-repo report dir   |
| Run/iteration/step data| SQL rows   | Database              |
| Structured logs        | JSON       | File system           |

---

## 5. Constraints

1. **API rate limits** — Azure OpenAI enforces TPM/RPM limits; parallelism must be throttled accordingly.
2. **Disk space** — Concurrent clones + Docker images can exhaust disk; cleanup is mandatory.
3. **Docker daemon** — Required for all build/verify operations; single shared resource across workers.
4. **LLM output quality** — Bounded by model capabilities; fallbacks required for malformed responses.
5. **Token budget** — Each step consumes tokens; 15 steps × 3 iterations bounds per-repo cost.
