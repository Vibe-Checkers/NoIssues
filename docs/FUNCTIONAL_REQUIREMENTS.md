# Functional Requirements

## BuildAgent v2.0

---

## FR-1: Repository Ingestion

**FR-1.1** The system shall clone repositories via `git clone --depth 1 --recurse-submodules`.

**FR-1.2** Each repo shall be assigned a filesystem-safe slug (`owner__repo`).

**FR-1.3** The system shall validate URLs and reject duplicates before processing.

---

## FR-2: Context Blueprint (Pre-Agent Analysis)

This phase runs **before** the ReAct agent and produces a structured context blueprint that is injected into the agent's initial prompt.

**FR-2.1 File Tree + README Collection**
- The system shall generate a full file tree of the cloned repository.
- The system shall read the repository's README file (if present).

**FR-2.2 Build-File Selection (gpt5-nano)**
- The file tree and README shall be sent to a gpt5-nano call.
- The LLM shall select 3–5 files most relevant to building the project (e.g., `package.json`, `pom.xml`, `Makefile`, `Dockerfile`, CI configs).
- If the LLM returns invalid paths, the system shall skip them and proceed with valid selections.

**FR-2.3 File Content Collection**
- The system shall read each selected file, truncating content at **20,000 characters** per file.
- Binary or unreadable files shall be skipped.

**FR-2.4 Docker Official Image Catalog**
- At batch start (once per batch, not per repo), the system shall fetch the full list of Docker Official Images (~178 images) and their recent tags from Docker Hub API.
- The catalog shall be formatted as a compact list (image name + top tags), cached for the duration of the batch.
- This catalog is injected into the metaprompting call so the LLM selects a base image from real, existing images.

**FR-2.5 Metaprompting LLM (Context Blueprint Generation)**
- The content of the selected files **and** the Docker Official Image catalog shall be fed into a gpt5-nano metaprompting call.
- The LLM shall output a **context blueprint** containing:
  - Primary language and build system
  - Detected dependencies and package manager
  - Build commands inferred from config files and CI
  - Environment requirements (runtime versions, system libraries)
  - Repository type (library, CLI tool, web service, etc.)
  - Known pitfalls and warnings (e.g., postinstall hooks, native extensions)
  - **Selected base Docker image** — must be chosen from the official image catalog provided. The LLM shall pick the image:tag that best matches the repo's language, version requirements, and build needs.
- The context blueprint format shall match the existing preparation module output (language, taxonomy dimensions, CI/CD facts, manifest warnings) so that downstream consumption is unchanged.
- If the repo requires a non-official image (e.g., `nvidia/cuda`, `mcr.microsoft.com/dotnet/sdk`), the LLM may note this and the agent can use `DockerImageSearch` during its steps to find it.

**FR-2.6 Fallback**
- If any LLM call in this phase fails, the system shall proceed with a minimal context (detected language only via file-extension heuristic).

---

## FR-3: ReAct Agent (Dockerfile Generation)

**FR-3.1 Iteration and Step Model**
- The agent shall run for up to **3 iterations**.
- Each iteration allows up to **15 steps**.
- A **step** is one LLM call that produces: a thought, a tool action, and the tool's result.
- If the agent produces a valid, verified Dockerfile within an iteration, the run succeeds.
- If 15 steps are exhausted without a verified Dockerfile, the iteration fails and the next begins.

**FR-3.2 Step Data Extraction**
- Each step shall produce a structured record with easily extractable fields:
  - `thought`: the agent's reasoning text
  - `action`: tool name + input parameters
  - `result`: the tool's output (after summarization if needed)
  - `token_usage`: prompt and completion token counts
  - `duration_ms`: wall-clock time of the step
- These records shall be persisted to the SQL database (see [Database Schema](./DATABASE_SCHEMA.md)).

**FR-3.3 Tool Output Summarization**
- If any tool returns output longer than **2,000 characters**, the output shall be summarized by a gpt5-nano call before being fed back to the agent.
- The summarization prompt shall instruct the LLM to preserve actionable information (error messages, version numbers, file paths, commands).
- The original full output shall still be saved to the database for auditability.

**FR-3.4 Initial Prompt**
- The agent's first-iteration prompt shall include:
  - The context blueprint from FR-2
  - The goal: generate a Dockerfile that builds the repository from source
  - The available tools and their descriptions
- Subsequent iteration prompts shall additionally include the lesson list from the prior iteration.

**FR-3.5 Lesson Injection (Cross-Iteration)**
- When an iteration fails, the full sequence of thoughts and actions shall be summarized by a **gpt5-chat** call.
- The summary shall produce a concise lesson list: what was tried, what failed, and what to try differently.
- This lesson list shall be prepended to the next iteration's prompt.
- The Dockerfile shall be deleted between iterations to prevent stale state.

**FR-3.6 Available Tools**
- The agent shall have access to the following tools, all sandboxed to `repo_root`:

| Tool              | Description                                        |
|-------------------|----------------------------------------------------|
| `ReadFile`        | Read a file (≤512KB)                               |
| `WriteFile`       | Write a file; validates Dockerfile FROM lines       |
| `ListDirectory`   | List directory contents                            |
| `FindFiles`       | Glob pattern search                                |
| `GrepFiles`       | Content search within files                        |
| `DockerImageSearch` | Search Docker Hub, verify tags                   |
| `SearchWeb`       | DuckDuckGo search for error solutions              |
| `VerifyBuild`     | Full build + smoke test pipeline (see FR-4)        |

---

## FR-4: VerifyBuild Tool

The VerifyBuild tool is the agent's primary validation mechanism. It is called as a tool action within a step.

**FR-4.1 Dockerfile Review (gpt5-nano)**
- Input: the generated Dockerfile content.
- The LLM shall evaluate whether the Dockerfile:
  - Builds the repository's application from source (not just installing a runtime)
  - Uses appropriate base images and dependencies
  - Follows reasonable Docker practices
- The LLM shall output:
  - `approved`: boolean — whether the Dockerfile should proceed to build
  - `score`: integer 1–10
  - `smoke_test_commands`: list of 1–3 shell commands to verify the built container
  - `concerns`: list of issues found (if any)
- If `approved` is false, VerifyBuild shall return the concerns to the agent without building.

**FR-4.2 Docker Build**
- If the Dockerfile is approved, the system shall run `docker build`.
- Build timeout shall be configurable (default: 600s).
- **Error handling:**
  - If build fails and error output is **>2,000 characters**, the error shall be summarized by a gpt5-nano call before returning to the agent.
  - If error output is **≤2,000 characters**, it shall be returned directly.
- Transient network errors (DNS, timeout, 503) shall trigger one automatic retry after 5s.
- BuildKit cache corruption shall trigger a cache prune + retry.

**FR-4.3 Smoke Test Execution**
- If the build succeeds, each smoke test command from FR-4.1 shall be executed inside the container.
- Containers run ephemerally (`--rm`) with a 30s timeout.
- If all smoke tests pass (exit code 0), the Dockerfile is **accepted** and the run succeeds.
- If any smoke test fails, the failure output (summarized if >2000 chars) is returned to the agent.

**FR-4.4 VerifyBuild Output**
- VerifyBuild shall return a JSON object with:
  - `status`: `"accepted"`, `"rejected"`, `"build_failed"`, `"smoke_failed"`
  - `review`: the LLM review (score, concerns, smoke commands)
  - `build_error`: error message (if build failed), summarized if needed
  - `smoke_results`: per-command exit code and output
  - `dockerfile_snapshot`: the Dockerfile content at time of acceptance (if accepted)

---

## FR-5: Logging and Persistence

**FR-5.1** Every run, iteration, and step shall be persisted to the SQL database with full inputs, outputs, token counts, and durations. See [Database Schema](./DATABASE_SCHEMA.md).

**FR-5.2** The generated Dockerfile and .dockerignore shall be saved to a per-repo report directory.

**FR-5.3** Structured JSON logs shall be written per repository for backward compatibility with existing analysis tooling.

---

## FR-6: Parallel Execution

**FR-6.1** The system shall process multiple repositories concurrently via a configurable worker pool.

**FR-6.2** Each worker shall operate with isolated state (clone directory, Docker image names, logging context).

**FR-6.3** Resources (cloned repos, Docker images) shall be cleaned up after each run completes, even on failure.

**FR-6.4** See [Parallel Execution Strategy](./PARALLEL_EXECUTION.md) for detailed design.
