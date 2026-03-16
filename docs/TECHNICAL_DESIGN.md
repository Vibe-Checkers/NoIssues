# Technical Design

## BuildAgent v2.0

---

## 1. Module Structure

```
src/
├── build_agent.py              # CLI entrypoint (single repo)
├── batch_runner.py             # CLI entrypoint (parallel batch)
├── agent/
│   ├── __init__.py
│   ├── llm.py                  # LLM client wrapper, rate limiter
│   ├── blueprint.py            # Phase 0+1: image catalog, context blueprint
│   ├── react_loop.py           # Phase 2: iteration/step loop
│   ├── tools.py                # Tool definitions (sandboxed)
│   ├── verify_build.py         # VerifyBuild tool implementation
│   ├── summarizer.py           # Output/error summarization (>2000 chars)
│   └── docker_ops.py           # Docker build, container run, cleanup
├── db/
│   ├── __init__.py
│   ├── schema.py               # Table creation, migrations
│   ├── models.py               # Dataclasses for batch_run, run, iteration, step, etc.
│   └── writer.py               # Thread-safe DB writer (SQLite WAL / PG pool)
└── parallel/
    ├── __init__.py
    ├── rate_limiter.py         # Global RPM+TPM token-bucket
    ├── disk_monitor.py         # Disk space backpressure
    └── worker.py               # Worker loop: clone → blueprint → agent → cleanup
```

---

## 2. Key Dataclasses

These are the in-memory representations that flow through the pipeline and get persisted to the database.

```python
@dataclass
class StepRecord:
    id: str                     # UUID
    step_number: int            # 1..15
    started_at: datetime
    finished_at: datetime | None
    duration_ms: int | None
    thought: str
    tool_name: str
    tool_input: dict
    tool_output_raw: str        # full output before summarization
    tool_output: str            # what the agent sees (summarized if >2000)
    was_summarized: bool
    prompt_tokens: int
    completion_tokens: int
    summary_prompt_tokens: int
    summary_completion_tokens: int

@dataclass
class IterationRecord:
    id: str
    iteration_number: int       # 1, 2, or 3
    status: str                 # 'success', 'failure', 'error'
    started_at: datetime
    finished_at: datetime | None
    duration_ms: int | None
    steps: list[StepRecord]
    injected_lessons: str | None
    verify_result: str | None   # 'accepted', 'rejected', 'build_failed', 'smoke_failed'
    lesson_extraction_tokens: tuple[int, int]  # (prompt, completion)

@dataclass
class RunRecord:
    id: str
    repo_url: str
    repo_slug: str
    status: str                 # 'success', 'failure', 'error', 'skipped'
    started_at: datetime
    finished_at: datetime | None
    duration_ms: int | None
    context_blueprint: str | None
    detected_language: str | None
    repo_type: str | None
    final_dockerfile: str | None
    verify_score: int | None
    smoke_test_passed: bool | None
    iterations: list[IterationRecord]
    worker_id: int

@dataclass
class VerifyBuildResult:
    status: str                 # 'accepted', 'rejected', 'build_failed', 'smoke_failed'
    review_approved: bool
    review_score: int
    review_concerns: list[str]
    smoke_test_commands: list[str]
    build_success: bool | None
    build_error: str | None     # summarized if >2000 chars
    build_error_raw: str | None
    build_duration_ms: int | None
    smoke_results: list[dict] | None  # [{command, exit_code, output, timed_out}]
    dockerfile_snapshot: str | None
    # token accounting
    review_tokens: tuple[int, int]
    error_summary_tokens: tuple[int, int]
```

---

## 3. Phase 0: Docker Official Image Catalog

Runs **once per batch**, before any worker starts.

### 3.1 Fetching

```
Docker Hub API
  GET https://hub.docker.com/v2/repositories/library/?page_size=100
  → paginate until all ~178 official images collected

For each image:
  GET https://hub.docker.com/v2/repositories/library/{name}/tags/?page_size=25&ordering=last_updated
  → collect top 25 tags with their architectures
```

### 3.2 Catalog Format

Stored as a single string injected into the metaprompting prompt. Format:

```
=== DOCKER OFFICIAL IMAGES ===
python: 3.12, 3.12-slim, 3.11, 3.11-slim, 3.12-alpine, 3.11-alpine
node: 22, 22-slim, 22-alpine, 20, 20-slim, 20-alpine
golang: 1.22, 1.22-alpine, 1.21, 1.21-alpine
rust: 1.77, 1.77-slim, 1.77-alpine
eclipse-temurin: 21, 21-jdk, 17, 17-jdk, 21-alpine, 17-alpine
gcc: 14, 13, 14-bookworm, 13-bookworm
ubuntu: 24.04, 22.04, 20.04
debian: bookworm, bookworm-slim, bullseye, bullseye-slim
alpine: 3.19, 3.18
...
(~178 images, one line each, top 6 tags)
```

Estimated size: ~5KB. Fits easily in a single prompt.

### 3.3 Caching

```python
class ImageCatalog:
    _catalog: str | None = None
    _fetched_at: datetime | None = None

    def get(self) -> str:
        if self._catalog is None:
            self._catalog = self._fetch_from_docker_hub()
            self._fetched_at = datetime.utcnow()
        return self._catalog

    def _fetch_from_docker_hub(self) -> str:
        # paginate /v2/repositories/library/, then /tags/ per image
        # format as compact "name: tag1, tag2, ..." lines
        # timeout: 60s total, skip images that fail
        ...
```

Called once in `batch_runner.py` before spawning workers. The string is passed to each worker by reference (read-only, no thread-safety concern).

---

## 4. Phase 1: Context Blueprint

### 4.1 Sequence

```
clone_repo(url)
    │
    ▼
generate_file_tree(repo_root)       ── tree output (string)
read_readme(repo_root)              ── README content (string)
    │
    ▼
LLM call: file_selector             ── gpt5-nano
  input:  file_tree + readme
  output: ["pom.xml", "src/main/java/App.java", ".github/workflows/build.yml"]
    │
    ▼
read_selected_files(paths, limit=20000)
    │
    ▼
LLM call: metaprompt_blueprint      ── gpt5-nano
  input:  file_contents + image_catalog
  output: context blueprint JSON
```

### 4.2 File Selector Prompt

```
You are analyzing a repository to determine which files are most relevant for
building and containerizing this project.

FILE TREE:
{file_tree}

README (first 5000 chars):
{readme}

Select 3-5 files that would be most useful for understanding how to build this
project in a Docker container. Prioritize: build configs, dependency manifests,
CI/CD configs, Makefiles, existing Dockerfiles.

Return ONLY a JSON array of file paths, e.g.:
["package.json", ".github/workflows/ci.yml", "tsconfig.json"]
```

### 4.3 Metaprompting Blueprint Prompt

```
You are a Docker expert analyzing a repository to produce a build blueprint.

SELECTED FILES:
--- {path1} ---
{content1}
--- {path2} ---
{content2}
...

AVAILABLE DOCKER OFFICIAL IMAGES:
{image_catalog}

Produce a JSON context blueprint:
{
  "language": "...",
  "build_system": "...",
  "package_manager": "...",
  "dependencies_summary": "...",
  "build_commands": ["..."],
  "runtime_requirements": "...",
  "repo_type": "library|cli_tool|web_service|data_pipeline|native_library|framework|desktop_app",
  "base_image": "python:3.12-slim",       ← MUST be from the official images list above
  "base_image_rationale": "...",
  "pitfalls": ["..."],
  "warnings": ["..."],
  "environment": {
    "language_version": "...",
    "system_libraries": ["..."]
  }
}

Rules for base_image:
- Pick from the official images list. Use the most specific tag (e.g., "python:3.12-slim" not "python:latest").
- Match the language version detected in the config files.
- Prefer -slim variants for smaller images unless build tools are needed.
- If the project needs a non-official image (nvidia/cuda, mcr.microsoft.com/...),
  set base_image to the closest official image and add a note in pitfalls.
```

### 4.4 Blueprint Parsing and Fallback

```python
def generate_blueprint(repo_root: str, image_catalog: str, llm: LLMClient) -> dict:
    tree = generate_file_tree(repo_root)
    readme = read_readme(repo_root)

    # Step 1: file selection
    try:
        selected = llm.call("file_selector", tree=tree, readme=readme)
        paths = json.loads(selected)
        paths = [p for p in paths if (Path(repo_root) / p).is_file()]  # validate
    except Exception:
        paths = _heuristic_file_selection(repo_root)  # fallback: known filenames

    # Step 2: read files
    contents = {}
    for p in paths[:5]:
        try:
            text = (Path(repo_root) / p).read_text(errors="replace")[:20_000]
            contents[p] = text
        except Exception:
            continue

    # Step 3: metaprompting
    try:
        raw = llm.call("metaprompt", files=contents, catalog=image_catalog)
        blueprint = json.loads(raw)
        return blueprint
    except Exception:
        # Fallback: minimal blueprint from file extensions
        lang = detect_language_by_extensions(repo_root)
        return {"language": lang, "base_image": None, "build_system": "unknown"}
```

---

## 5. Phase 2: ReAct Agent Loop

### 5.1 Architecture

```python
def run_agent(repo_root: str, blueprint: dict, max_iterations: int = 3) -> RunRecord:
    lessons = None
    for iteration_num in range(1, max_iterations + 1):
        # Clean slate
        delete_dockerfile(repo_root)

        # Build prompt
        prompt = build_prompt(blueprint, lessons)
        tools = create_tools(repo_root)

        # Run iteration
        iteration = run_iteration(prompt, tools, max_steps=15)

        # Check outcome
        if iteration.verify_result == "accepted":
            return success(iteration)

        # Extract lessons for next iteration
        lessons = extract_lessons(iteration.steps)

    return failure()
```

### 5.2 Single Iteration Loop

```python
def run_iteration(prompt: str, tools: list, max_steps: int = 15) -> IterationRecord:
    messages = [system_prompt, user_prompt(prompt)]
    steps = []

    for step_num in range(1, max_steps + 1):
        t0 = time.monotonic()

        # LLM call → thought + tool selection
        response = llm.call(messages)
        thought, tool_name, tool_input = parse_react_response(response)

        # Execute tool
        raw_output = tools[tool_name].execute(tool_input)

        # Summarize if needed
        if len(raw_output) > 2000:
            output = summarize(raw_output, tool_name)
            was_summarized = True
        else:
            output = raw_output
            was_summarized = False

        # Record step
        step = StepRecord(
            step_number=step_num,
            thought=thought,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output_raw=raw_output,
            tool_output=output,
            was_summarized=was_summarized,
            duration_ms=elapsed(t0),
            ...
        )
        steps.append(step)
        db.write_step(step)  # persist incrementally

        # Append to conversation
        messages.append(assistant_message(thought, tool_name, tool_input))
        messages.append(tool_result_message(output))

        # Check if VerifyBuild accepted
        if tool_name == "VerifyBuild" and is_accepted(output):
            return IterationRecord(status="success", steps=steps, ...)

    return IterationRecord(status="failure", steps=steps, ...)
```

### 5.3 ReAct Message Format

Each turn in the conversation follows:

```
Assistant (thought + action):
  Thought: I need to check the package.json to find build commands.
  Action: ReadFile
  Action Input: {"path": "package.json"}

Tool result:
  Observation: {"name": "express", "scripts": {"build": "tsc", ...}}

Assistant (next thought + action):
  Thought: This is a TypeScript project using tsc for build. I'll use node:22-slim...
  Action: WriteFile
  Action Input: {"path": "Dockerfile", "content": "FROM node:22-slim\n..."}
```

The `thought` and `action` are extracted from the assistant response via regex:
```python
THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:)", re.DOTALL)
ACTION_RE  = re.compile(r"Action:\s*(\w+)\s*\nAction Input:\s*(.+)", re.DOTALL)
```

This makes thoughts and actions directly extractable for database persistence.

---

## 6. VerifyBuild Implementation

### 6.1 Sequence

```
Agent calls VerifyBuild
    │
    ▼
Step 1: Read Dockerfile from disk
    │
    ▼
Step 2: LLM Review (gpt5-nano)
  input:  dockerfile content + repo_type from blueprint
  output: {approved, score, smoke_test_commands, concerns}
    │
    ├── approved=false → return {status: "rejected", review: ...}
    │
    ▼
Step 3: docker build
    │
    ├── build fails
    │     │
    │     ├── error > 2000 chars → summarize(error) → return {status: "build_failed", ...}
    │     └── error ≤ 2000 chars → return {status: "build_failed", build_error: error}
    │
    ▼
Step 4: Run smoke tests (1-3 commands)
    │
    ├── any fails → return {status: "smoke_failed", smoke_results: [...]}
    │
    ▼
Step 5: return {status: "accepted", dockerfile_snapshot: ..., review: ...}
```

### 6.2 Review Prompt

```
You are reviewing a Dockerfile for building the following repository type: {repo_type}

DOCKERFILE:
{dockerfile_content}

Evaluate:
1. Does it build the application from source? (not just install a runtime)
2. Are base images and dependencies appropriate?
3. Are there obvious errors (wrong WORKDIR, missing COPY, broken RUN)?

Return JSON:
{
  "approved": true/false,
  "score": 1-10,
  "smoke_test_commands": ["cmd1", "cmd2"],
  "concerns": ["issue1", "issue2"]
}

For smoke_test_commands, design 1-3 commands that prove the build worked:
- For libraries: import the package (e.g., python -c "import flask")
- For CLI tools: run --version or --help
- For web services: check the binary/process exists
- For compiled projects: verify artifacts exist (find /app -name "*.jar")
```

### 6.3 Error Summarization

```python
def summarize_build_error(error: str) -> tuple[str, int, int]:
    """Returns (summarized_error, prompt_tokens, completion_tokens)."""
    if len(error) <= 2000:
        return error, 0, 0

    prompt = f"""Summarize this Docker build error. Preserve:
- The exact error message and exit code
- The failing command/step
- Missing packages or files mentioned
- Any version mismatch info

ERROR OUTPUT:
{error}

Return a concise summary (under 1500 chars)."""

    response = llm.call(prompt)
    return response.content, response.prompt_tokens, response.completion_tokens
```

### 6.4 Smoke Test Execution

```python
def run_smoke_tests(image_name: str, commands: list[str], timeout: int = 30) -> list[dict]:
    results = []
    for cmd in commands:
        try:
            proc = subprocess.run(
                ["docker", "run", "--rm", "--entrypoint", "", image_name, "sh", "-c", cmd],
                capture_output=True, text=True, timeout=timeout
            )
            results.append({
                "command": cmd,
                "exit_code": proc.returncode,
                "output": (proc.stdout + proc.stderr)[:2000],
                "timed_out": False
            })
        except subprocess.TimeoutExpired:
            results.append({
                "command": cmd,
                "exit_code": -1,
                "output": f"Timed out after {timeout}s",
                "timed_out": True
            })
    return results
```

---

## 7. Summarizer Module

Shared logic used by both the agent loop (tool output summarization) and VerifyBuild (error summarization).

```python
SUMMARY_PROMPT = """Summarize the following {context_type} output.
Preserve all actionable information: error messages, file paths, version numbers,
commands, package names. Remove boilerplate, progress bars, and repeated lines.

OUTPUT ({length} chars):
{content}

Summary (under 1500 chars):"""

def summarize_output(content: str, context_type: str = "tool") -> tuple[str, int, int]:
    if len(content) <= 2000:
        return content, 0, 0
    response = llm.call(SUMMARY_PROMPT.format(
        context_type=context_type,
        length=len(content),
        content=content
    ))
    return response.content, response.prompt_tokens, response.completion_tokens
```

---

## 8. Lesson Extraction

When an iteration fails, gpt5-chat summarizes the full step history into lessons.

### 8.1 Input Construction

```python
def build_lesson_input(steps: list[StepRecord]) -> str:
    lines = []
    for s in steps:
        lines.append(f"Step {s.step_number}: {s.tool_name}")
        lines.append(f"  Thought: {s.thought[:300]}")
        lines.append(f"  Result: {s.tool_output[:300]}")
    return "\n".join(lines)
```

### 8.2 Lesson Prompt

```
You are analyzing a failed Dockerfile generation attempt. The agent had 15 steps
but could not produce a working Dockerfile.

STEP HISTORY:
{step_history}

Produce a concise lesson list for the next attempt:
1. What approaches were tried and why they failed
2. Specific errors encountered (package names, versions, commands)
3. What to try differently next time

Be specific and actionable. Max 500 words.
```

### 8.3 Model Choice

Lesson extraction uses **gpt5-chat** (not gpt5-nano) because it must compress a long context (up to 15 steps × ~600 chars = ~9K chars of step history) into a coherent summary. gpt5-chat has a larger context window and better summarization quality for this task.

---

## 9. LLM Client Wrapper

### 9.1 Unified Interface

```python
class LLMClient:
    def __init__(self, rate_limiter: GlobalRateLimiter):
        self.nano = AzureChatOpenAI(deployment="gpt5-nano", ...)
        self.chat = AzureChatOpenAI(deployment="gpt5-chat", ...)
        self.limiter = rate_limiter

    def call(self, model: str, messages: list, estimated_tokens: int = 2000) -> LLMResponse:
        self.limiter.acquire(estimated_tokens)
        try:
            client = self.nano if model == "nano" else self.chat
            response = client.invoke(messages)
            actual_tokens = response.usage.total_tokens
            self.limiter.release(actual_tokens)
            return LLMResponse(
                content=response.content,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens
            )
        except RateLimitError as e:
            self.limiter.backoff(e.retry_after)
            raise
```

### 9.2 Retry Decorator

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError))
)
def llm_call_with_retry(client, model, messages, estimated_tokens):
    return client.call(model, messages, estimated_tokens)
```

---

## 10. Tool Sandbox

### 10.1 Path Resolution

```python
def resolve_path(repo_root: Path, user_path: str) -> Path:
    resolved = (repo_root / user_path).resolve()
    if not resolved.is_relative_to(repo_root.resolve()):
        raise SecurityError(f"Path traversal blocked: {user_path}")
    return resolved
```

### 10.2 Tool Registration

Each tool is a callable with a Pydantic input schema:

```python
class ReadFileInput(BaseModel):
    path: str

class ReadFileTool:
    name = "ReadFile"
    description = "Read a file from the repository (max 512KB)"
    input_schema = ReadFileInput

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root

    def execute(self, input: ReadFileInput) -> str:
        path = resolve_path(self.repo_root, input.path)
        if path.stat().st_size > 512 * 1024:
            return f"Error: file exceeds 512KB limit ({path.stat().st_size} bytes)"
        return path.read_text(errors="replace")
```

### 10.3 WriteFile: FROM Validation

```python
class WriteFileTool:
    def execute(self, input: WriteFileInput) -> str:
        path = resolve_path(self.repo_root, input.path)
        if path.name == "Dockerfile":
            self._validate_from_lines(input.content)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(input.content)
        return f"Written {len(input.content)} bytes to {input.path}"

    def _validate_from_lines(self, content: str):
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("FROM "):
                image_ref = stripped.split()[1]
                if not self._image_exists(image_ref):
                    raise ValueError(f"Base image not found: {image_ref}")
```

---

## 11. Docker Operations

### 11.1 Build

```python
class DockerOps:
    def __init__(self, build_semaphore: threading.Semaphore, timeout: int = 600):
        self.semaphore = build_semaphore
        self.timeout = timeout

    def build(self, context_dir: str, image_name: str) -> tuple[bool, str, int]:
        """Returns (success, error_output, duration_ms)."""
        with self.semaphore:
            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    ["docker", "build", "-t", image_name, "-f", f"{context_dir}/Dockerfile", context_dir],
                    capture_output=True, text=True, timeout=self.timeout
                )
                duration = int((time.monotonic() - t0) * 1000)
                if proc.returncode == 0:
                    return True, "", duration
                error = proc.stderr or proc.stdout
                # Check transient errors
                if self._is_transient(error):
                    time.sleep(5)
                    return self._retry_build(context_dir, image_name)
                # Check cache corruption
                if self._is_cache_corrupt(error):
                    self._prune_cache()
                    return self._retry_build(context_dir, image_name)
                return False, error, duration
            except subprocess.TimeoutExpired:
                return False, f"Build timed out after {self.timeout}s", int((time.monotonic() - t0) * 1000)

    def cleanup(self, image_name: str):
        subprocess.run(["docker", "rmi", "-f", image_name], capture_output=True)
```

---

## 12. Database Writer

### 12.1 Thread-Safe Writer

```python
class DBWriter:
    def __init__(self, db_url: str):
        if db_url.startswith("sqlite"):
            self.conn = sqlite3.connect(db_url.replace("sqlite:///", ""))
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=5000")
            self.lock = threading.Lock()
        else:
            self.pool = ThreadedConnectionPool(2, 10, db_url)
            self.lock = None  # PG handles concurrency

    def write_step(self, iteration_id: str, step: StepRecord):
        sql = """INSERT INTO step (id, iteration_id, step_number, started_at, finished_at,
                 duration_ms, thought, tool_name, tool_input, tool_output_raw, tool_output,
                 was_summarized, prompt_tokens, completion_tokens,
                 summary_prompt_tokens, summary_completion_tokens)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        self._execute(sql, (
            step.id, iteration_id, step.step_number, step.started_at, step.finished_at,
            step.duration_ms, step.thought, step.tool_name, json.dumps(step.tool_input),
            step.tool_output_raw, step.tool_output, step.was_summarized,
            step.prompt_tokens, step.completion_tokens,
            step.summary_prompt_tokens, step.summary_completion_tokens
        ))

    def _execute(self, sql, params):
        if self.lock:  # SQLite
            with self.lock:
                self.conn.execute(sql, params)
                self.conn.commit()
        else:  # PG
            conn = self.pool.getconn()
            try:
                conn.cursor().execute(sql, params)
                conn.commit()
            finally:
                self.pool.putconn(conn)
```

---

## 13. Worker Lifecycle

```python
def worker_loop(worker_id: int, repo_url: str, batch_id: str,
                image_catalog: str, rate_limiter: GlobalRateLimiter,
                build_semaphore: threading.Semaphore,
                disk_monitor: DiskSpaceMonitor, db: DBWriter):
    slug = make_slug(repo_url)
    clone_dir = Path(f"workdir/{batch_id}/{worker_id}/{slug}")
    image_name = f"buildagent-{slug}-{worker_id}"
    llm = LLMClient(rate_limiter)

    run_record = RunRecord(repo_url=repo_url, repo_slug=slug, worker_id=worker_id, ...)
    db.write_run_start(run_record)

    try:
        # Disk check
        disk_monitor.check_or_wait()

        # Clone
        clone_repo(repo_url, clone_dir)

        # Phase 1: Blueprint
        blueprint = generate_blueprint(str(clone_dir), image_catalog, llm)
        run_record.context_blueprint = json.dumps(blueprint)
        run_record.detected_language = blueprint.get("language")
        run_record.repo_type = blueprint.get("repo_type")
        db.update_run_blueprint(run_record)

        # Phase 2: Agent
        docker_ops = DockerOps(build_semaphore)
        result = run_agent(str(clone_dir), blueprint, llm, docker_ops, image_name, db)

        run_record.status = result.status
        run_record.final_dockerfile = result.dockerfile
        run_record.verify_score = result.verify_score
        run_record.iterations = result.iterations

    except Exception as e:
        run_record.status = "error"
        run_record.error_message = str(e)
        logger.exception(f"[worker-{worker_id}] {slug}: {e}")
    finally:
        # Cleanup always runs
        shutil.rmtree(clone_dir, ignore_errors=True)
        docker_ops.cleanup(image_name)
        run_record.finished_at = datetime.utcnow()
        db.write_run_finish(run_record)
```

---

## 14. Batch Runner

```python
def main(repo_list: str, workers: int = 4):
    repos = read_and_validate(repo_list)

    # Phase 0: fetch image catalog once
    image_catalog = ImageCatalog().get()

    # Shared resources
    rate_limiter = GlobalRateLimiter(rpm=config.RPM, tpm=config.TPM)
    build_semaphore = threading.Semaphore(config.DOCKER_BUILD_CONCURRENCY)
    disk_monitor = DiskSpaceMonitor(threshold_gb=config.DISK_SPACE_THRESHOLD_GB)
    db = DBWriter(config.DATABASE_URL)
    db.create_tables()

    batch = BatchRun(worker_count=workers, repo_count=len(repos))
    db.write_batch_start(batch)

    # Skip already-successful repos (crash recovery)
    completed = db.get_successful_slugs()
    repos = [r for r in repos if make_slug(r) not in completed]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(worker_loop, i % workers, url, batch.id,
                       image_catalog, rate_limiter, build_semaphore,
                       disk_monitor, db): url
            for i, url in enumerate(repos)
        }
        for future in as_completed(futures):
            url = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error(f"Unhandled: {url}: {e}")

    db.write_batch_finish(batch)
    print_summary(db, batch.id)
```

---

## 15. Configuration

All configuration via environment variables (loaded from `.env`):

```
# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NANO=gpt5-nano
AZURE_OPENAI_DEPLOYMENT_CHAT=gpt5-chat

# Rate limits
AZURE_OPENAI_RPM=60
AZURE_OPENAI_TPM=80000

# Docker
DOCKER_BUILD_TIMEOUT=600
DOCKER_BUILD_CONCURRENCY=2
DOCKER_SMOKE_TIMEOUT=30
DOCKER_KEEP_STORAGE_GB=10
DOCKER_PRUNE_INTERVAL=10

# Parallel execution
WORKERS=4
DISK_SPACE_THRESHOLD_GB=5

# Database
DATABASE_URL=sqlite:///results.db

# Agent
MAX_ITERATIONS=3
MAX_STEPS_PER_ITERATION=15
SUMMARIZE_THRESHOLD=2000
```
