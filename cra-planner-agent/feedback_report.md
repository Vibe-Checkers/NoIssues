# Agent Feedback Report

Generated: 2026-03-12T19:49:43

## Data Summary
- Transcripts analyzed: 498
- Successes: 82
- Failures: 416
- Skipped: 3

## Failure Analysis

### Distribution
- **RATE_LIMIT**: 254
- **UNKNOWN**: 84
- **SYNTAX**: 37
- **FILE_COPY_MISSING**: 22
- **NO_SPACE**: 12
- **IMAGE_NOT_FOUND**: 4
- **COMPLIANCE**: 1

## Tool Usage

### Overall Tool Counts
- WriteToFile: 6862
- VerifyBuild: 5715
- SearchDockerError: 3574
- ListDirectory: 2737
- ReadLocalFile: 1305
- _Exception: 449
- DockerImageSearch: 155
- SearchWeb: 56
- FindFiles: 33
- GrepFiles: 4
- FetchWebPage: 4
- WriteToFile\n: 3
- ReadLocalFile;: 3
- CreateDirectoryTree: 2
- VerifyBuild\n: 1
- DockerImageSearch;: 1
- ExtractJsonField: 1

### Wasted Calls
- ListDirectory_duplicate: 1811
- ReadLocalFile_missing: 453
- WriteToFile_blocked: 367

### Avg Steps to First VerifyBuild
- success: 4.5
- failure: 4.5

## Success Patterns
- Total: 82
- First attempt: 33
- Avg steps: 26.7
- Avg tokens: 406183

### By Language
- Python: 17
- JavaScript: 7
- Go: 6
- Rust: 4
- TypeScript: 4
- C++: 3
- C: 3
- Java: 2
- TypeScript/JavaScript: 2
- PHP: 2
- unknown: 2
- Shell: 1
- TypeScript/JavaScript and Python: 1
- SCSS: 1
- None: 1
- Python, C, C++, Fortran: 1
- Ruby: 1
- C#: 1
- C/C++: 1

## Anti-Patterns
- **search_advice_ignored**: 297
- **premature_dockerfile_write**: 173
- **base_image_guessing_loop**: 62
- **repeated_search_queries**: 4

## Token Efficiency
- Avg tokens per success: 406,183
- Avg tokens per failure: 527,270
- Total tokens spent: 250,014,815
- Tokens wasted on failures: 216,707,844

## LLM: Failure Analysis
Excellent — this dataset gives us enough signal to perform a structured failure analysis.  
Below is a detailed breakdown addressing each requested item.

---

## 1. Root Cause Analysis per Failure Type

### **RATE_LIMIT (254)**
**Symptoms:** “Unknown error” or “failed to build” with no clear Docker command failure.  
**Root cause:**  
- The agent or its build orchestrator exceeded API or resource limits (e.g., Docker Hub pulls, OpenAI API calls, or CI concurrency).  
- Often these appear as transient infrastructure errors rather than logic errors in the Dockerfile.

**Underlying issue:**  
Not a content-generation failure; rather, the agent retried too aggressively or lacked backoff logic.

---

### **UNKNOWN (84)**
**Symptoms:** “Unknown error - check full log” without a clear failed command.  
**Root cause:**  
- The classifier cannot parse the failure message.  
- Could be due to truncated logs, missing error tokens, or multi-line errors not matched by regex.  
- Some may actually be runtime or syntax errors misclassified.

**Underlying issue:**  
Classifier coverage gaps; lack of structured error parsing.

---

### **SYNTAX (37)**
**Symptoms:** Dockerfile syntax invalid, or COPY/ADD commands malformed.  
**Root cause:**  
- The agent generated Dockerfiles with missing backslashes, invalid directives, or misordered instructions.  
- Sometimes caused by mixing shell syntax with Dockerfile syntax.

**Underlying issue:**  
Prompt insufficiently constrains Dockerfile formatting and validation; no syntax linting before build.

---

### **FILE_COPY_MISSING (22)**
**Symptoms:** “COPY command references missing file.”  
**Root cause:**  
- The agent assumes source files exist in the build context but they don’t.  
- Often happens when the repo structure is not fully analyzed before writing COPY commands.

**Underlying issue:**  
Agent lacks file-system awareness or pre-check for existence of referenced paths.

---

### **NO_SPACE (12)**
**Symptoms:** “failed to update builder last activity time: write /home…”  
**Root cause:**  
- Disk quota or ephemeral storage exhausted during build.  
- Large builds or multiple concurrent builds fill up temporary storage.

**Underlying issue:**  
Infrastructure resource exhaustion, not Dockerfile logic.

---

### **IMAGE_NOT_FOUND (4)**
**Symptoms:** “Failed to pull base image.”  
**Root cause:**  
- The agent references non-existent or private base images.  
- Example: `ziglang/zig:0.11.0` not available on Docker Hub.

**Underlying issue:**  
Agent does not validate image availability before using it.

---

### **COMPLIANCE (1)**
**Symptoms:** Build blocked for policy reasons (e.g., non-compliant content).  
**Root cause:**  
- Generated Dockerfile or build context violated compliance rules (e.g., license, prohibited content).  
- Rare but critical.

**Underlying issue:**  
Prompt or repo scanning did not include compliance checks.

---

## 2. Concrete Prompt Additions to Help the Agent Avoid Each Failure

| Failure Type | Prompt Additions |
|---------------|------------------|
| **RATE_LIMIT** | “If build or API requests fail due to rate limits, implement exponential backoff and retry after delay. Avoid excessive concurrent builds.” |
| **UNKNOWN** | “Always output structured error messages with clear stage and command context. When parsing logs, ensure multi-line errors are captured.” |
| **SYNTAX** | “Validate Dockerfile syntax using `dockerfile-lint` or `hadolint` before build. Ensure each instruction follows official Dockerfile grammar.” |
| **FILE_COPY_MISSING** | “Before writing COPY/ADD commands, verify that referenced files exist in the repository. If missing, skip or generate placeholder.” |
| **NO_SPACE** | “Monitor disk usage and clean intermediate images. Use smaller base images and multi-stage builds to reduce space.” |
| **IMAGE_NOT_FOUND** | “Check that base images exist and are publicly accessible before referencing them. Prefer official images from Docker Hub.” |
| **COMPLIANCE** | “Ensure generated Dockerfiles and build steps comply with open-source licensing and security policies. Avoid proprietary or restricted content.” |

---

## 3. New Failure Categories to Add to the Classifier

1. **NETWORK_TIMEOUT** – Distinguish from RATE_LIMIT; covers transient network or registry connectivity issues.  
2. **DEPENDENCY_INSTALL_FAILED** – When `apt-get`, `npm install`, etc. fail due to missing packages or version conflicts.  
3. **PERMISSION_DENIED** – When file copy or chmod fails due to permission issues in build context.  
4. **INVALID_BASE_IMAGE_TAG** – Specific subtype of IMAGE_NOT_FOUND where tag exists but is invalid or deprecated.  
5. **LINT_ERROR** – Syntax or formatting issues detected pre-build (to separate from runtime SYNTAX).  
6. **TEST_SCRIPT_FAILED** – For runtime smoke tests that fail due to incorrect entrypoint or missing dependencies.  
7. **RESOURCE_EXHAUSTION** – General category for NO_SPACE, memory, or CPU quota exceeded.

---

## 4. Whether New Tools Would Help Address Specific Failures

| Failure Type | Helpful Tools |
|---------------|---------------|
| **RATE_LIMIT** | **Yes** – Add a rate-limiting middleware or build queue manager (e.g., Redis-based throttler). |
| **UNKNOWN** | **Yes** – Structured log parser (e.g., regex-based classifier or JSON log emitter). |
| **SYNTAX** | **Yes** – Integrate `hadolint` or Dockerfile parser to validate syntax before build. |
| **FILE_COPY_MISSING** | **Yes** – File existence checker integrated with repo tree scanning. |
| **NO_SPACE** | **Yes** – Disk quota monitor and cleanup tool (e.g., `docker system prune` automation). |
| **IMAGE_NOT_FOUND** | **Yes** – Image registry validator (Docker Hub API check). |
| **COMPLIANCE** | **Yes** – License scanner (e.g., `scancode-toolkit`) and policy enforcement layer. |

---

### Summary Table

| Failure Type | Root Cause | Prompt Fix | New Tool Needed |
|---------------|-------------|-------------|-----------------|
| RATE_LIMIT | API/resource throttling | Add retry/backoff logic | ✅ |
| UNKNOWN | Unparsed logs | Structured error output | ✅ |
| SYNTAX | Invalid Dockerfile | Enforce linting | ✅ |
| FILE_COPY_MISSING | Missing files | Verify existence | ✅ |
| NO_SPACE | Disk exhaustion | Optimize builds | ✅ |
| IMAGE_NOT_FOUND | Invalid base image | Validate image | ✅ |
| COMPLIANCE | Policy violation | Add compliance checks | ✅ |

---

### Overall Recommendation

- **Classifier improvements:** Add fine-grained categories for network, dependency, and permission failures.  
- **Prompt improvements:** Explicitly instruct the agent to validate syntax, file existence, and image availability before build.  
- **Tooling:** Integrate lightweight pre-build checks (lint, file scan, registry validation) and post-build smoke tests with structured error reporting.  
- **Infrastructure:** Implement rate-limit handling and resource monitoring to reduce transient failures.

---

This structured approach should reduce the 414 total failures significantly — especially the 254 RATE_LIMIT and 84 UNKNOWN cases — by combining better prompt constraints, classifier refinement, and pre-build validation tooling.

## LLM: Success Synthesis
Here’s a structured analysis based on the successful Dockerfile-generation transcripts and the aggregate statistics you provided:

---

## **1. Common Winning Strategies (Tool Usage Patterns Leading to Success)**

Across 82 successful runs, several consistent patterns emerge:

### **A. Early Repository Inspection**
- **ListDirectory** is almost always the first step (appears in 9/10 top start sequences).
- Successful agents begin by mapping the repo structure before making any assumptions.
- This enables correct identification of language and build system (Python, Go, Rust, C, etc.).

### **B. Targeted File Reads**
- **ReadLocalFile** is used selectively—only after identifying key files like `setup.py`, `Cargo.toml`, or `Makefile`.
- Successful agents avoid reading too many files; they focus on one or two decisive ones.

### **C. Incremental Dockerfile Construction**
- **WriteToFile** often occurs twice:
  - First to create an initial Dockerfile draft.
  - Second to refine or add supporting files (`.dockerignore`, build optimizations).
- This two-stage writing pattern correlates strongly with success.

### **D. Validation and Iteration**
- **VerifyBuild** is used to confirm correctness early.
- When errors occur, successful agents use **SearchDockerError** to interpret build logs and then **WriteToFile** again to fix issues.
- This “verify → search → fix → verify” loop is a hallmark of robust agents.

### **E. Minimalism and Reproducibility**
- Successful Dockerfiles use slim base images (`python:3.11-slim`, `golang:1.20-alpine`, etc.).
- They include `.dockerignore` files to reduce context size and improve build speed.

---

## **2. Optimal Tool-Call Sequences by Project Type**

| **Project Type** | **Optimal Sequence** | **Notes** |
|------------------|----------------------|------------|
| **Python (library/framework)** | `ListDirectory → ReadLocalFile → WriteToFile → VerifyBuild → SearchDockerError → WriteToFile → VerifyBuild` | Reads `setup.py` or `pyproject.toml`; uses slim Python base; multi-stage builds for dependencies. |
| **Go (CLI tool)** | `ListDirectory → WriteToFile → WriteToFile → VerifyBuild` | Often creates `.dockerignore`; uses `golang:1.20-alpine`; single-stage build for binaries. |
| **Rust (native library)** | `ListDirectory → ReadLocalFile → WriteToFile → VerifyBuild` | Reads `Cargo.toml`; uses multi-stage builds with pinned Rust toolchain. |
| **C/C++ (library)** | `ListDirectory → ReadLocalFile → WriteToFile → VerifyBuild` | Reads `Makefile` or `configure.ac`; uses Ubuntu or Debian base; installs build-essential. |
| **Mixed JS/Python (desktop/web app)** | `ListDirectory → ReadLocalFile → WriteToFile → VerifyBuild → SearchDockerError → WriteToFile → VerifyBuild` | Two-stage build: Node.js builder + Python runtime; error search used to fix dependency issues. |

---

## **3. “Golden Example” Few-Shot Snippets for System Prompt**

These represent concise, high-quality exemplars of successful reasoning and tool usage.

---

### **Example 1: Python Library (s3tools/s3cmd)**
```
Thought: I will inspect the repository structure to determine language and packaging files.
Action: ListDirectory
Thought: I see setup.py — this is a Python package. I’ll create a lean Dockerfile using python:3.11-slim.
Action: WriteToFile (Dockerfile)
Thought: I’ll verify the build to ensure dependencies install correctly.
Action: VerifyBuild
```

---

### **Example 2: Go CLI Tool (filosottile/mkcert)**
```
Thought: I will inspect the repository to tailor the Dockerfile.
Action: ListDirectory
Thought: This is a Go CLI tool. I’ll create a multi-stage Dockerfile using golang:1.20-alpine.
Action: WriteToFile (Dockerfile)
Thought: To optimize build context, I’ll add a .dockerignore file.
Action: WriteToFile (.dockerignore)
Action: VerifyBuild
```

---

### **Example 3: Rust Library (git/git)**
```
Thought: I will inspect the repository structure to identify files and languages.
Action: ListDirectory
Thought: Cargo.toml found — this is a Rust project. I’ll read it to determine crate details.
Action: ReadLocalFile (Cargo.toml)
Thought: I’ll create a two-stage Dockerfile with a pinned Rust toolchain to build and copy artifacts.
Action: WriteToFile (Dockerfile)
Action: VerifyBuild
```

---

### **Example 4: Mixed JS/Python App (zen-browser/desktop)**
```
Thought: I will inspect the repository structure.
Action: ListDirectory
Thought: Contains both Node.js and Python components. I’ll create a two-stage Dockerfile.
Action: WriteToFile (Dockerfile)
Action: VerifyBuild
Thought: Build failed; I’ll search for Docker error logs to identify missing dependencies.
Action: SearchDockerError
Thought: I’ll fix the dependency issue and retry.
Action: WriteToFile (Dockerfile)
Action: VerifyBuild
```

---

### **Example 5: C Library (libevent/libevent)**
```
Thought: I will inspect the project layout with ListDirectory.
Action: ListDirectory
Thought: Found configure.ac — this uses autotools. I’ll read it to confirm build steps.
Action: ReadLocalFile (configure.ac)
Thought: I’ll create a multi-stage Dockerfile using Ubuntu 22.04 to build and install the library.
Action: WriteToFile (Dockerfile)
Action: VerifyBuild
```

---

## **4. Key Decision Points Differentiating Successful vs. Failing Agents**

| **Decision Point** | **Successful Behavior** | **Failing Behavior** |
|--------------------|-------------------------|----------------------|
| **Initial inspection** | Always runs `ListDirectory` before writing anything. | Skips inspection; guesses language incorrectly. |
| **File reading** | Reads only relevant files (`setup.py`, `Cargo.toml`, etc.). | Reads too many files or irrelevant ones, wasting tokens. |
| **Dockerfile writing** | Writes incrementally, often twice, refining after verification. | Writes once without validation; errors remain uncorrected. |
| **Error handling** | Uses `SearchDockerError` to interpret and fix build failures. | Ignores errors or retries blindly. |
| **Build verification** | Always runs `VerifyBuild` before finalizing. | Omits verification, leading to untested Dockerfiles. |
| **Optimization** | Adds `.dockerignore` and uses slim base images. | Uses heavy base images or omits context optimization. |

---

✅ **Summary Insight:**  
Successful Dockerfile-generation agents follow a disciplined, inspection-first, iterative refinement workflow. They use minimal, targeted tool calls, validate early, and fix issues intelligently using error search and re-write cycles. The most predictive success marker is the presence of both `VerifyBuild` and `SearchDockerError` in the sequence, indicating active debugging and validation.

## LLM: Prompt Improvement Suggestions
Excellent — we have enough telemetry to make targeted, diff‑ready improvements.  
Below are **specific additions and modifications** to improve compliance, reduce anti‑patterns, and increase success rate.

---

## 1️⃣  New Rules to Add to **ABSOLUTE RULES**

**Rationale:**  
Failures show repeated violations of search advice, premature writes, and guessing base images.  
We’ll add explicit enforcement and cooldowns.

```diff
═══════════════════════════════════════════════════════════════════════════════
ABSOLUTE RULES:
═══════════════════════════════════════════════════════════════════════════════
+ 11. You MUST NOT ignore or override SearchDockerError advice. If advice conflicts with your prior plan, follow the advice exactly and note the conflict in context.
+ 12. After any VerifyBuild failure, you MUST wait for SearchDockerError results before any WriteToFile. Do not emit WriteToFile until SearchDockerError has completed.
+ 13. You MUST NOT emit multiple SearchDockerError calls for the same error text unless the previous advice failed twice. Repeated identical searches are forbidden.
+ 14. You MUST NOT guess or change the base image unless explicitly instructed by SearchDockerError or VerifyBuild output.
+ 15. You MUST include a COPY validation step: before writing Dockerfile, confirm all source paths exist via ListDirectory results.
+ 16. If rate‑limit or UNKNOWN errors occur, pause and retry once with minimal context; do not abandon workflow.
```

---

## 2️⃣  New Entries for **COMMON FIX PATTERNS**

**Rationale:**  
We can codify recurring fixes for top failure types (syntax, missing COPY, image not found, no space).

```diff
═══════════════════════════════════════════════════════════════════════════════
COMMON FIX PATTERNS:
═══════════════════════════════════════════════════════════════════════════════
+ [SYNTAX]
+   - Re‑read Dockerfile line numbers from error log.
+   - Ensure each RUN, COPY, and CMD line uses valid Docker syntax.
+   - Validate JSON formatting in WriteToFile action before sending.

+ [FILE_COPY_MISSING]
+   - Re‑run ListDirectory to confirm missing file paths.
+   - Add COPY instructions only for files that exist.
+   - If build context missing, add .dockerignore excluding node_modules, not source files.

+ [IMAGE_NOT_FOUND]
+   - Use SearchDockerError to confirm correct base image tag.
+   - Do NOT guess; prefer official image from Docker Hub or verified registry.

+ [NO_SPACE]
+   - Add cleanup steps (e.g., `RUN apt-get clean && rm -rf /var/lib/apt/lists/*`).
+   - Reduce intermediate layers; combine RUN commands.

+ [RATE_LIMIT]
+   - Implement exponential backoff (wait 10s, retry once).
+   - Avoid repeated identical SearchDockerError queries.

+ [UNKNOWN]
+   - Capture full VerifyBuild output and re‑run SearchDockerError with expanded context.
+   - Do not proceed until SearchDockerError returns actionable advice.
```

---

## 3️⃣  Workflow Phase Modifications

**Rationale:**  
Premature writes and ignored search advice show the need for explicit gating and cooldown logic.

```diff
═══════════════════════════════════════════════════════════════════════════════
CRITICAL WORKFLOW - YOU MUST FOLLOW THIS EXACTLY:
═══════════════════════════════════════════════════════════════════════════════

PHASE 1 - ANALYZE:
  1. ListDirectory to see project structure
  2. ReadLocalFile to inspect key manifest files (package.json, requirements.txt, pom.xml, etc.)
  3. Identify language, framework, and dependencies
  4. Confirm all required source files exist before proceeding to Dockerfile creation
+ 4.1. Validate that all COPY source paths are present using ListDirectory results.

PHASE 2 - CREATE:
  5. WriteToFile to create Dockerfile (only after PHASE 1 is complete)
  6. WriteToFile to create .dockerignore
+ 6.1. Before WriteToFile, perform syntax self‑check (no missing quotes, valid Docker instructions).

PHASE 3 - VERIFY (MANDATORY):
  7. VerifyBuild to test the Dockerfile
+ 7.1. Capture full VerifyBuild output for later SearchDockerError context.

PHASE 4 - IF BUILD FAILS (MANDATORY LOOP):
  8. Read the error message carefully
  9. IMMEDIATELY use SearchDockerError(error_keywords="...", agent_context="...") to get a fix
  10. Do NOT guess or try to fix it yourself without searching first
  11. Apply the fix from the AI analysis
  12. VerifyBuild again
+ 12.1. If same error repeats twice, expand SearchDockerError context instead of guessing.
+ 12.2. If rate‑limit or UNKNOWN errors occur, retry once with minimal context before escalation.
```

---

## 4️⃣  Summary of **Exact Diff‑Ready Snippets**

Below is the consolidated diff block you can apply directly to the system prompt:

```diff
@@ ABSOLUTE RULES @@
+ 11. You MUST NOT ignore or override SearchDockerError advice. If advice conflicts with your prior plan, follow the advice exactly and note the conflict in context.
+ 12. After any VerifyBuild failure, you MUST wait for SearchDockerError results before any WriteToFile. Do not emit WriteToFile until SearchDockerError has completed.
+ 13. You MUST NOT emit multiple SearchDockerError calls for the same error text unless the previous advice failed twice. Repeated identical searches are forbidden.
+ 14. You MUST NOT guess or change the base image unless explicitly instructed by SearchDockerError or VerifyBuild output.
+ 15. You MUST include a COPY validation step: before writing Dockerfile, confirm all source paths exist via ListDirectory results.
+ 16. If rate‑limit or UNKNOWN errors occur, pause and retry once with minimal context; do not abandon workflow.

@@ COMMON FIX PATTERNS @@
+ [SYNTAX]
+   - Re‑read Dockerfile line numbers from error log.
+   - Ensure each RUN, COPY, and CMD line uses valid Docker syntax.
+   - Validate JSON formatting in WriteToFile action before sending.
+
+ [FILE_COPY_MISSING]
+   - Re‑run ListDirectory to confirm missing file paths.
+   - Add COPY instructions only for files that exist.
+   - If build context missing, add .dockerignore excluding node_modules, not source files.
+
+ [IMAGE_NOT_FOUND]
+   - Use SearchDockerError to confirm correct base image tag.
+   - Do NOT guess; prefer official image from Docker Hub or verified registry.
+
+ [NO_SPACE]
+   - Add cleanup steps (e.g., `RUN apt-get clean && rm -rf /var/lib/apt/lists/*`).
+   - Reduce intermediate layers; combine RUN commands.
+
+ [RATE_LIMIT]
+   - Implement exponential backoff (wait 10s, retry once).
+   - Avoid repeated identical SearchDockerError queries.
+
+ [UNKNOWN]
+   - Capture full VerifyBuild output and re‑run SearchDockerError with expanded context.
+   - Do not proceed until SearchDockerError returns actionable advice.

@@ WORKFLOW PHASES @@
+ 4.1. Validate that all COPY source paths are present using ListDirectory results.
+ 6.1. Before WriteToFile, perform syntax self‑check (no missing quotes, valid Docker instructions).
+ 7.1. Capture full VerifyBuild output for later SearchDockerError context.
+ 12.1. If same error repeats twice, expand SearchDockerError context instead of guessing.
+ 12.2. If rate‑limit or UNKNOWN errors occur, retry once with minimal context before escalation.
```

---

### ✅ Expected Outcomes

- **Reduce “search_advice_ignored”** by enforcing rule 11–12  
- **Eliminate “premature_dockerfile_write”** via 4.1 and 6.1 gating  
- **Prevent “base_image_guessing_loop”** via rule 14  
- **Lower “RATE_LIMIT” and “UNKNOWN”** via retry logic (rule 16, pattern additions)  
- **Increase first‑attempt success** through syntax self‑check and COPY validation  

These changes are minimal, diff‑ready, and empirically targeted to the observed failure clusters.

## LLM: Tool Gap Analysis
Excellent — this is exactly the kind of data that helps refine an AI coding agent’s tool ecosystem.  
Let’s analyze the failure and wasted-call data systematically, then propose **new tools**, **enhancements**, and **description improvements**.

---

## 1. Diagnosis

### Wasted Tool Calls
| Pattern | Root Cause | Implication |
|----------|-------------|--------------|
| `search_advice_ignored` (297) | Agent performs searches but doesn’t integrate results or misinterprets them. | Tool output format or guidance integration unclear. |
| `premature_dockerfile_write` (173) | Agent writes Dockerfile before dependencies or base image are known. | Missing dependency resolution or planning stage. |
| `base_image_guessing_loop` (62) | Agent repeatedly guesses base image instead of querying registry or metadata. | Lack of a “base image resolver” tool. |
| `repeated_search_queries` (4) | Agent reissues same query. | No caching or query deduplication. |

### Failure Types
| Type | Root Cause | Possible Tool Gap |
|------|-------------|-------------------|
| `RATE_LIMIT` (254) | External API throttling. | Need rate-aware batching or caching tool. |
| `UNKNOWN` (84) | Poor error surface or logging. | Need diagnostic/error introspection tool. |
| `SYNTAX` (37) | Generated code invalid. | Need syntax validation or “lint before write” tool. |
| `FILE_COPY_MISSING` (22) | Agent assumes file exists but didn’t copy. | Need file existence verification or dependency graph. |
| `NO_SPACE` (12) | Disk quota exceeded. | Need resource monitor tool. |
| `IMAGE_NOT_FOUND` (4) | Docker base image missing. | Need registry query or fallback mechanism. |
| `COMPLIANCE` (1) | Probably content policy violation. | Tool description improvement to clarify constraints. |

---

## 2. New Tools to Add

### 🧩 **Tool: `plan_build_context`**
**Purpose:** Generate a structured plan before writing Dockerfiles or build scripts.  
**Inputs:** Project files, dependency list.  
**Outputs:** Ordered build steps, required base image, estimated space usage.  
**Benefits:** Prevents premature Dockerfile writes and missing file copies.

---

### 🔍 **Tool: `base_image_resolver`**
**Purpose:** Query Docker registries or known image metadata to find the correct base image.  
**Inputs:** Language/runtime, version constraints.  
**Outputs:** Valid image name + digest.  
**Benefits:** Eliminates “base_image_guessing_loop” and “IMAGE_NOT_FOUND”.

---

### 🧠 **Tool: `search_result_integrator`**
**Purpose:** Convert search results into actionable summaries or structured data.  
**Inputs:** Search results.  
**Outputs:** Key insights, recommended next actions.  
**Benefits:** Reduces “search_advice_ignored”.

---

### 🧾 **Tool: `syntax_validator`**
**Purpose:** Validate code snippets before writing to disk.  
**Inputs:** Code text, language identifier.  
**Outputs:** Syntax errors, fix suggestions.  
**Benefits:** Reduces “SYNTAX” failures.

---

### 💾 **Tool: `resource_monitor`**
**Purpose:** Check disk space, memory, and quota before large writes or builds.  
**Inputs:** None or target directory.  
**Outputs:** Available space, warnings.  
**Benefits:** Prevents “NO_SPACE” failures.

---

### 🧰 **Tool: `file_dependency_checker`**
**Purpose:** Verify required files exist before build or copy operations.  
**Inputs:** File list or manifest.  
**Outputs:** Missing files report.  
**Benefits:** Prevents “FILE_COPY_MISSING”.

---

### 🕵️ **Tool: `error_introspector`**
**Purpose:** Capture and classify unknown errors.  
**Inputs:** Raw error logs.  
**Outputs:** Structured error type, probable cause, suggested recovery.  
**Benefits:** Reduces “UNKNOWN” category.

---

### ⚙️ **Tool: `rate_limit_manager`**
**Purpose:** Handle API rate limits via caching, batching, or backoff.  
**Inputs:** API endpoint, request frequency.  
**Outputs:** Delay schedule or cached results.  
**Benefits:** Reduces “RATE_LIMIT” failures.

---

## 3. Enhancements to Existing Tools

| Existing Tool | Enhancement | Rationale |
|----------------|--------------|------------|
| **Search Tool** | Add caching + semantic deduplication. | Prevent repeated queries. |
| **Dockerfile Writer** | Require validated build plan before execution. | Prevent premature writes. |
| **File Copy Tool** | Integrate with `file_dependency_checker`. | Avoid missing files. |
| **Build Tool** | Integrate resource monitor + syntax validator. | Prevent NO_SPACE and SYNTAX errors. |
| **Logging/Error Handling** | Include structured error codes + context. | Reduce UNKNOWN failures. |

---

## 4. Tool Description Improvements

1. **Explicit Preconditions:**  
   Each tool description should list required context (e.g., “Must have validated build plan before calling”).

2. **Structured Outputs:**  
   Use JSON schemas for all tool outputs so agents can parse and act on them deterministically.

3. **Failure Recovery Guidance:**  
   Include “If this fails, try…” section in tool docs to guide fallback behavior.

4. **Rate Limit Awareness:**  
   Add “expected latency and quota” metadata to tools that hit external APIs.

5. **Compliance Metadata:**  
   Add “content policy boundaries” to prevent misuse (e.g., clarify that copyrighted text cannot be reproduced).

---

## 5. Summary of Expected Impact

| Improvement | Expected Reduction |
|--------------|--------------------|
| `search_result_integrator` | ↓ 80% of `search_advice_ignored` |
| `plan_build_context` | ↓ 90% of `premature_dockerfile_write` |
| `base_image_resolver` | ↓ 100% of `base_image_guessing_loop` + `IMAGE_NOT_FOUND` |
| `syntax_validator` | ↓ 90% of `SYNTAX` |
| `resource_monitor` | ↓ 100% of `NO_SPACE` |
| `file_dependency_checker` | ↓ 100% of `FILE_COPY_MISSING` |
| `rate_limit_manager` | ↓ 80% of `RATE_LIMIT` |
| `error_introspector` | ↓ 70% of `UNKNOWN` |

---

**In short:**  
Add planning, validation, and introspection tools; enhance existing ones with caching and precondition checks; and rewrite tool descriptions to make dependencies, outputs, and compliance boundaries explicit.  
This will drastically reduce wasted calls and failure rates while improving agent reliability and interpretability.

## LLM: Metaprompt Improvement
Excellent — you’ve got a strong, structured metaprompt, but the failure-rate data shows where reasoning depth and specificity are insufficient.  
Below are **targeted improvements** grouped by metaprompt section and driven by observed failure patterns.

---

## 1. Introduce Failure-Aware Reasoning Control
**Problem:** The current template only adjusts reasoning depth in the last section (“FAILURE-WEIGHTED PRIORITY”). That’s too late — the agent needs to reason differently *throughout* the prompt.

**Improvement:**
Add a global directive near the top:

> **FAILURE-AWARE DIRECTIVE:**  
> Before generating outputs, dynamically weight reasoning depth per dimension according to historical failure rates.  
> - If domain or build_tool failure rate > 80%, expand reasoning for that dimension by adding explicit command examples, fallback logic, and cross-checks.  
> - If < 50%, keep concise.  
> - Always surface uncertainty explicitly (“if version unknown, fallback to X”).

This ensures the agent adapts its reasoning early, not just at the end.

---

## 2. Add a “FAILURE PROFILE SUMMARY” Section
**Problem:** The agent doesn’t contextualize which dimensions are high-risk before generating recommendations.

**Improvement:**
Insert a new section **before “RECOMMENDED BASE IMAGE”**:

```
FAILURE PROFILE SUMMARY:
<list domain and build_tool failure rates, classify as LOW/MEDIUM/HIGH risk>
<state which dimensions require expanded reasoning and validation>
```

This primes the agent to allocate attention proportionally.

---

## 3. Strengthen Cross-Dimension Consistency Logic
**Problem:** “CROSS-DIMENSION CONSISTENCY CHECK” is too qualitative; failures often stem from mismatched OS base + build tool + dependency manager.

**Improvement:**
Require explicit compatibility matrix reasoning:

```
CROSS-DIMENSION CONSISTENCY CHECK:
<tabulate base image vs build_tool vs dependency manager compatibility>
<flag known incompatibilities (e.g., Alpine + glibc-dependent tools)>
<recommend corrective substitutions>
```

---

## 4. Expand “BUILD TOOL” Dimension Guidance
**Problem:** High failure rates for build tools with complex dependency graphs (Gradle, Maven, npm, cargo, cmake).

**Improvement:**
Add sub-directives under “BUILD_TOOL”:

- Require the agent to output **tool-specific canonical install patterns** (e.g., Gradle wrapper vs system install).
- Include **cache layer optimization hints** (npm cache, cargo target dir, Maven local repo).
- Mandate **version fallback hierarchy**: CI config → lockfile → manifest → latest stable.

---

## 5. Add “VALIDATION CHECKPOINTS” Section
**Problem:** Many failures are due to missing verification of assumptions.

**Improvement:**
Add a new section after “CRITICAL WARNINGS”:

```
VALIDATION CHECKPOINTS:
<list concrete checks the Dockerfile-writing agent should perform to verify correctness>
- Confirm tool versions installed match inferred versions.
- Validate lockfile presence and integrity.
- Test ENTRYPOINT command resolves correctly.
```

This converts reasoning into actionable QA steps.

---

## 6. Improve “ENVIRONMENT_SPECIFICITY” Handling
**Problem:** High failure rates in domains like machine-learning, data-science, and systems-programming often stem from incorrect base image choice or missing OS packages.

**Improvement:**
Require:
- Explicit justification for OS family selection (glibc vs musl).
- Domain-specific package heuristics (e.g., ML → CUDA/cuDNN; systems → build-essential).
- Fallback OS selection rule: prefer Debian-based if toolchain complexity > 2 layers.

---

## 7. Add “DOMAIN-SPECIFIC HEURISTICS” Section
**Problem:** Domain-level failures (utilities, web-dev, mobile, devops) show the agent lacks domain heuristics.

**Improvement:**
Insert after “FAILURE PROFILE SUMMARY”:

```
DOMAIN-SPECIFIC HEURISTICS:
<for high-failure domains, list known Dockerfile patterns>
- web-development: node-based multi-stage with static asset copy
- mobile-development: Android SDK layers, emulator dependencies
- devops: include CLI tools (kubectl, helm)
```

---

## 8. Clarify “FAILURE-WEIGHTED PRIORITY” Behavior
**Problem:** Currently vague; doesn’t specify how to “increase reasoning depth.”

**Improvement:**
Define explicit scaling rules:

```
FAILURE-WEIGHTED PRIORITY:
If domain failure rate > 80% → add 2 extra reasoning layers per affected section.
If build_tool failure rate > 80% → include explicit install + verification commands.
If both > 80% → trigger “deep diagnostic mode”: output extended compatibility matrix and fallback logic.
```

---

## 9. Add “CONFIDENCE SCORING” Across Sections
**Problem:** The agent doesn’t quantify uncertainty.

**Improvement:**
After each major section, append:

```
CONFIDENCE SCORE: <High/Medium/Low> (based on data completeness and failure history)
```

This helps downstream agents prioritize human review.

---

## 10. Add “EXEMPLAR SNIPPET” for High-Failure Cases
**Problem:** Agents fail to translate reasoning into Dockerfile syntax.

**Improvement:**
For domains/tools with >80% failure rate, require a short, sanitized Dockerfile snippet illustrating the pattern (no copyrighted material, just generic syntax).

```
EXEMPLAR SNIPPET (for high-failure domain/tool):
<minimal Dockerfile fragment showing correct stage structure and install pattern>
```

---

## 11. Meta-Structural Improvement
**Problem:** The template is long but linear; reasoning could benefit from modular sub-prompts.

**Improvement:**
Group dimensions into three reasoning clusters:

- **Image Construction Cluster:** DOMAIN, BUILD_TOOL, ENVIRONMENT_SPECIFICITY  
- **Dependency Cluster:** DEPENDENCY_TRANSPARENCY, TOOLING_COMPLEXITY  
- **Reproducibility Cluster:** AUTOMATION_LEVEL, REPRODUCIBILITY_SUPPORT  

Add directive:

> “For each cluster, perform internal consistency validation before generating final sections.”

This reduces cross-dimension drift.

---

## 12. Add “FAILURE-DRIVEN EXPLANATION DEPTH” Table
**Problem:** The agent doesn’t know how much to elaborate per dimension.

**Improvement:**
Append a table mapping failure rate ranges to explanation depth:

| Failure Rate | Reasoning Depth | Example Output Additions |
|---------------|----------------|---------------------------|
| <50% | Concise | 1-line summary |
| 50–80% | Moderate | Include fallback logic |
| >80% | Deep | Include explicit commands, validation checkpoints, compatibility matrix |

---

## 13. Add “POST-GENERATION SELF-CHECK”
**Problem:** No self-validation loop.

**Improvement:**
End metaprompt with:

```
SELF-CHECK:
Before finalizing, verify:
- Each section populated.
- All high-failure dimensions received expanded reasoning.
- No contradictions between base image and build tool.
```

---

### Summary of Key Additions
| Category | New Section / Directive | Purpose |
|-----------|------------------------|----------|
| Global | FAILURE-AWARE DIRECTIVE | Dynamic reasoning depth |
| Context | FAILURE PROFILE SUMMARY | Risk awareness |
| Domain | DOMAIN-SPECIFIC HEURISTICS | Pattern injection |
| Validation | VALIDATION CHECKPOINTS | QA enforcement |
| Confidence | CONFIDENCE SCORE | Quantify uncertainty |
| Example | EXEMPLAR SNIPPET | Concrete guidance |
| Structure | Clustered reasoning + Self-check | Reduce drift |

---

### Example of Revised Metaprompt Opening

```
You are an expert DevOps architect specializing in Docker containerization.

FAILURE-AWARE DIRECTIVE:
Adjust reasoning depth per dimension according to historical failure rates.
Use expanded reasoning for domains/tools with >80% failure rate.

FAILURE PROFILE SUMMARY:
Domain: <domain> (failure rate X/Y → HIGH/MEDIUM/LOW)
Build Tool: <tool> (failure rate X/Y → HIGH/MEDIUM/LOW)
Dimensions requiring expanded reasoning: <list>

DOMAIN-SPECIFIC HEURISTICS:
<insert known Dockerfile patterns for high-failure domains>

Then proceed with the standard output format...
```

---

By integrating these improvements, the metaprompt becomes **failure-adaptive, self-validating, and domain-aware**, significantly reducing misgeneration rates across high-risk dimensions.