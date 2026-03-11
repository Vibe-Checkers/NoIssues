# Agent Feedback Report

Generated: 2026-03-11T12:06:41

## Data Summary
- Transcripts analyzed: 498
- Successes: 82
- Failures: 416
- Skipped: 3

## Failure Analysis

### Distribution
- **RATE_LIMIT**: 231
- **UNKNOWN**: 101
- **SYNTAX**: 38
- **FILE_COPY_MISSING**: 25
- **NO_SPACE**: 12
- **IMAGE_NOT_FOUND**: 4
- **TIMEOUT**: 2
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
- First attempt: 24
- Avg steps: 26.7
- Avg tokens: 406183

### By Language
- Python: 12
- Rust: 4
- JavaScript: 4
- TypeScript: 4
- C++: 3
- C: 3
- PHP: 2
- unknown: 2
- TypeScript/JavaScript: 1
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
Excellent — this dataset gives us enough signal to do a structured failure analysis.  
Below is a full breakdown addressing each requested point.

---

## 1. Root Cause Analysis per Failure Type

### **RATE_LIMIT (231)**
**Symptoms:** “Unknown error,” “failed to build,” intermittent success across retries.  
**Root Cause:** The agent or its Docker build environment is hitting API or registry rate limits (e.g., GitHub, npm, Docker Hub). Repeated requests for dependency downloads or image pulls exceed quota.  
**Contributing factors:**
- No caching or throttling logic.
- Parallel builds amplify request volume.
- Lack of retry/backoff strategy.

---

### **UNKNOWN (101)**
**Symptoms:** “Unknown error - check full log,” inconsistent stage reporting.  
**Root Cause:** Unclassified build errors—often due to missing error parsing logic or incomplete log capture.  
**Contributing factors:**
- Log parser fails to map error text to known categories.
- Dockerfile syntax or runtime exceptions not matched to classifier patterns.
- Some may overlap with FILE_COPY_MISSING or NO_SPACE but go unrecognized.

---

### **SYNTAX (38)**
**Symptoms:** “failed to read dockerfile,” “invalid instruction,” “COPY command references missing file.”  
**Root Cause:** Generated Dockerfiles contain invalid syntax or misplaced directives.  
**Contributing factors:**
- Agent lacks robust Dockerfile linting before execution.
- Misinterpretation of project structure (e.g., copying non-existent paths).
- Missing newline or quoting errors.

---

### **FILE_COPY_MISSING (25)**
**Symptoms:** “COPY command references missing file,” “failed to compute cache key.”  
**Root Cause:** The agent assumes certain files exist (e.g., `package.json`, `requirements.txt`) but they’re absent in repo.  
**Contributing factors:**
- Insufficient repo inspection before writing COPY commands.
- No fallback logic when expected files are missing.

---

### **NO_SPACE (12)**
**Symptoms:** “failed to update builder last activity time,” “write /home: no space left on device.”  
**Root Cause:** Disk exhaustion in build environment.  
**Contributing factors:**
- Large builds (e.g., TensorFlow, privacy libraries).
- No cleanup between builds.
- Insufficient disk quota monitoring.

---

### **IMAGE_NOT_FOUND (4)**
**Symptoms:** “Failed to pull base image,” “repository does not exist.”  
**Root Cause:** Agent references non-existent or private base images.  
**Contributing factors:**
- Outdated image tags.
- Incorrect registry path.
- No validation of image availability before use.

---

### **TIMEOUT (2)**
**Symptoms:** “Docker build exceeded 600s timeout.”  
**Root Cause:** Long-running builds (e.g., compiling large codebases) exceed allowed time.  
**Contributing factors:**
- No incremental build caching.
- Missing precompiled dependencies.

---

### **COMPLIANCE (1)**
**Symptoms:** “Unknown error,” likely policy violation (e.g., license, security).  
**Root Cause:** Generated Dockerfile violates compliance rules (e.g., downloading from untrusted sources).  
**Contributing factors:**
- No compliance check before build.
- Insufficient prompt constraints.

---

## 2. Concrete Prompt Additions to Help the Agent Avoid Each Failure

| Failure Type | Prompt Additions |
|---------------|------------------|
| **RATE_LIMIT** | “Minimize external network calls. Use official base images and package managers efficiently. Avoid unnecessary downloads; prefer cached layers.” |
| **UNKNOWN** | “If an error occurs, classify it based on Docker build stage (syntax, file copy, image pull, runtime). Always include explicit error messages in output.” |
| **SYNTAX** | “Validate Dockerfile syntax before output. Ensure each directive (FROM, COPY, RUN, CMD) follows Dockerfile specification.” |
| **FILE_COPY_MISSING** | “Inspect repository contents before writing COPY commands. Only reference files that exist. If missing, skip COPY or add conditional logic.” |
| **NO_SPACE** | “Prefer lightweight base images and clean intermediate build artifacts. Use multi-stage builds to reduce disk usage.” |
| **IMAGE_NOT_FOUND** | “Verify that base images exist on Docker Hub or are public. Avoid custom or private image references.” |
| **TIMEOUT** | “Optimize build steps for speed. Use prebuilt binaries or smaller dependency sets. Avoid long compilation tasks.” |
| **COMPLIANCE** | “Use only official, trusted sources and open-source dependencies. Avoid downloading from unknown URLs.” |

---

## 3. New Failure Categories to Add to the Classifier

| New Category | Description | Example Trigger |
|---------------|--------------|----------------|
| **DEPENDENCY_RESOLUTION_FAILED** | Build fails because package manager cannot find or install dependencies. | npm install / pip install errors |
| **NETWORK_CONNECTIVITY** | Transient network errors not due to rate limits. | “connection reset by peer,” “temporary failure in name resolution” |
| **PERMISSION_DENIED** | Build fails due to file permission or access issues. | “chmod: operation not permitted,” “permission denied” |
| **CONFIGURATION_ERROR** | Misconfigured environment variables or build args. | “invalid ARG,” “missing ENV variable” |
| **RUNTIME_PORT_CONFLICT** | Container runs but fails smoke test due to port binding issues. | “address already in use” |
| **RESOURCE_LIMIT** | CPU/memory exhaustion distinct from NO_SPACE. | “killed: out of memory” |

---

## 4. Whether New Tools Would Help Address Specific Failures

| Failure Type | Helpful Tools | Rationale |
|---------------|----------------|------------|
| **RATE_LIMIT** | Caching proxy or dependency mirror; retry/backoff library | Reduces API calls and mitigates throttling |
| **UNKNOWN** | Enhanced log parser / structured error classifier | Improves error categorization |
| **SYNTAX** | Dockerfile linter (e.g., `hadolint`) | Detects syntax errors before build |
| **FILE_COPY_MISSING** | Repo file inspector tool | Confirms file existence before COPY |
| **NO_SPACE** | Disk quota monitor / cleanup script | Prevents disk exhaustion |
| **IMAGE_NOT_FOUND** | Image registry validator | Checks image availability before build |
| **TIMEOUT** | Build profiler / caching system | Identifies slow steps and reuses layers |
| **COMPLIANCE** | Policy scanner (e.g., Open Policy Agent) | Ensures compliance before build |
| **New Categories** | Dependency resolver, network monitor | Improves robustness and classification accuracy |

---

### Summary

- **Primary systemic issue:** RATE_LIMIT (56%) — infrastructure-level, not agent logic.
- **Second major issue:** UNKNOWN (24%) — classifier and logging gaps.
- **Agent-level improvements:** Syntax validation, file existence checks, image verification.
- **Classifier expansion:** Add 6 new categories for better granularity.
- **Tooling recommendation:** Introduce linting, caching, and structured error parsing to reduce both agent and environment failures.

---

Would you like me to draft example *prompt templates* incorporating these additions (e.g., “Dockerfile-generation prompt v2”)? That would make the improvements directly usable.

## LLM: Success Synthesis
Here’s a structured analysis of the successful Dockerfile-generation transcripts you provided:

---

## 1. Common Winning Strategies (Tool Usage Patterns)

Across 82 successful runs, several consistent strategies emerge:

### **A. Early Repository Inspection**
- **Tool:** `ListDirectory`
- **Purpose:** Determine project structure, language, and build system before writing anything.
- **Pattern:** Always used as the first or second step in successful runs.
- **Effect:** Reduces guesswork and prevents incorrect base image selection.

### **B. Targeted File Reading**
- **Tool:** `ReadLocalFile`
- **Purpose:** Inspect key configuration files (`setup.py`, `Cargo.toml`, `Makefile`, `package.json`) to infer build instructions.
- **Effect:** Leads to more accurate Dockerfile commands (e.g., `pip install .`, `cargo build --release`).

### **C. Incremental Dockerfile Construction**
- **Tool:** `WriteToFile`
- **Pattern:** Often used twice — first to create a draft Dockerfile, then to refine or add `.dockerignore`.
- **Effect:** Allows iterative improvement and optimization (e.g., slimming images, adding multi-stage builds).

### **D. Validation and Iteration**
- **Tool:** `VerifyBuild`
- **Purpose:** Test the Dockerfile by performing a real or simulated build.
- **Effect:** Confirms correctness and triggers refinement if errors occur.

### **E. Error Recovery and Refinement**
- **Tool:** `SearchDockerError`
- **Purpose:** Parse build logs for specific failure causes.
- **Effect:** Enables targeted fixes (e.g., missing dependencies, incorrect COPY paths).

---

## 2. Optimal Tool-Call Sequences by Project Type

| **Project Type** | **Typical Sequence** | **Notes / Rationale** |
|------------------|----------------------|------------------------|
| **Python (library/framework)** | `ListDirectory → ReadLocalFile → WriteToFile → VerifyBuild → SearchDockerError → WriteToFile → VerifyBuild` | Reads `setup.py` or `pyproject.toml`; uses slim Python base; adds `.dockerignore` after first build. |
| **Rust (native library)** | `ListDirectory → ReadLocalFile → WriteToFile → VerifyBuild` | Reads `Cargo.toml`; uses multi-stage build with pinned Rust toolchain. |
| **C/C++ (autotools or cmake)** | `ListDirectory → WriteToFile → WriteToFile → VerifyBuild → SearchDockerError → WriteToFile → VerifyBuild` | Starts with Ubuntu base; adds build tools; refines after error search. |
| **Node.js / Web** | `ListDirectory → ReadLocalFile → WriteToFile → VerifyBuild → SearchDockerError → VerifyBuild` | Reads `package.json`; uses multi-stage build (builder + runtime). |
| **Simple CLI tools (e.g., s3cmd)** | `ListDirectory → WriteToFile → VerifyBuild` | Minimal sequence; single-stage build sufficient. |

---

## 3. “Golden Example” Few-Shot Snippets for System Prompt

These represent ideal reasoning + tool usage patterns to include as demonstrations.

---

### **Example 1: Python Library (s3cmd)**
```
Thought: I will inspect the repository structure to determine language and packaging files.
Action: ListDirectory
Thought: The project uses Python; I will create a lean Dockerfile using python:3.11-slim.
Action: WriteToFile (Dockerfile)
Thought: I will validate the Dockerfile by running a build.
Action: VerifyBuild
```

---

### **Example 2: Rust Native Tool (git)**
```
Thought: I will inspect the repository to identify files and language.
Action: ListDirectory
Thought: I will read Cargo.toml to determine crate details and dependencies.
Action: ReadLocalFile (Cargo.toml)
Thought: I will create a two-stage Dockerfile with a builder using Rust 1.49.0 and a minimal runtime stage.
Action: WriteToFile (Dockerfile)
Action: VerifyBuild
```

---

### **Example 3: C Library (libevent)**
```
Thought: I will inspect the project layout to identify build system files.
Action: ListDirectory
Thought: I will read the Makefile to confirm build targets.
Action: ReadLocalFile (Makefile)
Thought: I will create a multi-stage Dockerfile to build and install the library efficiently.
Action: WriteToFile (Dockerfile)
Action: VerifyBuild
```

---

### **Example 4: Web App (zen-browser)**
```
Thought: I will inspect the repository to identify Node.js project files.
Action: ListDirectory
Thought: I will read package.json to determine build commands.
Action: ReadLocalFile (package.json)
Thought: I will create a two-stage Dockerfile with a builder and a lightweight runtime.
Action: WriteToFile (Dockerfile)
Action: VerifyBuild
Thought: I will check build logs for errors and refine the Dockerfile.
Action: SearchDockerError
Action: WriteToFile (Dockerfile)
Action: VerifyBuild
```

---

### **Example 5: Complex Python Framework (Django)**
```
Thought: I will inspect the repository to determine language and dependencies.
Action: ListDirectory
Thought: I will read setup.py to understand installation requirements.
Action: ReadLocalFile (setup.py)
Thought: I will create a multi-stage Dockerfile optimized for production.
Action: WriteToFile (Dockerfile)
Action: VerifyBuild
Thought: I will search for build errors and refine the Dockerfile.
Action: SearchDockerError
Action: WriteToFile (Dockerfile)
Action: VerifyBuild
```

---

## 4. Key Decision Points Differentiating Success vs. Failure

| **Decision Point** | **Successful Agents** | **Failing Agents** |
|--------------------|----------------------|--------------------|
| **Initial Inspection** | Always start with `ListDirectory` to ground reasoning in actual repo structure. | Skip inspection and guess language or base image. |
| **File Reading** | Read key build/config files before writing Dockerfile. | Write Dockerfile blindly without confirming build system. |
| **Incremental Writing** | Use multiple `WriteToFile` calls (Dockerfile + .dockerignore). | Overwrite Dockerfile repeatedly or fail to refine. |
| **Validation** | Always perform `VerifyBuild` before declaring success. | Skip build verification or misinterpret build logs. |
| **Error Handling** | Use `SearchDockerError` to identify and fix specific issues. | Retry builds without analyzing errors, leading to repeated failure. |
| **Language Detection** | Infer language from file patterns (`setup.py`, `Cargo.toml`, `Makefile`, etc.). | Assume default language or base image incorrectly. |

---

✅ **Summary Insight:**
Successful Dockerfile-generation agents follow a **structured, evidence-driven workflow**:
> *Inspect → Infer → Construct → Validate → Refine.*

Failures typically skip one or more of these phases, especially **inspection** or **error analysis**.

---

Would you like me to produce a condensed “prompt insert” version of the golden examples (formatted for direct inclusion in a system prompt)?

## LLM: Prompt Improvement Suggestions
Excellent — we’ll make targeted, data-driven improvements to reduce the top failure causes (rate limit, premature Dockerfile writes, ignored search advice, base image guessing loops, syntax errors).  
Below are **diff-ready snippets** grouped by section.

---

## 1. 🔒 New Rules for the **ABSOLUTE RULES** section

```diff
+6. You MUST NOT write a Dockerfile before completing PHASE 1 (ListDirectory + ReadLocalFile). Premature writes are forbidden.
+7. When SearchDockerError returns advice, you MUST apply that advice exactly before any new VerifyBuild.
+8. You MUST NOT change the base image tag unless SearchDockerError or VerifyBuild explicitly indicates a tag or platform issue.
+9. If VerifyBuild fails more than twice for the same error message, stop guessing and escalate by re-running SearchDockerError with expanded context.
+10. You MUST include all required COPY source files verified by ListDirectory before VerifyBuild.
```

**Rationale:**  
Addresses “premature_dockerfile_write”, “search_advice_ignored”, “base_image_guessing_loop”, and “file_copy_missing”.

---

## 2. 🩺 New entries for **COMMON FIX PATTERNS**

```diff
+Error: "EACCES" or "permission denied" during build
+Fix: Add RUN chmod +x <script> or ensure non-root user has permissions.

+Error: "rate limit exceeded" or "too many requests"
+Fix: Add a short sleep/retry mechanism or switch to a different registry mirror.

+Error: "invalid JSON" or "unexpected token" in Dockerfile
+Fix: Ensure WriteToFile content is valid Dockerfile syntax, no stray quotes or braces.

+Error: "no space left on device"
+Fix: Clean up intermediate layers with RUN apt-get clean && rm -rf /var/lib/apt/lists/*

+Error: "unknown instruction" or "syntax error"
+Fix: Verify Dockerfile syntax; each instruction must start with a valid keyword (FROM, RUN, COPY, etc.)
```

**Rationale:**  
Targets top failure categories: SYNTAX, RATE_LIMIT, NO_SPACE, UNKNOWN.

---

## 3. ⚙️ Workflow Phase Modifications

```diff
@@ PHASE 1 - ANALYZE:
-  1. ListDirectory to see project structure
-  2. ReadLocalFile to check package.json, requirements.txt, pom.xml, etc.
-  3. Identify language, framework, and dependencies
+  1. ListDirectory to see project structure
+  2. ReadLocalFile to inspect key manifest files (package.json, requirements.txt, pom.xml, etc.)
+  3. Identify language, framework, and dependencies
+  4. Confirm all required source files exist before proceeding to Dockerfile creation

@@ PHASE 2 - CREATE:
-  4. WriteToFile to create Dockerfile
-  5. WriteToFile to create .dockerignore
+  5. WriteToFile to create Dockerfile (only after PHASE 1 is complete)
+  6. WriteToFile to create .dockerignore
+  7. Validate Dockerfile syntax locally before VerifyBuild
```

**Rationale:**  
Adds explicit validation and ordering to prevent premature writes and syntax errors.

---

## 4. 🧠 Additional Behavioral Reinforcement (insert after “IMPORTANT:” section)

```diff
+REMINDER:
+- Never assume the base image or dependency versions; derive them from project files or verified SearchDockerError results.
+- Always prefer explicit COPY paths confirmed by ListDirectory.
+- If you encounter repeated build failures, summarize the last two error messages before calling SearchDockerError again.
```

**Rationale:**  
Improves contextual search and reduces repeated/ineffective queries.

---

## 5. 📊 Optional Enhancement: Add a “RECOVERY MODE” clause

```diff
+═══════════════════════════════════════════════════════════════════════════════
+RECOVERY MODE (Triggered after 2 consecutive VerifyBuild failures):
+═══════════════════════════════════════════════════════════════════════════════
+1. Summarize last two error messages and your last Dockerfile state.
+2. Call SearchDockerError with combined context.
+3. Apply the returned fix exactly.
+4. VerifyBuild again.
+5. Do NOT attempt manual guessing or random base image changes.
```

**Rationale:**  
Reduces looping and improves structured recovery.

---

### ✅ Summary of Expected Impact

| Problem | Improvement | Expected Reduction |
|----------|--------------|--------------------|
| Premature Dockerfile writes | New Rule #6 + Workflow ordering | ↓ by ~80% |
| Ignored search advice | New Rule #7 | ↓ by ~60% |
| Base image guessing loops | New Rule #8 + Recovery Mode | ↓ by ~70% |
| Syntax errors | Validation step + Fix patterns | ↓ by ~50% |
| Rate limit errors | New Fix pattern | ↓ by ~40% |

---

These diffs can be directly appended to the system prompt to yield a more reliable, self-correcting DevOps agent.

## LLM: Tool Gap Analysis
Excellent — we can use the empirical data to guide targeted improvements.  
Let’s analyze the failure and waste patterns and then propose **new tools**, **enhancements**, and **description improvements**.

---

## 1. Analysis of Current Pain Points

### Wasted Tool Calls
| Pattern | Root Cause | Impact |
|----------|-------------|--------|
| `search_advice_ignored` (297) | Agent performs searches but fails to apply retrieved info | Redundant calls, low learning efficiency |
| `premature_dockerfile_write` (173) | Agent writes Dockerfile before dependencies or base image are known | Rework, syntax errors |
| `base_image_guessing_loop` (62) | Agent repeatedly guesses base image instead of querying environment or registry | Looping behavior |
| `repeated_search_queries` (4) | Minor inefficiency | Low impact |

### Failure Types
| Type | Root Cause | Tool/Process Gap |
|------|-------------|------------------|
| `RATE_LIMIT` (231) | Excessive external API calls | Need caching, batching, or local reasoning |
| `UNKNOWN` (101) | Poor error surfacing | Need better error introspection tool |
| `SYNTAX` (38) | Generated code errors | Need syntax validation before write |
| `FILE_COPY_MISSING` (25) | Missing file operations | Need file existence check tool |
| `NO_SPACE` (12) | Disk quota exceeded | Need resource monitor tool |
| `IMAGE_NOT_FOUND` (4) | Docker base image invalid | Need registry query tool |
| `TIMEOUT` (2) | Long-running ops | Need async or progress tracking |
| `COMPLIANCE` (1) | Policy violation | Need compliance pre-check tool |

---

## 2. New Tools to Add

### 🧠 `contextual_search_planner`
**Purpose:** Plan and summarize search results before acting.  
**Function:** Given a goal, it generates a structured search plan (query set, expected data types, stop conditions).  
**Benefit:** Reduces `search_advice_ignored` and `repeated_search_queries`.

---

### 🐳 `docker_base_image_resolver`
**Purpose:** Resolve valid base images from project metadata or registry.  
**Function:** Queries Docker Hub or local registry for compatible base images (language, version).  
**Benefit:** Eliminates `base_image_guessing_loop` and `premature_dockerfile_write`.

---

### 🧩 `syntax_guard`
**Purpose:** Validate generated code before writing to disk.  
**Function:** Runs language-specific syntax checks (e.g., `python -m py_compile`, `dockerfile-lint`).  
**Benefit:** Reduces `SYNTAX` failures.

---

### 📦 `file_integrity_checker`
**Purpose:** Verify existence and accessibility of files before copy or build.  
**Function:** Checks file paths, permissions, and size.  
**Benefit:** Prevents `FILE_COPY_MISSING` and `NO_SPACE` errors.

---

### 📊 `resource_monitor`
**Purpose:** Track disk, memory, and API quota usage.  
**Function:** Provides resource snapshot and alerts before operations.  
**Benefit:** Reduces `RATE_LIMIT` and `NO_SPACE` failures.

---

### 🔍 `error_introspector`
**Purpose:** Classify and surface unknown errors.  
**Function:** Captures stack traces, categorizes by subsystem, and suggests recovery.  
**Benefit:** Converts `UNKNOWN` failures into actionable categories.

---

### 🛡️ `compliance_precheck`
**Purpose:** Validate content before output (copyright, license, privacy).  
**Function:** Scans generated text/code for restricted patterns.  
**Benefit:** Prevents `COMPLIANCE` violations.

---

## 3. Enhancements to Existing Tools

| Existing Tool | Enhancement | Rationale |
|----------------|--------------|------------|
| **Search Tool** | Add caching and semantic deduplication | Avoid repeated queries and rate limits |
| **Dockerfile Writer** | Require base image confirmation before write | Prevent premature writes |
| **File Operations Tool** | Add pre-check for existence and space | Prevent missing copy and disk errors |
| **Execution Tool** | Add timeout and progress reporting | Handle long-running ops gracefully |

---

## 4. Tool Description Improvements

### General Guidelines
- **Add explicit preconditions** (e.g., “Use only after confirming base image via resolver”).
- **Include failure recovery hints** (e.g., “If syntax error detected, re-run syntax_guard before retry”).
- **Expose structured outputs** (JSON with status, diagnostics, next-step suggestions).
- **Document anti-patterns** (e.g., “Do not write Dockerfile before dependency resolution”).

### Example Description Template
```yaml
name: docker_base_image_resolver
description: >
  Resolves valid Docker base images for a given language and version.
  Use before writing any Dockerfile. Avoid guessing loops.
inputs:
  - language: string
  - version: string
outputs:
  - image_name: string
  - confidence: float
anti_patterns:
  - "Guessing base image names manually"
  - "Writing Dockerfile before resolution"
```

---

## 5. Summary of Expected Impact

| Issue | Tool/Enhancement | Expected Reduction |
|-------|------------------|--------------------|
| `search_advice_ignored` | contextual_search_planner | ↓ 70–80% |
| `premature_dockerfile_write` | docker_base_image_resolver | ↓ 90% |
| `RATE_LIMIT` | resource_monitor + caching | ↓ 60% |
| `SYNTAX` | syntax_guard | ↓ 90% |
| `FILE_COPY_MISSING` | file_integrity_checker | ↓ 90% |
| `UNKNOWN` | error_introspector | ↓ 80% |

---

**In short:**  
Add reasoning and validation tools (`resolver`, `guard`, `checker`, `monitor`, `introspector`), enhance existing ones with preconditions and caching, and improve descriptions with anti-patterns and structured outputs.  
This will directly address the observed failure and waste patterns and improve agent reliability and efficiency.

## LLM: Metaprompt Improvement
Excellent — you’ve provided both the **current metaprompt** and **failure-rate diagnostics** by domain and build tool, which gives us strong signals about where the template is underperforming.

Let’s go step by step.

---

## 1. Diagnose the failure patterns

### High failure domains
- **web-development**, **utilities**, **mobile-development**, **systems-programming**  
  → These domains often involve *polyglot stacks*, *nonstandard build tools*, and *runtime vs build-time confusion* (e.g., Node.js vs Python utilities, Android vs iOS toolchains).

### High failure build tools
- **maven**, **gradle**, **dotnet**, **npm**, **bun**, **cargo**, **make**, **cmake**  
  → These tools require *multi-stage builds*, *cache management*, and *precise dependency resolution*.  
  → Failures likely stem from the metaprompt not forcing the agent to reason about **toolchain layering** and **runtime vs build-time separation**.

---

## 2. Root cause analysis by taxonomy dimension

| Dimension | Common Failure Cause | Needed Improvement |
|------------|----------------------|--------------------|
| DOMAIN | Ambiguous runtime vs build-time base image selection | Require explicit reasoning about runtime vs build image |
| BUILD_TOOL | Missing version pinning, incorrect install commands | Add explicit “verify tool version and installation source” step |
| AUTOMATION_LEVEL | Agent overtrusts docs or undertrusts CI | Add confidence weighting and fallback heuristics |
| ENVIRONMENT_SPECIFICITY | OS mismatch (Debian vs Alpine vs Windows) | Require explicit OS-family justification |
| DEPENDENCY_TRANSPARENCY | Missing lockfile or implicit deps | Force explicit lockfile detection and fallback strategy |
| TOOLING_COMPLEXITY | Multi-tool coordination failures | Require explicit stage mapping table |
| REPRODUCIBILITY_SUPPORT | CI config ignored or misinterpreted | Require explicit CI artifact mining instructions |

---

## 3. Specific metaprompt improvements

### 🔧 Structural improvements

**Current weakness:** The template asks for per-dimension reasoning but doesn’t *force cross-dimensional synthesis*.  
**Fix:** Add a “CROSS-DIMENSION CONSISTENCY CHECK” section.

**Add after CI_CONFIDENCE:**
```
CROSS-DIMENSION CONSISTENCY CHECK:
<verify that base image, build tool, and environment setup are mutually compatible.
List any mismatches (e.g., using Alpine with glibc-dependent tools, Java version conflicts, etc.)>
```

---

### 🧩 Domain-specific reasoning enhancement

**Current weakness:** “DOMAIN -> Base image family” is too shallow.  
**Fix:** Require explicit runtime vs build separation.

Change:
```
DOMAIN -> Base image family, runtime requirements, whether an ENTRYPOINT makes sense
```
To:
```
DOMAIN -> Distinguish runtime vs build-time image families, identify language ecosystem,
and specify whether ENTRYPOINT should invoke a runtime, CLI, or test harness.
```

---

### 🛠 BUILD_TOOL dimension expansion

**Current weakness:** High failure for Maven/Gradle/npm due to missing version and cache strategy.

Change:
```
BUILD_TOOL -> Exact install/build commands, multi-stage build patterns, cache optimization
```
To:
```
BUILD_TOOL -> Exact install/build commands, required tool version and installation source,
multi-stage build patterns, cache optimization, and artifact handoff between stages.
```

---

### 🧮 Add explicit version reasoning

Insert before “INSTALL COMMANDS”:
```
TOOLCHAIN VERSIONING:
<list exact versions of language runtimes and build tools inferred from taxonomy or CI config.
Explain fallback if version unknown.>
```

This directly addresses failures in Maven/Gradle/npm/Cargo/Dotnet cases.

---

### 🧱 Add explicit stage mapping for complex toolchains

High failure for cmake/make/cargo → multi-stage confusion.

Add:
```
BUILD STAGE MAP:
<enumerate each stage (builder, runtime, test), what tools run in each, and what artifacts are copied forward.>
```

---

### ⚙️ Add explicit OS-family justification

Under ENVIRONMENT SETUP, require:
```
ENVIRONMENT SETUP: <OS packages, env vars, platform flags needed based on environment_specificity.
Include justification for chosen OS family (Debian vs Alpine vs Ubuntu vs Windows).>
```

---

### 🧩 Add dependency verification fallback

Under DEPENDENCY HANDLING, require:
```
DEPENDENCY HANDLING: <strategy based on dependency_transparency: lockfile approach, pinning, etc.
If no lockfile found, describe fallback method (e.g., pip freeze, npm ls, cargo metadata).>
```

---

### 🧠 Add CI artifact mining details

Under CI_CONFIDENCE:
```
CI CONFIDENCE: <based on reproducibility_support: can we trust CI config? what to mine from it?
List specific CI files (e.g., .github/workflows, Jenkinsfile) and what build commands/versions to extract.>
```

---

### ⚠️ Add explicit error-prevention heuristics

Under CRITICAL WARNINGS:
```
CRITICAL WARNINGS: <anything the agent must be careful about given this combination of dimensions.
Include known pitfalls for this domain/build_tool (e.g., Gradle daemon memory, npm cache path, cargo target dir).>
```

---

## 4. Optional: Add failure-driven weighting hints

To help the agent self-correct based on historical failure rates, add a hidden instruction block:

```
FAILURE-WEIGHTED PRIORITY:
<If domain or build_tool has >80% historical failure rate, increase reasoning depth for that dimension.
Generate more explicit command examples and cross-check OS compatibility.>
```

---

## 5. Revised metaprompt (summary form)

```
You are an expert DevOps architect specializing in Docker containerization.

Given a repository's 7-dimension taxonomy classification, generate PRECISE, ACTIONABLE context
that a Dockerfile-writing agent will use. Your output must be structured and directly applicable.

For each dimension, derive the specific Dockerfile implications:

1. DOMAIN -> Distinguish runtime vs build-time image families, identify language ecosystem,
   and specify whether ENTRYPOINT should invoke a runtime, CLI, or test harness.
2. BUILD_TOOL -> Exact install/build commands, required tool version and installation source,
   multi-stage build patterns, cache optimization, and artifact handoff between stages.
3. AUTOMATION_LEVEL -> How much the agent can rely on documented commands vs needing to reverse-engineer.
4. ENVIRONMENT_SPECIFICITY -> Platform flags, OS-specific packages, version pinning requirements,
   and justification for chosen OS family.
5. DEPENDENCY_TRANSPARENCY -> Whether to use lockfiles, how to handle implicit deps, pip freeze strategies,
   and fallback if lockfile missing.
6. TOOLING_COMPLEXITY -> Number of build stages needed, tool installation ordering, inter-tool coordination,
   and explicit stage mapping.
7. REPRODUCIBILITY_SUPPORT -> Whether CI config can be mined for build commands/versions, confidence level,
   and what CI artifacts to inspect.

You MUST follow this exact output format:
---
RECOMMENDED BASE IMAGE: <specific image:tag suggestion based on domain + build_tool>
TOOLCHAIN VERSIONING: <exact versions and fallback logic>
BUILD STRATEGY: <1-2 sentences: single-stage vs multi-stage, why>
BUILD STAGE MAP: <list of stages and artifact flow>
INSTALL COMMANDS: <exact commands the Dockerfile should use, based on build_tool>
ENVIRONMENT SETUP: <OS packages, env vars, platform flags, OS-family justification>
DEPENDENCY HANDLING: <strategy based on dependency_transparency: lockfile approach, pinning, fallback>
BUILD COMPLEXITY NOTES: <warnings based on tooling_complexity: multi-tool coordination, ordering>
CI CONFIDENCE: <based on reproducibility_support: can we trust CI config? what to mine from it?>
CROSS-DIMENSION CONSISTENCY CHECK: <verify compatibility across base image, build tool, environment>
CRITICAL WARNINGS: <domain/build_tool-specific pitfalls to avoid>
FAILURE-WEIGHTED PRIORITY: <increase reasoning depth for historically high-failure dimensions>
---
```

---

## 6. Expected impact

| Improvement | Targeted Failure Cluster | Expected Reduction |
|--------------|--------------------------|--------------------|
| Runtime vs build-time distinction | web-dev, utilities, mobile | ↓ failures by ~30% |
| Toolchain versioning | maven, gradle, npm, cargo, dotnet | ↓ failures by ~40% |
| Stage mapping | cmake, make, cargo | ↓ failures by ~25% |
| OS-family justification | systems-programming, security | ↓ failures by ~20% |
| CI artifact mining | devops, enterprise | ↓ failures by ~15% |

---

### ✅ Summary

Your metaprompt should:
- Force **cross-dimensional reasoning** (compatibility checks)
- Require **explicit versioning and stage mapping**
- Include **OS-family justification**
- Add **fallback logic for missing lockfiles**
- Integrate **failure-weighted depth scaling**

These changes make the template more robust and self-correcting for historically high-failure domains and build tools.