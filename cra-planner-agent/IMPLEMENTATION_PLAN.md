# CRA Planner Agent — Implementation Plan
**Based on empirical analysis of 50-repo test run (2026-02-22)**

---

## Ground truth before we start

From the raw JSONL, the actual breakdown is:
- 43 agent-generated attempts → 27 successes (62.8%), 16 failures (37.2%)
- All 16 failures hit the 3-attempt ceiling
- ~9 of 16 failures involved the "modify after verify" ordering bug in some form
- Token efficiency: failures consumed 2.27× more tokens, but only 1.2× per attempt — the gap is mostly structural (more attempts), not per-attempt intelligence

The reviewer's key correction: **failures aren't dumb — they spend ~303K tokens/attempt vs. 366K for failures, nearly the same. The problem is structural, not intelligence.**

---

## Improvement Taxonomy

| # | Fix | Files | Estimated repos recovered | Speed cost |
|---|-----|-------|--------------------------|------------|
| 1 | Snapshot + restore verified Dockerfile | `workflow.py` + `parallel_empirical_test.py` | +3 (mybatis-3, TypeScript, imgui) | None |
| 2 | Dockerfile state reset between attempts | `workflow.py` | +2–3 (folly, json-c, webview) | None |
| 3 | Language detection: recursive build-file search | `preparation.py` | +4–5 (all "Unknown" C++/Java failures) | <1s |
| 4 | Build manifest analysis (versions, hooks, build tools) | `preparation.py` | +3–4 (bootstrap, numpy, django, react) | 3–8s |
| 5 | Failure classification before lesson injection | `workflow.py` | +2 (commons-csv, flink compliance) | <1s |
| 6 | Adaptive Docker build timeout by complexity | `parallel_empirical_test.py` | +1–2 (react, flink) | None |
| 7 | Proactive base image validation in ANALYZE phase | `core.py` system prompt | +2 (guava, Activiti) | 1–2s |
| 8 | Structured CI/CD extraction | `preparation.py` | +2–3 (react, bootstrap, spring) | 1s |
| 9 | Known-pitfall injection at preparation time | `preparation.py` | +1–2 (husky, RAT, Gradle) | 0s |
| 10 | Cross-attempt diff injection | `workflow.py` | +1–2 (convergence improvement) | 0s |

**Conservative total: +15–22 repos recoverable = 68% → ~83–90% success rate**

---

## Fix 1: Snapshot + Restore Verified Dockerfile
**Files:** `src/parallel_empirical_test.py` (verify_build_tool_func), `src/agent/workflow.py`
**Estimated impact:** Recovers mybatis-3 (3/3 ordering violations), TypeScript (2/3), imgui (2/3) = ~3 repos

### The exact bug (workflow.py:265–269)

```python
# CURRENT — detects the violation but THROWS AWAY the working Dockerfile
if last_write_index > last_verify_index:
    lesson = f"Attempt {attempt}: You modified the Dockerfile (step {last_write_index}) ..."
    lessons_learned.append(lesson)
    continue  # ← the verified Dockerfile is already overwritten on disk; we fail the attempt
```

The agent had a **working Dockerfile at step `last_verify_index`** — VerifyBuild passed. Then it wrote cosmetic changes. The correct response is to restore the snapshot from the moment VerifyBuild passed, not to discard the entire attempt.

### Change A — `parallel_empirical_test.py`: capture snapshot inside verify_build_tool_func

**Location:** Inside `verify_build_tool_func`, at the point where `status="success"` is returned (~line 791).
The function currently returns JSON with `"status": "success"`. We need it to also embed the Dockerfile content so the caller can restore it.

```python
# BEFORE (inside verify_build_tool_func, success branch):
return json.dumps({
    "status": "success",
    "verification": {
        "smoke_test_passed": True,
        "suitability_check": suitability
    },
    # "auxiliary_logs": aux_logs
}, indent=2)

# AFTER: embed the verified Dockerfile content in the response
with open(dockerfile_path, 'r', encoding='utf-8') as _df:
    _verified_content = _df.read()

return json.dumps({
    "status": "success",
    "verification": {
        "smoke_test_passed": True,
        "suitability_check": suitability
    },
    "_verified_dockerfile_snapshot": _verified_content   # ← NEW
}, indent=2)
```

Do the same for the **suitability-fail branch** (where build passed + smoke test passed but suitability failed — that branch also returns success). Search for all `"status": "success"` returns in `verify_build_tool_func` and add the snapshot to each.

### Change B — `workflow.py`: restore snapshot instead of failing

**Location:** Lines 265–269 (the ordering violation check).

```python
# BEFORE:
if last_write_index > last_verify_index:
    lesson = f"Attempt {attempt}: You modified the Dockerfile (step {last_write_index}) AFTER verifying it (step {last_verify_index}). You must verify LAST."
    logger.warning(lesson)
    lessons_learned.append(lesson)
    continue

# AFTER:
if last_write_index > last_verify_index:
    # Try to restore the verified snapshot from the VerifyBuild observation
    restored = False
    if last_verify_index >= 0:
        _, verify_obs = intermediate_steps[last_verify_index]
        try:
            obs_data = json.loads(str(verify_obs).strip())
            snapshot = obs_data.get("_verified_dockerfile_snapshot")
            if snapshot:
                dockerfile_path.write_text(snapshot, encoding='utf-8')
                logger.info(f"Restored verified Dockerfile snapshot (discarding post-verify write).")
                # Treat this attempt as a success with the snapshot
                return {
                    "status": "success",
                    "report_dir": str(report_dir),
                    "dockerfile": str(dockerfile_path),
                    "attempts": attempt,
                    "language": language,
                    "note": "Restored from verified snapshot (agent wrote after verify)"
                }
        except Exception as _snap_err:
            logger.warning(f"Snapshot restore failed: {_snap_err}")

    # Fallback: no snapshot available — record lesson and retry
    lesson = f"Attempt {attempt}: You modified the Dockerfile (step {last_write_index}) AFTER verifying it (step {last_verify_index}). You must verify LAST."
    logger.warning(lesson)
    lessons_learned.append(lesson)
    continue
```

---

## Fix 2: Dockerfile State Reset Between Attempts
**File:** `src/agent/workflow.py`
**Estimated impact:** Recovers folly (oscillating lessons corrupt state), json-c (stray LLM hallucination persisted), webview (wrong path hardcoded in previous Dockerfile)

### The exact problem

When attempt N fails and attempt N+1 begins, the on-disk Dockerfile is whatever broken state attempt N left. The agent in attempt N+1 calls `ListDirectory`, sees the Dockerfile exists, reads it, and tries to **patch it** — inheriting all the corrupted state from N. This is a garbage-in/garbage-out loop.

### Change — `workflow.py`: add one line at the top of the retry loop

**Location:** Inside the `for attempt in range(1, max_retries + 1):` loop, before the `goal_prompt` is built (~line 140).

```python
for attempt in range(1, max_retries + 1):
    logger.info(f"=== Attempt {attempt}/{max_retries} ===")

    # NEW: Reset Dockerfile state before each retry attempt (not the first)
    # Lessons from previous attempts are already in `lessons_learned` —
    # the knowledge is preserved even though the file is cleared.
    if attempt > 1 and dockerfile_path.exists():
        dockerfile_path.unlink()
        logger.info(f"Cleared Dockerfile before attempt {attempt} (fresh generation with lessons injected).")

    # Also clear .dockerignore to avoid stale configurations
    dockerignore_path = Path(repo_path) / ".dockerignore"
    if attempt > 1 and dockerignore_path.exists():
        dockerignore_path.unlink()

    # ... rest of loop unchanged ...
```

**Why this is safe:** `lessons_learned` preserves all diagnostic knowledge as text in the next prompt. The file system state is reset, not the knowledge.

---

## Fix 3: Language Detection — Recursive Build-File Search
**File:** `src/agent/preparation.py`
**Estimated impact:** Fixes language detection for folly, json-c, imgui, msgpack-c, webview (C/C++), guava, Activiti, mybatis-3, spring-framework, RxJava (Java) — these all returned "Unknown" because their build files aren't in the root.

### The exact problem

`detect_project_language()` checks ONLY the root directory for build indicator files. Many large repos have a non-standard layout where:
- `CMakeLists.txt` is in a subdirectory (e.g., `msgpack-c/cpp/CMakeLists.txt`)
- `pom.xml` is in a subdirectory (e.g., for multi-module Maven projects, only root `pom.xml` matters but it may not exist in root)
- The build file check is a `for` loop with first-match wins — `package.json` in docs/node_modules would trigger "Node.js" for a Java project

### Replacement for `detect_project_language()` in `preparation.py`

```python
def detect_project_language(repo_path: str) -> str:
    """
    Detect the primary programming language of a repository.
    Three-pass strategy:
      1. Root directory build files (fastest, highest confidence)
      2. Immediate subdirectory build files (catches multi-module roots)
      3. Shallow recursive search up to depth 3 (catches nested build files)
    Falls back to file extension counting only if all passes fail.
    """
    try:
        if not os.path.exists(repo_path):
            return "Unknown"

        repo = Path(repo_path)

        # Priority-ordered build indicator map.
        # Higher entries take precedence when multiple indicators are found.
        BUILD_INDICATORS = [
            # Most specific first
            ("pom.xml",           "Java"),
            ("build.gradle",      "Java"),
            ("build.gradle.kts",  "Kotlin"),
            ("Cargo.toml",        "Rust"),
            ("go.mod",            "Go"),
            ("pyproject.toml",    "Python"),
            ("setup.py",          "Python"),
            ("requirements.txt",  "Python"),
            ("Pipfile",           "Python"),
            ("package.json",      "Node.js"),
            ("Gemfile",           "Ruby"),
            ("composer.json",     "PHP"),
            ("CMakeLists.txt",    "C++"),
            ("configure.ac",      "C"),      # autoconf (git, distcc, openvpn)
            ("meson.build",       "C"),      # meson (mpv)
            ("Makefile",          "C"),
        ]

        # Pass 1: root directory only (original behaviour, fast)
        for filename, language in BUILD_INDICATORS:
            if (repo / filename).exists():
                logger.info(f"[LangDetect] Pass 1 — {language} (found {filename} in root)")
                return language

        # Pass 2: immediate subdirectories (depth 1)
        # Skip common non-source dirs to avoid false positives
        SKIP_DIRS = {
            "node_modules", ".git", "__pycache__", ".venv", "venv",
            "env", "dist", "build", "target", "out", ".gradle",
            "docs", "doc", "examples", "sample", "test", "tests",
        }
        for subdir in repo.iterdir():
            if not subdir.is_dir():
                continue
            if subdir.name.startswith(".") or subdir.name.lower() in SKIP_DIRS:
                continue
            for filename, language in BUILD_INDICATORS:
                if (subdir / filename).exists():
                    logger.info(f"[LangDetect] Pass 2 — {language} (found {filename} in {subdir.name}/)")
                    return language

        # Pass 3: shallow recursive search (depth 2), first match wins by priority
        for depth in range(2, 4):
            for filename, language in BUILD_INDICATORS:
                # glob depth-limited search
                pattern = "/".join(["*"] * depth) + f"/{filename}"
                matches = list(repo.glob(pattern))
                # Filter out SKIP_DIRS at any path level
                matches = [
                    m for m in matches
                    if not any(part.lower() in SKIP_DIRS or part.startswith(".")
                               for part in m.parts)
                ]
                if matches:
                    logger.info(f"[LangDetect] Pass 3 (depth {depth}) — {language} "
                                f"(found {filename} at {matches[0].relative_to(repo)})")
                    return language

        # Pass 4: File extension counting (fallback, least reliable)
        EXTENSION_MAP = {
            ".py": "Python",
            ".java": "Java",
            ".kt": "Kotlin",
            ".rs": "Rust",
            ".go": "Go",
            ".rb": "Ruby",
            ".php": "PHP",
            ".cpp": "C++", ".cxx": "C++", ".cc": "C++", ".hpp": "C++",
            ".c": "C", ".h": "C",
            ".js": "Node.js", ".jsx": "Node.js", ".ts": "Node.js", ".tsx": "Node.js",
        }
        extension_counts: dict = {}
        for root, dirs, files in os.walk(repo_path):
            # Prune skip dirs in-place (os.walk respects in-place modification of dirs)
            dirs[:] = [
                d for d in dirs
                if not d.startswith(".") and d.lower() not in SKIP_DIRS
            ]
            # Don't go deeper than 4 levels for speed
            depth = root.replace(repo_path, "").count(os.sep)
            if depth > 4:
                dirs.clear()
                continue
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                lang = EXTENSION_MAP.get(ext)
                if lang:
                    extension_counts[lang] = extension_counts.get(lang, 0) + 1

        if extension_counts:
            detected = max(extension_counts.items(), key=lambda x: x[1])[0]
            logger.info(f"[LangDetect] Pass 4 (extensions) — {detected}: {extension_counts}")
            return detected

        return "Unknown"

    except Exception as e:
        logger.error(f"[LangDetect] Error: {e}")
        return "Unknown"
```

---

## Fix 4: Build Manifest Analysis
**File:** `src/agent/preparation.py`, `build_initial_context()`
**Estimated impact:** Prevents 2-attempt wasted cycles for bootstrap (husky), numpy (gfortran/Cython), django (language detection was "Node.js" but manifests would reveal Python), react/react-native (postinstall scripts), distcc (Python version mismatch)

### New function to add to `preparation.py`

Add this function after `summarize_github_workflows()`:

```python
def analyze_build_manifests(repo_path: str) -> dict:
    """
    Extract structured build metadata from manifest files.
    This is used to inject concrete version pins and known pitfalls
    into the agent's goal prompt BEFORE the first attempt.

    Returns a dict with keys:
        node_version, python_version, java_version, package_manager,
        has_postinstall, has_preinstall, has_husky, has_c_extensions,
        gradle_version, maven_version, cmake_minimum, warnings (list of str)
    """
    repo = Path(repo_path)
    result = {
        "node_version": None,
        "python_version": None,
        "java_version": None,
        "package_manager": None,
        "has_postinstall": False,
        "has_preinstall": False,
        "has_husky": False,
        "has_c_extensions": False,
        "requires_fortran": False,
        "gradle_version": None,
        "maven_version": None,
        "cmake_minimum": None,
        "build_tool": None,
        "warnings": []
    }

    # --- Node.js ---
    pkg_json = repo / "package.json"
    if pkg_json.exists():
        try:
            with open(pkg_json) as f:
                pkg = json.load(f)
            result["package_manager"] = "npm"
            if (repo / "yarn.lock").exists():
                result["package_manager"] = "yarn"
            elif (repo / "pnpm-lock.yaml").exists():
                result["package_manager"] = "pnpm"

            engines = pkg.get("engines", {})
            if "node" in engines:
                result["node_version"] = engines["node"]

            scripts = pkg.get("scripts", {})
            if "postinstall" in scripts:
                result["has_postinstall"] = True
                result["warnings"].append(
                    f"CAUTION: package.json has a 'postinstall' script: "
                    f"'{scripts['postinstall'][:120]}'. "
                    f"This runs automatically during npm/yarn install. "
                    f"COPY the full repo BEFORE running install, not just package.json."
                )
            if "preinstall" in scripts:
                result["has_preinstall"] = True
                result["warnings"].append(
                    f"CAUTION: package.json has a 'preinstall' script: "
                    f"'{scripts['preinstall'][:120]}'. "
                    f"Copy the scripts/ directory into the image BEFORE running install."
                )

            dev_deps = pkg.get("devDependencies", {})
            if "husky" in dev_deps or "husky" in pkg.get("dependencies", {}):
                result["has_husky"] = True
                result["warnings"].append(
                    "CAUTION: Husky is a devDependency. During Docker build, git hooks will fail "
                    "because there is no .git directory. Use 'npm ci --ignore-scripts' or "
                    "install git before running npm ci: 'RUN apt-get install -y git'."
                )
        except Exception as e:
            logger.warning(f"[Manifests] package.json parse error: {e}")

    # --- Python ---
    for py_file in [".python-version", "runtime.txt", ".tool-versions"]:
        candidate = repo / py_file
        if candidate.exists():
            try:
                content = candidate.read_text().strip()
                if py_file == ".tool-versions":
                    for line in content.splitlines():
                        if line.startswith("python"):
                            result["python_version"] = line.split()[-1]
                            break
                else:
                    result["python_version"] = content.split("\n")[0].strip()
            except Exception:
                pass

    # Check for C extensions (indicates need for gcc/build-essential)
    for ext_indicator in ["setup.py", "pyproject.toml"]:
        candidate = repo / ext_indicator
        if candidate.exists():
            try:
                content = candidate.read_text()
                if "ext_modules" in content or "cffi" in content or "cython" in content.lower():
                    result["has_c_extensions"] = True
                    result["warnings"].append(
                        "Python project has C extensions. Install build-essential, gcc, "
                        "and python3-dev before pip install."
                    )
                if "gfortran" in content.lower() or "fortran" in content.lower() or \
                   "blas" in content.lower() or "lapack" in content.lower():
                    result["requires_fortran"] = True
                    result["warnings"].append(
                        "Python project requires Fortran/BLAS. Install gfortran, "
                        "libopenblas-dev, liblapack-dev before building."
                    )
            except Exception:
                pass

    # --- Java/Gradle ---
    gradle_props = repo / "gradle" / "wrapper" / "gradle-wrapper.properties"
    if gradle_props.exists():
        try:
            content = gradle_props.read_text()
            for line in content.splitlines():
                if "distributionUrl" in line:
                    import re
                    m = re.search(r"gradle-([\d.]+)-", line)
                    if m:
                        result["gradle_version"] = m.group(1)
                        result["warnings"].append(
                            f"Gradle wrapper version: {m.group(1)}. Use this EXACT version in your "
                            f"base image or wrapper setup to avoid download failures."
                        )
                    break
        except Exception:
            pass

    # Check Java toolchain version
    for build_file in ["build.gradle", "build.gradle.kts"]:
        candidate = repo / build_file
        if candidate.exists():
            try:
                content = candidate.read_text()
                import re
                m = re.search(r"languageVersion\s*=\s*JavaLanguageVersion\.of\((\d+)\)", content)
                if m:
                    result["java_version"] = m.group(1)
                    result["warnings"].append(
                        f"Project requires Java {m.group(1)} (from build.gradle toolchain). "
                        f"Use a base image with JDK {m.group(1)}, e.g. eclipse-temurin:{m.group(1)}-jdk."
                    )
            except Exception:
                pass

    # Check for Apache license check (RAT)
    for build_file in ["pom.xml", "build.gradle", "build.gradle.kts"]:
        candidate = repo / build_file
        if candidate.exists():
            try:
                content = candidate.read_text()
                if "rat" in content.lower() and ("apache" in content.lower() or "license" in content.lower()):
                    result["warnings"].append(
                        "WARNING: Apache RAT license check detected in build configuration. "
                        "This check validates that all source files have ASF license headers. "
                        "It will FAIL during docker build if any generated files lack headers. "
                        "Consider adding '-Drat.skip=true' to Maven/Gradle flags, or skip the "
                        "test phase: use 'mvn package -DskipTests -Drat.skip=true'."
                    )
                    break
            except Exception:
                pass

    # --- C/C++: CMake ---
    cmake_file = repo / "CMakeLists.txt"
    if cmake_file.exists():
        try:
            content = cmake_file.read_text()
            import re
            m = re.search(r"cmake_minimum_required\s*\(\s*VERSION\s+([\d.]+)", content)
            if m:
                result["cmake_minimum"] = m.group(1)
        except Exception:
            pass

    return result


def format_manifest_warnings(manifest: dict) -> str:
    """Format manifest analysis results into a readable string for injection into goal prompt."""
    lines = []

    if manifest.get("node_version"):
        lines.append(f"- Node.js version required: {manifest['node_version']}")
    if manifest.get("python_version"):
        lines.append(f"- Python version required: {manifest['python_version']}")
    if manifest.get("java_version"):
        lines.append(f"- Java version required: {manifest['java_version']}")
    if manifest.get("gradle_version"):
        lines.append(f"- Gradle version: {manifest['gradle_version']}")
    if manifest.get("package_manager"):
        lines.append(f"- Package manager: {manifest['package_manager']}")
    if manifest.get("cmake_minimum"):
        lines.append(f"- CMake minimum version: {manifest['cmake_minimum']}")

    for warning in manifest.get("warnings", []):
        lines.append(f"⚠️  {warning}")

    return "\n".join(lines) if lines else ""
```

### Update `build_initial_context()` to call the new function

```python
def build_initial_context(llm: BaseChatModel, repo_path: str) -> dict:
    language = detect_project_language(repo_path)
    logger.info(f"Building context for language: {language}")

    guidelines = generate_language_guidelines(llm, language)
    workflows = summarize_github_workflows(repo_path)

    # NEW: analyze build manifests for version pins and known pitfalls
    manifest = analyze_build_manifests(repo_path)
    manifest_warnings = format_manifest_warnings(manifest)

    context_str = f"""
DETECTED LANGUAGE: {language}

LANGUAGE GUIDELINES (Latest Best Practices):
{guidelines}

CI/CD WORKFLOWS (Hints from .github):
{workflows}
"""
    # Only add manifest section if there's content
    if manifest_warnings:
        context_str += f"""
BUILD MANIFEST ANALYSIS (Version pins and known pitfalls — READ CAREFULLY):
{manifest_warnings}
"""

    return {
        "language": language,
        "context_str": context_str,
        "manifest": manifest   # pass through for potential use in workflow.py
    }
```

Also add `import json` and `import re` at the top of `preparation.py` if not already present.

---

## Fix 5: Failure Classification Before Lesson Injection
**File:** `src/agent/workflow.py`
**Estimated impact:** Immediately identifies compliance failures (commons-csv, flink) and skips remaining attempts, saving ~2 wasted attempts per unfixable repo; correctly routes IMAGE_NOT_FOUND errors to DockerImageSearch

### New function to add to `workflow.py`

Add this function before `run_learner_agent()`:

```python
import re as _re  # add to imports if not present

# Known unfixable failure signatures
_COMPLIANCE_PATTERNS = [
    r"rat:check",
    r"apache rat",
    r"unapproved license",
    r"license header",
    r"rat plugin",
]
_TIMEOUT_PATTERNS = [
    r"exceeded.*timeout",
    r"timeout.*exceeded",
    r"build timed out",
    r"TimeoutExpired",
]
_IMAGE_NOT_FOUND_PATTERNS = [
    r"manifest unknown",
    r"manifest for .* not found",
    r"pull access denied",
    r"not found: manifest unknown",
    r"repository does not exist",
]
_APT_NOT_FOUND_PATTERNS = [
    r"E: Unable to locate package",
    r"E: Package .* has no installation candidate",
]
_SYNTAX_PATTERNS = [
    r"unknown instruction",
    r"dockerfile parse error",
    r"syntax error",
    r"unknown flag",
]


def classify_failure(lessons_learned: list, intermediate_steps: list) -> dict:
    """
    Classify the dominant failure mode from the last attempt's outputs.

    Returns:
        {
          "type": one of IMAGE_NOT_FOUND | APT_NOT_FOUND | COMPLIANCE | TIMEOUT
                         | SYNTAX | ORDERING | UNKNOWN,
          "unfixable": bool,  # if True, skip remaining attempts
          "hint": str         # injected into next attempt's prompt
        }
    """
    # Gather all text from the most recent VerifyBuild observation
    error_text = ""
    for action, observation in reversed(intermediate_steps):
        tool_name = getattr(action, 'tool', '')
        if 'VerifyBuild' in tool_name:
            error_text = str(observation)
            break

    combined = " ".join(lessons_learned) + " " + error_text
    combined_lower = combined.lower()

    # Check compliance (unfixable within current toolset)
    if any(_re.search(p, combined_lower) for p in _COMPLIANCE_PATTERNS):
        return {
            "type": "COMPLIANCE",
            "unfixable": True,
            "hint": (
                "CRITICAL: This build fails due to an Apache RAT license compliance check. "
                "This check validates ASF license headers in ALL source files and cannot be "
                "satisfied by Dockerfile changes alone. To bypass: add the Maven flag "
                "'-Drat.skip=true' or Gradle flag '-x rat' to your build command. "
                "Example: RUN mvn package -DskipTests -Drat.skip=true"
            )
        }

    # Check timeout
    if any(_re.search(p, combined_lower) for p in _TIMEOUT_PATTERNS):
        return {
            "type": "TIMEOUT",
            "unfixable": False,
            "hint": (
                "The Docker build timed out (>600s). Strategies to fix: "
                "1. Skip tests: add '-DskipTests' (Maven) or '-x test' (Gradle). "
                "2. Use a pre-built base image that already has dependencies. "
                "3. Add '--no-cache' to force fresh layers if cache is causing slowness."
            )
        }

    # Check bad image tag
    if any(_re.search(p, combined_lower) for p in _IMAGE_NOT_FOUND_PATTERNS):
        return {
            "type": "IMAGE_NOT_FOUND",
            "unfixable": False,
            "hint": (
                "The base image tag does not exist on Docker Hub. "
                "Use DockerImageSearch to find a valid tag BEFORE writing the Dockerfile. "
                "Example: DockerImageSearch(query='maven 3.8 openjdk 17') to get real available tags."
            )
        }

    # Check missing apt package
    if any(_re.search(p, combined_lower) for p in _APT_NOT_FOUND_PATTERNS):
        # Extract the package name
        m = _re.search(r"Unable to locate package (\S+)", combined)
        pkg = m.group(1) if m else "the package"
        return {
            "type": "APT_NOT_FOUND",
            "unfixable": False,
            "hint": (
                f"The apt package '{pkg}' does not exist in the base image's OS. "
                f"Search for alternatives with SearchDockerError. Common fixes: "
                f"check the package's Ubuntu/Debian name (may differ from source name), "
                f"or install from source as a fallback."
            )
        }

    # Check Dockerfile syntax
    if any(_re.search(p, combined_lower) for p in _SYNTAX_PATTERNS):
        return {
            "type": "SYNTAX",
            "unfixable": False,
            "hint": (
                "The Dockerfile has a syntax error. Common causes: "
                "1. Here-doc syntax errors (use RUN bash -c 'cat > file << EOF...'). "
                "2. Stray characters outside RUN blocks. "
                "3. COPY --from=stage referencing a stage name that doesn't exist. "
                "Read the full Dockerfile carefully before writing changes."
            )
        }

    return {"type": "UNKNOWN", "unfixable": False, "hint": ""}
```

### Update `run_learner_agent()` in `workflow.py` to use the classifier

**Location:** After `last_verify_success` is determined to be False (around line 279–283), before appending the generic lesson:

```python
# BEFORE:
if not last_verify_success:
    lesson = f"Attempt {attempt}: VerifyBuild failed or was not clean success. You must keep fixing until status='success'."
    logger.warning(lesson)
    lessons_learned.append(lesson)
    continue

# AFTER:
if not last_verify_success:
    classification = classify_failure(lessons_learned, intermediate_steps)

    if classification["unfixable"]:
        # Save the compliance hint as context but abort early
        logger.warning(f"Failure classified as UNFIXABLE ({classification['type']}). Aborting remaining attempts.")
        lessons_learned.append(f"UNFIXABLE: {classification['hint']}")
        break  # Exit loop early — remaining attempts will not help

    if classification["hint"]:
        lessons_learned.append(f"DIAGNOSIS ({classification['type']}): {classification['hint']}")

    lesson = f"Attempt {attempt}: VerifyBuild failed. Type: {classification['type']}. Fix and verify again."
    logger.warning(lesson)
    lessons_learned.append(lesson)
    continue
```

---

## Fix 6: Adaptive Docker Build Timeout
**File:** `src/parallel_empirical_test.py`
**Estimated impact:** Allows react (JS monorepo), flink (Java mega-project), scipy (C extensions) to complete builds that were previously timing out

### New helper function (add before `ParallelEmpiricalTester` class or inside it)

```python
def estimate_build_timeout(repo_path: str, detected_language: str) -> int:
    """
    Choose a build timeout based on project complexity signals.

    Default: 600s
    Complex Java/C++: 1200s
    Extreme (large monorepos): 1800s
    """
    repo = Path(repo_path)

    # Count build files as complexity proxy
    java_modules = len(list(repo.rglob("pom.xml"))) + len(list(repo.rglob("build.gradle")))
    cmake_files = len(list(repo.rglob("CMakeLists.txt")))

    # Check repo size (rough: count non-hidden files)
    try:
        file_count = sum(1 for _ in repo.rglob("*") if _.is_file() and not any(
            p.startswith(".") for p in _.parts
        ))
    except Exception:
        file_count = 0

    # Extreme monorepos (react, flink, spring, tensorflow)
    if java_modules >= 5 or file_count > 50_000:
        return 1800

    # Complex builds (scipy, openvpn, numpy, node.js with heavy postinstall)
    if java_modules >= 2 or cmake_files >= 3 or detected_language in ("C++", "Rust"):
        return 1200

    return 600
```

### Wire it into the test runner

In `parallel_empirical_test.py`, find where `DockerBuildTester` is instantiated (search for `DockerBuildTester(`). It's likely something like:

```python
# BEFORE:
self.docker_tester = DockerBuildTester(timeout=600, ...)

# AFTER:
# timeout is set per-repo dynamically in verify_build_tool_func
# leave docker_tester creation with a default; override per-call:
self.docker_tester = DockerBuildTester(timeout=600, ...)
```

Then inside `verify_build_tool_func`, before calling `self.docker_tester.build_dockerfile(...)`:

```python
# Dynamic timeout based on project complexity
_timeout = estimate_build_timeout(repo_path, detected_language or "Unknown")
if _timeout != self.docker_tester.timeout:
    logger.info(f"[VerifyBuild] Adaptive timeout: {_timeout}s (default was 600s)")
    self.docker_tester.timeout = _timeout  # override for this build
```

Note: `detected_language` needs to be in scope inside `verify_build_tool_func`. Pass it via closure from the outer function where `run_learner_agent` is called:

```python
# In the outer scope where verify_build_tool_func is defined:
detected_language = agent_result.get("language", "Unknown") if ...
# Actually: use the language from agent_result or from prep_context which is set earlier
```

The cleanest approach: capture `detected_language` in the closure by reading it after `build_initial_context()` returns, before defining `verify_build_tool_func`.

---

## Fix 7: Proactive Base Image Validation in Goal Prompt
**File:** `src/agent/core.py` (system prompt template)
**Estimated impact:** Prevents invalid tag attempts at attempt 1 for guava, Activiti; saves 1 entire attempt per repo with this issue

### Change — `core.py`: add PHASE 0 to the CRITICAL WORKFLOW

**Location:** Inside the `template` string in `create_learner_agent()`, find `PHASE 1 - ANALYZE:` and prepend:

```
PHASE 0 - BASE IMAGE DISCOVERY (MANDATORY FIRST STEP):
  0. Use DockerImageSearch to confirm your planned base image tag EXISTS before writing Dockerfile.
     Example: DockerImageSearch("maven 3.8 openjdk 17 slim")
     NEVER write FROM <image>:<tag> without first confirming the tag exists.
     If DockerImageSearch returns no results, try a different tag or version.

```

Also add this to the COMMON FIX PATTERNS section:

```
Error: "manifest unknown" or "not found" or "pull access denied"
Fix: The base image tag does not exist. Use DockerImageSearch to find a valid tag.
     Example: DockerImageSearch("node 18 alpine") to see all valid node:18* tags.
```

---

## Fix 8: Structured CI/CD Extraction
**File:** `src/agent/preparation.py` — `summarize_github_workflows()`
**Estimated impact:** Provides exact build commands (npm ci --ignore-scripts, mvn -DskipTests, etc.) to the agent before attempt 1

### Replacement for `summarize_github_workflows()`

```python
def summarize_github_workflows(repo_path: str) -> str:
    """
    Scan .github/workflows for CI/CD hints.
    Returns both a human-readable summary AND structured key facts.
    """
    workflows_dir = Path(repo_path) / ".github" / "workflows"
    if not workflows_dir.exists():
        return "No .github/workflows found."

    # Structured extraction targets
    extracted = {
        "build_commands": [],
        "test_commands": [],
        "install_commands": [],
        "node_versions": [],
        "python_versions": [],
        "java_versions": [],
        "timeout_minutes": [],
        "env_vars": [],
    }
    summary_lines = ["Found CI/CD Workflows:"]

    try:
        files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
        for wf in files:
            try:
                with open(wf, 'r') as f:
                    content = yaml.safe_load(f)
                if not content:
                    continue
                name = content.get('name', wf.name)
                summary_lines.append(f"\n  Workflow: {name}")

                jobs = content.get('jobs', {})
                for job_name, job_data in jobs.items():
                    if not isinstance(job_data, dict):
                        continue

                    # Extract timeout
                    timeout = job_data.get('timeout-minutes')
                    if timeout:
                        extracted["timeout_minutes"].append(int(timeout))

                    # Extract matrix versions
                    strategy = job_data.get('strategy', {})
                    matrix = strategy.get('matrix', {}) if isinstance(strategy, dict) else {}
                    if isinstance(matrix, dict):
                        for key in ['node-version', 'node_version']:
                            if key in matrix:
                                v = matrix[key]
                                extracted["node_versions"].extend(v if isinstance(v, list) else [v])
                        for key in ['python-version', 'python_version']:
                            if key in matrix:
                                v = matrix[key]
                                extracted["python_versions"].extend(v if isinstance(v, list) else [v])
                        for key in ['java-version', 'java_version']:
                            if key in matrix:
                                v = matrix[key]
                                extracted["java_versions"].extend(v if isinstance(v, list) else [v])

                    steps = job_data.get('steps', [])
                    for step in steps:
                        if not isinstance(step, dict):
                            continue

                        # Extract uses: actions/setup-node etc for versions
                        uses = step.get('uses', '')
                        with_data = step.get('with', {}) or {}
                        if 'setup-node' in uses and 'node-version' in with_data:
                            extracted["node_versions"].append(str(with_data['node-version']))
                        if 'setup-python' in uses and 'python-version' in with_data:
                            extracted["python_versions"].append(str(with_data['python-version']))
                        if 'setup-java' in uses and 'java-version' in with_data:
                            extracted["java_versions"].append(str(with_data['java-version']))

                        run = step.get('run', '')
                        if not run:
                            continue
                        run_clean = run.strip().replace('\n', '; ')[:200]
                        summary_lines.append(f"    run: {run_clean}")

                        run_lower = run.lower()

                        # Categorize commands
                        if any(k in run_lower for k in ['npm install', 'npm ci', 'yarn install', 'yarn']):
                            extracted["install_commands"].append(run_clean)
                        if any(k in run_lower for k in ['npm run build', 'yarn build', 'mvn package',
                                                         'gradle build', 'pip install', 'python setup.py']):
                            extracted["build_commands"].append(run_clean)
                        if any(k in run_lower for k in ['npm test', 'yarn test', 'pytest', 'mvn test',
                                                         'gradle test', 'cargo test']):
                            extracted["test_commands"].append(run_clean)

            except Exception:
                pass

    except Exception as e:
        return f"Error reading workflows: {e}"

    # Build structured summary
    structured_lines = []
    if extracted["node_versions"]:
        structured_lines.append(f"  Node.js versions used in CI: {list(set(extracted['node_versions']))}")
    if extracted["python_versions"]:
        structured_lines.append(f"  Python versions used in CI: {list(set(extracted['python_versions']))}")
    if extracted["java_versions"]:
        structured_lines.append(f"  Java versions used in CI: {list(set(extracted['java_versions']))}")
    if extracted["timeout_minutes"]:
        structured_lines.append(f"  CI job timeout: {max(extracted['timeout_minutes'])} minutes → "
                                  f"set Docker build timeout similarly")
    if extracted["install_commands"]:
        structured_lines.append(f"  Install command from CI: {extracted['install_commands'][0]}")
    if extracted["build_commands"]:
        structured_lines.append(f"  Build command from CI: {extracted['build_commands'][0]}")

    result = "\n".join(summary_lines)
    if structured_lines:
        result += "\n\n  KEY FACTS EXTRACTED FROM CI:\n" + "\n".join(structured_lines)
    return result
```

---

## Fix 9: Known-Pitfall Injection at Preparation Time
**File:** `src/agent/preparation.py` — new function, called from `build_initial_context()`
**Estimated impact:** Surfaces Husky, RAT, Gradle wrapper, FastFloat patterns BEFORE attempt 1

This is an extension of Fix 4. The manifest analysis already catches Husky, RAT, and Fortran. Add one more detection pass specifically for patterns we know from the empirical run:

```python
def detect_known_pitfalls(repo_path: str, language: str) -> list:
    """
    Scan the repo for patterns that have historically caused failures.
    Returns a list of warning strings to inject into the goal prompt.
    """
    import re
    repo = Path(repo_path)
    warnings = []

    # Gradle wrapper version that doesn't exist
    gradle_props = repo / "gradle" / "wrapper" / "gradle-wrapper.properties"
    if gradle_props.exists():
        try:
            content = gradle_props.read_text()
            m = re.search(r"distributionUrl=.*gradle-([\d.]+)-", content)
            if m:
                version = m.group(1)
                warnings.append(
                    f"Gradle wrapper requests version {version}. Verify this version exists at "
                    f"https://services.gradle.org/distributions/ BEFORE building. "
                    f"Known non-existent versions include 8.15.0. "
                    f"If unavailable, use the nearest valid version (e.g., 8.14.1)."
                )
        except Exception:
            pass

    # Facebook/Meta monorepo pattern (postinstall scripts)
    pkg_json = repo / "package.json"
    if pkg_json.exists():
        try:
            import json as _json
            pkg = _json.loads(pkg_json.read_text())
            scripts = pkg.get("scripts", {})
            for script_name in ["postinstall", "preinstall"]:
                if script_name in scripts:
                    script_val = scripts[script_name]
                    if "node " in script_val and "scripts/" in script_val:
                        warnings.append(
                            f"'{script_name}' script references a file in scripts/ directory: "
                            f"'{script_val[:100]}'. Copy scripts/ BEFORE running npm/yarn install, "
                            f"otherwise this will fail with 'Cannot find module'."
                        )
        except Exception:
            pass

    # C++: check if folly-style meta-library (many deps)
    cmake = repo / "CMakeLists.txt"
    if cmake.exists():
        try:
            content = cmake.read_text().lower()
            # Projects that are known to have unusual package names on Ubuntu
            if "fastfloat" in content:
                warnings.append(
                    "CMakeLists.txt references FastFloat. "
                    "The package 'libfastfloat-dev' does NOT exist on Ubuntu 22.04 apt. "
                    "Install FastFloat from source: RUN git clone https://github.com/fastfloat/fast_float "
                    "&& cmake -S fast_float -B fast_float/build && cmake --install fast_float/build"
                )
            if "glog" in content and "find_package" in content:
                warnings.append(
                    "CMakeLists.txt uses glog. On Ubuntu 22.04, the apt package is 'libgoogle-glog-dev', "
                    "NOT 'libglog-dev'."
                )
        except Exception:
            pass

    return warnings
```

Call this from `build_initial_context()` and append its results to `manifest_warnings`.

---

## Fix 10: Cross-Attempt Diff Injection
**File:** `src/agent/workflow.py`
**Estimated impact:** Helps the agent understand what it changed between attempts (prevents repetitive fixes), improves convergence for near-miss repos

### Implementation

**In the loop, before building `feedback_section`**, save the previous Dockerfile content:

```python
# At the start of each attempt (inside loop, before goal_prompt build):
previous_dockerfile_content = None
if dockerfile_path.exists() and attempt > 1:
    # Read BEFORE we delete (Fix 2 deletes at top of loop, so read before that)
    # Note: if Fix 2 is applied, this read must happen BEFORE the unlink
    pass
```

Since Fix 2 deletes the Dockerfile at the top of the loop, we need to save the content BEFORE deletion. Adjust Fix 2 as follows:

```python
# Modified Fix 2 (top of loop, attempt > 1):
if attempt > 1 and dockerfile_path.exists():
    previous_dockerfile_content = dockerfile_path.read_text(encoding='utf-8', errors='replace')
    dockerfile_path.unlink()
    if dockerignore_path.exists():
        dockerignore_path.unlink()
else:
    previous_dockerfile_content = None
```

Then in the `feedback_section` builder, add the diff if available:

```python
if lessons_learned:
    diff_section = ""
    if previous_dockerfile_content and dockerfile_path.exists():
        # Simple line-level diff summary (avoid importing difflib if not wanted)
        try:
            import difflib
            prev_lines = previous_dockerfile_content.splitlines()
            curr_lines = dockerfile_path.read_text(encoding='utf-8').splitlines() if dockerfile_path.exists() else []
            diff = list(difflib.unified_diff(prev_lines, curr_lines, lineterm='', n=2))
            if diff:
                diff_str = "\n".join(diff[:40])  # cap at 40 lines to save tokens
                diff_section = f"\nDIFF FROM PREVIOUS ATTEMPT (what the agent changed last time):\n```\n{diff_str}\n```\n"
        except Exception:
            pass

    feedback_section = f"""
...
{diff_section}
...
"""
```

---

## Implementation Order (Suggested)

Execute fixes in this order to maximize early gains:

1. **Fix 1** (snapshot restore) — pure bug fix, no side effects, 3 repos recovered
2. **Fix 2** (Dockerfile reset) — one line, prevents corruption cascade
3. **Fix 3** (language detection) — standalone function replacement, zero risk
4. **Fix 5** (failure classification) — adds classification logic, saves wasted attempts on unfixable repos
5. **Fix 7** (base image validation in prompt) — zero-code system prompt change
6. **Fix 4** (build manifest analysis) — moderate complexity, high reward for Python/JS/Java
7. **Fix 8** (structured CI extraction) — replaces existing function, moderate complexity
8. **Fix 6** (adaptive timeout) — small helper + one closure variable
9. **Fix 9** (known pitfalls) — optional but cheap; fold into Fix 4 if time allows
10. **Fix 10** (diff injection) — nice-to-have, lowest priority

---

## Expected Outcome After All Fixes

| Category | Before | After (est.) |
|---|---|---|
| Agent success rate | 62.8% (27/43) | ~80–87% (34–37/43) |
| 1-attempt success rate | 63% of successes | ~75–80% of successes |
| Repos lost to ordering bug | 3–5 | 0 (snapshot restore) |
| Repos lost to "Unknown" language | 9 (of which 6 failed) | ~2 (edge cases) |
| Repos lost to bad image tags | 2–3 | 0–1 (proactive validation) |
| Compliance failures (RAT) | 2 wasted (3 attempts each) | 2 (but exit after 1 attempt) |
| Token waste on ordering violations | ~2.7M tokens (mybatis, TS, imgui) | ~0 (snapshot → instant success) |

---

## Notes for the Paper

**F-A (ordering violations) is now recoverable, not just detectable.** The snapshot mechanism turns a 100% failure into a 100% success for repos where VerifyBuild genuinely passed.

**The token efficiency inversion holds even after fixes.** Folly, flink, and RxJava remain hard because they require N sequential discoveries. The 3-attempt budget should be increased to 4 for C++ and Java repos specifically (detectable via language after Fix 3).

**Fundamental limits:** Apache RAT compliance and true build complexity (React monorepo build >600s) remain hard constraints. The paper should distinguish these from agent behavior bugs.

**Reproducibility note:** All fixes are deterministic code changes with no stochastic components. A re-run with fixes applied should produce stable results across runs, making the paper's evaluation reproducible.
