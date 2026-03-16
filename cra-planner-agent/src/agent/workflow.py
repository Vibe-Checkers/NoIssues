
import os
import re as _re
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from .core import create_learner_agent
from langchain_core.messages import SystemMessage, HumanMessage
from .preparation import build_initial_context

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Failure classification patterns
# ---------------------------------------------------------------------------

# Signals that the build failed due to a compliance/QA plugin check.
# Patterns use word boundaries to avoid false positives on substrings
# (e.g. "rat" matching "operator", "pmd" matching "command").
_COMPLIANCE_PATTERNS = [
    r"\brat\b.*(?:check|plugin|fail)",
    r"unapproved license",
    r"license header",
    r"\bcheckstyle\b.*(?:fail|violation)",
    r"\bspotbugs\b.*fail",
    r"\bfindbugs\b.*fail",
    r"\bpmd\b.*(?:fail|violation)",
    r"\bjacoco\b.*(?:fail|threshold)",
    r"license.*plugin.*fail",
]
_TIMEOUT_PATTERNS = [
    r"exceeded.*timeout",
    r"timeout.*exceeded",
    r"build timed out",
    r"timeoutexpired",
]
_IMAGE_NOT_FOUND_PATTERNS = [
    r"manifest unknown",
    r"manifest for .* not found",
    r"pull access denied",
    r"not found: manifest unknown",
    r"repository does not exist",
]
_APT_NOT_FOUND_PATTERNS = [
    r"e: unable to locate package",
    r"e: package .* has no installation candidate",
]
_SYNTAX_PATTERNS = [
    r"unknown instruction",
    r"dockerfile parse error",
    r"syntax error",
    r"unknown flag",
]
_UNFIXABLE_PATTERNS = [
    # Dependency versions that don't exist in any public registry
    r"(?:no matching version|version not found|does not exist in the npm registry)",
    # Toolchains requiring unreleased/unavailable JDK versions
    r"cannot find a java installation.*languageversion\s*=\s*(\d+)",
    # Private registry auth required
    r"(?:unauthorized|authentication required).*(?:pull|registry)",
]


def classify_failure(lessons_learned: List[str], intermediate_steps: List[Any]) -> dict:
    """
    Classify the dominant failure mode from the last attempt's outputs.

    Returns:
        {
          "type": one of IMAGE_NOT_FOUND | APT_NOT_FOUND | COMPLIANCE | TIMEOUT
                         | SYNTAX | UNFIXABLE_DEPENDENCY | UNKNOWN,
          "unfixable": bool,  # True only when retrying is structurally pointless
          "hint": str         # category description injected into next attempt's prompt
        }
    """
    # Gather text from the most recent VerifyBuild observation
    error_text = ""
    for action, observation in reversed(intermediate_steps):
        if 'VerifyBuild' in getattr(action, 'tool', ''):
            error_text = str(observation)
            break

    combined_lower = (" ".join(lessons_learned) + " " + error_text).lower()

    # Unfixable: dependency/toolchain requirements that cannot be satisfied
    if any(_re.search(p, combined_lower) for p in _UNFIXABLE_PATTERNS):
        return {
            "type": "UNFIXABLE_DEPENDENCY",
            "unfixable": True,
            "hint": (
                "The build requires a dependency or toolchain version that is not "
                "publicly available (e.g., unpublished npm package, unreleased JDK, "
                "private registry). Retrying will not resolve this."
            )
        }

    # Compliance / QA check failure — give the agent one attempt with the hint
    # rather than breaking early, since the agent may be able to pass the skip flag.
    if any(_re.search(p, combined_lower) for p in _COMPLIANCE_PATTERNS):
        return {
            "type": "COMPLIANCE",
            "unfixable": False,
            "hint": (
                "The build is failing due to a QA or compliance check plugin "
                "(e.g., license header validation, static analysis, code coverage). "
                "This check is unrelated to whether the Dockerfile is correct. "
                "Use SearchDockerError with the plugin name and error message to find "
                "the build-tool flag that skips or configures this check."
            )
        }

    # Timeout
    if any(_re.search(p, combined_lower) for p in _TIMEOUT_PATTERNS):
        return {
            "type": "TIMEOUT",
            "unfixable": False,
            "hint": (
                "The Docker build timed out. Strategies: "
                "1. Skip the test phase using the build tool's skip-tests flag. "
                "2. Use a pre-built base image that already has dependencies cached. "
                "3. Use SearchDockerError to find the correct flag for this build tool."
            )
        }

    # Bad image tag
    if any(_re.search(p, combined_lower) for p in _IMAGE_NOT_FOUND_PATTERNS):
        return {
            "type": "IMAGE_NOT_FOUND",
            "unfixable": False,
            "hint": (
                "The base image tag does not exist on Docker Hub. "
                "Use DockerImageSearch to find a valid tag BEFORE rewriting the Dockerfile. "
                "Example: DockerImageSearch(query='node 18 alpine') lists real available tags."
            )
        }

    # Missing apt package
    if any(_re.search(p, combined_lower) for p in _APT_NOT_FOUND_PATTERNS):
        m = _re.search(r"Unable to locate package (\S+)", " ".join(lessons_learned) + " " + error_text)
        pkg = m.group(1) if m else "the package"
        return {
            "type": "APT_NOT_FOUND",
            "unfixable": False,
            "hint": (
                f"The package '{pkg}' was not found in the base image's package registry. "
                f"Use SearchDockerError to find the correct package name for this OS, "
                f"or find an alternative installation method."
            )
        }

    # Dockerfile syntax error
    if any(_re.search(p, combined_lower) for p in _SYNTAX_PATTERNS):
        return {
            "type": "SYNTAX",
            "unfixable": False,
            "hint": (
                "The Dockerfile has a syntax error. Read the full Dockerfile carefully. "
                "Common causes: here-doc formatting, stray characters outside RUN blocks, "
                "or COPY --from referencing an undefined build stage."
            )
        }

    return {"type": "UNKNOWN", "unfixable": False, "hint": ""}


# ---------------------------------------------------------------------------
# Technical lesson extraction
# ---------------------------------------------------------------------------

def extract_technical_lesson(
    intermediate_steps: List[Any],
    llm: Any,
    use_llm: bool = True
) -> Optional[str]:
    """
    Scan agent history to extract technical insights.

    Args:
        intermediate_steps: Agent action/observation pairs from this attempt.
        llm: LLM instance for summarization.
        use_llm: If False, skip the LLM summarization call and return the raw
                 context snippet directly (saves tokens; useful for ablation).

    Returns:
        A concise string explaining the technical root cause/fix, or None.
    """
    if not intermediate_steps:
        return None

    raw_context = None

    # Iterate backwards to find the most recent relevant tool output
    for action, observation in reversed(intermediate_steps):
        tool_name = action.tool if hasattr(action, 'tool') else ""

        # 1. High Priority: SearchDockerError (contains AI analysis)
        if "SearchDockerError" in tool_name:
            if "=== AI ANALYSIS ===" in str(observation):
                try:
                    analysis_part = (
                        str(observation)
                        .split("=== AI ANALYSIS ===")[1]
                        .split("=== SEARCH SOURCES")[0]
                        .strip()
                    )
                    raw_context = f"SearchDockerError Analysis: {analysis_part}"
                    break
                except Exception:
                    pass

        # 2. Medium Priority: VerifyBuild failure (raw error info)
        if "VerifyBuild" in tool_name and not raw_context:
            try:
                obs_data = json.loads(str(observation))
                if obs_data.get("status") == "failed":
                    error_msg = obs_data.get("error_snippet", "")
                    analysis = obs_data.get("error_analysis", {})
                    raw_context = f"VerifyBuild Failed. Error: {error_msg}. Analysis: {analysis}"
                    break
            except Exception:
                pass

    if not raw_context:
        return None

    if not use_llm:
        return raw_context[:300] + ("..." if len(raw_context) > 300 else "")

    # Use LLM to compress the raw context into one actionable sentence
    try:
        summary_prompt = (
            "Analyze the following error context from a failed Docker build attempt.\n"
            "Summarize it into a SINGLE, CONCISE sentence that explains the specific fix.\n"
            'Format: "Previous failure caused by [Cause]. Fix: [Action]."\n\n'
            f"CONTEXT:\n{raw_context[:4000]}"
        )
        response = llm.invoke([
            SystemMessage(content="You are a technical summarizer. Be extremely concise. Focus on the FIX."),
            HumanMessage(content=summary_prompt)
        ])
        return response.content.strip()
    except Exception as e:
        logger.warning(f"LLM summarization of lesson failed: {e}")
        return raw_context[:300] + "..."


# ---------------------------------------------------------------------------
# Main agent runner
# ---------------------------------------------------------------------------

def run_learner_agent(
    repo_path: str,
    repo_name: str,
    repo_url: str,
    repo_taxonomy: dict = None,
    max_retries: int = 3,
    max_iterations: int = 25,
    callback_handler=None,
    validation_callback=None,
    extra_tools: list = None,
    use_llm_lessons: bool = True,
) -> Dict[str, Any]:
    """
    Run the Learner Agent on a repository with intelligent refinement.

    This function:
    1. Prepares context (language detection, guidelines, CI summary, manifest warnings).
    2. Runs the agent with a high-level goal ("Containerize this").
    3. If the agent fails, feeds back classified failure hints and accumulated
       lessons and re-invokes — up to max_retries times.

    Args:
        repo_path: Path to the cloned repository.
        repo_name: Name of the repository.
        repo_url: URL of the repository.
        max_retries: Maximum number of attempts (default: 3).
        max_iterations: Max tool calls per single attempt (default: 25).
            Complex C++/CMake repos like folly need 20-30+ steps per attempt;
            the previous default of 15 caused every attempt to silently exhaust
            its step budget before VerifyBuild could be called.
        callback_handler: Optional callback handler for logging.
        validation_callback: Unused; kept for API compatibility.
        extra_tools: Additional tools to provide to the agent (e.g., VerifyBuild).
        use_llm_lessons: If True (default), use LLM to compress technical lessons
            after each failed attempt. Set False to save tokens during ablation.

    Returns:
        Dict with status, report_dir, dockerfile path, attempts, and any errors.
    """

    # Setup report directory
    ts = int(time.time())
    report_dir = Path(repo_path) / "agent_reports" / f"report_{ts}"
    report_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting analysis for {repo_name}...")

    # Initial agent creation — provides base_llm for build_initial_context
    executor, handler, base_llm = create_learner_agent(
        repository_path=repo_path,
        verbose=True,
        extra_tools=extra_tools,
        max_iterations=max_iterations,
    )

    prep_context = build_initial_context(base_llm, repo_path,
                                         repo_taxonomy=repo_taxonomy)
    language = prep_context["language"]

    # Build taxonomy context for the agent prompt
    taxonomy_context = ""
    if repo_taxonomy:
        taxonomy_context = f"""
## Repository Characterization (taxonomy-first)
- domain: {repo_taxonomy.get('domain', 'unknown')}
- build_tool: {repo_taxonomy.get('build_tool', 'unknown')}
- automation_level: {repo_taxonomy.get('automation_level', 'unknown')}
- environment_specificity: {repo_taxonomy.get('environment_specificity', 'unknown')}
- dependency_transparency: {repo_taxonomy.get('dependency_transparency', 'unknown')}
- tooling_complexity: {repo_taxonomy.get('tooling_complexity', 'unknown')}
- repro_support: {repo_taxonomy.get('repro_support', 'unknown')}

You MUST infer an initial operating profile from repository evidence during your
first exploration steps (repo_type, build_system, likely verification approach),
then implement/verify accordingly.
"""

    logger.info(f"=== Starting Agent Execution (max {max_retries} attempts) ===")

    lessons_learned: List[str] = []
    _seen_lessons: set = set()  # deduplication guard

    def _add_lesson(text: str) -> None:
        """Append a lesson only if its content hasn't been seen before."""
        if text not in _seen_lessons:
            _seen_lessons.add(text)
            lessons_learned.append(text)

    dockerfile_path = Path(repo_path) / "Dockerfile"
    dockerignore_path = Path(repo_path) / ".dockerignore"

    for attempt in range(1, max_retries + 1):
        # Adaptive iteration budget: give later attempts more room
        # so complex repos don't exhaust steps on initial exploration.
        current_max_iterations = max_iterations + (attempt - 1) * 10
        logger.info(f"=== Attempt {attempt}/{max_retries} (max_iterations={current_max_iterations}) ===")

        # Re-create agent on retries so it gets the updated budget
        if attempt > 1:
            executor, handler, base_llm = create_learner_agent(
                repository_path=repo_path,
                verbose=True,
                extra_tools=extra_tools,
                max_iterations=current_max_iterations,
            )

        # Save previous Dockerfile BEFORE deleting it so we can:
        # (a) restore the verified snapshot on Fix-1B, and
        # (b) show the previous attempt's content in the feedback prompt.
        previous_dockerfile_content: Optional[str] = None
        if attempt > 1:
            if dockerfile_path.exists():
                try:
                    previous_dockerfile_content = dockerfile_path.read_text(
                        encoding='utf-8', errors='replace'
                    )
                except Exception:
                    pass
                dockerfile_path.unlink()
                logger.info(
                    f"Cleared Dockerfile before attempt {attempt} "
                    f"(fresh generation with lessons injected)."
                )
            if dockerignore_path.exists():
                dockerignore_path.unlink()

        # Build feedback section
        feedback_section = ""
        if lessons_learned:
            # Show the previous Dockerfile content as-is so the agent can
            # see exactly what it tried. This is intentionally presented as
            # "what you wrote last time", not as a diff.
            prev_dockerfile_section = ""
            if previous_dockerfile_content:
                prev_dockerfile_section = (
                    f"\nDOCKERFILE FROM PREVIOUS ATTEMPT "
                    f"(do not copy this verbatim — it failed; study it to avoid repeating mistakes):\n"
                    f"```dockerfile\n"
                    f"{previous_dockerfile_content[:3000]}"
                    f"{'...(truncated)' if len(previous_dockerfile_content) > 3000 else ''}"
                    f"\n```\n"
                )

            feedback_section = f"""

═══════════════════════════════════════════════════════════════════════════════
⚠️ PREVIOUS ATTEMPT FAILED - YOU MUST FIX THESE ISSUES:
═══════════════════════════════════════════════════════════════════════════════
{"".join(f"• {lesson}" + chr(10) for lesson in lessons_learned)}
{prev_dockerfile_section}
REQUIRED ACTION:
1. Use SearchDockerError with the error keywords from above
2. The AI will analyze the error and provide a specific fix
3. Apply the fix and verify with VerifyBuild
═══════════════════════════════════════════════════════════════════════════════
"""

            # Cap feedback to prevent prompt bloat on later attempts
            MAX_FEEDBACK_CHARS = 4000
            if len(feedback_section) > MAX_FEEDBACK_CHARS:
                feedback_section = feedback_section[:MAX_FEEDBACK_CHARS] + "\n... [feedback truncated]\n"

        goal_prompt = f"""
GOAL: Containerize the '{repo_name}' repository.
{taxonomy_context}
OBJECTIVES:
1. Use ListDirectory to see what files exist (pyproject.toml, package.json, etc.)
2. Based on ONLY the files that exist, infer repo type, verification approach, and build strategy
3. Create a production-ready 'Dockerfile' in the repository root
4. Create a '.dockerignore' to exclude unnecessary files
5. CRITICAL: Use the 'VerifyBuild' tool to test your Dockerfile
6. You MUST get a SUCCESS result from VerifyBuild before finishing

ERROR HANDLING WORKFLOW (When VerifyBuild fails):
1. STOP. Do not edit the file yet.
2. Call SearchDockerError with the error message and your context.
   Example: SearchDockerError(error_keywords="...", agent_context="...")
3. Read the AI ANALYSIS section to understand:
   - Root Cause (what went wrong)
   - Fix (specific changes needed)
   - Example (code to apply)
4. Apply the suggested fix to your Dockerfile
5. Run VerifyBuild again
6. Repeat until SUCCESS

IMPORTANT RULES:
- DO NOT check for files that don't exist (e.g. checking package.json in a Python project)
- Use ListDirectory FIRST, then only read files you know exist
- Be efficient - don't waste API calls on trial-and-error file reads
- ALWAYS use SearchDockerError when VerifyBuild fails (don't guess fixes!)

CONTEXT:
{prep_context['context_str']}
{feedback_section}
You have full autonomy. Use your tools to explore, build, and VERIFY.
Do NOT ask for user permission. Just do it.

Begin!
"""

        try:
            active_handler = callback_handler if callback_handler else handler

            result = executor.invoke(
                {"input": goal_prompt},
                config={"callbacks": [active_handler]}
            )

            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])

            steps_used = len(intermediate_steps)
            logger.info(
                f"[Attempt {attempt}] steps_used={steps_used}/{current_max_iterations} "
                f"(utilization={steps_used/current_max_iterations*100:.0f}%)"
            )
            logger.info(f"Agent output: {output[:500]}...")

            # Track VerifyBuild and WriteToFile positions to enforce ordering
            verify_called = False
            last_verify_success = False
            last_verify_index = -1
            last_write_index = -1

            for idx, step in enumerate(intermediate_steps):
                action, observation = step

                if hasattr(action, 'tool') and action.tool == 'WriteToFile':
                    tool_input = action.tool_input
                    if isinstance(tool_input, str):
                        if 'Dockerfile' in tool_input:
                            last_write_index = idx
                    elif isinstance(tool_input, dict):
                        fpath = tool_input.get('file_path', '')
                        if 'Dockerfile' in fpath or fpath.endswith('Dockerfile'):
                            last_write_index = idx

                if hasattr(action, 'tool') and 'VerifyBuild' in action.tool:
                    verify_called = True
                    last_verify_index = idx
                    try:
                        if isinstance(observation, str):
                            obs_data = json.loads(observation.strip())
                            last_verify_success = (
                                isinstance(obs_data, dict) and obs_data.get("status") == "success"
                            )
                        else:
                            last_verify_success = False
                    except Exception:
                        last_verify_success = False

            if not verify_called:
                lesson = (
                    f"Attempt {attempt}: Agent did not call VerifyBuild. "
                    f"You MUST verify your Dockerfile before finishing."
                )
                logger.warning(lesson)
                _add_lesson(lesson)
                continue

            # Fix 1B: agent wrote AFTER a passing VerifyBuild → restore snapshot
            if last_write_index > last_verify_index:
                restored = False
                if last_verify_index >= 0:
                    _, verify_obs = intermediate_steps[last_verify_index]
                    try:
                        obs_data = json.loads(str(verify_obs).strip())
                        snapshot = obs_data.get("_verified_dockerfile_snapshot")
                        if snapshot:
                            dockerfile_path.write_text(snapshot, encoding='utf-8')
                            logger.info(
                                f"[Fix 1B] Restored verified Dockerfile snapshot for attempt {attempt} "
                                f"(agent wrote at step {last_write_index} after verify at step "
                                f"{last_verify_index})."
                            )
                            restored = True
                            return {
                                "status": "success",
                                "report_dir": str(report_dir),
                                "dockerfile": str(dockerfile_path),
                                "attempts": attempt,
                                "language": language,
                                "note": "Restored from verified snapshot (agent wrote after verify)"
                            }
                    except Exception as _snap_err:
                        logger.warning(f"[Fix 1B] Snapshot restore failed: {_snap_err}")

                if not restored:
                    lesson = (
                        f"Attempt {attempt}: You modified the Dockerfile "
                        f"(step {last_write_index}) AFTER verifying it "
                        f"(step {last_verify_index}). You must verify LAST."
                    )
                    logger.warning(lesson)
                    _add_lesson(lesson)
                    continue

            # Extract technical insight from tool outputs
            tech_lesson = extract_technical_lesson(
                intermediate_steps, base_llm, use_llm=use_llm_lessons
            )
            if tech_lesson:
                _add_lesson(f"TECHNICAL INSIGHT: {tech_lesson}")
                logger.info(f"Extracted technical lesson: {tech_lesson[:100]}...")

            if not last_verify_success:
                classification = classify_failure(lessons_learned, intermediate_steps)

                if classification["unfixable"]:
                    logger.warning(
                        f"[Classifier] Failure classified as UNFIXABLE "
                        f"({classification['type']}). Aborting remaining attempts."
                    )
                    _add_lesson(f"UNFIXABLE: {classification['hint']}")
                    break

                if classification["hint"]:
                    _add_lesson(
                        f"DIAGNOSIS ({classification['type']}): {classification['hint']}"
                    )

                lesson = (
                    f"Attempt {attempt}: VerifyBuild failed. "
                    f"Type: {classification['type']}. Fix and verify again."
                )
                logger.warning(lesson)
                _add_lesson(lesson)
                continue

            # Sanity-check: Dockerfile should always exist after a successful verify
            if not dockerfile_path.exists() or dockerfile_path.stat().st_size == 0:
                lesson = (
                    f"Attempt {attempt}: No Dockerfile produced. "
                    f"You MUST use WriteToFile to create a Dockerfile."
                )
                logger.warning(lesson)
                _add_lesson(lesson)
                continue

            logger.info("✓ VerifyBuild PASSED! Dockerfile successfully built and verified.")
            return {
                "status": "success",
                "report_dir": str(report_dir),
                "dockerfile": str(dockerfile_path),
                "attempts": attempt,
                "language": language
            }

        except Exception as e:
            lesson = f"Attempt {attempt}: Agent crashed with error: {str(e)}"
            logger.error(lesson)
            _add_lesson(lesson)
            continue

    logger.error(f"All {max_retries} attempts failed.")
    return {
        "status": "failure",
        "error": f"Failed after {max_retries} attempts",
        "lessons_learned": lessons_learned,
        "report_dir": str(report_dir),
        "dockerfile": str(dockerfile_path) if dockerfile_path.exists() else None
    }
