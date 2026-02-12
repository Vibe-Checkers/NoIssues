
import os
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from .core import create_learner_agent, create_test_agent
from .preparation import build_initial_context
# from .tools import _set_report_directory

logger = logging.getLogger(__name__)


def _validate_agent_steps(intermediate_steps, attempt, phase_name="build"):
    """
    Validate that the agent called VerifyBuild and it succeeded.
    
    Returns:
        (success: bool, lesson: str or None)
    """
    verify_called = False
    last_verify_success = False
    last_verify_index = -1
    last_write_index = -1

    for idx, step in enumerate(intermediate_steps):
        action, observation = step

        if hasattr(action, 'tool') and action.tool == 'WriteToFile':
            tool_input = action.tool_input
            if isinstance(tool_input, str):
                if 'Dockerfile' in tool_input or 'run_tests' in tool_input:
                    last_write_index = idx
            elif isinstance(tool_input, dict):
                fpath = tool_input.get('file_path', '')
                if 'Dockerfile' in fpath or 'run_tests' in fpath:
                    last_write_index = idx

        if hasattr(action, 'tool') and 'VerifyBuild' in action.tool:
            verify_called = True
            last_verify_index = idx

            try:
                if isinstance(observation, str):
                    obs_data = json.loads(observation.strip())
                    if isinstance(obs_data, dict):
                        status = obs_data.get("status", "")
                        # Phase A: "success" or "incomplete" both count
                        # Phase B: only "success" counts
                        if phase_name == "build":
                            last_verify_success = status in ("success", "incomplete")
                        else:
                            last_verify_success = status == "success"
                    else:
                        last_verify_success = False
                else:
                    last_verify_success = False
            except Exception:
                last_verify_success = False

    if not verify_called:
        return False, f"Attempt {attempt} ({phase_name}): Agent did not call VerifyBuild. You MUST verify before finishing."

    if last_write_index > last_verify_index:
        return False, (
            f"Attempt {attempt} ({phase_name}): Modified files (step {last_write_index})"
            f" AFTER verifying (step {last_verify_index}). You must verify LAST."
        )

    if not last_verify_success:
        return False, f"Attempt {attempt} ({phase_name}): VerifyBuild failed. You must keep fixing until status='success'."

    return True, None


def _run_phase(executor, handler, goal_prompt, callback_handler, phase_name, attempt, max_retries):
    """
    Run a single agent phase and validate the result.
    
    Returns:
        (success: bool, lesson: str or None, intermediate_steps: list)
    """
    try:
        active_handler = callback_handler if callback_handler else handler
        
        result = executor.invoke(
            {"input": goal_prompt}, 
            config={"callbacks": [active_handler]}
        )
        
        output = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])
        
        logger.info(f"[{phase_name}] Agent output: {output[:500]}...")
        
        success, lesson = _validate_agent_steps(intermediate_steps, attempt, phase_name)
        return success, lesson, intermediate_steps
        
    except Exception as e:
        lesson = f"Attempt {attempt} ({phase_name}): Agent crashed with error: {str(e)}"
        logger.error(lesson)
        return False, lesson, []


def run_learner_agent(
    repo_path: str,
    repo_name: str,
    repo_url: str,
    max_retries: int = 5,
    callback_handler = None,
    validation_callback = None,
    extra_tools: list = None
) -> Dict[str, Any]:
    """
    Run the two-phase containerization + test suite agent.
    
    Phase A: Create Dockerfile + verify build succeeds.
    Phase B: Discover test command, create run_tests.sh, verify tests pass.
    
    Args:
        repo_path: Path to the cloned repository
        repo_name: Name of the repository
        repo_url: URL of the repository
        max_retries: Maximum number of refinement attempts per phase
        callback_handler: Optional callback handler for logging
        validation_callback: Optional function to validate the build
        extra_tools: Additional tools to provide to the agent (e.g., VerifyBuild)
    
    Returns:
        Dict with status, report_dir, dockerfile path, and any errors
    """
    
    # Setup report directory
    ts = int(time.time())
    report_dir = Path(repo_path) / "agent_reports" / f"report_{ts}"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Preparation Phase
    logger.info(f"Starting analysis for {repo_name}...")
    
    # Create Phase A agent (containerization)
    build_executor, handler, base_llm = create_learner_agent(
        repository_path=repo_path,
        repo_name=repo_name,
        verbose=True,
        extra_tools=extra_tools
    )

    # Build Context with base LLM
    prep_context = build_initial_context(base_llm, repo_path)
    language = prep_context["language"]
    
    dockerfile_path = Path(repo_path) / "Dockerfile"
    run_tests_path = Path(repo_path) / "run_tests.sh"
    
    # =========================================================================
    # PHASE A: Containerization (Dockerfile + build verification)
    # =========================================================================
    logger.info(f"=== PHASE A: Containerization (max {max_retries} attempts) ===")
    
    build_lessons: List[str] = []
    phase_a_success = False
    phase_a_attempts = 0
    
    for attempt in range(1, max_retries + 1):
        phase_a_attempts = attempt
        logger.info(f"=== Phase A - Attempt {attempt}/{max_retries} ===")
        
        feedback_section = ""
        if build_lessons:
            feedback_section = f"""

═══════════════════════════════════════════════════════════════════════════════
⚠️ PREVIOUS ATTEMPT FAILED - FIX THESE ISSUES:
═══════════════════════════════════════════════════════════════════════════════
{"".join(f"• {lesson}" + chr(10) for lesson in build_lessons)}

REQUIRED ACTION:
1. Use SearchDockerError with the error keywords from above
2. Apply the fix and verify with VerifyBuild
═══════════════════════════════════════════════════════════════════════════════
"""
        
        goal_prompt = f"""
GOAL: Containerize the '{repo_name}' repository.

OBJECTIVES:
1. Use ListDirectory to see what files exist
2. Create a 'Dockerfile' that installs ALL dependencies (including dev/test deps)
3. Create a '.dockerignore' (do NOT exclude test directories or Makefile)
4. CRITICAL: Use 'VerifyBuild' to verify the Docker image builds successfully
5. Install build tools the project needs (git, make, gcc, etc.)

IMPORTANT RULES:
- READ the REPOSITORY ANALYSIS in the context below — it contains critical insights for this specific repo
- Use ListDirectory FIRST, then only read files you know exist
- Be efficient — don't waste API calls on trial-and-error file reads
- Include dev/test dependencies in the Dockerfile (they'll be needed for testing later)

WHEN VERIFYBUILD FAILS — MANDATORY:
- Call SearchDockerError IMMEDIATELY with error_keywords + full_error_log + dockerfile_content
- Apply the COMPLETE suggested fix — not just part of it (if it says chown -R /app, do the full path)
- If the SAME error appears after 2 attempts: try a fundamentally different approach, not another tweak

CONTEXT:
{prep_context['context_str']}
{feedback_section}
You have full autonomy. Use your tools to explore, build, and VERIFY.
Do NOT ask for user permission. Just do it.

Begin!
"""
        
        success, lesson, _ = _run_phase(
            build_executor, handler, goal_prompt, callback_handler,
            "build", attempt, max_retries
        )
        
        if not success:
            if lesson:
                build_lessons.append(lesson)
            continue
        
        # Check Dockerfile exists
        if not dockerfile_path.exists() or dockerfile_path.stat().st_size == 0:
            build_lessons.append(
                f"Attempt {attempt}: No Dockerfile produced. You MUST use WriteToFile."
            )
            continue
        
        logger.info("✓ Phase A PASSED! Dockerfile builds successfully.")
        phase_a_success = True
        break
    
    if not phase_a_success:
        logger.error(f"Phase A failed after {max_retries} attempts.")
        return {
            "status": "failure",
            "error": f"Phase A (containerization) failed after {max_retries} attempts",
            "phase": "build",
            "lessons_learned": build_lessons,
            "report_dir": str(report_dir),
            "dockerfile": str(dockerfile_path) if dockerfile_path.exists() else None,
            "preparation_token_usage": prep_context.get("preparation_token_usage", {}),
        }
    
    # =========================================================================
    # PHASE B: Test Suite (discover test command + run_tests.sh + verify tests)
    # =========================================================================
    logger.info(f"=== PHASE B: Test Suite (max {max_retries} attempts) ===")
    
    # Create Phase B agent (test suite)
    test_executor, test_handler, _ = create_test_agent(
        repository_path=repo_path,
        repo_name=repo_name,
        verbose=True,
        extra_tools=extra_tools
    )
    
    test_lessons: List[str] = []
    phase_b_success = False
    phase_b_attempts = 0
    
    for attempt in range(1, max_retries + 1):
        phase_b_attempts = attempt
        logger.info(f"=== Phase B - Attempt {attempt}/{max_retries} ===")
        
        feedback_section = ""
        if test_lessons:
            feedback_section = f"""

═══════════════════════════════════════════════════════════════════════════════
⚠️ PREVIOUS ATTEMPT FAILED - FIX THESE ISSUES:
═══════════════════════════════════════════════════════════════════════════════
{"".join(f"• {lesson}" + chr(10) for lesson in test_lessons)}

REQUIRED ACTION:
1. Call DiagnoseTestFailure with test_output + Dockerfile + run_tests.sh
2. Apply the fix and VerifyBuild again
═══════════════════════════════════════════════════════════════════════════════
"""
        
        # Include auto-discovered test command if available
        test_command_hint = ""
        if prep_context.get("test_command"):
            tc = prep_context["test_command"]
            test_command_hint = f"""

AUTO-DISCOVERED TEST COMMAND:
  Command: {tc.get('command', 'N/A')}
  Source: {tc.get('source', 'N/A')}
  Confidence: {tc.get('confidence', 'N/A')}
  
Use this as your primary test command unless you find a better one.
"""
        
        goal_prompt = f"""
GOAL: Create run_tests.sh for '{repo_name}' and verify the test suite passes.

A Dockerfile already exists and builds successfully. Your job:
1. Discover the test command (check CI workflows, Makefile, README, package.json, etc.)
2. Create 'run_tests.sh' that installs test deps and runs the test suite
3. Use 'VerifyBuild' — it rebuilds the image and runs run_tests.sh inside the container
4. If tests fail, call 'DiagnoseTestFailure' to get specific fix suggestions
5. You MUST get status="success" from VerifyBuild before finishing

WHEN TESTS FAIL — MANDATORY:
1. Call DiagnoseTestFailure with test_output + current Dockerfile + run_tests.sh
2. Apply the COMPLETE suggested fix — not just part of it
3. If the SAME test failure appears after 2 attempts: try a fundamentally different approach
   (e.g., different test command, different env vars, modify Dockerfile environment)
4. VerifyBuild again

You can also use RunInContainer to run diagnostic commands inside the container.

NOTE: The REPOSITORY ANALYSIS below contains test environment insights — use it.

CONTEXT:
{prep_context['context_str']}
{test_command_hint}
{feedback_section}
You have full autonomy. Explore, create run_tests.sh, and VERIFY.

Begin!
"""
        
        success, lesson, _ = _run_phase(
            test_executor, test_handler, goal_prompt, callback_handler,
            "test", attempt, max_retries
        )
        
        if not success:
            if lesson:
                test_lessons.append(lesson)
            continue
        
        # Check run_tests.sh exists
        if not run_tests_path.exists() or run_tests_path.stat().st_size == 0:
            test_lessons.append(
                f"Attempt {attempt}: No run_tests.sh produced. You MUST create run_tests.sh."
            )
            continue
        
        logger.info("✓ Phase B PASSED! Test suite runs successfully.")
        phase_b_success = True
        break
    
    if not phase_b_success:
        logger.error(f"Phase B failed after {max_retries} attempts.")
        return {
            "status": "partial",
            "error": f"Phase B (test suite) failed after {max_retries} attempts",
            "phase": "test",
            "build_success": True,
            "lessons_learned": test_lessons,
            "report_dir": str(report_dir),
            "dockerfile": str(dockerfile_path),
            "attempts": {"build": phase_a_attempts, "test": phase_b_attempts},
            "language": language,
            "preparation_token_usage": prep_context.get("preparation_token_usage", {}),
        }
    
    # Both phases succeeded
    logger.info("✓ BOTH PHASES PASSED! Dockerfile + test suite verified.")
    return {
        "status": "success",
        "report_dir": str(report_dir),
        "dockerfile": str(dockerfile_path),
        "run_tests": str(run_tests_path),
        "attempts": {"build": phase_a_attempts, "test": phase_b_attempts},
        "language": language,
        "preparation_token_usage": prep_context.get("preparation_token_usage", {}),
    }
