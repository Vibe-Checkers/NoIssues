
import os
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from .core import create_learner_agent, create_test_agent
from .preparation import build_initial_context
from .tools import reset_search_docker_error_counter

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
        reset_search_docker_error_counter()
        logger.info(f"=== Phase A - Attempt {attempt}/{max_retries} ===")
        
        feedback_section = ""
        if build_lessons:
            # Carry forward: include the FULL Dockerfile content and explain what went wrong
            last_dockerfile_content = ""
            if dockerfile_path.exists():
                try:
                    content = dockerfile_path.read_text(encoding='utf-8')
                    # Include full Dockerfile — agent needs complete context
                    if len(content) > 8000:
                        content = content[:8000] + "\n# ... [truncated]"
                    last_dockerfile_content = f"""
YOUR PREVIOUS DOCKERFILE (this did NOT build successfully — fix the issues below):
```dockerfile
{content}
```
"""
                except Exception:
                    pass

            feedback_section = f"""

═══════════════════════════════════════════════════════════════════════════════
⚠️ ATTEMPT {attempt-1} FAILED — HERE IS WHAT HAPPENED AND WHAT TO FIX:
═══════════════════════════════════════════════════════════════════════════════

FAILURE REASON:
{"".join(f"• {lesson}" + chr(10) for lesson in build_lessons[-3:])}
{last_dockerfile_content}
WHAT YOU MUST DO NOW:
1. The Dockerfile above FAILED to build. Read it and understand the error.
2. Fix the SPECIFIC issue described above — do NOT rewrite from scratch.
3. If the error is about a missing package: add it to apt-get install.
4. If the error is about a wrong base image: change the FROM line.
5. If you've seen the SAME error twice: try a completely different approach
   (different base image, different install strategy, etc.)
6. Write the fixed Dockerfile with WriteToFile and run VerifyBuild.
═══════════════════════════════════════════════════════════════════════════════
"""

        goal_prompt = f"""
GOAL: Containerize the '{repo_name}' repository.

OBJECTIVES:
{"1. READ YOUR PREVIOUS DOCKERFILE above — it failed to build. Fix the specific error." if build_lessons else "1. Use ListDirectory to see what files exist"}
2. Create a 'Dockerfile' that installs ALL dependencies (including dev/test deps)
3. Create a '.dockerignore' (do NOT exclude test directories or Makefile)
4. CRITICAL: Use 'VerifyBuild' to verify the Docker image builds successfully
5. Install build tools the project needs (git, make, gcc, etc.)

IMPORTANT RULES:
- READ the REPOSITORY ANALYSIS in the context below — it contains critical insights for this specific repo
{"- You already explored the repo in the previous attempt. Skip ListDirectory and go straight to fixing the Dockerfile." if build_lessons else "- Use ListDirectory FIRST, then only read files you know exist"}
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
        reset_search_docker_error_counter()
        logger.info(f"=== Phase B - Attempt {attempt}/{max_retries} ===")
        
        feedback_section = ""
        if test_lessons:
            # Carry forward: include FULL run_tests.sh and Dockerfile
            last_run_tests = ""
            if run_tests_path.exists():
                try:
                    content = run_tests_path.read_text(encoding='utf-8')
                    last_run_tests = f"""
YOUR PREVIOUS run_tests.sh (this FAILED — fix it):
```bash
{content}
```
"""
                except Exception:
                    pass

            last_dockerfile = ""
            if dockerfile_path.exists():
                try:
                    content = dockerfile_path.read_text(encoding='utf-8')
                    last_dockerfile = f"""
YOUR WORKING DOCKERFILE (builds successfully — only modify if tests need extra system deps):
```dockerfile
{content}
```
"""
                except Exception:
                    pass

            feedback_section = f"""

═══════════════════════════════════════════════════════════════════════════════
⚠️ ATTEMPT {attempt-1} FAILED — THE BUILD WORKS BUT TESTS FAILED:
═══════════════════════════════════════════════════════════════════════════════

The Dockerfile builds successfully. Do NOT break it.
The problem is in run_tests.sh or missing test dependencies.

FAILURE REASON:
{"".join(f"• {lesson}" + chr(10) for lesson in test_lessons[-3:])}
{last_run_tests}
{last_dockerfile}
WHAT YOU MUST DO NOW:
1. The Dockerfile WORKS. Focus on fixing run_tests.sh.
2. If tests need extra system packages (e.g. chromium, xvfb), add them to the Dockerfile.
3. If the test command is wrong, fix it in run_tests.sh.
4. If tests need env vars, add ENV lines to the Dockerfile.
5. Call DiagnoseTestFailure if you need help understanding the error.
6. Write the fixed file(s) and run VerifyBuild.
═══════════════════════════════════════════════════════════════════════════════
"""
        
        # Include auto-discovered test command with full GPT-5 analysis
        test_command_hint = ""
        if prep_context.get("test_command"):
            tc = prep_context["test_command"]
            setup_cmds = tc.get('setup_commands', [])
            env_vars = tc.get('env_vars', {})
            skip_pats = tc.get('skip_patterns', [])

            setup_section = ""
            if setup_cmds:
                setup_section = "\n  Setup commands (run BEFORE test command in run_tests.sh):\n"
                for cmd in setup_cmds:
                    setup_section += f"    - {cmd}\n"

            env_section = ""
            if env_vars:
                env_section = "\n  ENV vars (add to Dockerfile as ENV lines):\n"
                for k, v in env_vars.items():
                    env_section += f"    ENV {k}={v}\n"

            skip_section = ""
            if skip_pats:
                skip_section = "\n  Tests to skip in Docker (add to test command):\n"
                for pat in skip_pats:
                    skip_section += f"    - {pat}\n"

            build_section = ""
            if tc.get('needs_build_first') and tc.get('build_command'):
                build_section = f"\n  Build before test: {tc['build_command']}\n"

            test_command_hint = f"""

AUTO-DISCOVERED TEST COMMAND (GPT-5 analyzed):
  Command: {tc.get('command', 'N/A')}
  Framework: {tc.get('test_framework', 'unknown')}
  Source: {tc.get('source', 'N/A')}
  Confidence: {tc.get('confidence', 'N/A')}
{setup_section}{env_section}{skip_section}{build_section}
  Notes: {tc.get('notes', 'None')}

IMPORTANT: Use this as your primary test command. The setup_commands MUST go in
run_tests.sh BEFORE the test command. The ENV vars MUST go in the Dockerfile.
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
