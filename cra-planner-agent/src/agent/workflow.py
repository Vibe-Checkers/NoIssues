
import os
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from .core import create_learner_agent
from langchain_core.messages import SystemMessage, HumanMessage
from .preparation import build_initial_context, _create_analysis_llm
# from .tools import _set_report_directory

logger = logging.getLogger(__name__)



def extract_technical_lesson(intermediate_steps: List[Any], llm: Any) -> Optional[str]:
    """
    Scan agent history to extract technical insights and uses LLM to summarize them.
    
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
                # meaningful analysis found
                try:
                    analysis_part = str(observation).split("=== AI ANALYSIS ===")[1].split("=== SEARCH SOURCES")[0].strip()
                    raw_context = f"SearchDockerError Analysis: {analysis_part}"
                    break
                except Exception:
                    pass
        
        # 2. Medium Priority: VerifyBuild (contains raw error info)
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

    # Use LLM to summarize the raw context into a concise lesson
    try:
        summary_prompt = f"""
        Analyze the following error context from a failed Docker build attempt.
        Summarize it into a SINGLE, CONCISE sentence that explains specific fix.
        Format: "Previous failure caused by [Cause]. Fix: [Action]."
        
        CONTEXT:
        {raw_context[:4000]}
        """
        
        response = llm.invoke([
            SystemMessage(content="You are a technical summarizer. Be extremely concise. Focus on the FIX."),
            HumanMessage(content=summary_prompt)
        ])
        
        return response.content.strip()
    except Exception as e:
        logger.warning(f"LLM summarization of lesson failed: {e}")
        return raw_context[:300] + "..." # Fallback to raw truncation

def run_learner_agent(
    repo_path: str,
    repo_name: str,
    repo_url: str,
    max_retries: int = 3,
    callback_handler = None,
    validation_callback = None,
    extra_tools: list = None
) -> Dict[str, Any]:
    """
    Run the Learner Agent on a repository with intelligent refinement.
    
    This function:
    1. Prepares context (Language detection, Guidelines, CI summary).
    2. Runs the agent with a high-level goal ("Containerize this").
    3. If the agent fails to produce a working Dockerfile, feeds back the error
       and re-invokes with accumulated lessons learned.
    
    Args:
        repo_path: Path to the cloned repository
        repo_name: Name of the repository
        repo_url: URL of the repository
        max_retries: Maximum number of refinement attempts (default: 3)
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
    # _set_report_directory removed (thread-local state elimination)
    
    # 1. Preparation Phase
    logger.info(f"Starting analysis for {repo_name}...")
    
    # Create the main agent (now returns base_llm too)
    executor, handler, base_llm = create_learner_agent(
        repository_path=repo_path,
        repo_name=repo_name,
        verbose=True,
        extra_tools=extra_tools
    )

    # Build Context with base LLM (not full agent)
    prep_context = build_initial_context(base_llm, repo_path)
    language = prep_context["language"]
    analysis_llm = _create_analysis_llm()

    # 2. Execution Loop with Refinement
    logger.info(f"=== Starting Agent Execution (max {max_retries} attempts) ===")
    
    lessons_learned: List[str] = []
    dockerfile_path = Path(repo_path) / "Dockerfile"
    
    for attempt in range(1, max_retries + 1):
        logger.info(f"=== Attempt {attempt}/{max_retries} ===")
        
        # Build feedback section from previous attempts
        feedback_section = ""
        if lessons_learned:
            feedback_section = f"""

═══════════════════════════════════════════════════════════════════════════════
⚠️ PREVIOUS ATTEMPT FAILED - YOU MUST FIX THESE ISSUES:
═══════════════════════════════════════════════════════════════════════════════
{"".join(f"• {lesson}" + chr(10) for lesson in lessons_learned)}

REQUIRED ACTION:
1. Use SearchDockerError with the error keywords from above
2. The AI will analyze the error and provide a specific fix
3. Apply the fix and verify with VerifyBuild
═══════════════════════════════════════════════════════════════════════════════
"""
        
        # Define Goal Prompt
        goal_prompt = f"""
GOAL: Containerize the '{repo_name}' repository.

OBJECTIVES:
1. Use ListDirectory to see what files exist (pyproject.toml, package.json, etc.)
2. Based on ONLY the files that exist, determine build approach
3. Create a production-ready 'Dockerfile' in the repository root
4. Create a '.dockerignore' to exclude unnecessary files
5. CRITICAL: Use the 'VerifyBuild' tool to test your Dockerfile
6. You MUST get a SUCCESS result from VerifyBuild before finishing

ERROR HANDLING WORKFLOW (When VerifyBuild fails):
1. STOP. Do not edit the file yet.
2. Call SearchDockerError with the error message and your context.
   Example: SearchDockerError(error_keywords="...", agent_context="...")
2. Read the AI ANALYSIS section to understand:
   - Root Cause (what went wrong)
   - Fix (specific changes needed)
   - Example (code to apply)
3. Apply the suggested fix to your Dockerfile
4. Run VerifyBuild again
5. Repeat until SUCCESS

IMPORTANT RULES:
- DO NOT check for files that don't exist (e.g. checking package.json in a Python project)
- Use ListDirectory FIRST, then only read files you know exist
- Be efficient - don't waste API calls on trial-and-error file reads
- ALWAYS use SearchDockerError when VerifyBuild fails (don't guess fixes!)
- The CONTEXT below includes a DOCKERFILE STRATEGY pre-analyzed by GPT-5 with a
  verified base image — use it as your starting point when writing the Dockerfile

CONTEXT:
{prep_context['context_str']}
{feedback_section}
You have full autonomy. Use your tools to explore, build, and VERIFY.
Do NOT ask for user permission. Just do it.

Begin!
"""
        
        try:
            # Use the provided callback handler if available, otherwise use the default
            active_handler = callback_handler if callback_handler else handler
            
            # Run Agent
            result = executor.invoke(
                {"input": goal_prompt}, 
                config={"callbacks": [active_handler]}
            )
            
            output = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])

            logger.info(f"Agent output: {output[:500]}...")

            # ========== NEW: Enforce VerifyBuild was called and succeeded ==========
            verify_called = False
            last_verify_success = False
            last_verify_index = -1
            last_write_index = -1
            
            # Track steps to enforce sequence
            for idx, step in enumerate(intermediate_steps):
                action, observation = step
                
                # Check for WriteToFile targeting Dockerfile
                if hasattr(action, 'tool') and action.tool == 'WriteToFile':
                    # Parse input to see if it's the Dockerfile
                    # Handle both dict and string input formats
                    tool_input = action.tool_input
                    if isinstance(tool_input, str):
                        # Heuristic for string input
                        if 'Dockerfile' in tool_input:
                            last_write_index = idx
                    elif isinstance(tool_input, dict):
                        fpath = tool_input.get('file_path', '')
                        if 'Dockerfile' in fpath or fpath.endswith('Dockerfile'):
                            last_write_index = idx

                # Check for VerifyBuild
                if hasattr(action, 'tool') and 'VerifyBuild' in action.tool:
                    verify_called = True
                    last_verify_index = idx
                    
                    # STRICT CHECK: Parse JSON observation
                    # The observation is a string (JSON dump), we need to parse it
                    try:
                        if isinstance(observation, str):
                            obs_data = json.loads(observation.strip())
                            if isinstance(obs_data, dict) and obs_data.get("status") == "success":
                                last_verify_success = True
                            else:
                                last_verify_success = False
                        else:
                            last_verify_success = False
                    except Exception:
                        # Fallback for non-JSON output (shouldn't happen with updated tool)
                        last_verify_success = False

            if not verify_called:
                lesson = f"Attempt {attempt}: Agent did not call VerifyBuild. You MUST verify your Dockerfile before finishing."
                logger.warning(lesson)
                lessons_learned.append(lesson)
                continue

            # Check sequence: Verification must happen AFTER the last write
            if last_write_index > last_verify_index:
                 lesson = f"Attempt {attempt}: You modified the Dockerfile (step {last_write_index}) AFTER verifying it (step {last_verify_index}). You must verify LAST."
                 logger.warning(lesson)
                 lessons_learned.append(lesson)
                 continue

            # Enforce SearchDockerError was called after every failed VerifyBuild.
            # If the agent tried to self-fix without searching, inject a hard lesson.
            last_failed_verify_idx = -1
            search_called_after_fail = False
            for idx, (action, observation) in enumerate(intermediate_steps):
                if hasattr(action, 'tool') and 'VerifyBuild' in action.tool:
                    try:
                        obs_data = json.loads(str(observation))
                        if obs_data.get("status") == "failed":
                            last_failed_verify_idx = idx
                            search_called_after_fail = False
                    except Exception:
                        pass
                if (hasattr(action, 'tool') and 'SearchDockerError' in action.tool
                        and last_failed_verify_idx >= 0 and idx > last_failed_verify_idx):
                    search_called_after_fail = True

            if last_failed_verify_idx >= 0 and not search_called_after_fail:
                lesson = (
                    f"Attempt {attempt}: VerifyBuild failed but you edited the Dockerfile "
                    f"without calling SearchDockerError first. MANDATORY: after every "
                    f"VerifyBuild failure you MUST call SearchDockerError before touching the file."
                )
                logger.warning(lesson)
                lessons_learned.append(lesson)

            # Extract technical insight from tool outputs
            tech_lesson = extract_technical_lesson(intermediate_steps, analysis_llm)
            if tech_lesson:
                lessons_learned.append(f"TECHNICAL INSIGHT: {tech_lesson}")
                logger.info(f"Extracted technical lesson: {tech_lesson[:100]}...")

            if not last_verify_success:
                lesson = f"Attempt {attempt}: VerifyBuild failed or was not clean success. You must keep fixing until status='success'."
                logger.warning(lesson)
                lessons_learned.append(lesson)
                continue
            # ========== END NEW CODE ==========

            # 1. Basic Validation: Check if Dockerfile exists
            if not dockerfile_path.exists() or dockerfile_path.stat().st_size == 0:
                lesson = f"Attempt {attempt}: No Dockerfile produced. You MUST use WriteToFile to create a Dockerfile."
                logger.warning(lesson)
                lessons_learned.append(lesson)
                continue

            # SUCCESS! VerifyBuild passed, which means Docker build succeeded
            # No need for external validation since VerifyBuild already did the Docker build
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
            lessons_learned.append(lesson)
            # Executor-level shutdown errors won't recover across retries — bail early
            if "interpreter shutdown" in str(e).lower() or "cannot schedule new futures" in str(e).lower():
                logger.error("Non-recoverable executor error — stopping retries early")
                break
            continue
    
    # All attempts exhausted
    logger.error(f"All {max_retries} attempts failed.")
    return {
        "status": "failure", 
        "error": f"Failed after {max_retries} attempts",
        "lessons_learned": lessons_learned,
        "report_dir": str(report_dir),
        "dockerfile": str(dockerfile_path) if dockerfile_path.exists() else None
    }
