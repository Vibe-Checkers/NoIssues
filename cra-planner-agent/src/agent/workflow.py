
import os
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from .core import create_learner_agent
from .preparation import build_initial_context
from .tools import _set_report_directory

logger = logging.getLogger(__name__)


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
    _set_report_directory(str(report_dir))
    
    # 1. Preparation Phase
    logger.info(f"Starting analysis for {repo_name}...")
    
    # Create the main agent
    executor, handler = create_learner_agent(
        repository_path=repo_path,
        repo_name=repo_name,
        verbose=True,
        extra_tools=extra_tools
    )
    
    # Build Context
    prep_context = build_initial_context(executor, repo_path)
    language = prep_context["language"]
    
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
Use SearchDockerError to research solutions for the errors above.
═══════════════════════════════════════════════════════════════════════════════
"""
        
        # Define Goal Prompt
        goal_prompt = f"""
GOAL: Containerize the '{repo_name}' repository.

OBJECTIVES:
1. Analyze the project structure and dependencies.
2. Create a production-ready 'Dockerfile' in the repository root.
3. Create a '.dockerignore' to exclude unnecessary files.
4. CRITICAL: Use the 'VerifyBuild' tool to test your Dockerfile. 
   - If it fails, use SearchDockerError to research the error
   - Fix the Dockerfile based on what you learned
   - Verify again until you get SUCCESS
5. You MUST get a SUCCESS result from VerifyBuild before finishing.

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
            logger.info(f"Agent output: {output[:500]}...")
            
            # 1. Basic Validation: Check if Dockerfile exists
            if not dockerfile_path.exists() or dockerfile_path.stat().st_size == 0:
                lesson = f"Attempt {attempt}: No Dockerfile produced. You MUST use WriteToFile to create a Dockerfile."
                logger.warning(lesson)
                lessons_learned.append(lesson)
                continue
            
            # 2. Advanced Validation: External Callback (e.g. Docker Build)
            if validation_callback:
                logger.info(f"Running external validation...")
                valid_result = validation_callback(repo_path)
                
                if valid_result.get("success"):
                    logger.info("✓ External validation PASSED!")
                    return {
                        "status": "success", 
                        "report_dir": str(report_dir),
                        "dockerfile": str(dockerfile_path),
                        "attempts": attempt,
                        "metrics": valid_result
                    }
                else:
                    error_msg = valid_result.get("error", "Validation failed")
                    lesson = f"Attempt {attempt}: Build failed - {error_msg}"
                    logger.warning(f"✗ External validation FAILED: {error_msg}")
                    lessons_learned.append(lesson)
                    continue
            
            # If no validation callback, success if file exists
            logger.info("✓ Success! Dockerfile generated (no external validation).")
            return {
                "status": "success", 
                "report_dir": str(report_dir),
                "dockerfile": str(dockerfile_path),
                "attempts": attempt
            }
                
        except Exception as e:
            lesson = f"Attempt {attempt}: Agent crashed with error: {str(e)}"
            logger.error(lesson)
            lessons_learned.append(lesson)
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
