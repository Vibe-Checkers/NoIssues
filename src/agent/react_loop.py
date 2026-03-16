"""Phase 2: ReAct Agent Loop for BuildAgent v2.0.

- run_agent: outer loop (up to 3 iterations)
- run_iteration: single iteration (up to 15 steps) via langgraph react agent
- extract_lessons: summarize failed iteration for next attempt
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from langgraph.prebuilt import create_react_agent
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from db.models import StepRecord, IterationRecord, RunRecord, VerifyBuildResult
from db.writer import DBWriter
from agent.summarizer import summarize_output
from agent.tools import create_tools
from agent.verify_build import VerifyBuildTool

logger = logging.getLogger(__name__)

# ─── Prompts ─────────────────────────────────────────

SYSTEM_PROMPT = """\
You are BuildAgent, an expert at creating Dockerfiles for open-source repositories. \
Your goal is to generate a Dockerfile that builds the project from source and produces a working container.

CONTEXT BLUEPRINT:
{context_blueprint_json}

{lessons_section}

WORKFLOW:
1. Explore the repository structure using ListDirectory and ReadFile to understand what needs to be built.
2. Write a Dockerfile using WriteFile. The Dockerfile must:
   - Use an appropriate base image (the blueprint suggests: {base_image})
   - Install system dependencies if needed
   - Copy source code
   - Install project dependencies
   - Build the project from source
   - Set a proper CMD or ENTRYPOINT
3. Write a .dockerignore file to exclude unnecessary files (.git, node_modules, etc.).
4. Call VerifyBuild to test the Dockerfile. This is MANDATORY — never finish without calling VerifyBuild.
5. If VerifyBuild reports issues, read the error, fix the Dockerfile, and call VerifyBuild again.

RULES:
- You MUST call VerifyBuild at least once.
- Do NOT write a Dockerfile that only installs a runtime without building the project.
- If the blueprint's suggested base image doesn't work, use DockerImageSearch to find a better one.
- If a build error is unclear, use SearchWeb to find solutions.
- Be precise with COPY paths — check what files exist with ListDirectory before writing COPY instructions."""

LESSONS_TEMPLATE = """\
LESSONS FROM PREVIOUS ATTEMPTS:
{lessons_text}

Apply these lessons. Do not repeat the same mistakes."""

LESSON_EXTRACTOR_SYSTEM = (
    "You analyze failed Dockerfile generation attempts and extract lessons for the next attempt."
)

LESSON_EXTRACTOR_USER = """\
The following attempt to generate a Dockerfile failed after {step_count} steps.

STEP HISTORY:
{step_history}

Each step shows what the agent thought, what tool it called, and what it got back.

Write a concise lesson list for the next attempt. Include:
1. What was tried and why it failed (specific errors, not vague descriptions)
2. Package names, versions, or commands that caused problems
3. What to do differently (specific, actionable instructions)

Format as a numbered list. Max 400 words. Be direct — the next attempt will read this verbatim."""


# ═══════════════════════════════════════════════════════
# Agent Loop
# ═══════════════════════════════════════════════════════

def run_agent(
    repo_root: Path,
    blueprint: dict,
    llm,
    docker_ops,
    image_name: str,
    db: DBWriter,
    run_record: RunRecord,
    max_iterations: int = 3,
) -> RunRecord:
    """Outer loop: up to max_iterations attempts to generate a working Dockerfile."""
    lessons = None

    for iteration_num in range(1, max_iterations + 1):
        # Clean slate — delete Dockerfile between iterations
        dockerfile = repo_root / "Dockerfile"
        if dockerfile.exists():
            dockerfile.unlink()

        # Build prompt
        prompt = _build_prompt(blueprint, lessons)

        # Create tools for this iteration
        tools = create_tools(repo_root)
        verify_tool = VerifyBuildTool(
            repo_root=repo_root,
            image_name=image_name,
            docker_ops=docker_ops,
            llm=llm,
            blueprint=blueprint,
        )
        tools.append(verify_tool)

        # Create iteration record
        iteration = IterationRecord(
            run_id=run_record.id,
            iteration_number=iteration_num,
            status="running",
            injected_lessons=lessons,
        )
        db.write_iteration_start(iteration)

        # Run iteration
        iteration = run_iteration(
            prompt=prompt,
            tools=tools,
            llm=llm,
            db=db,
            iteration=iteration,
            verify_tool=verify_tool,
        )

        run_record.iterations.append(iteration)
        run_record.iteration_count = iteration_num

        if iteration.verify_result == "accepted":
            # Read the final Dockerfile
            if dockerfile.exists():
                run_record.final_dockerfile = dockerfile.read_text(errors="replace")
            run_record.smoke_test_passed = True
            return run_record

        # Extract lessons for next iteration
        if iteration_num < max_iterations:
            lessons, les_pt, les_ct = extract_lessons(iteration.steps, llm)
            iteration.lesson_extraction_tokens = (les_pt, les_ct)

        db.write_iteration_finish(iteration)

    return run_record


def run_iteration(
    prompt: str,
    tools: list,
    llm,
    db: DBWriter,
    iteration: IterationRecord,
    verify_tool: VerifyBuildTool,
    max_steps: int = 15,
) -> IterationRecord:
    """Single iteration: run langgraph react agent and extract steps."""
    t0 = time.monotonic()

    # Convert tools to LangChain StructuredTool format
    lc_tools = _to_langchain_tools(tools)

    try:
        # Create the react agent graph
        agent = create_react_agent(
            model=llm.nano,
            tools=lc_tools,
            prompt=prompt,
        )

        # Invoke with recursion limit (each tool call = 2 graph steps: AI + tool)
        result = agent.invoke(
            {"messages": [HumanMessage(
                content="Generate a Dockerfile for this repository. Follow the workflow above.",
            )]},
            config={"recursion_limit": max_steps * 2 + 1},
        )

        messages = result.get("messages", [])

        # Extract steps from the message history
        # Each step = one AIMessage with tool_calls + corresponding ToolMessage
        step_num = 0
        for i, msg in enumerate(messages):
            if not isinstance(msg, AIMessage) or not msg.tool_calls:
                continue

            for tool_call in msg.tool_calls:
                step_num += 1
                tool_name = tool_call.get("name", "unknown")
                tool_input = tool_call.get("args", {})
                thought = msg.content if msg.content else ""

                # Find the corresponding ToolMessage
                raw_output = ""
                tool_call_id = tool_call.get("id", "")
                for later_msg in messages[i + 1:]:
                    if (isinstance(later_msg, ToolMessage)
                            and later_msg.tool_call_id == tool_call_id):
                        raw_output = later_msg.content or ""
                        break

                # Summarize if needed
                output, sum_pt, sum_ct = summarize_output(raw_output, llm=llm)
                was_summarized = output != raw_output

                step = StepRecord(
                    step_number=step_num,
                    thought=thought,
                    tool_name=tool_name,
                    tool_input=tool_input if isinstance(tool_input, dict) else {"raw": str(tool_input)},
                    tool_output_raw=raw_output,
                    tool_output=output,
                    was_summarized=was_summarized,
                    summary_prompt_tokens=sum_pt,
                    summary_completion_tokens=sum_ct,
                )
                iteration.steps.append(step)
                db.write_step(iteration.id, step)

                # Track token usage
                iteration.prompt_tokens += step.prompt_tokens + sum_pt
                iteration.completion_tokens += step.completion_tokens + sum_ct

                # Check for VerifyBuild acceptance
                if tool_name == "VerifyBuild" and "accepted" in raw_output.lower():
                    iteration.verify_attempted = True
                    iteration.verify_result = "accepted"
                    iteration.dockerfile_generated = True
                elif tool_name == "VerifyBuild":
                    iteration.verify_attempted = True
                    if "rejected" in raw_output.lower():
                        iteration.verify_result = "rejected"
                    elif "build_failed" in raw_output.lower():
                        iteration.verify_result = "build_failed"
                    elif "smoke_failed" in raw_output.lower():
                        iteration.verify_result = "smoke_failed"

                if tool_name == "WriteFile" and "Dockerfile" in str(tool_input.get("path", "")):
                    iteration.dockerfile_generated = True

    except Exception as e:
        logger.error("Iteration %d failed with error: %s", iteration.iteration_number, e, exc_info=True)
        iteration.error_message = str(e)

    # Finalize iteration
    elapsed = int((time.monotonic() - t0) * 1000)
    iteration.duration_ms = elapsed
    iteration.finished_at = datetime.now(timezone.utc)

    if iteration.verify_result == "accepted":
        iteration.status = "success"
    elif iteration.error_message:
        iteration.status = "error"
    else:
        iteration.status = "failure"

    return iteration


# ═══════════════════════════════════════════════════════
# Lesson Extraction
# ═══════════════════════════════════════════════════════

def extract_lessons(steps: list[StepRecord], llm) -> tuple[str, int, int]:
    """Summarize a failed iteration's steps into lessons for the next attempt.

    Returns (lessons_text, prompt_tokens, completion_tokens).
    """
    step_history = _format_step_history(steps)

    try:
        response = llm.call_chat([
            {"role": "system", "content": LESSON_EXTRACTOR_SYSTEM},
            {"role": "user", "content": LESSON_EXTRACTOR_USER.format(
                step_count=len(steps),
                step_history=step_history,
            )},
        ])
        return response.content, response.prompt_tokens, response.completion_tokens
    except Exception:
        logger.warning("Lesson extraction LLM call failed, using fallback", exc_info=True)
        return _fallback_lessons(steps), 0, 0


def _format_step_history(steps: list[StepRecord]) -> str:
    """Format steps for the lesson extractor prompt."""
    lines = []
    for s in steps:
        lines.append(f"Step {s.step_number} [{s.tool_name}]")
        lines.append(f"  Thought: {s.thought[:200]}")
        lines.append(f"  Input: {json.dumps(s.tool_input)[:200]}")
        lines.append(f"  Result: {s.tool_output[:200]}")
        lines.append("")
    return "\n".join(lines)


def _fallback_lessons(steps: list[StepRecord]) -> str:
    """Build raw lessons from the last VerifyBuild error."""
    last_verify = [s for s in steps if s.tool_name == "VerifyBuild"]
    if last_verify:
        return f"Previous attempt failed. Last VerifyBuild error:\n{last_verify[-1].tool_output[:1000]}"
    return (
        f"Previous attempt exhausted {len(steps)} steps without calling VerifyBuild. "
        "Make sure to write a Dockerfile and call VerifyBuild."
    )


# ─── Helpers ─────────────────────────────────────────

def _build_prompt(blueprint: dict, lessons: str | None) -> str:
    """Build the system prompt with blueprint and optional lessons."""
    lessons_section = ""
    if lessons:
        lessons_section = LESSONS_TEMPLATE.format(lessons_text=lessons)

    base_image = blueprint.get("base_image", "auto-detect")
    blueprint_json = json.dumps(blueprint, indent=2)

    return SYSTEM_PROMPT.format(
        context_blueprint_json=blueprint_json,
        lessons_section=lessons_section,
        base_image=base_image,
    )


def _to_langchain_tools(tools: list) -> list[StructuredTool]:
    """Convert our tool classes to LangChain StructuredTool instances."""
    lc_tools = []
    for tool in tools:
        # Build the execution function based on the tool's args_schema
        schema = tool.args_schema
        field_names = list(schema.model_fields.keys())

        if not field_names:
            # No-arg tool (VerifyBuild)
            func = tool.execute
        else:
            func = lambda _t=tool, **kwargs: _t.execute(**kwargs)

        lc_tool = StructuredTool(
            name=tool.name,
            description=tool.description,
            func=func,
            args_schema=schema,
        )
        lc_tools.append(lc_tool)

    return lc_tools
