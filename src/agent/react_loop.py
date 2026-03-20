"""Phase 2: ReAct Agent Loop for BuildAgent v2.0.

- run_agent: outer loop (up to 3 iterations)
- run_iteration: single iteration (up to 15 steps) via langgraph react agent
- extract_lessons: summarize failed iteration for next attempt
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from langgraph.prebuilt import create_react_agent
from langgraph.errors import GraphRecursionError
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from db.models import StepRecord, IterationRecord, RunRecord, VerifyBuildResult
from db.writer import DBWriter
from agent.blueprint import CollectedContext
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

REPOSITORY CONTEXT:
The repository file tree and key build file contents are provided in the first message below. \
You already have all the information needed to write the Dockerfile.

WORKFLOW:
1. Based on the repository context provided, write a Dockerfile using WriteFile. The Dockerfile must:
   - Use an appropriate base image (the blueprint suggests: {base_image})
   - Install system dependencies if needed
   - Copy source code
   - Install project dependencies
   - Build the project from source
   - Set a proper CMD or ENTRYPOINT
2. Write a .dockerignore file to exclude unnecessary files (.git, node_modules, etc.).
3. Call VerifyBuild to test the Dockerfile. This is MANDATORY — never finish without calling VerifyBuild.
4. If VerifyBuild reports issues, read the error carefully. Use ReadFile or ListDirectory ONLY to \
investigate specific files mentioned in the error. Fix the Dockerfile and call VerifyBuild again.

RULES:
- Your first action should be WriteFile to create the Dockerfile.
- Do NOT explore the repository — the file tree and build files are already provided.
- Do NOT call ListDirectory(".") or ReadFile unless VerifyBuild has failed and you need to check a specific file.
- You MUST call VerifyBuild at least once.
- Do NOT write a Dockerfile that only installs a runtime without building the project.
- If the blueprint's suggested base image doesn't work, use DockerImageSearch to find a better one.
- If a build error is unclear, use SearchWeb to find solutions."""

SYSTEM_PROMPT_SIMPLE = """\
You are BuildAgent, an expert at creating Dockerfiles for open-source repositories. \
Your goal is to generate a Dockerfile that builds the project from source and produces a working container.

{lessons_section}

REPOSITORY CONTEXT:
The repository file tree and key build file contents are provided in the first message below. \
You already have all the information needed to write the Dockerfile.

WORKFLOW:
1. Based on the repository context provided, write a Dockerfile using WriteFile. The Dockerfile must:
   - Use an appropriate base image
   - Install system dependencies if needed
   - Copy source code
   - Install project dependencies
   - Build the project from source
   - Set a proper CMD or ENTRYPOINT
2. Write a .dockerignore file to exclude unnecessary files (.git, node_modules, etc.).
3. Call VerifyBuild to test the Dockerfile. This is MANDATORY — never finish without calling VerifyBuild.
4. If VerifyBuild reports issues, read the error carefully. Use ReadFile or ListDirectory ONLY to \
investigate specific files mentioned in the error. Fix the Dockerfile and call VerifyBuild again.

RULES:
- Your first action should be WriteFile to create the Dockerfile.
- Do NOT explore the repository — the file tree and build files are already provided.
- Do NOT call ListDirectory(".") or ReadFile unless VerifyBuild has failed and you need to check a specific file.
- You MUST call VerifyBuild at least once.
- Do NOT write a Dockerfile that only installs a runtime without building the project.
- If a build error is unclear, use SearchWeb to find solutions."""

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
    collected_context: CollectedContext | None = None,
    ablation: str = "default",
) -> RunRecord:
    """Outer loop: up to max_iterations attempts to generate a working Dockerfile."""
    lessons = None
    initial_message = _build_initial_message(collected_context)

    for iteration_num in range(1, max_iterations + 1):
        # Clean slate — delete Dockerfile between iterations
        dockerfile = repo_root / "Dockerfile"
        if dockerfile.exists():
            dockerfile.unlink()

        # Build prompt
        prompt = _build_prompt(blueprint, lessons, ablation)

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
            max_steps=int(os.environ.get("MAX_STEPS_PER_ITERATION", "25")),
            initial_message=initial_message,
        )

        run_record.iterations.append(iteration)
        run_record.iteration_count = iteration_num

        if iteration.verify_result == "accepted":
            db.write_iteration_finish(iteration)
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
    max_steps: int = 25,
    initial_message: str = "Generate a Dockerfile for this repository. Follow the workflow above.",
) -> IterationRecord:
    """Single iteration: run langgraph react agent and extract steps."""
    t0 = time.monotonic()

    # Convert tools to LangChain StructuredTool format
    lc_tools = _to_langchain_tools(tools)

    # Create the react agent graph
    agent = create_react_agent(
        model=llm.nano,
        tools=lc_tools,
        prompt=_make_messages_modifier(prompt),
    )

    step_num = 0
    # tool_call_id -> {name, args, thought, started_at}
    pending: dict[str, dict] = {}

    try:
        for chunk in agent.stream(
            {"messages": [HumanMessage(content=initial_message)]},
            config={"recursion_limit": max_steps * 2 + 1},
            stream_mode="updates",
        ):
            # Agent node: LLM decided to call tool(s)
            if "agent" in chunk:
                for msg in chunk["agent"].get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        usage = msg.usage_metadata or {}
                        n = len(msg.tool_calls)
                        pt = usage.get("input_tokens", 0) // n
                        ct = usage.get("output_tokens", 0) // n
                        for tc in msg.tool_calls:
                            pending[tc["id"]] = {
                                "name": tc["name"],
                                "args": tc.get("args", {}),
                                "thought": msg.content or "",
                                "started_at": datetime.now(timezone.utc),
                                "prompt_tokens": pt,
                                "completion_tokens": ct,
                            }

            # Tools node: tool finished executing
            elif "tools" in chunk:
                for msg in chunk["tools"].get("messages", []):
                    if not isinstance(msg, ToolMessage):
                        continue
                    tc = pending.pop(msg.tool_call_id, None)
                    if tc is None:
                        continue

                    step_num += 1
                    raw_output = msg.content or ""
                    output, sum_pt, sum_ct = summarize_output(raw_output, llm=llm)

                    finished_at = datetime.now(timezone.utc)
                    duration_ms = int(
                        (finished_at - tc["started_at"]).total_seconds() * 1000
                    )
                    step = StepRecord(
                        step_number=step_num,
                        started_at=tc["started_at"],
                        finished_at=finished_at,
                        duration_ms=duration_ms,
                        thought=tc["thought"],
                        tool_name=tc["name"],
                        tool_input=tc["args"] if isinstance(tc["args"], dict) else {"raw": str(tc["args"])},
                        tool_output_raw=raw_output,
                        tool_output=output,
                        was_summarized=output != raw_output,
                        prompt_tokens=tc.get("prompt_tokens", 0),
                        completion_tokens=tc.get("completion_tokens", 0),
                        summary_prompt_tokens=sum_pt,
                        summary_completion_tokens=sum_ct,
                    )
                    iteration.steps.append(step)
                    db.write_step(iteration.id, step)

                    iteration.prompt_tokens += step.prompt_tokens + sum_pt
                    iteration.completion_tokens += step.completion_tokens + sum_ct

                    tool_name = tc["name"]
                    if tool_name == "VerifyBuild":
                        iteration.verify_attempted = True
                        if verify_tool._last_result is not None:
                            db.write_verify_detail(step.id, verify_tool._last_result)
                        if "accepted" in raw_output.lower():
                            iteration.verify_result = "accepted"
                            iteration.dockerfile_generated = True
                            break  # stop consuming the stream — we're done
                        elif "rejected" in raw_output.lower():
                            iteration.verify_result = "rejected"
                        elif "build_failed" in raw_output.lower():
                            iteration.verify_result = "build_failed"
                        elif "smoke_failed" in raw_output.lower():
                            iteration.verify_result = "smoke_failed"

                    if tool_name == "WriteFile" and "Dockerfile" in str(tc["args"].get("path", "")):
                        iteration.dockerfile_generated = True
                else:
                    continue
                break  # inner break propagates out of outer for-loop

    except GraphRecursionError:
        logger.warning(
            "Iteration %d hit recursion limit after %d steps (steps already saved)",
            iteration.iteration_number, step_num,
        )
    except Exception as e:
        logger.error("Iteration %d failed: %s", iteration.iteration_number, e, exc_info=True)
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

_OLD_TOOL_MSG_MAX = 4000  # chars to keep from non-recent tool messages


def _build_initial_message(collected_context: CollectedContext | None) -> str:
    """Build the initial HumanMessage content with pre-seeded repo context."""
    if collected_context is None or not collected_context.file_tree:
        return "Generate a Dockerfile for this repository. Follow the workflow above."

    parts = ["Generate a Dockerfile for this repository based on the context below."]

    parts.append("\n\n=== REPOSITORY FILE TREE ===")
    parts.append(collected_context.file_tree)

    if collected_context.readme and collected_context.readme != "(no README found)":
        parts.append("\n\n=== README ===")
        parts.append(collected_context.readme)

    if collected_context.files:
        parts.append("\n\n=== KEY BUILD FILES ===")
        for path, content in collected_context.files.items():
            parts.append(f"\n--- {path} ---")
            parts.append(content)

    parts.append("\n\nWrite the Dockerfile now. Do not explore — all needed context is above.")

    return "\n".join(parts)


def _make_messages_modifier(system_prompt: str):
    """Messages modifier: inject system prompt and truncate old tool outputs.

    Keeps the last 16 messages fully intact (8 tool exchanges). Older tool
    messages are truncated to _OLD_TOOL_MSG_MAX chars to prevent the
    accumulated history from exceeding the model's context limit.

    Note: langgraph passes the full state dict {"messages": [...]},
    not a bare message list.
    """
    def modifier(state) -> list:
        # langgraph passes state dict {"messages": [...]}, not bare list
        if isinstance(state, dict):
            messages = state.get("messages", [])
        else:
            messages = state

        keep_full = 16
        cutoff = max(0, len(messages) - keep_full)
        result = []
        for i, msg in enumerate(messages):
            if i < cutoff and isinstance(msg, ToolMessage):
                content = msg.content or ""
                if len(content) > _OLD_TOOL_MSG_MAX:
                    msg = ToolMessage(
                        content=content[:_OLD_TOOL_MSG_MAX] + "\n...[truncated]",
                        tool_call_id=msg.tool_call_id,
                    )
            result.append(msg)
        return [SystemMessage(content=system_prompt)] + result

    return modifier


def _build_prompt(blueprint: dict, lessons: str | None, ablation: str = "default") -> str:
    """Build the system prompt with blueprint and optional lessons."""
    lessons_section = ""
    if lessons:
        lessons_section = LESSONS_TEMPLATE.format(lessons_text=lessons)

    if ablation == "no-metaprompt":
        return SYSTEM_PROMPT_SIMPLE.format(
            lessons_section=lessons_section,
        )

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
