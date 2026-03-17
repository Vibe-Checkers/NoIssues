"""In-memory dataclasses that flow through the pipeline and get persisted to the database."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


def _new_id() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StepRecord:
    """One agent step: LLM reasoning + tool call + result."""

    id: str = field(default_factory=_new_id)
    step_number: int = 0
    started_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None
    duration_ms: int | None = None
    thought: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_output_raw: str = ""
    tool_output: str = ""
    was_summarized: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    summary_prompt_tokens: int = 0
    summary_completion_tokens: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tool_input"] = json.dumps(self.tool_input)
        return d


@dataclass
class VerifyBuildResult:
    """Result of a VerifyBuild tool invocation."""

    status: str = ""  # 'accepted', 'rejected', 'build_failed', 'smoke_failed'
    review_approved: bool = False
    review_concerns: list[str] = field(default_factory=list)
    smoke_test_commands: list[str] = field(default_factory=list)
    review_duration_ms: int | None = None
    build_success: bool | None = None
    build_error: str | None = None
    build_error_raw: str | None = None
    build_duration_ms: int | None = None
    smoke_results: list[dict] | None = None
    smoke_duration_ms: int | None = None
    dockerfile_snapshot: str | None = None
    # Token accounting
    review_tokens: tuple[int, int] = (0, 0)  # (prompt, completion)
    error_summary_tokens: tuple[int, int] = (0, 0)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["review_concerns"] = json.dumps(self.review_concerns)
        d["smoke_test_commands"] = json.dumps(self.smoke_test_commands)
        if self.smoke_results is not None:
            d["smoke_results"] = json.dumps(self.smoke_results)
        return d


@dataclass
class IterationRecord:
    """One iteration within a run (up to 3 per run)."""

    id: str = field(default_factory=_new_id)
    run_id: str = ""
    iteration_number: int = 0
    status: str = "pending"  # 'success', 'failure', 'error'
    started_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None
    duration_ms: int | None = None
    steps: list[StepRecord] = field(default_factory=list)
    injected_lessons: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    lesson_extraction_tokens: tuple[int, int] = (0, 0)  # (prompt, completion)
    dockerfile_generated: bool = False
    verify_attempted: bool = False
    verify_result: str | None = None  # 'accepted', 'rejected', 'build_failed', 'smoke_failed'
    error_message: str | None = None

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("steps")  # steps are written separately
        d["step_count"] = self.step_count
        d["lesson_extraction_tokens_prompt"] = self.lesson_extraction_tokens[0]
        d["lesson_extraction_tokens_completion"] = self.lesson_extraction_tokens[1]
        d.pop("lesson_extraction_tokens")
        return d


@dataclass
class RunRecord:
    """One row per repository processed."""

    id: str = field(default_factory=_new_id)
    batch_id: str | None = None
    repo_url: str = ""
    repo_slug: str = ""
    status: str = "pending"  # 'success', 'failure', 'error', 'skipped'
    started_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None
    duration_ms: int | None = None
    iteration_count: int = 0
    detected_language: str | None = None
    repo_type: str | None = None
    context_blueprint: str | None = None
    blueprint_tokens_prompt: int = 0
    blueprint_tokens_completion: int = 0
    blueprint_duration_ms: int | None = None
    final_dockerfile: str | None = None
    smoke_test_passed: bool | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_steps: int = 0
    error_message: str | None = None
    worker_id: int = 0
    iterations: list[IterationRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("iterations")  # iterations are written separately
        return d


@dataclass
class BatchRun:
    """Top-level grouping for a parallel batch execution."""

    id: str = field(default_factory=_new_id)
    started_at: datetime = field(default_factory=_now)
    finished_at: datetime | None = None
    worker_count: int = 0
    repo_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    config_json: str | None = None
    tag: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)
