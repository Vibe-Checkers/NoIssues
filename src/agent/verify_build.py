"""VerifyBuild tool for BuildAgent v2.0.

Reads the Dockerfile, calls LLM reviewer for approval + smoke test design,
builds the image, runs smoke tests, and returns a VerifyBuildResult.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pydantic import BaseModel

from db.models import VerifyBuildResult
from agent.summarizer import summarize_output

logger = logging.getLogger(__name__)

# ─── Prompts ─────────────────────────────────────────

REVIEWER_SYSTEM = "You review Dockerfiles and design smoke tests for built containers."

REVIEWER_USER = """\
REPOSITORY TYPE: {repo_type}
LANGUAGE: {language}

DOCKERFILE:
{dockerfile_content}

TASK 1 — REVIEW:
Decide if this Dockerfile should be built. It should be APPROVED if:
- It builds the application from source (not just installing a runtime with no build steps)
- The FROM image looks valid
- COPY, RUN, and CMD instructions are reasonable
- There are no obvious syntax errors

It should be REJECTED if:
- It only installs a language runtime without building anything
- It has clearly broken instructions (copying files that don't exist, missing FROM)
- It's essentially empty or placeholder

TASK 2 — SMOKE TESTS:
Design 1 to 3 shell commands to verify the container works after building. \
These commands will run inside the container with \
`docker run --rm --entrypoint "" <image> sh -c "<command>"`.

Examples by repo type:
- Library: python -c "import {{package}}; print('ok')" or node -e "require('{{package}}')"
- CLI tool: /app/binary --version or which binary_name
- Web service: test -f /app/server or ls /app/dist/index.html
- Compiled project: find /app -name "*.jar" | head -1 or test -x /app/build/main

Return ONLY valid JSON:
{{
  "approved": true or false,
  "concerns": ["<issue1>", "<issue2>"],
  "smoke_test_commands": ["<cmd1>", "<cmd2>"]
}}

smoke_test_commands must have 1 to 3 commands. Never return an empty list."""

REVIEW_FALLBACK = {
    "approved": True,
    "concerns": ["LLM review failed — building without review"],
    "smoke_test_commands": ["ls /app || ls /usr/src || echo 'checking root' && ls /"],
}


# ═══════════════════════════════════════════════════════
# VerifyBuild Tool
# ═══════════════════════════════════════════════════════

class VerifyBuildInput(BaseModel):
    pass  # No parameters — reads Dockerfile from repo root


class VerifyBuildTool:
    """Build the Dockerfile and run smoke tests."""

    name = "VerifyBuild"
    description = (
        "Build the Dockerfile and run smoke tests. Input: {} "
        "(no parameters — reads the Dockerfile from the repo root). "
        "Returns build status, smoke test results, and any errors. YOU MUST CALL THIS."
    )
    args_schema = VerifyBuildInput

    def __init__(self, repo_root: Path, image_name: str, docker_ops, llm,
                 blueprint: dict | None = None):
        self.repo_root = repo_root
        self.image_name = image_name
        self.docker_ops = docker_ops
        self.llm = llm
        self.blueprint = blueprint or {}

    def execute(self) -> str:
        """Run the full VerifyBuild pipeline. Returns a JSON string summary."""
        result = self._run()
        # Return a concise summary for the agent
        return self._format_for_agent(result)

    def _run(self) -> VerifyBuildResult:
        """Internal: full verify pipeline returning VerifyBuildResult."""
        # Step 1: Read Dockerfile
        dockerfile_path = self.repo_root / "Dockerfile"
        if not dockerfile_path.is_file():
            return VerifyBuildResult(
                status="rejected",
                review_approved=False,
                review_concerns=["No Dockerfile found at repo root"],
            )

        dockerfile_content = dockerfile_path.read_text(errors="replace")

        # Step 2: LLM review
        review = self._review_dockerfile(dockerfile_content)
        review_tokens = review.pop("_tokens", (0, 0))

        approved = review.get("approved", False)
        concerns = review.get("concerns", [])
        smoke_commands = review.get("smoke_test_commands", [])

        # Enforce at least 1 smoke test
        if not smoke_commands:
            smoke_commands = ["echo 'no smoke test designed'"]

        # Sanitize commands
        smoke_commands = [cmd.strip().strip("`").strip("'\"") for cmd in smoke_commands]

        if not approved:
            return VerifyBuildResult(
                status="rejected",
                review_approved=False,
                review_concerns=concerns,
                smoke_test_commands=smoke_commands,
                dockerfile_snapshot=dockerfile_content,
                review_tokens=review_tokens,
            )

        # Step 3: Docker build
        success, build_error, build_duration = self.docker_ops.build(
            str(self.repo_root), self.image_name,
        )

        if not success:
            # Summarize error if needed
            build_error_raw = build_error
            summarized_error, err_pt, err_ct = summarize_output(
                build_error, context_type="build_error", llm=self.llm,
            )
            return VerifyBuildResult(
                status="build_failed",
                review_approved=True,
                review_concerns=concerns,
                smoke_test_commands=smoke_commands,
                build_success=False,
                build_error=summarized_error,
                build_error_raw=build_error_raw,
                build_duration_ms=build_duration,
                dockerfile_snapshot=dockerfile_content,
                review_tokens=review_tokens,
                error_summary_tokens=(err_pt, err_ct),
            )

        # Step 4: Run smoke tests
        smoke_results = []
        all_passed = True
        for cmd in smoke_commands:
            exit_code, output, timed_out = self.docker_ops.run_container(
                self.image_name, cmd,
            )
            smoke_results.append({
                "command": cmd,
                "exit_code": exit_code,
                "output": output,
                "timed_out": timed_out,
            })
            if exit_code != 0:
                all_passed = False

        status = "accepted" if all_passed else "smoke_failed"

        return VerifyBuildResult(
            status=status,
            review_approved=True,
            review_concerns=concerns,
            smoke_test_commands=smoke_commands,
            build_success=True,
            build_duration_ms=build_duration,
            smoke_results=smoke_results,
            dockerfile_snapshot=dockerfile_content,
            review_tokens=review_tokens,
        )

    def _review_dockerfile(self, dockerfile_content: str) -> dict:
        """Call LLM reviewer. Returns review dict with _tokens key."""
        repo_type = self.blueprint.get("repo_type", "unknown")
        language = self.blueprint.get("language", "unknown")

        try:
            response = self.llm.call_nano([
                {"role": "system", "content": REVIEWER_SYSTEM},
                {"role": "user", "content": REVIEWER_USER.format(
                    repo_type=repo_type,
                    language=language,
                    dockerfile_content=dockerfile_content,
                )},
            ])
            review = json.loads(response.content)
            review["_tokens"] = (response.prompt_tokens, response.completion_tokens)
            return review
        except Exception:
            logger.warning("VerifyBuild LLM review failed, using fallback", exc_info=True)
            fallback = dict(REVIEW_FALLBACK)
            fallback["_tokens"] = (0, 0)
            return fallback

    @staticmethod
    def _format_for_agent(result: VerifyBuildResult) -> str:
        """Format VerifyBuildResult as a concise string for the agent."""
        lines = [f"VerifyBuild status: {result.status}"]

        if result.review_concerns:
            lines.append(f"Concerns: {', '.join(result.review_concerns)}")

        if result.status == "build_failed" and result.build_error:
            lines.append(f"Build error:\n{result.build_error}")

        if result.smoke_results:
            for sr in result.smoke_results:
                status = "PASS" if sr["exit_code"] == 0 else "FAIL"
                lines.append(f"Smoke [{status}]: {sr['command']}")
                if sr["exit_code"] != 0:
                    lines.append(f"  Output: {sr['output'][:500]}")

        return "\n".join(lines)

    def get_last_result(self) -> VerifyBuildResult:
        """Run and return the full VerifyBuildResult (used by the agent loop for DB writes)."""
        return self._run()
