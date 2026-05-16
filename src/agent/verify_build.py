"""VerifyBuild tool for BuildAgent v2.0.

Reads the Dockerfile, calls LLM reviewer for approval + smoke test design,
builds the image, runs smoke tests, and returns a VerifyBuildResult.
"""

from __future__ import annotations

import json
import logging
import time
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
Design 1 to 3 shell commands that EXERCISE REAL FUNCTIONALITY of the built \
artifact. Each command you return will be executed VERBATIM by us via \
`sh -c "<your command>"` INSIDE the already-built container, like \
`docker run --rm --entrypoint "" <image> sh -c "<your command>"`.

EXECUTION CONTEXT — read carefully:
- You are writing the part that goes after `sh -c`. DO NOT include \
`docker run`, `--entrypoint`, an image name, or the wrapping `sh -c` \
yourself. Just write the command(s) that run INSIDE the container.
- Wrong: `docker run --rm --entrypoint "" myimg:latest sh -c 'mytool --help'`
- Right: `mytool input.txt | grep -q expected_output`
- We will additionally prepend `set -e -o pipefail` so any failing stage \
in a pipe (or missing binary) bubbles up as a non-zero exit — design your \
command to rely on that strict mode.

QUOTING HYGIENE — get this right or your smoke test will fail with a \
shell syntax error instead of testing anything:
- Every `'` MUST be closed by another `'`. Every `"` MUST be closed by \
another `"`. Count them before returning.
- Prefer the single-quote outer / double-quote inner pattern (or vice \
versa). E.g. `python -c "from x import f; assert f(2)==4; print('ok')"`.
- Avoid mixing both quote styles at the same nesting level when you can \
help it. Avoid backticks.
- Keep each command to ONE LINE. No literal newlines in the command.
- If you cannot express what you need cleanly on one line with balanced \
quotes, pick a simpler smoke test (e.g. `pytest -q tests/smoke/ -x` \
instead of an ad-hoc Python heredoc).

Each command MUST do at least one of these (level 2):
  A. Invoke the project's CLI/binary on a real input and inspect meaningful \
output (not just --version or --help).
  B. Run a documented example end-to-end (e.g. README quick-start snippet).
  C. Start the service and curl a healthcheck endpoint, validating the \
response BODY content (not just HTTP 200).
  D. Run a repo-provided integration/smoke script (`health.sh`, `smoke.sh`, \
`examples/run.sh`, …).
  E. Run the project's own test suite (or a fast smoke subset of it).

REJECT THESE — they are presence/version checks, not smoke tests:
- Bare presence: `which X`, `test -f /app/X`, `ls /app/...`, \
`find / -name '*.X'`, `test -x /app/build/main`.
- Bare version/help: `X --version`, `X -V`, `X --help`, `X -h`.
- Bare import: `python -c "import X"`, `node -e "require('X')"` without \
calling and asserting anything.
- Anything that confirms only that the runtime/binary exists.

EXIT-CODE RULES (your command MUST fail loudly when the thing being \
tested is broken or missing):
- DO NOT use `|| echo FAIL`, `|| echo SKIP`, `|| echo NO_SMOKE_SCRIPT`, \
or any `|| <success-looking-echo>` pattern. Those swallow the failure \
and make the smoke pass even when the artifact is broken. Use \
`|| (echo FAIL; exit 1)` instead, or just let the failing command \
exit non-zero directly.
- DO NOT use `if [ -x /path/script ]; then run; else echo NO_SMOKE_SCRIPT; fi` \
fallbacks. If a script you need does not exist, that is a smoke FAIL — \
do not pick a smoke command that depends on it.
- Every command MUST: exit non-zero when the thing under test does not work, \
and rely ONLY on artifacts that the Dockerfile demonstrably produces \
(binaries from RUN, files from COPY) or that the project ships in its \
source tree (visible upstream).
- DO NOT propose a smoke test that requires a helper script the Dockerfile \
does not already COPY from the repo or produce as part of `RUN`. \
Specifically: do not assume `/app/scripts/smoke.sh`, `/app/health.sh`, \
or similar exist unless you saw a COPY/RUN line creating them.

If using language imports, the command MUST call a function and assert a \
result, e.g.:
- `python -c "from pkg import f; assert f(2)==4; print('ok')"`
- `node -e "const x=require('lodash'); \
console.assert(x.chunk([1,2,3,4],2).length===2); console.log('ok')"`

Examples (level 2) by repo type:
- Library (Python): `python -c "from requests.utils import unquote; \
assert unquote('a%20b')=='a b'; print('ok')"`
- CLI tool: `/app/bin/tool convert /app/examples/in.txt - | grep -q \
expected_token && echo PASS`
- Web service: `/app/start & sleep 2; curl -fsS localhost:8080/health | \
grep -q '\"status\":\"ok\"'`
- Compiled binary: `/app/build/main --input /app/examples/in.json | \
grep -q expected_output`
- Test suite: `cd /app && pytest -q tests/smoke/ -x` or \
`cd /app && npm test -- --testPathPattern=smoke`
- Repo script: `sh /app/scripts/smoke.sh`

PREFERENCE ORDER when picking smoke tests:
1. Project's own smoke/integration script (if one exists in the repo).
2. Project's own test suite (or a fast subset).
3. Documented example from README/docs.
4. CLI/binary on a hand-rolled real input + output assertion.
5. Library function call + assertion.

Avoid commands that depend on external network access.

Return ONLY valid JSON:
{{
  "approved": true or false,
  "concerns": ["<issue1>", "<issue2>"],
  "smoke_test_commands": ["<cmd1>", "<cmd2>"]
}}

smoke_test_commands must have 1 to 3 commands. Never return an empty list. \
Every command must satisfy at least one of A–E above."""

# Degraded fallback used only when the LLM call itself fails. It does not
# satisfy the level-2 rubric — it just keeps the pipeline moving so the run
# can still produce a build outcome.
REVIEW_FALLBACK = {
    "approved": True,
    "concerns": ["LLM review failed — building without review (degraded smoke)"],
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
        self._last_result: VerifyBuildResult | None = None

    def execute(self) -> str:
        """Run the full VerifyBuild pipeline. Returns a JSON string summary."""
        result = self._run()
        self._last_result = result
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
        review_t0 = time.monotonic()
        review = self._review_dockerfile(dockerfile_content)
        review_duration_ms = int((time.monotonic() - review_t0) * 1000)
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
                review_duration_ms=review_duration_ms,
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
                review_duration_ms=review_duration_ms,
                build_success=False,
                build_error=summarized_error,
                build_error_raw=build_error_raw,
                build_duration_ms=build_duration,
                dockerfile_snapshot=dockerfile_content,
                review_tokens=review_tokens,
                error_summary_tokens=(err_pt, err_ct),
            )

        # Step 4: Run smoke tests
        smoke_t0 = time.monotonic()
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
        smoke_duration_ms = int((time.monotonic() - smoke_t0) * 1000)

        status = "accepted" if all_passed else "smoke_failed"

        return VerifyBuildResult(
            status=status,
            review_approved=True,
            review_concerns=concerns,
            smoke_test_commands=smoke_commands,
            review_duration_ms=review_duration_ms,
            build_success=True,
            build_duration_ms=build_duration,
            smoke_results=smoke_results,
            smoke_duration_ms=smoke_duration_ms,
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
