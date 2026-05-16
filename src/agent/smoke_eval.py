"""Heuristic grader for smoke-test commands produced by the VerifyBuild reviewer.

Classifies a single shell command as one of:
    0 — non-test (empty / shell noise / pure cd-or-echo).
    1 — presence or version/help check (forbidden by the rubric).
    2 — exercises real functionality (the only acceptable level).

The grader is deliberately conservative: when in doubt between 1 and 2 it
returns 2 only if the command shows evidence of *using* the artifact (asserts,
greps over output, runs the test suite, runs a repo smoke script, hits an
endpoint and validates body content, etc.). Otherwise it falls back to 1.

This is a regex-based heuristic, not a perfect oracle. Use it for batch eval
and unit testing; do not rely on it as a hard production gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ─── Patterns ─────────────────────────────────────────

# Level-1 (forbidden) signals. If the *only* meaningful thing a command does
# matches one of these, it's a presence/version check.
_FORBIDDEN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("version_flag",   re.compile(r"(^|\s|&&|\|\|)\S+\s+--?(version|V)\b")),
    ("help_flag",      re.compile(r"(^|\s|&&|\|\|)\S+\s+--?(help|h)\b")),
    ("which",          re.compile(r"(^|\s|&&|\|\|)which\s+\S+")),
    ("type_builtin",   re.compile(r"(^|\s|&&|\|\|)type\s+-[pP]?\s*\S+")),
    ("test_f",         re.compile(r"(^|\s|&&|\|\|)test\s+-[fdsxr]\s+\S+")),
    ("bracket_test",   re.compile(r"\[\s+-[fdsxr]\s+\S+\s+]")),
    ("ls_only",        re.compile(r"^\s*ls(\s+-\w+)?\s+\S+\s*$")),
    ("find_name",      re.compile(r"(^|\s|&&|\|\|)find\s+\S+\s+-name\b")),
    ("bare_import_py", re.compile(
        r"""python3?\s+-c\s+["'][^"']*\bimport\s+\w+[^"']*["']"""
    )),
    ("bare_require_js", re.compile(
        r"""node\s+-e\s+["'][^"']*require\(['"][^'"]+['"]\)[^"']*["']"""
    )),
]

# Level-2 (acceptable) signals.
_LEVEL_2_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Repo-provided smoke / health / integration scripts.
    # Matches names like smoke.sh, health.py, healthcheck.py, smoke_test.sh,
    # health-check.sh, integration.sh, e2e.sh, examples/run.sh, run-smoke.sh.
    ("repo_script", re.compile(
        r"(^|/|\s)([\w.-]*?(smoke|health|healthcheck|integration|e2e)[\w.-]*|examples?/run)\.(sh|py|rb|js|mjs|ts)\b"
    )),
    # Project test suites.
    ("pytest",      re.compile(r"\bpytest\b")),
    ("unittest",    re.compile(r"\bpython\s+-m\s+unittest\b")),
    ("npm_test",    re.compile(r"\bnpm\s+(run\s+)?test\b")),
    ("yarn_test",   re.compile(r"\byarn\s+test\b")),
    ("pnpm_test",   re.compile(r"\bpnpm\s+test\b")),
    ("jest",        re.compile(r"\bjest\b")),
    ("mocha",       re.compile(r"\bmocha\b")),
    ("go_test",     re.compile(r"\bgo\s+test\b")),
    ("cargo_test",  re.compile(r"\bcargo\s+test\b")),
    ("mvn_test",    re.compile(r"\bmvn\s+(\S+\s+)?test\b")),
    ("gradle_test", re.compile(r"\bgradle\s+(\S+\s+)?test\b")),
    ("rake_test",   re.compile(r"\brake\s+test\b")),
    ("phpunit",     re.compile(r"\bphpunit\b")),
    ("rspec",       re.compile(r"\brspec\b")),
    ("ctest",       re.compile(r"\bctest\b")),
    # Endpoint hit with body content validation.
    ("curl_grep", re.compile(
        r"\bcurl\b[^|]*\|\s*(grep|jq|python|node|awk|sed|head)\b"
    )),
    ("wget_grep", re.compile(
        r"\bwget\b[^|]*-O-[^|]*\|\s*(grep|jq|python|node|awk|sed|head)\b"
    )),
    # Inline import + assertion (Python). Match `python -c` followed
    # eventually by an `assert`/`raise` token; tolerate embedded quotes.
    ("py_assert", re.compile(
        r"\bpython3?\s+-c\b.*?\b(assert|raise)\b"
    )),
    # Inline require + assertion (Node).
    ("node_assert", re.compile(
        r"\bnode\s+-e\b.*?\b(assert|console\.assert|throw)\b"
    )),
    # Generic "run something AND grep its output" — implies inspecting output.
    ("pipe_grep", re.compile(r"\|\s*grep\b")),
    ("pipe_jq",   re.compile(r"\|\s*jq\b")),
    # Compare command output against expected (diff/cmp).
    ("output_diff", re.compile(r"\b(diff|cmp)\b\s+\S")),
]

# Things that look like noise / not really a test.
_NOISE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("echo_only",  re.compile(r"^\s*echo\b[^|;&]*$")),
    ("true_only",  re.compile(r"^\s*(true|:)\s*$")),
    ("cd_only",    re.compile(r"^\s*cd\s+\S+\s*$")),
]


@dataclass
class CommandGrade:
    command: str
    level: int            # 0, 1, or 2
    reasons: list[str]    # named patterns that fired
    forbidden_hits: list[str]


def grade_command(cmd: str) -> CommandGrade:
    """Grade a single shell command. Higher level = better smoke test."""
    if not cmd or not cmd.strip():
        return CommandGrade(command=cmd, level=0, reasons=["empty"], forbidden_hits=[])

    reasons: list[str] = []
    forbidden_hits: list[str] = []

    for name, pat in _LEVEL_2_PATTERNS:
        if pat.search(cmd):
            reasons.append(name)

    for name, pat in _FORBIDDEN_PATTERNS:
        if pat.search(cmd):
            forbidden_hits.append(name)

    # Level-2 evidence wins, even if a forbidden token also appears (e.g. a
    # `--help` substring inside a larger pipeline that also asserts output).
    # We require the level-2 evidence to be substantive: at least one match.
    if reasons:
        return CommandGrade(command=cmd, level=2, reasons=reasons,
                            forbidden_hits=forbidden_hits)

    if forbidden_hits:
        return CommandGrade(command=cmd, level=1, reasons=[],
                            forbidden_hits=forbidden_hits)

    for name, pat in _NOISE_PATTERNS:
        if pat.search(cmd):
            return CommandGrade(command=cmd, level=0, reasons=[name],
                                forbidden_hits=[])

    # Unknown shape — treat as level 1 (ambiguous; not provably exercising
    # functionality). The rubric requires level 2; ambiguous fails.
    return CommandGrade(command=cmd, level=1, reasons=["unrecognized"],
                        forbidden_hits=[])


@dataclass
class ResponseGrade:
    commands: list[CommandGrade]
    min_level: int
    max_level: int
    passes_rubric: bool   # every command is level 2

    @property
    def num_commands(self) -> int:
        return len(self.commands)


def grade_response(commands: list[str]) -> ResponseGrade:
    """Grade a full smoke_test_commands list. Passes the rubric only if EVERY
    command is level 2 (the user's stated bar)."""
    graded = [grade_command(c) for c in commands]
    if not graded:
        return ResponseGrade(commands=[], min_level=0, max_level=0,
                             passes_rubric=False)
    levels = [g.level for g in graded]
    return ResponseGrade(
        commands=graded,
        min_level=min(levels),
        max_level=max(levels),
        passes_rubric=all(level == 2 for level in levels),
    )
