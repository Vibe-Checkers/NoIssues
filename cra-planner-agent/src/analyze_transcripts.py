#!/usr/bin/env python3
"""
Transcript Analysis Feedback Pipeline.

Reads agent transcripts + results JSONL, performs statistical analysis,
then uses gpt-5-chat to synthesize actionable improvements for the agent.

Usage:
    cd ~/Clones/capstone/NoIssues/cra-planner-agent
    python src/analyze_transcripts.py
"""

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = PROJECT_ROOT / "downloaded_transcripts"
RESULTS_DIRS = [Path("/tmp")]  # r3 results saved here
RESULTS_DIRS.append(PROJECT_ROOT / "parallel_empirical_results")

load_dotenv(PROJECT_ROOT / ".env")

# Failure patterns (mirrored from workflow.py)
_COMPLIANCE_RE = [re.compile(p, re.I) for p in [
    r"\brat\b.*(?:check|plugin|fail)", r"unapproved license", r"license header",
    r"\bcheckstyle\b.*(?:fail|violation)", r"\bspotbugs\b.*fail",
    r"\bpmd\b.*(?:fail|violation)", r"\bjacoco\b.*(?:fail|threshold)",
]]
_TIMEOUT_RE = [re.compile(p, re.I) for p in [
    r"exceeded.*timeout", r"timeout.*exceeded", r"build timed out", r"timeoutexpired",
]]
_IMAGE_NOT_FOUND_RE = [re.compile(p, re.I) for p in [
    r"manifest unknown", r"manifest for .* not found", r"pull access denied",
    r"not found: manifest unknown", r"repository does not exist",
]]
_APT_NOT_FOUND_RE = [re.compile(p, re.I) for p in [
    r"e: unable to locate package", r"e: package .* has no installation candidate",
]]
_SYNTAX_RE = [re.compile(p, re.I) for p in [
    r"unknown instruction", r"dockerfile parse error", r"syntax error", r"unknown flag",
]]
_UNFIXABLE_RE = [re.compile(p, re.I) for p in [
    r"(?:no matching version|version not found|does not exist in the npm registry)",
    r"cannot find a java installation.*languageversion",
    r"(?:unauthorized|authentication required).*(?:pull|registry)",
]]
_RATE_LIMIT_RE = re.compile(r"429|ratelimit|rate.limit", re.I)
_NO_SPACE_RE = re.compile(r"no space left on device", re.I)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TranscriptStep:
    step_num: int
    tool: str
    thought: str
    input_data: str
    observation: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ParsedTranscript:
    filename: str
    repo_slug: str
    machine: str
    steps: List[TranscriptStep] = field(default_factory=list)
    total_tokens: int = 0
    final_outcome: str = "unknown"  # success, failure, incomplete
    tools_used: Counter = field(default_factory=Counter)
    file_size: int = 0


# ---------------------------------------------------------------------------
# Stage 1: Parse & Ingest
# ---------------------------------------------------------------------------

_STEP_HEADER_RE = re.compile(
    r"^═{10,}\nSTEP\s+(\d+):\s*(.+?)\n═{10,}$", re.MULTILINE
)
_TOKENS_RE = re.compile(
    r"\[TOKENS\]\s*Input:\s*(\d+),\s*Output:\s*(\d+),\s*Total:\s*(\d+)"
)
_FINAL_TOKENS_RE = re.compile(
    r"\[FINAL TOKEN USAGE\]\s*Input:\s*(\d+),\s*Output:\s*(\d+),\s*Total:\s*(\d+)"
)
_AGENT_FINISHED_RE = re.compile(r"\[AGENT FINISHED\]")


def _slug_from_filename(fname: str) -> str:
    """Extract repo slug from transcript filename like 'owner__repo_20260311_012100.log'."""
    # Remove timestamp suffix: _YYYYMMDD_HHMMSS.log
    base = re.sub(r"_\d{8}_\d{6}\.log$", "", fname)
    return base


def parse_transcript(filepath: Path, machine: str) -> ParsedTranscript:
    """Parse a single transcript .log file into structured data."""
    text = filepath.read_text(errors="replace")
    slug = _slug_from_filename(filepath.name)

    transcript = ParsedTranscript(
        filename=filepath.name,
        repo_slug=slug,
        machine=machine,
        file_size=filepath.stat().st_size,
    )

    # Find all step boundaries
    step_matches = list(_STEP_HEADER_RE.finditer(text))

    for i, m in enumerate(step_matches):
        step_num = int(m.group(1))
        tool = m.group(2).strip()
        # Step content is from end of header to start of next header (or end of file)
        start = m.end()
        end = step_matches[i + 1].start() if i + 1 < len(step_matches) else len(text)
        content = text[start:end]

        # Extract sections
        thought = ""
        input_data = ""
        observation = ""

        thought_match = re.search(r"\[THOUGHT\]\s*\n(.*?)(?=\n\[ACTION\]|\n\[INPUT\]|\n═{10,}|\Z)", content, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
            # Remove "Thought: " prefix if present
            thought = re.sub(r"^Thought:\s*", "", thought)

        input_match = re.search(r"\[INPUT\]\s*\n(.*?)(?=\n\[OBSERVATION\]|\n═{10,}|\Z)", content, re.DOTALL)
        if input_match:
            input_data = input_match.group(1).strip()

        obs_match = re.search(r"\[OBSERVATION\]\s*\n(.*?)(?=\n─{10,}|\n═{10,}|\Z)", content, re.DOTALL)
        if obs_match:
            observation = obs_match.group(1).strip()

        # Extract tokens for this step
        tokens_match = _TOKENS_RE.search(content)
        inp_t = out_t = tot_t = 0
        if tokens_match:
            inp_t, out_t, tot_t = int(tokens_match.group(1)), int(tokens_match.group(2)), int(tokens_match.group(3))

        step = TranscriptStep(
            step_num=step_num, tool=tool, thought=thought,
            input_data=input_data, observation=observation,
            input_tokens=inp_t, output_tokens=out_t, total_tokens=tot_t,
        )
        transcript.steps.append(step)
        transcript.tools_used[tool] += 1

    # Final token usage
    final_match = _FINAL_TOKENS_RE.search(text)
    if final_match:
        transcript.total_tokens = int(final_match.group(3))
    elif transcript.steps:
        transcript.total_tokens = sum(s.total_tokens for s in transcript.steps)

    # Determine outcome
    if _AGENT_FINISHED_RE.search(text):
        # Check if the text around AGENT FINISHED mentions success
        finish_idx = text.rfind("[AGENT FINISHED]")
        context = text[finish_idx:finish_idx + 500]
        if re.search(r"success|verified|passed", context, re.I):
            transcript.final_outcome = "success"
        else:
            transcript.final_outcome = "failure"
    else:
        transcript.final_outcome = "incomplete"

    return transcript


def parse_all_transcripts() -> List[ParsedTranscript]:
    """Parse all transcript files across all machines."""
    transcripts = []
    for machine_dir in sorted(TRANSCRIPTS_DIR.iterdir()):
        if not machine_dir.is_dir():
            continue
        machine = machine_dir.name
        agent_dir = machine_dir / "agent_transcripts"
        if not agent_dir.exists():
            # Try flat structure
            agent_dir = machine_dir
        for logfile in sorted(agent_dir.glob("*.log")):
            try:
                t = parse_transcript(logfile, machine)
                transcripts.append(t)
            except Exception as e:
                print(f"  [WARN] Failed to parse {logfile.name}: {e}", file=sys.stderr)
    return transcripts


def load_results_jsonl() -> Dict[str, dict]:
    """Load all results JSONL files, keyed by repo_slug."""
    results = {}
    # Check /tmp for r1, r2, r3 results
    for d in RESULTS_DIRS:
        for f in sorted(d.glob("*_r*.jsonl")) if d == Path("/tmp") else sorted(d.glob("results_*.jsonl")):
            try:
                for line in f.read_text().splitlines():
                    if not line.strip():
                        continue
                    entry = json.loads(line)
                    slug = entry.get("repo_slug", "")
                    if slug and slug not in results:
                        results[slug] = entry
            except Exception as e:
                print(f"  [WARN] Failed to read {f}: {e}", file=sys.stderr)
    return results


def join_data(transcripts: List[ParsedTranscript], results: Dict[str, dict]) -> List[Tuple[ParsedTranscript, Optional[dict]]]:
    """Join transcripts with JSONL results by repo_slug."""
    joined = []
    for t in transcripts:
        result = results.get(t.repo_slug)
        joined.append((t, result))
    return joined


# ---------------------------------------------------------------------------
# Stage 2: Local Statistical Analysis
# ---------------------------------------------------------------------------

def classify_failure_type(text: str) -> str:
    """Classify a failure based on error text."""
    if _RATE_LIMIT_RE.search(text):
        return "RATE_LIMIT"
    if _NO_SPACE_RE.search(text):
        return "NO_SPACE"
    for p in _TIMEOUT_RE:
        if p.search(text):
            return "TIMEOUT"
    for p in _IMAGE_NOT_FOUND_RE:
        if p.search(text):
            return "IMAGE_NOT_FOUND"
    for p in _APT_NOT_FOUND_RE:
        if p.search(text):
            return "APT_NOT_FOUND"
    for p in _SYNTAX_RE:
        if p.search(text):
            return "SYNTAX"
    for p in _UNFIXABLE_RE:
        if p.search(text):
            return "UNFIXABLE"
    for p in _COMPLIANCE_RE:
        if p.search(text):
            return "COMPLIANCE"
    if re.search(r"file not found|no such file|checksum.*not found", text, re.I):
        return "FILE_COPY_MISSING"
    if re.search(r"WRITE BLOCKED", text):
        return "WRITE_BLOCKED_LOOP"
    return "UNKNOWN"


def analyze_failures(joined: List[Tuple[ParsedTranscript, Optional[dict]]]) -> dict:
    """Analyze failure modes across all transcripts."""
    failure_types = Counter()
    failure_examples = defaultdict(list)  # type -> [(slug, error_snippet)]

    for t, result in joined:
        if t.final_outcome == "success":
            continue
        if result and result.get("skip_reason"):
            continue

        # Collect error text from last VerifyBuild observation + lessons_learned
        error_text = ""
        for step in reversed(t.steps):
            if step.tool == "VerifyBuild" and step.observation:
                error_text = step.observation[:2000]
                break

        # Also check JSONL lessons_learned
        if result:
            lessons = result.get("agent_analysis", {}).get("lessons_learned", [])
            if lessons:
                error_text += " " + " ".join(str(l) for l in lessons[:3])

        ftype = classify_failure_type(error_text)
        failure_types[ftype] += 1
        if len(failure_examples[ftype]) < 5:
            failure_examples[ftype].append({
                "slug": t.repo_slug,
                "machine": t.machine,
                "error_snippet": error_text[:500],
                "steps": len(t.steps),
                "tools_used": dict(t.tools_used),
            })

    return {
        "distribution": dict(failure_types.most_common()),
        "examples": {k: v for k, v in failure_examples.items()},
        "total_failures": sum(failure_types.values()),
    }


def analyze_tool_usage(joined: List[Tuple[ParsedTranscript, Optional[dict]]]) -> dict:
    """Analyze tool usage patterns."""
    by_outcome = defaultdict(Counter)  # outcome -> tool -> count
    wasted_calls = Counter()  # type -> count
    steps_to_first_verify = {"success": [], "failure": []}
    total_tool_counts = Counter()

    for t, result in joined:
        outcome = t.final_outcome
        first_verify_step = None

        for i, step in enumerate(t.steps):
            total_tool_counts[step.tool] += 1
            by_outcome[outcome][step.tool] += 1

            if step.tool == "VerifyBuild" and first_verify_step is None:
                first_verify_step = i + 1

            # Detect wasted calls
            if step.tool == "ReadLocalFile" and re.search(r"not found|error|no such", step.observation, re.I):
                wasted_calls["ReadLocalFile_missing"] += 1
            if step.tool == "WriteToFile" and "WRITE BLOCKED" in step.observation:
                wasted_calls["WriteToFile_blocked"] += 1
            if step.tool == "ListDirectory" and i > 0:
                # Check if same directory was listed before
                for prev in t.steps[:i]:
                    if prev.tool == "ListDirectory" and prev.input_data == step.input_data:
                        wasted_calls["ListDirectory_duplicate"] += 1
                        break

        if first_verify_step is not None:
            bucket = "success" if outcome == "success" else "failure"
            steps_to_first_verify[bucket].append(first_verify_step)

    avg_first_verify = {}
    for k, vals in steps_to_first_verify.items():
        avg_first_verify[k] = round(sum(vals) / len(vals), 1) if vals else 0

    return {
        "total_tool_counts": dict(total_tool_counts.most_common()),
        "by_outcome": {k: dict(v.most_common()) for k, v in by_outcome.items()},
        "wasted_calls": dict(wasted_calls.most_common()),
        "avg_steps_to_first_verify": avg_first_verify,
    }


def analyze_success_patterns(joined: List[Tuple[ParsedTranscript, Optional[dict]]]) -> dict:
    """Analyze what successful transcripts have in common."""
    success_sequences = []
    by_language = Counter()
    by_repo_type = Counter()
    first_attempt_success = 0
    total_success = 0

    for t, result in joined:
        if t.final_outcome != "success":
            continue
        total_success += 1

        # Tool sequence
        seq = [s.tool for s in t.steps]
        success_sequences.append({
            "slug": t.repo_slug,
            "sequence": seq,
            "steps": len(seq),
            "tokens": t.total_tokens,
        })

        if result:
            lang = result.get("classification", {}).get("primary_language", "unknown")
            rtype = result.get("classification", {}).get("repo_type", "unknown")
            by_language[lang] += 1
            by_repo_type[rtype] += 1

            attempts = result.get("agent_analysis", {}).get("attempts", 1)
            if attempts == 1:
                first_attempt_success += 1

    # Find most common starting sequences
    start_seqs = Counter()
    for s in success_sequences:
        # First 4 tools
        key = " → ".join(s["sequence"][:4])
        start_seqs[key] += 1

    return {
        "total_success": total_success,
        "first_attempt_success": first_attempt_success,
        "retry_success": total_success - first_attempt_success,
        "by_language": dict(by_language.most_common()),
        "by_repo_type": dict(by_repo_type.most_common()),
        "common_start_sequences": dict(start_seqs.most_common(10)),
        "avg_steps": round(sum(s["steps"] for s in success_sequences) / max(len(success_sequences), 1), 1),
        "avg_tokens": round(sum(s["tokens"] for s in success_sequences) / max(len(success_sequences), 1)),
    }


def detect_anti_patterns(joined: List[Tuple[ParsedTranscript, Optional[dict]]]) -> dict:
    """Detect behavioral anti-patterns in transcripts."""
    patterns = Counter()
    examples = defaultdict(list)

    for t, result in joined:
        if not t.steps:
            continue

        # 1. Premature Dockerfile writing: WriteToFile before any ReadLocalFile
        wrote_before_read = False
        has_read = False
        for step in t.steps:
            if step.tool == "ReadLocalFile":
                has_read = True
            if step.tool == "WriteToFile" and "Dockerfile" in step.input_data and not has_read:
                wrote_before_read = True
                break
        if wrote_before_read:
            patterns["premature_dockerfile_write"] += 1
            if len(examples["premature_dockerfile_write"]) < 3:
                examples["premature_dockerfile_write"].append(t.repo_slug)

        # 2. SearchDockerError called but advice ignored (next WriteToFile doesn't change)
        for i, step in enumerate(t.steps):
            if step.tool == "SearchDockerError" and i + 1 < len(t.steps):
                next_step = t.steps[i + 1]
                if next_step.tool != "WriteToFile" and next_step.tool != "ReadLocalFile":
                    # Agent didn't apply the fix
                    patterns["search_advice_ignored"] += 1
                    break

        # 3. Base image guessing loops (multiple WRITE BLOCKED in same transcript)
        blocked_count = sum(1 for s in t.steps if "WRITE BLOCKED" in s.observation)
        if blocked_count >= 2:
            patterns["base_image_guessing_loop"] += 1
            if len(examples["base_image_guessing_loop"]) < 3:
                examples["base_image_guessing_loop"].append(f"{t.repo_slug} ({blocked_count}x)")

        # 4. Empty or minimal thoughts
        minimal_thoughts = sum(1 for s in t.steps if len(s.thought) < 20)
        if minimal_thoughts > len(t.steps) * 0.5 and len(t.steps) > 3:
            patterns["minimal_reasoning"] += 1

        # 5. Repeated identical SearchDockerError queries
        search_queries = [s.input_data for s in t.steps if s.tool == "SearchDockerError"]
        if len(search_queries) != len(set(search_queries)) and len(search_queries) >= 2:
            patterns["repeated_search_queries"] += 1

    return {
        "counts": dict(patterns.most_common()),
        "examples": {k: v for k, v in examples.items()},
    }


def analyze_token_efficiency(joined: List[Tuple[ParsedTranscript, Optional[dict]]]) -> dict:
    """Analyze token consumption efficiency."""
    success_tokens = []
    failure_tokens = []

    for t, result in joined:
        if t.total_tokens == 0:
            continue
        if t.final_outcome == "success":
            success_tokens.append(t.total_tokens)
        elif t.final_outcome in ("failure", "incomplete"):
            failure_tokens.append(t.total_tokens)

    return {
        "success_avg_tokens": round(sum(success_tokens) / max(len(success_tokens), 1)),
        "failure_avg_tokens": round(sum(failure_tokens) / max(len(failure_tokens), 1)),
        "success_count": len(success_tokens),
        "failure_count": len(failure_tokens),
        "total_tokens_spent": sum(success_tokens) + sum(failure_tokens),
        "tokens_wasted_on_failures": sum(failure_tokens),
    }


# ---------------------------------------------------------------------------
# Stage 3: LLM Synthesis
# ---------------------------------------------------------------------------

def _get_llm_client():
    """Create Azure OpenAI client for analysis."""
    from openai import AzureOpenAI
    return AzureOpenAI(
        api_key=os.getenv("ANALYSIS_MODEL_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
        azure_endpoint=os.getenv("ANALYSIS_MODEL_ENDPOINT"),
    )


def _llm_call(client, deployment: str, system: str, user: str, retries: int = 3) -> str:
    """Make an LLM call with retry logic."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.3,
                max_tokens=4000,
            )
            time.sleep(1)  # Rate limiting
            return resp.choices[0].message.content
        except Exception as e:
            print(f"  [WARN] LLM call attempt {attempt + 1} failed: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    return "[LLM call failed after retries]"


def _read_system_prompt() -> str:
    """Read the current system prompt from core.py."""
    core_path = PROJECT_ROOT / "src" / "agent" / "core.py"
    text = core_path.read_text()
    # Extract the template string
    match = re.search(r'template = """(.+?)"""', text, re.DOTALL)
    return match.group(1) if match else ""


def _read_metaprompt() -> str:
    """Read the current metaprompt from metaprompt.py."""
    path = PROJECT_ROOT / "src" / "agent" / "metaprompt.py"
    text = path.read_text()
    match = re.search(r'METAPROMPT_SYSTEM = """(.+?)"""', text, re.DOTALL)
    return match.group(1) if match else ""


def _get_tool_descriptions() -> str:
    """Get tool descriptions from tools.py (just names and purposes)."""
    tools_path = PROJECT_ROOT / "src" / "agent" / "tools.py"
    text = tools_path.read_text()
    # Extract tool names from create_structured_tools
    tools = re.findall(r'StructuredTool\.from_function\(\s*func=\w+.*?name="(\w+)".*?description="([^"]+)"', text, re.DOTALL)
    return "\n".join(f"- {name}: {desc[:100]}" for name, desc in tools)


def llm_failure_analysis(client, deployment: str, failure_data: dict) -> str:
    """Use LLM to analyze failure patterns and suggest improvements."""
    print("  [LLM] Analyzing failure patterns...", file=sys.stderr)

    system = """You are an expert at analyzing AI agent failure logs.
Given failure statistics and examples from a Dockerfile-generation agent, produce:
1. Root cause analysis for each failure type
2. Concrete prompt additions to help the agent avoid each failure
3. New failure categories that should be added to the classifier
4. Whether new tools would help address specific failures"""

    # Build user prompt with failure data
    lines = [f"Total failures: {failure_data['total_failures']}\n"]
    lines.append("Failure distribution:")
    for ftype, count in failure_data["distribution"].items():
        lines.append(f"  {ftype}: {count}")
        examples = failure_data["examples"].get(ftype, [])
        for ex in examples[:3]:
            lines.append(f"    - {ex['slug']}: {ex['error_snippet'][:200]}")

    return _llm_call(client, deployment, system, "\n".join(lines))


def llm_success_synthesis(client, deployment: str, success_data: dict,
                          joined: List[Tuple[ParsedTranscript, Optional[dict]]]) -> str:
    """Use LLM to synthesize success patterns into golden examples."""
    print("  [LLM] Synthesizing success patterns...", file=sys.stderr)

    system = """You are an expert at analyzing AI agent transcripts.
Given successful Dockerfile-generation transcripts, extract:
1. Common winning strategies (tool usage patterns that lead to success)
2. Optimal tool-call sequences by project type
3. 3-5 "golden example" snippets that could be added to the agent's system prompt as few-shot demonstrations
4. Key decision points where successful agents differ from failing ones"""

    # Select 10 best success transcripts (fewest steps)
    successes = [(t, r) for t, r in joined if t.final_outcome == "success"]
    successes.sort(key=lambda x: len(x[0].steps))
    top = successes[:10]

    lines = [f"Overall stats: {success_data['total_success']} successes, "
             f"avg {success_data['avg_steps']} steps, avg {success_data['avg_tokens']} tokens\n"]
    lines.append(f"Common start sequences: {json.dumps(success_data['common_start_sequences'])}\n")

    for t, r in top:
        lang = r.get("classification", {}).get("primary_language", "?") if r else "?"
        rtype = r.get("classification", {}).get("repo_type", "?") if r else "?"
        seq = " → ".join(s.tool for s in t.steps)
        # Get key thoughts
        thoughts = [s.thought[:100] for s in t.steps if s.thought][:3]
        lines.append(f"\n--- {t.repo_slug} ({lang}, {rtype}, {len(t.steps)} steps) ---")
        lines.append(f"Sequence: {seq}")
        lines.append(f"Key thoughts: {'; '.join(thoughts)}")

    return _llm_call(client, deployment, system, "\n".join(lines))


def llm_prompt_improvement(client, deployment: str, failure_data: dict,
                           anti_patterns: dict, success_data: dict) -> str:
    """Use LLM to suggest system prompt improvements."""
    print("  [LLM] Generating prompt improvements...", file=sys.stderr)

    current_prompt = _read_system_prompt()

    system = """You are an expert prompt engineer for AI coding agents.
Given the current system prompt and empirical failure/success data, propose SPECIFIC improvements:
1. New rules to add to the ABSOLUTE RULES section
2. New entries for the COMMON FIX PATTERNS section
3. Workflow phase modifications
4. Provide the EXACT text to add/change (as diff-ready snippets)"""

    user = f"""CURRENT SYSTEM PROMPT:
{current_prompt[:3000]}

FAILURE STATISTICS:
{json.dumps(failure_data['distribution'], indent=2)}

ANTI-PATTERNS DETECTED:
{json.dumps(anti_patterns['counts'], indent=2)}

SUCCESS STATS:
- Total: {success_data['total_success']}
- First-attempt: {success_data['first_attempt_success']}
- Common sequences: {json.dumps(success_data['common_start_sequences'])}

Propose concrete prompt improvements."""

    return _llm_call(client, deployment, system, user)


def llm_tool_gap_analysis(client, deployment: str, anti_patterns: dict,
                          failure_data: dict) -> str:
    """Use LLM to identify tool gaps and improvements."""
    print("  [LLM] Analyzing tool gaps...", file=sys.stderr)

    tool_descs = _get_tool_descriptions()

    system = """You are an expert at designing tools for AI coding agents.
Given the current tool set and empirical data on agent failures and anti-patterns,
suggest: new tools to add, existing tools to enhance, and tool description improvements."""

    user = f"""CURRENT TOOLS:
{tool_descs}

WASTED TOOL CALLS:
{json.dumps(anti_patterns.get('counts', {}), indent=2)}

FAILURE TYPES:
{json.dumps(failure_data['distribution'], indent=2)}

Suggest tool improvements."""

    return _llm_call(client, deployment, system, user)


def llm_metaprompt_improvement(client, deployment: str, failure_data: dict,
                               joined: List[Tuple[ParsedTranscript, Optional[dict]]]) -> str:
    """Use LLM to suggest metaprompt template improvements."""
    print("  [LLM] Analyzing metaprompt...", file=sys.stderr)

    current_metaprompt = _read_metaprompt()

    # Failure rates by taxonomy dimension
    by_domain = Counter()
    by_build_tool = Counter()
    domain_fail = Counter()
    build_fail = Counter()
    for t, r in joined:
        if not r:
            continue
        domain = r.get("taxonomy", {}).get("domain", "?")
        btool = r.get("taxonomy", {}).get("build_tool", "?")
        by_domain[domain] += 1
        by_build_tool[btool] += 1
        if t.final_outcome != "success":
            domain_fail[domain] += 1
            build_fail[btool] += 1

    system = """You are an expert at designing metaprompt templates for DevOps AI agents.
Given the current metaprompt template and failure rates by taxonomy dimension,
suggest specific improvements to the template."""

    user = f"""CURRENT METAPROMPT:
{current_metaprompt}

FAILURE RATES BY DOMAIN:
{json.dumps({d: f"{domain_fail[d]}/{by_domain[d]}" for d in by_domain}, indent=2)}

FAILURE RATES BY BUILD TOOL:
{json.dumps({b: f"{build_fail[b]}/{by_build_tool[b]}" for b in by_build_tool}, indent=2)}

Suggest metaprompt improvements."""

    return _llm_call(client, deployment, system, user)


# ---------------------------------------------------------------------------
# Stage 4: Output Generation
# ---------------------------------------------------------------------------

def generate_reports(stage2: dict, stage3: dict):
    """Generate JSON and Markdown reports."""
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "data_summary": stage2["summary"],
        "failure_analysis": stage2["failures"],
        "tool_usage": stage2["tool_usage"],
        "success_patterns": stage2["success"],
        "anti_patterns": stage2["anti_patterns"],
        "token_efficiency": stage2["tokens"],
        "llm_analysis": stage3,
    }

    # JSON report
    json_path = PROJECT_ROOT / "feedback_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"  Wrote {json_path}", file=sys.stderr)

    # Markdown report
    md_path = PROJECT_ROOT / "feedback_report.md"
    md_lines = [
        "# Agent Feedback Report",
        f"\nGenerated: {report['generated_at']}",
        f"\n## Data Summary",
        f"- Transcripts analyzed: {report['data_summary']['total_transcripts']}",
        f"- Successes: {report['data_summary']['successes']}",
        f"- Failures: {report['data_summary']['failures']}",
        f"- Skipped: {report['data_summary']['skipped']}",
        f"\n## Failure Analysis",
        f"\n### Distribution",
    ]
    for ftype, count in report["failure_analysis"]["distribution"].items():
        md_lines.append(f"- **{ftype}**: {count}")

    md_lines.append(f"\n## Tool Usage")
    md_lines.append(f"\n### Overall Tool Counts")
    for tool, count in report["tool_usage"]["total_tool_counts"].items():
        md_lines.append(f"- {tool}: {count}")
    md_lines.append(f"\n### Wasted Calls")
    for wtype, count in report["tool_usage"]["wasted_calls"].items():
        md_lines.append(f"- {wtype}: {count}")
    md_lines.append(f"\n### Avg Steps to First VerifyBuild")
    for k, v in report["tool_usage"]["avg_steps_to_first_verify"].items():
        md_lines.append(f"- {k}: {v}")

    md_lines.append(f"\n## Success Patterns")
    md_lines.append(f"- Total: {report['success_patterns']['total_success']}")
    md_lines.append(f"- First attempt: {report['success_patterns']['first_attempt_success']}")
    md_lines.append(f"- Avg steps: {report['success_patterns']['avg_steps']}")
    md_lines.append(f"- Avg tokens: {report['success_patterns']['avg_tokens']}")
    md_lines.append(f"\n### By Language")
    for lang, count in report["success_patterns"]["by_language"].items():
        md_lines.append(f"- {lang}: {count}")

    md_lines.append(f"\n## Anti-Patterns")
    for pattern, count in report["anti_patterns"]["counts"].items():
        md_lines.append(f"- **{pattern}**: {count}")

    md_lines.append(f"\n## Token Efficiency")
    te = report["token_efficiency"]
    md_lines.append(f"- Avg tokens per success: {te['success_avg_tokens']:,}")
    md_lines.append(f"- Avg tokens per failure: {te['failure_avg_tokens']:,}")
    md_lines.append(f"- Total tokens spent: {te['total_tokens_spent']:,}")
    md_lines.append(f"- Tokens wasted on failures: {te['tokens_wasted_on_failures']:,}")

    # LLM analysis sections
    for section, title in [
        ("failure_analysis", "LLM: Failure Analysis"),
        ("success_synthesis", "LLM: Success Synthesis"),
        ("prompt_improvement", "LLM: Prompt Improvement Suggestions"),
        ("tool_gap_analysis", "LLM: Tool Gap Analysis"),
        ("metaprompt_improvement", "LLM: Metaprompt Improvement"),
    ]:
        content = report["llm_analysis"].get(section, "")
        if content:
            md_lines.append(f"\n## {title}")
            md_lines.append(content)

    md_path.write_text("\n".join(md_lines))
    print(f"  Wrote {md_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60, file=sys.stderr)
    print("STAGE 1: Parsing transcripts & loading results", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    transcripts = parse_all_transcripts()
    print(f"  Parsed {len(transcripts)} transcripts", file=sys.stderr)

    results = load_results_jsonl()
    print(f"  Loaded {len(results)} result entries", file=sys.stderr)

    joined = join_data(transcripts, results)
    matched = sum(1 for _, r in joined if r is not None)
    print(f"  Joined: {matched}/{len(joined)} transcripts have JSONL match", file=sys.stderr)

    # Summary stats
    outcomes = Counter(t.final_outcome for t, _ in joined)
    skipped = sum(1 for _, r in joined if r and r.get("skip_reason"))
    summary = {
        "total_transcripts": len(transcripts),
        "successes": outcomes.get("success", 0),
        "failures": outcomes.get("failure", 0) + outcomes.get("incomplete", 0),
        "skipped": skipped,
        "matched_with_jsonl": matched,
    }
    print(f"  Outcomes: {dict(outcomes)}", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print("STAGE 2: Local statistical analysis", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    failures = analyze_failures(joined)
    print(f"  Failure types: {failures['distribution']}", file=sys.stderr)

    tool_usage = analyze_tool_usage(joined)
    print(f"  Tool counts: {tool_usage['total_tool_counts']}", file=sys.stderr)

    success = analyze_success_patterns(joined)
    print(f"  Success patterns: {success['total_success']} successes, avg {success['avg_steps']} steps", file=sys.stderr)

    anti_patterns = detect_anti_patterns(joined)
    print(f"  Anti-patterns: {anti_patterns['counts']}", file=sys.stderr)

    tokens = analyze_token_efficiency(joined)
    print(f"  Token efficiency: success avg={tokens['success_avg_tokens']}, failure avg={tokens['failure_avg_tokens']}", file=sys.stderr)

    stage2 = {
        "summary": summary,
        "failures": failures,
        "tool_usage": tool_usage,
        "success": success,
        "anti_patterns": anti_patterns,
        "tokens": tokens,
    }

    print("\n" + "=" * 60, file=sys.stderr)
    print("STAGE 3: LLM synthesis", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    stage3 = {}
    try:
        client = _get_llm_client()
        deployment = os.getenv("ANALYSIS_MODEL_DEPLOYMENT", "gpt-5-chat")

        stage3["failure_analysis"] = llm_failure_analysis(client, deployment, failures)
        stage3["success_synthesis"] = llm_success_synthesis(client, deployment, success, joined)
        stage3["prompt_improvement"] = llm_prompt_improvement(client, deployment, failures, anti_patterns, success)
        stage3["tool_gap_analysis"] = llm_tool_gap_analysis(client, deployment, anti_patterns, failures)
        stage3["metaprompt_improvement"] = llm_metaprompt_improvement(client, deployment, failures, joined)

        print(f"  Completed {len(stage3)} LLM analyses", file=sys.stderr)
    except Exception as e:
        print(f"  [ERROR] LLM stage failed: {e}", file=sys.stderr)
        print("  Continuing with Stage 2 results only...", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print("STAGE 4: Generating reports", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    generate_reports(stage2, stage3)

    print("\n" + "=" * 60, file=sys.stderr)
    print("DONE", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
