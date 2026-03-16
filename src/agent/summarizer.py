"""Output and error summarization for BuildAgent v2.0.

Handles the >2000 char gate. No LLM call if content is under threshold.
Fallback: truncate + append [truncated] if LLM is unavailable or fails.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SUMMARY_THRESHOLD = 2000
SUMMARY_TARGET = 1500

TOOL_OUTPUT_PROMPT = """Summarize the following {context_type} output.
Preserve all actionable information: error messages, file paths, version numbers,
commands, package names. Remove boilerplate, progress bars, and repeated lines.

OUTPUT ({length} chars):
{content}

Summary (under {target} chars):"""

ERROR_SUMMARY_PROMPT = """Summarize this Docker build error. Preserve:
- The exact error message and exit code
- The failing command/step
- Missing packages or files mentioned
- Any version mismatch info

Remove:
- Successful build steps that passed
- Download progress
- Cache hit/miss messages
- Repeated warnings

ERROR OUTPUT ({length} chars):
{content}

Summary (under {target} chars):"""


def summarize_output(
    content: str,
    context_type: str = "tool",
    llm=None,
) -> tuple[str, int, int]:
    """Summarize content if it exceeds the threshold.

    Args:
        content: The raw output string.
        context_type: Type of output for prompt context ('tool', 'build_error', etc.).
        llm: Optional LLM client with a call_nano(messages) method.
             If None or call fails, falls back to truncation.

    Returns:
        (summary_or_content, prompt_tokens, completion_tokens)
        Tokens are 0 if no LLM call was made.
    """
    if len(content) <= SUMMARY_THRESHOLD:
        return content, 0, 0

    # Try LLM summarization
    if llm is not None:
        try:
            if context_type == "build_error":
                prompt = ERROR_SUMMARY_PROMPT.format(
                    length=len(content),
                    content=content,
                    target=SUMMARY_TARGET,
                )
            else:
                prompt = TOOL_OUTPUT_PROMPT.format(
                    context_type=context_type,
                    length=len(content),
                    content=content,
                    target=SUMMARY_TARGET,
                )

            response = llm.call_nano([
                {"role": "system", "content": "You summarize outputs concisely, preserving actionable information."},
                {"role": "user", "content": prompt},
            ])
            return response.content, response.prompt_tokens, response.completion_tokens
        except Exception:
            logger.warning("LLM summarization failed, falling back to truncation", exc_info=True)

    # Fallback: truncate
    return _truncate(content), 0, 0


def _truncate(content: str) -> str:
    """Truncate content to threshold with a marker."""
    return content[:SUMMARY_THRESHOLD] + "\n... [truncated]"
