"""Tests for Phase H fixes:
- ListDirectory dotfile subpath crash
- VerifyBuildTool._last_result caching
- _make_messages_modifier history truncation
- worker_loop writes run_artifact on success
"""

import json
import os
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent.tools import ListDirectoryTool
from agent.verify_build import VerifyBuildTool
from agent.react_loop import _make_messages_modifier
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from db.models import BatchRun, RunRecord, VerifyBuildResult
from db.writer import DBWriter
from parallel.worker import worker_loop


# ═══════════════════════════════════════════════════════
# H1: ListDirectory — dotfile subpath fix
# ═══════════════════════════════════════════════════════

class TestListDirectoryDotfiles:
    """item.relative_to(repo_root.resolve()) must handle dotfiles and hidden dirs."""

    def test_list_with_hidden_files(self, tmp_path):
        (tmp_path / ".editorconfig").write_text("[*]\nindent_size=2\n")
        (tmp_path / ".gitignore").write_text("*.pyc\n")
        (tmp_path / "main.py").write_text("pass")

        tool = ListDirectoryTool(tmp_path)
        result = tool.execute(".")

        assert "Error" not in result
        assert ".editorconfig" in result
        assert ".gitignore" in result
        assert "main.py" in result

    def test_list_subdirectory_with_dotfiles(self, tmp_path):
        subdir = tmp_path / "src"
        subdir.mkdir()
        (subdir / ".hidden").write_text("hidden")
        (subdir / "code.py").write_text("pass")

        tool = ListDirectoryTool(tmp_path)
        result = tool.execute("src")

        assert "Error" not in result
        assert "src/.hidden" in result
        assert "src/code.py" in result

    def test_list_relative_repo_root(self, tmp_path, monkeypatch):
        """Regression: relative repo_root (e.g. Path('workdir/...')) must not crash."""
        (tmp_path / ".asf.yaml").write_text("asf: config")
        (tmp_path / "README.md").write_text("readme")

        # Simulate a relative path as repo_root
        monkeypatch.chdir(tmp_path.parent)
        relative_root = Path(tmp_path.name)
        tool = ListDirectoryTool(relative_root)
        result = tool.execute(".")

        assert "Error" not in result
        assert ".asf.yaml" in result
        assert "README.md" in result


# ═══════════════════════════════════════════════════════
# H2: VerifyBuildTool._last_result caching
# ═══════════════════════════════════════════════════════

class TestVerifyBuildLastResult:
    def _make_tool(self, tmp_path, review, build_ok=True, smoke_exit=0):
        (tmp_path / "Dockerfile").write_text("FROM alpine:3.19\nRUN echo ok\n")
        mock_llm = MagicMock()
        resp = MagicMock()
        resp.content = json.dumps(review)
        resp.prompt_tokens = 10
        resp.completion_tokens = 5
        mock_llm.call_nano.return_value = resp

        mock_ops = MagicMock()
        mock_ops.build.return_value = (build_ok, "" if build_ok else "build err", 500)
        mock_ops.run_container.return_value = (smoke_exit, "output", False)

        return VerifyBuildTool(tmp_path, "img", mock_ops, mock_llm)

    def test_last_result_none_before_execute(self, tmp_path):
        tool = VerifyBuildTool(tmp_path, "img", MagicMock(), MagicMock())
        assert tool._last_result is None

    def test_last_result_set_after_execute_accepted(self, tmp_path):
        review = {"approved": True, "concerns": [], "smoke_test_commands": ["echo ok"]}
        tool = self._make_tool(tmp_path, review)
        tool.execute()
        assert isinstance(tool._last_result, VerifyBuildResult)
        assert tool._last_result.status == "accepted"

    def test_last_result_set_after_execute_rejected(self, tmp_path):
        review = {"approved": False, "concerns": ["bad"], "smoke_test_commands": ["echo ok"]}
        tool = self._make_tool(tmp_path, review)
        tool.execute()
        assert tool._last_result is not None
        assert tool._last_result.status == "rejected"

    def test_last_result_set_after_execute_build_failed(self, tmp_path):
        review = {"approved": True, "concerns": [], "smoke_test_commands": ["echo ok"]}
        tool = self._make_tool(tmp_path, review, build_ok=False)
        tool.execute()
        assert tool._last_result.status == "build_failed"

    def test_last_result_updated_on_second_call(self, tmp_path):
        review = {"approved": True, "concerns": [], "smoke_test_commands": ["echo ok"]}
        tool = self._make_tool(tmp_path, review)
        tool.execute()
        first = tool._last_result
        tool.execute()
        second = tool._last_result
        assert isinstance(second, VerifyBuildResult)
        assert second is not first  # new instance each call

    def test_execute_return_matches_last_result_status(self, tmp_path):
        review = {"approved": True, "concerns": [], "smoke_test_commands": ["echo ok"]}
        tool = self._make_tool(tmp_path, review)
        output = tool.execute()
        assert tool._last_result.status in output


# ═══════════════════════════════════════════════════════
# H3: _make_messages_modifier
# ═══════════════════════════════════════════════════════

class TestMessagesModifier:
    def test_prepends_system_message_always(self):
        modifier = _make_messages_modifier("You are BuildAgent.")
        result = modifier([HumanMessage(content="hello")])
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "You are BuildAgent."

    def test_empty_messages_returns_only_system(self):
        modifier = _make_messages_modifier("sys")
        result = modifier([])
        assert len(result) == 1
        assert isinstance(result[0], SystemMessage)

    def test_fewer_than_keep_full_not_truncated(self):
        """With ≤8 messages, all tool messages preserved intact."""
        modifier = _make_messages_modifier("sys")
        long = "x" * 3000
        messages = [ToolMessage(content=long, tool_call_id=f"id{i}") for i in range(6)]
        result = modifier(messages)
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert all(len(m.content) == 3000 for m in tool_msgs)

    def test_old_long_tool_messages_truncated(self):
        """Messages beyond the keep_full window that exceed 1500 chars are truncated."""
        modifier = _make_messages_modifier("sys")
        long = "x" * 3000
        messages = [ToolMessage(content=long, tool_call_id=f"id{i}") for i in range(10)]
        result = modifier(messages)
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]

        # First 2 are old (10 - 8 = 2)
        assert "[truncated]" in tool_msgs[0].content
        assert "[truncated]" in tool_msgs[1].content
        # Remaining 8 intact
        for msg in tool_msgs[2:]:
            assert len(msg.content) == 3000
            assert "[truncated]" not in msg.content

    def test_old_short_tool_messages_not_truncated(self):
        modifier = _make_messages_modifier("sys")
        messages = [ToolMessage(content="short", tool_call_id=f"id{i}") for i in range(10)]
        result = modifier(messages)
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        assert all(m.content == "short" for m in tool_msgs)

    def test_old_ai_messages_not_truncated(self):
        """Only ToolMessages are truncated — AIMessages are left alone."""
        modifier = _make_messages_modifier("sys")
        long = "x" * 3000
        messages = [AIMessage(content=long) for _ in range(10)]
        result = modifier(messages)
        ai_msgs = [m for m in result if isinstance(m, AIMessage)]
        assert all(len(m.content) == 3000 for m in ai_msgs)

    def test_truncated_content_length(self):
        """Truncated content is ≤ _OLD_TOOL_MSG_MAX + len('\n...[truncated]')."""
        from agent.react_loop import _OLD_TOOL_MSG_MAX
        modifier = _make_messages_modifier("sys")
        messages = [ToolMessage(content="y" * 5000, tool_call_id=f"id{i}") for i in range(10)]
        result = modifier(messages)
        tool_msgs = [m for m in result if isinstance(m, ToolMessage)]
        for msg in tool_msgs[:2]:  # old messages
            assert len(msg.content) <= _OLD_TOOL_MSG_MAX + 20  # small slack for "\n...[truncated]"

    def test_modifier_callable_is_returned(self):
        """_make_messages_modifier returns a callable."""
        modifier = _make_messages_modifier("sys")
        assert callable(modifier)

    def test_original_messages_list_not_mutated(self):
        """The modifier should not mutate the input list."""
        modifier = _make_messages_modifier("sys")
        long = "x" * 3000
        messages = [ToolMessage(content=long, tool_call_id=f"id{i}") for i in range(10)]
        original_contents = [m.content for m in messages]
        modifier(messages)
        for msg, orig in zip(messages, original_contents):
            assert msg.content == orig


# ═══════════════════════════════════════════════════════
# H4: worker_loop writes run_artifact
# ═══════════════════════════════════════════════════════

class TestWorkerArtifact:
    def _run_worker(self, db, tmp_path, final_dockerfile, smoke_passed):
        batch = BatchRun(worker_count=1, repo_count=1)
        db.write_batch_start(batch)

        def mock_run_agent(*_args, run_record, **_kwargs):
            run_record.smoke_test_passed = smoke_passed
            run_record.final_dockerfile = final_dockerfile
            return run_record

        with patch("parallel.worker.LLMClient"):
            with patch("parallel.worker.clone_repo"):
                with patch("parallel.worker.generate_blueprint",
                           return_value=({"base_image": "python:3.11", "language": "python",
                                          "repo_type": "library"}, 100, 50)):
                    with patch("parallel.worker.run_agent", side_effect=mock_run_agent):
                        worker_loop(
                            worker_id=0,
                            repo_url="https://github.com/test/repo",
                            batch_id=batch.id,
                            image_catalog="catalog",
                            rate_limiter=MagicMock(),
                            build_semaphore=threading.Semaphore(1),
                            disk_monitor=MagicMock(),
                            db=db,
                            workdir=str(tmp_path / "work"),
                        )
        return batch

    def test_artifact_written_on_success(self, tmp_path):
        db = DBWriter("sqlite:///:memory:")
        self._run_worker(db, tmp_path,
                         final_dockerfile="FROM python:3.11\nRUN echo ok\n",
                         smoke_passed=True)
        rows = db._query("SELECT artifact_type, file_name, content FROM run_artifact")
        assert len(rows) == 1
        assert rows[0][0] == "dockerfile"
        assert rows[0][1] == "Dockerfile"
        assert "FROM python:3.11" in rows[0][2]
        db.close()

    def test_artifact_not_written_when_no_dockerfile(self, tmp_path):
        db = DBWriter("sqlite:///:memory:")
        self._run_worker(db, tmp_path,
                         final_dockerfile=None,
                         smoke_passed=False)
        rows = db._query("SELECT COUNT(*) FROM run_artifact")
        assert rows[0][0] == 0
        db.close()

    def test_artifact_not_written_on_error(self, tmp_path):
        """If clone fails (error path), no artifact written."""
        db = DBWriter("sqlite:///:memory:")
        batch = BatchRun(worker_count=1, repo_count=1)
        db.write_batch_start(batch)

        with patch("parallel.worker.LLMClient"):
            with patch("parallel.worker.clone_repo", side_effect=Exception("network")):
                worker_loop(
                    worker_id=0,
                    repo_url="https://github.com/test/repo",
                    batch_id=batch.id,
                    image_catalog="catalog",
                    rate_limiter=MagicMock(),
                    build_semaphore=threading.Semaphore(1),
                    disk_monitor=MagicMock(),
                    db=db,
                    workdir=str(tmp_path / "work"),
                )

        rows = db._query("SELECT COUNT(*) FROM run_artifact")
        assert rows[0][0] == 0
        db.close()


# ═══════════════════════════════════════════════════════
# H5: Step token counting from AIMessage.usage_metadata
# ═══════════════════════════════════════════════════════

class TestStepTokenCounting:
    """Tokens from AIMessage.usage_metadata flow into StepRecord and IterationRecord."""

    def _make_stream_chunks(self, tool_call_id, tool_name, tool_args,
                            tool_output, prompt_tokens, completion_tokens):
        """Build a minimal pair of LangGraph stream chunks (agent + tools)."""
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": tool_call_id, "name": tool_name, "args": tool_args}],
        )
        ai_msg.usage_metadata = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        }
        agent_chunk = {"agent": {"messages": [ai_msg]}}
        tool_msg = ToolMessage(content=tool_output, tool_call_id=tool_call_id)
        tools_chunk = {"tools": {"messages": [tool_msg]}}
        return [agent_chunk, tools_chunk]

    def _run_iteration(self, db, chunks):
        from agent.react_loop import run_iteration
        from db.models import IterationRecord

        mock_llm = MagicMock()
        mock_llm.nano = MagicMock()
        mock_verify = MagicMock()
        mock_verify._last_result = None
        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter(chunks)

        iteration = IterationRecord(run_id="run-1", iteration_number=1, status="running")
        db._execute(
            "INSERT INTO iteration (id, run_id, iteration_number, status, started_at, injected_lessons) VALUES (?,?,?,?,?,?)",
            (iteration.id, iteration.run_id, iteration.iteration_number,
             iteration.status, iteration.started_at, None),
        )
        with patch("agent.react_loop.create_react_agent", return_value=mock_agent):
            with patch("agent.react_loop.summarize_output", return_value=("output", 50, 10)):
                with patch("agent.react_loop._to_langchain_tools", return_value=[]):
                    return run_iteration(
                        prompt="system",
                        tools=[],
                        llm=mock_llm,
                        db=db,
                        iteration=iteration,
                        verify_tool=mock_verify,
                        max_steps=10,
                    )

    def test_step_records_prompt_and_completion_tokens(self):
        db = DBWriter("sqlite:///:memory:")
        chunks = self._make_stream_chunks(
            "tc1", "ListDirectory", {"path": "."}, "main.py", 1200, 80
        )
        result = self._run_iteration(db, chunks)
        assert result.steps[0].prompt_tokens == 1200
        assert result.steps[0].completion_tokens == 80
        db.close()

    def test_iteration_accumulates_step_plus_summary_tokens(self):
        db = DBWriter("sqlite:///:memory:")
        # Two sequential tool calls
        chunks = (
            self._make_stream_chunks("tc1", "ListDirectory", {"path": "."}, "main.py", 500, 30) +
            self._make_stream_chunks("tc2", "ReadFile", {"path": "main.py"}, "pass", 600, 40)
        )
        result = self._run_iteration(db, chunks)
        # prompt_tokens = (500 + 50) + (600 + 50) = 1200  (step + summarizer per step)
        assert result.prompt_tokens == 1200
        # completion_tokens = (30 + 10) + (40 + 10) = 90
        assert result.completion_tokens == 90
        db.close()

    def test_step_duration_ms_is_non_negative(self):
        db = DBWriter("sqlite:///:memory:")
        chunks = self._make_stream_chunks("tc1", "ListDirectory", {"path": "."}, "f.py", 100, 10)
        result = self._run_iteration(db, chunks)
        assert result.steps[0].duration_ms is not None
        assert result.steps[0].duration_ms >= 0
        db.close()

    def test_missing_usage_metadata_defaults_to_zero(self):
        """If AIMessage has no usage_metadata, tokens default to 0 without crashing."""
        from agent.react_loop import run_iteration
        from db.models import IterationRecord

        db = DBWriter("sqlite:///:memory:")

        ai_msg = AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "ListDirectory", "args": {"path": "."}}],
        )
        # Explicitly no usage_metadata
        chunks = [
            {"agent": {"messages": [ai_msg]}},
            {"tools": {"messages": [ToolMessage(content="file.py", tool_call_id="tc1")]}},
        ]

        mock_llm = MagicMock()
        mock_llm.nano = MagicMock()
        mock_verify = MagicMock()
        mock_verify._last_result = None
        mock_agent = MagicMock()
        mock_agent.stream.return_value = iter(chunks)

        iteration = IterationRecord(run_id="run-1", iteration_number=1, status="running")
        db._execute(
            "INSERT INTO iteration (id, run_id, iteration_number, status, started_at, injected_lessons) VALUES (?,?,?,?,?,?)",
            (iteration.id, iteration.run_id, iteration.iteration_number,
             iteration.status, iteration.started_at, None),
        )

        with patch("agent.react_loop.create_react_agent", return_value=mock_agent):
            with patch("agent.react_loop.summarize_output", return_value=("output", 0, 0)):
                with patch("agent.react_loop._to_langchain_tools", return_value=[]):
                    result = run_iteration(
                        prompt="system", tools=[], llm=mock_llm, db=db,
                        iteration=iteration, verify_tool=mock_verify, max_steps=10,
                    )

        assert result.steps[0].prompt_tokens == 0
        assert result.steps[0].completion_tokens == 0
        db.close()


# ═══════════════════════════════════════════════════════
# I1: Multi-deployment round-robin for gpt-5-nano
# ═══════════════════════════════════════════════════════

class TestMultiDeploymentRoundRobin:
    """LLMClient selects nano deployment by worker_id % len(deployments)."""

    def _make_client(self, deployment_env: str, worker_id: int):
        env = {
            "AZURE_OPENAI_DEPLOYMENT_NANO": deployment_env,
            "AZURE_OPENAI_DEPLOYMENT_CHAT": "chat-deploy",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "fake-key",
        }
        mock_limiter = MagicMock()
        with patch.dict(os.environ, env, clear=False):
            with patch("agent.llm.AzureChatOpenAI") as mock_azure:
                mock_azure.return_value = MagicMock()
                from agent.llm import LLMClient
                client = LLMClient(mock_limiter, worker_id=worker_id)
                # Return list of (kwargs, instance) from each AzureChatOpenAI call
                calls = mock_azure.call_args_list
                return client, calls

    def test_single_deployment_always_selected(self):
        """With one deployment, all workers use it."""
        for wid in range(5):
            _, calls = self._make_client("only-nano", wid)
            nano_call = calls[0]
            assert nano_call.kwargs["azure_deployment"] == "only-nano"

    def test_two_deployments_alternates(self):
        """Worker 0 → first, worker 1 → second, worker 2 → first, etc."""
        for wid, expected in [(0, "nano-a"), (1, "nano-b"), (2, "nano-a"), (3, "nano-b")]:
            _, calls = self._make_client("nano-a,nano-b", wid)
            nano_call = calls[0]
            assert nano_call.kwargs["azure_deployment"] == expected, \
                f"worker {wid}: expected {expected}, got {nano_call.kwargs['azure_deployment']}"

    def test_three_deployments_cycles(self):
        """Three deployments cycle across workers."""
        mapping = {0: "d1", 1: "d2", 2: "d3", 3: "d1", 4: "d2", 5: "d3"}
        for wid, expected in mapping.items():
            _, calls = self._make_client("d1,d2,d3", wid)
            nano_call = calls[0]
            assert nano_call.kwargs["azure_deployment"] == expected

    def test_whitespace_in_env_var_stripped(self):
        """Spaces around deployment names are stripped."""
        _, calls = self._make_client("  nano-x , nano-y  ", 1)
        nano_call = calls[0]
        assert nano_call.kwargs["azure_deployment"] == "nano-y"

    def test_chat_deployment_unaffected(self):
        """Chat deployment is always the single configured value regardless of worker_id."""
        for wid in range(4):
            _, calls = self._make_client("nano-a,nano-b", wid)
            chat_call = calls[1]
            assert chat_call.kwargs["azure_deployment"] == "chat-deploy"

    def test_default_worker_id_zero(self):
        """worker_id defaults to 0, selecting the first deployment."""
        env = {
            "AZURE_OPENAI_DEPLOYMENT_NANO": "first,second",
            "AZURE_OPENAI_DEPLOYMENT_CHAT": "chat-deploy",
            "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "fake-key",
        }
        mock_limiter = MagicMock()
        with patch.dict(os.environ, env, clear=False):
            with patch("agent.llm.AzureChatOpenAI") as mock_azure:
                mock_azure.return_value = MagicMock()
                from agent.llm import LLMClient
                LLMClient(mock_limiter)  # no worker_id
                nano_call = mock_azure.call_args_list[0]
                assert nano_call.kwargs["azure_deployment"] == "first"
