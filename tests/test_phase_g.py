"""Tests for Phase G: CLI + Batch Runner + Worker.

All tests use mocked LLM, Docker, and git operations — no real external calls.
"""

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parallel.worker import make_slug, clone_repo, worker_loop
from db.models import BatchRun, RunRecord
from db.writer import DBWriter


# ═══════════════════════════════════════════════════════
# G1: Worker Utilities
# ═══════════════════════════════════════════════════════

class TestMakeSlug:
    def test_github_url(self):
        assert make_slug("https://github.com/pallets/flask") == "pallets-flask"

    def test_github_url_with_git_suffix(self):
        assert make_slug("https://github.com/pallets/flask.git") == "pallets-flask"

    def test_trailing_slash(self):
        assert make_slug("https://github.com/pallets/flask/") == "pallets-flask"

    def test_special_chars(self):
        slug = make_slug("https://github.com/some.org/my_project")
        # dots become dashes, underscores are preserved
        assert slug == "some-org-my_project"

    def test_lowercase(self):
        assert make_slug("https://github.com/Owner/REPO") == "owner-repo"


class TestCloneRepo:
    def test_clone_calls_git(self, tmp_path):
        dest = tmp_path / "clone_target"
        with patch("parallel.worker.subprocess.run") as mock_run:
            clone_repo("https://github.com/test/repo", dest)
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "git"
            assert cmd[1] == "clone"
            assert "--depth" in cmd
            assert "https://github.com/test/repo" in cmd
            assert str(dest) in cmd

    def test_clone_creates_parent_dir(self, tmp_path):
        dest = tmp_path / "a" / "b" / "c" / "repo"
        with patch("parallel.worker.subprocess.run"):
            clone_repo("https://github.com/test/repo", dest)
        assert dest.parent.exists()


# ═══════════════════════════════════════════════════════
# G2: Worker Loop (mocked)
# ═══════════════════════════════════════════════════════

class TestWorkerLoop:
    @pytest.fixture
    def db(self):
        writer = DBWriter("sqlite:///:memory:")
        yield writer
        writer.close()

    @pytest.fixture
    def shared_resources(self):
        return {
            "rate_limiter": MagicMock(),
            "build_semaphore": threading.Semaphore(2),
            "disk_monitor": MagicMock(),
        }

    def test_worker_records_error_on_clone_failure(self, db, shared_resources, tmp_path):
        """If clone fails, run is recorded as error."""
        batch = BatchRun(worker_count=1, repo_count=1)
        db.write_batch_start(batch)

        with patch("parallel.worker.LLMClient"):
            with patch("parallel.worker.clone_repo", side_effect=Exception("clone failed")):
                worker_loop(
                    worker_id=0,
                    repo_url="https://github.com/test/repo",
                    batch_id=batch.id,
                    image_catalog="catalog",
                    **shared_resources,
                    db=db,
                    workdir=str(tmp_path / "work"),
                )

        rows = db._query("SELECT status, error_message FROM run WHERE batch_id=?", (batch.id,))
        assert len(rows) == 1
        assert rows[0][0] == "error"
        assert "clone failed" in rows[0][1]

    def test_worker_records_error_on_blueprint_failure(self, db, shared_resources, tmp_path):
        """If blueprint generation fails, run is recorded as error."""
        batch = BatchRun(worker_count=1, repo_count=1)
        db.write_batch_start(batch)

        with patch("parallel.worker.LLMClient"):
            with patch("parallel.worker.clone_repo"):
                with patch("parallel.worker.generate_blueprint", side_effect=Exception("LLM down")):
                    worker_loop(
                        worker_id=0,
                        repo_url="https://github.com/test/repo",
                        batch_id=batch.id,
                        image_catalog="catalog",
                        **shared_resources,
                        db=db,
                        workdir=str(tmp_path / "work"),
                    )

        rows = db._query("SELECT status, error_message FROM run WHERE batch_id=?", (batch.id,))
        assert rows[0][0] == "error"
        assert "LLM down" in rows[0][1]

    def test_worker_cleanup_always_runs(self, db, shared_resources, tmp_path):
        """Cleanup (rmtree + docker cleanup) runs even on error."""
        batch = BatchRun(worker_count=1, repo_count=1)
        db.write_batch_start(batch)

        mock_docker = MagicMock()

        with patch("parallel.worker.LLMClient"):
            with patch("parallel.worker.clone_repo", side_effect=Exception("fail")):
                with patch("parallel.worker.DockerOps", return_value=mock_docker):
                    with patch("parallel.worker.shutil.rmtree") as mock_rmtree:
                        worker_loop(
                            worker_id=0,
                            repo_url="https://github.com/test/repo",
                            batch_id=batch.id,
                            image_catalog="catalog",
                            **shared_resources,
                            db=db,
                            workdir=str(tmp_path / "work"),
                        )

                        mock_rmtree.assert_called_once()
                        mock_docker.cleanup.assert_called_once()

    def test_worker_calls_disk_monitor(self, db, shared_resources, tmp_path):
        """Worker checks disk space before starting."""
        batch = BatchRun(worker_count=1, repo_count=1)
        db.write_batch_start(batch)

        with patch("parallel.worker.LLMClient"):
            with patch("parallel.worker.clone_repo", side_effect=Exception("fail")):
                worker_loop(
                    worker_id=0,
                    repo_url="https://github.com/test/repo",
                    batch_id=batch.id,
                    image_catalog="catalog",
                    **shared_resources,
                    db=db,
                    workdir=str(tmp_path / "work"),
                )

        shared_resources["disk_monitor"].check_or_wait.assert_called_once()


# ═══════════════════════════════════════════════════════
# G3: Build Agent CLI
# ═══════════════════════════════════════════════════════

class TestBuildAgentCLI:
    def test_cli_runs_and_handles_error(self):
        """CLI parses args, runs pipeline, and handles errors gracefully."""
        from build_agent import main

        mock_llm = MagicMock()

        with patch("build_agent.LLMClient", return_value=mock_llm):
            with patch("build_agent.GlobalRateLimiter"):
                with patch("build_agent.DBWriter") as mock_db_cls:
                    mock_db = MagicMock()
                    mock_db_cls.return_value = mock_db
                    with patch("build_agent.DiskSpaceMonitor"):
                        with patch("build_agent.ImageCatalog") as mock_cat:
                            mock_cat.return_value.get.return_value = "catalog"
                            with patch("build_agent.clone_repo", side_effect=Exception("no network")):
                                result = main([
                                    "https://github.com/test/repo",
                                    "--db", "sqlite:///:memory:",
                                ])
                                assert result == 1
                                # Verify DB was written to
                                mock_db.write_batch_start.assert_called_once()
                                mock_db.write_run_start.assert_called_once()
                                mock_db.write_run_finish.assert_called_once()


# ═══════════════════════════════════════════════════════
# G4: Batch Runner
# ═══════════════════════════════════════════════════════

class TestBatchRunner:
    def test_read_repo_list(self, tmp_path):
        from batch_runner import read_repo_list

        repo_file = tmp_path / "repos.txt"
        repo_file.write_text(
            "# Comment line\n"
            "https://github.com/test/repo1\n"
            "\n"
            "https://github.com/test/repo2\n"
            "# Another comment\n"
            "https://github.com/test/repo3\n"
        )

        repos = read_repo_list(str(repo_file))
        assert repos == [
            "https://github.com/test/repo1",
            "https://github.com/test/repo2",
            "https://github.com/test/repo3",
        ]

    def test_read_empty_repo_list(self, tmp_path):
        from batch_runner import read_repo_list

        repo_file = tmp_path / "empty.txt"
        repo_file.write_text("# Only comments\n")

        repos = read_repo_list(str(repo_file))
        assert repos == []

    def test_print_summary(self):
        """print_summary queries DB and prints stats."""
        from batch_runner import print_summary

        db = DBWriter("sqlite:///:memory:")
        batch = BatchRun(worker_count=2, repo_count=3)
        db.write_batch_start(batch)

        for i, status in enumerate(["success", "success", "failure"]):
            run = RunRecord(
                batch_id=batch.id,
                repo_url=f"https://github.com/test/repo{i}",
                repo_slug=f"test-repo{i}",
                status=status,
                total_prompt_tokens=1000 * (i + 1),
                total_completion_tokens=200 * (i + 1),
                iteration_count=i + 1,
            )
            db.write_run_start(run)
            db.write_run_finish(run)

        # Should not raise
        print_summary(db, batch.id, 120.5)
        db.close()

    def test_batch_crash_recovery(self):
        """Already-successful slugs are skipped."""
        db = DBWriter("sqlite:///:memory:")
        batch = BatchRun(worker_count=1, repo_count=2)
        db.write_batch_start(batch)

        run = RunRecord(
            batch_id=batch.id,
            repo_url="https://github.com/test/already-done",
            repo_slug="test-already-done",
            status="success",
        )
        db.write_run_start(run)
        db.write_run_finish(run)

        completed = db.get_successful_slugs(batch.id)
        assert "test-already-done" in completed
        db.close()


# ═══════════════════════════════════════════════════════
# G5: Integration — Worker writes full DB chain
# ═══════════════════════════════════════════════════════

class TestWorkerDBChain:
    def test_worker_writes_run_record(self):
        """Even on early failure, a complete run record is persisted."""
        db = DBWriter("sqlite:///:memory:")
        batch = BatchRun(worker_count=1, repo_count=1)
        db.write_batch_start(batch)

        disk_monitor = MagicMock()
        rate_limiter = MagicMock()

        with patch("parallel.worker.LLMClient"):
            with patch("parallel.worker.clone_repo", side_effect=Exception("network error")):
                worker_loop(
                    worker_id=0,
                    repo_url="https://github.com/test/myrepo",
                    batch_id=batch.id,
                    image_catalog="catalog",
                    rate_limiter=rate_limiter,
                    build_semaphore=threading.Semaphore(1),
                    disk_monitor=disk_monitor,
                    db=db,
                    workdir="/tmp/test_workdir",
                )

        rows = db._query(
            "SELECT repo_slug, status, finished_at, duration_ms FROM run WHERE batch_id=?",
            (batch.id,),
        )
        assert len(rows) == 1
        slug, status, finished, duration = rows[0]
        assert slug == "test-myrepo"
        assert status == "error"
        assert finished is not None
        assert duration >= 0
        db.close()
