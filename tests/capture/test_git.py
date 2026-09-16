import shutil
import subprocess
from pathlib import Path

import pytest

from mlforensics.capture import capture_git_diff, capture_git_metadata

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def test_git_metadata_and_diff_capture_fake_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "capture@example.test")
    _run_git(repo, "config", "user.name", "Capture Test")
    tracked = repo / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    _run_git(repo, "add", "tracked.txt")
    _run_git(repo, "commit", "-m", "initial")
    tracked.write_text("after\n", encoding="utf-8")
    (repo / "untracked.txt").write_text("new\n", encoding="utf-8")

    metadata = capture_git_metadata(repo)
    diff = capture_git_diff(repo)
    assert metadata["available"] is True
    assert metadata["commit"]
    assert metadata["is_dirty"] is True
    assert "before" in diff["patch"] and "after" in diff["patch"]
    assert diff["untracked"] == ["untracked.txt"]


def test_git_capture_gracefully_handles_non_repo(tmp_path: Path) -> None:
    assert capture_git_metadata(tmp_path)["available"] is False
    assert capture_git_diff(tmp_path)["available"] is False
