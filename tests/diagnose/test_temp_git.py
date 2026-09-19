import subprocess

import pytest

from mlforensics.diagnose.bisect import SubprocessGit


def test_subprocess_git_adapter_reads_temp_repository(tmp_path):
    def git(*args):
        subprocess.run(
            ["git", *args],
            cwd=tmp_path,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (tmp_path / "file.txt").write_text("one")
    git("add", "file.txt")
    git("commit", "-qm", "one")
    (tmp_path / "file.txt").write_text("two")
    git("commit", "-qam", "two")
    revisions = SubprocessGit(str(tmp_path)).commits()
    assert len(revisions) == 2


def test_subprocess_git_preflight_rejects_dirty_checkout(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("clean")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=tmp_path, check=True)
    (tmp_path / "untracked.txt").write_text("keep")

    with pytest.raises(RuntimeError, match="clean checkout"):
        SubprocessGit(str(tmp_path)).preflight()
