import subprocess

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
