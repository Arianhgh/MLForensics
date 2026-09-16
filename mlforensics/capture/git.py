"""Best-effort Git evidence capture."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _git(args: list[str], cwd: str | Path | None = None, timeout: float = 10.0) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_root(path: str | Path | None = None) -> Path | None:
    result = _git(["rev-parse", "--show-toplevel"], path)
    return Path(result) if result else None


def capture_git(
    path: str | Path | None = None, *, include_diff: bool = True, max_diff_bytes: int = 250_000
) -> dict[str, Any]:
    """Return Git metadata without failing when the current directory is not a repository."""

    root = git_root(path)
    if root is None:
        return {"available": False}
    commit = _git(["rev-parse", "HEAD"], root)
    branch = _git(["branch", "--show-current"], root)
    status = _git(["status", "--porcelain=v1"], root) or ""
    result: dict[str, Any] = {
        "available": True,
        "root": str(root),
        "commit": commit,
        "branch": branch or None,
        "dirty": bool(status),
        "status": status.splitlines(),
    }
    if include_diff:
        diff = _git(["diff", "--no-ext-diff", "--binary"], root) or ""
        result["diff"] = diff[:max_diff_bytes]
        result["diff_truncated"] = len(diff) > max_diff_bytes
    return result


def capture_git_metadata(path: str | Path | None = None) -> dict[str, Any]:
    """Return metadata even when the caller is outside a Git work tree."""

    result = capture_git(path, include_diff=False)
    if not result.get("available"):
        result.setdefault("path", str(path) if path is not None else None)
        return result
    result["is_dirty"] = result.get("dirty")
    result["git_dir"] = _git(["rev-parse", "--git-dir"], result.get("root"))
    result["describe"] = _git(["describe", "--always", "--long", "--dirty"], result.get("root"))
    return result


def capture_git_diff(
    path: str | Path | None = None, *, max_diff_bytes: int = 250_000
) -> dict[str, Any]:
    """Return working-tree and staged patches plus untracked paths."""

    root = git_root(path)
    if root is None:
        return {"available": False, "path": str(path) if path is not None else None}
    working = _git(["diff", "--no-ext-diff", "--binary"], root) or ""
    staged = _git(["diff", "--cached", "--no-ext-diff", "--binary"], root) or ""
    status = _git(["status", "--porcelain=v1"], root) or ""
    untracked = sorted(line[3:] for line in status.splitlines() if line.startswith("?? "))
    return {
        "available": True,
        "root": str(root),
        "patch": working[:max_diff_bytes],
        "working_tree_patch": working[:max_diff_bytes],
        "staged_patch": staged[:max_diff_bytes],
        "untracked": untracked,
        "status": status.splitlines(),
        "diff_truncated": len(working) > max_diff_bytes,
    }


git_metadata = capture_git_metadata


def git_diff(base: str, head: str = "HEAD", path: str | Path | None = None) -> str:
    """Return a textual diff, raising a useful error for invalid revisions."""

    root = git_root(path)
    if root is None:
        raise RuntimeError("not a Git repository")
    try:
        result = subprocess.run(
            ["git", "diff", "--no-ext-diff", f"{base}..{head}"],
            cwd=root,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(exc.stderr.strip() or "git diff failed") from exc
    return result.stdout


def git_commits(good: str, bad: str = "HEAD", path: str | Path | None = None) -> list[str]:
    root = git_root(path)
    if root is None:
        raise RuntimeError("not a Git repository")
    result = _git(["rev-list", "--ancestry-path", "--reverse", f"{good}..{bad}"], root)
    if result is None:
        raise RuntimeError(f"unable to enumerate commits between {good} and {bad}")
    return [good, *[line for line in result.splitlines() if line]]
