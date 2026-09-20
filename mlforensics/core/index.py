"""Local JSON index for run IDs, aliases, and capsule paths."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]


def index_path(root: str | Path) -> Path:
    return Path(root) / "index.json"


def lock_path(root: str | Path) -> Path:
    return Path(root) / "index.lock"


@contextmanager
def _index_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    handle = lock_path(root).open("a+b")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        elif msvcrt is not None:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        yield
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        elif msvcrt is not None:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        handle.close()


def _empty_index() -> dict[str, Any]:
    return {"runs": {}, "aliases": {}}


def _parse_index(data: Any) -> dict[str, Any] | None:
    if not isinstance(data, Mapping):
        return None
    runs = data.get("runs", {})
    aliases = data.get("aliases", {})
    if not isinstance(runs, Mapping) or not isinstance(aliases, Mapping):
        return None
    return {"runs": dict(runs), "aliases": dict(aliases)}


def rebuild_index(root: str | Path) -> dict[str, Any]:
    """Rebuild an index by reading capsule metadata under ``root``."""
    payload = _empty_index()
    target = Path(root)
    if not target.is_dir():
        return payload
    for child in sorted(target.iterdir()):
        if child.name.startswith("index.") or child.suffix in {".lock", ".tmp"}:
            continue
        if not (child.is_dir() or child.suffix in {".mlcap", ".zip"}):
            continue
        try:
            from .capsule import RunCapsule

            capsule = RunCapsule.load(child, include_artifacts=False)
        except Exception:
            continue
        payload["runs"][capsule.run.run_id] = {
            "path": str(child.resolve() if child.exists() else child),
            "status": capsule.run.status,
        }
    return payload


def _recover_index(root: str | Path, path: Path) -> dict[str, Any]:
    corrupt = path.with_name(f"index.json.corrupt-{int(time.time())}")
    try:
        path.replace(corrupt)
    except OSError:
        pass
    rebuilt = rebuild_index(root)
    try:
        record_index(root, rebuilt)
    except OSError:
        pass
    return rebuilt


def load_index(root: str | Path, *, recover: bool = True) -> dict[str, Any]:
    path = index_path(root)
    if not path.is_file():
        return _empty_index()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _recover_index(root, path) if recover else _empty_index()
    parsed = _parse_index(data)
    if parsed is not None:
        return parsed
    return _recover_index(root, path) if recover else _empty_index()


def record_index(root: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix="index.", suffix=".tmp", dir=target)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
        Path(temporary_name).replace(index_path(target))
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def record_run(
    root: str | Path,
    run_id: str,
    capsule_path: str | Path,
    *,
    alias: str | None = None,
    status: str | None = None,
) -> None:
    """Atomically record a run ID under a storage root."""
    target = Path(root)
    with _index_lock(target):
        payload = load_index(target)
        recorded = Path(capsule_path)
        if recorded.exists():
            recorded = recorded.resolve()
        payload["runs"][run_id] = {
            "path": str(recorded),
            "status": status,
        }
        if alias:
            payload["aliases"][alias] = run_id
        record_index(target, payload)


def resolve_run(root: str | Path, identifier: str) -> Path | None:
    """Resolve a path, run ID, or alias under a storage root."""
    direct = Path(identifier)
    if direct.exists():
        return direct
    payload = load_index(root)
    run_id = payload["aliases"].get(identifier, identifier)
    record = payload["runs"].get(run_id)
    if isinstance(record, Mapping) and record.get("path"):
        candidate = Path(str(record["path"]))
        if candidate.exists():
            return candidate
    return None
