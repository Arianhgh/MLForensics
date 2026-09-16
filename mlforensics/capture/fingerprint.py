"""Deterministic file and dataset fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _digest_stream(stream: Any, algorithm: str, chunk_size: int) -> str:
    digest = hashlib.new(algorithm)
    while True:
        chunk = stream.read(chunk_size)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(
    path: str | Path, algorithm: str = "sha256", chunk_size: int = 1024 * 1024
) -> dict[str, Any]:
    """Hash file bytes; symlinks are represented by their link target."""

    target = Path(path)
    if target.is_symlink():
        link_target = os.readlink(target)
        digest = hashlib.new(algorithm, link_target.encode("utf-8")).hexdigest()
        return {
            "path": str(target),
            "kind": "symlink",
            "target": link_target,
            "size": len(link_target),
            "algorithm": algorithm,
            "digest": digest,
            algorithm: digest,
        }
    if not target.is_file():
        raise ValueError(f"not a regular file: {target}")
    with target.open("rb") as stream:
        digest = _digest_stream(stream, algorithm, chunk_size)
    size = target.stat().st_size
    return {
        "path": str(target),
        "kind": "file",
        "size": size,
        "algorithm": algorithm,
        "digest": digest,
        algorithm: digest,
    }


def _relative_record(root: Path, path: Path, algorithm: str, chunk_size: int) -> dict[str, Any]:
    record = fingerprint_file(path, algorithm=algorithm, chunk_size=chunk_size)
    record["path"] = path.relative_to(root).as_posix()
    return record


def fingerprint_directory(
    path: str | Path, algorithm: str = "sha256", chunk_size: int = 1024 * 1024
) -> dict[str, Any]:
    """Fingerprint a directory from sorted relative paths and file contents.

    Modification times, inode numbers, and absolute roots are intentionally not
    included, so copying the same dataset to another directory preserves the
    dataset digest.
    """

    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    records: list[dict[str, Any]] = []
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(dirs)
        for filename in sorted(files):
            candidate = Path(current) / filename
            records.append(_relative_record(root, candidate, algorithm, chunk_size))
    records.sort(key=lambda item: item["path"])
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.new(algorithm, encoded).hexdigest()
    return {
        "path": str(root),
        "kind": "directory",
        "algorithm": algorithm,
        "digest": digest,
        algorithm: digest,
        "files": records,
        "count": len(records),
    }


def fingerprint_dataset(
    path: str | Path | Iterable[str | Path],
    algorithm: str = "sha256",
    chunk_size: int = 1024 * 1024,
) -> dict[str, Any]:
    """Fingerprint one file/directory or an ordered-independent collection of them."""

    if isinstance(path, (str, Path)):
        target = Path(path)
        return (
            fingerprint_file(target, algorithm, chunk_size)
            if target.is_file() or target.is_symlink()
            else fingerprint_directory(target, algorithm, chunk_size)
        )

    entries: list[dict[str, Any]] = []
    for item in sorted((Path(value) for value in path), key=lambda value: str(value)):
        entries.append(fingerprint_dataset(item, algorithm, chunk_size))
    # Absolute source paths are useful in the returned diagnostic payload but
    # must not influence the content digest of a copied dataset.
    canonical_entries = [
        {
            key: entry[key]
            for key in ("kind", "size", "count", "algorithm", "digest", algorithm)
            if key in entry
        }
        for entry in entries
    ]
    encoded = json.dumps(canonical_entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.new(algorithm, encoded).hexdigest()
    return {
        "kind": "dataset",
        "algorithm": algorithm,
        "digest": digest,
        algorithm: digest,
        "entries": entries,
        "count": len(entries),
    }


fingerprint_path = fingerprint_dataset
file_fingerprint = fingerprint_file
dataset_fingerprint = fingerprint_dataset
