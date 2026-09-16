"""Deterministic dataset and file fingerprinting."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _update_json_hash(hasher, value: Any) -> None:
    hasher.update(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=repr
        ).encode("utf-8")
    )
    hasher.update(b"\n")


def fingerprint_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    hasher = hashlib.sha256()
    size = 0
    with source.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            hasher.update(chunk)
            size += len(chunk)
    return {
        "path": str(source.resolve()),
        "kind": "file",
        "sha256": hasher.hexdigest(),
        "digest": hasher.hexdigest(),
        "algorithm": "sha256",
        "size_bytes": size,
        "size": size,
    }


def infer_schema(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    columns = sorted({str(key) for row in rows for key in row})
    schema: dict[str, Any] = {}
    for column in columns:
        values = [row.get(column) for row in rows if row.get(column) is not None]
        types = sorted({type(value).__name__ for value in values})
        schema[column] = {"types": types, "nullable": len(values) != len(rows)}
    return schema


def _sample_rows(path: Path, sample_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = [dict(row) for _, row in zip(range(sample_size), reader)]
            return rows, {"format": "csv", "columns": reader.fieldnames or []}
    if suffix in {".jsonl", ".ndjson"}:
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip() and len(rows) < sample_size:
                    value = json.loads(line)
                    rows.append(value if isinstance(value, dict) else {"value": value})
        return rows, {"format": "jsonl"}
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        values = value if isinstance(value, list) else [value]
        rows = [
            item if isinstance(item, dict) else {"value": item} for item in values[:sample_size]
        ]
        return rows, {"format": "json"}
    return [], {"format": "binary_or_unknown"}


def fingerprint_dataset(
    source: str | Path | Iterable[Mapping[str, Any]],
    *,
    name: str | None = None,
    sample_size: int = 128,
) -> dict[str, Any]:
    """Fingerprint a file or an iterable of records without consuming it twice."""

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            from .fingerprint import fingerprint_directory as _fingerprint_directory

            directory = _fingerprint_directory(path)
            return {
                "name": name or path.name,
                "path": str(path.resolve()),
                "sha256": directory["digest"],
                "kind": "directory",
                "file_count": directory["count"],
                "files": directory["files"],
                "format": "directory",
                "sample_count": 0,
                "schema": {},
                "sample": [],
            }
        file_record = fingerprint_file(path)
        rows, details = _sample_rows(path, sample_size)
        record: dict[str, Any] = {
            "name": name or path.name,
            **file_record,
            **details,
            "sample_count": len(rows),
        }
    else:
        source_iter = iter(source)
        try:
            first = next(source_iter)
        except StopIteration:
            first = None
        if isinstance(first, (str, Path)):
            from .fingerprint import fingerprint_dataset as _fingerprint_dataset

            return _fingerprint_dataset([first, *source_iter])
        rows = []
        hasher = hashlib.sha256()
        count = 0
        if first is not None:
            source_iter = iter([first, *source_iter])
        for row in source_iter:
            count += 1
            if len(rows) < sample_size:
                rows.append(dict(row))
            _update_json_hash(hasher, row)
        record = {
            "name": name or "dataset",
            "sha256": hasher.hexdigest(),
            "sample_count": len(rows),
            "row_count": count,
            "format": "records",
        }
    record["schema"] = infer_schema(rows)
    record["sample"] = rows
    return record


def fingerprint_paths(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    return [
        fingerprint_file(path)
        for path in sorted((Path(path) for path in paths), key=lambda p: str(p))
    ]


def fingerprint_directory(path: str | Path, *, chunk_size: int = 1024 * 1024) -> dict[str, Any]:
    from .fingerprint import fingerprint_directory as _fingerprint_directory

    return _fingerprint_directory(path, chunk_size=chunk_size)
