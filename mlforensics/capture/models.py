"""Data model and compatibility helpers for captured runs.

The capture package deliberately owns a small fallback model.  Projects that
provide a richer ``mlforensics.core.RunCapsule`` can pass that object to
``CaptureContext``; the context only relies on ordinary attributes and a few
optional recording methods.
"""

from __future__ import annotations

import json
import traceback
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    """Return a JSON-compatible representation without importing dependencies."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    if is_dataclass(value):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [json_safe(item) for item in value]
        return (
            items
            if not isinstance(value, (set, frozenset))
            else sorted(items, key=lambda item: repr(item))
        )
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return repr(value)
    return value


@dataclass
class RunCapsule:
    """Portable record containing the reproducibility context of one run."""

    schema_version: str = "1"
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    manifest: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    code: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, Any] = field(default_factory=dict)
    data_fingerprints: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    exceptions: list[dict[str, Any]] = field(default_factory=list)
    rng_snapshots: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep, JSON-safe mapping suitable for serialization."""

        return json_safe(asdict(self))

    as_dict = to_dict

    def record_metric(self, name: str, value: Any) -> None:
        self.metrics[str(name)] = json_safe(value)

    def record_exception(self, exception: BaseException, where: str | None = None) -> None:
        entry: dict[str, Any] = {
            "type": type(exception).__name__,
            "module": type(exception).__module__,
            "message": str(exception),
            "traceback": "".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            ),
        }
        if where:
            entry["where"] = where
        self.exceptions.append(entry)

    def record_rng_snapshot(self, name: str, snapshot: Any) -> None:
        self.rng_snapshots[str(name)] = json_safe(snapshot)

    def merge(self, section: str, values: Mapping[str, Any]) -> None:
        """Merge a section, useful when adapting an existing core capsule."""

        current = getattr(self, section)
        if isinstance(current, dict):
            current.update(json_safe(values))
        else:
            raise TypeError(f"section {section!r} is not a mapping")


def record_on_capsule(capsule: Any, section: str, value: Any) -> None:
    """Write a section to either the fallback model or a compatible core object."""

    safe_value = json_safe(value)
    if hasattr(capsule, section):
        current = getattr(capsule, section)
        if isinstance(current, dict) and isinstance(safe_value, Mapping):
            current.update(safe_value)
        else:
            try:
                setattr(capsule, section, safe_value)
            except Exception:
                pass
        return
    setter = getattr(capsule, f"set_{section}", None)
    if callable(setter):
        setter(safe_value)


def record_exception_on_capsule(
    capsule: Any, exception: BaseException, where: str | None = None
) -> None:
    recorder = getattr(capsule, "record_exception", None)
    if callable(recorder):
        try:
            recorder(exception, where=where)
        except TypeError:
            # Older core capsules may only accept the exception itself.
            recorder(exception)
        return
    record = {"type": type(exception).__name__, "message": str(exception)}
    if where:
        record["where"] = where
    exceptions = getattr(capsule, "exceptions", None)
    if isinstance(exceptions, list):
        exceptions.append(record)
