"""Stable structured report conversion for diagnosis results."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


class Report:
    def to_dict(self) -> dict[str, Any]:
        return as_dict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return to_json(self, indent=indent)


@dataclass(frozen=True)
class DiagnosisReport(Report):
    kind: str
    status: str
    summary: str = ""
    evidence: Mapping[str, Any] = field(default_factory=dict)
    recommendations: tuple[str, ...] = ()


def as_dict(value: Any) -> Any:
    # Prefer a record's explicit wire representation. ``dataclasses.asdict``
    # would erase the type/schema markers carried by core evidence records.
    if hasattr(value, "to_dict") and not isinstance(value, (str, bytes, Report)):
        return as_dict(value.to_dict())
    if dataclasses.is_dataclass(value):
        return {item.name: as_dict(getattr(value, item.name)) for item in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): as_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [as_dict(v) for v in value]
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value
    return value


def to_json(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(as_dict(value), indent=indent, sort_keys=True, default=str)
