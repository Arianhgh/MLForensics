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
    if dataclasses.is_dataclass(value):
        return {k: as_dict(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): as_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [as_dict(v) for v in value]
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value
    if hasattr(value, "to_dict") and not isinstance(value, (str, bytes)):
        return value.to_dict()
    return value


def to_json(value: Any, *, indent: int | None = 2) -> str:
    return json.dumps(as_dict(value), indent=indent, sort_keys=True, default=str)
