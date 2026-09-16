"""Small core-model substitutes used only when an optional core import fails."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _safe(value.to_dict())
    return value


@dataclass
class ArtifactRef:
    uri: str
    kind: str = "file"
    sha256: str | None = None
    size_bytes: int | None = None
    media_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _safe(asdict(self))


@dataclass
class DatasetRef:
    name: str
    uri: str | None = None
    fingerprint: str | None = None
    schema: dict[str, Any] = field(default_factory=dict)
    sample_count: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _safe(asdict(self))


@dataclass
class ModelRef:
    name: str
    framework: str | None = None
    architecture: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _safe(asdict(self))


@dataclass
class MetricSeries:
    name: str
    values: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    timestamps: list[str] = field(default_factory=list)
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(
        cls,
        name: str,
        value: float,
        *,
        step: int | None = None,
        timestamp: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MetricSeries:
        return cls(
            name=name,
            values=[float(value)],
            steps=[] if step is None else [step],
            timestamps=[] if timestamp is None else [timestamp],
            metadata=dict(metadata or {}),
        )

    def append(
        self, value: float, *, step: int | None = None, timestamp: str | None = None
    ) -> None:
        self.values.append(float(value))
        if step is not None or self.steps:
            self.steps.append(int(step if step is not None else len(self.values) - 1))
        if timestamp is not None or self.timestamps:
            self.timestamps.append(timestamp or utc_now())

    def to_dict(self) -> dict[str, Any]:
        return _safe(asdict(self))


@dataclass
class ResourceSeries:
    name: str
    values: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    units: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def append(self, value: float, *, step: int | None = None) -> None:
        self.values.append(float(value))
        if step is not None or self.steps:
            self.steps.append(int(step if step is not None else len(self.values) - 1))

    def to_dict(self) -> dict[str, Any]:
        return _safe(asdict(self))


@dataclass
class RNGState:
    states: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"states": _safe(self.states)}


@dataclass
class FailureSignature:
    kind: str
    exception_type: str | None = None
    normalized_message: str | None = None

    def matches(self, other: FailureSignature) -> bool:
        return (self.kind, self.exception_type, self.normalized_message) == (
            other.kind,
            other.exception_type,
            other.normalized_message,
        )

    def to_dict(self) -> dict[str, Any]:
        return _safe(asdict(self))


def exception_signature(exc: BaseException) -> FailureSignature:
    message = " ".join(str(exc).split())[:500]
    message = re.sub(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<uuid>", message, flags=re.I)
    message = re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", message)
    return FailureSignature(
        "exception", f"{type(exc).__module__}.{type(exc).__qualname__}", message
    )


@dataclass
class TraceEvent:
    name: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _safe(asdict(self))


@dataclass
class Run:
    run_id: str
    command: list[str] = field(default_factory=list)
    git: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    dependencies: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    failure: FailureSignature | None = None
    ended_at: str | None = None
    metrics: list[MetricSeries] = field(default_factory=list)
    resources: list[ResourceSeries] = field(default_factory=list)
    datasets: list[DatasetRef] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    model: ModelRef | None = None
    started_at: str = field(default_factory=utc_now)

    def metric(self, name: str, split: str | None = None) -> MetricSeries | None:
        return next(
            (item for item in self.metrics if item.name == name and item.split == split), None
        )

    def to_dict(self) -> dict[str, Any]:
        return _safe(asdict(self))


class RunCapsule:
    """Fallback shape accepted by the pre-core CaptureSession."""

    def __init__(self, *, run: Run, **kwargs: Any) -> None:
        self.run = run
        self.artifacts = tuple(kwargs.pop("artifacts", ()))
        self.payloads = dict(kwargs.pop("payloads", {}))
        self.code = kwargs.pop("code", {})
        self.environment = kwargs.pop("environment", {})
        self.hardware = kwargs.pop("hardware", {})
        self.dependencies = kwargs.pop("dependencies", {})
        self.randomness = kwargs.pop("randomness", {})
        self.training = kwargs.pop("training", {})
        self.system = kwargs.pop("system", {})
        self.data = kwargs.pop("data", {})
        self.model = kwargs.pop("model", None)
        self.failure = kwargs.pop("failure", None)
        self.extra = kwargs

    def to_dict(self) -> dict[str, Any]:
        return _safe(
            {
                "type": "run_capsule",
                "run": self.run.to_dict(),
                "code": self.code,
                "environment": self.environment,
                "hardware": self.hardware,
                "dependencies": self.dependencies,
                "randomness": self.randomness,
                "training": self.training,
                "system": self.system,
                "data": self.data,
                "model": self.model,
                "failure": self.failure,
                "extra": self.extra,
            }
        )

    @property
    def schema_version(self) -> str:
        return "1"

    @property
    def run_id(self) -> str:
        return self.run.run_id

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, default=repr).encode()
        ).hexdigest()

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        target = Path(path)
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), sort_keys=True, indent=2, default=repr), encoding="utf-8"
        )
        return target


class LocalArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(
        self,
        source: str | Path | bytes | bytearray,
        *,
        kind: str = "file",
        media_type: str | None = None,
    ) -> ArtifactRef:
        content = (
            bytes(source) if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
        )
        digest = hashlib.sha256(content).hexdigest()
        (self.root / digest).write_bytes(content)
        return ArtifactRef(
            uri=str(self.root / digest),
            kind=kind,
            sha256=digest,
            size_bytes=len(content),
            media_type=media_type,
        )
