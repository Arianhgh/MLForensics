"""Stable, framework-neutral records used by :mod:`mlforensics`."""

from __future__ import annotations

import hashlib
import json
import math
import re
import traceback as traceback_module
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .errors import ValidationError

SCHEMA_VERSION = 1
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SECRET = re.compile(r"(?i)(password|token|secret|api[_-]?key)(\s*[=:]\s*)[^,;\s]+")
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", re.I)
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


# Public spelling used by capture and integration adapters.
utc_now = _now


def _check_schema(data: Mapping[str, Any]) -> None:
    version = data.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValidationError(f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION}")


def _str(value: Any, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValidationError(f"{name} must be a non-empty string")
    return value


def _metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValidationError("metadata must be a mapping")
    result = dict(value)
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValidationError("metadata must contain JSON-compatible finite values") from exc
    return result


def _record(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _record(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_record(item) for item in value]
    return value


def _require(data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ValidationError(f"missing field {key!r}")
    return data[key]


def _records(data: Sequence[Mapping[str, Any]] | None, cls: Any) -> tuple[Any, ...]:
    return tuple(
        cls.from_dict(item) if not isinstance(item, cls) else item for item in (data or ())
    )


class EvidenceState(str, Enum):
    """The common evidence outcome vocabulary used by analyzers and CI."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class EvidenceStatus:
    """Explain an evidence-backed decision without collapsing missing data into pass.

    ``required_evidence`` and ``observed_evidence`` are names, not values. The
    actual observations remain in the owning record so this small object can
    safely travel through JSON reports and policy decisions.
    """

    status: str
    reason: str = ""
    required_evidence: Sequence[str] = field(default_factory=tuple)
    observed_evidence: Sequence[str] = field(default_factory=tuple)
    policy: Mapping[str, Any] = field(default_factory=dict)
    remediation: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = (
            self.status.value
            if isinstance(self.status, EvidenceState)
            else str(self.status).casefold()
        )
        if status not in {item.value for item in EvidenceState}:
            raise ValidationError("evidence status must be one of: pass, fail, warn, inconclusive")
        if not isinstance(self.reason, str):
            raise ValidationError("evidence reason must be a string")
        for values, label in (
            (self.required_evidence, "required_evidence"),
            (self.observed_evidence, "observed_evidence"),
        ):
            converted = tuple(str(item) for item in values)
            if any(not item.strip() for item in converted):
                raise ValidationError(f"{label} must contain non-empty strings")
            object.__setattr__(self, label, converted)
        if self.remediation is not None and not isinstance(self.remediation, str):
            raise ValidationError("remediation must be a string or None")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "policy", _metadata(self.policy))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def state(self) -> str:
        """Alias used by callers that call the outcome a state."""
        return self.status

    @property
    def conclusive(self) -> bool:
        return self.status in {EvidenceState.PASS.value, EvidenceState.FAIL.value}

    @property
    def passed(self) -> bool:
        return self.status == EvidenceState.PASS.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "evidence_status",
            "status": self.status,
            "reason": self.reason,
            "required_evidence": list(self.required_evidence),
            "observed_evidence": list(self.observed_evidence),
            "policy": _record(self.policy),
            "remediation": self.remediation,
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceStatus:
        _check_schema(data)
        return cls(
            status=_require(data, "status"),
            reason=data.get("reason", ""),
            required_evidence=data.get("required_evidence", ()),
            observed_evidence=data.get("observed_evidence", ()),
            policy=data.get("policy", {}),
            remediation=data.get("remediation"),
            metadata=data.get("metadata", {}),
        )


_OBSERVATION_STATES = {"finite", "nan", "pos_inf", "neg_inf", "missing", "failed"}


@dataclass(frozen=True)
class Observation:
    """A scalar observation that can preserve failure states safely.

    Strict metric/resource series intentionally contain finite numbers only.
    Capture paths use this record for NaN, infinity, missing, and failed
    observations so forensic evidence is retained without emitting invalid
    JSON numbers.
    """

    name: str
    value: float | None = None
    state: str = "finite"
    identity: Any = None
    step: int | float | None = None
    timestamp: str | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _str(self.name, "name")
        state = str(self.state).casefold()
        if state not in _OBSERVATION_STATES:
            raise ValidationError(f"unknown observation state {self.state!r}")
        value = self.value
        if state == "finite":
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValidationError("finite observations require a finite numeric value")
            value = float(value)
        else:
            # Never retain a NaN/Inf float in a strict record. A caller may
            # pass the original value for convenience; the typed state is the
            # canonical representation.
            if value is not None:
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    numeric = None
                if numeric is not None and math.isfinite(numeric):
                    raise ValidationError(
                        f"{state} observations cannot contain a finite numeric value"
                    )
            value = None
        if self.step is not None and (
            isinstance(self.step, bool)
            or not isinstance(self.step, (int, float))
            or not math.isfinite(float(self.step))
        ):
            raise ValidationError("observation step must be a finite number or None")
        if self.timestamp is not None and not isinstance(self.timestamp, str):
            raise ValidationError("observation timestamp must be a string or None")
        if self.error is not None and not isinstance(self.error, str):
            raise ValidationError("observation error must be a string or None")
        try:
            json.dumps(self.identity, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError("observation identity must be JSON-compatible") from exc
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_value(
        cls,
        name: str,
        value: Any,
        *,
        identity: Any = None,
        step: int | float | None = None,
        timestamp: str | None = None,
        error: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Observation:
        if value is None:
            state = "missing"
            numeric = None
        else:
            if isinstance(value, bool):
                state = "failed"
                numeric = None
                error = error or "value is not numeric: bool"
                return cls(
                    name=name,
                    value=numeric,
                    state=state,
                    identity=identity,
                    step=step,
                    timestamp=timestamp,
                    error=error,
                    metadata=metadata or {},
                )
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                state = "failed"
                numeric = None
                error = error or f"value is not numeric: {type(value).__name__}"
            else:
                if math.isnan(numeric):
                    state = "nan"
                    numeric = None
                elif math.isinf(numeric):
                    state = "pos_inf" if numeric > 0 else "neg_inf"
                    numeric = None
                else:
                    state = "finite"
        return cls(
            name=name,
            value=numeric,
            state=state,
            identity=identity,
            step=step,
            timestamp=timestamp,
            error=error,
            metadata=metadata or {},
        )

    @property
    def finite(self) -> bool:
        return self.state == "finite"

    @property
    def numeric_value(self) -> float | None:
        return self.value if self.finite else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "observation",
            "name": self.name,
            "value": self.value,
            "state": self.state,
            "identity": _record(self.identity),
            "step": self.step,
            "timestamp": self.timestamp,
            "error": self.error,
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Observation:
        _check_schema(data)
        return cls(
            name=_require(data, "name"),
            value=data.get("value"),
            state=data.get("state", "finite"),
            identity=data.get("identity"),
            step=data.get("step"),
            timestamp=data.get("timestamp"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class ArtifactRef:
    name: str
    uri: str = ""
    sha256: str = ""
    size_bytes: int | None = None
    media_type: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _str(self.name, "name")
        if not isinstance(self.uri, str):
            raise ValidationError("uri must be a string")
        if self.sha256 and not _HEX64.fullmatch(self.sha256):
            raise ValidationError("sha256 must be a lowercase 64-character hex digest")
        if self.size_bytes is not None and (
            not isinstance(self.size_bytes, int) or self.size_bytes < 0
        ):
            raise ValidationError("size_bytes must be a non-negative integer or None")
        if self.media_type is not None:
            _str(self.media_type, "media_type")
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @classmethod
    def from_bytes(
        cls,
        name: str,
        content: bytes,
        *,
        media_type: str | None = None,
        uri: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRef:
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        return cls(
            name=name,
            uri=uri,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            media_type=media_type,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "artifact_ref",
            "name": self.name,
            "uri": self.uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "media_type": self.media_type,
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactRef:
        _check_schema(data)
        return cls(
            name=_require(data, "name"),
            uri=data.get("uri", ""),
            sha256=data.get("sha256", ""),
            size_bytes=data.get("size_bytes"),
            media_type=data.get("media_type"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class DatasetRef:
    name: str
    artifact: ArtifactRef | None = None
    split: str | None = None
    format: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _str(self.name, "name")
        if self.artifact is not None and not isinstance(self.artifact, ArtifactRef):
            raise ValidationError("artifact must be an ArtifactRef or None")
        for value, label in ((self.split, "split"), (self.format, "format")):
            if value is not None:
                _str(value, label)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "dataset_ref",
            "name": self.name,
            "artifact": _record(self.artifact),
            "split": self.split,
            "format": self.format,
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetRef:
        _check_schema(data)
        artifact = data.get("artifact")
        return cls(
            name=_require(data, "name"),
            artifact=ArtifactRef.from_dict(artifact) if artifact else None,
            split=data.get("split"),
            format=data.get("format"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class ModelRef:
    name: str
    artifact: ArtifactRef | None = None
    framework: str | None = None
    version: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _str(self.name, "name")
        if self.artifact is not None and not isinstance(self.artifact, ArtifactRef):
            raise ValidationError("artifact must be an ArtifactRef or None")
        for value, label in ((self.framework, "framework"), (self.version, "version")):
            if value is not None:
                _str(value, label)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "model_ref",
            "name": self.name,
            "artifact": _record(self.artifact),
            "framework": self.framework,
            "version": self.version,
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelRef:
        _check_schema(data)
        artifact = data.get("artifact")
        return cls(
            name=_require(data, "name"),
            artifact=ArtifactRef.from_dict(artifact) if artifact else None,
            framework=data.get("framework"),
            version=data.get("version"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class MetricSeries:
    name: str
    values: Sequence[float]
    steps: Sequence[int | float] = field(default_factory=tuple)
    timestamps: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    identities: Sequence[Any] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _str(self.name, "name")
        raw_values = tuple(self.values)
        try:
            values = tuple(float(value) for value in raw_values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("metric values must be finite numbers") from exc
        if any(
            isinstance(raw, bool) or not math.isfinite(value)
            for raw, value in zip(raw_values, values)
        ):
            raise ValidationError("metric values must be finite numbers")
        steps = tuple(self.steps)
        if steps and len(steps) != len(values):
            raise ValidationError("steps must have the same length as values")
        for value in steps:
            try:
                finite = not isinstance(value, bool) and math.isfinite(float(value))
            except (TypeError, ValueError, OverflowError):
                finite = False
            if not finite:
                raise ValidationError("steps must be finite numbers")
        timestamps = tuple(self.timestamps)
        if timestamps and len(timestamps) != len(values):
            raise ValidationError("timestamps must have the same length as values")
        if any(not isinstance(v, str) or not v for v in timestamps):
            raise ValidationError("timestamps must be non-empty strings")
        identities = tuple(self.identities)
        if identities and len(identities) != len(values):
            raise ValidationError("identities must have the same length as values")
        try:
            json.dumps(identities, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError("identities must be JSON-compatible") from exc
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "identities", identities)

    @classmethod
    def from_value(
        cls,
        name: str,
        value: float,
        *,
        step: int | float | None = None,
        timestamp: str | None = None,
        identity: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> MetricSeries:
        """Construct a one-observation metric series."""
        return cls(
            name=name,
            values=(value,),
            steps=(step,) if step is not None else (),
            timestamps=(timestamp,) if timestamp is not None else (),
            metadata=metadata or {},
            identities=(identity if identity is not None else step,)
            if identity is not None or step is not None
            else (),
        )

    @classmethod
    def from_values(
        cls,
        name: str,
        values: Sequence[float],
        *,
        steps: Sequence[int | float] = (),
        timestamps: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        identities: Sequence[Any] = (),
    ) -> MetricSeries:
        return cls(
            name=name,
            values=values,
            steps=steps,
            timestamps=timestamps,
            metadata=metadata or {},
            identities=identities,
        )

    @property
    def observation_ids(self) -> tuple[Any, ...]:
        """Stable pairing keys, if supplied by the producer."""
        if self.identities:
            return tuple(self.identities)
        for key in ("observation_ids", "sample_ids", "seed_ids", "seeds", "case_ids", "ids"):
            value = self.metadata.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) == len(self.values):
                    return tuple(value)
            if (
                len(self.values) == 1
                and value is not None
                and not isinstance(value, (Mapping, list, tuple, set))
            ):
                return (value,)
        if "seed" in self.metadata and len(self.values) == 1:
            return (self.metadata["seed"],)
        if self.steps:
            return tuple(self.steps)
        return tuple(range(len(self.values)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "metric_series",
            "name": self.name,
            "values": list(self.values),
            "steps": list(self.steps),
            "timestamps": list(self.timestamps),
            "metadata": _record(self.metadata),
            "identities": _record(self.identities),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricSeries:
        _check_schema(data)
        return cls(
            name=_require(data, "name"),
            values=_require(data, "values"),
            steps=data.get("steps", ()),
            timestamps=data.get("timestamps", ()),
            metadata=data.get("metadata", {}),
            identities=data.get("identities", ()),
        )


@dataclass(frozen=True)
class ResourceSeries:
    name: str
    values: Sequence[float]
    steps: Sequence[int | float] = field(default_factory=tuple)
    units: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    identities: Sequence[Any] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _str(self.name, "name")
        raw_values = tuple(self.values)
        try:
            values = tuple(float(value) for value in raw_values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError("resource values must be finite numbers") from exc
        if any(
            isinstance(raw, bool) or not math.isfinite(value)
            for raw, value in zip(raw_values, values)
        ):
            raise ValidationError("resource values must be finite numbers")
        steps = tuple(self.steps)
        if steps and len(steps) != len(values):
            raise ValidationError("steps must have the same length as values")
        for value in steps:
            try:
                finite = not isinstance(value, bool) and math.isfinite(float(value))
            except (TypeError, ValueError, OverflowError):
                finite = False
            if not finite:
                raise ValidationError("steps must be finite numbers")
        if self.units is not None:
            _str(self.units, "units")
        identities = tuple(self.identities)
        if identities and len(identities) != len(values):
            raise ValidationError("identities must have the same length as values")
        try:
            json.dumps(identities, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError("identities must be JSON-compatible") from exc
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "identities", identities)

    @classmethod
    def from_value(
        cls,
        name: str,
        value: float,
        *,
        step: int | float | None = None,
        units: str | None = None,
        identity: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ResourceSeries:
        return cls(
            name=name,
            values=(value,),
            steps=(step,) if step is not None else (),
            units=units,
            metadata=metadata or {},
            identities=(identity if identity is not None else step,)
            if identity is not None or step is not None
            else (),
        )

    @property
    def observation_ids(self) -> tuple[Any, ...]:
        if self.identities:
            return tuple(self.identities)
        for key in ("observation_ids", "sample_ids", "seed_ids", "seeds", "case_ids", "ids"):
            value = self.metadata.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) == len(self.values):
                    return tuple(value)
            if (
                len(self.values) == 1
                and value is not None
                and not isinstance(value, (Mapping, list, tuple, set))
            ):
                return (value,)
        if "seed" in self.metadata and len(self.values) == 1:
            return (self.metadata["seed"],)
        if self.steps:
            return tuple(self.steps)
        return tuple(range(len(self.values)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "resource_series",
            "name": self.name,
            "values": list(self.values),
            "steps": list(self.steps),
            "units": self.units,
            "metadata": _record(self.metadata),
            "identities": _record(self.identities),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResourceSeries:
        _check_schema(data)
        return cls(
            name=_require(data, "name"),
            values=_require(data, "values"),
            steps=data.get("steps", ()),
            units=data.get("units"),
            metadata=data.get("metadata", {}),
            identities=data.get("identities", ()),
        )


@dataclass(frozen=True)
class RNGState:
    python: str | None = None
    numpy: str | None = None
    frameworks: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value, label in ((self.python, "python"), (self.numpy, "numpy")):
            if value is not None:
                _str(value, label, allow_empty=True)
        if not isinstance(self.frameworks, Mapping) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in self.frameworks.items()
        ):
            raise ValidationError("frameworks must map strings to strings")
        object.__setattr__(self, "frameworks", dict(self.frameworks))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "rng_state",
            "python": self.python,
            "numpy": self.numpy,
            "frameworks": _record(self.frameworks),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RNGState:
        _check_schema(data)
        return cls(
            python=data.get("python"),
            numpy=data.get("numpy"),
            frameworks=data.get("frameworks", {}),
        )


@dataclass(frozen=True)
class StateSnapshot:
    """A named, codec-described piece of state used by replay."""

    name: str
    codec: str
    value: Any = None
    artifact: ArtifactRef | None = None
    dtype: str | None = None
    shape: Sequence[int] = field(default_factory=tuple)
    device: str | None = None
    captured_at: str = field(default_factory=_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _str(self.name, "name")
        _str(self.codec, "codec")
        if self.artifact is not None and not isinstance(self.artifact, ArtifactRef):
            raise ValidationError("artifact must be an ArtifactRef or None")
        shape = tuple(self.shape)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape):
            raise ValidationError("shape must contain non-negative integers")
        if self.dtype is not None:
            _str(self.dtype, "dtype")
        if self.device is not None:
            _str(self.device, "device")
        _str(self.captured_at, "captured_at")
        try:
            json.dumps(self.value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError("snapshot value must be JSON-compatible") from exc
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "state_snapshot",
            "name": self.name,
            "codec": self.codec,
            "value": _record(self.value),
            "artifact": _record(self.artifact),
            "dtype": self.dtype,
            "shape": list(self.shape),
            "device": self.device,
            "captured_at": self.captured_at,
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StateSnapshot:
        _check_schema(data)
        artifact = data.get("artifact")
        return cls(
            name=_require(data, "name"),
            codec=_require(data, "codec"),
            value=data.get("value"),
            artifact=ArtifactRef.from_dict(artifact) if artifact else None,
            dtype=data.get("dtype"),
            shape=data.get("shape", ()),
            device=data.get("device"),
            captured_at=data.get("captured_at", _now()),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class CheckpointRef:
    """Reference to a replay checkpoint and the state it contains."""

    checkpoint_id: str
    step: int | float
    state: Sequence[StateSnapshot] = field(default_factory=tuple)
    batch: Any = None
    epoch: int | None = None
    sampler_position: int | None = None
    created_at: str = field(default_factory=_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    rng_state: RNGState | None = None

    def __post_init__(self) -> None:
        _str(self.checkpoint_id, "checkpoint_id")
        if (
            isinstance(self.step, bool)
            or not isinstance(self.step, (int, float))
            or not math.isfinite(float(self.step))
        ):
            raise ValidationError("checkpoint step must be a finite number")
        state = tuple(self.state)
        if any(not isinstance(item, StateSnapshot) for item in state):
            raise ValidationError("checkpoint state must contain StateSnapshot records")
        if self.rng_state is not None and not isinstance(self.rng_state, RNGState):
            raise ValidationError("checkpoint rng_state must be RNGState or None")
        for value, label in ((self.epoch, "epoch"), (self.sampler_position, "sampler_position")):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValidationError(f"{label} must be a non-negative integer or None")
        try:
            json.dumps(self.batch, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError("checkpoint batch must be JSON-compatible") from exc
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "checkpoint_ref",
            "checkpoint_id": self.checkpoint_id,
            "step": self.step,
            "state": [_record(item) for item in self.state],
            "batch": _record(self.batch),
            "epoch": self.epoch,
            "sampler_position": self.sampler_position,
            "created_at": self.created_at,
            "metadata": _record(self.metadata),
            "rng_state": _record(self.rng_state),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckpointRef:
        _check_schema(data)
        return cls(
            checkpoint_id=_require(data, "checkpoint_id"),
            step=_require(data, "step"),
            state=_records(data.get("state"), StateSnapshot),
            batch=data.get("batch"),
            epoch=data.get("epoch"),
            sampler_position=data.get("sampler_position"),
            created_at=data.get("created_at", _now()),
            metadata=data.get("metadata", {}),
            rng_state=RNGState.from_dict(data["rng_state"]) if data.get("rng_state") else None,
        )


@dataclass(frozen=True)
class ReplayPlan:
    """Portable description of how an incident should be reconstructed."""

    input: Any = None
    state: Sequence[StateSnapshot] = field(default_factory=tuple)
    entrypoint: str | None = None
    expected_failure: FailureSignature | None = None
    determinism: str = "best_effort"
    restore_order: Sequence[str] = field(default_factory=tuple)
    checkpoints: Sequence[CheckpointRef] = field(default_factory=tuple)
    limitations: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = tuple(self.state)
        checkpoints = tuple(self.checkpoints)
        if any(not isinstance(item, StateSnapshot) for item in state):
            raise ValidationError("replay state must contain StateSnapshot records")
        if any(not isinstance(item, CheckpointRef) for item in checkpoints):
            raise ValidationError("replay checkpoints must contain CheckpointRef records")
        if self.expected_failure is not None and not isinstance(
            self.expected_failure, FailureSignature
        ):
            raise ValidationError("expected_failure must be a FailureSignature or None")
        _str(self.determinism, "determinism")
        if self.entrypoint is not None:
            _str(self.entrypoint, "entrypoint")
        for values, label in (
            (self.restore_order, "restore_order"),
            (self.limitations, "limitations"),
        ):
            converted = tuple(str(item) for item in values)
            if any(not item.strip() for item in converted):
                raise ValidationError(f"{label} must contain non-empty strings")
            object.__setattr__(self, label, converted)
        try:
            json.dumps(self.input, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError("replay input must be JSON-compatible") from exc
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "checkpoints", checkpoints)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "replay_plan",
            "input": _record(self.input),
            "state": [_record(item) for item in self.state],
            "entrypoint": self.entrypoint,
            "expected_failure": _record(self.expected_failure),
            "determinism": self.determinism,
            "restore_order": list(self.restore_order),
            "checkpoints": [_record(item) for item in self.checkpoints],
            "limitations": list(self.limitations),
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReplayPlan:
        _check_schema(data)
        expected = data.get("expected_failure")
        return cls(
            input=data.get("input"),
            state=_records(data.get("state"), StateSnapshot),
            entrypoint=data.get("entrypoint"),
            expected_failure=FailureSignature.from_dict(expected) if expected else None,
            determinism=data.get("determinism", "best_effort"),
            restore_order=data.get("restore_order", ()),
            checkpoints=_records(data.get("checkpoints"), CheckpointRef),
            limitations=data.get("limitations", ()),
            metadata=data.get("metadata", {}),
        )


def normalize_failure_message(message: str) -> str:
    """Remove volatile values from an exception message for grouping."""
    normalized = _SECRET.sub(r"\1=<redacted>", message)
    normalized = _UUID.sub("<uuid>", normalized)
    normalized = _NUMBER.sub("<number>", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


_BUILTIN_EXCEPTION_MODULES = frozenset({"", "builtins", "exceptions", "__builtin__"})


def compatible_exception_types(left: str, right: str) -> bool:
    """Match ordinary Python exception names with or without a builtins prefix."""
    if left == right:
        return True
    left_short = left.rsplit(".", 1)[-1]
    right_short = right.rsplit(".", 1)[-1]
    if left_short != right_short:
        return False
    left_mod = left.rsplit(".", 1)[0] if "." in left else ""
    right_mod = right.rsplit(".", 1)[0] if "." in right else ""
    return (
        left_mod in _BUILTIN_EXCEPTION_MODULES
        or right_mod in _BUILTIN_EXCEPTION_MODULES
        or left_mod == right_mod
    )


def _compatible_exception_chains(left: Sequence[str], right: Sequence[str]) -> bool:
    if len(left) != len(right):
        return False
    return all(compatible_exception_types(first, second) for first, second in zip(left, right))


@dataclass(frozen=True)
class FailureSignature:
    error_type: str = ""
    message: str = ""
    normalized_message: str = ""
    traceback_hash: str = ""
    exception_chain: Sequence[str] = field(default_factory=tuple)
    kind: str = "exception"
    phase: str | None = None
    module: str | None = None
    operation: str | None = None
    top_frame: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = str(self.kind).casefold()
        if not kind or not re.fullmatch(r"[a-z][a-z0-9_.-]*", kind):
            raise ValidationError("failure kind must be a non-empty stable identifier")
        # Structured signatures do not necessarily have an exception type.
        if kind == "exception":
            _str(self.error_type, "error_type")
        elif self.error_type and not isinstance(self.error_type, str):
            raise ValidationError("error_type must be a string")
        if not isinstance(self.message, str) or not isinstance(self.normalized_message, str):
            raise ValidationError("failure messages must be strings")
        trace_hash = self.traceback_hash
        if trace_hash and not _HEX64.fullmatch(trace_hash):
            raise ValidationError("traceback_hash must be a lowercase 64-character hex digest")
        chain = tuple(self.exception_chain)
        if any(not isinstance(item, str) or not item for item in chain):
            raise ValidationError("exception_chain must contain non-empty strings")
        for value, label in (
            (self.phase, "phase"),
            (self.module, "module"),
            (self.operation, "operation"),
            (self.top_frame, "top_frame"),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValidationError(f"{label} must be a non-empty string or None")
        if not isinstance(self.details, Mapping):
            raise ValidationError("failure details must be a mapping")
        try:
            json.dumps(self.details, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                "failure details must contain JSON-compatible finite values"
            ) from exc
        if not trace_hash:
            canonical = json.dumps(
                {
                    "kind": kind,
                    "error_type": self.error_type,
                    "normalized_message": self.normalized_message,
                    "exception_chain": list(chain),
                    "phase": self.phase,
                    "module": self.module,
                    "operation": self.operation,
                    "top_frame": self.top_frame,
                    "details": self.details,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            trace_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "traceback_hash", trace_hash)
        object.__setattr__(self, "exception_chain", chain)
        object.__setattr__(self, "details", dict(self.details))

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        traceback_text: str | None = None,
        top_frame: str | None = None,
        phase: str | None = None,
        module: str | None = None,
        operation: str | None = None,
        **details: Any,
    ) -> FailureSignature:
        if not isinstance(exc, BaseException):
            raise TypeError("exc must be an exception")
        trace = (
            traceback_text
            if traceback_text is not None
            else "".join(traceback_module.format_exception(exc))
        )
        chain: list[str] = []
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(f"{type(current).__module__}.{type(current).__qualname__}")
            current = current.__cause__ or (
                None if current.__suppress_context__ else current.__context__
            )
        error_type = f"{type(exc).__module__}.{type(exc).__qualname__}"
        message = str(exc)
        if top_frame is None:
            extracted = traceback_module.extract_tb(exc.__traceback__)
            if extracted:
                frame = extracted[-1]
                top_frame = f"{frame.filename}:{frame.lineno}"
        return cls(
            error_type=error_type,
            message=message,
            normalized_message=normalize_failure_message(message),
            traceback_hash=hashlib.sha256(trace.encode("utf-8", "replace")).hexdigest(),
            exception_chain=tuple(chain),
            kind="exception",
            phase=phase,
            module=module,
            operation=operation,
            top_frame=top_frame,
            details=details,
        )

    @classmethod
    def structured(
        cls,
        kind: str,
        *,
        message: str = "",
        normalized_message: str | None = None,
        phase: str | None = None,
        module: str | None = None,
        operation: str | None = None,
        top_frame: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> FailureSignature:
        """Construct a non-exception failure predicate record."""
        return cls(
            error_type="",
            message=message,
            normalized_message=normalize_failure_message(message)
            if normalized_message is None
            else normalized_message,
            traceback_hash="",
            kind=kind,
            phase=phase,
            module=module,
            operation=operation,
            top_frame=top_frame,
            details=details or {},
        )

    @classmethod
    def from_non_finite(
        cls,
        *,
        operation: str | None = None,
        module: str | None = None,
        phase: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> FailureSignature:
        return cls.structured(
            "non_finite_tensor",
            message="non-finite tensor value",
            operation=operation,
            module=module,
            phase=phase,
            details=details,
        )

    @classmethod
    def from_metric_regression(
        cls,
        metric: str,
        threshold: float,
        *,
        details: Mapping[str, Any] | None = None,
    ) -> FailureSignature:
        return cls.structured(
            "metric_regression",
            message=f"metric {metric} regressed beyond threshold {threshold}",
            details={"metric": metric, "threshold": threshold, **dict(details or {})},
        )

    @classmethod
    def from_parity_mismatch(
        cls, *, path: str | None = None, details: Mapping[str, Any] | None = None
    ) -> FailureSignature:
        return cls.structured(
            "parity_mismatch",
            message="backend outputs differ",
            operation=path,
            details=details,
        )

    @classmethod
    def from_timeout(
        cls,
        *,
        seconds: float | None = None,
        phase: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> FailureSignature:
        payload = dict(details or {})
        if seconds is not None:
            payload.setdefault("seconds", float(seconds))
        return cls.structured(
            "timeout",
            message="operation timed out",
            phase=phase,
            details=payload,
        )

    @classmethod
    def from_oom(
        cls,
        *,
        device: str | None = None,
        phase: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> FailureSignature:
        payload = dict(details or {})
        if device is not None:
            payload.setdefault("device", device)
        return cls.structured(
            "out_of_memory", message="out of memory", phase=phase, details=payload
        )

    @classmethod
    def from_hang(
        cls,
        *,
        phase: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> FailureSignature:
        return cls.structured("hang", message="operation hung", phase=phase, details=details)

    @classmethod
    def from_user_predicate(
        cls,
        name: str,
        *,
        message: str = "user-defined failure predicate matched",
        details: Mapping[str, Any] | None = None,
    ) -> FailureSignature:
        payload = {"predicate": name, **dict(details or {})}
        return cls.structured("user_predicate", message=message, operation=name, details=payload)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FailureSignature:
        _check_schema(data)
        kind = data.get("kind", "exception")
        # Older capsules may have omitted normalized_message. Recompute it so
        # loading old evidence does not weaken matching semantics.
        message = data.get("message", "")
        normalized = data.get("normalized_message")
        return cls(
            error_type=data.get("error_type", ""),
            message=message,
            normalized_message=(
                normalize_failure_message(message) if normalized is None else normalized
            ),
            traceback_hash=data.get("traceback_hash", ""),
            exception_chain=data.get("exception_chain", ()),
            kind=kind,
            phase=data.get("phase"),
            module=data.get("module"),
            operation=data.get("operation"),
            top_frame=data.get("top_frame"),
            details=data.get("details", {}),
        )

    @property
    def exception_type(self) -> str:
        """Compatibility alias used by the older capture model."""
        return self.error_type

    def grouping_key(self) -> tuple[Any, ...]:
        stable_details = json.dumps(self.details, sort_keys=True, separators=(",", ":"))
        return (
            self.kind,
            self.error_type,
            self.normalized_message,
            tuple(self.exception_chain),
            self.phase,
            self.module,
            self.operation,
            self.top_frame,
            stable_details,
        )

    def matches(self, other: FailureSignature | BaseException) -> bool:
        """Return whether ``other`` represents the same reproducible failure.

        Traceback hashes intentionally do not participate in matching because
        a replay normally runs from a different call site.  The stable error
        type, normalized message, and exception chain are the useful identity
        for replay, shrinking, and bisect predicates.
        """
        candidate = (
            other if isinstance(other, FailureSignature) else FailureSignature.from_exception(other)
        )
        if self.kind != candidate.kind:
            return False

        def frame_key(value: str | None) -> str | None:
            if value is None:
                return None
            # Replay normally moves the line number but should retain the
            # source file/module identity.  Callers that need an exact frame
            # can still place it in ``details`` and compare that explicitly.
            return re.sub(r":\d+(?::\d+)?$", "", value)

        def context_matches() -> bool:
            return (
                (self.phase is None or self.phase == candidate.phase)
                and (self.module is None or self.module == candidate.module)
                and (self.operation is None or self.operation == candidate.operation)
                and (
                    self.top_frame is None
                    or frame_key(self.top_frame) == frame_key(candidate.top_frame)
                )
            )

        if self.kind == "exception":
            return (
                compatible_exception_types(self.error_type, candidate.error_type)
                and self.normalized_message == candidate.normalized_message
                and (
                    not self.exception_chain
                    or tuple(self.exception_chain) == tuple(candidate.exception_chain)
                    or _compatible_exception_chains(self.exception_chain, candidate.exception_chain)
                )
                and context_matches()
            )

        # Structured predicates identify the operation/module when present;
        # traceback locations and volatile details are never required.
        return (
            self.normalized_message == candidate.normalized_message
            and (self.operation is None or self.operation == candidate.operation)
            and (self.module is None or self.module == candidate.module)
            and (self.phase is None or self.phase == candidate.phase)
            and (
                self.top_frame is None
                or frame_key(self.top_frame) == frame_key(candidate.top_frame)
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "failure_signature",
            "error_type": self.error_type,
            "message": self.message,
            "normalized_message": self.normalized_message,
            "traceback_hash": self.traceback_hash,
            "exception_chain": list(self.exception_chain),
            "kind": self.kind,
            "phase": self.phase,
            "module": self.module,
            "operation": self.operation,
            "top_frame": self.top_frame,
            "details": _record(self.details),
        }


@dataclass(frozen=True)
class TraceEvent:
    kind: str
    message: str = ""
    timestamp: str = field(default_factory=_now)
    step: int | float | None = None
    data: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _str(self.kind, "kind")
        if not isinstance(self.message, str) or not isinstance(self.timestamp, str):
            raise ValidationError("message and timestamp must be strings")
        if self.step is not None and (
            isinstance(self.step, bool)
            or not isinstance(self.step, (int, float))
            or not math.isfinite(self.step)
        ):
            raise ValidationError("step must be a finite number or None")
        object.__setattr__(self, "data", _metadata(self.data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "trace_event",
            "kind": self.kind,
            "message": self.message,
            "timestamp": self.timestamp,
            "step": self.step,
            "data": _record(self.data),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TraceEvent:
        _check_schema(data)
        return cls(
            kind=_require(data, "kind"),
            message=data.get("message", ""),
            timestamp=data.get("timestamp", _now()),
            step=data.get("step"),
            data=data.get("data", {}),
        )


@dataclass(frozen=True)
class LineageNode:
    node_id: str
    kind: str
    label: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _str(self.node_id, "node_id")
        _str(self.kind, "kind")
        if not isinstance(self.label, str):
            raise ValidationError("label must be a string")
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "lineage_node",
            "node_id": self.node_id,
            "kind": self.kind,
            "label": self.label,
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LineageNode:
        _check_schema(data)
        return cls(
            node_id=_require(data, "node_id"),
            kind=_require(data, "kind"),
            label=data.get("label", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class LineageEdge:
    source: str
    target: str
    relation: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _str(self.source, "source")
        _str(self.target, "target")
        _str(self.relation, "relation")
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "lineage_edge",
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LineageEdge:
        _check_schema(data)
        return cls(
            source=_require(data, "source"),
            target=_require(data, "target"),
            relation=_require(data, "relation"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Run:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    status: str = "running"
    started_at: str = field(default_factory=_now)
    ended_at: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    datasets: Sequence[DatasetRef] = field(default_factory=tuple)
    models: Sequence[ModelRef] = field(default_factory=tuple)
    metrics: Sequence[MetricSeries] = field(default_factory=tuple)
    resources: Sequence[ResourceSeries] = field(default_factory=tuple)
    events: Sequence[TraceEvent] = field(default_factory=tuple)
    rng_state: RNGState | None = None
    failure_signature: FailureSignature | None = None
    lineage_nodes: Sequence[LineageNode] = field(default_factory=tuple)
    lineage_edges: Sequence[LineageEdge] = field(default_factory=tuple)
    observations: Sequence[Observation] = field(default_factory=tuple)
    replay_plan: ReplayPlan | None = None

    def __post_init__(self) -> None:
        _str(self.run_id, "run_id")
        if (
            not isinstance(self.name, str)
            or not isinstance(self.status, str)
            or not isinstance(self.started_at, str)
        ):
            raise ValidationError("name, status, and started_at must be strings")
        if self.ended_at is not None and not isinstance(self.ended_at, str):
            raise ValidationError("ended_at must be a string or None")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        for values, cls, label in (
            (self.datasets, DatasetRef, "datasets"),
            (self.models, ModelRef, "models"),
            (self.metrics, MetricSeries, "metrics"),
            (self.resources, ResourceSeries, "resources"),
            (self.events, TraceEvent, "events"),
            (self.lineage_nodes, LineageNode, "lineage_nodes"),
            (self.lineage_edges, LineageEdge, "lineage_edges"),
            (self.observations, Observation, "observations"),
        ):
            converted = tuple(values)
            if any(not isinstance(item, cls) for item in converted):
                raise ValidationError(f"{label} contains an invalid record")
            object.__setattr__(self, label, converted)
        if self.rng_state is not None and not isinstance(self.rng_state, RNGState):
            raise ValidationError("rng_state must be RNGState or None")
        if self.failure_signature is not None and not isinstance(
            self.failure_signature, FailureSignature
        ):
            raise ValidationError("failure_signature must be FailureSignature or None")
        if self.replay_plan is not None and not isinstance(self.replay_plan, ReplayPlan):
            raise ValidationError("replay_plan must be ReplayPlan or None")

    def validate(self) -> Run:
        self.__post_init__()
        return self

    def metric(self, name: str) -> MetricSeries | None:
        """Return the first metric series with ``name``, if present."""
        return next((item for item in self.metrics if item.name == name), None)

    def model(self, name: str | None = None) -> ModelRef | None:
        """Return a recorded model, optionally selected by name."""
        return next((item for item in self.models if name is None or item.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "run",
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "metadata": _record(self.metadata),
            "datasets": _record(self.datasets),
            "models": _record(self.models),
            "metrics": _record(self.metrics),
            "resources": _record(self.resources),
            "events": _record(self.events),
            "rng_state": _record(self.rng_state),
            "failure_signature": _record(self.failure_signature),
            "lineage_nodes": _record(self.lineage_nodes),
            "lineage_edges": _record(self.lineage_edges),
            "observations": _record(self.observations),
            "replay_plan": _record(self.replay_plan),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Run:
        _check_schema(data)
        rng = data.get("rng_state")
        failure = data.get("failure_signature")
        return cls(
            run_id=_require(data, "run_id"),
            name=data.get("name", ""),
            status=data.get("status", "running"),
            started_at=_require(data, "started_at"),
            ended_at=data.get("ended_at"),
            metadata=data.get("metadata", {}),
            datasets=_records(data.get("datasets"), DatasetRef),
            models=_records(data.get("models"), ModelRef),
            metrics=_records(data.get("metrics"), MetricSeries),
            resources=_records(data.get("resources"), ResourceSeries),
            events=_records(data.get("events"), TraceEvent),
            rng_state=RNGState.from_dict(rng) if rng else None,
            failure_signature=FailureSignature.from_dict(failure) if failure else None,
            lineage_nodes=_records(data.get("lineage_nodes"), LineageNode),
            lineage_edges=_records(data.get("lineage_edges"), LineageEdge),
            observations=_records(data.get("observations"), Observation),
            replay_plan=ReplayPlan.from_dict(data["replay_plan"])
            if data.get("replay_plan")
            else None,
        )


@dataclass(frozen=True)
class Comparison:
    left_run_id: str
    right_run_id: str
    metric_deltas: Mapping[str, float] = field(default_factory=dict)
    resource_deltas: Mapping[str, float] = field(default_factory=dict)
    regressions: Sequence[str] = field(default_factory=tuple)
    notes: Sequence[str] = field(default_factory=tuple)
    evidence: Mapping[str, Any] = field(default_factory=dict)
    status: str = "pass"
    decisions: Mapping[str, EvidenceStatus] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _str(self.left_run_id, "left_run_id")
        _str(self.right_run_id, "right_run_id")
        for mapping, label in (
            (self.metric_deltas, "metric_deltas"),
            (self.resource_deltas, "resource_deltas"),
        ):
            if not isinstance(mapping, Mapping) or any(
                not isinstance(k, str)
                or isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                for k, v in mapping.items()
            ):
                raise ValidationError(f"{label} must map names to finite numbers")
        object.__setattr__(self, "metric_deltas", dict(self.metric_deltas))
        object.__setattr__(self, "resource_deltas", dict(self.resource_deltas))
        object.__setattr__(self, "evidence", _metadata(self.evidence))
        status = (
            self.status.value
            if isinstance(self.status, EvidenceState)
            else str(self.status).casefold()
        )
        if status not in {item.value for item in EvidenceState}:
            raise ValidationError(
                "comparison status must be one of: pass, fail, warn, inconclusive"
            )
        decisions: dict[str, EvidenceStatus] = {}
        for key, value in self.decisions.items():
            if not isinstance(key, str) or not isinstance(value, EvidenceStatus):
                raise ValidationError("decisions must map names to EvidenceStatus records")
            decisions[key] = value
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "decisions", decisions)
        for values, label in ((self.regressions, "regressions"), (self.notes, "notes")):
            if any(not isinstance(item, str) for item in values):
                raise ValidationError(f"{label} must contain strings")
            object.__setattr__(self, label, tuple(values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "comparison",
            "left_run_id": self.left_run_id,
            "right_run_id": self.right_run_id,
            "metric_deltas": _record(self.metric_deltas),
            "resource_deltas": _record(self.resource_deltas),
            "regressions": list(self.regressions),
            "notes": list(self.notes),
            "evidence": _record(self.evidence),
            "status": self.status,
            "decisions": {key: value.to_dict() for key, value in self.decisions.items()},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Comparison:
        _check_schema(data)
        return cls(
            left_run_id=_require(data, "left_run_id"),
            right_run_id=_require(data, "right_run_id"),
            metric_deltas=data.get("metric_deltas", {}),
            resource_deltas=data.get("resource_deltas", {}),
            regressions=data.get("regressions", ()),
            notes=data.get("notes", ()),
            evidence=data.get("evidence", {}),
            status=data.get("status", "pass"),
            decisions={
                str(key): EvidenceStatus.from_dict(value)
                for key, value in data.get("decisions", {}).items()
            },
        )


@dataclass(frozen=True)
class Counterexample:
    description: str
    inputs: Any = None
    expected: Any = None
    actual: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _str(self.description, "description")
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        try:
            json.dumps((self.inputs, self.expected, self.actual), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValidationError("counterexample values must be JSON-compatible") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "counterexample",
            "description": self.description,
            "inputs": _record(self.inputs),
            "expected": _record(self.expected),
            "actual": _record(self.actual),
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Counterexample:
        _check_schema(data)
        return cls(
            description=_require(data, "description"),
            inputs=data.get("inputs"),
            expected=data.get("expected"),
            actual=data.get("actual"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class Incident:
    incident_id: str
    category: str
    severity: str = "error"
    title: str = ""
    description: str = ""
    run_id: str | None = None
    counterexamples: Sequence[Counterexample] = field(default_factory=tuple)
    evidence: Sequence[ArtifactRef] = field(default_factory=tuple)
    created_at: str = field(default_factory=_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _str(self.incident_id, "incident_id")
        _str(self.category, "category")
        _str(self.severity, "severity")
        if (
            not isinstance(self.title, str)
            or not isinstance(self.description, str)
            or not isinstance(self.created_at, str)
        ):
            raise ValidationError("title, description, and created_at must be strings")
        if self.run_id is not None:
            _str(self.run_id, "run_id")
        for values, cls, label in (
            (self.counterexamples, Counterexample, "counterexamples"),
            (self.evidence, ArtifactRef, "evidence"),
        ):
            converted = tuple(values)
            if any(not isinstance(item, cls) for item in converted):
                raise ValidationError(f"{label} contains an invalid record")
            object.__setattr__(self, label, converted)
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "incident",
            "incident_id": self.incident_id,
            "category": self.category,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "run_id": self.run_id,
            "counterexamples": _record(self.counterexamples),
            "evidence": _record(self.evidence),
            "created_at": self.created_at,
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Incident:
        _check_schema(data)
        return cls(
            incident_id=_require(data, "incident_id"),
            category=_require(data, "category"),
            severity=data.get("severity", "error"),
            title=data.get("title", ""),
            description=data.get("description", ""),
            run_id=data.get("run_id"),
            counterexamples=_records(data.get("counterexamples"), Counterexample),
            evidence=_records(data.get("evidence"), ArtifactRef),
            created_at=data.get("created_at", _now()),
            metadata=data.get("metadata", {}),
        )


RECORD_TYPES: dict[str, Any] = {
    "artifact_ref": ArtifactRef,
    "dataset_ref": DatasetRef,
    "model_ref": ModelRef,
    "metric_series": MetricSeries,
    "resource_series": ResourceSeries,
    "rng_state": RNGState,
    "evidence_status": EvidenceStatus,
    "observation": Observation,
    "state_snapshot": StateSnapshot,
    "checkpoint_ref": CheckpointRef,
    "replay_plan": ReplayPlan,
    "failure_signature": FailureSignature,
    "trace_event": TraceEvent,
    "lineage_node": LineageNode,
    "lineage_edge": LineageEdge,
    "run": Run,
    "comparison": Comparison,
    "counterexample": Counterexample,
    "incident": Incident,
}
