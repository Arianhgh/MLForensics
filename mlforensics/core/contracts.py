"""Shared workflow contracts used by capture, replay, shrink, bisect, and CI."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .errors import UnresolvedEvaluation, ValidationError
from .models import SCHEMA_VERSION, FailureSignature, _metadata, _now, _record, _str

CHILD_RESULT_ENV = "MLFORENSICS_CHILD_RESULT"
CHILD_CAPSULE_ENV = "MLFORENSICS_CHILD_CAPSULE"
REPLAY_CAPSULE_ENV = "MLFORENSICS_REPLAY_CAPSULE"
REPLAY_STEP_ENV = "MLFORENSICS_REPLAY_STEP"


@dataclass(frozen=True)
class ExecutionSpec:
    """How to execute one forensic command or harness."""

    command: Sequence[str] | str
    working_directory: str | None = None
    seed: int | None = None
    timeout: float | None = None
    identity: Mapping[str, Any] = field(default_factory=dict)
    capsule: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.command, str):
            if not self.command.strip():
                raise ValidationError("command must be a non-empty string")
        else:
            converted = tuple(str(item) for item in self.command)
            if not converted or any(not item.strip() for item in converted):
                raise ValidationError("command must contain non-empty strings")
            object.__setattr__(self, "command", converted)
        object.__setattr__(self, "identity", _metadata(self.identity))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        command = self.command if isinstance(self.command, str) else list(self.command)
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "execution_spec",
            "command": command,
            "working_directory": self.working_directory,
            "seed": self.seed,
            "timeout": self.timeout,
            "identity": _record(self.identity),
            "capsule": self.capsule,
            "metadata": _record(self.metadata),
        }


@dataclass(frozen=True)
class ExecutionResult:
    """Structured child outcome, separate from stdout/stderr."""

    status: str
    returncode: int | None = None
    failure: FailureSignature | None = None
    error: str | None = None
    resources: Mapping[str, Any] = field(default_factory=dict)
    capsule: str | None = None
    unresolved: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _str(self.status, "status")
        object.__setattr__(self, "resources", _metadata(self.resources))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "execution_result",
            "status": self.status,
            "returncode": self.returncode,
            "failure": _record(self.failure),
            "error": self.error,
            "resources": _record(self.resources),
            "capsule": self.capsule,
            "unresolved": self.unresolved,
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionResult:
        if data.get("type") not in {None, "execution_result"}:
            raise ValidationError("not an execution_result record")
        failure = data.get("failure")
        return cls(
            status=str(data.get("status", "inconclusive")),
            returncode=data.get("returncode"),
            failure=FailureSignature.from_dict(failure) if isinstance(failure, Mapping) else None,
            error=data.get("error"),
            resources=data.get("resources", {})
            if isinstance(data.get("resources"), Mapping)
            else {},
            capsule=data.get("capsule"),
            unresolved=bool(data.get("unresolved", False)),
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata"), Mapping) else {},
        )


@dataclass(frozen=True)
class PredicateResult:
    preserved: bool
    status: str
    actual: FailureSignature | None = None
    reason: str = ""
    unresolved: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "predicate_result",
            "preserved": self.preserved,
            "status": self.status,
            "actual": _record(self.actual),
            "reason": self.reason,
            "unresolved": self.unresolved,
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PredicateResult:
        if data.get("type") not in {None, "predicate_result"}:
            raise ValidationError("not a predicate_result record")
        actual = data.get("actual")
        return cls(
            preserved=bool(data.get("preserved", False)),
            status=str(data.get("status", "inconclusive")),
            actual=FailureSignature.from_dict(actual) if isinstance(actual, Mapping) else None,
            reason=str(data.get("reason", "")),
            unresolved=bool(data.get("unresolved", False)),
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata"), Mapping) else {},
        )


def normalize_predicate_result(value: Any) -> PredicateResult:
    """Convert booleans, signatures, and typed results into PredicateResult."""
    if isinstance(value, PredicateResult):
        return value
    if isinstance(value, UnresolvedEvaluation):
        return PredicateResult(
            preserved=False, status="inconclusive", unresolved=True, reason=str(value)
        )
    if isinstance(value, FailureSignature):
        return PredicateResult(preserved=True, status="fail", actual=value)
    if isinstance(value, bool):
        return PredicateResult(
            preserved=value, status="fail" if value else "pass", unresolved=False
        )
    if value is None:
        return PredicateResult(preserved=False, status="pass")
    raise TypeError(
        "predicate must return bool, FailureSignature, PredicateResult, or UnresolvedEvaluation"
    )


def evaluate_predicate(predicate: Callable[..., Any], value: Any) -> PredicateResult:
    """Run a predicate and normalize its result, preserving unresolved evaluations."""
    try:
        return normalize_predicate_result(predicate(value))
    except UnresolvedEvaluation as exc:
        return normalize_predicate_result(exc)


class ReplayFixture(Protocol):
    """Resettable application objects used by replay and shrinking."""

    def construct(self) -> Any: ...

    def restore(self, checkpoint: Any) -> None: ...

    def execute(self, value: Any) -> Any: ...

    def close(self) -> None: ...


class FailurePredicate(Protocol):
    def __call__(self, value: Any) -> PredicateResult: ...


@dataclass(frozen=True)
class Budget:
    run_count: int | None = None
    wall_clock_s: float | None = None
    gpu_hours: float | None = None
    currency: float | None = None
    consumed: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "budget",
            "run_count": self.run_count,
            "wall_clock_s": self.wall_clock_s,
            "gpu_hours": self.gpu_hours,
            "currency": self.currency,
            "consumed": _record(self.consumed),
            "metadata": _record(self.metadata),
        }


@dataclass
class WorkflowSession:
    session_id: str
    kind: str
    created_at: str = field(default_factory=_now)
    cache_identity: Mapping[str, Any] = field(default_factory=dict)
    budget: Budget | None = None
    pending: list[str] = field(default_factory=list)
    consumed: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "workflow_session",
            "session_id": self.session_id,
            "kind": self.kind,
            "created_at": self.created_at,
            "cache_identity": _record(self.cache_identity),
            "budget": _record(self.budget),
            "pending": list(self.pending),
            "consumed": dict(self.consumed),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ValidationCheck:
    check_id: str
    kind: str
    target: str
    required: bool = True
    budget: Budget | None = None
    depends_on: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "validation_check",
            "check_id": self.check_id,
            "kind": self.kind,
            "target": self.target,
            "required": self.required,
            "budget": _record(self.budget),
            "depends_on": list(self.depends_on),
            "metadata": _record(self.metadata),
        }


@dataclass(frozen=True)
class ValidationPlan:
    checks: Sequence[ValidationCheck] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "validation_plan",
            "checks": [_record(item) for item in self.checks],
            "metadata": _record(self.metadata),
        }


@dataclass(frozen=True)
class PluginDescriptor:
    name: str
    kind: str
    protocol_version: str
    capabilities: Sequence[str] = field(default_factory=tuple)
    optional_dependencies: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "plugin_descriptor",
            "name": self.name,
            "kind": self.kind,
            "protocol_version": self.protocol_version,
            "capabilities": list(self.capabilities),
            "optional_dependencies": list(self.optional_dependencies),
            "metadata": _record(self.metadata),
        }


@dataclass(frozen=True)
class RunGroup:
    group_id: str
    run_ids: Sequence[str]
    pairing_key: str | None = None
    experimental_unit: str | None = None
    dataset: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "run_group",
            "group_id": self.group_id,
            "run_ids": list(self.run_ids),
            "pairing_key": self.pairing_key,
            "experimental_unit": self.experimental_unit,
            "dataset": self.dataset,
            "metadata": _record(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunGroup:
        if data.get("type") not in {None, "run_group"}:
            raise ValidationError("not a run_group record")
        run_ids = data.get("run_ids", ())
        if not isinstance(run_ids, Sequence) or isinstance(run_ids, (str, bytes)):
            raise ValidationError("run_ids must be a sequence of identifiers")
        return cls(
            group_id=str(data.get("group_id", "")),
            run_ids=tuple(str(item) for item in run_ids),
            pairing_key=data.get("pairing_key"),
            experimental_unit=data.get("experimental_unit"),
            dataset=data.get("dataset"),
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata"), Mapping) else {},
        )


def write_child_result(path: str | Path, result: ExecutionResult) -> Path:
    """Atomically write a structured child result separate from stdout."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.to_dict(), sort_keys=True, allow_nan=False, indent=2)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(payload + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def load_child_result(path: str | Path) -> ExecutionResult:
    """Load a child result file. Malformed envelopes are unresolved."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise UnresolvedEvaluation(f"malformed child result envelope: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise UnresolvedEvaluation("malformed child result envelope: expected an object")
    try:
        result = ExecutionResult.from_dict(raw)
    except (TypeError, ValueError, ValidationError) as exc:
        raise UnresolvedEvaluation(f"malformed child result envelope: {exc}") from exc
    return result


def parse_failure_envelope(value: Any) -> FailureSignature | None | UnresolvedEvaluation:
    """Parse a structured failure mapping. Malformed envelopes are unresolved."""
    if isinstance(value, FailureSignature):
        return value
    if not isinstance(value, Mapping):
        return None
    looks_like_envelope = (
        any(key in value for key in ("failure", "failure_signature", "signature"))
        or value.get("type") == "failure_signature"
    )
    for key in ("failure", "failure_signature", "signature"):
        candidate = value.get(key)
        if candidate is None:
            continue
        if isinstance(candidate, FailureSignature):
            return candidate
        if isinstance(candidate, Mapping):
            try:
                return FailureSignature.from_dict(candidate)
            except (TypeError, ValueError, ValidationError):
                return UnresolvedEvaluation("malformed failure envelope")
        return UnresolvedEvaluation("malformed failure envelope")
    if value.get("type") == "failure_signature":
        try:
            return FailureSignature.from_dict(value)
        except (TypeError, ValueError, ValidationError):
            return UnresolvedEvaluation("malformed failure envelope")
    nested = value.get("mlforensics")
    if isinstance(nested, Mapping):
        return parse_failure_envelope(nested)
    if looks_like_envelope:
        return UnresolvedEvaluation("malformed failure envelope")
    return None


CallableFactory = Callable[..., ReplayFixture]


__all__ = [
    "CHILD_CAPSULE_ENV",
    "CHILD_RESULT_ENV",
    "Budget",
    "ExecutionResult",
    "ExecutionSpec",
    "FailurePredicate",
    "PluginDescriptor",
    "PredicateResult",
    "REPLAY_CAPSULE_ENV",
    "REPLAY_STEP_ENV",
    "ReplayFixture",
    "RunGroup",
    "ValidationCheck",
    "ValidationPlan",
    "WorkflowSession",
    "evaluate_predicate",
    "load_child_result",
    "normalize_predicate_result",
    "parse_failure_envelope",
    "write_child_result",
]
