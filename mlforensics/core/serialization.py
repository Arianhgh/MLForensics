"""Canonical JSON serialization for mlforensics records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from .errors import ValidationError
from .models import RECORD_TYPES, SCHEMA_VERSION


def _contract_types() -> dict[str, Any]:
    from .contracts import ExecutionResult, ExecutionSpec, PredicateResult, RunGroup
    from .execution import ExecutionRecord

    return {
        "execution_result": ExecutionResult,
        "execution_spec": ExecutionSpec,
        "predicate_result": PredicateResult,
        "run_group": RunGroup,
        "execution_record": ExecutionRecord,
    }


T = TypeVar("T")


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            # JSON object keys are strings.  Silently coercing two distinct
            # Python keys to the same string would make the serialized record
            # depend on insertion order and lose evidence.
            if not isinstance(key, str):
                raise TypeError("canonical JSON mappings must use string keys")
            if key in result:
                raise TypeError(f"duplicate canonical JSON key {key!r}")
            result[key] = _plain(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _reject_constant(value: str) -> Any:
    raise ValidationError(f"non-finite JSON constant {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def dumps(value: Any) -> str:
    """Return canonical JSON: UTF-8-safe, sorted, compact, and finite-only."""
    return json.dumps(
        _plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def dump_bytes(value: Any) -> bytes:
    return dumps(value).encode("utf-8")


def loads(data: str | bytes | bytearray, type_: type[T] | None = None) -> T | Any:
    """Decode JSON, optionally reconstructing a known record class."""
    try:
        raw = json.loads(
            data,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError("invalid JSON") from exc
    if not isinstance(raw, dict):
        if type_ is None:
            return raw
        raise ValidationError("a record JSON document must contain an object")
    if raw.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValidationError(
            f"unsupported schema_version {raw.get('schema_version')!r}; expected {SCHEMA_VERSION}"
        )
    if type_ is None:
        record_type = raw.get("type")
        type_ = RECORD_TYPES.get(record_type)
        if type_ is None:
            type_ = _contract_types().get(record_type)
        if type_ is None and record_type == "run_capsule":
            # Lazy import avoids the capsule -> serialization import cycle.
            from .capsule import RunCapsule

            type_ = RunCapsule
        if type_ is None:
            return raw
    factory = getattr(type_, "from_dict", None)
    if factory is None:
        raise TypeError(f"{type_!r} does not provide from_dict")
    return factory(raw)


def load(path: str | Path, type_: type[T] | None = None) -> T | Any:
    return loads(Path(path).read_bytes(), type_)


def validate(value: T) -> T:
    """Validate a public record and return it for convenient composition."""
    method = getattr(value, "validate", None)
    if method is not None:
        method()
    elif not hasattr(value, "to_dict"):
        raise TypeError("validate expects an mlforensics record")
    return value
