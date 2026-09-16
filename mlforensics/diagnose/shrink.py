"""Hierarchical delta debugging and ML-shaped shrinking strategies.

The predicate passed to this module uses the conventional delta-debugging
meaning: it returns :data:`True` while the failure is still present. Every
entry point verifies that precondition before attempting to minimize a value.
Predicate exceptions are never considered evidence that a failure was
preserved.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from numbers import Number
from typing import Any

from ..core import (
    ArtifactRef,
    Counterexample,
    FailureSignature,
    Run,
    RunCapsule,
    default_state_codecs,
    utc_now,
)
from ..core.contracts import PredicateResult, evaluate_predicate

Predicate = Callable[[Any], bool]


class InitialPredicateError(ValueError):
    """Raised when a shrink request is not given a failing input."""


@dataclass(slots=True)
class ShrinkResult:
    value: Any
    original_size: int | float | None
    final_size: int | float | None
    evaluations: int
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def predicate_calls(self) -> int:
        return self.evaluations

    @property
    def minimal(self) -> Any:
        """Compatibility alias for the minimized value."""
        return self.value

    @property
    def counterexample(self) -> Counterexample:
        return Counterexample(
            "minimized failing input",
            inputs=self.value,
            metadata={
                "original_size": self.original_size,
                "final_size": self.final_size,
                **self.metadata,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "original_size": self.original_size,
            "final_size": self.final_size,
            "evaluations": self.evaluations,
            "history": list(self.history),
            "metadata": dict(self.metadata),
        }


ShrinkReport = ShrinkResult


def _evaluate(predicate: Predicate, value: Any, *, initial: bool = False) -> PredicateResult:
    try:
        result = evaluate_predicate(predicate, value)
    except Exception as exc:
        if initial:
            raise InitialPredicateError(
                "cannot shrink: the predicate raised for the initial value: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return PredicateResult(
            preserved=False, status="error", reason=f"{type(exc).__name__}: {exc}"
        )
    if result.unresolved and initial:
        raise InitialPredicateError(
            "cannot shrink: the initial predicate evaluation is unresolved"
            + (f": {result.reason}" if result.reason else "")
        )
    return result


def _accepted(result: PredicateResult) -> bool:
    return result.preserved and not result.unresolved


def _minimality_verified(final: PredicateResult, history: Sequence[Mapping[str, Any]]) -> bool:
    if not _accepted(final) or final.unresolved:
        return False
    return not any(bool(item.get("unresolved")) for item in history)


def _require_failing(predicate: Predicate, value: Any) -> None:
    if not _accepted(_evaluate(predicate, value, initial=True)):
        raise InitialPredicateError(
            "cannot shrink: the initial value does not satisfy the failure predicate"
        )


def _sequence_factory(prototype: Sequence[Any]) -> Callable[[Iterable[Any]], Any]:
    if isinstance(prototype, tuple):
        return tuple
    return list


def ddmin(
    items: Sequence[Any],
    predicate: Predicate,
    *,
    min_chunk: int = 1,
    max_evaluations: int | None = None,
) -> ShrinkResult:
    """Minimize a sequence while preserving its concrete list/tuple type.

    ``min_chunk`` is the minimum number of items in the result (despite its
    historical name). Set it to zero when an empty counterexample is useful.
    """

    if isinstance(items, (str, bytes, bytearray)):
        raise TypeError("ddmin expects an item sequence, not text; use shrink(..., kind='tokens')")
    if min_chunk < 0:
        raise ValueError("min_chunk must be non-negative")
    if max_evaluations is not None and max_evaluations < 1:
        raise ValueError("max_evaluations must be at least 1")

    factory = _sequence_factory(items)
    original = factory(copy.deepcopy(list(items)))
    _require_failing(predicate, original)
    current = list(original)
    evaluations = 1
    history: list[dict[str, Any]] = []
    granularity = 2

    while len(current) > min_chunk:
        chunk_size = max(1, math.ceil(len(current) / granularity))
        reduced = False
        for start in range(0, len(current), chunk_size):
            candidate_items = current[:start] + current[start + chunk_size :]
            if len(candidate_items) < min_chunk:
                continue
            if max_evaluations is not None and evaluations >= max_evaluations:
                break
            candidate = factory(candidate_items)
            evaluations += 1
            accepted = _evaluate(predicate, candidate)
            history.append(
                {
                    "operation": "remove_chunk",
                    "start": start,
                    "count": min(chunk_size, len(current) - start),
                    "accepted": _accepted(accepted),
                    "unresolved": accepted.unresolved,
                    "size": len(candidate_items),
                }
            )
            if _accepted(accepted):
                current = candidate_items
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if max_evaluations is not None and evaluations >= max_evaluations:
            break
        if not reduced:
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)

    minimized = factory(current)
    verified = _evaluate(predicate, minimized)
    evaluations += 1
    if verified.unresolved or not _accepted(verified):
        if not verified.unresolved:
            raise RuntimeError("shrink predicate was not stable when the final value was verified")
    return ShrinkResult(
        minimized,
        len(original),
        len(current),
        evaluations,
        history,
        {
            "initial_failing": True,
            "verified": _minimality_verified(verified, history),
            "strategy": "ddmin",
        },
    )


def shrink_rows(rows: Sequence[Any], predicate: Predicate, **kwargs: Any) -> ShrinkResult:
    return ddmin(rows, predicate, **kwargs)


def shrink_columns(
    row_data: Sequence[Mapping[str, Any]],
    predicate: Predicate,
    *,
    columns: Iterable[str] | None = None,
    max_evaluations: int | None = None,
) -> ShrinkResult:
    """Remove columns with delta debugging while preserving rows and key order."""

    original = [dict(row) for row in copy.deepcopy(row_data)]
    _require_failing(predicate, original)
    all_names = list(dict.fromkeys(key for row in original for key in row))
    selected = list(dict.fromkeys(columns if columns is not None else all_names))
    unknown = [name for name in selected if name not in all_names]
    if unknown:
        raise ValueError(f"unknown columns: {', '.join(map(str, unknown))}")
    fixed = [name for name in all_names if name not in selected]

    def project(kept: Sequence[str]) -> list[dict[str, Any]]:
        allowed = set(fixed) | set(kept)
        return [{key: value for key, value in row.items() if key in allowed} for row in original]

    reduced = ddmin(
        selected,
        lambda kept: _evaluate(predicate, project(kept)),
        min_chunk=0,
        max_evaluations=max_evaluations,
    )
    value = project(reduced.value)
    final_names = list(dict.fromkeys(key for row in value for key in row))
    history = []
    for item in reduced.history:
        entry = dict(item)
        entry["operation"] = "remove_column_chunk"
        history.append(entry)
    return ShrinkResult(
        value,
        len(all_names),
        len(final_names),
        reduced.evaluations + 1,
        history,
        {
            **reduced.metadata,
            "strategy": "columns",
            "removed_columns": [name for name in all_names if name not in final_names],
        },
    )


def shrink_tokens(tokens: Sequence[Any], predicate: Predicate, **kwargs: Any) -> ShrinkResult:
    return ddmin(tokens, predicate, **kwargs)


def _tensor_to_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        try:
            return value.detach().cpu().tolist()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return copy.deepcopy(value)


def _flatten(value: Any) -> list[Any]:
    result: list[Any] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            result.append(item)

    visit(value)
    return result


def _shape(value: Any) -> tuple[int, ...]:
    declared = getattr(value, "shape", None)
    if declared is not None:
        try:
            return tuple(int(item) for item in declared)
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        children = [_shape(item) for item in value]
        if all(child == children[0] for child in children):
            return (len(value), *children[0])
        return (len(value),)
    return ()


def _zero_like(value: Any) -> Any:
    if isinstance(value, bool):
        return False
    try:
        return type(value)(0)
    except Exception:
        return 0


def _restore_like(prototype: Any, flat: Sequence[Any]) -> Any:
    iterator = iter(flat)

    def rebuild(template: Any) -> Any:
        if isinstance(template, list):
            return [rebuild(item) for item in template]
        if isinstance(template, tuple):
            return tuple(rebuild(item) for item in template)
        return next(iterator)

    if hasattr(prototype, "copy") and hasattr(prototype, "flat"):
        candidate = prototype.copy()
        candidate.flat[:] = list(flat)
        return candidate
    if hasattr(prototype, "detach") and hasattr(prototype, "clone"):
        candidate = prototype.detach().clone()
        view = candidate.reshape(-1)
        for index, item in enumerate(flat):
            view[index] = item
        return candidate
    return rebuild(prototype)


def _is_zero(value: Any) -> bool:
    try:
        return bool(value == 0)
    except Exception:
        return False


def shrink_tensor(
    value: Any,
    predicate: Predicate,
    *,
    max_elements: int = 100_000,
    max_evaluations: int | None = None,
) -> ShrinkResult:
    """Zero tensor regions while preserving dtype, container type, and shape."""

    if max_elements < 0:
        raise ValueError("max_elements must be non-negative")
    _require_failing(predicate, value)
    raw = _tensor_to_list(value)
    flat = _flatten(raw)
    original_shape = _shape(value)
    if len(flat) > max_elements:
        return ShrinkResult(
            value,
            len(flat),
            len(flat),
            1,
            metadata={
                "initial_failing": True,
                "verified": True,
                "skipped": "tensor exceeds max_elements",
                "shape": original_shape,
            },
        )

    active = [index for index, item in enumerate(flat) if not _is_zero(item)]

    def build(kept: Sequence[int]) -> Any:
        kept_set = set(kept)
        candidate_flat = [
            item if index in kept_set else _zero_like(item) for index, item in enumerate(flat)
        ]
        return _restore_like(value, candidate_flat)

    reduced = ddmin(
        active,
        lambda kept: _evaluate(predicate, build(kept)),
        min_chunk=0,
        max_evaluations=max_evaluations,
    )
    result = build(reduced.value)
    history = []
    for item in reduced.history:
        entry = dict(item)
        entry["operation"] = "zero_region"
        history.append(entry)
    return ShrinkResult(
        result,
        len(flat),
        len(reduced.value),
        reduced.evaluations + 1,
        history,
        {
            **reduced.metadata,
            "strategy": "tensor_zeroing",
            "representation": type(value).__name__,
            "shape": original_shape,
            "shape_preserved": _shape(result) == original_shape,
        },
    )


def _complexity(value: Any) -> tuple[float, float, int]:
    if isinstance(value, Mapping):
        child = [_complexity(item) for item in value.values()]
        return (1 + sum(item[0] for item in child), sum(item[1] for item in child), len(value))
    if isinstance(value, (list, tuple)):
        child = [_complexity(item) for item in value]
        return (1 + sum(item[0] for item in child), sum(item[1] for item in child), len(value))
    if isinstance(value, str):
        return (1, float(len(value)), len(value))
    if isinstance(value, Number):
        try:
            magnitude = abs(float(value))
            return (1, magnitude if math.isfinite(magnitude) else math.inf, 0)
        except (TypeError, ValueError, OverflowError):
            pass
    return (1, float(len(repr(value))), 0)


def _structured_candidates(value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        keys = list(value)
        if keys:
            yield "clear_mapping", {}
        for key in keys:
            yield f"remove_key:{key}", {item: child for item, child in value.items() if item != key}
        for key in keys:
            for operation, child in _structured_candidates(value[key]):
                candidate = dict(value)
                candidate[key] = child
                yield f"key:{key}/{operation}", candidate
        return
    if isinstance(value, (list, tuple)):
        factory = tuple if isinstance(value, tuple) else list
        if value:
            yield "clear_sequence", factory()
        for index in range(len(value)):
            yield f"remove_item:{index}", factory((*value[:index], *value[index + 1 :]))
        for index, item in enumerate(value):
            for operation, child in _structured_candidates(item):
                replacement = list(value)
                replacement[index] = child
                yield f"item:{index}/{operation}", factory(replacement)
        return
    if isinstance(value, str):
        if value:
            yield "clear_text", ""
            yield "halve_text", value[: len(value) // 2]
        return
    if isinstance(value, bool):
        if value:
            yield "false", False
        return
    if isinstance(value, Number):
        candidates = [0, value / 2]
        for candidate in candidates:
            if candidate != value:
                yield "reduce_number", candidate


def shrink_structure(
    value: Any,
    predicate: Predicate,
    *,
    max_evaluations: int = 256,
) -> ShrinkResult:
    """Recursively simplify JSON-like mappings and sequences."""

    if max_evaluations < 1:
        raise ValueError("max_evaluations must be at least 1")
    current = copy.deepcopy(value)
    _require_failing(predicate, current)
    evaluations = 1
    history: list[dict[str, Any]] = []
    while evaluations < max_evaluations:
        changed = False
        current_complexity = _complexity(current)
        for operation, candidate in _structured_candidates(current):
            if _complexity(candidate) >= current_complexity:
                continue
            evaluations += 1
            accepted = _evaluate(predicate, candidate)
            history.append(
                {
                    "operation": operation,
                    "accepted": _accepted(accepted),
                    "unresolved": accepted.unresolved,
                    "size": _complexity(candidate)[0],
                }
            )
            if _accepted(accepted):
                current = candidate
                changed = True
                break
            if evaluations >= max_evaluations:
                break
        if not changed:
            break
    verified = _evaluate(predicate, current)
    evaluations += 1
    if verified.unresolved or not _accepted(verified):
        if not verified.unresolved:
            raise RuntimeError("shrink predicate was not stable when the final value was verified")
    return ShrinkResult(
        current,
        _complexity(value)[0],
        _complexity(current)[0],
        evaluations,
        history,
        {
            "initial_failing": True,
            "verified": _minimality_verified(verified, history),
            "strategy": "structured",
        },
    )


def _shrink_text(value: str, predicate: Predicate, **kwargs: Any) -> ShrinkResult:
    tokens = re.findall(r"\S+", value)
    result = ddmin(tokens, lambda items: _evaluate(predicate, " ".join(items)), **kwargs)
    result.value = " ".join(result.value)
    result.metadata.update({"source_type": "str", "strategy": "tokens"})
    return result


def shrink(value: Any, predicate: Predicate, *, kind: str = "auto", **kwargs: Any) -> ShrinkResult:
    """Select a deterministic shrink strategy for common ML inputs."""

    valid_kinds = {"auto", "rows", "columns", "tokens", "sequence", "tensor", "structured"}
    if kind not in valid_kinds:
        raise ValueError(f"unknown shrink kind: {kind}")
    if isinstance(value, str) and kind in {"auto", "tokens", "sequence"}:
        return _shrink_text(value, predicate, **kwargs)
    if kind == "columns" or (
        kind == "auto"
        and isinstance(value, list)
        and value
        and all(isinstance(item, Mapping) for item in value)
    ):
        return shrink_columns(value, predicate, **kwargs)
    if kind == "structured" or (kind == "auto" and isinstance(value, Mapping)):
        return shrink_structure(value, predicate, **kwargs)
    if kind in {"rows", "tokens", "sequence"}:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{kind} shrinking requires a list or tuple")
        return ddmin(value, predicate, **kwargs)
    if kind == "auto" and isinstance(value, (list, tuple)):
        return ddmin(value, predicate, **kwargs)
    return shrink_tensor(value, predicate, **kwargs)


def _failure_from_result(value: Any) -> FailureSignature | None:
    """Read only the structured failure envelope emitted by a replay child."""
    if isinstance(value, FailureSignature):
        return value
    if isinstance(value, Mapping):
        for key in ("failure", "failure_signature", "signature"):
            candidate = value.get(key)
            if isinstance(candidate, FailureSignature):
                return candidate
            if isinstance(candidate, Mapping):
                try:
                    return FailureSignature.from_dict(candidate)
                except (TypeError, ValueError):
                    return None
        nested = value.get("mlforensics")
        if isinstance(nested, Mapping):
            return _failure_from_result(nested)
    return None


def capsule_replay_input(capsule: RunCapsule) -> Any:
    """Return the canonical replay input, including an embedded artifact."""
    from .replay import _artifact_value  # local import keeps the core dependency-free

    replay = capsule.evidence.get("replay", {})
    if isinstance(replay, Mapping) and "input" in replay:
        return _artifact_value(capsule, replay["input"])
    plan = capsule.run.replay_plan
    if plan is not None and plan.input is not None:
        return _artifact_value(capsule, plan.input)
    raise ValueError("capsule does not contain evidence.replay.input")


def persist_shrink_capsule(
    source: RunCapsule,
    result: ShrinkResult,
    *,
    failure: FailureSignature | None = None,
    output: str | Any | None = None,
) -> RunCapsule:
    """Create a child capsule containing a minimized replay input and history."""
    source_digest = source.digest
    child_id = (
        f"{source.run.run_id}-shrink-{hashlib.sha256(source_digest.encode()).hexdigest()[:8]}"
    )
    artifacts = list(source.artifacts)
    payloads = dict(source.payloads)
    encoded_artifact = False

    def encode_input(value: Any, name: str = "counterexample") -> Any:
        """Encode a shrunk tensor/array as an embedded artifact marker."""
        nonlocal artifacts, encoded_artifact, payloads
        if isinstance(value, Mapping):
            return {str(key): encode_input(item, f"{name}-{key}") for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode_input(item, f"{name}-{index}") for index, item in enumerate(value)]
        try:
            encoded = default_state_codecs.encode(value)
        except (ImportError, RuntimeError, TypeError, ValueError):
            encoded = None
        if encoded is None:
            try:
                json.dumps(value, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "the minimized counterexample is not JSON-compatible and has no state codec"
                ) from exc
            return value
        if encoded.payload is None:
            return encoded.value
        encoded_artifact = True
        ref = ArtifactRef.from_bytes(
            f"shrink/{name}.bin",
            encoded.payload,
            media_type="application/octet-stream",
            metadata={
                "role": "shrink_counterexample",
                "codec": encoded.codec,
                **dict(encoded.metadata),
            },
        )
        artifacts = [item for item in artifacts if item.sha256 != ref.sha256]
        artifacts.append(ref)
        payloads[ref.sha256] = encoded.payload
        return {"artifact_sha256": ref.sha256, "codec": encoded.codec, **dict(encoded.metadata)}

    encoded_value = encode_input(result.value)
    result_metadata = dict(result.metadata)
    try:
        json.dumps(result_metadata, allow_nan=False)
    except (TypeError, ValueError):
        result_metadata = {"unserializable_metadata": repr(result_metadata)}
    result_record = {
        "value": encoded_value,
        "original_size": result.original_size,
        "final_size": result.final_size,
        "evaluations": result.evaluations,
        "history": result.history,
        "metadata": result_metadata,
    }
    result_failure = failure or source.run.failure_signature
    replay_plan = source.run.replay_plan
    if replay_plan is not None:
        replay_plan = replace(replay_plan, input=encoded_value, expected_failure=result_failure)
    metadata = {
        **dict(source.run.metadata),
        "parent_run_id": source.run.run_id,
        "parent_capsule_digest": source_digest,
        "shrink": {
            "original_size": result.original_size,
            "final_size": result.final_size,
            "evaluations": result.evaluations,
            "history": result.history,
            "metadata": result_metadata,
        },
        "counterexample": {
            "description": "minimized failing input",
            "input_codec": "embedded" if encoded_artifact else "json",
        },
    }
    child_run = Run(
        run_id=child_id,
        name=f"{source.run.name} (shrunk)",
        status="failed" if result_failure is not None else source.run.status,
        started_at=source.run.started_at,
        ended_at=utc_now(),
        metadata=metadata,
        datasets=source.run.datasets,
        models=source.run.models,
        metrics=source.run.metrics,
        resources=source.run.resources,
        events=source.run.events,
        rng_state=source.run.rng_state,
        failure_signature=result_failure,
        lineage_nodes=source.run.lineage_nodes,
        lineage_edges=source.run.lineage_edges,
        observations=source.run.observations,
        replay_plan=replay_plan,
    )
    replay_source = (
        dict(source.evidence.get("replay", {}))
        if isinstance(source.evidence.get("replay", {}), Mapping)
        else {}
    )
    checkpoints = replay_source.get("checkpoints", [])
    sanitized_checkpoints = []
    target_step = replay_source.get("step")
    if isinstance(checkpoints, Sequence) and not isinstance(checkpoints, (str, bytes)):
        if target_step is None:
            for item in checkpoints:
                if isinstance(item, Mapping) and item.get("step") is not None:
                    target_step = item.get("step")
        for item in checkpoints:
            if not isinstance(item, Mapping):
                sanitized_checkpoints.append(item)
                continue
            copied = dict(item)
            copied["historical_batch"] = copied.get("historical_batch", copied.get("batch"))
            if target_step is not None and copied.get("step") == target_step:
                copied["batch"] = encoded_value
            sanitized_checkpoints.append(copied)
        replay_source["checkpoints"] = sanitized_checkpoints
    replay_source["input"] = encoded_value
    replay_source["execution_input"] = encoded_value
    replay_source["execution_input_step"] = target_step
    replay_source["parent_capsule_digest"] = source_digest
    child = RunCapsule(
        child_run,
        tuple(artifacts),
        payloads,
        evidence={
            **dict(source.evidence),
            "replay": replay_source,
            "shrink": {
                "parent_run_id": source.run.run_id,
                "parent_capsule_digest": source_digest,
                "history": result.history,
                "result": result_record,
                "counterexample": Counterexample(
                    "minimized failing input",
                    inputs=encoded_value,
                    metadata={"parent_capsule_digest": source_digest},
                ).to_dict(),
                "failure_signature": result_failure.to_dict() if result_failure else None,
            },
        },
    )
    if output is not None:
        child.save(output, overwrite=True)
    return child


def shrink_capsule(
    capsule: RunCapsule | str,
    predicate: Predicate,
    *,
    kind: str = "auto",
    output: str | Any | None = None,
    failure: FailureSignature | None = None,
    **kwargs: Any,
) -> tuple[ShrinkResult, RunCapsule]:
    """Shrink ``evidence.replay.input`` and persist a linked child capsule."""
    source = RunCapsule.load(capsule) if isinstance(capsule, (str, bytes)) else capsule
    if not isinstance(source, RunCapsule):
        raise TypeError("capsule must be a RunCapsule or capsule path")
    value = capsule_replay_input(source)
    result = shrink(value, predicate, kind=kind, **kwargs)
    child = persist_shrink_capsule(source, result, failure=failure, output=output)
    return result, child
