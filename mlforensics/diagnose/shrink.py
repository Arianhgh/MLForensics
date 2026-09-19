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
import time
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
    UnresolvedEvaluation,
    default_state_codecs,
    digest_state_tree,
    normalize_predicate_result,
    utc_now,
)
from ..core.contracts import PredicateResult

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
        return int(self.metadata.get("predicate_calls", self.evaluations))

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
            "resume": self.resume_state(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ShrinkResult:
        """Restore a JSON result for an interrupted shrink session.

        The value is intentionally not decoded here.  Capsule callers may use
        an artifact marker and should resolve it through the capsule handoff
        before resuming; plain JSON values are ready to use as-is.
        """
        if not isinstance(data, Mapping):
            raise TypeError("shrink result must be a mapping")
        return cls(
            value=data.get("value"),
            original_size=data.get("original_size"),
            final_size=data.get("final_size"),
            evaluations=int(data.get("evaluations", 0)),
            history=[dict(item) for item in data.get("history", ()) if isinstance(item, Mapping)],
            metadata=dict(data.get("metadata", {}))
            if isinstance(data.get("metadata", {}), Mapping)
            else {},
        )

    def resume_state(self) -> dict[str, Any]:
        """Return a self-contained, JSON-friendly continuation token.

        Predicate results are cached by content digest.  Persisting that cache
        makes a resumed run safe and cheap without pretending that a time
        budget from the previous process is still available.
        """
        metadata = self.metadata
        cache = metadata.get("candidate_cache", {})
        if not isinstance(cache, Mapping):
            cache = {}
        return {
            "version": 1,
            "value": self.value,
            "original_size": self.original_size,
            "original_digest": metadata.get("original_digest"),
            "history": list(self.history),
            "evaluations": self.evaluations,
            "predicate_calls": self.predicate_calls,
            "cache": dict(cache),
            "metadata": {
                key: metadata[key]
                for key in ("strategy", "trials", "quorum", "signature_checks")
                if key in metadata
            },
        }


ShrinkReport = ShrinkResult


def _is_tensorish(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    return shape is not None and (
        hasattr(value, "detach") or hasattr(value, "reshape") or hasattr(value, "numpy")
    )


def _nested_size(value: Any) -> float:
    if _is_tensorish(value):
        numel = getattr(value, "numel", None)
        if callable(numel):
            return float(numel())
        shape = getattr(value, "shape", ())
        total = 1
        for dim in shape:
            total *= int(dim)
        return float(total)
    if isinstance(value, Mapping):
        return sum(_nested_size(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_nested_size(item) for item in value)
    return _complexity(value)[0]


def _slice_tensor(value: Any, axis: int, length: int) -> Any:
    slicers: list[Any] = [slice(None)] * len(value.shape)
    slicers[axis] = slice(0, length)
    sliced = value[tuple(slicers)]
    clone = getattr(sliced, "contiguous", None)
    return clone() if callable(clone) else sliced


def _walk_tensors(value: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any]]:
    found: list[tuple[tuple[Any, ...], Any]] = []
    if _is_tensorish(value):
        found.append((prefix, value))
        return found
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.extend(_walk_tensors(child, prefix + (("key", key),)))
        return found
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found.extend(_walk_tensors(child, prefix + (("item", index),)))
    return found


def _replace_at(value: Any, path: tuple[Any, ...], replacement: Any) -> Any:
    if not path:
        return replacement
    kind, token = path[0]
    rest = path[1:]
    if kind == "key" and isinstance(value, Mapping):
        result = dict(value)
        result[token] = _replace_at(value[token], rest, replacement)
        return result
    if kind == "item" and isinstance(value, (list, tuple)):
        factory = tuple if isinstance(value, tuple) else list
        items = list(value)
        items[int(token)] = _replace_at(items[int(token)], rest, replacement)
        return factory(items)
    return value


def _linked_tensor_candidates(value: Any) -> Iterable[tuple[str, Any]]:
    tensors = _walk_tensors(value)
    if not tensors:
        return
    groups: dict[int, list[tuple[tuple[Any, ...], int, Any]]] = {}
    for path, tensor in tensors:
        shape = tuple(int(dim) for dim in tensor.shape)
        for axis, dim in enumerate(shape):
            if dim > 1:
                groups.setdefault(dim, []).append((path, axis, tensor))
    for dim, members in groups.items():
        lengths = sorted({1, max(1, dim // 2), dim - 1})
        for length in lengths:
            if length >= dim:
                continue
            candidate = value
            for path, axis, tensor in members:
                if int(tensor.shape[axis]) != dim:
                    continue
                candidate = _replace_at(candidate, path, _slice_tensor(tensor, axis, length))
            if _nested_size(candidate) < _nested_size(value):
                yield f"linked_slice:{dim}->{length}", candidate


def _resetting_predicate(predicate: Predicate, fixture: Any, checkpoint: Any) -> Predicate:
    def wrapped(value: Any) -> Any:
        if fixture is not None:
            close = getattr(fixture, "close", None)
            construct = getattr(fixture, "construct", None)
            restore = getattr(fixture, "restore", None)
            if callable(close):
                close()
            if callable(construct):
                construct()
            if callable(restore) and checkpoint is not None:
                restore(checkpoint)
        return predicate(value)

    return wrapped


def _cached_predicate(predicate: Predicate) -> Predicate:
    cache: dict[str, Any] = {}

    def wrapped(value: Any) -> Any:
        digest = digest_state_tree(value) or repr(value)
        if digest in cache:
            return cache[digest]
        result = predicate(value)
        cache[digest] = result
        return result

    return wrapped


def _candidate_digest(value: Any) -> str:
    """Return a content identity for a candidate, including tensor payloads."""
    digest = digest_state_tree(value)
    if digest is not None:
        return digest
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr)
    except (TypeError, ValueError):
        encoded = repr(value)
    return hashlib.sha256(encoded.encode("utf-8", "replace")).hexdigest()


def _resume_parts(
    resume: Any,
) -> tuple[Any, list[dict[str, Any]], int | float | None, dict[str, Any]]:
    """Normalize the supported continuation-token shapes."""
    if resume is None:
        return None, [], None, {}
    if isinstance(resume, ShrinkResult):
        return resume.value, list(resume.history), resume.original_size, resume.resume_state()
    if not isinstance(resume, Mapping):
        raise TypeError("resume must be a ShrinkResult or mapping")
    nested = resume.get("resume")
    state = dict(nested) if isinstance(nested, Mapping) else dict(resume)
    value = state.get("value", resume.get("value"))
    history_value = state.get("history", resume.get("history", ()))
    history = [dict(item) for item in history_value if isinstance(item, Mapping)]
    original_size = state.get("original_size", resume.get("original_size"))
    return value, history, original_size, state


def _validate_resume_input(original: Any, state: Mapping[str, Any]) -> None:
    expected = state.get("original_digest")
    if expected is not None and expected != _candidate_digest(original):
        raise ValueError("resume token belongs to a different initial input")


def _copy_resume_cache(state: Mapping[str, Any]) -> dict[str, Any]:
    cache = state.get("cache", state.get("candidate_cache", {}))
    return dict(cache) if isinstance(cache, Mapping) else {}


def _signature_expected(
    value: FailureSignature | Mapping[str, Any] | None,
) -> FailureSignature | None:
    if value is None or isinstance(value, FailureSignature):
        return value
    if isinstance(value, Mapping):
        return FailureSignature.from_dict(value)
    raise TypeError("expected_failure must be a FailureSignature, mapping, or None")


def _with_signature_check(
    result: PredicateResult, expected: FailureSignature | None
) -> PredicateResult:
    """Apply the common replay/shrink failure identity when a predicate reports one."""
    if expected is None or result.actual is None or not result.preserved:
        if expected is not None and result.preserved and result.actual is None:
            return replace(result, metadata={**dict(result.metadata), "signature_checked": False})
        return result
    matched = expected.matches(result.actual)
    metadata = {**dict(result.metadata), "signature_checked": True, "signature_match": matched}
    if matched:
        return replace(result, metadata=metadata)
    return PredicateResult(
        preserved=False,
        status="pass",
        actual=result.actual,
        reason="failure signature mismatch",
        unresolved=False,
        metadata=metadata,
    )


class _CandidateEvaluator:
    """Bounded, content-addressed and optionally quorum-based predicate runner."""

    def __init__(
        self,
        predicate: Predicate,
        *,
        expected_failure: FailureSignature | Mapping[str, Any] | None = None,
        trials: int = 1,
        quorum: int | None = None,
        max_evaluations: int | None = None,
        max_seconds: float | None = None,
        resume: Any = None,
    ) -> None:
        if trials < 1:
            raise ValueError("trials must be at least 1")
        if quorum is None:
            quorum = trials // 2 + 1
        if quorum < 1 or quorum > trials:
            raise ValueError("quorum must be between 1 and trials")
        if max_evaluations is not None and max_evaluations < 1:
            raise ValueError("max_evaluations must be at least 1")
        if max_seconds is not None and (max_seconds < 0 or not math.isfinite(float(max_seconds))):
            raise ValueError("max_seconds must be a finite non-negative number")
        self.predicate = predicate
        self.expected_failure = _signature_expected(expected_failure)
        self.trials = trials
        self.quorum = quorum
        self.max_evaluations = max_evaluations
        self.max_seconds = max_seconds
        self.started = time.monotonic()
        self.cache: dict[str, PredicateResult] = {}
        self.cache_hits = 0
        self.evaluations = 0
        self.predicate_calls = 0
        self.flaky_candidates = 0
        self.signature_checks = 0
        self.signature_mismatches = 0
        self.last_cached = False
        _value, _history, _original_size, resume_state = _resume_parts(resume)
        if resume_state:
            self.evaluations = max(0, int(resume_state.get("evaluations", 0)))
            self.predicate_calls = max(0, int(resume_state.get("predicate_calls", 0)))
            for digest, record in _copy_resume_cache(resume_state).items():
                try:
                    self.cache[str(digest)] = (
                        record
                        if isinstance(record, PredicateResult)
                        else PredicateResult.from_dict(record)
                    )
                except (TypeError, ValueError, KeyError):
                    # A partial/corrupt cache must never turn a resume into a
                    # false pass.  It is safe to recompute that candidate.
                    continue

    @property
    def exhausted(self) -> bool:
        return bool(
            (self.max_evaluations is not None and self.evaluations >= self.max_evaluations)
            or (
                self.max_seconds is not None and time.monotonic() - self.started >= self.max_seconds
            )
        )

    def __call__(self, value: Any, *, initial: bool = False) -> PredicateResult:
        key = _candidate_digest(value)
        self.last_cached = key in self.cache
        if self.last_cached:
            self.cache_hits += 1
            return self.cache[key]
        if self.exhausted:
            return PredicateResult(
                preserved=False,
                status="inconclusive",
                unresolved=True,
                reason="shrink evaluation budget exhausted",
                metadata={"budget_exhausted": True},
            )

        self.evaluations += 1
        observations: list[PredicateResult] = []
        for trial in range(self.trials):
            if trial and self.exhausted:
                observations.append(
                    PredicateResult(
                        preserved=False,
                        status="inconclusive",
                        unresolved=True,
                        reason="shrink evaluation budget exhausted",
                        metadata={"budget_exhausted": True},
                    )
                )
                break
            self.predicate_calls += 1
            observed = _evaluate(self.predicate, value, initial=initial and trial == 0)
            observed = _with_signature_check(observed, self.expected_failure)
            observations.append(observed)
        votes = [_accepted(item) for item in observations]
        accepted_votes = sum(votes)
        unresolved = any(item.unresolved for item in observations)
        actuals = [item.actual for item in observations if item.actual is not None]
        signature_keys = {item.grouping_key() for item in actuals}
        flaky = len(set(votes)) > 1 or len(signature_keys) > 1
        if flaky:
            self.flaky_candidates += 1
        if any(item.metadata.get("signature_checked") for item in observations):
            self.signature_checks += 1
        if any(item.metadata.get("signature_match") is False for item in observations):
            self.signature_mismatches += 1
        actual = next((item for item in actuals if item is not None), None)
        preserved = accepted_votes >= self.quorum and not unresolved
        if preserved:
            status = "fail"
        elif unresolved or (flaky and accepted_votes > 0):
            status = "inconclusive"
        else:
            status = "pass"
        reasons = [item.reason for item in observations if item.reason]
        result = PredicateResult(
            preserved=preserved,
            status=status,
            actual=actual,
            reason=reasons[0] if reasons else "",
            unresolved=unresolved,
            metadata={
                "trials": self.trials,
                "quorum": self.quorum,
                "votes": votes,
                "flaky": flaky,
                "signature_checked": bool(
                    any(item.metadata.get("signature_checked") for item in observations)
                ),
                "signature_match": (
                    all(item.metadata.get("signature_match", True) for item in observations)
                    if actuals
                    else None
                ),
            },
        )
        self.cache[key] = result
        return result

    def metadata(self) -> dict[str, Any]:
        exhausted = self.exhausted
        if self.expected_failure is not None and self.signature_mismatches:
            guarantee = "signature_mismatch_observed"
        elif exhausted:
            guarantee = "bounded"
        else:
            guarantee = "locally_minimal"
        cache = {key: value.to_dict() for key, value in self.cache.items()}
        return {
            "evaluations": self.evaluations,
            "predicate_calls": self.predicate_calls,
            "cache_hits": self.cache_hits,
            "cached_candidates": len(self.cache),
            "trials": self.trials,
            "quorum": self.quorum,
            "flaky_candidates": self.flaky_candidates,
            "signature_checks": self.signature_checks,
            "signature_mismatches": self.signature_mismatches,
            "budget_exhausted": exhausted,
            "wall_time_s": time.monotonic() - self.started,
            "minimality": guarantee,
            "candidate_cache": cache,
        }


def _history_entry(
    operation: str, result: PredicateResult, evaluator: _CandidateEvaluator, **extra: Any
) -> dict[str, Any]:
    return {
        "operation": operation,
        "accepted": _accepted(result),
        "unresolved": result.unresolved,
        "flaky": bool(result.metadata.get("flaky", False)),
        "cached": evaluator.last_cached,
        "signature_match": result.metadata.get("signature_match"),
        **extra,
    }


def _budget_metadata(
    history: Sequence[Mapping[str, Any]],
    evaluations: int,
    max_evaluations: int | None,
) -> dict[str, Any]:
    unresolved = any(bool(item.get("unresolved")) for item in history)
    flaky = any(bool(item.get("flaky")) for item in history)
    exhausted = max_evaluations is not None and evaluations >= max_evaluations
    if unresolved:
        guarantee = "unresolved"
    elif exhausted:
        guarantee = "bounded"
    else:
        guarantee = "locally_minimal"
    return {
        "budget_exhausted": exhausted,
        "unresolved_evaluations": unresolved,
        "flakiness": flaky,
        "minimality": guarantee,
    }


def _evaluate(predicate: Predicate, value: Any, *, initial: bool = False) -> PredicateResult:
    try:
        raw = predicate(value)
        try:
            result = normalize_predicate_result(raw)
        except TypeError:
            # NumPy and Torch scalar booleans are common predicate results but
            # are intentionally not accepted by the core JSON contract.
            item = raw.item() if callable(getattr(raw, "item", None)) else raw
            if not isinstance(item, bool):
                raise
            result = normalize_predicate_result(item)
    except UnresolvedEvaluation as exc:
        result = normalize_predicate_result(exc)
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


def _require_failing(predicate: Predicate, value: Any) -> PredicateResult:
    result = (
        predicate(value, initial=True)
        if isinstance(predicate, _CandidateEvaluator)
        else _evaluate(predicate, value, initial=True)
    )
    if not _accepted(result):
        detail = f": {result.reason}" if result.reason else ""
        raise InitialPredicateError(
            "cannot shrink: the initial value does not satisfy the failure predicate" + detail
        )
    return result


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
    expected_failure: FailureSignature | Mapping[str, Any] | None = None,
    trials: int = 1,
    quorum: int | None = None,
    max_seconds: float | None = None,
    _evaluator: _CandidateEvaluator | None = None,
    resume: Any = None,
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
    _resume_value, resume_history, resume_original_size, resume_state = _resume_parts(resume)
    _validate_resume_input(original, resume_state)
    evaluator = _evaluator or _CandidateEvaluator(
        predicate,
        expected_failure=expected_failure,
        trials=trials,
        quorum=quorum,
        max_evaluations=max_evaluations,
        max_seconds=max_seconds,
        resume=resume,
    )
    starting = original if _resume_value is None else _resume_value
    _require_failing(evaluator, starting)
    current = list(starting)
    history: list[dict[str, Any]] = resume_history
    granularity = 2

    while len(current) > min_chunk:
        chunk_size = max(1, math.ceil(len(current) / granularity))
        reduced = False
        for start in range(0, len(current), chunk_size):
            candidate_items = current[:start] + current[start + chunk_size :]
            if len(candidate_items) < min_chunk:
                continue
            if evaluator.exhausted:
                break
            candidate = factory(candidate_items)
            accepted = evaluator(candidate)
            history.append(
                _history_entry(
                    "remove_chunk",
                    accepted,
                    evaluator,
                    start=start,
                    count=min(chunk_size, len(current) - start),
                    size=len(candidate_items),
                )
            )
            if _accepted(accepted):
                current = candidate_items
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if evaluator.exhausted:
            break
        if not reduced:
            if granularity >= len(current):
                break
            granularity = min(len(current), granularity * 2)

    minimized = factory(current)
    verified = evaluator(minimized)
    if verified.unresolved or not _accepted(verified):
        if not verified.unresolved:
            raise RuntimeError("shrink predicate was not stable when the final value was verified")
    metadata = {
        "initial_failing": True,
        "verified": _minimality_verified(verified, history),
        "strategy": "ddmin",
        "original_digest": _candidate_digest(original),
        "resume_used": _resume_value is not None,
        **evaluator.metadata(),
    }
    return ShrinkResult(
        minimized,
        resume_original_size if resume_original_size is not None else len(original),
        len(current),
        evaluator.evaluations,
        history,
        metadata,
    )


def shrink_rows(rows: Sequence[Any], predicate: Predicate, **kwargs: Any) -> ShrinkResult:
    return ddmin(rows, predicate, **kwargs)


def shrink_columns(
    row_data: Sequence[Mapping[str, Any]],
    predicate: Predicate,
    *,
    columns: Iterable[str] | None = None,
    max_evaluations: int | None = None,
    expected_failure: FailureSignature | Mapping[str, Any] | None = None,
    trials: int = 1,
    quorum: int | None = None,
    max_seconds: float | None = None,
    resume: Any = None,
) -> ShrinkResult:
    """Remove columns with delta debugging while preserving rows and key order."""

    original = [dict(row) for row in copy.deepcopy(row_data)]
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
        lambda kept: predicate(project(kept)),
        min_chunk=0,
        max_evaluations=max_evaluations,
        expected_failure=expected_failure,
        trials=trials,
        quorum=quorum,
        max_seconds=max_seconds,
        resume=resume,
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
        reduced.evaluations,
        history,
        {
            **reduced.metadata,
            "strategy": "columns",
            "removed_columns": [name for name in all_names if name not in final_names],
            "original_digest": _candidate_digest(original),
            "resume_used": resume is not None,
        },
    )


def shrink_tokens(tokens: Sequence[Any], predicate: Predicate, **kwargs: Any) -> ShrinkResult:
    return ddmin(tokens, predicate, **kwargs)


def _scalar_candidates(value: Any) -> Iterable[tuple[str, Any]]:
    """Yield deterministic, type-preserving scalar simplifications."""
    if isinstance(value, bool):
        if value:
            yield "bool_false", False
        return
    if not isinstance(value, Number):
        return
    try:
        numeric = complex(value)
    except (TypeError, ValueError, OverflowError):
        return
    if isinstance(value, complex):
        candidates: list[Any] = [0j, value / 2, complex(round(numeric.real), round(numeric.imag))]
    else:
        try:
            candidates = [type(value)(0), value / 2, type(value)(1 if value > 0 else -1)]
        except (TypeError, ValueError, OverflowError):
            candidates = [0, value / 2]
        if isinstance(value, float) and math.isfinite(value):
            candidates.extend([type(value)(round(value)), type(value)(math.copysign(1.0, value))])
    seen: set[str] = set()
    for candidate in candidates:
        try:
            different = candidate != value
            if hasattr(different, "item"):
                different = different.item()
            different = bool(different)
        except Exception:
            different = True
        marker = repr(candidate)
        if different and marker not in seen:
            seen.add(marker)
            yield "reduce_scalar", candidate


def _strictly_smaller(candidate: Any, current: Any) -> bool:
    before = _complexity(current)
    after = _complexity(candidate)
    if after < before:
        return True
    return _nested_size(candidate) < _nested_size(current)


def _domain_reduce(
    value: Any,
    predicate: Predicate,
    candidates: Callable[[Any], Iterable[tuple[str, Any]]],
    *,
    strategy: str,
    max_evaluations: int | None = 256,
    expected_failure: FailureSignature | Mapping[str, Any] | None = None,
    trials: int = 1,
    quorum: int | None = None,
    max_seconds: float | None = None,
    resume: Any = None,
) -> ShrinkResult:
    """Run a bounded greedy reducer with one shared evaluator.

    Domain reducers deliberately share the same evaluator as ddmin.  This
    means a candidate is tested at most once per session, signatures and
    quorums have the same meaning everywhere, and a saved result can resume
    without silently resetting its budget or candidate cache.
    """
    if max_evaluations is not None and max_evaluations < 1:
        raise ValueError("max_evaluations must be at least 1")
    original = copy.deepcopy(value)
    resume_value, resume_history, resume_original_size, resume_state = _resume_parts(resume)
    _validate_resume_input(original, resume_state)
    evaluator = _CandidateEvaluator(
        predicate,
        expected_failure=expected_failure,
        trials=trials,
        quorum=quorum,
        max_evaluations=max_evaluations,
        max_seconds=max_seconds,
        resume=resume,
    )
    current = copy.deepcopy(original if resume_value is None else resume_value)
    _require_failing(evaluator, current)
    history = resume_history
    while not evaluator.exhausted:
        changed = False
        for operation, candidate in candidates(current):
            if evaluator.exhausted:
                break
            if _candidate_digest(candidate) == _candidate_digest(current):
                continue
            if not _strictly_smaller(candidate, current):
                continue
            result = evaluator(candidate)
            history.append(
                _history_entry(operation, result, evaluator, size=_nested_size(candidate))
            )
            if _accepted(result):
                current = candidate
                changed = True
                break
        if not changed:
            break
    verified = evaluator(current)
    if not verified.unresolved and not _accepted(verified):
        raise RuntimeError("shrink predicate was not stable when the final value was verified")
    metadata = {
        "initial_failing": True,
        "verified": _minimality_verified(verified, history),
        "strategy": strategy,
        "original_digest": _candidate_digest(original),
        "resume_used": resume_value is not None,
        **_budget_metadata(history, evaluator.evaluations, max_evaluations),
        **evaluator.metadata(),
    }
    return ShrinkResult(
        current,
        resume_original_size if resume_original_size is not None else _nested_size(original),
        _nested_size(current),
        evaluator.evaluations,
        history,
        metadata,
    )


def _table_parts(value: Any) -> tuple[list[Any], Callable[[Sequence[Any]], Any], set[str], bool]:
    """Return rows, a preserving builder, declared columns, and schema presence."""
    if isinstance(value, (list, tuple)):
        rows = list(copy.deepcopy(value))
        columns = set(key for row in rows if isinstance(row, Mapping) for key in row)
        factory = tuple if isinstance(value, tuple) else list
        return rows, lambda items: factory(copy.deepcopy(list(items))), columns, False
    if isinstance(value, Mapping):
        rows_key = next(
            (
                key
                for key in ("rows", "records", "data", "examples")
                if isinstance(value.get(key), (list, tuple))
            ),
            None,
        )
        if rows_key is None:
            raise TypeError("tabular input must be rows or a mapping containing rows")
        rows = list(copy.deepcopy(value[rows_key]))
        columns = set(key for row in rows if isinstance(row, Mapping) for key in row)
        schema = value.get("schema", value.get("columns"))
        declared = set(schema) if isinstance(schema, (list, tuple, set)) else set()
        if isinstance(schema, Mapping):
            declared.update(schema)
        declared.update(columns)

        def build(items: Sequence[Any]) -> Any:
            result = dict(value)
            result[rows_key] = copy.deepcopy(list(items))
            if declared:
                for row in result[rows_key]:
                    if isinstance(row, Mapping):
                        for name in declared:
                            row.setdefault(name, None)
            return result

        return rows, build, declared, "schema" in value or "columns" in value
    # Pandas is optional and is inspected only when a DataFrame is supplied.
    if type(value).__module__.split(".", 1)[0] == "pandas" and hasattr(value, "to_dict"):
        rows = list(value.to_dict(orient="records"))
        columns = set(str(item) for item in getattr(value, "columns", ()))

        def build(items: Sequence[Any]) -> Any:
            try:
                import pandas as pd  # type: ignore[import-not-found]

                return pd.DataFrame(list(items), columns=list(value.columns)).astype(
                    value.dtypes.to_dict()
                )
            except Exception:
                return list(items)

        return rows, build, columns, True
    raise TypeError("tabular input must be rows or a mapping containing rows")


def _tabular_cell_candidates(value: Any) -> Iterable[tuple[str, Any]]:
    yield from _scalar_candidates(value)
    if value is None:
        yield "null_to_zero", 0
    elif isinstance(value, str):
        if value:
            yield "clear_string", ""
            words = value.split()
            if len(words) > 1:
                yield "shorten_string", words[0]
            if value not in {"0", "1", "x"}:
                yield "canonical_category", "x"
    elif isinstance(value, Mapping):
        for operation, candidate in _structured_candidates(value):
            yield f"mapping/{operation}", candidate
    elif isinstance(value, (list, tuple)):
        for operation, candidate in _structured_candidates(value):
            yield f"sequence/{operation}", candidate


def shrink_tabular(
    value: Any,
    predicate: Predicate,
    *,
    preserve_schema: bool = True,
    drop_columns: bool = False,
    max_evaluations: int | None = 256,
    expected_failure: FailureSignature | Mapping[str, Any] | None = None,
    trials: int = 1,
    quorum: int | None = None,
    max_seconds: float | None = None,
    resume: Any = None,
) -> ShrinkResult:
    """Shrink rows and cell values while keeping tabular schema coherent."""
    rows, build, columns, has_schema = _table_parts(value)
    original = copy.deepcopy(value)

    def candidates(current: Any) -> Iterable[tuple[str, Any]]:
        current_rows, _current_build, _current_columns, _ = _table_parts(current)
        count = len(current_rows)
        for length in sorted({1, max(1, count // 2), count - 1}):
            if length <= 0 or length > count:
                continue
            for start in range(0, count - length + 1):
                yield (
                    f"remove_rows:{start}:{length}",
                    build(current_rows[:start] + current_rows[start + length :]),
                )
        if (drop_columns or not preserve_schema) and not has_schema:
            names = list(
                dict.fromkeys(
                    key for row in current_rows if isinstance(row, Mapping) for key in row
                )
            )
            for name in names:
                candidate_rows = [
                    {key: item for key, item in row.items() if key != name}
                    if isinstance(row, Mapping)
                    else row
                    for row in current_rows
                ]
                yield f"remove_column:{name}", build(candidate_rows)
        for row_index, row in enumerate(current_rows):
            if not isinstance(row, Mapping):
                continue
            for name, cell in row.items():
                for operation, reduced in _tabular_cell_candidates(cell):
                    candidate_rows = copy.deepcopy(current_rows)
                    candidate_rows[row_index] = dict(candidate_rows[row_index])
                    candidate_rows[row_index][name] = reduced
                    yield f"cell:{row_index}:{name}/{operation}", build(candidate_rows)

    result = _domain_reduce(
        original,
        predicate,
        candidates,
        strategy="tabular",
        max_evaluations=max_evaluations,
        expected_failure=expected_failure,
        trials=trials,
        quorum=quorum,
        max_seconds=max_seconds,
        resume=resume,
    )
    result.metadata.update(
        {
            "schema_preserved": bool(preserve_schema and (has_schema or columns)),
            "columns": sorted(str(item) for item in columns),
        }
    )
    return result


def _nlp_token_candidates(token: Any) -> Iterable[tuple[str, Any]]:
    yield from _scalar_candidates(token)
    if isinstance(token, str) and token:
        if not token.startswith(("<", "[")):
            yield "unknown_token", "<unk>"
        if len(token) > 1:
            yield "shorten_token", token[:1]


def _remap_span(span: Any, positions: Sequence[int], position_map: Mapping[int, int]) -> Any | None:
    if isinstance(span, Mapping):
        start = span.get("start", span.get("begin", span.get("token_start")))
        end = span.get("end", span.get("stop", span.get("token_end")))
        if isinstance(start, int) and isinstance(end, int):
            kept = [item for item in positions if start <= item < end]
            if not kept:
                return None
            result = dict(span)
            result["start"] = position_map[kept[0]]
            result["end"] = position_map[kept[-1]] + 1
            if "begin" in result:
                result["begin"] = result["start"]
            if "stop" in result:
                result["stop"] = result["end"]
            if "token_start" in result:
                result["token_start"] = result["start"]
            if "token_end" in result:
                result["token_end"] = result["end"]
            return result
        return dict(span)
    if isinstance(span, (list, tuple)) and len(span) >= 2:
        start, end = span[0], span[1]
        if isinstance(start, int) and isinstance(end, int):
            kept = [item for item in positions if start <= item < end]
            if not kept:
                return None
            values = list(span)
            values[0], values[1] = position_map[kept[0]], position_map[kept[-1]] + 1
            return tuple(values) if isinstance(span, tuple) else values
    return copy.deepcopy(span)


def _nlp_parts(value: Any) -> tuple[list[Any], Callable[[Sequence[Any], Sequence[int]], Any]]:
    if isinstance(value, str):
        tokens = re.findall(r"\S+", value)
        return tokens, lambda items, _positions: " ".join(str(item) for item in items)
    if isinstance(value, (list, tuple)):
        tokens = list(copy.deepcopy(value))
        factory = tuple if isinstance(value, tuple) else list
        return tokens, lambda items, _positions: factory(copy.deepcopy(list(items)))
    if isinstance(value, Mapping):
        token_key = next(
            (
                key
                for key in ("tokens", "input_ids", "ids", "words", "subwords")
                if isinstance(value.get(key), (list, tuple))
            ),
            None,
        )
        if token_key is None and isinstance(value.get("text"), str):
            tokens = re.findall(r"\S+", value["text"])
            token_key = "text"
        if token_key is None:
            raise TypeError("NLP input must contain tokens, input_ids, ids, words, or text")
        tokens = (
            re.findall(r"\S+", value["text"])
            if token_key == "text"
            else list(copy.deepcopy(value[token_key]))
        )

        def build(items: Sequence[Any], positions: Sequence[int]) -> Any:
            result = copy.deepcopy(dict(value))
            if token_key == "text":
                result["text"] = " ".join(str(item) for item in items)
            else:
                result[token_key] = copy.deepcopy(list(items))
            for key, field_value in list(result.items()):
                if key in {token_key, "text", "spans", "entities", "annotations"}:
                    continue
                if isinstance(field_value, (list, tuple)) and len(field_value) == len(tokens):
                    factory = tuple if isinstance(field_value, tuple) else list
                    result[key] = factory(
                        copy.deepcopy([field_value[index] for index in positions])
                    )
            position_map = {old: new for new, old in enumerate(positions)}
            for key in ("spans", "entities", "annotations"):
                if isinstance(result.get(key), (list, tuple)):
                    mapped = [_remap_span(item, positions, position_map) for item in value[key]]
                    result[key] = [item for item in mapped if item is not None]
            return result

        return tokens, build
    raise TypeError("unsupported NLP input")


def shrink_nlp(
    value: Any,
    predicate: Predicate,
    *,
    preserve_spans: bool = True,
    max_evaluations: int | None = 256,
    expected_failure: FailureSignature | Mapping[str, Any] | None = None,
    trials: int = 1,
    quorum: int | None = None,
    max_seconds: float | None = None,
    resume: Any = None,
) -> ShrinkResult:
    """Reduce token sequences/documents and remap token-indexed spans."""
    tokens, build = _nlp_parts(value)

    def candidates(current: Any) -> Iterable[tuple[str, Any]]:
        current_tokens, current_build = _nlp_parts(current)
        count = len(current_tokens)
        for length in sorted({1, max(1, count // 2), count - 1}):
            if length <= 0 or length > count:
                continue
            for start in range(0, count - length + 1):
                positions = [index for index in range(count) if not start <= index < start + length]
                yield (
                    f"remove_tokens:{start}:{length}",
                    current_build([current_tokens[index] for index in positions], positions),
                )
        for index, token in enumerate(current_tokens):
            for operation, reduced in _nlp_token_candidates(token):
                items = list(current_tokens)
                items[index] = reduced
                yield f"token:{index}/{operation}", current_build(items, list(range(count)))
        if current_tokens:
            canonical = current_tokens[0]
            if any(item != canonical for item in current_tokens):
                yield (
                    "vocabulary_canonicalize",
                    current_build([canonical for _item in current_tokens], list(range(count))),
                )

    result = _domain_reduce(
        value,
        predicate,
        candidates,
        strategy="nlp",
        max_evaluations=max_evaluations,
        expected_failure=expected_failure,
        trials=trials,
        quorum=quorum,
        max_seconds=max_seconds,
        resume=resume,
    )
    result.metadata.update({"token_count": len(tokens), "spans_remapped": preserve_spans})
    return result


shrink_document = shrink_nlp


def _array_shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    try:
        return tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        return ()


def _array_copy(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "clone"):
        try:
            return value.detach().clone()
        except Exception:
            pass
    copier = getattr(value, "copy", None)
    if callable(copier):
        try:
            return copier()
        except Exception:
            pass
    return copy.deepcopy(value)


def _array_slice(value: Any, slices: Sequence[Any]) -> Any:
    candidate = value[tuple(slices)]
    contiguous = getattr(candidate, "contiguous", None)
    return contiguous() if callable(contiguous) else candidate


def _array_zeros(value: Any) -> Any:
    candidate = _array_copy(value)
    try:
        candidate[...] = 0
        return candidate
    except Exception:
        try:
            return candidate * 0
        except Exception:
            return candidate


def _vision_axes(shape: Sequence[int], layout: str = "auto") -> tuple[int, int, int | None]:
    rank = len(shape)
    normalized = layout.casefold().replace("-", "")
    if normalized in {"hw", "2d"} and rank >= 2:
        return rank - 2, rank - 1, None
    if normalized in {"chw", "channelsfirst"} and rank >= 3:
        return rank - 2, rank - 1, rank - 3
    if normalized in {"hwc", "channelslast"} and rank >= 3:
        return rank - 3, rank - 2, rank - 1
    if normalized in {"nchw", "batchchannelsfirst"} and rank >= 4:
        return rank - 2, rank - 1, rank - 3
    if normalized in {"nhwc", "batchchannelslast"} and rank >= 4:
        return rank - 3, rank - 2, rank - 1
    if rank == 2:
        return 0, 1, None
    if rank == 3:
        if shape[0] <= 4 < shape[-1]:
            return 1, 2, 0
        return 0, 1, 2
    if rank >= 4:
        if shape[1] <= 8 and shape[-1] > 8:
            return rank - 2, rank - 1, rank - 3
        return rank - 3, rank - 2, rank - 1
    return 0, 0, None


def _spatial_slices(
    shape: Sequence[int],
    height_axis: int,
    width_axis: int,
    start_h: int,
    height: int,
    start_w: int,
    width: int,
) -> list[Any]:
    slices: list[Any] = [slice(None)] * len(shape)
    slices[height_axis] = slice(start_h, start_h + height)
    slices[width_axis] = slice(start_w, start_w + width)
    return slices


def _vision_transform(
    value: Any,
    operation: str,
    *,
    layout: str = "auto",
    preserve_shape: bool = False,
) -> Any:
    """Apply a spatial/channel operation to an array or an input mapping."""
    if isinstance(value, Mapping):
        image_key = next(
            (
                key
                for key in ("image", "images", "pixels", "input", "inputs", "x", "data")
                if key in value and _array_shape(value[key])
            ),
            None,
        )
        if image_key is None:
            raise TypeError("vision input mapping has no image-like tensor")
        image = value[image_key]
        shape = _array_shape(image)
        height_axis, width_axis, channel_axis = _vision_axes(shape, layout)
        result = dict(value)
        transformed_image = _vision_transform(
            image,
            operation,
            layout=layout,
            preserve_shape=preserve_shape,
        )
        result[image_key] = transformed_image
        for key, item in value.items():
            item_shape = _array_shape(item)
            if key == image_key or len(item_shape) < 2:
                continue
            item_h, item_w, _item_channel = _vision_axes(item_shape, layout)
            if item_h >= len(item_shape) or item_w >= len(item_shape):
                continue
            if operation.startswith("crop:"):
                _prefix, start_h, height, start_w, width = operation.split(":")
                slices = _spatial_slices(
                    item_shape,
                    item_h,
                    item_w,
                    int(start_h),
                    min(int(height), item_shape[item_h]),
                    int(start_w),
                    min(int(width), item_shape[item_w]),
                )
                try:
                    result[key] = _array_slice(item, slices)
                except (IndexError, TypeError, ValueError):
                    pass
            elif operation.startswith("resolution:"):
                factor = max(1, int(operation.split(":", 1)[1]))
                slices = [slice(None)] * len(item_shape)
                slices[item_h] = slice(None, None, factor)
                slices[item_w] = slice(None, None, factor)
                try:
                    result[key] = _array_slice(item, slices)
                except (IndexError, TypeError, ValueError):
                    pass
        return result
    shape = _array_shape(value)
    if len(shape) < 2:
        return value
    height_axis, width_axis, channel_axis = _vision_axes(shape, layout)
    if operation.startswith("crop:"):
        _prefix, start_h, height, start_w, width = operation.split(":")
        slices = _spatial_slices(
            shape,
            height_axis,
            width_axis,
            int(start_h),
            int(height),
            int(start_w),
            int(width),
        )
        if preserve_shape:
            candidate = _array_zeros(value)
            try:
                target = tuple(slices)
                candidate[target] = value[target]
                return candidate
            except (IndexError, TypeError, ValueError):
                return candidate
        return _array_slice(value, slices)
    if operation.startswith("channel:") and channel_axis is not None:
        _prefix, start, count = operation.split(":")
        slices = [slice(None)] * len(shape)
        slices[channel_axis] = slice(int(start), int(start) + int(count))
        if preserve_shape:
            candidate = _array_zeros(value)
            target = tuple(slices)
            try:
                candidate[target] = value[target]
                return candidate
            except (IndexError, TypeError, ValueError):
                return candidate
        return _array_slice(value, slices)
    if operation.startswith("resolution:"):
        factor = max(1, int(operation.split(":", 1)[1]))
        slices = [slice(None)] * len(shape)
        slices[height_axis] = slice(None, None, factor)
        slices[width_axis] = slice(None, None, factor)
        return _array_slice(value, slices)
    return value


def _vision_candidates_factory(
    layout: str, preserve_shape: bool
) -> Callable[[Any], Iterable[tuple[str, Any]]]:
    def candidates(value: Any) -> Iterable[tuple[str, Any]]:
        image = value
        if isinstance(value, Mapping):
            image = next(
                (
                    value[key]
                    for key in ("image", "images", "pixels", "input", "inputs", "x", "data")
                    if key in value and _array_shape(value[key])
                ),
                None,
            )
        shape = _array_shape(image)
        if len(shape) < 2:
            return
        height_axis, width_axis, channel_axis = _vision_axes(shape, layout)
        height, width = shape[height_axis], shape[width_axis]
        for new_height in sorted({1, max(1, height // 2), height - 1}):
            if new_height >= height or new_height <= 0:
                continue
            for start_h in sorted({0, max(0, (height - new_height) // 2), height - new_height}):
                for new_width in sorted({1, max(1, width // 2), width - 1}):
                    if new_width >= width or new_width <= 0:
                        continue
                    for start_w in sorted({0, max(0, (width - new_width) // 2), width - new_width}):
                        operation = f"crop:{start_h}:{new_height}:{start_w}:{new_width}"
                        yield (
                            operation,
                            _vision_transform(
                                value, operation, layout=layout, preserve_shape=preserve_shape
                            ),
                        )
        if channel_axis is not None and shape[channel_axis] > 1:
            channels = shape[channel_axis]
            for count in sorted({1, max(1, channels // 2), channels - 1}):
                if count < channels:
                    yield (
                        f"channel:0:{count}",
                        _vision_transform(
                            value,
                            f"channel:0:{count}",
                            layout=layout,
                            preserve_shape=preserve_shape,
                        ),
                    )
        for factor in (2, 4):
            if height // factor >= 1 and width // factor >= 1:
                yield (
                    f"resolution:{factor}",
                    _vision_transform(
                        value, f"resolution:{factor}", layout=layout, preserve_shape=preserve_shape
                    ),
                )

    return candidates


def shrink_vision(
    value: Any,
    predicate: Predicate,
    *,
    layout: str = "auto",
    preserve_shape: bool = False,
    max_evaluations: int | None = 256,
    expected_failure: FailureSignature | Mapping[str, Any] | None = None,
    trials: int = 1,
    quorum: int | None = None,
    max_seconds: float | None = None,
    resume: Any = None,
) -> ShrinkResult:
    """Shrink image-like arrays through crops, masks, channels, and resolution."""
    result = _domain_reduce(
        value,
        predicate,
        _vision_candidates_factory(layout, preserve_shape),
        strategy="vision",
        max_evaluations=max_evaluations,
        expected_failure=expected_failure,
        trials=trials,
        quorum=quorum,
        max_seconds=max_seconds,
        resume=resume,
    )
    result.metadata.update({"layout": layout, "shape_preserved": preserve_shape})
    return result


shrink_image = shrink_vision


def _sequence_candidates_factory(
    dependencies: Mapping[int, Iterable[int]] | None = None,
) -> Callable[[Any], Iterable[tuple[str, Any]]]:
    dependency_map = {
        int(key): {int(item) for item in values} for key, values in (dependencies or {}).items()
    }

    def candidates(value: Any) -> Iterable[tuple[str, Any]]:
        if isinstance(value, Mapping):
            key = next(
                (
                    name
                    for name in ("steps", "sequence", "events", "items")
                    if isinstance(value.get(name), (list, tuple))
                ),
                None,
            )
            if key is None:
                return
            items = list(value[key])
            factory = tuple if isinstance(value[key], tuple) else list

            def build(selected: Sequence[Any]) -> Any:
                result = dict(value)
                result[key] = factory(selected)
                return result
        elif isinstance(value, (list, tuple)):
            items = list(value)
            factory = tuple if isinstance(value, tuple) else list

            def build(selected: Sequence[Any]) -> Any:
                return factory(selected)
        else:
            return
        count = len(items)
        for length in sorted({1, max(1, count // 2), count - 1}):
            if length <= 0 or length > count:
                continue
            for start in range(0, count - length + 1):
                removed = set(range(start, start + length))
                kept = set(range(count)).difference(removed)
                changed = True
                while changed:
                    changed = False
                    for index, required in dependency_map.items():
                        if index in kept and not required.issubset(kept):
                            kept.update(required)
                            changed = True
                selected = [item for index, item in enumerate(items) if index in kept]
                if len(selected) < count:
                    yield f"remove_steps:{start}:{length}", build(selected)
        for index, item in enumerate(items):
            for operation, reduced in _scalar_candidates(item):
                selected = list(items)
                selected[index] = reduced
                yield f"step:{index}/{operation}", build(selected)

    return candidates


def shrink_sequence(
    value: Any,
    predicate: Predicate,
    *,
    dependencies: Mapping[int, Iterable[int]] | None = None,
    max_evaluations: int | None = 256,
    expected_failure: FailureSignature | Mapping[str, Any] | None = None,
    trials: int = 1,
    quorum: int | None = None,
    max_seconds: float | None = None,
    resume: Any = None,
) -> ShrinkResult:
    """Reduce ordered sequences while retaining declared prerequisite steps."""
    return _domain_reduce(
        value,
        predicate,
        _sequence_candidates_factory(dependencies),
        strategy="sequence",
        max_evaluations=max_evaluations,
        expected_failure=expected_failure,
        trials=trials,
        quorum=quorum,
        max_seconds=max_seconds,
        resume=resume,
    )


def _edge_endpoints(edge: Any) -> tuple[Any, Any] | None:
    if isinstance(edge, Mapping):
        source = edge.get("source", edge.get("src", edge.get("from")))
        target = edge.get("target", edge.get("dst", edge.get("to")))
        if source is not None and target is not None:
            return source, target
    if isinstance(edge, (list, tuple)) and len(edge) >= 2:
        return edge[0], edge[1]
    return None


def _graph_parts(
    value: Any,
) -> tuple[list[Any], list[Any], Callable[[Sequence[Any], Sequence[Any]], Any]]:
    if not isinstance(value, Mapping):
        if hasattr(value, "nodes") and hasattr(value, "edges"):
            nodes = list(value.nodes(data=True))
            edges = list(value.edges(data=True))
            try:
                prototype = value.copy()
            except Exception:
                prototype = None

            def build(new_nodes: Sequence[Any], new_edges: Sequence[Any]) -> Any:
                if prototype is None:
                    return {"nodes": list(new_nodes), "edges": list(new_edges)}
                graph = prototype.copy()
                graph.clear()
                for node in new_nodes:
                    if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], Mapping):
                        graph.add_node(node[0], **dict(node[1]))
                    else:
                        graph.add_node(node)
                for edge in new_edges:
                    endpoints = _edge_endpoints(edge)
                    if endpoints is not None:
                        graph.add_edge(*endpoints)
                return graph

            return nodes, edges, build
        raise TypeError("graph input must contain nodes and edges")
    nodes = value.get("nodes")
    edges = value.get("edges", value.get("links"))
    if not isinstance(nodes, (list, tuple, Mapping)) or not isinstance(edges, (list, tuple)):
        raise TypeError("graph input must contain sequence nodes and edges")
    if isinstance(nodes, Mapping):
        node_items = [(key, copy.deepcopy(item)) for key, item in nodes.items()]
    else:
        node_items = list(copy.deepcopy(nodes))
    edge_items = list(copy.deepcopy(edges))

    def build(new_nodes: Sequence[Any], new_edges: Sequence[Any]) -> Any:
        result = dict(value)
        if isinstance(nodes, Mapping):
            result["nodes"] = {
                item[0]: item[1] for item in new_nodes if isinstance(item, tuple) and len(item) == 2
            }
        else:
            result["nodes"] = tuple(new_nodes) if isinstance(nodes, tuple) else list(new_nodes)
        result["edges"] = tuple(new_edges) if isinstance(edges, tuple) else list(new_edges)
        return result

    return node_items, edge_items, build


def _graph_candidates(value: Any) -> Iterable[tuple[str, Any]]:
    nodes, edges, build = _graph_parts(value)
    node_ids = []
    for node in nodes:
        if isinstance(node, Mapping):
            node_ids.append(node.get("id", node.get("node")))
        elif isinstance(node, (tuple, list)) and node:
            node_ids.append(node[0])
        else:
            node_ids.append(node)
    for length in sorted({1, max(1, len(nodes) // 2), len(nodes) - 1}):
        if length <= 0 or length > len(nodes):
            continue
        for start in range(0, len(nodes) - length + 1):
            kept_nodes = nodes[:start] + nodes[start + length :]
            kept_ids = set(node_ids[:start] + node_ids[start + length :])
            kept_edges = [
                edge
                for edge in edges
                if (endpoints := _edge_endpoints(edge)) is not None
                and endpoints[0] in kept_ids
                and endpoints[1] in kept_ids
            ]
            yield f"remove_nodes:{start}:{length}", build(kept_nodes, kept_edges)
    for length in sorted({1, max(1, len(edges) // 2), len(edges) - 1}):
        if length <= 0 or length > len(edges):
            continue
        for start in range(0, len(edges) - length + 1):
            yield (
                f"remove_edges:{start}:{length}",
                build(nodes, edges[:start] + edges[start + length :]),
            )
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            continue
        node_id = node_ids[index]
        for key, item in node.items():
            if key in {"id", "node"}:
                continue
            for operation, reduced in _structured_candidates(item):
                candidate_nodes = copy.deepcopy(nodes)
                candidate_nodes[index] = dict(candidate_nodes[index])
                candidate_nodes[index][key] = reduced
                yield f"node:{node_id}:{key}/{operation}", build(candidate_nodes, edges)
    for index, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            continue
        for key, item in edge.items():
            if key in {"source", "target", "src", "dst", "from", "to"}:
                continue
            for operation, reduced in _structured_candidates(item):
                candidate_edges = copy.deepcopy(edges)
                candidate_edges[index] = dict(candidate_edges[index])
                candidate_edges[index][key] = reduced
                yield f"edge:{index}:{key}/{operation}", build(nodes, candidate_edges)


def shrink_graph(
    value: Any,
    predicate: Predicate,
    *,
    max_evaluations: int | None = 256,
    expected_failure: FailureSignature | Mapping[str, Any] | None = None,
    trials: int = 1,
    quorum: int | None = None,
    max_seconds: float | None = None,
    resume: Any = None,
) -> ShrinkResult:
    """Shrink graph nodes/edges while preserving edge endpoint validity."""
    result = _domain_reduce(
        value,
        predicate,
        _graph_candidates,
        strategy="graph",
        max_evaluations=max_evaluations,
        expected_failure=expected_failure,
        trials=trials,
        quorum=quorum,
        max_seconds=max_seconds,
        resume=resume,
    )
    result.metadata["graph_validity"] = "edges_filtered_to_retained_nodes"
    return result


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


def _tensor_dimension_candidates(value: Any) -> Iterable[tuple[str, Any]]:
    """Yield bounded prefix/suffix windows for legal dimension reduction."""
    shape = _shape(value)
    if not shape:
        return
    for axis, dimension in enumerate(shape):
        if dimension <= 1:
            continue
        lengths = sorted({1, max(1, dimension // 2), dimension - 1})
        for length in lengths:
            if length >= dimension:
                continue
            # All starts are required for adversarial signals in the middle of
            # a dimension. The candidate count is bounded by the tensor rank
            # and the dimension itself, and the caller still owns the trial
            # budget.
            for start in range(dimension - length + 1):
                slicers: list[Any] = [slice(None)] * len(shape)
                slicers[axis] = slice(start, start + length)
                try:
                    candidate = value[tuple(slicers)]
                    contiguous = getattr(candidate, "contiguous", None)
                    if callable(contiguous):
                        candidate = contiguous()
                except (IndexError, TypeError, ValueError):
                    continue
                yield f"slice_axis:{axis}:{start}:{length}", candidate


def _tensor_magnitude_candidates(value: Any) -> Iterable[tuple[str, Any]]:
    """Yield element-wise magnitude reductions without changing shape/dtype."""
    flat = _flatten(_tensor_to_list(value))
    for index, item in enumerate(flat):
        if isinstance(item, bool) or not isinstance(item, Number):
            continue
        try:
            numeric = complex(item)
            if not math.isfinite(numeric.real) or not math.isfinite(numeric.imag):
                continue
        except (TypeError, ValueError, OverflowError):
            continue
        candidates: list[Any]
        if isinstance(item, complex):
            candidates = [0j, item / 2, complex(round(numeric.real), round(numeric.imag))]
        else:
            try:
                candidates = [type(item)(0), item / 2, type(item)(1 if item > 0 else -1)]
            except (TypeError, ValueError, OverflowError):
                candidates = [0, item / 2]
            if isinstance(item, float):
                candidates.append(type(item)(round(item)))
        for reduced in candidates:
            if reduced == item:
                continue
            candidate_flat = list(flat)
            candidate_flat[index] = reduced
            try:
                yield f"reduce_magnitude:{index}", _restore_like(value, candidate_flat)
            except (IndexError, TypeError, ValueError, RuntimeError):
                continue


def _tensor_dtype_candidates(
    value: Any, dtypes: Sequence[Any] | None = None
) -> Iterable[tuple[str, Any]]:
    """Return legal, progressively smaller dtype casts using lazy imports."""
    current = getattr(value, "dtype", None)
    if current is None or not hasattr(value, "astype") and not hasattr(value, "to"):
        return
    targets: list[Any] = list(dtypes or ())
    module_name = type(value).__module__.split(".", 1)[0]
    if not targets and module_name == "numpy":
        try:
            import numpy as np  # type: ignore[import-not-found]

            dtype = np.dtype(current)
            if dtype.kind in "fc":
                targets = [np.float32, np.float16] if dtype.itemsize > 2 else []
            elif dtype.kind in "iu":
                targets = [np.int32, np.int16, np.int8] if dtype.itemsize > 1 else []
            elif dtype.kind == "b":
                targets = []
        except (ImportError, TypeError, ValueError):
            targets = []
    elif not targets and module_name == "torch":
        try:
            import torch  # type: ignore[import-not-found]

            if current in {torch.float64, torch.complex128}:
                targets = [torch.float32] if current == torch.float64 else [torch.complex64]
                targets.append(torch.float16 if current == torch.float64 else torch.complex64)
            elif current in {torch.int64, torch.int32, torch.int16}:
                targets = [torch.int32, torch.int16, torch.int8]
        except (ImportError, AttributeError, TypeError):
            targets = []
    seen: set[str] = set()
    for target in targets:
        marker = str(target)
        if marker in seen or str(target) == str(current):
            continue
        seen.add(marker)
        try:
            candidate = (
                value.astype(target, copy=True)
                if hasattr(value, "astype")
                else value.to(dtype=target)
            )
        except (TypeError, ValueError, RuntimeError):
            continue
        yield f"reduce_dtype:{marker}", candidate


def _tensor_measure(value: Any) -> tuple[float, float, int]:
    flat = _flatten(_tensor_to_list(value))
    magnitude = 0.0
    for item in flat:
        try:
            numeric = abs(float(item))
            if math.isfinite(numeric):
                magnitude += numeric
        except (TypeError, ValueError, OverflowError):
            magnitude = math.inf
            break
    dtype = str(getattr(value, "dtype", ""))
    return (float(len(flat)), magnitude, len(dtype))


def _reduce_tensor_values(
    value: Any,
    predicate: Predicate,
    *,
    reduce_magnitude: bool,
    reduce_dtype: bool,
    dtype_candidates: Sequence[Any] | None,
    max_evaluations: int | None,
    expected_failure: FailureSignature | Mapping[str, Any] | None,
    trials: int,
    quorum: int | None,
    max_seconds: float | None,
) -> tuple[Any, list[dict[str, Any]], int, dict[str, Any]]:
    if not reduce_magnitude and not reduce_dtype:
        return value, [], 0, {}
    evaluator = _CandidateEvaluator(
        predicate,
        expected_failure=expected_failure,
        trials=trials,
        quorum=quorum,
        max_evaluations=max_evaluations,
        max_seconds=max_seconds,
    )
    _require_failing(evaluator, value)
    history: list[dict[str, Any]] = []
    current = value
    while not evaluator.exhausted:
        changed = False
        candidates: list[tuple[str, Any]] = []
        if reduce_magnitude:
            candidates.extend(_tensor_magnitude_candidates(current))
        if reduce_dtype:
            candidates.extend(_tensor_dtype_candidates(current, dtype_candidates))
        for operation, candidate in candidates:
            if evaluator.exhausted:
                break
            if _tensor_measure(candidate) >= _tensor_measure(current) and operation.startswith(
                "reduce_magnitude"
            ):
                continue
            result = evaluator(candidate)
            history.append(_history_entry(operation, result, evaluator, shape=_shape(candidate)))
            if _accepted(result):
                current = candidate
                changed = True
                break
        if not changed:
            break
    verified = evaluator(current)
    if not verified.unresolved and not _accepted(verified):
        raise RuntimeError("shrink predicate was not stable when the tensor result was verified")
    return current, history, evaluator.evaluations, evaluator.metadata()


def shrink_tensor(
    value: Any,
    predicate: Predicate,
    *,
    max_elements: int = 100_000,
    max_evaluations: int | None = None,
    expected_failure: FailureSignature | Mapping[str, Any] | None = None,
    trials: int = 1,
    quorum: int | None = None,
    max_seconds: float | None = None,
    preserve_shape: bool = True,
    reduce_magnitude: bool = True,
    reduce_dtype: bool = False,
    dtype_candidates: Sequence[Any] | None = None,
    resume: Any = None,
) -> ShrinkResult:
    """Reduce tensor dimensions, values, magnitudes, and optionally dtype.

    Shape and dtype changes are opt-in.  Magnitude reduction is enabled by
    default but only changes a candidate after the failure predicate accepts it.
    """

    if max_elements < 0:
        raise ValueError("max_elements must be non-negative")
    source_value = copy.deepcopy(value)
    dimension_history: list[dict[str, Any]] = []
    dimension_evaluator: _CandidateEvaluator | None = None
    dimension_evaluations = 0
    resume_value, resume_history, resume_original_size, resume_state = _resume_parts(resume)
    _validate_resume_input(value, resume_state)
    if resume_value is not None:
        value = copy.deepcopy(resume_value)
    if not preserve_shape:
        dimension_evaluator = _CandidateEvaluator(
            predicate,
            expected_failure=expected_failure,
            trials=trials,
            quorum=quorum,
            max_evaluations=max_evaluations,
            max_seconds=max_seconds,
            resume=resume,
        )
        _require_failing(dimension_evaluator, value)
        current = value
        changed = True
        while changed and not dimension_evaluator.exhausted:
            changed = False
            for operation, candidate in _tensor_dimension_candidates(current):
                if _nested_size(candidate) >= _nested_size(current):
                    continue
                accepted = dimension_evaluator(candidate)
                dimension_history.append(
                    _history_entry(
                        operation,
                        accepted,
                        dimension_evaluator,
                        shape=_shape(candidate),
                    )
                )
                if _accepted(accepted):
                    current = candidate
                    changed = True
                    break
                if dimension_evaluator.exhausted:
                    break
        dimension_evaluations = dimension_evaluator.evaluations
        value = current
        if max_evaluations is not None:
            remaining = max_evaluations - dimension_evaluations
            if remaining < 1:
                return ShrinkResult(
                    value,
                    _nested_size(value),
                    _nested_size(value),
                    dimension_evaluations,
                    resume_history + dimension_history,
                    {
                        "initial_failing": True,
                        "verified": True,
                        "strategy": "tensor_dimensions",
                        "original_digest": _candidate_digest(value),
                        "resume_used": resume_value is not None,
                        **dimension_evaluator.metadata(),
                    },
                )
            max_evaluations = remaining
    raw = _tensor_to_list(value)
    flat = _flatten(raw)
    original_shape = _shape(value)
    if len(flat) > max_elements:
        checker = _CandidateEvaluator(
            predicate,
            expected_failure=expected_failure,
            trials=trials,
            quorum=quorum,
            max_evaluations=max_evaluations,
            max_seconds=max_seconds,
        )
        _require_failing(checker, value)
        return ShrinkResult(
            value,
            len(flat),
            len(flat),
            checker.evaluations,
            metadata={
                "initial_failing": True,
                "verified": not checker.exhausted,
                "skipped": "tensor exceeds max_elements",
                "shape": original_shape,
                **checker.metadata(),
            },
        )

    active = [index for index, item in enumerate(flat) if not _is_zero(item)]

    def build(kept: Sequence[int]) -> Any:
        kept_set = set(kept)
        candidate_flat = [
            item if index in kept_set else _zero_like(item) for index, item in enumerate(flat)
        ]
        return _restore_like(value, candidate_flat)

    evaluator = _CandidateEvaluator(
        lambda kept: predicate(build(kept)),
        expected_failure=expected_failure,
        trials=trials,
        quorum=quorum,
        max_evaluations=max_evaluations,
        max_seconds=max_seconds,
    )
    active_resume = None
    if resume_value is not None:
        resumed_flat = _flatten(_tensor_to_list(value))
        active_resume = {
            "value": [index for index, item in enumerate(resumed_flat) if not _is_zero(item)],
            "history": [],
            "evaluations": max(0, int(resume_state.get("evaluations", 0))),
            "predicate_calls": max(0, int(resume_state.get("predicate_calls", 0))),
            "cache": _copy_resume_cache(resume_state),
        }
    reduced = ddmin(
        active,
        lambda kept: predicate(build(kept)),
        min_chunk=0,
        _evaluator=evaluator,
        resume=active_resume,
    )
    result = build(reduced.value)
    history = []
    for item in reduced.history:
        entry = dict(item)
        entry["operation"] = "zero_region"
        history.append(entry)
    remaining = None
    if max_evaluations is not None:
        remaining = max_evaluations - dimension_evaluations - reduced.evaluations
    magnitude_value, magnitude_history, magnitude_evaluations, magnitude_metadata = (
        _reduce_tensor_values(
            result,
            predicate,
            reduce_magnitude=reduce_magnitude,
            reduce_dtype=reduce_dtype,
            dtype_candidates=dtype_candidates,
            max_evaluations=remaining,
            expected_failure=expected_failure,
            trials=trials,
            quorum=quorum,
            max_seconds=max_seconds,
        )
        if remaining is None or remaining > 0
        else (result, [], 0, {})
    )
    all_history = resume_history + dimension_history + history + magnitude_history
    total_evaluations = dimension_evaluations + reduced.evaluations + magnitude_evaluations
    merged_metadata = {
        **reduced.metadata,
        **(
            {"dimension_evaluations": dimension_evaluations}
            if dimension_evaluator is not None
            else {}
        ),
        **magnitude_metadata,
        "strategy": "tensor_reduction",
        "representation": type(value).__name__,
        "shape": original_shape,
        "shape_preserved": _shape(magnitude_value) == original_shape,
        "preserve_shape": preserve_shape,
        "dtype": str(getattr(magnitude_value, "dtype", "")) or None,
        "dtype_reduced": str(getattr(magnitude_value, "dtype", ""))
        != str(getattr(value, "dtype", "")),
        "magnitude_reduced": bool(magnitude_history),
        "original_digest": _candidate_digest(source_value),
        "resume_used": resume_value is not None,
    }
    return ShrinkResult(
        magnitude_value,
        _nested_size(source_value) if resume_original_size is None else resume_original_size,
        len(reduced.value) if preserve_shape else _nested_size(magnitude_value),
        total_evaluations,
        all_history,
        merged_metadata,
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
        yield from _linked_tensor_candidates(value)
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
        yield from _linked_tensor_candidates(value)
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
    if _is_tensorish(value):
        yield from _linked_tensor_candidates(value)
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
    expected_failure: FailureSignature | Mapping[str, Any] | None = None,
    trials: int = 1,
    quorum: int | None = None,
    max_seconds: float | None = None,
    resume: Any = None,
) -> ShrinkResult:
    """Recursively simplify JSON-like mappings and sequences."""

    if max_evaluations < 1:
        raise ValueError("max_evaluations must be at least 1")
    source = copy.deepcopy(value)
    resume_value, resume_history, resume_original_size, resume_state = _resume_parts(resume)
    _validate_resume_input(source, resume_state)
    current = copy.deepcopy(source if resume_value is None else resume_value)
    evaluator = _CandidateEvaluator(
        predicate,
        expected_failure=expected_failure,
        trials=trials,
        quorum=quorum,
        max_evaluations=max_evaluations,
        max_seconds=max_seconds,
        resume=resume,
    )
    _require_failing(evaluator, current)
    history: list[dict[str, Any]] = resume_history
    while not evaluator.exhausted:
        changed = False
        current_complexity = _complexity(current)
        current_size = _nested_size(current)
        for operation, candidate in _structured_candidates(current):
            if (
                _complexity(candidate) >= current_complexity
                and _nested_size(candidate) >= current_size
            ):
                continue
            accepted = evaluator(candidate)
            history.append(
                _history_entry(operation, accepted, evaluator, size=_complexity(candidate)[0])
            )
            if _accepted(accepted):
                current = candidate
                changed = True
                break
            if evaluator.exhausted:
                break
        if not changed:
            break
    verified = evaluator(current)
    if verified.unresolved or not _accepted(verified):
        if not verified.unresolved:
            raise RuntimeError("shrink predicate was not stable when the final value was verified")
    metadata = {
        "initial_failing": True,
        "verified": _minimality_verified(verified, history),
        "strategy": "structured",
        "original_digest": _candidate_digest(source),
        "resume_used": resume_value is not None,
        **_budget_metadata(history, evaluator.evaluations, max_evaluations),
        **evaluator.metadata(),
        "original_nested_size": _nested_size(value),
        "final_nested_size": _nested_size(current),
    }
    return ShrinkResult(
        current,
        resume_original_size if resume_original_size is not None else _complexity(source)[0],
        _complexity(current)[0],
        evaluator.evaluations,
        history,
        metadata,
    )


def _shrink_text(value: str, predicate: Predicate, **kwargs: Any) -> ShrinkResult:
    tokens = re.findall(r"\S+", value)
    result = ddmin(tokens, lambda items: _evaluate(predicate, " ".join(items)), **kwargs)
    result.value = " ".join(result.value)
    result.metadata.update({"source_type": "str", "strategy": "tokens"})
    return result


def shrink(
    value: Any,
    predicate: Predicate,
    *,
    kind: str = "auto",
    fixture: Any = None,
    checkpoint: Any = None,
    expected_failure: FailureSignature | Mapping[str, Any] | None = None,
    failure: FailureSignature | Mapping[str, Any] | None = None,
    trials: int = 1,
    quorum: int | None = None,
    max_seconds: float | None = None,
    **kwargs: Any,
) -> ShrinkResult:
    """Select a deterministic shrink strategy for common ML inputs."""

    valid_kinds = {"auto", "rows", "columns", "tokens", "sequence", "tensor", "structured"}
    if kind not in valid_kinds:
        raise ValueError(f"unknown shrink kind: {kind}")
    if fixture is not None:
        predicate = _resetting_predicate(predicate, fixture, checkpoint)
    if expected_failure is not None and failure is not None:
        raise ValueError("pass only one of expected_failure or failure")
    expected_failure = expected_failure or failure
    max_evaluations = kwargs.get("max_evaluations")
    strategy_options = {
        "expected_failure": expected_failure,
        "trials": trials,
        "quorum": quorum,
        "max_seconds": max_seconds,
    }
    strategy_options.update(kwargs)
    if isinstance(value, str) and kind in {"auto", "tokens", "sequence"}:
        result = _shrink_text(value, predicate, **strategy_options)
    elif kind == "columns" or (
        kind == "auto"
        and isinstance(value, list)
        and value
        and all(isinstance(item, Mapping) for item in value)
    ):
        result = shrink_columns(value, predicate, **strategy_options)
    elif kind == "structured" or (
        kind == "auto" and (isinstance(value, Mapping) or _walk_tensors(value))
    ):
        result = shrink_structure(value, predicate, **strategy_options)
    elif kind in {"rows", "tokens", "sequence"}:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{kind} shrinking requires a list or tuple")
        result = ddmin(value, predicate, **strategy_options)
    elif kind == "auto" and isinstance(value, (list, tuple)):
        result = ddmin(value, predicate, **strategy_options)
    else:
        result = shrink_tensor(value, predicate, **strategy_options)
    budget = _budget_metadata(result.history, result.evaluations, max_evaluations)
    budget["flakiness"] = bool(budget["flakiness"] or result.metadata.get("flaky_candidates", 0))
    budget["budget_exhausted"] = bool(
        budget["budget_exhausted"] or result.metadata.get("budget_exhausted", False)
    )
    if budget["budget_exhausted"] and not budget["unresolved_evaluations"]:
        budget["minimality"] = "bounded"
    result.metadata.update(budget)
    if expected_failure is not None:
        checked = result.metadata.get("signature_checks", 0)
        result.metadata["signature_preservation"] = (
            "verified"
            if checked and not result.metadata.get("signature_mismatches")
            else "not_observed"
        )
    result.metadata.setdefault("original_nested_size", _nested_size(value))
    result.metadata.setdefault("final_nested_size", _nested_size(result.value))
    return result


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
    if isinstance(replay, Mapping) and replay.get("input") is not None:
        return _artifact_value(capsule, replay["input"])
    plan = capsule.run.replay_plan
    if plan is not None and getattr(plan, "input", None) is not None:
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
    result_failure = _signature_expected(failure) or source.run.failure_signature
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
    fixture: Any = None,
    verify: bool = True,
    state_restorers: Mapping[str, Any] | None = None,
    runner: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> tuple[ShrinkResult, RunCapsule]:
    """Shrink ``evidence.replay.input`` and persist a linked child capsule."""
    source = RunCapsule.load(capsule) if isinstance(capsule, (str, bytes)) else capsule
    if not isinstance(source, RunCapsule):
        raise TypeError("capsule must be a RunCapsule or capsule path")
    value = capsule_replay_input(source)
    checkpoint = None
    replay = source.evidence.get("replay", {})
    if isinstance(replay, Mapping):
        named = replay.get("state", {})
        if isinstance(named, Mapping) and named:
            from .replay import _snapshot_value

            try:
                checkpoint = {
                    str(name): _snapshot_value(source, value) for name, value in named.items()
                }
            except Exception:
                checkpoint = None
    expected_failure = _signature_expected(failure) or source.run.failure_signature
    result = shrink(
        value,
        predicate,
        kind=kind,
        fixture=fixture,
        checkpoint=checkpoint,
        expected_failure=expected_failure,
        **kwargs,
    )
    child = persist_shrink_capsule(source, result, failure=expected_failure, output=output)
    if verify:
        from .replay import ReplayEngine

        loaded = RunCapsule.load(output) if output is not None else child
        execute = runner
        if execute is None and fixture is not None:
            execute = fixture.execute
        restorers = dict(state_restorers or {})
        if execute is not None:
            replayed = ReplayEngine(state_restorers=restorers).replay(loaded, execute)
            result.metadata["child_replay_verified"] = bool(replayed.reproduced)
            result.metadata["child_replay"] = replayed.to_dict()
            if expected_failure is not None and replayed.reproduced:
                result.metadata["signature_preservation"] = "verified"
        elif fixture is not None:
            from .factory import replay_with_factory

            replayed = replay_with_factory(loaded, factory=fixture)
            result.metadata["child_replay_verified"] = bool(replayed.reproduced)
            result.metadata["child_replay"] = replayed.to_dict()
            if expected_failure is not None and replayed.reproduced:
                result.metadata["signature_preservation"] = "verified"
        # The verification result is part of the child evidence as well as the
        # in-memory result; otherwise a saved capsule cannot explain why the
        # minimized input was accepted.
        child_evidence = dict(child.evidence)
        shrink_evidence = dict(child_evidence.get("shrink", {}))
        shrink_evidence["verification"] = {
            "verified": result.metadata.get("child_replay_verified"),
            "replay": result.metadata.get("child_replay"),
        }
        child_evidence["shrink"] = shrink_evidence
        object.__setattr__(child, "evidence", child_evidence)
        if output is not None:
            child.save(output, overwrite=True)
    return result, child
