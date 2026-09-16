"""Deterministic input shrinking for parity counterexamples."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterable, Mapping
from numbers import Number
from typing import Any

try:
    import numpy as _np
except ImportError:  # pragma: no cover - exercised in minimal installations
    _np = None


class ShrinkPreconditionError(ValueError):
    """Raised when the original value does not reproduce a parity failure."""


def _safe_predicate(predicate: Callable[[Any], bool], value: Any, *, initial: bool = False) -> bool:
    try:
        return bool(predicate(value))
    except Exception as exc:
        if initial:
            raise ShrinkPreconditionError(
                "cannot shrink: the predicate raised for the initial value: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        # An invalid reduced input is not the counterexample we are looking for.
        return False


def _numeric_candidates(value: Number) -> Iterable[Number]:
    if isinstance(value, bool):
        if value:
            yield False
        return
    candidates: list[Number] = [type(value)(0) if not isinstance(value, complex) else 0j]
    try:
        candidates.extend([value / 2, type(value)(1 if value > 0 else -1)])
    except (TypeError, ValueError):
        pass
    seen: set[str] = set()
    for candidate in candidates:
        marker = repr(candidate)
        if marker not in seen and candidate != value:
            seen.add(marker)
            yield candidate


def _array_candidates(value: Any) -> Iterable[Any]:
    if value.size == 0:
        return
    yield _np.zeros_like(value)
    try:
        yield value / 2
    except (TypeError, ValueError, FloatingPointError):
        pass
    # Zero progressively smaller chunks. All candidates retain shape and dtype.
    granularity = 2
    while granularity <= min(value.size, 16):
        chunk = max(1, math.ceil(value.size / granularity))
        for start in range(0, value.size, chunk):
            candidate = value.copy()
            candidate.flat[start : start + chunk] = 0
            yield candidate
        granularity *= 2


def _candidates(value: Any) -> Iterable[Any]:
    if isinstance(value, Number):
        yield from _numeric_candidates(value)
        return
    if _np is not None and isinstance(value, _np.ndarray):
        yield from _array_candidates(value)
        return
    if isinstance(value, Mapping):
        keys = list(value)
        if keys:
            yield {}
        for key in keys:
            yield {item: child for item, child in value.items() if item != key}
        for key in keys:
            for shrunk in _candidates(value[key]):
                replacement = dict(value)
                replacement[key] = shrunk
                yield replacement
        return
    if isinstance(value, tuple):
        if value:
            yield ()
        for index in range(len(value)):
            yield value[:index] + value[index + 1 :]
        for index, item in enumerate(value):
            for shrunk in _candidates(item):
                replacement = list(value)
                replacement[index] = shrunk
                yield tuple(replacement)
        return
    if isinstance(value, list):
        if value:
            yield []
        for index in range(len(value)):
            yield value[:index] + value[index + 1 :]
        for index, item in enumerate(value):
            for shrunk in _candidates(item):
                replacement = list(value)
                replacement[index] = shrunk
                yield replacement
        return
    if isinstance(value, str):
        if value:
            yield ""
            words = value.split()
            if len(words) > 1:
                for index in range(len(words)):
                    yield " ".join(words[:index] + words[index + 1 :])


def _call_shrinker(
    shrinker: Any, value: Any, predicate: Callable[[Any], bool], max_steps: int
) -> Any:
    function = getattr(shrinker, "shrink", shrinker)
    if not callable(function):
        raise TypeError("shrinker must be callable or expose shrink(value, predicate)")
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(value, predicate)
    parameters = signature.parameters
    kwargs = {}
    if "max_steps" in parameters:
        kwargs["max_steps"] = max_steps
    if "predicate" in parameters:
        kwargs["predicate"] = predicate
        return function(value, **kwargs)
    return function(value, predicate, **kwargs)


def shrink_input(
    value: Any,
    predicate: Callable[[Any], bool],
    *,
    shrinker: Any = None,
    max_steps: int = 64,
) -> Any:
    """Return a smaller value that demonstrably preserves a parity failure.

    Candidate predicate exceptions are rejected instead of being confused with
    the original mismatch. Custom shrinker output is verified before return.
    """

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    if not _safe_predicate(predicate, value, initial=True):
        raise ShrinkPreconditionError(
            "cannot shrink: the initial value does not satisfy the failure predicate"
        )
    if shrinker is not None:
        candidate = _call_shrinker(shrinker, value, predicate, max_steps)
        if not _safe_predicate(predicate, candidate):
            raise ValueError("custom shrinker returned a value that does not preserve the failure")
        return candidate

    current = value
    steps = 0
    while steps < max_steps:
        changed = False
        for candidate in _candidates(current):
            steps += 1
            if _safe_predicate(predicate, candidate):
                current = candidate
                changed = True
                break
            if steps >= max_steps:
                break
        if not changed:
            break
    if not _safe_predicate(predicate, current):
        raise RuntimeError("shrink predicate was not stable when the result was verified")
    return current


InputShrinker = Callable[[Any, Callable[[Any], bool]], Any]
