"""Protocols and deterministic execution helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RunResult:
    """Normalised result of one seeded experiment run."""

    seed: int | None = None
    ok: bool = True
    metric: float | None = None
    value: Any = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "ok": self.ok,
            "metric": self.metric,
            "value": self.value,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    as_dict = to_dict

    @classmethod
    def from_value(cls, value: Any, seed: int | None = None) -> RunResult:
        if isinstance(value, cls):
            return (
                value
                if value.seed is not None or seed is None
                else cls(
                    seed=seed,
                    ok=value.ok,
                    metric=value.metric,
                    value=value.value,
                    error=value.error,
                    metadata=value.metadata,
                )
            )
        if isinstance(value, Mapping):
            ok = value.get("ok", value.get("success", not value.get("error")))
            return cls(
                seed=seed if value.get("seed") is None else value.get("seed"),
                ok=bool(ok),
                metric=_number(value.get("metric")),
                value=value.get("value", value),
                error=_error_text(value.get("error")),
                metadata=value.get("metadata", {}),
            )
        if isinstance(value, bool):
            return cls(seed=seed, ok=value, value=value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return cls(seed=seed, ok=True, metric=float(value), value=value)
        return cls(seed=seed, ok=True, value=value)


@dataclass(frozen=True)
class SeedRun:
    seed: int
    result: RunResult


@runtime_checkable
class ExperimentRunner(Protocol):
    """A runner can accept a seed as a keyword or positional argument.

    Returning ``RunResult``, a mapping, a boolean, or a numeric metric is
    supported by :func:`run_many`; this keeps adapters small for user code.
    """

    def run(self, seed: int, **kwargs: Any) -> Any: ...


def run_many(
    runner: ExperimentRunner | Callable[..., Any],
    seeds: Iterable[int],
    *,
    kwargs: Mapping[str, Any] | None = None,
    fail_fast: bool = False,
) -> tuple[SeedRun, ...]:
    """Run each seed exactly once, in the supplied order."""
    options = dict(kwargs or {})
    output: list[SeedRun] = []
    for seed in seeds:
        try:
            raw = _call_seeded(runner, seed, options)
            result = RunResult.from_value(raw, seed)
        except Exception as exc:  # runners are experiments; capture failures as data
            result = RunResult(seed=seed, ok=False, error=f"{type(exc).__name__}: {exc}")
            if fail_fast:
                raise
        output.append(SeedRun(seed, result))
    return tuple(output)


def evaluate_runner(*args: Any, **kwargs: Any) -> tuple[SeedRun, ...]:
    """Alias kept as the readable name for callers building reports."""
    return run_many(*args, **kwargs)


def _call_seeded(
    runner: ExperimentRunner | Callable[..., Any], seed: int, options: Mapping[str, Any]
) -> Any:
    """Call positional or keyword-only seed runners without catching body errors."""
    target = runner.run if hasattr(runner, "run") else runner
    try:
        import inspect

        parameters = inspect.signature(target).parameters
        seed_parameter = parameters.get("seed")
        if seed_parameter is not None and seed_parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            return target(seed=seed, **dict(options))
    except (TypeError, ValueError):
        pass
    return target(seed, **dict(options))


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _error_text(value: Any) -> str | None:
    return None if value is None else str(value)
