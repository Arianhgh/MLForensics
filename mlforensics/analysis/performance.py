"""Resource and latency regression analysis."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# Paired observations required before a resource interval is estimated. A single
# measurement per side carries no information about run-to-run variability.
MIN_RESOURCE_OBSERVATIONS = 2


def _finite_number(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _default_higher_is_better(name: str) -> bool:
    normalized = name.casefold().replace("-", "_")
    return any(
        token in normalized
        for token in ("throughput", "utilization", "bandwidth", "samples_per", "items_per")
    )


@dataclass(slots=True)
class PerformanceMetric:
    name: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    relative_delta: float | None
    units: str | None = None
    regression: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence_interval: tuple[float, float] | None = None
    confidence: float | None = None
    sample_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "delta": self.delta,
            "relative_delta": self.relative_delta,
            "units": self.units,
            "regression": self.regression,
            "metadata": self.metadata,
            "confidence_interval": (
                list(self.confidence_interval) if self.confidence_interval is not None else None
            ),
            "confidence": self.confidence,
            "sample_size": self.sample_size,
        }

    @property
    def is_regression(self) -> bool:
        return self.regression


@dataclass(slots=True)
class PerformanceDiff:
    metrics: list[PerformanceMetric] = field(default_factory=list)
    likely_contributors: list[str] = field(default_factory=list)
    configuration_changes: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": [item.to_dict() for item in self.metrics],
            "likely_contributors": self.likely_contributors,
            "configuration_changes": self.configuration_changes,
        }

    @property
    def has_regression(self) -> bool:
        return bool(self.likely_contributors)

    @property
    def regressions(self) -> tuple[str, ...]:
        return tuple(self.likely_contributors)

    def metric(self, name: str) -> PerformanceMetric | None:
        return next((item for item in self.metrics if item.name == name), None)


def _raw_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "values"):
        value = value.values
    if isinstance(value, (str, bytes)):
        try:
            return [float(value)]
        except ValueError:
            return []
    try:
        return list(value)
    except TypeError:
        try:
            return [value]
        except (TypeError, ValueError):
            return []


def _values(value: Any) -> list[float]:
    values: list[float] = []
    for item in _raw_values(value):
        if isinstance(item, bool):
            continue
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return values


def _identity_token(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)


def _sample_ids(value: Any, count: int) -> tuple[Any, ...]:
    explicit = getattr(value, "observation_ids", None)
    if explicit is not None:
        try:
            ids = tuple(explicit() if callable(explicit) else explicit)
            if len(ids) == count:
                return ids
        except (TypeError, ValueError):
            pass
    raw = getattr(value, "identities", ())
    if raw:
        ids = tuple(raw)
        if len(ids) == count:
            return ids
    metadata = getattr(value, "metadata", {})
    if isinstance(metadata, Mapping):
        for key in ("observation_ids", "sample_ids", "seed_ids", "seeds", "case_ids", "ids"):
            ids = metadata.get(key)
            if isinstance(ids, (list, tuple)) and len(ids) == count:
                return tuple(ids)
        if "seed" in metadata and count == 1:
            return (metadata["seed"],)
    steps = getattr(value, "steps", ())
    if steps and len(steps) == count:
        return tuple(steps)
    return tuple(range(count))


def _aligned_finite_samples(
    baseline: Any, candidate: Any
) -> tuple[list[float], list[float], dict[str, int]]:
    old_raw = _raw_values(baseline)
    new_raw = _raw_values(candidate)
    old_ids = _sample_ids(baseline, len(old_raw))
    new_ids = _sample_ids(candidate, len(new_raw))
    buckets: dict[str, list[Any]] = {}
    for identity, value in zip(new_ids, new_raw):
        buckets.setdefault(_identity_token(identity), []).append(value)
    old: list[float] = []
    new: list[float] = []
    shared = 0
    for identity, old_value in zip(old_ids, old_raw):
        bucket = buckets.get(_identity_token(identity))
        if not bucket:
            continue
        new_value = bucket.pop(0)
        shared += 1
        try:
            old_number, new_number = float(old_value), float(new_value)
        except (TypeError, ValueError):
            continue
        if not isinstance(old_value, bool) and not isinstance(new_value, bool):
            if math.isfinite(old_number) and math.isfinite(new_number):
                old.append(old_number)
                new.append(new_number)
    return (
        old,
        new,
        {
            "baseline_observations": len(old_raw),
            "candidate_observations": len(new_raw),
            "shared_observations": shared,
            "unpaired_baseline_observations": len(old_raw) - shared,
            "unpaired_candidate_observations": len(new_raw) - shared,
            "baseline_nonfinite": sum(not _finite_number(v) for v in old_raw),
            "candidate_nonfinite": sum(not _finite_number(v) for v in new_raw),
        },
    )


def _summary(value: Any, aggregation: str = "mean") -> float | None:
    values = _values(value)
    if not values:
        return None
    if aggregation == "p95":
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]
    if aggregation == "max":
        return max(values)
    if aggregation == "min":
        return min(values)
    return statistics.fmean(values)


def _config_changes(
    baseline: Mapping[str, Any] | None, candidate: Mapping[str, Any] | None
) -> dict[str, dict[str, Any]]:
    """Return changed configuration leaves without claiming causality."""

    def flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(item, Mapping):
                result.update(flatten(item, path))
            else:
                result[path] = item
        return result

    before = flatten(baseline or {})
    after = flatten(candidate or {})
    missing = object()
    changes: dict[str, dict[str, Any]] = {}
    for name in sorted(set(before) | set(after)):
        old = before.get(name, missing)
        new = after.get(name, missing)
        if old != new:
            changes[name] = {
                "baseline": None if old is missing else old,
                "candidate": None if new is missing else new,
            }
    return changes


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_delta_interval(
    before: list[float],
    after: list[float],
    *,
    aggregation: str,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    paired = len(before) == len(after)
    distribution: list[float] = []
    for _ in range(n_resamples):
        if paired:
            indices = [rng.randrange(len(before)) for _ in before]
            old_sample = [before[index] for index in indices]
            new_sample = [after[index] for index in indices]
        else:
            old_sample = [before[rng.randrange(len(before))] for _ in before]
            new_sample = [after[rng.randrange(len(after))] for _ in after]
        old_summary = _summary(old_sample, aggregation)
        new_summary = _summary(new_sample, aggregation)
        if old_summary is not None and new_summary is not None:
            distribution.append(new_summary - old_summary)
    tail = (1 - confidence) / 2
    return (_percentile(distribution, tail), _percentile(distribution, 1 - tail))


def performance_diff(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float] | None = None,
    aggregations: Mapping[str, str] | None = None,
    higher_is_better: Mapping[str, bool] | None = None,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
    baseline_configuration: Mapping[str, Any] | None = None,
    candidate_configuration: Mapping[str, Any] | None = None,
    required_names: Sequence[str] | None = None,
    min_observations: int = MIN_RESOURCE_OBSERVATIONS,
) -> PerformanceDiff:
    """Compare resource samples with uncertainty-aware relative thresholds.

    Thresholds are fractional changes (``0.2`` means 20%). Configuration
    changes are reported as context only; they are not asserted to be causal.

    A resource needs at least ``min_observations`` paired observations before an
    interval is estimated. Below that, a regression is reported only against an
    explicitly configured threshold, because one sample per side says nothing
    about run-to-run variability.
    """
    if isinstance(confidence, bool) or not _finite_number(confidence):
        raise ValueError("confidence must be between 0 and 1")
    confidence = float(confidence)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if (
        isinstance(min_observations, bool)
        or not isinstance(min_observations, int)
        or min_observations < 1
    ):
        raise ValueError("min_observations must be a positive integer")
    minimum_observations = int(min_observations)
    thresholds = dict(thresholds or {})
    aggregations = dict(aggregations or {})
    higher_is_better = dict(higher_is_better or {})
    mappings = (baseline, candidate, thresholds, aggregations, higher_is_better)
    if any(not isinstance(name, str) for mapping in mappings for name in mapping):
        raise ValueError("performance evidence names must be strings")
    if any(not _finite_number(value) for value in thresholds.values()):
        raise ValueError("performance thresholds must be finite numbers")
    thresholds = {name: float(value) for name, value in thresholds.items()}
    if any(not isinstance(value, bool) for value in higher_is_better.values()):
        raise ValueError("higher_is_better values must be booleans")
    if required_names is None:
        required_names = ()
    if isinstance(required_names, (str, bytes)):
        raise TypeError("required_names must be a sequence of names")
    required_names = tuple(str(name) for name in required_names)
    if any(not name.strip() for name in required_names):
        raise ValueError("required_names must contain non-empty names")
    allowed_aggregations = {"mean", "p95", "min", "max"}
    unknown_aggregations = set(aggregations.values()) - allowed_aggregations
    if unknown_aggregations:
        raise ValueError(f"unsupported aggregations: {sorted(unknown_aggregations)!r}")
    result = PerformanceDiff(
        configuration_changes=_config_changes(baseline_configuration, candidate_configuration)
    )
    for index, name in enumerate(sorted(set(baseline) | set(candidate) | set(required_names))):
        baseline_value = baseline.get(name)
        candidate_value = candidate.get(name)
        old_values, new_values, pairing = _aligned_finite_samples(baseline_value, candidate_value)
        # Summaries are computed from every finite observation on each side;
        # inference below uses only the stable-identity intersection.
        old_summary_values = _values(baseline_value)
        new_summary_values = _values(candidate_value)
        before = _summary(old_summary_values, aggregations.get(name, "mean"))
        after = _summary(new_summary_values, aggregations.get(name, "mean"))
        delta = None if before is None or after is None else after - before
        relative = None if delta is None or before in (None, 0) else delta / abs(before)
        configured = name in thresholds
        limit = thresholds.get(name, 0.0)
        beneficial = higher_is_better.get(name, _default_higher_is_better(name))
        interval: tuple[float, float] | None = None
        # Resampling a single observation cannot describe variability: it yields a
        # zero-width interval that excludes zero for any non-zero delta, which
        # would report every run-to-run wobble as a significant regression.
        if min(len(old_values), len(new_values)) >= minimum_observations and before not in (
            None,
            0,
        ):
            delta_interval = _bootstrap_delta_interval(
                old_values,
                new_values,
                aggregation=aggregations.get(name, "mean"),
                confidence=confidence,
                n_resamples=n_resamples,
                seed=seed + index,
            )
            interval = tuple(value / abs(before) for value in delta_interval)
        if interval is not None:
            evidence_mode = "interval"
            regression = bool(interval[1] < -abs(limit) if beneficial else interval[0] > abs(limit))
        elif configured and relative is not None:
            # Too few observations to infer, but the caller declared an explicit
            # budget, so compare the point estimate against that budget instead.
            evidence_mode = "threshold"
            regression = relative < -abs(limit) if beneficial else relative > abs(limit)
        else:
            evidence_mode = "none"
            regression = False
        item = PerformanceMetric(
            name,
            before,
            after,
            delta,
            relative,
            regression=regression,
            metadata={
                "threshold": abs(limit),
                "threshold_mode": "relative",
                "threshold_configured": configured,
                "evidence_mode": evidence_mode,
                "higher_is_better": beneficial,
                "baseline_samples": len(old_values),
                "candidate_samples": len(new_values),
                "minimum_observations": minimum_observations,
                "insufficient_evidence": evidence_mode == "none",
                "pairing": "stable_identity",
                **pairing,
            },
            confidence_interval=interval,
            confidence=confidence if interval is not None else None,
            sample_size=min(len(old_values), len(new_values)),
        )
        result.metrics.append(item)
        if regression:
            result.likely_contributors.append(name)
    return result


compare_performance = performance_diff
performance_delta = performance_diff
