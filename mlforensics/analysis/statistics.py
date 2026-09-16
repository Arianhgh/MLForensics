"""Small, deterministic statistical primitives used by comparison and CI."""

from __future__ import annotations

import json
import math
import random
import statistics as _statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any


def _as_list(values: Any) -> list[Any]:
    if values is None or isinstance(values, (str, bytes)):
        return [values]
    try:
        return list(values)
    except TypeError:
        return [values]


def finite_values(values: Iterable[float] | Any) -> list[float]:
    clean: list[float] = []
    for value in _as_list(values):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    return clean


def _identity_token(value: Any) -> str:
    """Return a deterministic token for a pairing identity.

    Identities are normally strings, integers, seeds, or sample IDs, but JSON
    input specs also permit structured IDs.  A canonical token lets the
    pairing code handle all of those without requiring them to be hashable.
    """

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)


def paired_observations(
    baseline: Sequence[Any] | Iterable[Any],
    candidate: Sequence[Any] | Iterable[Any],
    *,
    baseline_ids: Sequence[Any] | None = None,
    candidate_ids: Sequence[Any] | None = None,
    strict: bool = False,
) -> list[tuple[Any, Any, Any]]:
    """Pair observations by identity while retaining failed values in place.

    When identities are omitted, the historical positional design is used.
    Once identities are supplied, only equal identities are paired; values
    with an identity present on one side are intentionally not matched to a
    different observation on the other side.  Duplicate identities are
    paired in occurrence order and remain visible to callers through the
    returned identity value.
    """

    before = _as_list(baseline)
    after = _as_list(candidate)
    old_ids = list(range(len(before))) if baseline_ids is None else list(baseline_ids)
    new_ids = list(range(len(after))) if candidate_ids is None else list(candidate_ids)
    if len(old_ids) != len(before) or len(new_ids) != len(after):
        raise ValueError("observation identities must have the same length as their values")
    if strict and baseline_ids is not None and candidate_ids is not None:
        old_keys = {_identity_token(value) for value in old_ids}
        new_keys = {_identity_token(value) for value in new_ids}
        if old_keys != new_keys:
            raise ValueError("strict paired samples must contain the same observation identities")
    candidates: dict[str, list[tuple[Any, Any]]] = {}
    for identity, value in zip(new_ids, after):
        candidates.setdefault(_identity_token(identity), []).append((identity, value))
    paired: list[tuple[Any, Any, Any]] = []
    for identity, value in zip(old_ids, before):
        bucket = candidates.get(_identity_token(identity))
        if bucket:
            candidate_identity, candidate_value = bucket.pop(0)
            # Return the baseline identity: it is the stable key callers used
            # to construct the pair and it is less surprising for mixed types.
            del candidate_identity
            paired.append((identity, value, candidate_value))
    return paired


def mean(values: Sequence[float]) -> float | None:
    return _statistics.fmean(values) if values else None


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _validate_sampling(confidence: float, n_resamples: int) -> float:
    try:
        valid_confidence = not isinstance(confidence, bool) and 0 < float(confidence) < 1
    except (TypeError, ValueError):
        valid_confidence = False
    if not valid_confidence:
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    return float(confidence)


@dataclass(slots=True)
class BootstrapResult:
    estimate: float
    confidence_interval: tuple[float, float]
    confidence: float
    resamples: int
    sample_size: int
    p_value: float | None = None
    distribution: list[float] = field(default_factory=list, repr=False)

    @property
    def lower(self) -> float:
        return self.confidence_interval[0]

    @property
    def upper(self) -> float:
        return self.confidence_interval[1]

    def to_dict(self) -> dict[str, Any]:
        def number(value: float | None) -> float | None:
            if value is None:
                return None
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value

        return {
            "estimate": number(self.estimate),
            "confidence_interval": [number(item) for item in self.confidence_interval],
            "confidence": self.confidence,
            "resamples": self.resamples,
            "sample_size": self.sample_size,
            "p_value": number(self.p_value),
        }


def bootstrap(
    values: Sequence[float],
    statistic: Callable[[Sequence[float]], float] = mean,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
) -> BootstrapResult:
    """Non-parametric bootstrap for a one-sample statistic."""

    clean = finite_values(values)
    if not clean:
        raise ValueError("bootstrap requires at least one finite observation")
    confidence = _validate_sampling(confidence, n_resamples)
    rng = random.Random(seed)
    estimate = float(statistic(clean))
    distribution = [
        float(statistic([clean[rng.randrange(len(clean))] for _ in clean]))
        for _ in range(n_resamples)
    ]
    tail = (1.0 - confidence) / 2.0
    return BootstrapResult(
        estimate,
        (_quantile(distribution, tail), _quantile(distribution, 1 - tail)),
        confidence,
        n_resamples,
        len(clean),
        distribution=distribution,
    )


def paired_differences(
    baseline: Sequence[float] | Iterable[float],
    candidate: Sequence[float] | Iterable[float],
    *,
    strict: bool = False,
    baseline_ids: Sequence[Any] | None = None,
    candidate_ids: Sequence[Any] | None = None,
) -> list[float]:
    """Return candidate-minus-baseline differences for matched observations.

    Failed/non-finite observations drop only their pair. Unequal lengths are
    tolerated by default.  If identities are supplied, unequal or missing
    identities are not shifted into a different pair; callers needing strict
    experimental design can pass ``strict=True``.
    """
    differences: list[float] = []
    if strict and baseline_ids is None and candidate_ids is None:
        before = _as_list(baseline)
        after = _as_list(candidate)
        if len(before) != len(after):
            raise ValueError("paired samples must have equal lengths")
    for _identity, old, new in paired_observations(
        baseline,
        candidate,
        baseline_ids=baseline_ids,
        candidate_ids=candidate_ids,
        strict=strict,
    ):
        try:
            delta = float(new) - float(old)
        except (TypeError, ValueError):
            continue
        if math.isfinite(delta):
            differences.append(delta)
    return differences


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
    baseline_ids: Sequence[Any] | None = None,
    candidate_ids: Sequence[Any] | None = None,
) -> BootstrapResult:
    differences = paired_differences(
        baseline,
        candidate,
        baseline_ids=baseline_ids,
        candidate_ids=candidate_ids,
    )
    result = bootstrap(differences, confidence=confidence, n_resamples=n_resamples, seed=seed)
    if result.distribution:
        beyond_zero = sum(1 for value in result.distribution if value <= 0) / len(
            result.distribution
        )
        result.p_value = min(1.0, 2.0 * min(beyond_zero, 1.0 - beyond_zero))
    return result


def effect_size(
    values: Sequence[float], *, baseline: Sequence[float] | None = None
) -> float | None:
    clean = finite_values(values)
    if baseline is not None:
        clean = paired_differences(baseline, values)
    if not clean:
        return None
    deviation = _statistics.stdev(clean) if len(clean) > 1 else 0.0
    if deviation == 0:
        return 0.0 if mean(clean) == 0 else math.copysign(float("inf"), mean(clean))
    return float(mean(clean) / deviation)


def relative_delta(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None:
        return None
    if baseline == 0:
        return None if candidate == 0 else math.copysign(float("inf"), candidate)
    return (candidate - baseline) / abs(baseline)


def regression_decision(
    delta: float | None,
    interval: tuple[float, float] | None,
    *,
    higher_is_better: bool = True,
    practical_threshold: float = 0.0,
) -> tuple[bool, bool, bool]:
    """Return ``(regression, improvement, insufficient_evidence)``.

    A result is a regression only when the whole confidence interval exceeds
    the practical threshold in the harmful direction. A point estimate alone
    therefore never fails a statistical gate.
    """

    if delta is None or interval is None:
        return False, False, True
    if higher_is_better:
        regression = interval[1] < -abs(practical_threshold)
        improvement = interval[0] > abs(practical_threshold)
    else:
        regression = interval[0] > abs(practical_threshold)
        improvement = interval[1] < -abs(practical_threshold)
    insufficient = not regression and not improvement
    return regression, improvement, insufficient


def noninferiority_decision(
    interval: tuple[float, float] | None,
    *,
    higher_is_better: bool = True,
    margin: float = 0.0,
) -> bool:
    """Return whether the confidence interval rules out harm beyond ``margin``."""
    if interval is None:
        return False
    bound = interval[0] if higher_is_better else -interval[1]
    return bound >= -abs(margin)


def compare_samples(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
    paired: bool = True,
    baseline_ids: Sequence[Any] | None = None,
    candidate_ids: Sequence[Any] | None = None,
) -> BootstrapResult:
    confidence = _validate_sampling(confidence, n_resamples)
    if paired:
        return paired_bootstrap(
            baseline,
            candidate,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
            baseline_ids=baseline_ids,
            candidate_ids=candidate_ids,
        )
    before = finite_values(baseline)
    after = finite_values(candidate)
    if not before or not after:
        raise ValueError("both samples must contain a finite observation")
    rng = random.Random(seed)
    observed = _statistics.fmean(after) - _statistics.fmean(before)
    distribution: list[float] = []
    for _ in range(n_resamples):
        b = [_statistics.fmean([before[rng.randrange(len(before))] for _ in before])]
        c = [_statistics.fmean([after[rng.randrange(len(after))] for _ in after])]
        distribution.append(c[0] - b[0])
    tail = (1 - confidence) / 2
    result = BootstrapResult(
        observed,
        (_quantile(distribution, tail), _quantile(distribution, 1 - tail)),
        confidence,
        n_resamples,
        min(len(before), len(after)),
        distribution=distribution,
    )
    beyond_zero = sum(value <= 0 for value in distribution) / len(distribution)
    result.p_value = min(1.0, 2 * min(beyond_zero, 1 - beyond_zero))
    return result
