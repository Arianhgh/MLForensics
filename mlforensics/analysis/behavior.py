"""Behavioral model diffs and automatic, lightweight slice discovery."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _finite_number(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_score_vector(value: Any) -> bool:
    """Return whether *value* looks like a vector of class scores.

    Scalar class labels are deliberately not converted to a one-element vector:
    doing that would turn every integer label into class zero.
    """
    if isinstance(value, (str, bytes, Mapping)):
        return False
    if isinstance(value, Sequence):
        return True
    shape = getattr(value, "shape", None)
    return shape is not None and len(shape) > 0


def _argmax(value: Any) -> Any:
    if not _is_score_vector(value):
        return value
    values = _as_list(value)
    if not values:
        return value
    try:
        return max(range(len(values)), key=lambda index: float(values[index]))
    except (TypeError, ValueError):
        return value


def _confidence(value: Any) -> float | None:
    if not _is_score_vector(value):
        return None
    values = _as_list(value)
    if not values:
        return None
    try:
        return float(max(values))
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def _values_differ(before: Any, after: Any) -> bool:
    """Compare score containers without relying on ambiguous array truth values."""
    try:
        result = before != after
        if isinstance(result, bool):
            return result
        all_method = getattr(result, "all", None)
        if callable(all_method):
            return bool(result.any())
    except (TypeError, ValueError):
        pass
    return repr(before) != repr(after)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _bootstrap_mean_interval(
    values: Sequence[float], *, confidence: float, n_resamples: int, seed: int
) -> tuple[float, float] | None:
    if not values:
        return None
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    rng = random.Random(seed)
    # ``choices`` draws a whole resample in one call, which matters because a
    # slice bootstrap runs thousands of resamples over thousands of rows.
    population = list(values)
    size = len(population)
    draw = rng.choices
    distribution = [sum(draw(population, k=size)) / size for _ in range(n_resamples)]
    tail = (1 - confidence) / 2
    return (_quantile(distribution, tail), _quantile(distribution, 1 - tail))


def _wilson_interval(successes: int, total: int, confidence: float) -> tuple[float, float] | None:
    if total == 0:
        return None
    # NormalDist is in the standard library and avoids making scipy a runtime dependency.
    z = statistics.NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total**2))
    radius /= denominator
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def _class_distribution(classes: Sequence[Any]) -> dict[str, float]:
    if not classes:
        return {}
    counts: dict[str, int] = {}
    for value in classes:
        key = repr(value)
        counts[key] = counts.get(key, 0) + 1
    return {key: count / len(classes) for key, count in sorted(counts.items())}


def _calibration_terms(
    predictions: Sequence[Any], labels: Sequence[Any], indices: Sequence[int]
) -> list[tuple[int, float, float]] | None:
    """Return the confidence bin, confidence, and correctness of each row.

    Decoding and validating a prediction vector is the expensive part of a
    calibration estimate, and it does not change between bootstrap resamples,
    so it is done once per row here.
    """
    terms: list[tuple[int, float, float]] = []
    for index in indices:
        prediction = predictions[index]
        if not _is_score_vector(prediction):
            return None
        try:
            scores = [float(value) for value in _as_list(prediction)]
        except (TypeError, ValueError):
            return None
        if (
            not scores
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores)
            or not math.isclose(sum(scores), 1.0, abs_tol=1e-3)
        ):
            return None
        predicted = max(range(len(scores)), key=scores.__getitem__)
        confidence = scores[predicted]
        terms.append((min(9, int(confidence * 10)), confidence, float(predicted == labels[index])))
    return terms


def _ece_from_terms(terms: Sequence[tuple[int, float, float]], indices: Sequence[int]) -> float:
    """Aggregate precomputed calibration terms into an expected calibration error."""
    counts = [0] * 10
    confidences = [0.0] * 10
    correctness = [0.0] * 10
    for index in indices:
        bin_index, confidence, correct = terms[index]
        counts[bin_index] += 1
        confidences[bin_index] += confidence
        correctness[bin_index] += correct
    total = len(indices)
    return sum(
        counts[position]
        / total
        * abs(confidences[position] / counts[position] - correctness[position] / counts[position])
        for position in range(10)
        if counts[position]
    )


def _expected_calibration_error(
    predictions: Sequence[Any], labels: Sequence[Any], indices: Sequence[int] | None = None
) -> float | None:
    selected = list(range(len(predictions))) if indices is None else list(indices)
    if not selected:
        return None
    terms = _calibration_terms(predictions, labels, selected)
    if terms is None:
        return None
    return _ece_from_terms(terms, range(len(terms)))


def _calibration_interval(
    baseline: Sequence[Any],
    candidate: Sequence[Any],
    labels: Sequence[Any],
    *,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> tuple[float, float] | None:
    if not labels:
        return None
    rows = range(len(labels))
    baseline_terms = _calibration_terms(baseline, labels, rows)
    candidate_terms = _calibration_terms(candidate, labels, rows)
    if baseline_terms is None or candidate_terms is None:
        return None
    rng = random.Random(seed)
    population = list(rows)
    size = len(population)
    draw = rng.choices
    differences: list[float] = []
    for _ in range(n_resamples):
        indices = draw(population, k=size)
        differences.append(
            _ece_from_terms(candidate_terms, indices) - _ece_from_terms(baseline_terms, indices)
        )
    tail = (1 - confidence) / 2
    return (_quantile(differences, tail), _quantile(differences, 1 - tail))


@dataclass(slots=True)
class SliceResult:
    name: str
    indices: list[int]
    baseline_score: float | None
    candidate_score: float | None
    delta: float | None
    support: int
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence_interval: tuple[float, float] | None = None
    confidence: float | None = None

    @property
    def regression(self) -> bool:
        if "regression" in self.metadata:
            return bool(self.metadata["regression"])
        return self.delta is not None and self.delta < 0

    def to_dict(self, *, max_example_indices: int = 20) -> dict[str, Any]:
        # A slice can cover millions of rows. The predicate in ``metadata`` and
        # ``support`` fully describe membership, so only a sample of row indices
        # is serialised; embedding all of them would dominate the report.
        return _json_safe(
            {
                "name": self.name,
                "example_indices": self.indices[:max_example_indices],
                "example_indices_truncated": len(self.indices) > max_example_indices,
                "baseline_score": self.baseline_score,
                "candidate_score": self.candidate_score,
                "delta": self.delta,
                "support": self.support,
                "metadata": self.metadata,
                "confidence_interval": (
                    list(self.confidence_interval) if self.confidence_interval is not None else None
                ),
                "confidence": self.confidence,
            }
        )


@dataclass(slots=True)
class BehavioralDiff:
    sample_count: int
    changed_predictions: int
    class_flips: int
    confidence_only_changes: int
    prediction_change_rate: float
    baseline_accuracy: float | None = None
    candidate_accuracy: float | None = None
    slices: list[SliceResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    confidence_delta_mean: float | None = None
    confidence_delta_median: float | None = None
    flips_to_correct: int = 0
    flips_to_incorrect: int = 0
    accuracy_delta: float | None = None
    accuracy_delta_interval: tuple[float, float] | None = None
    prediction_change_interval: tuple[float, float] | None = None
    confidence: float = 0.95
    regression: bool = False
    baseline_class_distribution: dict[str, float] = field(default_factory=dict)
    candidate_class_distribution: dict[str, float] = field(default_factory=dict)
    distribution_total_variation: float = 0.0
    baseline_calibration_error: float | None = None
    candidate_calibration_error: float | None = None
    calibration_delta: float | None = None
    calibration_delta_interval: tuple[float, float] | None = None
    calibration_regression: bool = False
    class_flip_rate: float = 0.0
    total_behavior_changes: int = 0
    total_behavior_change_rate: float = 0.0
    slice_claim: str = "exploratory"

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(
            {
                "sample_count": self.sample_count,
                "changed_predictions": self.changed_predictions,
                "class_flips": self.class_flips,
                "confidence_only_changes": self.confidence_only_changes,
                "prediction_change_rate": self.prediction_change_rate,
                "class_flip_rate": self.class_flip_rate,
                "total_behavior_changes": self.total_behavior_changes,
                "total_behavior_change_rate": self.total_behavior_change_rate,
                "slice_claim": self.slice_claim,
                "baseline_accuracy": self.baseline_accuracy,
                "candidate_accuracy": self.candidate_accuracy,
                "confidence_delta_mean": self.confidence_delta_mean,
                "confidence_delta_median": self.confidence_delta_median,
                "flips_to_correct": self.flips_to_correct,
                "flips_to_incorrect": self.flips_to_incorrect,
                "accuracy_delta": self.accuracy_delta,
                "accuracy_delta_interval": (
                    list(self.accuracy_delta_interval)
                    if self.accuracy_delta_interval is not None
                    else None
                ),
                "prediction_change_interval": (
                    list(self.prediction_change_interval)
                    if self.prediction_change_interval is not None
                    else None
                ),
                "confidence": self.confidence,
                "regression": self.regression,
                "baseline_class_distribution": self.baseline_class_distribution,
                "candidate_class_distribution": self.candidate_class_distribution,
                "distribution_total_variation": self.distribution_total_variation,
                "baseline_calibration_error": self.baseline_calibration_error,
                "candidate_calibration_error": self.candidate_calibration_error,
                "calibration_delta": self.calibration_delta,
                "calibration_delta_interval": (
                    list(self.calibration_delta_interval)
                    if self.calibration_delta_interval is not None
                    else None
                ),
                "calibration_regression": self.calibration_regression,
                "slices": [item.to_dict() for item in self.slices],
                "metadata": self.metadata,
            }
        )

    @property
    def n(self) -> int:
        return self.sample_count

    @property
    def changed(self) -> int:
        return self.changed_predictions

    @property
    def change_rate(self) -> float:
        return self.prediction_change_rate


def _accuracy(predictions: Sequence[Any], labels: Sequence[Any] | None) -> float | None:
    if labels is None or not predictions or len(predictions) != len(labels):
        return None
    return sum(
        _argmax(prediction) == label for prediction, label in zip(predictions, labels)
    ) / len(labels)


def behavioral_diff(
    baseline_predictions: Sequence[Any],
    candidate_predictions: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
    features: Sequence[Mapping[str, Any]] | None = None,
    min_support: int = 5,
    max_slices: int = 20,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
    practical_threshold: float = 0.0,
    calibration_threshold: float = 0.0,
    confirmatory_slices: bool = False,
    heldout_confirmed: bool = False,
    min_slice_support: int | None = None,
) -> BehavioralDiff:
    # ``compare_runs``/``ci_gate`` spell this parameter ``min_slice_support``;
    # accept either so moving between the two APIs is not a TypeError.
    if min_slice_support is not None:
        min_support = min_slice_support
    if isinstance(confidence, bool) or not _finite_number(confidence):
        raise ValueError("confidence must be between 0 and 1")
    confidence = float(confidence)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
        raise ValueError("min_support must be a positive integer")
    if isinstance(max_slices, bool) or not isinstance(max_slices, int) or max_slices < 0:
        raise ValueError("max_slices must be a non-negative integer")
    if not _finite_number(practical_threshold) or not _finite_number(calibration_threshold):
        raise ValueError("behavioral thresholds must be finite")
    practical_threshold = float(practical_threshold)
    calibration_threshold = float(calibration_threshold)
    if len(baseline_predictions) != len(candidate_predictions):
        raise ValueError("behavioral comparison requires equal sample counts")
    n = len(baseline_predictions)
    if labels is not None and len(labels) != n:
        raise ValueError("labels and predictions must have equal lengths")
    if features is not None and len(features) != n:
        raise ValueError("features and predictions must have equal lengths")
    baseline_classes = [_argmax(value) for value in baseline_predictions]
    candidate_classes = [_argmax(value) for value in candidate_predictions]
    baseline_distribution = _class_distribution(baseline_classes)
    candidate_distribution = _class_distribution(candidate_classes)
    distribution_keys = set(baseline_distribution) | set(candidate_distribution)
    changed = [
        index
        for index, (before, after) in enumerate(zip(baseline_classes, candidate_classes))
        if before != after
    ]
    confidence_deltas: list[float] = []
    for before, after in zip(baseline_predictions, candidate_predictions):
        old_confidence = _confidence(before)
        new_confidence = _confidence(after)
        if old_confidence is not None and new_confidence is not None:
            confidence_deltas.append(new_confidence - old_confidence)
    confidence_changes = [
        index
        for index, (before, after) in enumerate(zip(baseline_predictions, candidate_predictions))
        if _is_score_vector(before)
        and _is_score_vector(after)
        and _values_differ(before, after)
        and _argmax(before) == _argmax(after)
    ]
    total_changes = len(set(changed) | set(confidence_changes))
    baseline_accuracy = _accuracy(baseline_predictions, labels)
    candidate_accuracy = _accuracy(candidate_predictions, labels)
    accuracy_differences: list[float] = []
    if labels is not None and len(labels) == n:
        accuracy_differences = [
            float(new == label) - float(old == label)
            for old, new, label in zip(baseline_classes, candidate_classes, labels)
        ]
    accuracy_interval = _bootstrap_mean_interval(
        accuracy_differences,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
    )
    accuracy_delta = (
        candidate_accuracy - baseline_accuracy
        if baseline_accuracy is not None and candidate_accuracy is not None
        else None
    )
    baseline_calibration = (
        _expected_calibration_error(baseline_predictions, labels) if labels is not None else None
    )
    candidate_calibration = (
        _expected_calibration_error(candidate_predictions, labels) if labels is not None else None
    )
    calibration_delta = (
        candidate_calibration - baseline_calibration
        if baseline_calibration is not None and candidate_calibration is not None
        else None
    )
    calibration_interval = (
        _calibration_interval(
            baseline_predictions,
            candidate_predictions,
            labels,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed + 1,
        )
        if labels is not None
        else None
    )
    diff = BehavioralDiff(
        sample_count=n,
        # A behavioral change includes a class flip or a confidence/output
        # change that preserves the predicted class.  ``class_flips`` remains
        # available when callers need the stricter label-level count.
        changed_predictions=total_changes,
        class_flips=len(changed),
        confidence_only_changes=len(confidence_changes),
        prediction_change_rate=(total_changes / n if n else 0.0),
        baseline_accuracy=baseline_accuracy,
        candidate_accuracy=candidate_accuracy,
        confidence_delta_mean=statistics.fmean(confidence_deltas) if confidence_deltas else None,
        confidence_delta_median=statistics.median(confidence_deltas) if confidence_deltas else None,
        accuracy_delta=accuracy_delta,
        accuracy_delta_interval=accuracy_interval,
        prediction_change_interval=_wilson_interval(total_changes, n, confidence),
        confidence=confidence,
        regression=(
            accuracy_interval is not None and accuracy_interval[1] < -abs(practical_threshold)
        ),
        baseline_class_distribution=baseline_distribution,
        candidate_class_distribution=candidate_distribution,
        distribution_total_variation=0.5
        * sum(
            abs(baseline_distribution.get(key, 0.0) - candidate_distribution.get(key, 0.0))
            for key in distribution_keys
        ),
        baseline_calibration_error=baseline_calibration,
        candidate_calibration_error=candidate_calibration,
        calibration_delta=calibration_delta,
        calibration_delta_interval=calibration_interval,
        calibration_regression=(
            calibration_interval is not None
            and calibration_interval[0] > abs(calibration_threshold)
        ),
        class_flip_rate=(len(changed) / n if n else 0.0),
        total_behavior_changes=total_changes,
        total_behavior_change_rate=(total_changes / n if n else 0.0),
        slice_claim="confirmatory" if confirmatory_slices or heldout_confirmed else "exploratory",
    )
    if labels is not None:
        for index in changed:
            if index >= len(labels):
                continue
            old_correct = baseline_classes[index] == labels[index]
            new_correct = candidate_classes[index] == labels[index]
            diff.flips_to_correct += int(not old_correct and new_correct)
            diff.flips_to_incorrect += int(old_correct and not new_correct)
    if features is not None:
        diff.slices = discover_slices(
            baseline_predictions,
            candidate_predictions,
            features,
            labels=labels,
            min_support=min_support,
            max_slices=max_slices,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed,
            practical_threshold=practical_threshold,
            confirmatory=confirmatory_slices,
            heldout_confirmed=heldout_confirmed,
        )
    return diff


def discover_slices(
    baseline_predictions: Sequence[Any],
    candidate_predictions: Sequence[Any],
    features: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[Any] | None = None,
    min_support: int = 5,
    max_slices: int = 20,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
    practical_threshold: float = 0.0,
    confirmatory: bool = True,
    heldout_confirmed: bool = False,
    correction: str = "bonferroni",
    include_intersections: bool = True,
    numeric_bins: int = 4,
    min_slice_support: int | None = None,
) -> list[SliceResult]:
    # ``compare_runs``/``ci_gate`` spell this parameter ``min_slice_support``;
    # accept either so moving between the two APIs is not a TypeError.
    if min_slice_support is not None:
        min_support = min_slice_support
    if isinstance(confidence, bool) or not _finite_number(confidence):
        raise ValueError("confidence must be between 0 and 1")
    confidence = float(confidence)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support < 1:
        raise ValueError("min_support must be a positive integer")
    if isinstance(max_slices, bool) or not isinstance(max_slices, int) or max_slices < 0:
        raise ValueError("max_slices must be a non-negative integer")
    if not _finite_number(practical_threshold):
        raise ValueError("practical_threshold must be finite")
    if correction not in {"none", "bonferroni", "holm"}:
        raise ValueError("correction must be one of: none, bonferroni, holm")
    if isinstance(numeric_bins, bool) or not isinstance(numeric_bins, int) or numeric_bins < 1:
        raise ValueError("numeric_bins must be a positive integer")
    practical_threshold = float(practical_threshold)
    if len(baseline_predictions) != len(candidate_predictions) or len(features) != len(
        baseline_predictions
    ):
        raise ValueError("predictions and features must have equal lengths")
    labels = list(labels) if labels is not None else None
    if labels is not None and len(labels) != len(baseline_predictions):
        raise ValueError("labels and predictions must have equal lengths")
    candidates: list[SliceResult] = []
    # Every candidate slice, and then every bootstrap resample of it, needs the
    # predicted class of the same rows. Decoding each prediction vector once
    # keeps slice discovery linear in the number of predictions.
    baseline_classes = [_argmax(item) for item in baseline_predictions]
    candidate_classes = [_argmax(item) for item in candidate_predictions]
    baseline_correct = (
        [float(predicted == label) for predicted, label in zip(baseline_classes, labels)]
        if labels is not None
        else None
    )
    candidate_correct = (
        [float(predicted == label) for predicted, label in zip(candidate_classes, labels)]
        if labels is not None
        else None
    )
    columns = sorted({str(key) for row in features for key in row})
    for column in columns:
        values = [row.get(column) for row in features]
        unique = []
        for value in values:
            if value not in unique and len(unique) < 50:
                unique.append(value)
        for value in unique:
            indices = [index for index, item in enumerate(values) if item == value]
            if len(indices) < min_support:
                continue
            if labels is None:
                baseline_score = None
                candidate_score = None
                changed = sum(
                    baseline_classes[index] != candidate_classes[index] for index in indices
                )
                delta = -(changed / len(indices)) if changed else 0.0
            else:
                baseline_score = statistics.fmean(baseline_correct[index] for index in indices)
                candidate_score = statistics.fmean(candidate_correct[index] for index in indices)
                delta = candidate_score - baseline_score
            candidates.append(
                SliceResult(
                    f"{column}={value!r}",
                    indices,
                    baseline_score,
                    candidate_score,
                    delta,
                    len(indices),
                    {
                        "feature": column,
                        "value": value,
                        "regression": False,
                    },
                )
            )
        numeric = [
            float(value)
            for value in values
            if not isinstance(value, bool) and _finite_number(value)
        ]
        if len(numeric) >= min_support and len(set(numeric)) > min(4, numeric_bins):
            ordered = sorted(set(numeric))
            cuts = [
                ordered[
                    min(len(ordered) - 1, max(0, round(index * (len(ordered) - 1) / numeric_bins)))
                ]
                for index in range(1, numeric_bins)
            ]
            bounds = [ordered[0], *sorted(set(cuts)), ordered[-1]]
            for lower, upper in zip(bounds, bounds[1:]):
                indices = [
                    index
                    for index, item in enumerate(values)
                    if _finite_number(item)
                    and float(item) >= lower
                    and (float(item) <= upper if upper == bounds[-1] else float(item) < upper)
                ]
                if len(indices) < min_support:
                    continue
                if labels is None:
                    baseline_score = candidate_score = None
                    delta = -(
                        sum(
                            baseline_classes[index] != candidate_classes[index] for index in indices
                        )
                        / len(indices)
                    )
                else:
                    baseline_score = statistics.fmean(baseline_correct[i] for i in indices)
                    candidate_score = statistics.fmean(candidate_correct[i] for i in indices)
                    delta = candidate_score - baseline_score
                candidates.append(
                    SliceResult(
                        f"{column} in [{lower!r}, {upper!r}]",
                        indices,
                        baseline_score,
                        candidate_score,
                        delta,
                        len(indices),
                        {
                            "feature": column,
                            "lower": lower,
                            "upper": upper,
                            "slice_type": "numeric",
                        },
                    )
                )

    if include_intersections and len(columns) > 1:
        for left_index, left in enumerate(columns):
            for right in columns[left_index + 1 :]:
                left_values = [row.get(left) for row in features]
                right_values = [row.get(right) for row in features]
                combinations = []
                for pair in zip(left_values, right_values):
                    if pair not in combinations and len(combinations) < 50:
                        combinations.append(pair)
                for left_value, right_value in combinations:
                    indices = [
                        index
                        for index, pair in enumerate(zip(left_values, right_values))
                        if pair == (left_value, right_value)
                    ]
                    if len(indices) < min_support:
                        continue
                    if labels is None:
                        old_score = new_score = None
                        delta = -(
                            sum(
                                baseline_classes[index] != candidate_classes[index]
                                for index in indices
                            )
                            / len(indices)
                        )
                    else:
                        old_score = statistics.fmean(baseline_correct[i] for i in indices)
                        new_score = statistics.fmean(candidate_correct[i] for i in indices)
                        delta = new_score - old_score
                    candidates.append(
                        SliceResult(
                            f"{left}={left_value!r} & {right}={right_value!r}",
                            indices,
                            old_score,
                            new_score,
                            delta,
                            len(indices),
                            {"features": [left, right], "slice_type": "intersection"},
                        )
                    )
    candidates.sort(key=lambda item: (item.delta if item.delta is not None else 0.0, -item.support))
    selected = candidates[:max_slices]
    if labels is not None:
        tests = max(1, len(candidates))
        adjusted_confidence = confidence
        if correction != "none":
            adjusted_confidence = 1.0 - (1.0 - confidence) / tests
        for index, item in enumerate(selected):
            differences = [candidate_correct[row] - baseline_correct[row] for row in item.indices]
            item.confidence_interval = _bootstrap_mean_interval(
                differences,
                confidence=adjusted_confidence,
                n_resamples=n_resamples,
                seed=seed + index,
            )
            item.confidence = confidence
            signal = bool(
                item.confidence_interval is not None
                and item.confidence_interval[1] < -abs(practical_threshold)
            )
            item.metadata.update(
                {
                    "regression": bool(signal and (confirmatory or heldout_confirmed)),
                    "exploratory_signal": signal,
                    "claim_type": "confirmatory"
                    if confirmatory or heldout_confirmed
                    else "exploratory",
                    "multiple_testing": correction,
                    "tests": tests,
                    "corrected_confidence": adjusted_confidence,
                    "heldout_confirmed": heldout_confirmed,
                }
            )
    return selected
