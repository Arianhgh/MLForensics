"""Forensic comparison of core runs and their metric evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..core import Comparison, EvidenceState, EvidenceStatus, Run, RunCapsule
from .behavior import behavioral_diff
from .performance import performance_diff
from .statistics import (
    BootstrapResult,
    compare_samples,
    effect_size,
    finite_values,
    noninferiority_decision,
    paired_differences,
    regression_decision,
    relative_delta,
)

# Independent repetitions needed before a metric difference can be attributed to
# anything but chance. Below this there is no variation to estimate, so the
# comparison reports the observed delta without a verdict.
MIN_REPLICATES_FOR_INFERENCE = 2


def _load(value: RunCapsule | Run | str | Path) -> tuple[str, Run, RunCapsule | None]:
    if isinstance(value, RunCapsule):
        return value.run.run_id, value.run, value
    if isinstance(value, Run):
        return value.run_id, value, None
    # The append-style capture compatibility capsule intentionally keeps a
    # mutable legacy run object.  Its metrics/resources are structurally the
    # same evidence needed for comparison, so accept it without forcing a
    # conversion just to compute deltas.
    embedded_run = getattr(value, "run", None)
    if embedded_run is not None and hasattr(embedded_run, "run_id"):
        return str(embedded_run.run_id), embedded_run, value
    capsule = RunCapsule.load(value)
    return capsule.run.run_id, capsule.run, capsule


def _load_group(
    value: Any,
) -> tuple[str, list[Run], RunCapsule | None]:
    """Load one side of a comparison, which may be several repeated runs.

    Multiple runs on a side are what make a run-level statistical claim
    possible, so a sequence of runs/capsules/paths is accepted here. The first
    capsule is kept for the evidence (behaviour, parity) attached to a side.
    """
    if isinstance(value, (str, Path)) or not isinstance(value, Sequence):
        identifier, run, capsule = _load(value)
        return identifier, [run], capsule
    items = list(value)
    if not items:
        raise ValueError("a comparison side must contain at least one run")
    loaded = [_load(item) for item in items]
    runs = [run for _identifier, run, _capsule in loaded]
    capsule = next((item for _identifier, _run, item in loaded if item is not None), None)
    identifier = loaded[0][0] if len(loaded) == 1 else f"{loaded[0][0]}+{len(loaded) - 1}"
    return identifier, runs, capsule


def _run_replicate_identity(run: Run, index: int) -> Any:
    """Return the identity that pairs this run with its counterpart."""
    metadata = getattr(run, "metadata", {})
    if isinstance(metadata, Mapping):
        for key in ("seed", "replicate", "trial", "fold", "run_index"):
            if metadata.get(key) is not None:
                return metadata[key]
        configuration = metadata.get("configuration")
        if isinstance(configuration, Mapping) and configuration.get("seed") is not None:
            return configuration["seed"]
    replay_plan = getattr(run, "replay_plan", None)
    plan_seed = getattr(replay_plan, "seed", None)
    if plan_seed is not None:
        return plan_seed
    rng_state = getattr(run, "rng_state", None)
    rng_seed = getattr(rng_state, "seed", None)
    if rng_seed is not None:
        return rng_seed
    return index


def _reduce_run_metric(records: Sequence[Mapping[str, Any]]) -> Any:
    """Collapse one run's within-run metric history to that run's outcome.

    The last finite observation is the result the run actually reached; earlier
    steps describe a model that no longer exists by the end of training.
    """
    finite = [record for record in records if _finite_number(record.get("value"))]
    if finite:
        return finite[-1]["value"]
    return records[-1].get("value") if records else None


def _replicate_records(runs: Sequence[Run], name: str, kind: str) -> list[dict[str, Any]]:
    """Return one record per independent repetition of ``name``.

    A producer that declared replicate identities (seeds, sample or case IDs)
    already provides repetitions, so those are used directly. Otherwise each run
    contributes exactly one repetition, because successive steps within a run
    are not independent repeats of the same measurement.
    """
    records: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        observed = _observation_records(run, name, kind)
        if not observed:
            continue
        if any(record.get("replicate") for record in observed):
            records.extend(observed)
            continue
        identity = _run_replicate_identity(run, index)
        value = _reduce_run_metric(observed)
        records.append(
            {
                "identity": identity,
                "value": value,
                "state": "finite" if _finite_number(value) else "failed",
                "source": "run",
                "replicate": len(runs) > 1,
                "within_run_observations": len(observed),
                # Collapsing a run to its outcome must not hide a NaN that
                # appeared mid-training: that is exactly the evidence a
                # forensics tool exists to surface.
                "nonfinite_within_run": sum(
                    1 for record in observed if not _finite_number(record.get("value"))
                ),
            }
        )
    return records


def _metric_values(run: Run, name: str) -> list[float]:
    metric = next((item for item in run.metrics if item.name == name), None)
    return finite_values(metric.values if metric else [])


def _raw_metric_values(run: Run, name: str) -> list[Any]:
    metric = next((item for item in run.metrics if item.name == name), None)
    return list(metric.values) if metric is not None else []


def _series_values(series: Any) -> list[Any]:
    value = getattr(series, "values", series)
    if value is None or isinstance(value, (str, bytes)):
        return [value] if value is not None else []
    try:
        return list(value)
    except TypeError:
        return [value]


def _identity_token(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)


def _is_replicate_identity(identities: Sequence[Any], series: Any, values: Sequence[Any]) -> bool:
    """Return whether identities name repetitions rather than positions in time.

    Seeds, sample IDs, and case IDs name independent repetitions of the same
    measurement, so they support inference. Capture assigns the step, or a
    running counter, when the producer declared nothing better; identities that
    merely restate the step or the position carry no repetition information.
    """
    ordered = tuple(identities)
    if ordered == tuple(range(len(values))):
        return False
    steps = tuple(getattr(series, "steps", ()) or ())
    return not (steps and ordered == steps)


def _series_ids(series: Any, values: Sequence[Any]) -> tuple[tuple[Any, ...], bool]:
    """Return observation identities and whether they identify true replicates."""
    identities = getattr(series, "observation_ids", None)
    if callable(identities):
        try:
            result = tuple(identities())
            if len(result) == len(values):
                return result, _is_replicate_identity(result, series, values)
        except (TypeError, ValueError):
            pass
    elif identities is not None:
        try:
            result = tuple(identities)
            if len(result) == len(values):
                return result, _is_replicate_identity(result, series, values)
        except TypeError:
            pass
    raw = getattr(series, "identities", ())
    if raw:
        result = tuple(raw)
        if len(result) == len(values):
            return result, _is_replicate_identity(result, series, values)
    metadata = getattr(series, "metadata", {})
    if isinstance(metadata, Mapping):
        for key in ("observation_ids", "sample_ids", "seed_ids", "seeds", "case_ids", "ids"):
            raw = metadata.get(key)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                if len(raw) == len(values):
                    return tuple(raw), True
        if "seed" in metadata and len(values) == 1:
            return (metadata["seed"],), True
    steps = tuple(getattr(series, "steps", ()) or ())
    if len(steps) == len(values) and steps:
        return steps, False
    return tuple(range(len(values))), False


def _observation_records(run: Run, name: str, kind: str) -> list[dict[str, Any]]:
    """Return finite and failed observations without losing their identities."""
    series = next(
        (
            item
            for item in getattr(run, "metrics" if kind == "metric" else "resources", ())
            if getattr(item, "name", None) == name
        ),
        None,
    )
    values = _series_values(series) if series is not None else []
    ids, declared = _series_ids(series, values) if series is not None else ((), False)
    records = [
        {
            "identity": identity,
            "value": value,
            "state": "finite" if _finite_number(value) else "failed",
            "source": "series",
            "replicate": declared,
        }
        for identity, value in zip(ids, values)
    ]
    for observation in getattr(run, "observations", ()):
        if getattr(observation, "name", None) != name:
            continue
        metadata = getattr(observation, "metadata", {})
        observation_kind = metadata.get("kind") if isinstance(metadata, Mapping) else None
        if observation_kind and observation_kind != kind:
            continue
        identity = getattr(observation, "identity", None)
        # An explicit identity marks an independent repetition; a bare step does not.
        replicate = identity is not None
        if identity is None:
            identity = getattr(observation, "step", None)
        if identity is None:
            identity = f"observation:{len(records)}"
        records.append(
            {
                "identity": identity,
                "value": getattr(observation, "numeric_value", None),
                "state": getattr(observation, "state", "failed"),
                "source": "observation",
                "replicate": replicate,
            }
        )
    return records


def _paired_records(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> tuple[
    list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    candidates: dict[str, list[Mapping[str, Any]]] = {}
    for record in candidate:
        candidates.setdefault(_identity_token(record.get("identity")), []).append(record)
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    unmatched_baseline: list[Mapping[str, Any]] = []
    for record in baseline:
        bucket = candidates.get(_identity_token(record.get("identity")))
        if bucket:
            pairs.append((record, bucket.pop(0)))
        else:
            unmatched_baseline.append(record)
    unmatched_candidate = [record for bucket in candidates.values() for record in bucket]
    return pairs, unmatched_baseline, unmatched_candidate


def _series_quality(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    unmatched_baseline: Sequence[Mapping[str, Any]],
    unmatched_candidate: Sequence[Mapping[str, Any]],
) -> tuple[list[float], list[float], dict[str, Any]]:
    old: list[float] = []
    new: list[float] = []
    shared_nonfinite = 0
    for before, after in pairs:
        if _finite_number(before.get("value")) and _finite_number(after.get("value")):
            old.append(float(before["value"]))
            new.append(float(after["value"]))
        else:
            shared_nonfinite += 1
    baseline_nonfinite = sum(not _finite_number(item.get("value")) for item in unmatched_baseline)
    candidate_nonfinite = sum(not _finite_number(item.get("value")) for item in unmatched_candidate)
    baseline_nonfinite += sum(not _finite_number(before.get("value")) for before, _after in pairs)
    candidate_nonfinite += sum(not _finite_number(after.get("value")) for _before, after in pairs)
    # A run reduced to its outcome still reports any non-finite value it contained.
    baseline_nonfinite += sum(
        int(item.get("nonfinite_within_run", 0))
        for item in (*unmatched_baseline, *(before for before, _after in pairs))
    )
    candidate_nonfinite += sum(
        int(item.get("nonfinite_within_run", 0))
        for item in (*unmatched_candidate, *(after for _before, after in pairs))
    )
    quality = {
        "baseline_observations": len(pairs) + len(unmatched_baseline),
        "candidate_observations": len(pairs) + len(unmatched_candidate),
        "shared_observations": len(pairs),
        "paired_finite_observations": len(old),
        "baseline_nonfinite": int(baseline_nonfinite),
        "candidate_nonfinite": int(candidate_nonfinite),
        "shared_nonfinite": shared_nonfinite,
        "unpaired_baseline_observations": len(unmatched_baseline),
        "unpaired_candidate_observations": len(unmatched_candidate),
        "unpaired_observations": len(unmatched_baseline) + len(unmatched_candidate),
        "candidate_missing": bool(unmatched_baseline),
        "baseline_missing": bool(unmatched_candidate),
        "pairing": "stable_identity",
    }
    return old, new, quality


def _required_names(values: Sequence[str] | None, label: str) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{label} must be a sequence of names, not a string")
    result = {str(value) for value in values}
    if any(not value.strip() for value in result):
        raise ValueError(f"{label} must contain non-empty names")
    return result


def _default_higher_is_better(name: str) -> bool:
    """Infer a conservative direction for common metric/resource names."""
    lowered = name.casefold().replace("-", "_")
    lower_is_better = (
        "loss",
        "error",
        "ece",
        "brier",
        "calibration_error",
        "latency",
        "runtime",
        "duration",
        "time",
        "memory",
        "ram",
        "vram",
        "throughput_time",
        "size",
        "cost",
    )
    return not any(token in lowered for token in lower_is_better)


def _failure_count(values: Any) -> int:
    try:
        return sum(1 for value in values if _invalid(value))
    except TypeError:
        return int(_invalid(values))


def _finite_number(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _paired_finite_values(
    baseline: Sequence[Any], candidate: Sequence[Any]
) -> tuple[list[float], list[float]]:
    before: list[float] = []
    after: list[float] = []
    for old, new in zip(baseline, candidate):
        try:
            old_number = float(old)
            new_number = float(new)
        except (TypeError, ValueError):
            continue
        if math.isfinite(old_number) and math.isfinite(new_number):
            before.append(old_number)
            after.append(new_number)
    return before, after


def _known_json_artifact(capsule: RunCapsule | None, names: set[str]) -> Mapping[str, Any] | None:
    if capsule is None:
        return None
    for artifact in capsule.artifacts:
        if Path(artifact.name).name.casefold() not in names or not artifact.sha256:
            continue
        payload = capsule.payloads.get(artifact.sha256)
        if payload is None:
            continue
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(decoded, Mapping):
            return decoded
    return None


def _behavior_evidence(run: Run, capsule: RunCapsule | None) -> Mapping[str, Any] | None:
    for key in ("behavior", "behavioral_evidence", "behavioral_fingerprint"):
        value = run.metadata.get(key)
        if isinstance(value, Mapping):
            return value
    if isinstance(run.metadata.get("predictions"), Sequence):
        return {
            "predictions": run.metadata["predictions"],
            "labels": run.metadata.get("labels"),
            "features": run.metadata.get("features"),
        }
    return _known_json_artifact(
        capsule,
        {
            "behavior.json",
            "behavioral_evidence.json",
            "behavioral_fingerprint.json",
            "predictions.json",
        },
    )


def _configuration(run: Run) -> Mapping[str, Any]:
    for key in ("configuration", "config", "parameters", "hyperparameters"):
        value = run.metadata.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _parity_evidence(run: Run, capsule: RunCapsule | None) -> Any:
    for key in ("parity", "parity_result", "export_parity"):
        if key in run.metadata:
            value = run.metadata[key]
            return value.to_dict() if hasattr(value, "to_dict") else value
    return _known_json_artifact(capsule, {"parity.json", "parity_result.json"})


def _parity_passed(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        for key in ("passed", "equal", "success"):
            if isinstance(value.get(key), bool):
                return bool(value[key])
        status = value.get("status")
        if isinstance(status, str):
            lowered = status.casefold()
            if lowered in {"pass", "passed", "success", "succeeded", "ok"}:
                return True
            if lowered in {"fail", "failed", "failure", "error", "mismatch"}:
                return False
    return None


def _json_safe(value: Any) -> Any:
    """Normalize optional adapter evidence for core Comparison metadata."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


def _invalid(value: Any) -> bool:
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError):
        return True


def _metric_note(
    name: str,
    result: BootstrapResult | None,
    *,
    regression: bool,
    improvement: bool,
    insufficient: bool,
    noninferior: bool = False,
) -> str:
    if result is None:
        return f"{name}: insufficient finite paired observations"
    state = (
        "regressed"
        if regression
        else "improved"
        if improvement
        else "non-inferior"
        if noninferior
        else "no statistically demonstrated change"
    )
    low, high = result.confidence_interval
    return (
        f"{name}: delta={result.estimate:.6g}, {result.confidence:.0%} "
        f"CI=[{low:.6g}, {high:.6g}] ({state})"
    )


def compare_runs(
    baseline: RunCapsule | Run | str | Path,
    candidate: RunCapsule | Run | str | Path,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
    practical_thresholds: Mapping[str, float] | None = None,
    noninferiority_margins: Mapping[str, float] | None = None,
    higher_is_better: Mapping[str, bool] | None = None,
    behavior: tuple[list[Any], list[Any]] | None = None,
    labels: list[Any] | None = None,
    features: list[Mapping[str, Any]] | None = None,
    min_slice_support: int = 5,
    required_metrics: Sequence[str] | None = None,
    required_resources: Sequence[str] | None = None,
    required_evidence: Sequence[str] | None = None,
    min_sample_count: int = 1,
) -> Comparison:
    """Compare two core runs and return the existing core ``Comparison`` record.

    Observations are paired by their stable identity (seed, sample ID, case ID,
    step, or an explicit positional fallback when the producer supplied no
    better identity). Failed/non-finite observations remain in the evidence
    accounting and never shift a later pair. Practical thresholds use metric
    units for metrics, fractional change for resources, and the reserved
    ``behavior:accuracy``/``behavior:calibration`` names for behavioral
    evidence.
    """
    if isinstance(confidence, bool) or not _finite_number(confidence):
        raise ValueError("confidence must be between 0 and 1")
    confidence = float(confidence)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if (
        isinstance(min_slice_support, bool)
        or not isinstance(min_slice_support, int)
        or min_slice_support < 1
    ):
        raise ValueError("min_slice_support must be a positive integer")
    if (
        isinstance(min_sample_count, bool)
        or not isinstance(min_sample_count, int)
        or min_sample_count < 1
    ):
        raise ValueError("min_sample_count must be a positive integer")
    baseline_id, before_runs, before_capsule = _load_group(baseline)
    candidate_id, after_runs, after_capsule = _load_group(candidate)
    before, after = before_runs[0], after_runs[0]
    thresholds = dict(practical_thresholds or {})
    margins = dict(noninferiority_margins or {})
    directions = dict(higher_is_better or {})
    if any(
        not isinstance(name, str) for values in (thresholds, margins, directions) for name in values
    ):
        raise ValueError("threshold, margin, and direction names must be strings")
    for label, values in (("practical thresholds", thresholds), ("margins", margins)):
        if any(not _finite_number(value) for value in values.values()):
            raise ValueError(f"{label} must be finite numbers")
    thresholds = {name: float(value) for name, value in thresholds.items()}
    margins = {name: float(value) for name, value in margins.items()}
    if any(not isinstance(value, bool) for value in directions.values()):
        raise ValueError("higher_is_better values must be booleans")
    required_metric_names = _required_names(required_metrics, "required_metrics")
    required_resource_names = _required_names(required_resources, "required_resources")
    required_evidence_names = _required_names(required_evidence, "required_evidence")
    resource_names = {getattr(item, "name", "") for item in (*before.resources, *after.resources)}
    reserved_evidence = {
        "behavior:accuracy",
        "behavior_accuracy",
        "behavior:calibration",
        "behavior_calibration",
    }
    # A threshold/margin is a contract, not merely an output formatting hint.
    # If a matching resource already exists, require resource evidence; an
    # otherwise unknown configured name is conservatively treated as a metric.
    for configured in (*thresholds, *margins):
        if configured in reserved_evidence:
            continue
        if configured in resource_names:
            required_resource_names.add(configured)
        else:
            required_metric_names.add(configured)
    names = sorted(
        {metric.name for run in before_runs for metric in run.metrics}
        | {metric.name for run in after_runs for metric in run.metrics}
        | required_metric_names
    )
    metric_deltas: dict[str, float] = {}
    evidence: dict[str, Any] = {}
    data_quality: dict[str, Any] = {"metrics": {}, "resources": {}}
    notes: list[str] = []
    regressions: list[str] = []
    decisions: dict[str, EvidenceStatus] = {}
    inconclusive_reasons: list[str] = []
    for index, name in enumerate(names):
        baseline_records = _replicate_records(before_runs, name, "metric")
        candidate_records = _replicate_records(after_runs, name, "metric")
        pairs, unmatched_baseline, unmatched_candidate = _paired_records(
            baseline_records, candidate_records
        )
        old, new, quality = _series_quality(pairs, unmatched_baseline, unmatched_candidate)
        quality["required"] = name in required_metric_names
        quality["replicate_unit"] = (
            "declared"
            if any(record.get("replicate") for record in (*baseline_records, *candidate_records))
            else "run"
        )
        quality["baseline_runs"] = len(before_runs)
        quality["candidate_runs"] = len(after_runs)
        data_quality["metrics"][name] = quality
        baseline_invalid = int(quality["baseline_nonfinite"])
        candidate_invalid = int(quality["candidate_nonfinite"])
        if baseline_records and not candidate_records:
            regressions.append(f"missing_metric:{name}")
            decisions[name] = EvidenceStatus(
                EvidenceState.FAIL.value,
                f"candidate metric {name!r} is absent",
                required_evidence=(f"metric:{name}",),
                observed_evidence=(f"metric:{name}",) if baseline_records else (),
                policy={"required": name in required_metric_names},
                remediation=f"record metric {name!r} for the candidate run",
            )
        elif candidate_invalid:
            regressions.append(f"nonfinite:{name}")
            decisions[name] = EvidenceStatus(
                EvidenceState.FAIL.value,
                f"candidate metric {name!r} contains non-finite or failed observations",
                required_evidence=(f"metric:{name}",),
                observed_evidence=(f"metric:{name}",),
                policy={"required": name in required_metric_names},
                remediation="inspect the captured observation and preserve a finite metric path",
            )
        if not old or not new:
            notes.append(f"{name}: insufficient finite observations")
            if name in required_metric_names or baseline_records or candidate_records:
                inconclusive_reasons.append(f"metric:{name}")
                decisions.setdefault(
                    name,
                    EvidenceStatus(
                        EvidenceState.INCONCLUSIVE.value,
                        f"metric {name!r} has no shared finite observations",
                        required_evidence=(f"metric:{name}",),
                        observed_evidence=(f"metric:{name}",) if candidate_records else (),
                        policy={
                            "minimum_samples": min_sample_count,
                            "required": name in required_metric_names,
                        },
                        remediation="collect matching finite observations for both runs",
                    ),
                )
            continue
        differences = paired_differences(old, new)
        if not differences:
            notes.append(f"{name}: insufficient finite paired observations")
            inconclusive_reasons.append(f"metric:{name}")
            decisions.setdefault(
                name,
                EvidenceStatus(
                    EvidenceState.INCONCLUSIVE.value,
                    f"metric {name!r} has no usable paired observations",
                    required_evidence=(f"metric:{name}",),
                    observed_evidence=(f"metric:{name}",),
                    policy={
                        "minimum_samples": min_sample_count,
                        "required": name in required_metric_names,
                    },
                    remediation="use the same seed/sample IDs and collect finite outcomes",
                ),
            )
            continue
        direction = directions.get(name, _default_higher_is_better(name))
        if len(differences) < MIN_REPLICATES_FOR_INFERENCE:
            # One repetition per side fixes the point estimate but says nothing
            # about run-to-run spread, and resampling it would manufacture a
            # zero-width interval that "proves" any difference. Report the
            # observed delta and withhold the verdict instead.
            observed_delta = (sum(new) / len(new)) - (sum(old) / len(old))
            metric_deltas[name] = observed_delta
            relative = relative_delta(sum(old) / len(old), sum(new) / len(new))
            evidence[name] = {
                "baseline": sum(old) / len(old),
                "candidate": sum(new) / len(new),
                "delta": observed_delta,
                "relative_delta": relative if relative is None or math.isfinite(relative) else None,
                "confidence_interval": None,
                "confidence": None,
                "p_value": None,
                "effect_size": None,
                "sample_size": len(differences),
                "baseline_observations": len(baseline_records),
                "candidate_observations": len(candidate_records),
                "excluded_baseline": baseline_invalid,
                "excluded_candidate": candidate_invalid,
                "practical_threshold": abs(thresholds.get(name, 0.0)),
                "higher_is_better": direction,
                "regression": False,
                "improvement": False,
                "insufficient_evidence": True,
                "replicate_unit": quality["replicate_unit"],
                "replicates": len(differences),
                "minimum_replicates": MIN_REPLICATES_FOR_INFERENCE,
                "noninferior": False,
                "noninferiority_margin": margins.get(name),
            }
            notes.append(
                f"{name}: delta={observed_delta:.6g} from {len(differences)} paired "
                f"repetition(s); at least {MIN_REPLICATES_FOR_INFERENCE} are required before "
                f"run-to-run variation can be estimated"
            )
            inconclusive_reasons.append(f"metric:{name}")
            decisions[name] = EvidenceStatus(
                EvidenceState.INCONCLUSIVE.value,
                f"metric {name!r} has {len(differences)} independent repetition(s)",
                required_evidence=(f"metric:{name}",),
                observed_evidence=(f"metric:{name}",),
                policy={
                    "minimum_replicates": MIN_REPLICATES_FOR_INFERENCE,
                    "replicate_unit": quality["replicate_unit"],
                    "required": name in required_metric_names,
                },
                remediation=(
                    "compare several seeded runs per side, or record observation "
                    "identities (seeds/sample IDs) so repetitions can be paired"
                ),
            )
            continue
        result = compare_samples(
            old,
            new,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed + index,
            paired=True,
        )
        threshold = thresholds.get(name, 0.0)
        regression, improvement, insufficient = regression_decision(
            result.estimate,
            result.confidence_interval,
            higher_is_better=direction,
            practical_threshold=threshold,
        )
        has_noninferiority_margin = name in margins
        noninferior = has_noninferiority_margin and noninferiority_decision(
            result.confidence_interval,
            higher_is_better=direction,
            margin=margins[name],
        )
        metric_deltas[name] = result.estimate
        relative = relative_delta(sum(old) / len(old), sum(new) / len(new))
        effect = effect_size(new, baseline=old)
        evidence[name] = {
            "baseline": sum(old) / len(old),
            "candidate": sum(new) / len(new),
            "delta": result.estimate,
            "relative_delta": relative if relative is None or math.isfinite(relative) else None,
            "confidence_interval": list(result.confidence_interval),
            "confidence": result.confidence,
            "p_value": result.p_value,
            "effect_size": effect if effect is None or math.isfinite(effect) else None,
            "sample_size": result.sample_size,
            "baseline_observations": len(baseline_records),
            "candidate_observations": len(candidate_records),
            "excluded_baseline": baseline_invalid,
            "excluded_candidate": candidate_invalid,
            "practical_threshold": abs(threshold),
            "higher_is_better": direction,
            "regression": regression,
            "improvement": improvement,
            "insufficient_evidence": insufficient,
            "replicate_unit": quality["replicate_unit"],
            "replicates": len(differences),
            "minimum_replicates": MIN_REPLICATES_FOR_INFERENCE,
            "noninferior": noninferior,
            "noninferiority_margin": margins.get(name),
        }
        if regression:
            regressions.append(name)
        if has_noninferiority_margin and not noninferior:
            regressions.append(f"noninferiority:{name}")
        notes.append(
            _metric_note(
                name,
                result,
                regression=regression,
                improvement=improvement,
                insufficient=insufficient,
                noninferior=noninferior,
            )
        )
        failed = baseline_invalid + candidate_invalid
        if failed:
            notes.append(f"{name}: excluded {failed} failed/non-finite observations")
        if len(old) < min_sample_count:
            inconclusive_reasons.append(f"metric:{name}")
            decisions[name] = EvidenceStatus(
                EvidenceState.INCONCLUSIVE.value,
                f"metric {name!r} has only {len(old)} shared finite observations",
                required_evidence=(f"metric:{name}",),
                observed_evidence=(f"metric:{name}",),
                policy={
                    "minimum_samples": min_sample_count,
                    "required": name in required_metric_names,
                },
                remediation=f"collect at least {min_sample_count} matching observations",
            )
        elif candidate_invalid or regression or (has_noninferiority_margin and not noninferior):
            decisions[name] = EvidenceStatus(
                EvidenceState.FAIL.value,
                f"metric {name!r} crossed its configured regression policy",
                required_evidence=(f"metric:{name}",),
                observed_evidence=(f"metric:{name}",),
                policy={
                    "practical_threshold": abs(threshold),
                    "noninferiority_margin": margins.get(name),
                    "higher_is_better": direction,
                },
                remediation="inspect the metric evidence and compare the matched observations",
            )
        else:
            decisions[name] = EvidenceStatus(
                EvidenceState.PASS.value,
                f"metric {name!r} satisfied its configured policy",
                required_evidence=(f"metric:{name}",),
                observed_evidence=(f"metric:{name}",),
                policy={
                    "practical_threshold": abs(threshold),
                    "noninferiority_margin": margins.get(name),
                    "higher_is_better": direction,
                },
            )

    performance_before = {item.name: item for item in before.resources}
    performance_after = {item.name: item for item in after.resources}
    for name in required_resource_names:
        performance_before.setdefault(name, ())
        performance_after.setdefault(name, ())
    performance_result = performance_diff(
        performance_before,
        performance_after,
        thresholds=thresholds,
        higher_is_better=directions,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed + len(names),
        baseline_configuration=_configuration(before),
        candidate_configuration=_configuration(after),
        required_names=required_resource_names,
    )
    resource_units = {item.name: item.units for item in (*before.resources, *after.resources)}
    for item in performance_result.metrics:
        item.units = resource_units.get(item.name)
    resource_deltas = {
        item.name: item.delta for item in performance_result.metrics if item.delta is not None
    }
    for name in performance_result.likely_contributors:
        regressions.append(f"resource:{name}")
    notes.extend(
        (
            f"resource {item.name}: delta={item.delta:.6g}, "
            f"relative {confidence:.0%} CI="
            f"[{item.confidence_interval[0]:.6g}, {item.confidence_interval[1]:.6g}]"
            if item.confidence_interval is not None
            else f"resource {item.name}: delta={item.delta:.6g}; insufficient evidence"
        )
        for item in performance_result.metrics
        if item.delta is not None
    )

    evidence["_performance"] = performance_result.to_dict()
    for item in performance_result.metrics:
        baseline_records = _observation_records(before, item.name, "resource")
        candidate_records = _observation_records(after, item.name, "resource")
        pairs, unmatched_baseline, unmatched_candidate = _paired_records(
            baseline_records, candidate_records
        )
        _old, _new, quality = _series_quality(pairs, unmatched_baseline, unmatched_candidate)
        quality.update(
            {
                "required": item.name in required_resource_names,
                "insufficient_evidence": item.metadata.get("insufficient_evidence", True),
            }
        )
        data_quality["resources"][item.name] = quality
        if baseline_records and not candidate_records:
            regressions.append(f"missing_resource:{item.name}")
        if quality["candidate_nonfinite"]:
            regressions.append(f"nonfinite:resource:{item.name}")
        if quality["paired_finite_observations"] < min_sample_count:
            inconclusive_reasons.append(f"resource:{item.name}")
        if not baseline_records and not candidate_records and item.name in required_resource_names:
            regressions.append(f"missing_resource:{item.name}")
        if quality["candidate_missing"] or quality["candidate_nonfinite"] or item.regression:
            decisions[f"resource:{item.name}"] = EvidenceStatus(
                EvidenceState.FAIL.value,
                f"resource {item.name!r} failed its evidence or regression policy",
                required_evidence=(f"resource:{item.name}",),
                observed_evidence=(f"resource:{item.name}",) if candidate_records else (),
                policy={"required": item.name in required_resource_names},
                remediation=(
                    "collect matching finite resource observations and inspect the resource delta"
                ),
            )
        elif quality["paired_finite_observations"] < min_sample_count:
            decisions[f"resource:{item.name}"] = EvidenceStatus(
                EvidenceState.INCONCLUSIVE.value,
                f"resource {item.name!r} has insufficient shared finite observations",
                required_evidence=(f"resource:{item.name}",),
                observed_evidence=(f"resource:{item.name}",) if candidate_records else (),
                policy={
                    "minimum_samples": min_sample_count,
                    "required": item.name in required_resource_names,
                },
                remediation=f"collect at least {min_sample_count} matching resource observations",
            )
        else:
            decisions[f"resource:{item.name}"] = EvidenceStatus(
                EvidenceState.PASS.value,
                f"resource {item.name!r} satisfied its configured policy",
                required_evidence=(f"resource:{item.name}",),
                observed_evidence=(f"resource:{item.name}",),
                policy={"required": item.name in required_resource_names},
            )
    if performance_result.configuration_changes:
        notes.append(
            "configuration changes observed (context, not causal attribution): "
            + ", ".join(performance_result.configuration_changes)
        )

    behavior_source = "explicit"
    behavior_labels = labels
    behavior_features = features
    if behavior is None:
        old_behavior = _behavior_evidence(before, before_capsule)
        new_behavior = _behavior_evidence(after, after_capsule)
        old_predictions = old_behavior.get("predictions") if old_behavior else None
        new_predictions = new_behavior.get("predictions") if new_behavior else None
        if (
            isinstance(old_predictions, Sequence)
            and not isinstance(old_predictions, (str, bytes))
            and isinstance(new_predictions, Sequence)
            and not isinstance(new_predictions, (str, bytes))
        ):
            behavior = (list(old_predictions), list(new_predictions))
            if behavior_labels is None:
                behavior_labels = new_behavior.get("labels") if new_behavior else None
            if behavior_labels is None:
                behavior_labels = old_behavior.get("labels") if old_behavior else None
            if behavior_features is None:
                behavior_features = new_behavior.get("features") if new_behavior else None
            if behavior_features is None:
                behavior_features = old_behavior.get("features") if old_behavior else None
            behavior_source = "capsule"
    if behavior is not None:
        behavior_threshold = thresholds.get(
            "behavior:accuracy", thresholds.get("behavior_accuracy", 0.0)
        )
        calibration_threshold = thresholds.get(
            "behavior:calibration", thresholds.get("behavior_calibration", 0.0)
        )
        behavior_result = behavioral_diff(
            *behavior,
            labels=behavior_labels,
            features=behavior_features,
            min_support=min_slice_support,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed + len(names) + len(performance_result.metrics),
            practical_threshold=behavior_threshold,
            calibration_threshold=calibration_threshold,
        )
        behavior_dict = behavior_result.to_dict()
        behavior_dict["source"] = behavior_source
        behavior_dict["practical_threshold"] = abs(behavior_threshold)
        behavior_dict["calibration_threshold"] = abs(calibration_threshold)
        evidence["_behavior"] = _json_safe(behavior_dict)
        notes.append(
            f"behavior: {behavior_result.changed_predictions}/"
            f"{behavior_result.sample_count} predictions changed"
        )
        if behavior_result.flips_to_incorrect:
            notes.append(f"behavior: {behavior_result.flips_to_incorrect} flips to incorrect")
        if behavior_result.regression:
            regressions.append("behavior:accuracy")
        if behavior_result.calibration_regression:
            regressions.append("behavior:calibration")
        regressed_slices = [
            item.name for item in behavior_result.slices if item.metadata.get("regression")
        ]
        if regressed_slices:
            regressions.extend(f"behavior:slice:{name}" for name in regressed_slices)
            notes.append("behavior: regressed slices: " + ", ".join(regressed_slices))
        behavior_failed = bool(
            behavior_result.regression or behavior_result.calibration_regression or regressed_slices
        )
        behavior_observations = behavior_result.sample_count
        if behavior_failed:
            decisions["behavior"] = EvidenceStatus(
                EvidenceState.FAIL.value,
                "behavioral evidence crossed an accuracy, calibration, or slice policy",
                required_evidence=("behavior",),
                observed_evidence=("behavior",),
                policy={
                    "minimum_samples": min_sample_count,
                    "accuracy_threshold": abs(behavior_threshold),
                    "calibration_threshold": abs(calibration_threshold),
                },
                remediation=(
                    "inspect divergent examples and confirm the affected slice on held-out data"
                ),
            )
        elif behavior_observations < min_sample_count:
            inconclusive_reasons.append("behavior")
            decisions["behavior"] = EvidenceStatus(
                EvidenceState.INCONCLUSIVE.value,
                "behavioral evidence has insufficient examples",
                required_evidence=("behavior",),
                observed_evidence=("behavior",),
                policy={"minimum_samples": min_sample_count},
                remediation=f"collect at least {min_sample_count} matched examples",
            )
        elif behavior_labels is None:
            decisions["behavior"] = EvidenceStatus(
                EvidenceState.WARN.value,
                "behavioral predictions are present but labels are unavailable",
                required_evidence=("behavior",),
                observed_evidence=("behavior",),
                policy={"minimum_samples": min_sample_count},
                remediation="attach labels or a confirmatory behavioral evaluation",
            )
        else:
            decisions["behavior"] = EvidenceStatus(
                EvidenceState.PASS.value,
                "behavioral evidence satisfied its configured policy",
                required_evidence=("behavior",),
                observed_evidence=("behavior",),
                policy={"minimum_samples": min_sample_count},
            )

    failed_statuses = {"failed", "failure", "error", "crashed", "cancelled", "canceled", "timeout"}
    incomplete_statuses = {"running", "pending", "queued", "created"}
    baseline_status = str(getattr(before, "status", "")).lower()
    candidate_status = str(getattr(after, "status", "")).lower()
    if baseline_status in failed_statuses:
        notes.append(f"baseline run status: {before.status}")
    if candidate_status in failed_statuses:
        notes.append(f"candidate run status: {after.status}")
        regressions.append(f"run:{candidate_status}")
    elif candidate_status in incomplete_statuses:
        notes.append(f"candidate run is incomplete: {after.status}")
        regressions.append(f"run:{candidate_status}")
    evidence["_run_status"] = {
        "baseline": before.status,
        "candidate": after.status,
        "baseline_failed": baseline_status in failed_statuses,
        "candidate_failed": candidate_status in failed_statuses,
        "baseline_incomplete": baseline_status in incomplete_statuses,
        "candidate_incomplete": candidate_status in incomplete_statuses,
        "baseline_failure_signature": (
            before.failure_signature.to_dict() if before.failure_signature is not None else None
        ),
        "candidate_failure_signature": (
            after.failure_signature.to_dict() if after.failure_signature is not None else None
        ),
    }
    if baseline_status in failed_statuses or candidate_status in failed_statuses:
        decisions["run"] = EvidenceStatus(
            EvidenceState.FAIL.value,
            "one or both compared runs did not complete successfully",
            required_evidence=("run_status",),
            observed_evidence=("run_status",),
            policy={
                "baseline_failed": baseline_status in failed_statuses,
                "candidate_failed": candidate_status in failed_statuses,
            },
            remediation=(
                "rerun the comparison with healthy, completed baseline and candidate capsules"
            ),
        )
    elif candidate_status in incomplete_statuses:
        decisions["run"] = EvidenceStatus(
            EvidenceState.INCONCLUSIVE.value,
            "candidate run is incomplete",
            required_evidence=("run_status",),
            observed_evidence=("run_status",),
            policy={"candidate_status": candidate_status},
            remediation="wait for the candidate run to finish before comparing it",
        )
        inconclusive_reasons.append("run")
    else:
        decisions["run"] = EvidenceStatus(
            EvidenceState.PASS.value,
            "both runs supplied a completed run status",
            required_evidence=("run_status",),
            observed_evidence=("run_status",),
        )
    parity = _parity_evidence(after, after_capsule)
    if parity is not None:
        evidence["_parity"] = _json_safe(parity)
        parity_result = _parity_passed(parity)
        if parity_result is False:
            decisions["parity"] = EvidenceStatus(
                EvidenceState.FAIL.value,
                "parity evidence reports a mismatch",
                required_evidence=("parity",),
                observed_evidence=("parity",),
                remediation="inspect the divergent input and backend output paths",
            )
        elif parity_result is None:
            decisions["parity"] = EvidenceStatus(
                EvidenceState.WARN.value,
                "parity evidence is present without a recognized outcome",
                required_evidence=("parity",),
                observed_evidence=("parity",),
                remediation="record an explicit parity pass/fail result",
            )
        else:
            decisions["parity"] = EvidenceStatus(
                EvidenceState.PASS.value,
                "parity evidence reports matching outputs",
                required_evidence=("parity",),
                observed_evidence=("parity",),
            )
    available_evidence = set(evidence)
    for required in required_evidence_names:
        candidates = {required, required.lstrip("_"), f"_{required.lstrip('_')}"}
        if required.startswith("_"):
            candidates.add(required[1:])
        if not candidates.intersection(available_evidence):
            regressions.append(f"missing_evidence:{required}")
            decisions[required] = EvidenceStatus(
                EvidenceState.FAIL.value,
                f"required evidence {required!r} is absent",
                required_evidence=(required,),
                observed_evidence=(),
                policy={"required": True},
                remediation=f"capture and attach {required!r} before running the gate",
            )
    evidence["_requirements"] = {
        "required_metrics": sorted(required_metric_names),
        "required_resources": sorted(required_resource_names),
        "required_evidence": sorted(required_evidence_names),
        "minimum_sample_count": min_sample_count,
    }
    evidence["_data_quality"] = data_quality
    has_comparable_evidence = bool(
        names or performance_result.metrics or behavior is not None or parity is not None
    )
    if not has_comparable_evidence and not regressions:
        inconclusive_reasons.append("no comparable evidence")
    # Preserve order while avoiding duplicate regression labels from overlapping checks.
    regressions = list(dict.fromkeys(regressions))
    if regressions:
        conclusion = "regression detected: " + ", ".join(regressions)
        status = EvidenceState.FAIL.value
    elif inconclusive_reasons or any(
        decision.status == EvidenceState.INCONCLUSIVE.value for decision in decisions.values()
    ):
        conclusion = "comparison inconclusive: " + ", ".join(dict.fromkeys(inconclusive_reasons))
        status = EvidenceState.INCONCLUSIVE.value
    elif any(decision.status == EvidenceState.WARN.value for decision in decisions.values()):
        conclusion = "comparison complete with warnings"
        status = EvidenceState.WARN.value
    else:
        conclusion = "no statistically demonstrated regression"
        status = EvidenceState.PASS.value
    notes.append(conclusion)
    return Comparison(
        left_run_id=baseline_id,
        right_run_id=candidate_id,
        metric_deltas=metric_deltas,
        resource_deltas=resource_deltas,
        regressions=tuple(regressions),
        notes=tuple(notes),
        evidence=evidence,
        status=status,
        decisions=decisions,
    )


compare_capsules = compare_runs


def render_comparison(comparison: Comparison) -> str:
    """Render a core comparison for logs, review comments, or CLI output."""
    lines = [f"Comparison: {comparison.left_run_id} → {comparison.right_run_id}"]
    if comparison.metric_deltas:
        lines.append("Metrics:")
        lines.extend(f"  {name}: Δ {value:.6g}" for name, value in comparison.metric_deltas.items())
    if comparison.resource_deltas:
        lines.append("Resources:")
        lines.extend(
            f"  {name}: Δ {value:.6g}" for name, value in comparison.resource_deltas.items()
        )
    if comparison.regressions:
        lines.append("Regressions: " + ", ".join(comparison.regressions))
    if comparison.notes:
        lines.append("Evidence:")
        lines.extend(f"  {note}" for note in comparison.notes)
    overall = "REGRESSION" if comparison.regressions else comparison.status.upper()
    lines.append("Overall: " + overall)
    return "\n".join(lines)


format_comparison = render_comparison

__all__ = ["compare_capsules", "compare_runs", "format_comparison", "render_comparison"]
