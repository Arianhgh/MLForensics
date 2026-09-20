"""Forensic comparison of core runs and their metric evidence."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


_RUN_IDENTITY_KEYS = (
    "seed",
    "sample_id",
    "case_id",
    "replicate",
    "trial",
    "fold",
    "dataset_version",
    "configuration_id",
    "run_index",
)
_IDENTITY_DISCRIMINATOR_KEYS = (
    "sample_id",
    "case_id",
    "replicate",
    "trial",
    "fold",
    "dataset_version",
    "configuration_id",
    "run_index",
)


def _load_group(
    value: Any,
) -> tuple[str, list[Run], list[RunCapsule | None]]:
    """Load one side of a comparison, which may be several repeated runs.

    Multiple runs on a side are what make a run-level statistical claim
    possible, so a sequence of runs/capsules/paths is accepted here. Every
    member is retained: run health, parity, and behavior can fail on any of
    them, not only the first.
    """
    if isinstance(value, (str, Path)) or not isinstance(value, Sequence):
        identifier, run, capsule = _load(value)
        return identifier, [run], [capsule]
    items = list(value)
    if not items:
        raise ValueError("a comparison side must contain at least one run")
    loaded = [_load(item) for item in items]
    runs = [run for _identifier, run, _capsule in loaded]
    capsules = [capsule for _identifier, _run, capsule in loaded]
    identifier = loaded[0][0] if len(loaded) == 1 else f"{loaded[0][0]}+{len(loaded) - 1}"
    return identifier, runs, capsules


def _declared_run_identity_parts(run: Run) -> dict[str, Any]:
    """Return producer-declared pairing keys; never invent a list position."""
    parts: dict[str, Any] = {}
    metadata = getattr(run, "metadata", {}) or {}
    if isinstance(metadata, Mapping):
        for key in _RUN_IDENTITY_KEYS:
            if metadata.get(key) is not None:
                parts[key] = metadata[key]
        configuration = metadata.get("configuration")
        if "seed" not in parts and isinstance(configuration, Mapping):
            if configuration.get("seed") is not None:
                parts["seed"] = configuration["seed"]
    if "seed" not in parts:
        replay_plan = getattr(run, "replay_plan", None)
        plan_seed = getattr(replay_plan, "seed", None)
        if plan_seed is not None:
            parts["seed"] = plan_seed
    if "seed" not in parts:
        rng_state = getattr(run, "rng_state", None)
        rng_seed = getattr(rng_state, "seed", None)
        if rng_seed is not None:
            parts["seed"] = rng_seed
    return parts


@dataclass(frozen=True, slots=True)
class _GroupMember:
    run: Run
    capsule: RunCapsule | None
    identity: Any
    pairing: str
    pairable: bool
    run_id: str


@dataclass(frozen=True, slots=True)
class _AlignedSamples:
    """A finite sample with the identities that established its alignment."""

    values: tuple[float, ...]
    identities: tuple[Any, ...]


def _group_members(
    runs: Sequence[Run], capsules: Sequence[RunCapsule | None] | None = None
) -> list[_GroupMember]:
    """Deduplicate capsules and assign pairing identities without using position."""
    aligned: list[tuple[Run, RunCapsule | None]] = []
    seen_ids: set[str] = set()
    capsule_list: Sequence[RunCapsule | None] = capsules or (None,) * len(runs)
    for index, run in enumerate(runs):
        capsule = capsule_list[index] if index < len(capsule_list) else None
        run_id = str(getattr(run, "run_id", "") or "")
        if run_id and run_id in seen_ids:
            continue
        if run_id:
            seen_ids.add(run_id)
        aligned.append((run, capsule))
    seed_counts: Counter[str] = Counter()
    parts_by_run = [_declared_run_identity_parts(run) for run, _capsule in aligned]
    for parts in parts_by_run:
        if "seed" in parts:
            seed_counts[_identity_token(parts["seed"])] += 1
    members: list[_GroupMember] = []
    pending: list[tuple[Run, RunCapsule | None, Any, str]] = []
    for (run, capsule), parts in zip(aligned, parts_by_run):
        run_id = str(getattr(run, "run_id", "") or f"run:{len(pending)}")
        if not parts:
            pending.append((run, capsule, ("unpaired", run_id), "missing"))
            continue
        trial_keys = tuple(key for key in _IDENTITY_DISCRIMINATOR_KEYS if key in parts)
        seed = parts.get("seed")
        if seed is not None and seed_counts[_identity_token(seed)] > 1:
            if trial_keys:
                identity = (seed, *(parts[key] for key in trial_keys))
                pending.append((run, capsule, identity, "composite"))
            else:
                pending.append((run, capsule, ("ambiguous", seed, run_id), "ambiguous"))
            continue
        if len(parts) == 1:
            identity = next(iter(parts.values()))
        else:
            identity = tuple(parts[key] for key in _RUN_IDENTITY_KEYS if key in parts)
        pending.append((run, capsule, identity, "declared"))
    identity_counts = Counter(
        _identity_token(identity)
        for _run, _capsule, identity, kind in pending
        if kind in {"declared", "composite"}
    )
    for run, capsule, identity, kind in pending:
        run_id = str(getattr(run, "run_id", "") or "")
        pairable = kind in {"declared", "composite"}
        resolved_kind = kind
        resolved_identity = identity
        if pairable and identity_counts[_identity_token(identity)] > 1:
            pairable = False
            resolved_kind = "ambiguous"
            resolved_identity = ("ambiguous", identity, run_id)
        members.append(
            _GroupMember(
                run=run,
                capsule=capsule,
                identity=resolved_identity,
                pairing=resolved_kind,
                pairable=pairable,
                run_id=run_id,
            )
        )
    return members


def _run_replicate_identity(run: Run, index: int) -> Any:
    """Return a declared pairing identity, or ``None`` when none was supplied."""
    parts = _declared_run_identity_parts(run)
    if not parts:
        return None
    if len(parts) == 1:
        return next(iter(parts.values()))
    return tuple(parts[key] for key in _RUN_IDENTITY_KEYS if key in parts)


def _reduce_run_metric(records: Sequence[Mapping[str, Any]]) -> Any:
    """Collapse one run's within-run metric history to that run's outcome.

    The last finite observation is the result the run actually reached; earlier
    steps describe a model that no longer exists by the end of training.
    """
    finite = [record for record in records if _finite_number(record.get("value"))]
    if finite:
        return finite[-1]["value"]
    return records[-1].get("value") if records else None


def _replicate_records(
    runs: Sequence[Run] | Sequence[_GroupMember], name: str, kind: str
) -> list[dict[str, Any]]:
    """Return one record per independent repetition of ``name``.

    A producer that declared replicate identities (seeds, sample or case IDs)
    already provides repetitions, so those are used directly. Otherwise each run
    contributes exactly one repetition, because successive steps within a run
    are not independent repeats of the same measurement. List position is never
    a pairing identity: runs without a declared identity stay unpaired.
    """
    members: list[_GroupMember]
    if runs and isinstance(runs[0], _GroupMember):
        members = list(runs)  # type: ignore[arg-type]
    else:
        members = _group_members(runs)  # type: ignore[arg-type]
    records: list[dict[str, Any]] = []
    for member in members:
        run = member.run
        observed = _observation_records(run, name, kind)
        if not observed:
            continue
        if any(record.get("replicate") for record in observed):
            observed = [dict(record) for record in observed]
            for record in observed:
                record.setdefault("run_id", member.run_id)
                record.setdefault("pairing", "declared")
                if len(members) > 1:
                    if member.pairable:
                        record["identity"] = (member.identity, record.get("identity"))
                        record["pairable"] = record.get("pairable", True)
                    else:
                        record["pairing"] = member.pairing
                        record["pairable"] = False
            records.extend(observed)
            continue
        value = _reduce_run_metric(observed)
        records.append(
            {
                "identity": member.identity,
                "value": value,
                "state": "finite" if _finite_number(value) else "failed",
                "source": "run",
                "replicate": member.pairable,
                "pairing": member.pairing,
                "pairable": member.pairable,
                "run_id": member.run_id,
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
    raw = getattr(series, "identities", ())
    if raw:
        result = tuple(raw)
        if len(result) == len(values):
            metadata = getattr(series, "metadata", {})
            if isinstance(metadata, Mapping) and metadata.get("_mlforensics_identity_kind") in {
                "position",
                "step",
                "mixed",
            }:
                return result, False
            return result, True
    metadata = getattr(series, "metadata", {})
    if isinstance(metadata, Mapping):
        for key in (
            "observation_ids",
            "sample_ids",
            "seed_ids",
            "seeds",
            "case_ids",
            "ids",
        ):
            raw = metadata.get(key)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                if len(raw) == len(values):
                    return tuple(raw), True
        if "seed" in metadata and len(values) == 1:
            return (metadata["seed"],), True
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
    if not hasattr(series, "identities"):
        explicit = getattr(series, "observation_ids", None)
        if explicit is not None:
            try:
                result = tuple(explicit() if callable(explicit) else explicit)
                if len(result) == len(values):
                    return result, _is_replicate_identity(result, series, values)
            except (TypeError, ValueError):
                pass
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
    identity_tokens = [_identity_token(identity) for identity in ids]
    duplicate_identities = declared and len(set(identity_tokens)) != len(identity_tokens)
    series_pairing = "ambiguous" if duplicate_identities else "declared" if declared else "missing"
    records = [
        {
            "identity": identity,
            "value": value,
            "state": "finite" if _finite_number(value) else "failed",
            "source": "series",
            "replicate": declared,
            "pairing": series_pairing,
            "pairable": declared and not duplicate_identities,
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
        if kind == "resource" and observation_kind is None:
            continue
        identity = getattr(observation, "identity", None)
        # An explicit identity marks an independent repetition; a bare step does not.
        # Capture uses ``identity == step`` for failed points from a time
        # series. Treat those as within-run observations so they cannot turn a
        # training history into artificial repetitions.
        step = getattr(observation, "step", None)
        replicate = identity is not None and not (step is not None and identity == step)
        pairing = "declared" if replicate else "missing"
        if identity is None:
            identity = step
        if identity is None:
            identity = f"observation:{len(records)}"
        observation_state = str(getattr(observation, "state", "failed")).casefold()
        records.append(
            {
                "identity": identity,
                "value": (
                    getattr(observation, "numeric_value", None)
                    if observation_state == "finite"
                    else None
                ),
                "state": observation_state,
                "source": "observation",
                "replicate": replicate,
                "pairing": pairing,
                "pairable": replicate,
            }
        )
    explicit_records = [record for record in records if record.get("replicate")]
    if explicit_records:
        tokens = [_identity_token(record.get("identity")) for record in explicit_records]
        if len(set(tokens)) != len(tokens):
            for record in explicit_records:
                record["pairing"] = "ambiguous"
                record["pairable"] = False
    return records


def _paired_records(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]]
) -> tuple[
    list[tuple[Mapping[str, Any], Mapping[str, Any]]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    def pairable(record: Mapping[str, Any]) -> bool:
        if record.get("pairable") is False:
            return False
        return record.get("pairing") not in {"missing", "ambiguous"}

    candidates: dict[str, list[Mapping[str, Any]]] = {}
    unmatched_candidate: list[Mapping[str, Any]] = []
    for record in candidate:
        if not pairable(record):
            unmatched_candidate.append(record)
            continue
        candidates.setdefault(_identity_token(record.get("identity")), []).append(record)
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    unmatched_baseline: list[Mapping[str, Any]] = []
    for record in baseline:
        if not pairable(record):
            unmatched_baseline.append(record)
            continue
        bucket = candidates.get(_identity_token(record.get("identity")))
        if bucket:
            pairs.append((record, bucket.pop(0)))
        else:
            unmatched_baseline.append(record)
    unmatched_candidate.extend(record for bucket in candidates.values() for record in bucket)
    # One observation per side is an explicit unpaired design, not positional
    # pairing of a larger group. Ambiguous identities still cannot pair.
    if (
        not pairs
        and len(baseline) == 1
        and len(candidate) == 1
        and baseline[0].get("pairing") != "ambiguous"
        and candidate[0].get("pairing") != "ambiguous"
    ):
        return [(baseline[0], candidate[0])], [], []
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

    # A run reduced to its outcome still reports any non-finite value it
    # contained, without counting the reduced non-finite outcome twice.
    def add_hidden_nonfinite(records: Sequence[Mapping[str, Any]], current: int) -> int:
        return current + sum(
            max(
                0,
                int(item.get("nonfinite_within_run", 0))
                - int(not _finite_number(item.get("value"))),
            )
            for item in records
        )

    baseline_nonfinite = add_hidden_nonfinite(
        (*unmatched_baseline, *(before for before, _after in pairs)), baseline_nonfinite
    )
    candidate_nonfinite = add_hidden_nonfinite(
        (*unmatched_candidate, *(after for _before, after in pairs)), candidate_nonfinite
    )
    baseline_count = len(pairs) + len(unmatched_baseline)
    candidate_count = len(pairs) + len(unmatched_candidate)

    def has_pairable(records: Sequence[Mapping[str, Any]]) -> bool:
        return any(
            record.get("pairable") is not False
            and record.get("pairing") not in {"missing", "ambiguous"}
            for record in records
        )

    # An identity mismatch is missing evidence only when the other side
    # supplied a pairable identity. A wholly unpaired or ambiguous sample is
    # present evidence with an invalid design, which must be reported as
    # insufficient rather than mislabeled as absent.
    candidate_missing = candidate_count == 0 or (
        bool(unmatched_baseline)
        and has_pairable((*unmatched_candidate, *(after for _before, after in pairs)))
    )
    baseline_missing = baseline_count == 0 or (
        bool(unmatched_candidate)
        and has_pairable((*unmatched_baseline, *(before for before, _after in pairs)))
    )
    quality = {
        "baseline_observations": baseline_count,
        "candidate_observations": candidate_count,
        "shared_observations": len(pairs),
        "paired_finite_observations": len(old),
        "baseline_nonfinite": int(baseline_nonfinite),
        "candidate_nonfinite": int(candidate_nonfinite),
        "shared_nonfinite": shared_nonfinite,
        "unpaired_baseline_observations": len(unmatched_baseline),
        "unpaired_candidate_observations": len(unmatched_candidate),
        "unpaired_observations": len(unmatched_baseline) + len(unmatched_candidate),
        "candidate_missing": candidate_missing,
        "baseline_missing": baseline_missing,
        "pairing": _pairing_label(pairs, unmatched_baseline, unmatched_candidate),
    }
    return old, new, quality


def _aligned_samples_from_pairs(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[_AlignedSamples, _AlignedSamples]:
    old: list[float] = []
    new: list[float] = []
    identities: list[Any] = []
    for before, after in pairs:
        if _finite_number(before.get("value")) and _finite_number(after.get("value")):
            old.append(float(before["value"]))
            new.append(float(after["value"]))
            identities.append(before.get("identity"))
    return (
        _AlignedSamples(tuple(old), tuple(identities)),
        _AlignedSamples(tuple(new), tuple(identities)),
    )


def _resource_input(member: _GroupMember, name: str) -> Any:
    """Build performance input without dropping observation-only failures."""
    series = next(
        (
            item
            for item in getattr(member.run, "resources", ())
            if getattr(item, "name", None) == name
        ),
        None,
    )
    observations = [
        item
        for item in _observation_records(member.run, name, "resource")
        if item.get("source") == "observation"
    ]
    if not observations:
        return series if series is not None else _AlignedSamples((), ())
    records = _observation_records(member.run, name, "resource")
    finite = [record for record in records if _finite_number(record.get("value"))]
    values = tuple(float(record["value"]) for record in finite)
    if finite and all(record.get("pairing") == "declared" for record in finite):
        identities = tuple(record.get("identity") for record in finite)
    else:
        identities = ()
    return _AlignedSamples(values, identities)


def _pairing_label(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    unmatched_baseline: Sequence[Mapping[str, Any]],
    unmatched_candidate: Sequence[Mapping[str, Any]],
) -> str:
    kinds = {
        str(record.get("pairing") or "")
        for record in (
            *(before for before, _after in pairs),
            *(after for _before, after in pairs),
            *unmatched_baseline,
            *unmatched_candidate,
        )
        if record.get("pairing")
    }
    if "ambiguous" in kinds:
        return "ambiguous"
    if "missing" in kinds:
        return "unpaired"
    if pairs:
        return "stable_identity"
    return "unpaired"


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


def _observation_names(members: Sequence[_GroupMember], kind: str) -> set[str]:
    names: set[str] = set()
    for member in members:
        for observation in getattr(member.run, "observations", ()):
            metadata = getattr(observation, "metadata", {})
            observation_kind = metadata.get("kind") if isinstance(metadata, Mapping) else None
            if observation_kind == kind or (kind == "metric" and observation_kind is None):
                name = getattr(observation, "name", None)
                if isinstance(name, str) and name:
                    names.add(name)
    return names


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


_FAILED_STATUSES = {
    "aborted",
    "failed",
    "failure",
    "error",
    "crashed",
    "hung",
    "killed",
    "oom",
    "oom_killed",
    "out_of_memory",
    "cancelled",
    "canceled",
    "timeout",
    "timed_out",
}
_INCOMPLETE_STATUSES = {"running", "pending", "queued", "created"}


def _member_health(member: _GroupMember) -> dict[str, Any]:
    status = str(getattr(member.run, "status", "")).strip().lower()
    signature = getattr(member.run, "failure_signature", None)
    return {
        "run_id": member.run_id,
        "identity": member.identity,
        "status": getattr(member.run, "status", None),
        "failed": status in _FAILED_STATUSES,
        "incomplete": status in _INCOMPLETE_STATUSES,
        "failure_signature": signature.to_dict() if signature is not None else None,
        "pairing": member.pairing,
    }


def _group_health(members: Sequence[_GroupMember], *, side: str) -> dict[str, Any]:
    records = [_member_health(member) for member in members]
    return {
        side: records[0]["status"] if records else None,
        f"{side}_failed": any(item["failed"] for item in records),
        f"{side}_incomplete": any(item["incomplete"] for item in records),
        f"{side}_failure_signature": records[0]["failure_signature"] if records else None,
        f"{side}_runs": records,
        f"{side}_run_ids": [item["run_id"] for item in records],
        f"{side}_failed_run_ids": [item["run_id"] for item in records if item["failed"]],
    }


def _member_parity(member: _GroupMember) -> dict[str, Any] | None:
    evidence = _parity_evidence(member.run, member.capsule)
    if evidence is None:
        return None
    return {
        "run_id": member.run_id,
        "identity": member.identity,
        "passed": _parity_passed(evidence),
        "evidence": evidence,
    }


def _group_parity(members: Sequence[_GroupMember]) -> dict[str, Any] | None:
    records = [item for item in (_member_parity(member) for member in members) if item is not None]
    if not records:
        return None
    outcomes = [item["passed"] for item in records]
    if any(item is False for item in outcomes):
        passed: bool | None = False
    elif any(item is None for item in outcomes):
        passed = None
    else:
        passed = True
    representative = next((item["evidence"] for item in records if item["passed"] is False), None)
    if representative is None:
        representative = records[0]["evidence"]
    if isinstance(representative, Mapping):
        summary: dict[str, Any] = dict(representative)
    else:
        summary = {"value": representative}
    summary["passed"] = passed
    summary["runs"] = records
    summary["failed_run_ids"] = [item["run_id"] for item in records if item["passed"] is False]
    return summary


def _pair_members(
    baseline: Sequence[_GroupMember], candidate: Sequence[_GroupMember]
) -> list[tuple[_GroupMember, _GroupMember]]:
    """Pair group members by declared identity.

    A one-run-per-side comparison is always paired with itself. Larger groups
    require explicit identities; list position is not a pairing key.
    """
    if len(baseline) == 1 and len(candidate) == 1:
        return [(baseline[0], candidate[0])]
    buckets: dict[str, list[_GroupMember]] = {}
    for member in candidate:
        if member.pairable:
            buckets.setdefault(_identity_token(member.identity), []).append(member)
    pairs: list[tuple[_GroupMember, _GroupMember]] = []
    for member in baseline:
        if not member.pairable:
            continue
        bucket = buckets.get(_identity_token(member.identity))
        if bucket:
            pairs.append((member, bucket.pop(0)))
    return pairs


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
    or an explicit trial discriminator). List position is never a pairing key.
    Duplicate capsules and duplicate seeds without a trial identity cannot
    inflate the sample size; those comparisons stay inconclusive. Failed and
    non-finite observations remain in the evidence accounting and never shift a
    later pair. Practical thresholds use metric units for metrics, fractional
    change for resources, and the reserved ``behavior:accuracy``/
    ``behavior:calibration`` names for behavioral evidence.
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
    baseline_id, before_runs, before_capsules = _load_group(baseline)
    candidate_id, after_runs, after_capsules = _load_group(candidate)
    before_members = _group_members(before_runs, before_capsules)
    after_members = _group_members(after_runs, after_capsules)
    before, after = before_members[0].run, after_members[0].run
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
    if any(value < 0 for value in (*thresholds.values(), *margins.values())):
        raise ValueError("thresholds and non-inferiority margins must be non-negative")
    if any(not isinstance(value, bool) for value in directions.values()):
        raise ValueError("higher_is_better values must be booleans")
    required_metric_names = _required_names(required_metrics, "required_metrics")
    required_resource_names = _required_names(required_resources, "required_resources")
    required_evidence_names = _required_names(required_evidence, "required_evidence")
    resource_names = {
        getattr(item, "name", "")
        for member in (*before_members, *after_members)
        for item in member.run.resources
    }
    resource_names.update(_observation_names((*before_members, *after_members), "resource"))
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
        {metric.name for member in before_members for metric in member.run.metrics}
        | {metric.name for member in after_members for metric in member.run.metrics}
        | _observation_names(before_members, "metric")
        | _observation_names(after_members, "metric")
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
        baseline_records = _replicate_records(before_members, name, "metric")
        candidate_records = _replicate_records(after_members, name, "metric")
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
        quality["baseline_runs"] = len(before_members)
        quality["candidate_runs"] = len(after_members)
        quality["baseline_run_ids"] = [member.run_id for member in before_members]
        quality["candidate_run_ids"] = [member.run_id for member in after_members]
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
        if quality.get("pairing") == "ambiguous":
            notes.append(
                f"{name}: pairing is ambiguous; duplicate seeds need a trial/replicate identity"
            )
            inconclusive_reasons.append(f"metric:{name}")
            decisions.setdefault(
                name,
                EvidenceStatus(
                    EvidenceState.INCONCLUSIVE.value,
                    f"metric {name!r} cannot be paired: duplicate identities without a trial key",
                    required_evidence=(f"metric:{name}",),
                    observed_evidence=(f"metric:{name}",) if candidate_records else (),
                    policy={
                        "pairing": "ambiguous",
                        "required": name in required_metric_names,
                    },
                    remediation=(
                        "give each repetition a unique seed or a composite trial identity "
                        "(trial/replicate/fold); do not submit the same capsule twice"
                    ),
                ),
            )
            continue
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
        matched_ids = tuple(range(len(old)))
        differences = paired_differences(
            old,
            new,
            baseline_ids=matched_ids,
            candidate_ids=matched_ids,
        )
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
            # ``_paired_records`` has already matched these values by the
            # producer's identity. Mark that established alignment explicitly
            # so the statistical primitive never invents positional pairing.
            baseline_ids=tuple(range(len(old))),
            candidate_ids=tuple(range(len(new))),
        )
        threshold = thresholds.get(name, 0.0)
        raw_regression, raw_improvement, raw_insufficient = regression_decision(
            result.estimate,
            result.confidence_interval,
            higher_is_better=direction,
            practical_threshold=threshold,
        )
        enough_samples = len(old) >= min_sample_count
        regression = raw_regression if enough_samples else False
        improvement = raw_improvement if enough_samples else False
        insufficient = raw_insufficient or not enough_samples
        has_noninferiority_margin = name in margins
        noninferior = (
            has_noninferiority_margin
            and enough_samples
            and noninferiority_decision(
                result.confidence_interval,
                higher_is_better=direction,
                margin=margins[name],
            )
        )
        metric_deltas[name] = result.estimate
        relative = relative_delta(sum(old) / len(old), sum(new) / len(new))
        effect = effect_size(differences)
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
        if enough_samples and regression:
            regressions.append(name)
        if enough_samples and has_noninferiority_margin and not noninferior:
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
        if candidate_invalid:
            decisions[name] = EvidenceStatus(
                EvidenceState.FAIL.value,
                f"metric {name!r} contains non-finite or failed observations",
                required_evidence=(f"metric:{name}",),
                observed_evidence=(f"metric:{name}",),
                policy={"required": name in required_metric_names},
                remediation="preserve a finite candidate metric path and inspect the failure",
            )
        elif len(old) < min_sample_count:
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
        elif regression or (enough_samples and has_noninferiority_margin and not noninferior):
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

    grouped_resources = len(before_members) > 1 or len(after_members) > 1
    if grouped_resources:
        performance_before = {}
        performance_after = {}
        for name in sorted(resource_names | required_resource_names):
            baseline_records = _replicate_records(before_members, name, "resource")
            candidate_records = _replicate_records(after_members, name, "resource")
            pairs, unmatched_baseline, unmatched_candidate = _paired_records(
                baseline_records, candidate_records
            )
            old_values, new_values = _aligned_samples_from_pairs(pairs)
            performance_before[name] = old_values
            performance_after[name] = new_values
    else:
        performance_before = {
            name: _resource_input(before_members[0], name) for name in resource_names
        }
        performance_after = {
            name: _resource_input(after_members[0], name) for name in resource_names
        }
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
    resource_units = {
        item.name: item.units
        for member in (*before_members, *after_members)
        for item in member.run.resources
    }
    for item in performance_result.metrics:
        item.units = resource_units.get(item.name)
    resource_deltas = {
        item.name: item.delta for item in performance_result.metrics if item.delta is not None
    }
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
        if grouped_resources:
            baseline_records = _replicate_records(before_members, item.name, "resource")
            candidate_records = _replicate_records(after_members, item.name, "resource")
        else:
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
                "effective_samples": (
                    item.sample_size
                    if item.metadata.get("pairing") == "independent"
                    else quality["paired_finite_observations"]
                ),
            }
        )
        effective_samples = int(quality["effective_samples"])
        enough_samples = effective_samples >= min_sample_count
        data_quality["resources"][item.name] = quality
        if baseline_records and not candidate_records:
            regressions.append(f"missing_resource:{item.name}")
        if quality["candidate_nonfinite"]:
            regressions.append(f"nonfinite:resource:{item.name}")
        if not enough_samples:
            inconclusive_reasons.append(f"resource:{item.name}")
        if item.regression and enough_samples:
            regressions.append(f"resource:{item.name}")
        if not baseline_records and not candidate_records and item.name in required_resource_names:
            regressions.append(f"missing_resource:{item.name}")
        if quality["candidate_missing"] or quality["candidate_nonfinite"]:
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
        elif item.regression and enough_samples:
            decisions[f"resource:{item.name}"] = EvidenceStatus(
                EvidenceState.FAIL.value,
                f"resource {item.name!r} crossed its configured regression policy",
                required_evidence=(f"resource:{item.name}",),
                observed_evidence=(f"resource:{item.name}",),
                policy={"required": item.name in required_resource_names},
                remediation="inspect the matched resource observations and configured budget",
            )
        elif not enough_samples:
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
    behavior_pairs: list[tuple[Any, Mapping[str, Any] | None, Mapping[str, Any] | None]] = []
    if behavior is not None:
        behavior_pairs.append((behavior, None, None))
    else:
        for left, right in _pair_members(before_members, after_members):
            old_behavior = _behavior_evidence(left.run, left.capsule)
            new_behavior = _behavior_evidence(right.run, right.capsule)
            old_predictions = old_behavior.get("predictions") if old_behavior else None
            new_predictions = new_behavior.get("predictions") if new_behavior else None
            if (
                isinstance(old_predictions, Sequence)
                and not isinstance(old_predictions, (str, bytes))
                and isinstance(new_predictions, Sequence)
                and not isinstance(new_predictions, (str, bytes))
            ):
                behavior_pairs.append(
                    ((list(old_predictions), list(new_predictions)), old_behavior, new_behavior)
                )
                behavior_source = "capsule"
    member_behavior: list[dict[str, Any]] = []
    behavior_failed = False
    behavior_observations = 0
    representative_behavior: dict[str, Any] | None = None
    behavior_threshold = thresholds.get(
        "behavior:accuracy", thresholds.get("behavior_accuracy", 0.0)
    )
    calibration_threshold = thresholds.get(
        "behavior:calibration", thresholds.get("behavior_calibration", 0.0)
    )
    for pair_index, (pair_behavior, old_behavior, new_behavior) in enumerate(behavior_pairs):
        pair_labels = behavior_labels
        pair_features = behavior_features
        if pair_labels is None and isinstance(new_behavior, Mapping):
            pair_labels = new_behavior.get("labels")
        if pair_labels is None and isinstance(old_behavior, Mapping):
            pair_labels = old_behavior.get("labels")
        if pair_features is None and isinstance(new_behavior, Mapping):
            pair_features = new_behavior.get("features")
        if pair_features is None and isinstance(old_behavior, Mapping):
            pair_features = old_behavior.get("features")
        behavior_result = behavioral_diff(
            *pair_behavior,
            labels=pair_labels,
            features=pair_features,
            min_support=min_slice_support,
            confidence=confidence,
            n_resamples=n_resamples,
            seed=seed + len(names) + len(performance_result.metrics) + pair_index,
            practical_threshold=behavior_threshold,
            calibration_threshold=calibration_threshold,
        )
        behavior_dict = behavior_result.to_dict()
        behavior_dict["source"] = behavior_source
        behavior_dict["practical_threshold"] = abs(behavior_threshold)
        behavior_dict["calibration_threshold"] = abs(calibration_threshold)
        member_behavior.append(behavior_dict)
        if representative_behavior is None or behavior_result.regression:
            representative_behavior = behavior_dict
        behavior = pair_behavior
        behavior_labels = pair_labels
        behavior_features = pair_features
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
        behavior_failed = behavior_failed or bool(
            behavior_result.regression or behavior_result.calibration_regression or regressed_slices
        )
        behavior_observations = max(behavior_observations, behavior_result.sample_count)
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
            decisions.setdefault(
                "behavior",
                EvidenceStatus(
                    EvidenceState.WARN.value,
                    "behavioral predictions are present but labels are unavailable",
                    required_evidence=("behavior",),
                    observed_evidence=("behavior",),
                    policy={"minimum_samples": min_sample_count},
                    remediation="attach labels or a confirmatory behavioral evaluation",
                ),
            )
        else:
            decisions.setdefault(
                "behavior",
                EvidenceStatus(
                    EvidenceState.PASS.value,
                    "behavioral evidence satisfied its configured policy",
                    required_evidence=("behavior",),
                    observed_evidence=("behavior",),
                    policy={"minimum_samples": min_sample_count},
                ),
            )
    if representative_behavior is not None:
        if len(member_behavior) > 1:
            representative_behavior = dict(representative_behavior)
            representative_behavior["runs"] = member_behavior
        evidence["_behavior"] = _json_safe(representative_behavior)

    baseline_health = _group_health(before_members, side="baseline")
    candidate_health = _group_health(after_members, side="candidate")
    evidence["_run_status"] = {**baseline_health, **candidate_health}
    if baseline_health["baseline_failed"]:
        notes.append(
            "baseline run status: " + ", ".join(baseline_health["baseline_failed_run_ids"])
        )
    if candidate_health["candidate_failed"]:
        failed_ids = candidate_health["candidate_failed_run_ids"]
        notes.append("candidate run status: " + ", ".join(failed_ids))
        regressions.append("run:failed")
    elif candidate_health["candidate_incomplete"]:
        notes.append("candidate run is incomplete")
        regressions.append("run:incomplete")
    if baseline_health["baseline_failed"] or candidate_health["candidate_failed"]:
        decisions["run"] = EvidenceStatus(
            EvidenceState.FAIL.value,
            "one or both compared runs did not complete successfully",
            required_evidence=("run_status",),
            observed_evidence=("run_status",),
            policy={
                "baseline_failed": baseline_health["baseline_failed"],
                "candidate_failed": candidate_health["candidate_failed"],
                "baseline_failed_run_ids": baseline_health["baseline_failed_run_ids"],
                "candidate_failed_run_ids": candidate_health["candidate_failed_run_ids"],
            },
            remediation=(
                "rerun the comparison with healthy, completed baseline and candidate capsules"
            ),
        )
    elif baseline_health["baseline_incomplete"] or candidate_health["candidate_incomplete"]:
        decisions["run"] = EvidenceStatus(
            EvidenceState.INCONCLUSIVE.value,
            "one or more compared runs are incomplete",
            required_evidence=("run_status",),
            observed_evidence=("run_status",),
            policy={
                "baseline_run_ids": baseline_health["baseline_run_ids"],
                "candidate_run_ids": candidate_health["candidate_run_ids"],
            },
            remediation="wait for all compared runs to finish before comparing them",
        )
        inconclusive_reasons.append("run")
    else:
        decisions["run"] = EvidenceStatus(
            EvidenceState.PASS.value,
            "both runs supplied a completed run status",
            required_evidence=("run_status",),
            observed_evidence=("run_status",),
        )
    parity = _group_parity(after_members)
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
            regressions.append("parity:mismatch")
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
    member_missing_required: dict[str, list[str]] = {}
    for member in (*before_members, *after_members):
        missing_for_member: list[str] = []
        for required in required_evidence_names:
            key = required.lstrip("_")
            if key in {"parity"} and _parity_evidence(member.run, member.capsule) is None:
                missing_for_member.append(required)
            elif key in {"behavior"} and _behavior_evidence(member.run, member.capsule) is None:
                missing_for_member.append(required)
            elif key in {"run_status", "run"}:
                continue
        if missing_for_member:
            member_missing_required[member.run_id] = missing_for_member
    missing_required_evidence: dict[str, list[str]] = {}
    for required in required_evidence_names:
        candidates = {required, required.lstrip("_"), f"_{required.lstrip('_')}"}
        if required.startswith("_"):
            candidates.add(required[1:])
        absent = not candidates.intersection(available_evidence)
        members_missing = [
            run_id for run_id, names in member_missing_required.items() if required in names
        ]
        missing_required_evidence[required] = sorted(set(members_missing))
        if absent or members_missing:
            regressions.append(f"missing_evidence:{required}")
            decisions[required] = EvidenceStatus(
                EvidenceState.FAIL.value,
                f"required evidence {required!r} is absent",
                required_evidence=(required,),
                observed_evidence=(),
                policy={
                    "required": True,
                    "missing_run_ids": members_missing,
                },
                remediation=f"capture and attach {required!r} before running the gate",
            )
    evidence["_requirements"] = {
        "required_metrics": sorted(required_metric_names),
        "required_resources": sorted(required_resource_names),
        "required_evidence": sorted(required_evidence_names),
        "minimum_sample_count": min_sample_count,
        "missing_evidence": missing_required_evidence,
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
