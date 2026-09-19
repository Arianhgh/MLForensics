"""Machine-learning-aware CI decisions built on the comparison engine."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .analysis import compare_runs
from .core import Comparison, EvidenceState, Run, RunCapsule


def _json_safe(value: Any) -> Any:
    """Make adapter and user-supplied CI details safe for strict JSON."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        import math

        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _parity_passed(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    record = _mapping(value)
    for key in ("passed", "equal", "success"):
        if isinstance(record.get(key), bool):
            return bool(record[key])
    status = record.get("status")
    if isinstance(status, str):
        lowered = status.casefold()
        if lowered in {"pass", "passed", "success", "succeeded", "ok"}:
            return True
        if lowered in {"fail", "failed", "failure", "error", "mismatch"}:
            return False
    return None


def _built_in_checks(
    comparison: Comparison,
    *,
    fail_on_failed_run: bool,
    fail_on_nonfinite: bool,
    fail_on_missing_evidence: bool,
    fail_on_behavior_regression: bool,
    fail_on_parity_failure: bool,
) -> list[CICheck]:
    evidence = comparison.evidence
    checks: list[CICheck] = []

    status = _mapping(evidence.get("_run_status"))
    baseline_failed = bool(status.get("baseline_failed"))
    candidate_failed = bool(status.get("candidate_failed"))
    baseline_incomplete = bool(status.get("baseline_incomplete"))
    candidate_incomplete = bool(status.get("candidate_incomplete"))
    if status:
        unhealthy = (
            baseline_failed or candidate_failed or baseline_incomplete or candidate_incomplete
        )
        check_status = (
            "fail" if unhealthy and fail_on_failed_run else "warn" if unhealthy else "pass"
        )
        checks.append(
            CICheck(
                "run completed successfully",
                check_status != "fail",
                check_status,
                dict(status),
            )
        )

    quality = _mapping(evidence.get("_data_quality"))
    nonfinite: dict[str, int] = {}
    baseline_nonfinite: dict[str, int] = {}
    missing: list[str] = []
    insufficient: list[str] = []
    required_missing: list[str] = []
    required_insufficient: list[str] = []
    for category in ("metrics", "resources"):
        for name, raw_details in _mapping(quality.get(category)).items():
            details = _mapping(raw_details)
            try:
                count = int(details.get("candidate_nonfinite", 0) or 0)
            except (TypeError, ValueError):
                count = 0
            if count:
                nonfinite[f"{category}:{name}"] = count
            try:
                baseline_count = int(details.get("baseline_nonfinite", 0) or 0)
            except (TypeError, ValueError):
                baseline_count = 0
            if baseline_count:
                baseline_nonfinite[f"{category}:{name}"] = baseline_count
            if details.get("candidate_missing"):
                missing.append(f"{category}:{name}")
            if details.get("required") and not details.get("candidate_observations", 0):
                required_missing.append(f"{category}:{name}")
            if details.get("insufficient_evidence") or (
                category == "metrics" and not details.get("paired_finite_observations", 0)
            ):
                insufficient.append(f"{category}:{name}")
                if details.get("required"):
                    required_insufficient.append(f"{category}:{name}")
    if quality:
        check_status = (
            "fail" if nonfinite and fail_on_nonfinite else "warn" if nonfinite else "pass"
        )
        checks.append(
            CICheck(
                "no non-finite candidate observations",
                check_status != "fail",
                check_status,
                {"nonfinite": nonfinite},
            )
        )
        if baseline_nonfinite:
            baseline_status = "fail" if fail_on_nonfinite else "warn"
            checks.append(
                CICheck(
                    "no non-finite baseline observations",
                    baseline_status != "fail",
                    baseline_status,
                    {"nonfinite": baseline_nonfinite},
                )
            )
        missing_status = (
            "fail"
            if (missing or required_missing) and fail_on_missing_evidence
            else "warn"
            if missing or required_missing
            else "pass"
        )
        checks.append(
            CICheck(
                "candidate evidence is present",
                missing_status != "fail",
                missing_status,
                {
                    "missing": sorted(set(missing + required_missing)),
                    "required_missing": required_missing,
                },
            )
        )
        if insufficient:
            checks.append(
                CICheck(
                    "comparison evidence is sufficient",
                    not required_insufficient or not fail_on_missing_evidence,
                    "fail" if required_insufficient and fail_on_missing_evidence else "warn",
                    {
                        "insufficient": insufficient,
                        "required_insufficient": required_insufficient,
                    },
                )
            )

    requirements = _mapping(evidence.get("_requirements"))
    required_evidence = [str(item) for item in requirements.get("required_evidence", ())]
    available = set(evidence)
    per_member_missing = _mapping(requirements.get("missing_evidence"))
    missing_required_evidence = [
        item
        for item in required_evidence
        if (
            not ({item, item.lstrip("_"), f"_{item.lstrip('_')}"} & available)
            or bool(per_member_missing.get(item))
        )
    ]
    if required_evidence:
        evidence_status = (
            "fail"
            if missing_required_evidence and fail_on_missing_evidence
            else ("warn" if missing_required_evidence else "pass")
        )
        checks.append(
            CICheck(
                "required evidence is present",
                evidence_status != "fail",
                evidence_status,
                {
                    "required": required_evidence,
                    "missing": missing_required_evidence,
                    "missing_run_ids": {
                        item: list(per_member_missing.get(item, ()))
                        for item in missing_required_evidence
                        if per_member_missing.get(item)
                    },
                },
            )
        )

    behavior = _mapping(evidence.get("_behavior"))
    if behavior:
        regressed_slices = [
            item.get("name")
            for item in behavior.get("slices", [])
            if isinstance(item, Mapping) and _mapping(item.get("metadata")).get("regression")
        ]
        regressed = (
            bool(behavior.get("regression"))
            or bool(behavior.get("calibration_regression"))
            or bool(regressed_slices)
        )
        directional = behavior.get("accuracy_delta") is not None
        if regressed:
            check_status = "fail" if fail_on_behavior_regression else "warn"
        elif not directional and behavior.get("changed_predictions", 0):
            check_status = "warn"
        else:
            check_status = "pass"
        checks.append(
            CICheck(
                "behavioral regression",
                check_status != "fail",
                check_status,
                {
                    "accuracy_delta": behavior.get("accuracy_delta"),
                    "accuracy_delta_interval": behavior.get("accuracy_delta_interval"),
                    "prediction_change_rate": behavior.get("prediction_change_rate"),
                    "flips_to_incorrect": behavior.get("flips_to_incorrect", 0),
                    "calibration_delta": behavior.get("calibration_delta"),
                    "calibration_delta_interval": behavior.get("calibration_delta_interval"),
                    "distribution_total_variation": behavior.get("distribution_total_variation"),
                    "regressed_slices": regressed_slices,
                },
            )
        )

    if "_parity" in evidence:
        parity = evidence["_parity"]
        parity_passed = _parity_passed(parity)
        if parity_passed is None:
            checks.append(
                CICheck(
                    "export parity",
                    True,
                    "warn",
                    {"reason": "parity evidence has no recognized outcome", "evidence": parity},
                )
            )
        else:
            check_status = (
                "fail"
                if not parity_passed and fail_on_parity_failure
                else "warn"
                if not parity_passed
                else "pass"
            )
            checks.append(
                CICheck(
                    "export parity",
                    check_status != "fail",
                    check_status,
                    {"evidence": parity},
                )
            )
    return checks


@dataclass(frozen=True)
class CICheck:
    """One named CI assertion with an optional structured explanation."""

    name: str
    passed: bool
    status: str = "pass"
    details: Mapping[str, Any] = field(default_factory=dict)
    check_id: str | None = None

    def __post_init__(self) -> None:
        status = self.status.lower()
        if status not in {"pass", "fail", "warn", "skip"}:
            raise ValueError("CI check status must be pass, fail, warn, or skip")
        if status == "fail" and self.passed:
            raise ValueError("a failing CI check cannot be marked passed")
        if status == "pass" and not self.passed:
            raise ValueError("a passing CI check must be marked passed")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "details", _json_safe(self.details))
        if self.check_id is None:
            normalized = re.sub(r"[^a-z0-9]+", "_", self.name.casefold()).strip("_")
            object.__setattr__(self, "check_id", normalized or "check")
        elif not isinstance(self.check_id, str) or not self.check_id.strip():
            raise ValueError("check_id must be a non-empty string or None")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "status": self.status,
            "details": dict(self.details),
            "check_id": self.check_id,
        }


@dataclass(frozen=True)
class CIResult:
    """Structured result of a CI evaluation."""

    passed: bool
    checks: tuple[CICheck, ...] = ()
    comparison: Comparison | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def failed_checks(self) -> tuple[CICheck, ...]:
        return tuple(check for check in self.checks if check.status == "fail")

    @property
    def warnings(self) -> tuple[CICheck, ...]:
        return tuple(check for check in self.checks if check.status == "warn")

    @property
    def exit_code(self) -> int:
        return 0 if self.passed else 1

    def __bool__(self) -> bool:
        return self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "exit_code": self.exit_code,
            "checks": [check.to_dict() for check in self.checks],
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "metadata": _json_safe(self.metadata),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=True, allow_nan=False, default=str
        )

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"ML FORENSICS CHECK: {status}"]
        for check in self.checks:
            marker = {"pass": "✓", "fail": "✗", "warn": "⚠", "skip": "–"}[check.status]
            lines.append(f"{marker} {check.name}")
        return "\n".join(lines)

    __str__ = summary


def evaluate_comparison(
    comparison: Comparison,
    *,
    fail_on_regression: bool = True,
    fail_on_failed_run: bool = True,
    fail_on_nonfinite: bool = True,
    fail_on_missing_evidence: bool = True,
    fail_on_behavior_regression: bool = True,
    fail_on_parity_failure: bool = True,
    extra_checks: Iterable[CICheck | Callable[[Comparison], CICheck]] = (),
) -> CIResult:
    """Turn comparison evidence into a conservative, process-friendly CI decision.

    Optional checks are only emitted when their corresponding evidence exists.
    This keeps absence of parity or behavioral capture distinct from a passing test.
    """
    checks: list[CICheck] = []
    statistical_regressions = [
        name
        for name in comparison.regressions
        if not name.startswith(
            ("behavior:", "run:", "nonfinite:", "missing_", "noninferiority:", "parity:")
        )
    ]
    if statistical_regressions and fail_on_regression:
        checks.append(
            CICheck(
                "statistical and resource regression",
                False,
                "fail",
                {"regressions": statistical_regressions},
            )
        )
    elif statistical_regressions:
        checks.append(
            CICheck(
                "statistical and resource regression",
                True,
                "warn",
                {"regressions": statistical_regressions},
            )
        )
    else:
        checks.append(CICheck("statistical and resource regression", True))
    checks.extend(
        _built_in_checks(
            comparison,
            fail_on_failed_run=fail_on_failed_run,
            fail_on_nonfinite=fail_on_nonfinite,
            fail_on_missing_evidence=fail_on_missing_evidence,
            fail_on_behavior_regression=fail_on_behavior_regression,
            fail_on_parity_failure=fail_on_parity_failure,
        )
    )
    # A failed comparison is already decomposed into the dedicated checks
    # above (regression, run health, non-finite, parity, and missing evidence).
    # Only an inconclusive comparison needs its own policy check; otherwise a
    # caller opting to warn on regressions would be failed a second time.
    if comparison.status == EvidenceState.INCONCLUSIVE.value:
        status = "fail" if fail_on_missing_evidence else "warn"
        checks.append(
            CICheck(
                "comparison evidence status",
                status != "fail",
                status,
                {
                    "status": comparison.status,
                    "reason": next(
                        (note for note in comparison.notes if "inconclusive" in note),
                        None,
                    ),
                    "decisions": {
                        name: decision.to_dict()
                        for name, decision in comparison.decisions.items()
                        if decision.status != EvidenceState.PASS.value
                    },
                },
            )
        )
    for extra in extra_checks:
        check = extra(comparison) if callable(extra) else extra
        checks.append(check)
    passed = not any(check.status == "fail" for check in checks)
    return CIResult(passed, tuple(checks), comparison)


def ci_gate(
    baseline: RunCapsule | Run | str,
    candidate: RunCapsule | Run | str,
    *,
    confidence: float = 0.95,
    n_resamples: int = 2_000,
    seed: int = 0,
    practical_thresholds: Mapping[str, float] | None = None,
    higher_is_better: Mapping[str, bool] | None = None,
    noninferiority_margins: Mapping[str, float] | None = None,
    behavior: tuple[list[Any], list[Any]] | None = None,
    labels: list[Any] | None = None,
    features: list[Mapping[str, Any]] | None = None,
    min_slice_support: int = 5,
    required_metrics: Sequence[str] | None = None,
    required_resources: Sequence[str] | None = None,
    required_evidence: Sequence[str] | None = None,
    min_sample_count: int = 1,
    fail_on_regression: bool = True,
    fail_on_failed_run: bool = True,
    fail_on_nonfinite: bool = True,
    fail_on_missing_evidence: bool = True,
    fail_on_behavior_regression: bool = True,
    fail_on_parity_failure: bool = True,
    extra_checks: Iterable[CICheck | Callable[[Comparison], CICheck]] = (),
) -> CIResult:
    """Compare two runs and return a process-friendly CI result."""
    comparison = compare_runs(
        baseline,
        candidate,
        confidence=confidence,
        n_resamples=n_resamples,
        seed=seed,
        practical_thresholds=practical_thresholds,
        noninferiority_margins=noninferiority_margins,
        higher_is_better=higher_is_better,
        behavior=behavior,
        labels=labels,
        features=features,
        min_slice_support=min_slice_support,
        required_metrics=required_metrics,
        required_resources=required_resources,
        required_evidence=required_evidence,
        min_sample_count=min_sample_count,
    )
    return evaluate_comparison(
        comparison,
        fail_on_regression=fail_on_regression,
        fail_on_failed_run=fail_on_failed_run,
        fail_on_nonfinite=fail_on_nonfinite,
        fail_on_missing_evidence=fail_on_missing_evidence,
        fail_on_behavior_regression=fail_on_behavior_regression,
        fail_on_parity_failure=fail_on_parity_failure,
        extra_checks=extra_checks,
    )


run_ci = ci_gate

__all__ = ["CICheck", "CIResult", "ci_gate", "evaluate_comparison", "run_ci"]
