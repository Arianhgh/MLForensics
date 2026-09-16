"""Human-readable and JSON reports."""

from __future__ import annotations

import dataclasses
import json
from typing import Any


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value
    return value


def to_json(value: Any, *, indent: int = 2) -> str:
    return (
        json.dumps(_jsonable(value), indent=indent, sort_keys=True, ensure_ascii=False, default=str)
        + "\n"
    )


def render_comparison(comparison: Any) -> str:
    if not hasattr(comparison, "metrics"):
        lines = [f"Comparison: {comparison.left_run_id} → {comparison.right_run_id}"]
        if getattr(comparison, "metric_deltas", {}):
            lines.append("Metrics:")
            lines.extend(
                f"  {name}: Δ {value:.6g}" for name, value in comparison.metric_deltas.items()
            )
        if getattr(comparison, "resource_deltas", {}):
            lines.append("Resources:")
            lines.extend(
                f"  {name}: Δ {value:.6g}" for name, value in comparison.resource_deltas.items()
            )
        if getattr(comparison, "regressions", ()):
            lines.append("Regressions: " + ", ".join(comparison.regressions))
        if getattr(comparison, "notes", ()):
            lines.append("Evidence:")
            lines.extend(f"  {note}" for note in comparison.notes)
        lines.append("Overall: " + ("REGRESSION" if comparison.regressions else "PASS"))
        return "\n".join(lines) + "\n"
    lines = [
        "MLForensics comparison",
        "=" * 23,
        "",
        "Metric                 baseline       candidate       change",
        "-" * 63,
    ]
    for metric in comparison.metrics:
        old = "n/a" if metric.baseline is None else f"{metric.baseline:.6g}"
        new = "n/a" if metric.candidate is None else f"{metric.candidate:.6g}"
        delta = "n/a" if metric.delta is None else f"{metric.delta:+.6g}"
        marker = " ⚠" if metric.regression else ""
        lines.append(f"{metric.name:<23} {old:>12} {new:>14} {delta:>14}{marker}")
        if metric.confidence_interval is not None:
            lines.append(
                f"  95% CI: [{metric.confidence_interval[0]:+.6g}, "
                f"{metric.confidence_interval[1]:+.6g}]"
            )
    if comparison.performance.get("likely_contributors"):
        lines.extend(
            [
                "",
                "Performance contributors",
                "-" * 24,
                *[f"⚠ {name}" for name in comparison.performance["likely_contributors"]],
            ]
        )
    lines.extend(["", f"Conclusion: {comparison.conclusion}"])
    return "\n".join(lines) + "\n"


def render_bisect(result: Any) -> str:
    lines = [
        "MLForensics bisect",
        "=" * 18,
        "",
        f"First regression-compatible change: {result.first_bad or 'none found'}",
    ]
    for item in result.evaluations:
        status = "GOOD" if item.passed else "BAD"
        score = "" if item.score is None else f" score={item.score:.6g}"
        lines.append(f"{status:<5} {item.target}{score} ({item.runs} run(s))")
    if result.inconclusive:
        lines.extend(["", "Inconclusive: " + ", ".join(result.inconclusive)])
    return "\n".join(lines) + "\n"


def render_impact(report: Any) -> str:
    lines = [
        "MLForensics impact",
        "=" * 18,
        "",
        "Changed files",
        *[f"  {path}" for path in report.changed_files],
        "",
        "Affected nodes",
        *[f"  {node}" for node in report.affected_nodes],
        "",
        "Recommended validation",
        *[f"  ✓ {item}" for item in report.recommended_validation],
    ]
    if report.skipped_validation:
        lines.extend(
            ["", "Likely unnecessary", *[f"  {item}" for item in report.skipped_validation]]
        )
    return "\n".join(lines) + "\n"


def render_parity(report: Any, *, max_divergences: int = 10) -> str:
    status = "PASS" if report.passed else "FAIL"
    lines = [
        "MLForensics parity",
        "=" * 19,
        "",
        f"{status}: {report.divergent_count}/{report.sample_count} inputs diverged",
        f"max absolute error: {report.max_absolute_error:.6g}",
        f"max relative error: {report.max_relative_error:.6g}",
    ]
    # A parity sweep can diverge on thousands of inputs. Show the worst offenders,
    # which are the ones worth shrinking, and point at --json for the rest.
    ranked = sorted(report.divergences, key=lambda item: item.max_absolute_error, reverse=True)
    for divergence in ranked[:max_divergences]:
        lines.append(
            f"  input #{divergence.index}: abs={divergence.max_absolute_error:.6g} "
            f"rel={divergence.max_relative_error:.6g}"
        )
    if len(ranked) > max_divergences:
        lines.append(
            f"  ... {len(ranked) - max_divergences} more divergent input(s); "
            f"use --json for the full list"
        )
    return "\n".join(lines) + "\n"


def render(value: Any, *, output_format: str = "text") -> str:
    if output_format == "json":
        return to_json(value)
    name = type(value).__name__.lower()
    if "comparison" in name:
        return render_comparison(value)
    if "bisect" in name:
        return render_bisect(value)
    if "impact" in name:
        return render_impact(value)
    if "parity" in name:
        return render_parity(value)
    return to_json(value)
