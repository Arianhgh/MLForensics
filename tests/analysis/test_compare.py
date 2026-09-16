from mlforensics.analysis import compare_runs, render_comparison
from mlforensics.core import MetricSeries, ResourceSeries, Run


def _run(run_id, accuracy, latency, seed=None):
    return Run(
        run_id=run_id,
        status="succeeded",
        metrics=(MetricSeries("accuracy", accuracy),),
        resources=(ResourceSeries("latency_ms", latency, units="ms"),),
        metadata={"seed": seed} if seed is not None else {},
    )


def _repeated(prefix, accuracies, latency):
    """Build one run per seed, which is what makes a run-level claim possible."""
    return [
        _run(f"{prefix}-{seed}", [value], latency, seed=seed)
        for seed, value in zip((11, 29, 37, 53, 71), accuracies)
    ]


def test_compare_runs_returns_existing_core_comparison_and_renders_it():
    result = compare_runs(
        _repeated("old", [0.80, 0.81, 0.79, 0.80, 0.80], [10, 10, 10]),
        _repeated("new", [0.70, 0.71, 0.69, 0.70, 0.70], [15, 15, 15]),
        practical_thresholds={"accuracy": 0.05, "latency_ms": 0.2},
        higher_is_better={"accuracy": True, "latency_ms": False},
        n_resamples=100,
    )
    assert result.left_run_id.startswith("old-11")
    assert result.metric_deltas["accuracy"] < 0
    assert "accuracy" in result.regressions
    assert "resource:latency_ms" in result.regressions
    text = render_comparison(result)
    assert "REGRESSION" in text


def test_single_run_per_side_reports_delta_without_claiming_a_change():
    """One run per side fixes the delta but cannot describe run-to-run spread."""
    result = compare_runs(
        _run("old", [0.80, 0.80, 0.80], [10, 10, 10]),
        _run("new", [0.70, 0.70, 0.70], [15, 15, 15]),
        higher_is_better={"accuracy": True},
        n_resamples=100,
    )
    evidence = result.evidence["accuracy"]
    assert evidence["delta"] < 0
    assert evidence["confidence_interval"] is None
    assert evidence["insufficient_evidence"] is True
    assert evidence["replicates"] == 1
    assert evidence["replicate_unit"] == "run"
    assert "accuracy" not in result.regressions


def test_seed_noise_between_identical_runs_is_not_reported_as_improvement():
    """Identical code, different seeds: the interval has to cover zero."""
    baseline = _repeated("old", [0.90, 0.93, 0.88, 0.91, 0.92], [10, 10, 10])
    candidate = _repeated("new", [0.91, 0.89, 0.93, 0.90, 0.92], [10, 10, 10])
    result = compare_runs(baseline, candidate, higher_is_better={"accuracy": True}, n_resamples=500)
    evidence = result.evidence["accuracy"]
    assert evidence["replicates"] == 5
    assert evidence["confidence_interval"][0] < 0 < evidence["confidence_interval"][1]
    assert evidence["improvement"] is False
    assert evidence["regression"] is False
    assert "accuracy" not in result.regressions


def test_declared_observation_identities_are_treated_as_repetitions():
    baseline = Run(
        run_id="old",
        status="succeeded",
        metrics=(MetricSeries("accuracy", [0.80, 0.81, 0.79], identities=[11, 29, 37]),),
    )
    candidate = Run(
        run_id="new",
        status="succeeded",
        metrics=(MetricSeries("accuracy", [0.70, 0.71, 0.69], identities=[11, 29, 37]),),
    )
    result = compare_runs(
        baseline,
        candidate,
        practical_thresholds={"accuracy": 0.05},
        higher_is_better={"accuracy": True},
        n_resamples=200,
    )
    assert result.evidence["accuracy"]["replicate_unit"] == "declared"
    assert result.evidence["accuracy"]["replicates"] == 3
    assert "accuracy" in result.regressions


def test_compare_uses_capsule_metadata_behavior_parity_and_run_health_evidence():
    baseline = Run(
        run_id="old",
        status="succeeded",
        metadata={"behavior": {"predictions": [[0.9, 0.1]] * 6, "labels": [0] * 6}},
    )
    candidate = Run(
        run_id="new",
        status="failed",
        metadata={
            "behavior": {"predictions": [[0.1, 0.9]] * 6, "labels": [0] * 6},
            "parity": {"passed": False, "mismatches": 2},
        },
    )
    result = compare_runs(baseline, candidate, n_resamples=100)
    assert "behavior:accuracy" in result.regressions
    assert "run:failed" in result.regressions
    assert result.evidence["_behavior"]["source"] == "capsule"
    assert result.evidence["_parity"]["passed"] is False
    assert result.evidence["_run_status"]["candidate_failed"] is True


def test_noninferiority_margin_is_a_real_gate():
    baseline = _repeated("old", [0.80, 0.80, 0.80, 0.80, 0.80], [10, 10, 10])
    candidate = _repeated("new", [0.76, 0.76, 0.76, 0.76, 0.76], [10, 10, 10])
    result = compare_runs(
        baseline,
        candidate,
        noninferiority_margins={"accuracy": 0.02},
        n_resamples=100,
    )
    assert "noninferiority:accuracy" in result.regressions
    assert result.evidence["accuracy"]["noninferior"] is False
