from mlforensics.analysis import compare_runs, render_comparison
from mlforensics.core import CaptureContext, MetricSeries, Observation, ResourceSeries, Run


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


def test_integer_observation_identities_are_not_mistaken_for_steps():
    baseline = Run(
        run_id="old",
        status="succeeded",
        metrics=(MetricSeries("accuracy", [0.8, 0.8, 0.8], identities=[0, 1, 2]),),
    )
    candidate = Run(
        run_id="new",
        status="succeeded",
        metrics=(MetricSeries("accuracy", [0.7, 0.7, 0.7], identities=[0, 1, 2]),),
    )
    result = compare_runs(baseline, candidate, n_resamples=100)
    assert result.evidence["accuracy"]["replicates"] == 3
    assert "accuracy" in result.regressions


def test_grouped_observation_pairing_is_invariant_to_run_order():
    def make(side, seed, value):
        return Run(
            run_id=f"{side}-{seed}",
            status="succeeded",
            metadata={"seed": seed},
            metrics=(MetricSeries("accuracy", [value, value], identities=["a", "b"]),),
        )

    baseline = [make("old", 11, 0.9), make("old", 29, 0.1)]
    candidate = [make("new", 11, 0.85), make("new", 29, 0.05)]
    ordered = compare_runs(baseline, candidate, n_resamples=100)
    reversed_result = compare_runs(baseline, candidate[::-1], n_resamples=100)
    assert ordered.metric_deltas == reversed_result.metric_deltas
    assert ordered.regressions == reversed_result.regressions
    assert (
        ordered.evidence["accuracy"]["confidence_interval"]
        == reversed_result.evidence["accuracy"]["confidence_interval"]
    )


def test_generated_capture_steps_are_not_independent_repetitions():
    captures = []
    for prefix, values in (("old", [0.8, 0.8, 0.8]), ("new", [0.7, 0.7, 0.7])):
        with CaptureContext(name=prefix) as capture:
            for step, value in enumerate(values):
                capture.record_metric("accuracy", value, step=step)
        captures.append(capture.capsule)
    result = compare_runs(*captures, n_resamples=100)
    assert result.evidence["accuracy"]["replicates"] == 1
    assert result.status == "inconclusive"


def test_grouped_observations_need_a_run_identity():
    def make(run_id, value):
        return Run(
            run_id=run_id,
            status="succeeded",
            metrics=(MetricSeries("accuracy", [value, value], identities=["a", "b"]),),
        )

    result = compare_runs(
        [make("old-1", 0.9), make("old-2", 0.1)],
        [make("new-2", 0.05), make("new-1", 0.85)],
        n_resamples=100,
    )
    assert result.status == "inconclusive"
    assert result.evidence["_data_quality"]["metrics"]["accuracy"]["pairing"] == "unpaired"


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


def test_duplicate_seed_without_trial_is_inconclusive():
    baseline = [_run(f"old-{index}", [0.8], [10], seed=7) for index in range(5)]
    candidate = [_run(f"new-{index}", [0.7], [10], seed=7) for index in range(5)]
    result = compare_runs(baseline, candidate, n_resamples=50)
    quality = result.evidence["_data_quality"]["metrics"]["accuracy"]
    assert quality["pairing"] == "ambiguous"
    assert result.status == "inconclusive"


def test_missing_identities_cannot_pair_a_group():
    baseline = [_run(f"old-{index}", [0.8 + index / 100], [10]) for index in range(5)]
    candidate = [_run(f"new-{index}", [0.7 + index / 100], [10]) for index in range(5)]
    result = compare_runs(baseline, candidate, n_resamples=50)
    quality = result.evidence["_data_quality"]["metrics"]["accuracy"]
    assert quality["pairing"] == "unpaired"
    assert result.status == "inconclusive"


def test_duplicate_capsules_do_not_inflate_sample_size():
    run = _run("shared", [0.8], [10], seed=3)
    result = compare_runs([run, run, run], [run, run, run], n_resamples=50)
    assert result.evidence["accuracy"]["replicates"] == 1


def test_duplicate_observation_identities_are_inconclusive():
    baseline = Run(
        run_id="old",
        status="succeeded",
        metrics=(MetricSeries("accuracy", [0.8, 0.8], identities=[7, 7]),),
    )
    candidate = Run(
        run_id="new",
        status="succeeded",
        metrics=(MetricSeries("accuracy", [0.7, 0.7], identities=[7, 7]),),
    )
    result = compare_runs(baseline, candidate, n_resamples=50)
    quality = result.evidence["_data_quality"]["metrics"]["accuracy"]
    assert quality["pairing"] == "ambiguous"
    assert quality["paired_finite_observations"] == 0
    assert quality["candidate_missing"] is False
    assert result.status == "inconclusive"


def test_run_identity_supports_sample_and_dataset_discriminators():
    def make(run_id, sample_id, value):
        return Run(
            run_id=run_id,
            status="succeeded",
            metrics=(MetricSeries("accuracy", [value]),),
            metadata={"seed": 3, "sample_id": sample_id, "dataset_version": "v1"},
        )

    result = compare_runs(
        [make("old-a", "a", 0.8), make("old-b", "b", 0.8)],
        [make("new-b", "b", 0.7), make("new-a", "a", 0.7)],
        n_resamples=50,
    )
    assert result.evidence["accuracy"]["replicates"] == 2
    assert result.evidence["_data_quality"]["metrics"]["accuracy"]["pairing"] == "stable_identity"


def test_failed_observation_without_a_metric_series_is_retained_as_evidence():
    baseline = Run(
        run_id="old",
        status="succeeded",
        observations=(Observation.from_value("accuracy", 0.8, identity="trial"),),
    )
    candidate = Run(
        run_id="new",
        status="failed",
        observations=(Observation.from_value("accuracy", float("nan"), identity="trial"),),
    )
    result = compare_runs(baseline, candidate, n_resamples=20)
    assert "accuracy" in result.evidence["_data_quality"]["metrics"]
    assert result.evidence["_data_quality"]["metrics"]["accuracy"]["candidate_nonfinite"] == 1
    assert "nonfinite:accuracy" in result.regressions


def test_failed_resource_observation_without_a_resource_series_is_retained():
    baseline = Run(
        run_id="old",
        status="succeeded",
        resources=(ResourceSeries("peak_rss", [100.0]),),
    )
    candidate = Run(
        run_id="new",
        status="succeeded",
        observations=(
            Observation.from_value(
                "peak_rss", float("inf"), identity="trial", metadata={"kind": "resource"}
            ),
        ),
    )
    result = compare_runs(baseline, candidate, n_resamples=20)
    assert result.evidence["_data_quality"]["resources"]["peak_rss"]["candidate_nonfinite"] == 1
    assert "nonfinite:resource:peak_rss" in result.regressions


def test_minimum_sample_count_withholds_regression_verdict():
    baseline = _repeated("old", [0.8, 0.8, 0.8, 0.8, 0.8], [10, 10, 10])[:2]
    candidate = _repeated("new", [0.6, 0.6, 0.6, 0.6, 0.6], [10, 10, 10])[:2]
    result = compare_runs(baseline, candidate, min_sample_count=3, n_resamples=50)
    assert result.status == "inconclusive"
    assert "accuracy" not in result.regressions
    assert result.evidence["accuracy"]["insufficient_evidence"] is True
