from mlforensics.analysis import performance_diff


def test_latency_and_memory_regressions_are_detected():
    result = performance_diff(
        {"latency_ms": [10, 11, 9, 10], "memory_mb": [100, 101, 99, 100]},
        {"latency_ms": [15, 16, 14, 15], "memory_mb": [120, 121, 119, 120]},
        thresholds={"latency_ms": 0.2, "memory_mb": 0.05},
    )
    assert result.has_regression
    assert set(result.regressions) == {"latency_ms", "memory_mb"}


def test_resource_summaries_ignore_failed_samples_and_handle_scalars():
    result = performance_diff(
        {"latency_ms": [10, None, 10], "peak_memory_mb": 100},
        {"latency_ms": [11, 12, 11], "peak_memory_mb": 101},
    )
    assert result.metric("latency_ms").baseline == 10
    assert result.metric("peak_memory_mb").delta == 1


def test_performance_uses_uncertainty_and_reports_config_changes_as_context():
    result = performance_diff(
        {"latency_ms": [8, 12, 8, 12]},
        {"latency_ms": [9, 14, 7, 13]},
        thresholds={"latency_ms": 0.05},
        baseline_configuration={"loader": {"workers": 8}},
        candidate_configuration={"loader": {"workers": 2}},
        n_resamples=200,
        seed=4,
    )
    metric = result.metric("latency_ms")
    assert metric.relative_delta > 0.05
    assert not metric.regression  # the confidence interval still overlaps the budget
    assert metric.confidence_interval[0] <= 0.05
    assert result.configuration_changes["loader.workers"] == {
        "baseline": 8,
        "candidate": 2,
    }


def test_throughput_and_utilization_default_to_higher_is_better():
    result = performance_diff(
        {"throughput": [100, 100], "gpu_utilization": [0.9, 0.9]},
        {"throughput": [80, 80], "gpu_utilization": [0.7, 0.7]},
        thresholds={"throughput": 0.1, "gpu_utilization": 0.1},
        n_resamples=20,
    )
    assert set(result.regressions) == {"throughput", "gpu_utilization"}


def test_single_sample_resource_is_not_a_regression_without_a_budget():
    """One measurement per side cannot show that a resource grew significantly."""
    result = performance_diff({"peak_rss": 1_000_000}, {"peak_rss": 1_180_224})
    metric = result.metric("peak_rss")
    assert metric.delta == 180_224
    assert metric.confidence_interval is None
    assert metric.metadata["evidence_mode"] == "none"
    assert not metric.regression
    assert not result.has_regression


def test_single_sample_resource_is_compared_against_an_explicit_budget():
    result = performance_diff(
        {"peak_rss": 1_000_000},
        {"peak_rss": 1_180_224},
        thresholds={"peak_rss": 0.1},
    )
    metric = result.metric("peak_rss")
    assert metric.metadata["evidence_mode"] == "threshold"
    assert metric.regression  # +18% exceeds the declared 10% budget

    within = performance_diff(
        {"peak_rss": 1_000_000},
        {"peak_rss": 1_050_000},
        thresholds={"peak_rss": 0.1},
    )
    assert not within.metric("peak_rss").regression
