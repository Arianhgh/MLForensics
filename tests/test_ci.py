from mlforensics.ci import ci_gate
from mlforensics.core import MetricSeries, Run


class _LegacyMetric:
    def __init__(self, name, values):
        self.name = name
        self.values = values


class _LegacyRun:
    def __init__(self, run_id, values):
        self.run_id = run_id
        self.status = "succeeded"
        self.metrics = (_LegacyMetric("accuracy", values),)
        self.resources = ()
        self.metadata = {}
        self.failure_signature = None


class _LegacyCapsule:
    def __init__(self, run):
        self.run = run
        self.artifacts = ()
        self.payloads = {}


def test_ci_uses_behavior_run_health_and_parity_evidence():
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
            "parity": {"passed": False},
        },
    )
    result = ci_gate(baseline, candidate, n_resamples=100)
    assert not result
    statuses = {check.name: check.status for check in result.checks}
    assert statuses["run completed successfully"] == "fail"
    assert statuses["behavioral regression"] == "fail"
    assert statuses["export parity"] == "fail"
    assert result.exit_code == 1


def test_ci_can_warn_instead_of_fail_for_configured_optional_policies():
    baseline = Run(run_id="old", status="succeeded")
    candidate = Run(
        run_id="new",
        status="succeeded",
        metadata={"parity": {"passed": False}},
    )
    result = ci_gate(
        baseline,
        candidate,
        n_resamples=10,
        fail_on_parity_failure=False,
    )
    assert result
    assert next(check for check in result.checks if check.name == "export parity").status == "warn"


def test_ci_fails_nonfinite_observations_without_shifting_paired_samples():
    baseline = _LegacyCapsule(_LegacyRun("old", [1.0, 2.0, 3.0]))
    candidate = _LegacyCapsule(_LegacyRun("new", [2.0, float("nan"), 4.0]))
    result = ci_gate(baseline, candidate, n_resamples=20)
    assert not result
    assert result.comparison.metric_deltas["accuracy"] == 1.0
    check = next(
        check for check in result.checks if check.name == "no non-finite candidate observations"
    )
    assert check.status == "fail"
    assert check.details["nonfinite"] == {"metrics:accuracy": 1}


def test_ci_fails_when_candidate_drops_baseline_evidence():
    baseline = Run(
        run_id="old",
        status="succeeded",
        metrics=(MetricSeries("accuracy", [0.8, 0.9]),),
    )
    candidate = Run(run_id="new", status="succeeded")
    result = ci_gate(baseline, candidate, n_resamples=20)
    assert not result
    check = next(check for check in result.checks if check.name == "candidate evidence is present")
    assert check.status == "fail"
    assert check.details["missing"] == ["metrics:accuracy"]
