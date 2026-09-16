from mlforensics.analysis import behavioral_diff, discover_slices


def test_behavior_reports_prediction_flips_confidence_only_changes_and_accuracy():
    baseline = [[0.8, 0.2], [0.1, 0.9], [0.7, 0.3], [0.6, 0.4]]
    candidate = [[0.7, 0.3], [0.2, 0.8], [0.2, 0.8], [0.65, 0.35]]
    result = behavioral_diff(baseline, candidate, labels=[0, 1, 0, 0])
    assert result.sample_count == 4
    assert result.class_flips == 1
    assert result.confidence_only_changes == 3
    assert result.baseline_accuracy == 1.0
    assert result.candidate_accuracy == 0.75
    assert result.flips_to_incorrect == 1
    assert result.confidence_delta_mean < 0


def test_slice_discovery_finds_supported_feature_slices():
    baseline = [[0.9, 0.1]] * 3 + [[0.1, 0.9]] * 3
    candidate = [[0.8, 0.2]] * 3 + [[0.8, 0.2]] * 3
    features = [{"region": "a"}] * 3 + [{"region": "b"}] * 3
    slices = discover_slices(baseline, candidate, features, labels=[0] * 3 + [1] * 3, min_support=3)
    assert {item.name for item in slices} == {"region='a'", "region='b'"}
    assert any(item.regression for item in slices)


def test_scalar_class_predictions_are_not_mistaken_for_score_vectors():
    result = behavioral_diff([0, 1, 1], [0, 0, 1], labels=[0, 1, 1], n_resamples=100)
    assert result.class_flips == 1
    assert result.confidence_only_changes == 0
    assert result.baseline_accuracy == 1.0
    assert result.candidate_accuracy == 2 / 3


def test_behavioral_regression_requires_confidence_interval_beyond_threshold():
    baseline = [[0.9, 0.1]] * 8
    candidate = [[0.1, 0.9]] * 8
    result = behavioral_diff(
        baseline,
        candidate,
        labels=[0] * 8,
        confidence=0.95,
        n_resamples=100,
        practical_threshold=0.1,
    )
    assert result.regression
    assert result.accuracy_delta == -1.0
    assert result.accuracy_delta_interval == (-1.0, -1.0)
    assert result.prediction_change_interval is not None


def test_behavior_reports_distribution_and_calibration_regression():
    baseline = [[0.9, 0.1], [0.1, 0.9]] * 5
    candidate = [[0.6, 0.4], [0.4, 0.6]] * 5
    result = behavioral_diff(
        baseline,
        candidate,
        labels=[0, 1] * 5,
        calibration_threshold=0.1,
        n_resamples=100,
    )
    assert result.distribution_total_variation == 0
    assert result.accuracy_delta == 0
    assert result.calibration_delta > 0.2
    assert result.calibration_delta_interval[0] > 0.1
    assert result.calibration_regression
