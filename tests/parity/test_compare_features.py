import pytest

from mlforensics.parity import InputCase, Tolerance, compare_models, compare_outputs


def test_precision_profiles_have_explicit_conservative_defaults_and_overrides():
    assert Tolerance.fp32().absolute == pytest.approx(1e-6)
    assert Tolerance.fp16().relative == pytest.approx(1e-2)
    assert Tolerance.bf16().absolute == pytest.approx(2e-2)
    assert Tolerance(profile="fp16", atol=1e-5).absolute == pytest.approx(1e-5)
    assert Tolerance(precision="float16", profile="fp16").profile == "fp16"


def test_output_specific_tolerances_follow_named_nested_outputs():
    result = compare_outputs(
        {"probabilities": [1.0], "logits": [1.0]},
        {"probabilities": [1.0005], "logits": [1.0005]},
        tolerance={
            "probabilities": {"atol": 1e-3, "rtol": 0},
            "logits": {"atol": 1e-6, "rtol": 0},
        },
    )
    assert not result.equal
    assert result.mismatch_paths == ("$['logits'][0]",)


def test_models_report_per_example_and_per_output_errors():
    report = compare_models(
        lambda value: {"stable": value, "drifting": value},
        lambda value: {"stable": value, "drifting": value + value / 10},
        [1, 2],
        tolerance={"stable": {"atol": 0, "rtol": 0}, "drifting": {"atol": 0, "rtol": 0}},
    )
    assert len(report.per_example_errors) == 2
    assert report.per_output_errors["$['drifting']"]["count"] == 2
    assert report.divergences[0].example_index in {0, 1}
    assert report.divergences[0].output_path == "$['drifting']"
    assert "per_output_errors" in report.to_dict()


def test_per_example_tolerances_can_use_input_case_labels():
    report = compare_models(
        lambda value: value,
        lambda value: value + 1e-4,
        [InputCase(1, "loose-case", "custom")],
        tolerance={"atol": 0, "rtol": 0},
        example_tolerances={"loose-case": {"atol": 1e-3, "rtol": 0}},
    )
    assert report.passed


def test_null_shape_dtype_and_order_policies_are_explicit():
    assert compare_outputs(None, None).equal
    assert compare_outputs(None, 0).reason == "null output differs"
    assert not compare_outputs([[1, 2]], [1, 2]).equal
    assert "shape" in compare_outputs([[1, 2]], [1, 2]).reason
    assert compare_outputs([1, 2], [2, 1], order="ignore").equal

    numpy = pytest.importorskip("numpy")
    dtype_result = compare_outputs(
        numpy.ones(2, dtype="float32"),
        numpy.ones(2, dtype="float64"),
        check_dtype=True,
    )
    assert not dtype_result.equal
    assert "dtype" in dtype_result.reason


def test_precision_aware_mode_infers_float16_profile():
    numpy = pytest.importorskip("numpy")
    expected = numpy.asarray([1000.0], dtype="float16")
    actual = numpy.asarray([1000.5], dtype="float16")
    assert compare_outputs(expected, actual, precision_aware=True).equal
