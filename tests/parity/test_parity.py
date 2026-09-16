import math

import pytest

from mlforensics.parity import (
    CallableBackendAdapter,
    DivergenceLocation,
    InputCase,
    ONNXRuntimeBackendAdapter,
    OptionalDependencyError,
    ShrinkPreconditionError,
    Tolerance,
    compare_batch,
    compare_models,
    compare_outputs,
    generate_adversarial_inputs,
    generate_edge_inputs,
    generate_representative_inputs,
    load_backend,
    parse_backend_spec,
    shrink_input,
)


def test_scalar_and_vector_outputs_respect_absolute_and_relative_tolerance():
    assert compare_outputs(1.0, 1.0000005, Tolerance(absolute=1e-6, relative=0)).equal
    assert not compare_outputs(1.0, 1.01, Tolerance(absolute=1e-6, relative=1e-4)).equal

    vector = compare_outputs([1.0, 2.0], [1.0, 2.0001], Tolerance(absolute=1e-6, relative=1e-6))
    assert not vector.equal
    assert vector.mismatch_count == 1
    assert vector.mismatch_paths == ("$[1]",)


def test_nested_outputs_and_nonfinite_policy():
    result = compare_outputs({"score": 2.0, "labels": [1, 0]}, {"score": 2.0, "labels": [1, 1]})
    assert not result.equal
    assert result.mismatch_paths == ("$['labels'][1]",)
    assert not compare_outputs(float("nan"), float("nan")).equal
    assert compare_outputs(float("nan"), float("nan"), Tolerance(nan_equal=True)).equal
    assert compare_outputs(float("inf"), float("inf")).equal
    assert not compare_outputs(float("inf"), float("-inf")).equal


def test_callable_adapters_and_report_summary():
    reference = CallableBackendAdapter(lambda x: x * 2, name="reference")
    candidate = CallableBackendAdapter(lambda x: x * 2 + (0.0000001 if x else 0), name="candidate")
    report = compare_models(reference, candidate, [0, 1, InputCase(2, "special", "example")])
    assert report.passed
    assert report.pass_count == 3
    assert "PASS: 3/3 cases matched" in report.summary()
    assert report.to_dict()["candidate"] == "candidate"


def test_localization_hook_receives_divergence_and_is_reported():
    calls = []

    def localize(*, input, comparison, **kwargs):
        calls.append(input)
        return DivergenceLocation(path=comparison.mismatch_paths[0], component="head")

    report = compare_models(
        lambda x: [x, x + 1],
        lambda x: [x, x + 2],
        [InputCase(3, "bad", "adversarial")],
        localization_hook=localize,
    )
    assert not report.passed
    assert calls == [3]
    assert report.results[0].localization.component == "head"
    assert report.results[0].comparison.mismatch_paths == ("$[1]",)


def test_batch_comparison_uses_backend_batch_methods():
    class BatchBackend:
        name = "batch"

        def __init__(self, offset):
            self.offset = offset
            self.calls = []

        def predict(self, value):
            return value + self.offset

        def predict_batch(self, values):
            self.calls.append(list(values))
            return [self.predict(value) for value in values]

    reference, candidate = BatchBackend(1), BatchBackend(1)
    report = compare_batch(reference, candidate, [[1, 2], [5]])
    assert report.passed
    assert reference.calls == [[1, 2], [5]]
    assert len(report.results) == 3


def test_input_generation_is_deterministic_and_has_edge_categories():
    first = generate_representative_inputs(shape=(2,), count=7, seed=17)
    second = generate_representative_inputs(shape=(2,), count=7, seed=17)
    assert all((a == b).all() for a, b in zip(first, second))
    assert len(generate_edge_inputs(shape=(2,))) >= 4
    adversarial = generate_adversarial_inputs(shape=(2,), seed=4)
    assert any(math.isinf(float(item.flat[0])) for item in adversarial)


def test_default_and_custom_shrinking_hooks():
    shrunk = shrink_input([8, 4], lambda value: sum(value) >= 1)
    assert shrunk == [0, 1] or sum(shrunk) < 12

    seen = []

    def custom(value, predicate):
        seen.append(value)
        return [0]

    assert shrink_input([10], lambda value: True, shrinker=custom) == [0]
    assert seen == [[10]]


def test_optional_onnxruntime_adapter_fails_only_when_used():
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        with pytest.raises(OptionalDependencyError):
            ONNXRuntimeBackendAdapter("model.onnx")


def test_backend_specs_are_lazy_and_python_targets_are_loadable():
    assert parse_backend_spec("artifact.onnx").kind == "onnx"
    assert parse_backend_spec("torchscript:model.pt").kind == "torchscript"
    backend = load_backend("python:math:sqrt", name="square-root")
    assert backend.name == "square-root"
    assert backend.predict(9) == 3


def test_existing_onnx_session_does_not_require_importing_the_optional_runtime():
    class Input:
        name = "features"

    class Session:
        def get_inputs(self):
            return [Input()]

        def run(self, outputs, feed):
            return [feed["features"]]

    adapter = ONNXRuntimeBackendAdapter(Session())
    output = adapter.predict([1.0, 2.0])
    assert output.tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match="input names differ"):
        adapter.predict({"wrong": [1.0]})


def test_compare_models_generates_labeled_inputs_when_none_are_supplied():
    report = compare_models(
        lambda value: value * 2,
        lambda value: value * 2,
        input_shape=None,
        representative_count=4,
        include_edge=False,
        include_adversarial=False,
        seed=7,
    )
    assert report.passed
    assert report.sample_count == 4
    assert {case.category for case in report.results} == {"representative"}
    assert report.metadata["input_source"] == "generated"


def test_shrinking_rejects_invalid_candidates_and_non_failing_inputs():
    def failure(value):
        if not value:
            raise ValueError("empty input is invalid")
        return sum(value) >= 2

    result = shrink_input([2, 1], failure)
    assert result
    assert failure(result)
    with pytest.raises(ShrinkPreconditionError, match="does not satisfy"):
        shrink_input([0], failure)


def test_custom_shrinker_output_must_preserve_the_failure():
    with pytest.raises(ValueError, match="does not preserve"):
        shrink_input([3], lambda value: bool(value), shrinker=lambda value, predicate: [])


def test_mismatch_shrinking_does_not_substitute_backend_input_errors():
    report = compare_models(
        lambda value: value[0],
        lambda value: value[0] + 1,
        [[3, 9]],
        shrink_failures=True,
    )
    assert not report.passed
    assert report.results[0].shrunk_input
    assert report.results[0].localization["path"] == "$"
    assert "shrunk_input" in report.to_dict()["cases"][0]


def test_localizer_failure_is_diagnostic_and_does_not_abort_the_report():
    def broken_localizer(**kwargs):
        raise RuntimeError("probe failed")

    report = compare_models(lambda x: x, lambda x: x + 1, [1], localizer=broken_localizer)
    assert not report.passed
    assert "probe failed" in report.results[0].localization["error"]
