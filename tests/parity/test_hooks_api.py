import pytest

from mlforensics.parity import (
    IntermediateRecorder,
    UnsupportedIntermediateLocalization,
    attach_torch_module_hooks,
    capture_callable_intermediates,
    first_intermediate_divergence,
    instrument_callable,
)


def test_callable_hook_captures_bounded_intermediates():
    def model(value, intermediate_hook):
        doubled = value * 2
        intermediate_hook("double", doubled)
        intermediate_hook("triple", value * 3)
        return doubled

    capture = capture_callable_intermediates(model, 4, max_records=1)
    assert capture.output == 8
    assert [record.name for record in capture.records] == ["triple"]
    assert capture.recorder.dropped_count == 1


def test_callable_without_opt_in_reports_unsupported_protocol():
    with pytest.raises(UnsupportedIntermediateLocalization, match="intermediate_hook"):
        instrument_callable(lambda value: value + 1)


def test_recorder_freezes_and_serializes_evidence():
    recorder = IntermediateRecorder(max_records=4, max_elements=2)
    record = recorder.record("layer", [1, 2, 3])
    assert record is not None
    assert record.truncated
    recorder.freeze()
    assert recorder.record("ignored", 1) is None
    evidence = recorder.to_dict()
    assert evidence["dropped_count"] == 1
    assert len(evidence["records"]) == 1


def test_first_intermediate_divergence_reports_layer_and_mismatch():
    reference = IntermediateRecorder()
    candidate = IntermediateRecorder()
    reference.record("encoder", [1.0, 2.0])
    candidate.record("encoder", [1.0, 2.5])
    location = first_intermediate_divergence(reference, candidate)
    assert location is not None
    assert location.path == "encoder"
    assert location.component == "intermediate"
    assert location.details["mismatch_paths"] == ["$[1]"]


def test_torch_module_hooks_capture_named_forward_outputs():
    torch = pytest.importorskip("torch")

    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
    with attach_torch_module_hooks(model) as handles:
        model(torch.ones(1, 2))
    assert [record.name for record in handles.records] == ["0", "1"]
