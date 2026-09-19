import importlib.util

import pytest

from mlforensics.parity.backends import (
    BackendSpec,
    CallableBackendAdapter,
    OpenVINOBackendAdapter,
    OptionalDependencyError,
    PyTorchBackendAdapter,
    TensorRTBackendAdapter,
    export_eager_pytorch,
    load_backend,
    parse_backend_spec,
)


def test_named_callable_inputs_and_adapter_managed_state():
    def model(*, left, right, state):
        return {"value": left + right + state, "next_state": state + 1}

    adapter = CallableBackendAdapter(
        model,
        unpack_inputs=True,
        stateful=True,
        state_input_name="state",
        state_output_key="next_state",
        return_state=False,
        initial_state=0,
    )

    assert adapter.predict({"left": 1, "right": 2}) == {"value": 3}
    assert adapter.state == 1
    assert adapter.predict({"left": 1, "right": 2}) == {"value": 4}
    adapter.reset_state()
    assert adapter.state == 0


def test_backend_specs_include_safe_eager_and_optional_runtime_kinds():
    assert parse_backend_spec("model.pt2").kind == "torch-export"
    assert parse_backend_spec("model.engine").kind == "tensorrt"
    assert parse_backend_spec("model.xml").kind == "openvino"
    assert parse_backend_spec({"kind": "torch-export", "source": "model.pt2"}) == BackendSpec(
        "torch-export", "model.pt2"
    )


def test_pytorch_precision_and_positional_multi_input_calls():
    torch = pytest.importorskip("torch")

    class Add(torch.nn.Module):
        def forward(self, left, right):
            return left + right

    adapter = PyTorchBackendAdapter(Add(), dtype="bf16", unpack_inputs=True)
    output = adapter.predict(([1.0, 2.0], [3.0, 4.0]))
    assert output.dtype == torch.bfloat16
    assert output.tolist() == [4.0, 6.0]


def test_pytorch_structured_inputs_are_recursively_moved():
    torch = pytest.importorskip("torch")

    class First(torch.nn.Module):
        def forward(self, payload):
            return payload["tokens"][0] + payload["bias"]

    adapter = PyTorchBackendAdapter(First(), structured_inputs=True, dtype="float32")
    output = adapter.predict({"tokens": [[1.0, 2.0]], "bias": 2.0})
    assert output.tolist() == [3.0, 4.0]


def test_eager_export_round_trip_does_not_call_torch_load(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    class AddOne(torch.nn.Module):
        def forward(self, value):
            return value + 1

    artifact = tmp_path / "add-one.pt2"
    export_eager_pytorch(AddOne(), artifact, torch.ones(2))
    original_load = torch.load

    def safe_load(*args, **kwargs):
        assert kwargs.get("weights_only") is True
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", safe_load)

    adapter = load_backend(artifact)
    assert adapter.predict([1.0, 2.0]).tolist() == [2.0, 3.0]


def test_tensorrt_and_openvino_existing_wrappers_are_dependency_free():
    class Engine:
        def infer(self, feed):
            return feed["x"]

    class CompiledModel:
        inputs = []

        def __call__(self, feed):
            return {"output": feed["x"]}

    assert TensorRTBackendAdapter(Engine(), input_names=("x",)).predict([1, 2]).tolist() == [1, 2]
    assert OpenVINOBackendAdapter(CompiledModel(), input_names=("x",)).predict([1, 2]).tolist() == [
        1,
        2,
    ]


@pytest.mark.parametrize(
    ("adapter", "package", "label"),
    [
        (TensorRTBackendAdapter, "tensorrt", "TensorRT"),
        (OpenVINOBackendAdapter, "openvino", "OpenVINO"),
    ],
)
def test_optional_runtime_errors_are_lazy_and_actionable(tmp_path, adapter, package, label):
    if importlib.util.find_spec(package) is not None:
        pytest.skip(f"{package} is installed")
    path = tmp_path / ("model.engine" if package == "tensorrt" else "model.xml")
    path.write_bytes(b"not a real model")
    with pytest.raises(OptionalDependencyError, match=label):
        adapter(path)
