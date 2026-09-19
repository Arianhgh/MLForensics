import pytest

from mlforensics.parity.export import (
    ExportInputBundle,
    dynamic_shapes_from_specs,
    export_torch_model,
    load_exported_module,
    load_exported_program,
    prepare_export_inputs,
    save_exported_program,
)
from mlforensics.parity.inputs import InputSpec
from mlforensics.parity.stateful import StatefulModelAdapter

torch = pytest.importorskip("torch")


def test_export_input_normalization_supports_named_and_structured_inputs():
    named = prepare_export_inputs((1, 2), input_names=("left", "right"))
    assert named.args == ()
    assert named.kwargs == {"left": 1, "right": 2}

    structured = prepare_export_inputs({"tokens": [1, 2]}, structured_inputs=True)
    assert structured.args == (({"tokens": [1, 2]},))
    assert structured.kwargs == {}

    bundle = ExportInputBundle((1,), {"scale": 2})
    assert prepare_export_inputs(bundle) is bundle


def test_eager_export_round_trip_is_safe_and_supports_multi_input(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, x, y):
            return {"sum": x + y, "product": x * y}

    destination = tmp_path / "model.pt2"
    exported = export_torch_model(
        Model(),
        {"x": torch.ones(2), "y": torch.full((2,), 3.0)},
        destination,
        metadata={"kind": "test", "version": 1},
    )
    assert destination.is_file()
    loaded, metadata = load_exported_program(destination, with_metadata=True)
    assert metadata == {"kind": "test", "version": 1}
    output = loaded.module()(x=torch.ones(2), y=torch.full((2,), 3.0))
    assert torch.equal(output["sum"], torch.full((2,), 4.0))
    loaded_output = load_exported_module(destination)(x=torch.ones(2), y=torch.full((2,), 3.0))
    assert torch.equal(loaded_output["product"], torch.full((2,), 3.0))
    assert exported is not None


def test_dynamic_shapes_can_be_derived_from_input_specs():
    spec = InputSpec(name="x", shape=("batch", 3), dynamic_dims={"batch": (1, 4)})
    shapes = dynamic_shapes_from_specs(spec)
    assert isinstance(shapes, tuple)
    assert shapes[0][0].min == 1
    assert shapes[0][0].max == 4

    class Model(torch.nn.Module):
        def forward(self, x):
            return x + 1

    exported = export_torch_model(
        Model(),
        torch.ones(2, 3),
        input_specs=spec,
    )
    assert torch.equal(exported.module()(torch.ones(4, 3)), torch.full((4, 3), 2.0))


def test_named_dynamic_shape_specs_follow_keyword_inputs():
    specs = {
        "x": InputSpec(name="x", shape=("batch", 2), dynamic_dims={"batch": (1, 4)}),
        "y": InputSpec(name="y", shape=("batch", 2), dynamic_dims={"batch": (1, 4)}),
    }
    shapes = dynamic_shapes_from_specs(specs)
    assert set(shapes) == {"x", "y"}

    class Model(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    exported = export_torch_model(
        Model(),
        {"x": torch.ones(2, 2), "y": torch.ones(2, 2)},
        input_specs=specs,
    )
    assert exported.module()(x=torch.ones(4, 2), y=torch.ones(4, 2)).shape == (4, 2)


def test_stateful_adapter_carries_and_resets_state():
    def model(value, state):
        next_state = state + value
        return value * 2, next_state

    adapter = StatefulModelAdapter(model, initial_state=0)
    assert adapter.predict(3) == 6
    assert adapter.state == 3
    assert adapter.predict(4) == 8
    assert adapter.state == 7
    adapter.reset_state(10)
    assert adapter.predict(1) == 2
    assert adapter.state == 11


def test_stateful_adapter_supports_named_structured_outputs():
    def model(inputs):
        return {"value": inputs["x"] + inputs["state"], "state": inputs["state"] + 1}

    adapter = StatefulModelAdapter(
        model,
        initial_state=5,
        state_output_key="state",
        state_input_name="state",
    )
    assert adapter.predict({"x": 2}) == {"value": 7}
    assert adapter.state == 6


def test_stateful_adapter_adds_state_to_named_positional_inputs():
    def model(x, y, state):
        return x + y, state + 1

    adapter = StatefulModelAdapter(model, initial_state=4, input_names=("x", "y"))
    assert adapter.predict((2, 3)) == 5
    assert adapter.state == 5


def test_save_helper_rejects_non_json_metadata(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, x):
            return x

    exported = export_torch_model(Model(), torch.ones(1))
    with pytest.raises(ValueError, match="JSON-compatible"):
        save_exported_program(exported, tmp_path / "bad.pt2", metadata={"value": object()})
