import json

import numpy as np
import pytest

from mlforensics.parity.inputs import (
    InputSpec,
    generate_input_cases,
    generate_representative_inputs,
)


def test_precision_aliases_generate_the_requested_host_precision():
    assert generate_representative_inputs(shape=(3,), dtype="fp32", count=1)[0].dtype == np.float32
    assert generate_representative_inputs(shape=(3,), dtype="fp16", count=1)[0].dtype == np.float16


def test_bfloat16_is_generated_as_portable_rounded_float32():
    value = generate_representative_inputs(shape=(3,), dtype="bf16", count=4)[-1]
    assert value.dtype == np.float32
    bits = value.view(np.uint32)
    assert np.all((bits & np.uint32(0xFFFF)) == 0)


def test_named_json_specs_support_named_and_sequence_forms(tmp_path):
    named = InputSpec.from_json(
        {
            "inputs": {
                "tokens": {"dtype": "int64", "shape": ["batch", 4], "dynamic": {"batch": [2]}},
                "mask": {"dtype": "fp16", "shape": ["batch", 4], "dynamic": {"batch": [2]}},
            }
        }
    )
    assert set(named) == {"tokens", "mask"}
    assert named["tokens"].resolve_shape(seed=99) == (2, 4)

    path = tmp_path / "input-spec.json"
    path.write_text(json.dumps([{"name": "x", "shape": [2]}, {"name": "y", "shape": [1]}]))
    parsed = InputSpec.from_json(path)
    assert list(parsed) == ["x", "y"]


def test_multi_input_cases_are_synchronized_and_preserve_each_contract():
    cases = generate_input_cases(
        input_spec={
            "left": {"shape": [2], "dtype": "fp16"},
            "right": {"shape": [3], "dtype": "bf16"},
        },
        representative_count=3,
        include_edge=False,
        include_adversarial=False,
    )
    assert len(cases) == 3
    assert set(cases[0].value) == {"left", "right"}
    assert cases[0].value["left"].dtype == np.float16
    assert cases[0].value["right"].dtype == np.float32
    assert cases[0].identity == "representative:0"


def test_dynamic_shape_resolution_is_seeded_and_validates_missing_choices():
    spec = InputSpec(shape=["batch", 2], dynamic_dims={"batch": [1, 3]})
    assert spec.resolve_shape(seed=4) == spec.resolve_shape(seed=4)
    with pytest.raises(ValueError, match="has no configured choices"):
        InputSpec(shape=["batch", 2]).resolve_shape()


@pytest.mark.parametrize(
    "bad",
    [
        {"dtype": "float128"},
        {"shape": [2, -1]},
        {"shape": ["batch"], "dynamic_dims": {"other": [1]}},
        {"shape": [2], "dynamic_dims": {0: []}},
        {"distribution": "made-up"},
        {"unknown": True},
    ],
)
def test_malformed_specs_fail_with_actionable_validation_errors(bad):
    with pytest.raises((TypeError, ValueError)):
        InputSpec.from_dict(bad)


def test_static_named_values_are_cast_and_structured_values_remain_mappings():
    cases = generate_input_cases(
        input_spec={"features": {"shape": [2], "dtype": "fp16", "value": [1, 2]}},
        include_edge=False,
        include_adversarial=False,
    )
    assert cases[0].value["features"].dtype == np.float16

    structured = generate_input_cases(
        input_spec=InputSpec(name="record", metadata={"value": {"id": 7, "text": "ok"}}),
        include_edge=False,
        include_adversarial=False,
    )
    assert structured[0].value == {"record": {"id": 7, "text": "ok"}}
