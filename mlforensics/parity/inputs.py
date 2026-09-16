"""Deterministic representative and adversarial input generation."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # NumPy is useful, but is not a required parity dependency.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised in minimal installations
    _np = None


@dataclass(frozen=True)
class InputSpec:
    """Portable contract for generated and named model inputs."""

    name: str = "input"
    dtype: Any = "float32"
    shape: Sequence[int | str | None] | None = None
    dynamic_dims: Mapping[int | str, Sequence[int]] = field(default_factory=dict)
    value_range: tuple[float, float] | None = None
    distribution: str = "uniform"
    semantic_constraints: Sequence[str] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("input spec name must be a non-empty string")
        if self.shape is not None:
            raw_shape = (self.shape,) if isinstance(self.shape, int) else self.shape
            shape = tuple(
                None if item is None else str(item) if isinstance(item, str) else int(item)
                for item in raw_shape
            )
            if any(
                isinstance(item, int) and item < 0 or isinstance(item, str) and not item.strip()
                for item in shape
            ):
                raise ValueError("input spec dimensions cannot be negative")
            object.__setattr__(self, "shape", shape)
        dynamic: dict[int | str, tuple[int, ...]] = {}
        for key, values in dict(self.dynamic_dims).items():
            raw_values = (values,) if isinstance(values, int) else values
            normalized = tuple(int(value) for value in raw_values)
            if not normalized or any(value < 0 for value in normalized):
                raise ValueError("dynamic dimension choices must be non-negative")
            dynamic[key] = normalized
        object.__setattr__(self, "dynamic_dims", dynamic)
        if self.value_range is not None:
            try:
                raw_range = tuple(self.value_range)
            except TypeError as exc:
                raise ValueError("value_range must be an ordered pair") from exc
            if len(raw_range) != 2 or float(raw_range[0]) > float(raw_range[1]):
                raise ValueError("value_range must be an ordered pair")
            if not all(_finite_number(item) for item in raw_range):
                raise ValueError("value_range must contain finite numbers")
            object.__setattr__(self, "value_range", (float(raw_range[0]), float(raw_range[1])))
        if not isinstance(self.distribution, str) or not self.distribution.strip():
            raise ValueError("distribution must be a non-empty string")
        constraints = (
            (self.semantic_constraints,)
            if isinstance(self.semantic_constraints, str)
            else self.semantic_constraints
        )
        object.__setattr__(self, "semantic_constraints", tuple(str(item) for item in constraints))
        if not isinstance(self.metadata, Mapping):
            raise ValueError("input spec metadata must be a mapping")
        try:
            json.dumps(self.metadata, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("input spec metadata must be JSON-compatible") from exc
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InputSpec:
        if not isinstance(value, Mapping):
            raise TypeError("input spec must be a mapping")
        raw_range = value.get("value_range", value.get("range"))
        return cls(
            name=str(value.get("name", "input")),
            dtype=value.get("dtype", "float32"),
            shape=value.get("shape"),
            dynamic_dims=value.get("dynamic_dims", value.get("dynamic", {})) or {},
            value_range=tuple(raw_range) if raw_range is not None else None,
            distribution=str(value.get("distribution", "uniform")),
            semantic_constraints=value.get("semantic_constraints", value.get("constraints", ()))
            or (),
            metadata=value.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, value: str | Path | Mapping[str, Any]) -> InputSpec | dict[str, InputSpec]:
        if isinstance(value, Mapping):
            if "inputs" in value and isinstance(value["inputs"], Mapping):
                return {
                    str(name): spec
                    if isinstance(spec, cls)
                    else cls.from_dict({"name": name, **dict(spec)})
                    if isinstance(spec, Mapping)
                    else cls(name=str(name), metadata={"value": spec})
                    for name, spec in value["inputs"].items()
                }
            return cls.from_dict(value)
        if isinstance(value, (list, tuple)):
            return {
                spec.name: spec
                for index, item in enumerate(value)
                for spec in [
                    cls.from_dict({"name": f"input-{index}", **dict(item)})
                    if isinstance(item, Mapping)
                    else cls(name=f"input-{index}", metadata={"value": item})
                ]
            }
        raw_text = str(value)
        if raw_text.lstrip().startswith(("{", "[")):
            raw = json.loads(raw_text)
        else:
            path = Path(raw_text)
            raw = json.loads(path.read_text(encoding="utf-8") if path.exists() else raw_text)
        return cls.from_json(raw)

    def resolve_shape(self, *, seed: int = 0) -> tuple[int, ...] | None:
        if self.shape is None:
            return None
        rng = random.Random(seed)
        result: list[int] = []
        for index, dimension in enumerate(self.shape):
            choices = self.dynamic_dims.get(index, self.dynamic_dims.get(str(index), ()))
            if not choices and isinstance(dimension, str):
                choices = self.dynamic_dims.get(dimension, ())
            if choices:
                result.append(int(choices[rng.randrange(len(choices))]))
            elif dimension is None:
                result.append(1)
            elif isinstance(dimension, str):
                raise ValueError(f"dynamic dimension {dimension!r} has no configured choices")
            else:
                result.append(int(dimension))
        return tuple(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": str(self.dtype),
            "shape": list(self.shape) if self.shape is not None else None,
            "dynamic_dims": {str(key): list(value) for key, value in self.dynamic_dims.items()},
            "value_range": list(self.value_range) if self.value_range is not None else None,
            "distribution": self.distribution,
            "semantic_constraints": list(self.semantic_constraints),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class InputCase:
    """An input plus stable metadata used in reports."""

    value: Any
    label: str = "input"
    category: str = "custom"
    metadata: dict[str, Any] = field(default_factory=dict)
    identity: Any = None

    @property
    def case_id(self) -> Any:
        return self.identity if self.identity is not None else self.label

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "category": self.category,
            "value": self.value,
            "metadata": dict(self.metadata),
            "identity": self.identity,
        }


def _normal_shape(shape: int | Sequence[int] | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    if isinstance(shape, int):
        if shape < 0:
            raise ValueError("shape dimensions cannot be negative")
        return (shape,)
    normalized = tuple(int(size) for size in shape)
    if any(size < 0 for size in normalized):
        raise ValueError("shape dimensions cannot be negative")
    return normalized


def _normalized_dtype(dtype: Any) -> Any:
    if isinstance(dtype, str):
        normalized = dtype.casefold().replace("numpy.", "").replace("torch.", "")
        aliases = {
            "float": "float32",
            "double": "float64",
            "half": "float16",
            "bf16": "float32",
            "long": "int64",
            "int": "int32",
        }
        value = aliases.get(normalized, normalized)
        if _np is not None:
            try:
                return _np.dtype(value)
            except TypeError:
                pass
        return value
    return dtype


def _finite_number(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _apply_constraints(value: Any, spec: InputSpec | None) -> Any:
    if spec is None:
        return value
    constraints = {item.casefold().replace("_", "-") for item in spec.semantic_constraints}
    if not constraints:
        return value
    if _np is not None:
        try:
            array = _np.asarray(value)
            if "non-negative" in constraints or "nonnegative" in constraints:
                array = _np.abs(array)
            if "integer" in constraints or "categorical" in constraints:
                array = _np.rint(array)
            if "one-hot" in constraints:
                array = _np.zeros_like(array)
                if array.size:
                    array.flat[0] = 1
            return array.astype(_normalized_dtype(spec.dtype), copy=False)
        except (TypeError, ValueError, OverflowError):
            pass
    return value


def _make_value(value: Any, shape: tuple[int, ...] | None, dtype: Any = float) -> Any:
    dtype = _normalized_dtype(dtype)
    if shape is None:
        return value
    size = 1
    for dim in shape:
        size *= dim
    values = [value] * size
    if _np is not None:
        try:
            return _np.asarray(values, dtype=dtype).reshape(shape)
        except (TypeError, ValueError, OverflowError):
            # Integer dtypes cannot represent +/-inf or nan.  Preserve the
            # edge case as a float array instead of making generation fail.
            return _np.asarray(values, dtype=float).reshape(shape)
    if len(shape) == 1:
        return values

    def nest(items: list[Any], dims: tuple[int, ...]) -> Any:
        if len(dims) == 1:
            return items[: dims[0]]
        stride = 1
        for dim in dims[1:]:
            stride *= dim
        return [nest(items[i * stride : (i + 1) * stride], dims[1:]) for i in range(dims[0])]

    return nest(values, shape)


def _random_value(
    rng: random.Random,
    shape: tuple[int, ...] | None,
    low: float,
    high: float,
    dtype: Any = float,
    distribution: str = "uniform",
) -> Any:
    dtype = _normalized_dtype(dtype)
    distribution_name = str(distribution).casefold().replace("_", "-")

    def draw() -> float:
        if distribution_name in {"uniform", "flat"}:
            return rng.uniform(low, high)
        if distribution_name in {"normal", "gaussian"}:
            mean = (low + high) / 2.0
            deviation = (high - low) / 6.0 or 1.0
            return max(low, min(high, rng.gauss(mean, deviation)))
        if distribution_name in {"log-normal", "lognormal"}:
            positive_low = max(abs(low), 1e-6)
            positive_high = max(abs(high), positive_low)
            mean = (math.log(positive_low) + math.log(positive_high)) / 2.0
            deviation = (math.log(positive_high) - math.log(positive_low)) / 6.0 or 1.0
            return min(positive_high, max(positive_low, rng.lognormvariate(mean, deviation)))
        if distribution_name in {"zeros", "zero"}:
            return 0.0
        if distribution_name in {"ones", "one"}:
            return 1.0
        raise ValueError(f"unsupported input distribution: {distribution}")

    if shape is None:
        return draw()
    size = 1
    for dim in shape:
        size *= dim
    values = [draw() for _ in range(size)]
    if _np is not None:
        return _np.asarray(values, dtype=dtype).reshape(shape)
    return values


def _dedupe(values: Iterable[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        duplicate = False
        for prior in result:
            try:
                if _np is not None and isinstance(value, _np.ndarray):
                    duplicate = bool(_np.array_equal(value, prior, equal_nan=True))
                else:
                    duplicate = value == prior
                    if not isinstance(duplicate, bool):
                        duplicate = bool(_np.all(duplicate)) if _np is not None else False
            except (TypeError, ValueError):
                duplicate = False
            if duplicate:
                break
        if not duplicate:
            result.append(value)
    return result


def _same_value(left: Any, right: Any) -> bool:
    try:
        if _np is not None and (isinstance(left, _np.ndarray) or isinstance(right, _np.ndarray)):
            return bool(_np.array_equal(left, right, equal_nan=True))
        result = left == right
        if isinstance(result, bool):
            return result
        return bool(_np.all(result)) if _np is not None else False
    except (TypeError, ValueError):
        return False


def generate_representative_inputs(
    shape: int | Sequence[int] | None = None,
    dtype: Any = float,
    count: int = 8,
    seed: int = 0,
    low: float = -1.0,
    high: float = 1.0,
    distribution: str = "uniform",
) -> list[Any]:
    """Generate stable typical values for a model input shape."""

    if count < 0:
        raise ValueError("count must be non-negative")
    normalized = _normal_shape(shape)
    constants = [
        _make_value(0, normalized, dtype),
        _make_value(1, normalized, dtype),
        _make_value(-1, normalized, dtype),
    ]
    rng = random.Random(seed)
    values = _dedupe(
        constants
        + [
            _random_value(rng, normalized, low, high, dtype, distribution)
            for _ in range(max(0, count - len(constants)))
        ]
    )
    return values[:count]


def generate_edge_inputs(
    shape: int | Sequence[int] | None = None,
    dtype: Any = float,
    include_non_finite: bool = False,
    count: int | None = None,
    seed: int = 0,
    low: float = -1.0,
    high: float = 1.0,
    distribution: str = "uniform",
) -> list[Any]:
    """Generate deterministic numerical boundary inputs.

    Non-finite values are opt-in because many production models reject them.
    """

    normalized = _normal_shape(shape)
    values: list[Any] = [
        _make_value(1e-12, normalized, dtype),
        _make_value(-1e-12, normalized, dtype),
        _make_value(1e6, normalized, dtype),
        _make_value(-1e6, normalized, dtype),
    ]
    if normalized and normalized[0] == 0:
        values.insert(0, _make_value(0, normalized, dtype))
    if include_non_finite:
        values.extend(
            [
                _make_value(float("inf"), normalized, dtype),
                _make_value(float("-inf"), normalized, dtype),
                _make_value(float("nan"), normalized, dtype),
            ]
        )
    values = _dedupe(values)
    if count is None:
        return values
    if count < 0:
        raise ValueError("count must be non-negative")
    if count <= len(values):
        return values[:count]
    rng = random.Random(seed)
    while len(values) < count:
        values.append(_random_value(rng, normalized, low, high, dtype, distribution))
    return values


def generate_adversarial_inputs(
    shape: int | Sequence[int] | None = None,
    dtype: Any = float,
    seed: int = 0,
    include_non_finite: bool = True,
    distribution: str = "uniform",
) -> list[Any]:
    """Generate inputs likely to expose backend numerical or shape drift."""

    normalized = _normal_shape(shape)
    edges = generate_edge_inputs(
        shape=normalized, dtype=dtype, include_non_finite=include_non_finite
    )
    if normalized is None:
        alternating: Any = -1.0
    else:
        size = 1
        for dim in normalized:
            size *= dim
        values = [1.0 if index % 2 == 0 else -1.0 for index in range(size)]
        if _np is not None:
            alternating = _np.asarray(values, dtype=dtype).reshape(normalized)
        else:
            alternating = values
    rng = random.Random(seed)
    near_boundary = _random_value(rng, normalized, -1e-7, 1e-7, dtype, distribution)
    return _dedupe(edges + [alternating, near_boundary])


def generate_input_cases(
    *,
    examples: Iterable[Any] | None = None,
    shape: int | Sequence[int] | None = None,
    dtype: Any = float,
    seed: int = 0,
    representative_count: int = 8,
    include_adversarial: bool = True,
    include_edge: bool = True,
    include_non_finite: bool = False,
    input_spec: InputSpec
    | Mapping[str, InputSpec | Mapping[str, Any]]
    | Mapping[str, Any]
    | None = None,
) -> list[InputCase]:
    """Build labeled cases suitable for a parity report."""

    cases: list[InputCase] = []
    if examples is not None:
        cases.extend(
            InputCase(value, f"example-{i}", "example", identity=f"example:{i}")
            for i, value in enumerate(examples)
        )

    def append_unique(values: Iterable[Any], category: str) -> None:
        category_index = 0
        for value in values:
            if any(_same_value(value, case.value) for case in cases):
                continue
            cases.append(
                InputCase(
                    value,
                    f"{category}-{category_index}",
                    category,
                    identity=f"{category}:{category_index}",
                )
            )
            category_index += 1

    normalized_specs: dict[str, InputSpec] = {}
    if input_spec is not None:
        if isinstance(input_spec, InputSpec):
            normalized_specs[input_spec.name] = input_spec
        elif isinstance(input_spec, Mapping):
            if "inputs" in input_spec and isinstance(input_spec["inputs"], Mapping):
                named_specs = input_spec["inputs"]
                for name, raw in named_specs.items():
                    spec = (
                        raw
                        if isinstance(raw, InputSpec)
                        else InputSpec.from_dict({"name": name, **dict(raw)})
                        if isinstance(raw, Mapping)
                        else InputSpec(name=str(name), metadata={"value": raw})
                    )
                    normalized_specs[str(name)] = spec
            elif any(key in input_spec for key in ("shape", "dtype", "name", "dynamic_dims")):
                spec = InputSpec.from_dict(input_spec)
                normalized_specs[spec.name] = spec
            else:
                for name, raw in input_spec.items():
                    spec = (
                        raw
                        if isinstance(raw, InputSpec)
                        else InputSpec.from_dict({"name": name, **dict(raw)})
                    )
                    normalized_specs[str(name)] = spec
        else:
            raise TypeError("input_spec must be an InputSpec or mapping")

    def generated_for_spec(spec: InputSpec | None, category: str) -> list[Any]:
        local_shape = spec.resolve_shape(seed=seed) if spec is not None else shape
        local_dtype = spec.dtype if spec is not None else dtype
        if spec is not None and "value" in spec.metadata:
            return [_apply_constraints(spec.metadata["value"], spec)]
        low, high = (
            spec.value_range if spec is not None and spec.value_range is not None else (-1.0, 1.0)
        )
        distribution = spec.distribution if spec is not None else "uniform"
        if category == "representative":
            values = generate_representative_inputs(
                shape=local_shape,
                dtype=local_dtype,
                count=representative_count,
                seed=seed,
                low=low,
                high=high,
                distribution=distribution,
            )
        elif category == "edge":
            values = generate_edge_inputs(
                shape=local_shape,
                dtype=local_dtype,
                include_non_finite=include_non_finite,
                low=low,
                high=high,
                distribution=distribution,
            )
        else:
            values = generate_adversarial_inputs(
                shape=local_shape,
                dtype=local_dtype,
                seed=seed,
                include_non_finite=include_non_finite,
                distribution=distribution,
            )
        return [_apply_constraints(value, spec) for value in values]

    def wrap(values: Iterable[Any], category: str = "representative") -> Iterable[Any]:
        if not normalized_specs:
            return values
        if len(normalized_specs) == 1:
            name = next(iter(normalized_specs))
            return ({name: value} for value in values)
        # Multi-input generation keeps the case identity aligned across all
        # named inputs while retaining each input's dtype and shape contract.
        generated = {
            name: list(generated_for_spec(spec, category))
            for name, spec in normalized_specs.items()
        }
        count = min(len(value) for value in generated.values()) if generated else 0
        return ({name: generated[name][index] for name in generated} for index in range(count))

    if normalized_specs and len(normalized_specs) > 1:
        # ``wrap`` already constructs the synchronized multi-input cases.
        append_unique(wrap((), "representative"), "representative")
        if include_edge:
            append_unique(wrap((), "edge"), "edge")
        if include_adversarial:
            append_unique(wrap((), "adversarial"), "adversarial")
    else:
        single_spec = next(iter(normalized_specs.values()), None)
        append_unique(wrap(generated_for_spec(single_spec, "representative")), "representative")
        if include_edge:
            append_unique(wrap(generated_for_spec(single_spec, "edge")), "edge")
        if include_adversarial:
            append_unique(wrap(generated_for_spec(single_spec, "adversarial")), "adversarial")
    return cases


generate_inputs = generate_input_cases
