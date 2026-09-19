"""Deterministic representative and adversarial input generation."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

try:  # NumPy is useful, but is not a required parity dependency.
    import numpy as _np
except ImportError:  # pragma: no cover - exercised in minimal installations
    _np = None


_SUPPORTED_DTYPES = frozenset(
    {
        "bool",
        "bfloat16",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "complex64",
        "complex128",
    }
)
_SUPPORTED_DISTRIBUTIONS = frozenset(
    {
        "uniform",
        "flat",
        "normal",
        "gaussian",
        "log-normal",
        "lognormal",
        "zeros",
        "zero",
        "ones",
        "one",
    }
)
_SPEC_FIELDS = frozenset(
    {
        "name",
        "dtype",
        "shape",
        "dynamic_dims",
        "dynamic",
        "value_range",
        "range",
        "distribution",
        "semantic_constraints",
        "constraints",
        "metadata",
        "value",
    }
)


def _canonical_dtype_name(dtype: Any) -> str:
    """Return a portable dtype name, including aliases used by model runtimes."""

    if dtype is float:
        normalized = "float64"
    elif dtype is int:
        normalized = "int32"
    elif dtype is bool:
        normalized = "bool"
    elif isinstance(dtype, str):
        normalized = dtype.strip().casefold().replace(" ", "")
        normalized = normalized.removeprefix("numpy.").removeprefix("np.")
        normalized = normalized.removeprefix("torch.")
    elif _np is not None:
        try:
            normalized = str(_np.dtype(dtype)).casefold()
        except (TypeError, ValueError):
            normalized = str(dtype).strip().casefold().replace("torch.", "")
    else:
        normalized = str(dtype).strip().casefold().replace("torch.", "")
    aliases = {
        "float": "float32",
        "single": "float32",
        "fp32": "float32",
        "double": "float64",
        "half": "float16",
        "fp16": "float16",
        "bf16": "bfloat16",
        "bfloat": "bfloat16",
        "long": "int64",
        "int": "int32",
    }
    return aliases.get(normalized, normalized)


def _is_spec_mapping(value: Mapping[str, Any]) -> bool:
    return bool(set(value).intersection(_SPEC_FIELDS))


def _canonical_distribution_name(distribution: Any) -> str:
    if not isinstance(distribution, str) or not distribution.strip():
        raise ValueError("distribution must be a non-empty string")
    normalized = distribution.strip().casefold().replace("_", "-")
    if normalized not in _SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"unsupported input distribution: {distribution}")
    return normalized


def _normalize_shape_dimension(item: Any) -> int | str | None:
    if item is None:
        return None
    if isinstance(item, str):
        stripped = item.strip()
        if not stripped:
            raise ValueError("input spec dimensions cannot be empty")
        try:
            return int(stripped)
        except ValueError:
            return stripped
    if isinstance(item, bool) or not isinstance(item, Integral):
        raise ValueError("input spec dimensions must be integers, names, or null")
    return int(item)


def _named_specs(spec_class: type[InputSpec], values: Mapping[Any, Any]) -> dict[str, InputSpec]:
    """Parse ``{"input-name": spec}`` without silently changing names."""

    result: dict[str, InputSpec] = {}
    for raw_name, raw_spec in values.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError("named input specifications require non-empty names")
        if isinstance(raw_spec, spec_class):
            spec = raw_spec
            if spec.name != name:
                raise ValueError(f"input specification name {spec.name!r} does not match {name!r}")
        elif isinstance(raw_spec, Mapping):
            spec_data = dict(raw_spec)
            declared_name = spec_data.get("name", name)
            if declared_name != name:
                raise ValueError(
                    f"input specification name {declared_name!r} does not match {name!r}"
                )
            spec_data["name"] = name
            spec = spec_class.from_dict(spec_data)
        else:
            spec = spec_class(name=name, metadata={"value": raw_spec})
        if name in result:
            raise ValueError(f"duplicate input specification name: {name!r}")
        result[name] = spec
    if not result:
        raise ValueError("input specification must define at least one input")
    return result


def _sequence_specs(spec_class: type[InputSpec], values: Sequence[Any]) -> dict[str, InputSpec]:
    result: dict[str, InputSpec] = {}
    for index, raw_spec in enumerate(values):
        default_name = f"input-{index}"
        if isinstance(raw_spec, spec_class):
            spec = raw_spec
        elif isinstance(raw_spec, Mapping):
            spec_data = dict(raw_spec)
            spec_data.setdefault("name", default_name)
            spec = spec_class.from_dict(spec_data)
        else:
            spec = spec_class(name=default_name, metadata={"value": raw_spec})
        if spec.name in result:
            raise ValueError(f"duplicate input specification name: {spec.name!r}")
        result[spec.name] = spec
    if not result:
        raise ValueError("input specification must define at least one input")
    return result


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
        object.__setattr__(self, "name", self.name.strip())
        if self.shape is not None:
            if isinstance(self.shape, str):
                parts = self.shape.split(",")
                if not self.shape.strip() or any(not part.strip() for part in parts):
                    raise ValueError("input spec shape must contain non-empty dimensions")
                raw_shape = tuple(part.strip() for part in parts)
            else:
                if isinstance(self.shape, (str, bytes)):
                    raise ValueError("input spec shape must be a sequence of dimensions")
                raw_shape = (self.shape,) if isinstance(self.shape, Integral) else self.shape
                try:
                    raw_shape = tuple(raw_shape)
                except TypeError as exc:
                    raise ValueError("input spec shape must be a sequence of dimensions") from exc
            shape = tuple(_normalize_shape_dimension(item) for item in raw_shape)
            if any(isinstance(item, int) and item < 0 for item in shape):
                raise ValueError("input spec dimensions cannot be negative")
            object.__setattr__(self, "shape", shape)
        normalized_dtype = _canonical_dtype_name(self.dtype)
        if normalized_dtype not in _SUPPORTED_DTYPES:
            raise ValueError(f"unsupported input dtype: {self.dtype}")
        object.__setattr__(self, "dtype", normalized_dtype)
        dynamic: dict[int | str, tuple[int, ...]] = {}
        if self.dynamic_dims is None:
            raw_dynamic_dims: Mapping[Any, Any] = {}
        elif isinstance(self.dynamic_dims, Mapping):
            raw_dynamic_dims = self.dynamic_dims
        else:
            raise ValueError("dynamic_dims must be a mapping")
        if self.shape is None and raw_dynamic_dims:
            raise ValueError("dynamic_dims require an input shape")
        for key, values in raw_dynamic_dims.items():
            if isinstance(key, bool) or not isinstance(key, (Integral, str)):
                raise ValueError("dynamic dimension keys must be indices or names")
            if isinstance(key, str):
                key = key.strip()
                if not key:
                    raise ValueError("dynamic dimension names cannot be empty")
            else:
                key = int(key)
            if key in dynamic:
                raise ValueError(f"duplicate dynamic dimension key: {key!r}")
            if isinstance(values, (str, bytes)) or values is None:
                raw_values = (values,) if isinstance(values, Integral) else None
            elif isinstance(values, Integral):
                raw_values = (values,)
            else:
                try:
                    raw_values = tuple(values)
                except TypeError:
                    raw_values = None
            if not raw_values:
                raise ValueError("dynamic dimension choices must be a non-empty sequence")
            normalized_values: list[int] = []
            for value in raw_values:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise ValueError("dynamic dimension choices must be integers")
                normalized_values.append(int(value))
            if any(value < 0 for value in normalized_values):
                raise ValueError("dynamic dimension choices must be non-negative")
            dynamic[key] = tuple(normalized_values)
        object.__setattr__(self, "dynamic_dims", dynamic)
        if self.shape is not None:
            dimension_names = {item for item in self.shape if isinstance(item, str)}
            for key in dynamic:
                if isinstance(key, int) and key >= len(self.shape):
                    raise ValueError(f"dynamic dimension index {key} is outside the input shape")
                if isinstance(key, str) and key not in dimension_names:
                    raise ValueError(f"dynamic dimension {key!r} is not present in the input shape")
        if self.value_range is not None:
            try:
                raw_range = tuple(self.value_range)
            except (TypeError, ValueError) as exc:
                raise ValueError("value_range must be an ordered pair") from exc
            if len(raw_range) != 2 or float(raw_range[0]) > float(raw_range[1]):
                raise ValueError("value_range must be an ordered pair")
            if not all(_finite_number(item) for item in raw_range):
                raise ValueError("value_range must contain finite numbers")
            object.__setattr__(self, "value_range", (float(raw_range[0]), float(raw_range[1])))
        object.__setattr__(self, "distribution", _canonical_distribution_name(self.distribution))
        constraints = (
            (self.semantic_constraints,)
            if isinstance(self.semantic_constraints, str)
            else self.semantic_constraints
        )
        if constraints is None or isinstance(constraints, (bytes, Mapping)):
            raise ValueError("semantic_constraints must be a string or sequence of strings")
        try:
            constraints = tuple(constraints)
        except TypeError as exc:
            raise ValueError(
                "semantic_constraints must be a string or sequence of strings"
            ) from exc
        if any(not isinstance(item, str) or not item.strip() for item in constraints):
            raise ValueError("semantic_constraints must contain non-empty strings")
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
        unknown = set(value) - _SPEC_FIELDS
        if unknown:
            raise ValueError(f"unknown input specification fields: {sorted(unknown, key=str)}")
        if "value_range" in value and "range" in value and value["value_range"] != value["range"]:
            raise ValueError("input spec cannot define conflicting value_range and range")
        if (
            "dynamic_dims" in value
            and "dynamic" in value
            and value["dynamic_dims"] != value["dynamic"]
        ):
            raise ValueError("input spec cannot define conflicting dynamic_dims and dynamic")
        if (
            "semantic_constraints" in value
            and "constraints" in value
            and value["semantic_constraints"] != value["constraints"]
        ):
            raise ValueError("input spec cannot define conflicting semantic constraints")
        raw_range = value.get("value_range", value.get("range"))
        metadata = value.get("metadata", {})
        if "value" in value:
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, Mapping):
                raise ValueError("input spec metadata must be a mapping")
            metadata = {**metadata, "value": value["value"]}
        return cls(
            name=value.get("name", "input"),
            dtype=value.get("dtype", "float32"),
            shape=value.get("shape"),
            dynamic_dims=value.get("dynamic_dims", value.get("dynamic", {})) or {},
            value_range=tuple(raw_range) if raw_range is not None else None,
            distribution=value.get("distribution", "uniform"),
            semantic_constraints=value.get("semantic_constraints", value.get("constraints", ()))
            or (),
            metadata=metadata,
        )

    @classmethod
    def from_json(
        cls, value: str | Path | Mapping[str, Any] | Sequence[Any]
    ) -> InputSpec | dict[str, InputSpec]:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            if "inputs" in value:
                raw_inputs = value["inputs"]
                if isinstance(raw_inputs, Mapping):
                    return _named_specs(cls, raw_inputs)
                if isinstance(raw_inputs, Sequence) and not isinstance(raw_inputs, (str, bytes)):
                    return _sequence_specs(cls, raw_inputs)
                raise TypeError("input spec 'inputs' must be a mapping or sequence")
            if _is_spec_mapping(value):
                return cls.from_dict(value)
            return _named_specs(cls, value)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path)):
            return _sequence_specs(cls, value)
        if not isinstance(value, (str, Path)):
            raise TypeError("input spec JSON must be an object, array, path, or JSON string")
        raw_text = str(value)
        if raw_text.lstrip().startswith(("{", "[")):
            try:
                raw = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid input spec JSON: {exc.msg}") from exc
        else:
            path = Path(raw_text)
            try:
                raw_text = path.read_text(encoding="utf-8") if path.exists() else raw_text
                raw = json.loads(raw_text)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid input spec JSON: {exc}") from exc
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


def _normal_shape(shape: int | Sequence[int] | str | None) -> tuple[int, ...] | None:
    if shape is None:
        return None
    if isinstance(shape, bool):
        raise ValueError("shape dimensions must be integers")
    if isinstance(shape, Integral):
        if shape < 0:
            raise ValueError("shape dimensions cannot be negative")
        return (int(shape),)
    if isinstance(shape, str):
        parts = shape.split(",")
        if not shape.strip() or any(not part.strip() for part in parts):
            raise ValueError("shape must contain non-empty dimensions")
        shape = parts
    if isinstance(shape, (bytes, Mapping)):
        raise ValueError("shape must be a sequence of dimensions")
    try:
        raw_shape = tuple(shape)
    except TypeError as exc:
        raise ValueError("shape must be a sequence of dimensions") from exc
    normalized = tuple(_normalize_shape_dimension(size) for size in raw_shape)
    if any(size is None or isinstance(size, str) for size in normalized):
        raise ValueError("generated input shapes must contain concrete integer dimensions")
    if any(size < 0 for size in normalized):
        raise ValueError("shape dimensions cannot be negative")
    return tuple(int(size) for size in normalized)


def _normalized_dtype(dtype: Any) -> Any:
    value = _canonical_dtype_name(dtype)
    if value not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported input dtype: {dtype}")
    if _np is None:
        return value
    # NumPy still has no portable bfloat16 dtype.  Host values use float32,
    # rounded to the BF16 mantissa below, while the InputSpec keeps the
    # contract name for reports and adapters.
    return _np.dtype("float32" if value == "bfloat16" else value)


def _finite_number(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _bfloat16_round(array: Any) -> Any:
    """Round float32 values to BF16 precision without requiring PyTorch."""

    if _np is None:
        return array
    values = _np.asarray(array, dtype=_np.float32)
    bits = values.view(_np.uint32)
    # Round-to-nearest-even before dropping the low 16 mantissa bits.
    rounding = _np.uint32(0x7FFF) + ((bits >> _np.uint32(16)) & _np.uint32(1))
    return ((bits + rounding) & _np.uint32(0xFFFF0000)).view(_np.float32)


def _cast_values(values: Any, dtype: Any, shape: tuple[int, ...] | None = None) -> Any:
    """Cast generated values and apply the portable BF16 emulation."""

    canonical = _canonical_dtype_name(dtype)
    if canonical not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported input dtype: {dtype}")
    if _np is None:
        return values
    host_dtype = _normalized_dtype(dtype)
    try:
        array = _np.asarray(values, dtype=host_dtype)
        if canonical == "bfloat16":
            array = _bfloat16_round(array)
        if shape is not None:
            array = array.reshape(shape)
        return array
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"values cannot be represented by input dtype {canonical}") from exc


def _supports_non_finite(dtype: Any) -> bool:
    canonical = _canonical_dtype_name(dtype)
    return (
        canonical.startswith("float") or canonical.startswith("complex") or canonical == "bfloat16"
    )


def _validate_range(low: Any, high: Any) -> tuple[float, float]:
    if not _finite_number(low) or not _finite_number(high):
        raise ValueError("input value range must contain finite numbers")
    normalized = (float(low), float(high))
    if normalized[0] > normalized[1]:
        raise ValueError("input value range must be ordered")
    return normalized


def _apply_constraints(value: Any, spec: InputSpec | None) -> Any:
    if spec is None:
        return value
    constraints = {item.casefold().replace("_", "-") for item in spec.semantic_constraints}
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
            return _cast_values(array, spec.dtype)
        except (TypeError, ValueError, OverflowError):
            if constraints:
                raise ValueError(f"value does not satisfy input dtype {spec.dtype}") from None
    return value


def _make_value(value: Any, shape: tuple[int, ...] | None, dtype: Any = float) -> Any:
    _normalized_dtype(dtype)
    if shape is None:
        return _cast_values(value, dtype) if _np is not None else value
    size = 1
    for dim in shape:
        size *= dim
    values = [value] * size
    if _np is not None:
        return _cast_values(values, dtype, shape)
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
    _normalized_dtype(dtype)
    distribution_name = _canonical_distribution_name(distribution)

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
        return _cast_values(draw(), dtype) if _np is not None else draw()
    size = 1
    for dim in shape:
        size *= dim
    values = [draw() for _ in range(size)]
    if _np is not None:
        return _cast_values(values, dtype, shape)
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
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if not isinstance(left, Mapping) or not isinstance(right, Mapping):
                return False
            return left.keys() == right.keys() and all(
                _same_value(left[key], right[key]) for key in left
            )
        if _np is not None and (isinstance(left, _np.ndarray) or isinstance(right, _np.ndarray)):
            return bool(_np.array_equal(left, right, equal_nan=True))
        if (
            isinstance(left, Sequence)
            and not isinstance(left, (str, bytes))
            or isinstance(right, Sequence)
            and not isinstance(right, (str, bytes))
        ):
            if not isinstance(left, Sequence) or not isinstance(right, Sequence):
                return False
            return len(left) == len(right) and all(_same_value(a, b) for a, b in zip(left, right))
        result = left == right
        if isinstance(result, bool):
            return result
        return bool(_np.all(result)) if _np is not None else False
    except (TypeError, ValueError):
        return False


def generate_representative_inputs(
    shape: int | Sequence[int] | None = None,
    dtype: Any = "float32",
    count: int = 8,
    seed: int = 0,
    low: float = -1.0,
    high: float = 1.0,
    distribution: str = "uniform",
) -> list[Any]:
    """Generate stable typical values for a model input shape."""

    if count < 0:
        raise ValueError("count must be non-negative")
    low, high = _validate_range(low, high)
    distribution = _canonical_distribution_name(distribution)
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
    dtype: Any = "float32",
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

    low, high = _validate_range(low, high)
    distribution = _canonical_distribution_name(distribution)
    normalized = _normal_shape(shape)
    values: list[Any] = [
        _make_value(1e-12, normalized, dtype),
        _make_value(-1e-12, normalized, dtype),
        _make_value(1e6, normalized, dtype),
        _make_value(-1e6, normalized, dtype),
    ]
    if normalized and normalized[0] == 0:
        values.insert(0, _make_value(0, normalized, dtype))
    if include_non_finite and _supports_non_finite(dtype):
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
    dtype: Any = "float32",
    seed: int = 0,
    include_non_finite: bool = True,
    distribution: str = "uniform",
) -> list[Any]:
    """Generate inputs likely to expose backend numerical or shape drift."""

    distribution = _canonical_distribution_name(distribution)
    normalized = _normal_shape(shape)
    edges = generate_edge_inputs(
        shape=normalized, dtype=dtype, include_non_finite=include_non_finite
    )
    if normalized is None:
        alternating: Any = _make_value(-1.0, None, dtype)
    else:
        size = 1
        for dim in normalized:
            size *= dim
        values = [1.0 if index % 2 == 0 else -1.0 for index in range(size)]
        if _np is not None:
            alternating = _cast_values(values, dtype, normalized)
        else:
            alternating = values
    rng = random.Random(seed)
    near_boundary = _random_value(rng, normalized, -1e-7, 1e-7, dtype, distribution)
    return _dedupe(edges + [alternating, near_boundary])


def generate_input_cases(
    *,
    examples: Iterable[Any] | None = None,
    shape: int | Sequence[int] | None = None,
    dtype: Any = "float32",
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
    if isinstance(examples, Mapping):
        examples = [examples]
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
        elif isinstance(input_spec, (str, Path)):
            parsed = InputSpec.from_json(input_spec)
            if isinstance(parsed, InputSpec):
                normalized_specs[parsed.name] = parsed
            else:
                normalized_specs.update(parsed)
        elif isinstance(input_spec, Mapping):
            if "inputs" in input_spec:
                parsed = InputSpec.from_json(input_spec)
                if isinstance(parsed, InputSpec):
                    normalized_specs[parsed.name] = parsed
                else:
                    normalized_specs.update(parsed)
            elif _is_spec_mapping(input_spec):
                spec = InputSpec.from_dict(input_spec)
                normalized_specs[spec.name] = spec
            else:
                normalized_specs.update(_named_specs(InputSpec, input_spec))
        else:
            raise TypeError("input_spec must be an InputSpec, mapping, JSON string, or path")

    def generated_for_spec(spec: InputSpec | None, category: str) -> list[Any]:
        local_shape = spec.resolve_shape(seed=seed) if spec is not None else shape
        local_dtype = spec.dtype if spec is not None else dtype
        if spec is not None and "value" in spec.metadata:
            value = _apply_constraints(spec.metadata["value"], spec)
            if spec.shape is not None and not isinstance(value, Mapping):
                value = _cast_values(value, spec.dtype, local_shape)
            return [value]
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
