"""Numerical output comparison and model parity execution."""

from __future__ import annotations

import inspect
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Number
from typing import Any

from .backends import as_backend, predict_batch
from .inputs import InputCase, InputSpec, generate_input_cases


@dataclass(frozen=True, init=False)
class Tolerance:
    """Absolute/relative tolerance policy for numeric outputs."""

    absolute: float
    relative: float
    nan_equal: bool
    inf_equal: bool
    precision: str | None

    # These are intentionally conservative defaults.  They describe the
    # precision loss normally introduced by each representation, rather than
    # attempting to make every output pass.
    PROFILES = {
        "fp32": (1e-6, 1e-5),
        "float32": (1e-6, 1e-5),
        "single": (1e-6, 1e-5),
        "fp16": (1e-3, 1e-2),
        "float16": (1e-3, 1e-2),
        "half": (1e-3, 1e-2),
        "bf16": (2e-2, 2e-2),
        "bfloat16": (2e-2, 2e-2),
        "bfloat": (2e-2, 2e-2),
    }

    def __init__(
        self,
        absolute: float | None = None,
        relative: float | None = None,
        *,
        atol: float | None = None,
        rtol: float | None = None,
        abs_tol: float | None = None,
        rel_tol: float | None = None,
        precision: str | None = None,
        profile: str | None = None,
        nan_equal: bool = False,
        inf_equal: bool = True,
    ) -> None:
        if precision is not None and profile is not None:
            if self._normalise_precision(precision) != self._normalise_precision(profile):
                raise ValueError("precision and profile must agree when both are supplied")
        precision = precision if precision is not None else profile
        if precision is not None:
            precision = self._normalise_precision(precision)
            profile_atol, profile_rtol = self.PROFILES[precision]
            if absolute is None and atol is None and abs_tol is None:
                absolute = profile_atol
            if relative is None and rtol is None and rel_tol is None:
                relative = profile_rtol
        if atol is not None:
            absolute = atol
        elif abs_tol is not None:
            absolute = abs_tol
        elif absolute is None:
            absolute = 1e-6
        if rtol is not None:
            relative = rtol
        elif rel_tol is not None:
            relative = rel_tol
        elif relative is None:
            relative = 1e-5
        if absolute is None or relative is None:
            raise ValueError("tolerance values cannot be None")
        if (
            isinstance(absolute, bool)
            or isinstance(relative, bool)
            or not math.isfinite(float(absolute))
            or not math.isfinite(float(relative))
            or absolute < 0
            or relative < 0
        ):
            raise ValueError("absolute and relative tolerances must be non-negative")
        object.__setattr__(self, "absolute", float(absolute))
        object.__setattr__(self, "relative", float(relative))
        object.__setattr__(self, "nan_equal", nan_equal)
        object.__setattr__(self, "inf_equal", inf_equal)
        object.__setattr__(self, "precision", precision)

    @classmethod
    def _normalise_precision(cls, precision: str) -> str:
        name = str(precision).casefold().replace("-", "").replace("_", "")
        aliases = {
            "float32": "fp32",
            "single": "fp32",
            "fp32": "fp32",
            "float16": "fp16",
            "half": "fp16",
            "fp16": "fp16",
            "bfloat16": "bf16",
            "bfloat": "bf16",
            "bf16": "bf16",
        }
        canonical = aliases.get(name, name)
        if canonical not in cls.PROFILES:
            raise ValueError("precision must be one of fp32, fp16, or bf16")
        return canonical

    @classmethod
    def from_profile(cls, profile: str, **kwargs: Any) -> Tolerance:
        return cls(profile=profile, **kwargs)

    @classmethod
    def for_precision(cls, precision: str, **kwargs: Any) -> Tolerance:
        return cls(precision=precision, **kwargs)

    @classmethod
    def fp32(cls, **kwargs: Any) -> Tolerance:
        return cls(profile="fp32", **kwargs)

    @classmethod
    def fp16(cls, **kwargs: Any) -> Tolerance:
        return cls(profile="fp16", **kwargs)

    @classmethod
    def bf16(cls, **kwargs: Any) -> Tolerance:
        return cls(profile="bf16", **kwargs)

    @property
    def atol(self) -> float:
        return self.absolute

    @property
    def rtol(self) -> float:
        return self.relative

    @property
    def profile(self) -> str | None:
        """Alias for the selected precision profile."""
        return self.precision


Tolerances = Tolerance


@dataclass
class OutputComparison:
    equal: bool
    max_absolute_diff: float = 0.0
    max_relative_diff: float = 0.0
    mismatch_count: int = 0
    mismatch_paths: tuple[str, ...] = ()
    reason: str | None = None
    expected_shape: tuple[int | None, ...] | None = None
    actual_shape: tuple[int | None, ...] | None = None
    expected_dtype: str | None = None
    actual_dtype: str | None = None
    tolerance_profile: str | None = None

    @property
    def passed(self) -> bool:
        return self.equal

    @property
    def max_abs_diff(self) -> float:
        return self.max_absolute_diff

    @property
    def max_rel_diff(self) -> float:
        return self.max_relative_diff

    def __bool__(self) -> bool:
        return self.equal

    def to_dict(self) -> dict[str, Any]:
        return {
            "equal": self.equal,
            "max_absolute_diff": self.max_absolute_diff,
            "max_relative_diff": self.max_relative_diff,
            "mismatch_count": self.mismatch_count,
            "mismatch_paths": list(self.mismatch_paths),
            "reason": self.reason,
            "expected_shape": list(self.expected_shape)
            if self.expected_shape is not None
            else None,
            "actual_shape": list(self.actual_shape) if self.actual_shape is not None else None,
            "expected_dtype": self.expected_dtype,
            "actual_dtype": self.actual_dtype,
            "tolerance_profile": self.tolerance_profile,
        }


def _plain(value: Any) -> Any:
    """Convert common tensor/array containers to Python containers."""

    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(item) for item in value)
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            value = detach().cpu()
        except (AttributeError, RuntimeError):
            value = detach()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _plain(tolist())
        except (TypeError, ValueError):
            pass
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


def _report_value(value: Any) -> Any:
    """Convert arrays, tensors, and localization records to report data."""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _report_value(value.to_dict())
    plain = _plain(value)
    if isinstance(plain, Mapping):
        return {str(key): _report_value(item) for key, item in plain.items()}
    if isinstance(plain, (list, tuple)):
        return [_report_value(item) for item in plain]
    return plain


def _precision_from_values(*values: Any) -> str | None:
    """Infer a supported floating-point profile from array/tensor dtypes."""
    dtypes = [_dtype_of(value).casefold() for value in values if _dtype_of(value)]
    if any("bfloat16" in dtype for dtype in dtypes):
        return "bfloat16"
    if any("float16" in dtype or dtype in {"half", "fp16"} for dtype in dtypes):
        return "float16"
    if any("float32" in dtype or dtype in {"single", "fp32"} for dtype in dtypes):
        return "float32"
    return None


def compare_outputs(
    expected: Any,
    actual: Any,
    tolerance: Tolerance | Mapping[str, Any] | str | None = None,
    *,
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    abs_tol: float | None = None,
    rel_tol: float | None = None,
    precision: str | None = None,
    profile: str | None = None,
    output_tolerances: Mapping[str, Any] | None = None,
    tolerance_profiles: Mapping[str, Any] | None = None,
    dtype_policy: str = "ignore",
    check_dtype: bool | None = None,
    precision_aware: bool = False,
    order: str = "strict",
    ignore_order: bool | None = None,
    path: str = "$",
) -> OutputComparison:
    """Compare scalar, vector, mapping, tuple, tensor, or array outputs.

    ``tolerance`` can be a global policy, a precision profile name, or a map
    from output names/paths to policies.  Dtypes are ignored by default so
    numerically equivalent float32 and float64 outputs remain compatible;
    ``dtype_policy='strict'`` makes dtype part of the output contract.
    """

    absolute_tolerance = (
        absolute_tolerance
        if absolute_tolerance is not None
        else (atol if atol is not None else abs_tol)
    )
    relative_tolerance = (
        relative_tolerance
        if relative_tolerance is not None
        else (rtol if rtol is not None else rel_tol)
    )
    if ignore_order is not None:
        order = "ignore" if ignore_order else "strict"
    order = str(order).casefold().replace("-", "_")
    if order in {"ordered", "sensitive", "strict"}:
        order = "strict"
    elif order in {"unordered", "insensitive", "ignore", "ignored"}:
        order = "ignore"
    else:
        raise ValueError("order must be 'strict' or 'ignore'")
    if check_dtype is not None:
        dtype_policy = "strict" if check_dtype else "ignore"
    dtype_policy = str(dtype_policy).casefold().replace("-", "_")
    if dtype_policy in {"none", "off", "ignore", "ignored"}:
        dtype_policy = "ignore"
    elif dtype_policy in {"exact", "strict", "required"}:
        dtype_policy = "strict"
    else:
        raise ValueError("dtype_policy must be 'ignore' or 'strict'")

    profile_value = precision if precision is not None else profile
    if profile_value is not None and str(profile_value).casefold() == "auto":
        profile_value = _precision_from_values(expected, actual)
    elif precision_aware and profile_value is None:
        profile_value = _precision_from_values(expected, actual)
    base_tolerance = _coerce_tolerance(tolerance, precision=profile_value)
    output_map = _normalise_tolerance_map(tolerance)
    if output_tolerances is not None:
        output_map = {
            **(output_map or {}),
            **{str(key): _coerce_tolerance(value) for key, value in output_tolerances.items()},
        }
    if tolerance_profiles:
        output_map = {
            **(output_map or {}),
            **{str(key): _coerce_tolerance(value) for key, value in tolerance_profiles.items()},
        }
    if absolute_tolerance is not None or relative_tolerance is not None:
        base_tolerance = Tolerance(
            absolute=(
                base_tolerance.absolute if absolute_tolerance is None else absolute_tolerance
            ),
            relative=(
                base_tolerance.relative if relative_tolerance is None else relative_tolerance
            ),
            nan_equal=base_tolerance.nan_equal,
            inf_equal=base_tolerance.inf_equal,
            precision=base_tolerance.precision,
        )
    resolver = _ToleranceResolver(base_tolerance, output_map)
    return _compare(expected, actual, resolver, path, dtype_policy=dtype_policy, order=order)


def _coerce_tolerance(value: Any = None, *, precision: str | None = None) -> Tolerance:
    """Convert the flexible public tolerance forms to a policy."""
    if isinstance(value, Tolerance):
        return value
    if isinstance(value, str):
        return Tolerance(profile=value)
    if isinstance(value, Mapping):
        fields = {
            "absolute",
            "relative",
            "atol",
            "rtol",
            "abs_tol",
            "rel_tol",
            "precision",
            "profile",
            "nan_equal",
            "inf_equal",
        }
        if fields.intersection(value):
            return Tolerance(**dict(value))
    return Tolerance(precision=precision)


def _normalise_tolerance_map(value: Any) -> dict[str, Tolerance] | None:
    if not isinstance(value, Mapping):
        return None
    fields = {
        "absolute",
        "relative",
        "atol",
        "rtol",
        "abs_tol",
        "rel_tol",
        "precision",
        "profile",
        "nan_equal",
        "inf_equal",
    }
    if fields.intersection(value):
        return None
    return {str(key): _coerce_tolerance(item) for key, item in value.items()}


class _ToleranceResolver:
    def __init__(self, default: Tolerance, policies: Mapping[str, Tolerance] | None) -> None:
        self.default = default
        self.policies = policies or {}

    def for_path(self, path: str) -> Tolerance:
        if not self.policies:
            return self.default
        candidates = [path]
        if path.startswith("$['"):
            closing = path.find("']")
            if closing > 3:
                root = path[3:closing]
                candidates.extend((root, "$." + root, path[: closing + 2]))
        if path.startswith("$[") and path.endswith("]"):
            candidates.append(path[2:-1].strip("'\""))
        if path.startswith("$."):
            candidates.append(path[2:])
        candidates.extend(("$", "*"))
        for candidate in candidates:
            if candidate in self.policies:
                return self.policies[candidate]
        prefixes = sorted(
            (key for key in self.policies if path.startswith(key + "[")),
            key=len,
            reverse=True,
        )
        return self.policies[prefixes[0]] if prefixes else self.default


def _example_tolerance(
    default: Any,
    policies: Mapping[Any, Any] | Callable[..., Any] | None,
    *,
    index: int,
    case: InputCase,
) -> Any:
    """Resolve a case policy by index, label, category, or callback."""
    if policies is None:
        return default
    if callable(policies):
        try:
            return policies(case=case, index=index)
        except TypeError:
            try:
                return policies(case, index)
            except TypeError:
                return policies(case)
    for key in (index, case.label, case.category, "*", str(index)):
        if key in policies:
            return policies[key]
    return default


def _failed(
    path: str,
    reason: str,
    *,
    absolute: float = 0.0,
    relative: float = 0.0,
) -> OutputComparison:
    return OutputComparison(False, absolute, relative, 1, (path,), reason)


def _combine(parts: list[OutputComparison]) -> OutputComparison:
    failures = [part for part in parts if not part.equal]
    return OutputComparison(
        not failures,
        max((part.max_absolute_diff for part in parts), default=0.0),
        max((part.max_relative_diff for part in parts), default=0.0),
        sum(part.mismatch_count for part in parts),
        tuple(path for part in failures for path in part.mismatch_paths),
        failures[0].reason if failures else None,
    )


def _shape_of(value: Any) -> tuple[int | None, ...] | None:
    """Return a portable shape for arrays and nested Python sequences."""
    shape = getattr(value, "shape", None)
    if shape is not None and not isinstance(value, (str, bytes)):
        try:
            return tuple(int(dimension) for dimension in shape)
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)):
        if not value:
            return (0,)
        child_shapes = [_shape_of(item) for item in value]
        first = child_shapes[0]
        if all(item == first for item in child_shapes):
            return (len(value),) + (first or ())
        return (len(value), None)
    return None


def _dtype_of(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype)


def _output_root(path: str) -> str:
    """Collapse an element mismatch path to its top-level output name."""
    if path == "$" or not path.startswith("$"):
        return path
    if path.startswith("$['"):
        closing = path.find("']")
        if closing > 3:
            return path[: closing + 2]
    if path.startswith("$."):
        return path[2:].split(".", 1)[0]
    if path.startswith("$["):
        closing = path.find("]")
        if closing > 2:
            return path[: closing + 1]
    return "$"


def _is_arrayish(value: Any) -> bool:
    return not isinstance(value, (str, bytes, list, tuple, Mapping)) and (
        getattr(value, "shape", None) is not None
        or callable(getattr(value, "tolist", None))
        or callable(getattr(value, "detach", None))
    )


def _compare(
    expected: Any,
    actual: Any,
    resolver: _ToleranceResolver,
    path: str,
    *,
    dtype_policy: str = "ignore",
    order: str = "strict",
) -> OutputComparison:
    expected_shape = _shape_of(expected)
    actual_shape = _shape_of(actual)
    expected_dtype = _dtype_of(expected)
    actual_dtype = _dtype_of(actual)
    if expected_shape is not None and actual_shape is not None and expected_shape != actual_shape:
        return OutputComparison(
            False,
            mismatch_count=1,
            mismatch_paths=(path,),
            reason=f"output shape differs: {expected_shape} != {actual_shape}",
            expected_shape=expected_shape,
            actual_shape=actual_shape,
            expected_dtype=expected_dtype,
            actual_dtype=actual_dtype,
        )
    if (
        dtype_policy == "strict"
        and expected_dtype is not None
        and actual_dtype is not None
        and expected_dtype != actual_dtype
    ):
        return OutputComparison(
            False,
            mismatch_count=1,
            mismatch_paths=(path,),
            reason=f"output dtype differs: {expected_dtype} != {actual_dtype}",
            expected_shape=expected_shape,
            actual_shape=actual_shape,
            expected_dtype=expected_dtype,
            actual_dtype=actual_dtype,
        )

    # Normalize only after inspecting metadata.  This keeps tensor/array
    # support optional and allows the scalar comparator below to stay tiny.
    if _is_arrayish(expected) or _is_arrayish(actual):
        expected = _plain(expected)
        actual = _plain(actual)

    if expected is None or actual is None:
        if expected is None and actual is None:
            return OutputComparison(True)
        return _failed(path, "null output differs")

    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            return _failed(path, "output container type differs")
        parts: list[OutputComparison] = []
        for key in sorted(expected.keys() - actual.keys(), key=repr):
            parts.append(_failed(f"{path}[{key!r}]", "missing output key"))
        for key in sorted(actual.keys() - expected.keys(), key=repr):
            parts.append(_failed(f"{path}[{key!r}]", "unexpected output key"))
        for key in sorted(expected.keys() & actual.keys(), key=repr):
            parts.append(
                _compare(
                    expected[key],
                    actual[key],
                    resolver,
                    f"{path}[{key!r}]",
                    dtype_policy=dtype_policy,
                    order=order,
                )
            )
        return _combine(parts)

    sequence_types = (list, tuple)
    if isinstance(expected, sequence_types) or isinstance(actual, sequence_types):
        if not isinstance(expected, sequence_types) or not isinstance(actual, sequence_types):
            return _failed(path, "output container type differs")
        if len(expected) != len(actual):
            return _failed(path, f"output length differs: {len(expected)} != {len(actual)}")
        if order == "ignore":
            unmatched = list(actual)
            parts = []
            for expected_index, expected_item in enumerate(expected):
                match_index = next(
                    (
                        index
                        for index, actual_item in enumerate(unmatched)
                        if _compare(
                            expected_item,
                            actual_item,
                            resolver,
                            f"{path}[{expected_index}]",
                            dtype_policy=dtype_policy,
                            order=order,
                        ).equal
                    ),
                    None,
                )
                if match_index is None:
                    parts.append(_failed(f"{path}[{expected_index}]", "unordered output differs"))
                else:
                    unmatched.pop(match_index)
            return _combine(parts)
        return _combine(
            [
                _compare(
                    a,
                    b,
                    resolver,
                    f"{path}[{i}]",
                    dtype_policy=dtype_policy,
                    order=order,
                )
                for i, (a, b) in enumerate(zip(expected, actual))
            ]
        )

    if isinstance(expected, bool) or isinstance(actual, bool):
        return (
            OutputComparison(True)
            if expected is actual
            else _failed(path, "boolean output differs")
        )

    if isinstance(expected, Number) and isinstance(actual, Number):
        try:
            tolerance = resolver.for_path(path)
            expected_number = (
                complex(expected) if isinstance(expected, complex) else float(expected)
            )
            actual_number = complex(actual) if isinstance(actual, complex) else float(actual)

            def is_nan(value: float | complex) -> bool:
                if isinstance(value, complex):
                    return math.isnan(value.real) or math.isnan(value.imag)
                return math.isnan(value)

            def is_infinite(value: float | complex) -> bool:
                if isinstance(value, complex):
                    return math.isinf(value.real) or math.isinf(value.imag)
                return math.isinf(value)

            if is_nan(expected_number) or is_nan(actual_number):
                if tolerance.nan_equal and is_nan(expected_number) and is_nan(actual_number):
                    return OutputComparison(True)
                return _failed(path, "NaN output differs")
            if is_infinite(expected_number) or is_infinite(actual_number):
                if tolerance.inf_equal and expected_number == actual_number:
                    return OutputComparison(True)
                return _failed(
                    path, "infinite output differs", absolute=math.inf, relative=math.inf
                )
            absolute = abs(expected - actual)
            absolute_float = float(absolute)
            relative = (
                absolute_float / abs(expected_number) if expected_number != 0 else absolute_float
            )
            allowed = tolerance.absolute + tolerance.relative * abs(expected_number)
            return (
                OutputComparison(True, absolute_float, relative)
                if absolute_float <= allowed
                else _failed(
                    path,
                    "numeric output exceeds tolerance",
                    absolute=absolute_float,
                    relative=relative,
                )
            )
        except (TypeError, ValueError, OverflowError):
            return _failed(path, "numeric outputs could not be compared")

    if expected == actual:
        return OutputComparison(True)
    return _failed(path, "output values differ")


@dataclass
class ParityCaseResult:
    input: Any
    label: str
    category: str
    reference_output: Any = None
    candidate_output: Any = None
    comparison: OutputComparison | None = None
    error: str | None = None
    localization: Any = None
    shrunk_input: Any = None
    shrink_error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.comparison is not None and self.comparison.equal

    def to_dict(self) -> dict[str, Any]:
        localization = self.localization
        if hasattr(localization, "to_dict") and callable(localization.to_dict):
            localization = localization.to_dict()
        return {
            "label": self.label,
            "category": self.category,
            "passed": self.passed,
            "input": _report_value(self.input),
            "reference_output": _report_value(self.reference_output),
            "candidate_output": _report_value(self.candidate_output),
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "error": self.error,
            "localization": _report_value(localization),
            "shrunk_input": _report_value(self.shrunk_input),
            "shrink_error": self.shrink_error,
        }


@dataclass
class ParityDivergence:
    """Backward-compatible divergence record for report consumers."""

    index: int
    max_absolute_error: float
    max_relative_error: float
    baseline: Any
    candidate: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    example_index: int | None = None
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "baseline": self.baseline,
            "candidate": self.candidate,
            "metadata": self.metadata,
            "example_index": self.example_index,
            "output_path": self.output_path,
        }


class _SummaryText(str):
    """String that also preserves the convenient ``report.summary()`` form."""

    def __call__(self) -> str:
        return str(self)


@dataclass
class ParityReport:
    reference_name: str
    candidate_name: str
    results: list[ParityCaseResult] = field(default_factory=list)
    tolerance_atol: float = 1e-6
    tolerance_rtol: float = 1e-5
    max_divergences: int = 100
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def baseline_name(self) -> str:
        return self.reference_name

    @property
    def sample_count(self) -> int:
        return len(self.results)

    @property
    def divergences(self) -> list[ParityDivergence]:
        records = []
        for index, result in enumerate(self.results):
            if result.passed:
                continue
            comparison = result.comparison
            output_path = (
                comparison.mismatch_paths[0]
                if comparison is not None and comparison.mismatch_paths
                else None
            )
            records.append(
                ParityDivergence(
                    index=index,
                    max_absolute_error=(
                        comparison.max_absolute_diff if comparison is not None else math.inf
                    ),
                    max_relative_error=(
                        comparison.max_relative_diff if comparison is not None else math.inf
                    ),
                    baseline=result.reference_output,
                    candidate=result.candidate_output,
                    metadata=(
                        {
                            **(
                                result.localization
                                if isinstance(result.localization, dict)
                                else {"localization": result.localization}
                            ),
                            "mismatch_paths": (
                                list(comparison.mismatch_paths) if comparison is not None else []
                            ),
                        }
                    ),
                    example_index=index,
                    output_path=output_path,
                )
            )
        return sorted(
            records,
            key=lambda record: (record.max_absolute_error, record.max_relative_error),
            reverse=True,
        )[: self.max_divergences]

    @property
    def per_example_errors(self) -> list[dict[str, Any]]:
        """Return one compact error record for every supplied example."""
        return [
            {
                "index": index,
                "label": result.label,
                "category": result.category,
                "passed": result.passed,
                "max_absolute_error": (
                    result.comparison.max_absolute_diff
                    if result.comparison is not None
                    else math.inf
                    if result.error is not None
                    else 0.0
                ),
                "max_relative_error": (
                    result.comparison.max_relative_diff
                    if result.comparison is not None
                    else math.inf
                    if result.error is not None
                    else 0.0
                ),
                "mismatch_count": (
                    result.comparison.mismatch_count if result.comparison is not None else 0
                ),
                "mismatch_paths": (
                    list(result.comparison.mismatch_paths) if result.comparison is not None else []
                ),
                "error": result.error,
            }
            for index, result in enumerate(self.results)
        ]

    @property
    def per_output_errors(self) -> dict[str, dict[str, Any]]:
        """Aggregate divergence counts and worst errors by output path."""
        output_errors: dict[str, dict[str, Any]] = {}
        for result in self.results:
            comparison = result.comparison
            if comparison is None:
                continue
            paths = comparison.mismatch_paths or ("$",)
            for path in paths:
                output_path = _output_root(path)
                record = output_errors.setdefault(
                    output_path,
                    {
                        "count": 0,
                        "max_absolute_error": 0.0,
                        "max_relative_error": 0.0,
                        "examples": [],
                    },
                )
                record["count"] += 1
                record["max_absolute_error"] = max(
                    record["max_absolute_error"], comparison.max_absolute_diff
                )
                record["max_relative_error"] = max(
                    record["max_relative_error"], comparison.max_relative_diff
                )
                record["examples"].append(result.label)
        return output_errors

    @property
    def output_errors(self) -> dict[str, dict[str, Any]]:
        """Backward-friendly short alias for :attr:`per_output_errors`."""
        return self.per_output_errors

    @property
    def example_errors(self) -> list[dict[str, Any]]:
        """Backward-friendly short alias for :attr:`per_example_errors`."""
        return self.per_example_errors

    @property
    def divergent_count(self) -> int:
        return sum(not result.passed for result in self.results)

    @property
    def max_absolute_error(self) -> float:
        return max(
            (
                result.comparison.max_absolute_diff
                for result in self.results
                if result.comparison is not None
            ),
            default=0.0,
        )

    @property
    def max_relative_error(self) -> float:
        return max(
            (
                result.comparison.max_relative_diff
                for result in self.results
                if result.comparison is not None
            ),
            default=0.0,
        )

    @property
    def layer_diagnostics(self) -> dict[str, Any]:
        return {
            str(key): value
            for result in self.results
            if isinstance(result.localization, Mapping)
            for key, value in result.localization.items()
        }

    @property
    def passed(self) -> bool:
        return (
            bool(self.results)
            and bool(self.metadata.get("reference_valid", True))
            and self.metadata.get("status", "pass") == "pass"
            and all(result.passed for result in self.results)
        )

    @property
    def status(self) -> str:
        if self.metadata.get("status"):
            return str(self.metadata["status"])
        return "pass" if self.passed else "fail"

    @property
    def exit_code(self) -> int:
        """CLI-compatible status code: pass, failed parity, or unusable reference."""
        return {"pass": 0, "fail": 1, "inconclusive": 2}.get(self.status, 2)

    @property
    def reference_valid(self) -> bool:
        return bool(self.metadata.get("reference_valid", True))

    @property
    def pass_count(self) -> int:
        return sum(result.passed for result in self.results)

    @property
    def fail_count(self) -> int:
        return sum(not result.passed for result in self.results)

    @property
    def error_count(self) -> int:
        return sum(result.error is not None for result in self.results)

    @property
    def mismatch_count(self) -> int:
        return sum(
            result.comparison.mismatch_count
            for result in self.results
            if result.comparison is not None
        )

    @property
    def summary(self) -> str:
        status = {"pass": "PASS", "inconclusive": "INCONCLUSIVE", "fail": "FAIL"}.get(
            self.status, "FAIL"
        )
        return _SummaryText(
            f"{status}: {self.pass_count}/{len(self.results)} cases matched "
            f"({self.error_count} errors)"
        )

    @property
    def failures(self) -> list[ParityCaseResult]:
        return [result for result in self.results if not result.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference_name,
            "candidate": self.candidate_name,
            "baseline_name": self.baseline_name,
            "candidate_name": self.candidate_name,
            "passed": self.passed,
            "status": self.status,
            "reference_valid": self.reference_valid,
            "reference_error_count": self.metadata.get("reference_error_count", 0),
            "total": len(self.results),
            "sample_count": self.sample_count,
            "divergent_count": self.divergent_count,
            "tolerance_atol": self.tolerance_atol,
            "tolerance_rtol": self.tolerance_rtol,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "layer_diagnostics": self.layer_diagnostics,
            "metadata": self.metadata,
            "divergences": [item.to_dict() for item in self.divergences],
            "passed_cases": self.pass_count,
            "failed_cases": self.fail_count,
            "errors": self.error_count,
            "mismatches": self.mismatch_count,
            "per_example_errors": self.per_example_errors,
            "per_output_errors": self.per_output_errors,
            "cases": [result.to_dict() for result in self.results],
        }

    def report(self, *, max_failures: int = 10) -> str:
        lines = [self.summary]
        # A sweep can diverge on every one of thousands of inputs. Listing them
        # all buries the summary, and the worst cases are the actionable ones.
        failures = [result for result in self.results if not result.passed]
        ranked = sorted(
            failures,
            key=lambda result: (
                result.comparison.max_absolute_diff
                if result.comparison is not None
                # A backend error has no numeric divergence but is the most
                # severe outcome, so it sorts ahead of any tolerance breach.
                else float("inf")
            ),
            reverse=True,
        )
        for result in ranked[:max_failures]:
            detail = result.error or (
                result.comparison.reason if result.comparison is not None else "unknown failure"
            )
            paths = (
                ", ".join(result.comparison.mismatch_paths[:3])
                if result.comparison is not None
                else ""
            )
            lines.append(f"- {result.label}: {detail}" + (f" at {paths}" if paths else ""))
            if result.shrunk_input is not None:
                lines.append(f"  minimized input: {result.shrunk_input!r}")
            elif result.shrink_error is not None:
                lines.append(f"  minimization skipped: {result.shrink_error}")
        if len(ranked) > max_failures:
            lines.append(
                f"- ... {len(ranked) - max_failures} more failing case(s); "
                f"use --json for the full list"
            )
        return "\n".join(lines)

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)

    __str__ = report


def _invoke_hook(
    hook: Callable[..., Any], *, result: ParityCaseResult, context: dict[str, Any]
) -> Any:
    """Call hooks with either the rich keyword form or a simple result form."""

    try:
        signature = inspect.signature(hook)
        names = signature.parameters
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in names.values()):
            return hook(result=result, **context)
        accepted = {
            key: value for key, value in {"result": result, **context}.items() if key in names
        }
        if accepted:
            return hook(**accepted)
    except (TypeError, ValueError):
        pass
    try:
        parameters = list(inspect.signature(hook).parameters.values())
    except (TypeError, ValueError):
        return hook(result)
    positional = [
        context.get("input"),
        context.get("reference_output"),
        context.get("candidate_output"),
        context.get("comparison"),
    ]
    positional_count = sum(
        parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for parameter in parameters
    )
    if positional_count == 1:
        return hook(result)
    return hook(*positional[:positional_count])


def _default_localization(comparison: OutputComparison) -> dict[str, Any]:
    """Return useful localization even when no framework hook is installed."""

    return {
        "path": comparison.mismatch_paths[0] if comparison.mismatch_paths else "$",
        "component": "output",
        "message": comparison.reason or "backend outputs diverged",
        "mismatch_paths": list(comparison.mismatch_paths),
    }


def _localize(
    hook: Callable[..., Any] | None,
    *,
    result: ParityCaseResult,
    context: dict[str, Any],
) -> Any:
    if hook is None:
        comparison = result.comparison
        return _default_localization(comparison) if comparison is not None else None
    try:
        return _invoke_hook(hook, result=result, context=context)
    except Exception as exc:
        return {
            "path": (
                result.comparison.mismatch_paths[0]
                if result.comparison and result.comparison.mismatch_paths
                else None
            ),
            "component": "localizer",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _infer_input_dtype(backend: Any) -> Any:
    """Infer a model input dtype without importing optional runtimes eagerly."""
    for attribute in ("input_dtype", "dtype"):
        value = getattr(backend, attribute, None)
        if value is not None:
            return value
    spec = getattr(backend, "input_spec", None)
    if isinstance(spec, InputSpec):
        return spec.dtype
    model = getattr(backend, "model", None)
    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            first = next(iter(parameters()))
            return getattr(first, "dtype", None) or "float32"
        except (StopIteration, TypeError, RuntimeError):
            pass
    session = getattr(backend, "session", None)
    get_inputs = getattr(session, "get_inputs", None)
    if callable(get_inputs):
        try:
            model_type = str(get_inputs()[0].type).casefold()
            if "double" in model_type or "float64" in model_type:
                return "float64"
            if "float16" in model_type:
                return "float16"
            if "int64" in model_type or "long" in model_type:
                return "int64"
            if "int32" in model_type:
                return "int32"
        except (IndexError, AttributeError, TypeError):
            pass
    return "float32"


def compare_models(
    reference: Any,
    candidate: Any,
    inputs: Iterable[Any] | None = None,
    *,
    tolerance: Tolerance | Mapping[str, Any] | str | None = None,
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    precision: str | None = None,
    profile: str | None = None,
    output_tolerances: Mapping[str, Any] | None = None,
    tolerance_profiles: Mapping[str, Any] | None = None,
    example_tolerances: Mapping[Any, Any] | Callable[..., Any] | None = None,
    per_example_tolerances: Mapping[Any, Any] | Callable[..., Any] | None = None,
    dtype_policy: str = "ignore",
    check_dtype: bool | None = None,
    precision_aware: bool = False,
    order: str = "strict",
    ignore_order: bool | None = None,
    max_divergences: int = 100,
    localizer: Callable[..., Any] | None = None,
    localization_hook: Callable[..., Any] | None = None,
    shrinker: Any = None,
    shrink_failures: bool = False,
    reference_name: str | None = None,
    candidate_name: str | None = None,
    input_shape: int | Sequence[int] | None = None,
    input_dtype: Any | None = None,
    input_spec: InputSpec | Mapping[str, Any] | str | None = None,
    seed: int = 0,
    representative_count: int = 8,
    include_edge: bool = True,
    include_adversarial: bool = True,
    include_non_finite: bool = False,
) -> ParityReport:
    """Run backend parity checks over supplied or generated inputs.

    When ``inputs`` is omitted, deterministic representative, edge, and
    adversarial cases are generated. Passing an explicit empty iterable keeps
    the report empty, which is distinct from requesting generated coverage.
    """

    ref = as_backend(reference, name=reference_name)
    cand = as_backend(candidate, name=candidate_name)
    if max_divergences < 0:
        raise ValueError("max_divergences must be non-negative")
    profile_value = precision if precision is not None else profile
    tol = _coerce_tolerance(tolerance, precision=profile_value)
    if absolute_tolerance is None:
        absolute_tolerance = atol
    if relative_tolerance is None:
        relative_tolerance = rtol
    if absolute_tolerance is not None or relative_tolerance is not None:
        tol = Tolerance(
            absolute=tol.absolute if absolute_tolerance is None else absolute_tolerance,
            relative=tol.relative if relative_tolerance is None else relative_tolerance,
            nan_equal=tol.nan_equal,
            inf_equal=tol.inf_equal,
            precision=tol.precision,
        )
    case_policies = per_example_tolerances or example_tolerances
    configured_output_tolerances = {
        **(_normalise_tolerance_map(tolerance) or {}),
        **(
            {str(key): _coerce_tolerance(value) for key, value in output_tolerances.items()}
            if output_tolerances is not None
            else {}
        ),
        **(
            {str(key): _coerce_tolerance(value) for key, value in tolerance_profiles.items()}
            if tolerance_profiles is not None
            else {}
        ),
    }
    hook = localizer or localization_hook
    generated = inputs is None
    parsed_input_spec: InputSpec | Mapping[str, InputSpec] | None = None
    if input_spec is not None:
        parsed_input_spec = (
            InputSpec.from_json(input_spec) if isinstance(input_spec, str) else input_spec
        )
    if isinstance(parsed_input_spec, InputSpec):
        if input_shape is None:
            input_shape = parsed_input_spec.resolve_shape(seed=seed)
        if input_dtype is None:
            input_dtype = parsed_input_spec.dtype
    if input_dtype is None:
        input_dtype = _infer_input_dtype(ref)
    elif isinstance(input_dtype, str):
        # Validate explicit CLI/API dtype values even when the caller supplies
        # inputs rather than asking the generator to materialize an array.
        InputSpec(dtype=input_dtype)
    cases = (
        generate_input_cases(
            shape=input_shape,
            dtype=input_dtype,
            seed=seed,
            representative_count=representative_count,
            include_edge=include_edge,
            include_adversarial=include_adversarial,
            include_non_finite=include_non_finite,
            input_spec=parsed_input_spec,
        )
        if generated
        else inputs
    )
    cases = list(cases)
    prepared_cases = [
        raw_case
        if isinstance(raw_case, InputCase)
        else InputCase(raw_case, f"input-{index}", "custom")
        for index, raw_case in enumerate(cases)
    ]
    reference_results: list[tuple[Any, BaseException | None]] = []
    for case in prepared_cases:
        try:
            reference_results.append((ref.predict(case.value), None))
        except Exception as exc:
            reference_results.append((None, exc))
    results: list[ParityCaseResult] = []
    for index, case in enumerate(prepared_cases):
        result = ParityCaseResult(case.value, case.label, case.category)
        selected_tolerance = _example_tolerance(
            tolerance,
            case_policies,
            index=index,
            case=case,
        )
        case_tolerance = _coerce_tolerance(selected_tolerance, precision=profile_value)
        case_output_tolerances = {
            **configured_output_tolerances,
            **(_normalise_tolerance_map(selected_tolerance) or {}),
        }
        error_stage: str | None = None
        error_type: type[BaseException] | None = None
        result.reference_output, reference_error = reference_results[index]
        if reference_error is not None:
            exc = reference_error
            error_stage = "reference"
            error_type = type(exc)
            result.error = f"reference {type(exc).__name__}: {exc}"
        try:
            if result.error is None:
                result.candidate_output = cand.predict(case.value)
        except Exception as exc:
            error_stage = "candidate"
            error_type = type(exc)
            result.error = f"candidate {type(exc).__name__}: {exc}"
        try:
            if result.error is None:
                result.comparison = compare_outputs(
                    result.reference_output,
                    result.candidate_output,
                    case_tolerance,
                    output_tolerances=case_output_tolerances,
                    tolerance_profiles=tolerance_profiles,
                    dtype_policy=dtype_policy,
                    check_dtype=check_dtype,
                    precision_aware=precision_aware,
                    order=order,
                    ignore_order=ignore_order,
                )
        except Exception as exc:
            error_stage = "comparison"
            error_type = type(exc)
            result.error = f"comparison {type(exc).__name__}: {exc}"

        if shrink_failures and (result.error is not None or not result.passed):
            from .shrinking import shrink_input

            if result.error is not None:

                def still_fails(value: Any) -> bool:
                    try:
                        expected = ref.predict(value)
                    except Exception as exc:
                        return error_stage == "reference" and type(exc) is error_type
                    try:
                        actual = cand.predict(value)
                    except Exception as exc:
                        return error_stage == "candidate" and type(exc) is error_type
                    try:
                        compare_outputs(
                            expected,
                            actual,
                            case_tolerance,
                            output_tolerances=case_output_tolerances,
                            tolerance_profiles=tolerance_profiles,
                            dtype_policy=dtype_policy,
                            check_dtype=check_dtype,
                            precision_aware=precision_aware,
                            order=order,
                            ignore_order=ignore_order,
                        )
                    except Exception as exc:
                        return error_stage == "comparison" and type(exc) is error_type
                    return False

            else:

                def still_fails(value: Any) -> bool:
                    try:
                        expected = ref.predict(value)
                        actual = cand.predict(value)
                    except Exception:
                        return False
                    return not compare_outputs(
                        expected,
                        actual,
                        case_tolerance,
                        output_tolerances=case_output_tolerances,
                        tolerance_profiles=tolerance_profiles,
                        dtype_policy=dtype_policy,
                        check_dtype=check_dtype,
                        precision_aware=precision_aware,
                        order=order,
                        ignore_order=ignore_order,
                    ).equal

            try:
                result.shrunk_input = shrink_input(case.value, still_fails, shrinker=shrinker)
            except Exception as exc:
                result.shrink_error = f"{type(exc).__name__}: {exc}"

        if result.error is None and result.comparison is not None and not result.comparison.equal:
            result.localization = _localize(
                hook,
                result=result,
                context={
                    "input": case.value,
                    "reference_output": result.reference_output,
                    "candidate_output": result.candidate_output,
                    "comparison": result.comparison,
                },
            )
        elif result.error is not None:
            result.localization = {
                "component": error_stage or "backend",
                "message": result.error,
            }
        results.append(result)
    reference_valid = all(error is None for _output, error in reference_results)
    if not prepared_cases:
        status = "inconclusive"
    elif not reference_valid:
        status = "inconclusive"
    elif any(result.error is not None for result in results):
        status = "fail"
    else:
        status = "pass" if all(result.passed for result in results) else "fail"
    return ParityReport(
        getattr(ref, "name", "reference"),
        getattr(cand, "name", "candidate"),
        results,
        tol.absolute,
        tol.relative,
        max_divergences,
        {
            "input_source": "generated" if generated else "provided",
            "generated_seed": seed if generated else None,
            "generated_shape": input_shape if generated else None,
            "generated_dtype": str(input_dtype) if generated else None,
            "input_spec": (
                parsed_input_spec.to_dict()
                if isinstance(parsed_input_spec, InputSpec)
                else {
                    str(name): spec.to_dict() if isinstance(spec, InputSpec) else str(spec)
                    for name, spec in parsed_input_spec.items()
                }
                if isinstance(parsed_input_spec, Mapping)
                else None
            ),
            "reference_valid": reference_valid,
            "reference_error_count": sum(error is not None for _output, error in reference_results),
            "tolerance_profile": tol.precision,
            "output_tolerances": {
                key: {"atol": value.absolute, "rtol": value.relative}
                for key, value in configured_output_tolerances.items()
            },
            "status": status,
        },
    )


def compare_batch(
    reference: Any,
    candidate: Any,
    batches: Iterable[Sequence[Any]],
    *,
    tolerance: Tolerance | Mapping[str, Any] | str | None = None,
    **kwargs: Any,
) -> ParityReport:
    """Compare backend batch predictions, one report case per batch item."""

    reference_name = kwargs.pop("reference_name", None)
    candidate_name = kwargs.pop("candidate_name", None)
    ref = as_backend(reference, name=reference_name)
    cand = as_backend(candidate, name=candidate_name)
    precision = kwargs.pop("precision", kwargs.pop("profile", None))
    tol = _coerce_tolerance(tolerance, precision=precision)
    absolute_tolerance = kwargs.pop(
        "absolute_tolerance", kwargs.pop("atol", kwargs.pop("abs_tol", None))
    )
    relative_tolerance = kwargs.pop(
        "relative_tolerance", kwargs.pop("rtol", kwargs.pop("rel_tol", None))
    )
    if absolute_tolerance is not None or relative_tolerance is not None:
        tol = Tolerance(
            absolute=tol.absolute if absolute_tolerance is None else absolute_tolerance,
            relative=tol.relative if relative_tolerance is None else relative_tolerance,
            nan_equal=tol.nan_equal,
            inf_equal=tol.inf_equal,
            precision=tol.precision,
        )
    output_tolerances = kwargs.pop("output_tolerances", None)
    tolerance_profiles = kwargs.pop("tolerance_profiles", None)
    example_tolerances = kwargs.pop(
        "per_example_tolerances", kwargs.pop("example_tolerances", None)
    )
    dtype_policy = kwargs.pop("dtype_policy", "ignore")
    check_dtype = kwargs.pop("check_dtype", None)
    precision_aware = kwargs.pop("precision_aware", False)
    order = kwargs.pop("order", "strict")
    ignore_order = kwargs.pop("ignore_order", None)
    configured_output_tolerances = {
        **(_normalise_tolerance_map(tolerance) or {}),
        **(
            {str(key): _coerce_tolerance(value) for key, value in output_tolerances.items()}
            if output_tolerances is not None
            else {}
        ),
        **(
            {str(key): _coerce_tolerance(value) for key, value in tolerance_profiles.items()}
            if tolerance_profiles is not None
            else {}
        ),
    }
    localizer = kwargs.pop("localizer", kwargs.pop("localization_hook", None))
    shrinker = kwargs.pop("shrinker", None)
    shrink_failures = kwargs.pop("shrink_failures", False)
    max_divergences = kwargs.pop("max_divergences", 100)
    if max_divergences < 0:
        raise ValueError("max_divergences must be non-negative")
    if kwargs:
        raise TypeError(f"unexpected comparison options: {', '.join(sorted(kwargs))}")
    results: list[ParityCaseResult] = []
    reference_valid = True
    for batch_index, batch in enumerate(batches):
        try:
            ref_outputs = predict_batch(ref, batch)
        except Exception as exc:
            reference_valid = False
            results.append(
                ParityCaseResult(
                    batch,
                    f"batch-{batch_index}",
                    "batch",
                    error=f"reference {type(exc).__name__}: {exc}",
                )
            )
            continue
        if len(ref_outputs) != len(batch):
            case = ParityCaseResult(
                batch,
                f"batch-{batch_index}",
                "batch",
                error=(
                    "reference batch output length differs from input length: "
                    f"inputs={len(batch)}, reference={len(ref_outputs)}"
                ),
            )
            reference_valid = False
            results.append(case)
            continue
        try:
            cand_outputs = predict_batch(cand, batch)
        except Exception as exc:
            results.append(
                ParityCaseResult(
                    batch,
                    f"batch-{batch_index}",
                    "batch",
                    error=f"candidate {type(exc).__name__}: {exc}",
                )
            )
            continue
        if len(cand_outputs) != len(batch):
            case = ParityCaseResult(
                batch,
                f"batch-{batch_index}",
                "batch",
                error=(
                    "candidate batch output length differs from input length: "
                    f"inputs={len(batch)}, candidate={len(cand_outputs)}"
                ),
            )
            results.append(case)
            continue
        for item_index, (item, expected, actual) in enumerate(
            zip(batch, ref_outputs, cand_outputs)
        ):
            case = InputCase(item, f"batch-{batch_index}[{item_index}]", "batch")
            selected_tolerance = _example_tolerance(
                tolerance,
                example_tolerances,
                index=len(results),
                case=case,
            )
            item_tolerance = _coerce_tolerance(selected_tolerance, precision=precision)
            item_output_tolerances = {
                **configured_output_tolerances,
                **(_normalise_tolerance_map(selected_tolerance) or {}),
            }
            comparison = compare_outputs(
                expected,
                actual,
                item_tolerance,
                output_tolerances=item_output_tolerances,
                tolerance_profiles=tolerance_profiles,
                dtype_policy=dtype_policy,
                check_dtype=check_dtype,
                precision_aware=precision_aware,
                order=order,
                ignore_order=ignore_order,
            )
            result = ParityCaseResult(
                item,
                f"batch-{batch_index}[{item_index}]",
                "batch",
                expected,
                actual,
                comparison,
            )
            if shrink_failures and not comparison.equal:
                from .shrinking import shrink_input

                def still_fails(value: Any) -> bool:
                    try:
                        return not compare_outputs(
                            ref.predict(value),
                            cand.predict(value),
                            item_tolerance,
                            output_tolerances=item_output_tolerances,
                            tolerance_profiles=tolerance_profiles,
                            dtype_policy=dtype_policy,
                            check_dtype=check_dtype,
                            precision_aware=precision_aware,
                            order=order,
                            ignore_order=ignore_order,
                        ).equal
                    except Exception:
                        return False

                try:
                    result.shrunk_input = shrink_input(item, still_fails, shrinker=shrinker)
                except Exception as exc:
                    result.shrink_error = f"{type(exc).__name__}: {exc}"
            if not comparison.equal:
                result.localization = _localize(
                    localizer,
                    result=result,
                    context={
                        "input": item,
                        "reference_output": expected,
                        "candidate_output": actual,
                        "comparison": comparison,
                    },
                )
            results.append(result)
    report = ParityReport(
        getattr(ref, "name", "reference"),
        getattr(cand, "name", "candidate"),
        results,
        tol.absolute,
        tol.relative,
        max_divergences,
        {
            "input_source": "batches",
            "reference_valid": reference_valid,
            "reference_error_count": sum(
                1
                for result in results
                if result.error is not None and result.error.startswith("reference ")
            ),
            "tolerance_profile": tol.precision,
            "output_tolerances": {
                key: {"atol": value.absolute, "rtol": value.relative}
                for key, value in configured_output_tolerances.items()
            },
            "status": (
                "inconclusive"
                if not results or not reference_valid
                else "fail"
                if any(result.error is not None for result in results)
                or any(not result.passed for result in results)
                else "pass"
            ),
        },
    )
    return report


class ParityComparator:
    """Reusable comparator configured with models and a tolerance policy."""

    def __init__(
        self,
        reference: Any,
        candidate: Any,
        *,
        tolerance: Tolerance | None = None,
        **kwargs: Any,
    ) -> None:
        self.reference = reference
        self.candidate = candidate
        self.tolerance = tolerance or Tolerance()
        self.options = kwargs

    def compare(self, inputs: Iterable[Any]) -> ParityReport:
        return compare_models(
            self.reference, self.candidate, inputs, tolerance=self.tolerance, **self.options
        )

    def compare_batch(self, batches: Iterable[Sequence[Any]]) -> ParityReport:
        return compare_batch(
            self.reference, self.candidate, batches, tolerance=self.tolerance, **self.options
        )


ParityChecker = ParityComparator
ParityResult = ParityCaseResult
ComparisonResult = OutputComparison
TolerancePolicy = Tolerance
ToleranceProfile = Tolerance
parity = compare_models
