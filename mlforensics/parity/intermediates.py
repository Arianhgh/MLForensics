"""Bounded intermediate-output instrumentation for parity localization.

The parity package cannot infer the internal operations of an arbitrary Python
callable.  Callables therefore opt in by accepting an ``intermediate_hook``
(``hook`` and ``trace_hook`` are also recognized) or by exposing a
``register_intermediate_hook`` method.  PyTorch modules have a concrete,
optional hook protocol and are supported through forward hooks.

All captured values are detached/snapshotted and bounded before they are kept;
the recorder is intended for diagnostics, not for retaining a model's entire
activation graph.
"""

from __future__ import annotations

import inspect
import math
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .compare import OutputComparison, Tolerance, compare_outputs
from .localization import DivergenceLocation


class UnsupportedIntermediateLocalization(RuntimeError):
    """Raised when a backend has no supported intermediate-hook protocol."""


def _safe_scalar(value: Any) -> Any:
    """Convert scalar-like values without requiring NumPy or Torch."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _safe_scalar(item())
        except Exception:
            pass
    return repr(value)


def _snapshot(value: Any, *, max_elements: int, depth: int = 0) -> tuple[Any, bool]:
    """Make a small, JSON-friendly snapshot and report whether it was clipped."""

    if depth > 8:
        return ("<maximum nesting depth>", True)
    if value is None or isinstance(value, (str, bool, int, float, complex)):
        return (_safe_scalar(value), False)

    # Torch tensors and NumPy arrays both support detach/cpu/tolist, but the
    # operations are deliberately duck-typed so importing this module remains
    # framework-free.
    original = value
    detach = getattr(value, "detach", None)
    if callable(detach):
        try:
            value = detach()
        except Exception:
            value = original
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        try:
            value = cpu()
        except Exception:
            pass
    shape = getattr(value, "shape", None)
    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            count = int(numel())
        except (TypeError, ValueError, RuntimeError):
            count = None
    else:
        try:
            count = math.prod(int(item) for item in shape) if shape is not None else None
        except (TypeError, ValueError):
            count = None
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            if count is not None and count > max_elements:
                flatten = getattr(value, "reshape", None)
                if callable(flatten):
                    value = flatten(-1)[:max_elements]
                else:
                    value = list(tolist())[:max_elements]
                return (_snapshot(value, max_elements=max_elements, depth=depth + 1)[0], True)
            return (_snapshot(tolist(), max_elements=max_elements, depth=depth + 1)[0], False)
        except Exception:
            pass

    if isinstance(value, Mapping):
        result: dict[Any, Any] = {}
        clipped = False
        for index, (key, item) in enumerate(value.items()):
            if index >= max_elements:
                clipped = True
                break
            result[key], item_clipped = _snapshot(item, max_elements=max_elements, depth=depth + 1)
            clipped |= item_clipped
        return (result, clipped)
    if isinstance(value, (list, tuple)):
        result = []
        clipped = len(value) > max_elements
        for item in value[:max_elements]:
            snapshot, item_clipped = _snapshot(item, max_elements=max_elements, depth=depth + 1)
            result.append(snapshot)
            clipped |= item_clipped
        return (type(value)(result), clipped)
    return (_safe_scalar(value), False)


def _shape(value: Any) -> tuple[int, ...] | None:
    raw = getattr(value, "shape", None)
    if raw is not None:
        try:
            return tuple(int(item) for item in raw)
        except (TypeError, ValueError):
            pass
    if isinstance(value, (list, tuple)):
        result: list[int] = []
        current: Any = value
        while isinstance(current, (list, tuple)):
            result.append(len(current))
            if not current:
                break
            current = current[0]
        return tuple(result)
    return None


def _finite_stats(value: Any) -> dict[str, float | int | None]:
    """Return lightweight numeric stats when the value exposes a flat form."""

    values: list[float] = []

    def visit(item: Any) -> None:
        if len(values) >= 4096:
            return
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            try:
                number = float(item)
            except (TypeError, ValueError):
                return
            if math.isfinite(number):
                values.append(number)

    visit(value)
    if not values:
        return {"finite_count": 0, "min": None, "max": None, "mean": None}
    return {
        "finite_count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


@dataclass(frozen=True)
class IntermediateRecord:
    """One bounded intermediate value captured from a backend execution."""

    name: str
    value: Any
    sequence: int = 0
    phase: str = "forward"
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    device: str | None = None
    truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "sequence": self.sequence,
            "phase": self.phase,
            "shape": list(self.shape) if self.shape is not None else None,
            "dtype": self.dtype,
            "device": self.device,
            "truncated": self.truncated,
            "metadata": dict(self.metadata),
        }


class IntermediateRecorder:
    """A bounded ring buffer for intermediate records.

    ``max_records`` bounds retained activations and ``max_elements`` bounds
    each retained value.  Once full, the oldest record is evicted and
    ``dropped_count`` increases.  Call :meth:`freeze` from a callback when a
    trigger has fired to preserve the evidence window.
    """

    def __init__(self, *, max_records: int = 256, max_elements: int = 4096) -> None:
        if isinstance(max_records, bool) or not isinstance(max_records, int) or max_records < 0:
            raise ValueError("max_records must be a non-negative integer")
        if isinstance(max_elements, bool) or not isinstance(max_elements, int) or max_elements <= 0:
            raise ValueError("max_elements must be a positive integer")
        self.max_records = int(max_records)
        self.max_elements = int(max_elements)
        self._records: deque[IntermediateRecord] = deque(maxlen=self.max_records or None)
        self._sequence = 0
        self.dropped_count = 0
        self._frozen = False

    @property
    def records(self) -> tuple[IntermediateRecord, ...]:
        return tuple(self._records)

    @property
    def events(self) -> tuple[IntermediateRecord, ...]:
        """Alias useful to callers that store other diagnostic events."""

        return self.records

    @property
    def frozen(self) -> bool:
        return self._frozen

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[IntermediateRecord]:
        return iter(self._records)

    def clear(self) -> None:
        self._records.clear()
        self.dropped_count = 0
        self._sequence = 0
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True

    def thaw(self) -> None:
        self._frozen = False

    def record(
        self, name: str, value: Any, *, phase: str = "forward", **metadata: Any
    ) -> IntermediateRecord | None:
        if self._frozen or self.max_records == 0:
            self.dropped_count += 1
            return None
        if len(self._records) == self._records.maxlen:
            self.dropped_count += 1
        snapshot, truncated = _snapshot(value, max_elements=self.max_elements)
        record_metadata = dict(metadata)
        record_metadata.setdefault("stats", _finite_stats(snapshot))
        record = IntermediateRecord(
            name=str(name),
            value=snapshot,
            sequence=self._sequence,
            phase=str(phase),
            shape=_shape(value),
            dtype=str(getattr(value, "dtype", "")) or None,
            device=str(getattr(value, "device", "")) or None,
            truncated=truncated,
            metadata=record_metadata,
        )
        self._sequence += 1
        self._records.append(record)
        return record

    __call__ = record

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": [record.to_dict() for record in self._records],
            "max_records": self.max_records,
            "max_elements": self.max_elements,
            "dropped_count": self.dropped_count,
            "frozen": self.frozen,
        }


BoundedIntermediateRecorder = IntermediateRecorder


class CallableIntermediateHook:
    """Callback passed to an instrumented callable.

    A model should call ``hook("layer-name", intermediate_value)``.  The hook
    is also a context manager, so the same object can be reused for one
    execution and inspected through ``hook.records`` afterwards.
    """

    def __init__(
        self,
        recorder: IntermediateRecorder | None = None,
        *,
        callback: Callable[[IntermediateRecord], Any] | None = None,
        max_records: int = 256,
        max_elements: int = 4096,
    ) -> None:
        self.recorder = (
            recorder
            if recorder is not None
            else IntermediateRecorder(max_records=max_records, max_elements=max_elements)
        )
        self.callback = callback

    @property
    def records(self) -> tuple[IntermediateRecord, ...]:
        return self.recorder.records

    def __call__(self, name: str, value: Any, **metadata: Any) -> IntermediateRecord | None:
        record = self.recorder.record(name, value, **metadata)
        if record is not None and self.callback is not None:
            self.callback(record)
        return record

    record = __call__

    def __enter__(self) -> CallableIntermediateHook:
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None


def _hook_parameter(model: Any, requested: str | None) -> str | None:
    if requested:
        return requested
    try:
        signature = inspect.signature(model)
    except (TypeError, ValueError):
        return None
    for candidate in ("intermediate_hook", "hook", "trace_hook"):
        parameter = signature.parameters.get(candidate)
        if parameter is not None and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY:
            return candidate
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return "intermediate_hook"
    return None


class InstrumentedCallable:
    """Callable wrapper for the explicit callable intermediate-hook protocol."""

    def __init__(
        self,
        model: Callable[..., Any],
        *,
        recorder: IntermediateRecorder | None = None,
        hook_parameter: str | None = None,
        callback: Callable[[IntermediateRecord], Any] | None = None,
        max_records: int = 256,
        max_elements: int = 4096,
    ) -> None:
        if not callable(model):
            raise TypeError("model must be callable")
        self.model = model
        self.recorder = (
            recorder
            if recorder is not None
            else IntermediateRecorder(max_records=max_records, max_elements=max_elements)
        )
        self.hook = CallableIntermediateHook(self.recorder, callback=callback)
        self.hook_parameter = _hook_parameter(model, hook_parameter)
        self._register = getattr(model, "register_intermediate_hook", None)
        if self.hook_parameter is None and not callable(self._register):
            raise UnsupportedIntermediateLocalization(
                "callable intermediate localization requires an intermediate_hook, hook, "
                "or trace_hook parameter, or register_intermediate_hook()"
            )

    @property
    def records(self) -> tuple[IntermediateRecord, ...]:
        return self.recorder.records

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.hook_parameter is not None:
            if self.hook_parameter in kwargs:
                raise TypeError(f"{self.hook_parameter} is reserved for the instrumentation hook")
            kwargs[self.hook_parameter] = self.hook
            return self.model(*args, **kwargs)
        registration = self._register(self.hook)
        try:
            return self.model(*args, **kwargs)
        finally:
            remove = getattr(registration, "remove", None)
            if callable(remove):
                remove()
            elif callable(getattr(self.model, "unregister_intermediate_hook", None)):
                self.model.unregister_intermediate_hook(self.hook)


def instrument_callable(
    model: Callable[..., Any],
    *,
    recorder: IntermediateRecorder | None = None,
    hook_parameter: str | None = None,
    callback: Callable[[IntermediateRecord], Any] | None = None,
    max_records: int = 256,
    max_elements: int = 4096,
) -> InstrumentedCallable:
    """Wrap an opt-in callable with a bounded intermediate recorder."""

    return InstrumentedCallable(
        model,
        recorder=recorder,
        hook_parameter=hook_parameter,
        callback=callback,
        max_records=max_records,
        max_elements=max_elements,
    )


attach_callable_hook = instrument_callable


@dataclass(frozen=True)
class CallableIntermediateCapture:
    """Output and records produced by :func:`capture_callable_intermediates`."""

    output: Any
    recorder: IntermediateRecorder

    @property
    def records(self) -> tuple[IntermediateRecord, ...]:
        return self.recorder.records


def capture_callable_intermediates(
    model: Callable[..., Any],
    *args: Any,
    recorder: IntermediateRecorder | None = None,
    hook_parameter: str | None = None,
    max_records: int = 256,
    max_elements: int = 4096,
    callback: Callable[[IntermediateRecord], Any] | None = None,
    **kwargs: Any,
) -> CallableIntermediateCapture:
    """Execute an instrumentable callable and return its bounded records."""

    instrumented = instrument_callable(
        model,
        recorder=recorder,
        hook_parameter=hook_parameter,
        callback=callback,
        max_records=max_records,
        max_elements=max_elements,
    )
    return CallableIntermediateCapture(instrumented(*args, **kwargs), instrumented.recorder)


class TorchHookHandles(list[Any]):
    """List-like collection of removable Torch hook handles."""

    def __init__(
        self,
        handles: Iterable[Any] = (),
        *,
        recorder: IntermediateRecorder | None = None,
    ) -> None:
        super().__init__(handles)
        self.recorder = recorder

    @property
    def records(self) -> tuple[IntermediateRecord, ...]:
        return self.recorder.records if self.recorder is not None else ()

    def close(self) -> None:
        for handle in self:
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()
        self.clear()

    remove_all = close

    def __enter__(self) -> TorchHookHandles:
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def attach_torch_module_hooks(
    module: Any,
    recorder: IntermediateRecorder | None = None,
    *,
    modules: Iterable[str] | str | None = None,
    include_root: bool = False,
    max_records: int = 256,
    max_elements: int = 4096,
) -> TorchHookHandles:
    """Attach bounded forward hooks to a PyTorch module tree.

    PyTorch is imported lazily.  A clear :class:`UnsupportedIntermediateLocalization`
    is raised when PyTorch is unavailable or the object does not expose the
    module-hook protocol.  ``modules`` filters by fully-qualified module name.
    """

    try:
        import torch  # type: ignore  # noqa: F401
    except ImportError as exc:
        raise UnsupportedIntermediateLocalization(
            "Torch intermediate localization requires the optional 'torch' dependency"
        ) from exc
    named_modules = getattr(module, "named_modules", None)
    if not callable(named_modules):
        raise UnsupportedIntermediateLocalization(
            "Torch intermediate localization requires a module exposing named_modules()"
        )
    recorder = (
        recorder
        if recorder is not None
        else IntermediateRecorder(max_records=max_records, max_elements=max_elements)
    )
    allowed = (
        {modules}
        if isinstance(modules, str)
        else {str(name) for name in modules}
        if modules is not None
        else None
    )
    handles = TorchHookHandles(recorder=recorder)
    for name, child in named_modules():
        if not name and not include_root:
            continue
        if allowed is not None and name not in allowed and type(child).__name__ not in allowed:
            continue
        register = getattr(child, "register_forward_hook", None)
        if not callable(register):
            continue

        def forward_hook(
            _child: Any,
            _inputs: Any,
            output: Any,
            operation: str = str(name),
        ) -> None:
            recorder.record(operation, output, module=operation, backend="torch")

        handles.append(register(forward_hook))
    if not handles:
        raise UnsupportedIntermediateLocalization(
            "no selected Torch modules support register_forward_hook()"
        )
    return handles


attach_torch_hooks = attach_torch_module_hooks


def _record_parts(
    records: Iterable[IntermediateRecord | Mapping[str, Any]],
) -> list[IntermediateRecord]:
    result: list[IntermediateRecord] = []
    for index, record in enumerate(records):
        if isinstance(record, IntermediateRecord):
            result.append(record)
        elif isinstance(record, Mapping) and "name" in record:
            result.append(
                IntermediateRecord(
                    name=str(record["name"]),
                    value=record.get("value"),
                    sequence=int(record.get("sequence", index)),
                    phase=str(record.get("phase", "forward")),
                    shape=tuple(record["shape"]) if record.get("shape") is not None else None,
                    dtype=record.get("dtype"),
                    device=record.get("device"),
                    truncated=bool(record.get("truncated", False)),
                    metadata=dict(record.get("metadata", {})),
                )
            )
    return result


def first_intermediate_divergence(
    reference: Iterable[IntermediateRecord | Mapping[str, Any]],
    candidate: Iterable[IntermediateRecord | Mapping[str, Any]],
    *,
    tolerance: Tolerance | None = None,
) -> DivergenceLocation | None:
    """Return the first aligned intermediate that differs beyond tolerance."""

    left, right = _record_parts(reference), _record_parts(candidate)
    tol = tolerance or Tolerance()
    for index in range(max(len(left), len(right))):
        expected = left[index] if index < len(left) else None
        actual = right[index] if index < len(right) else None
        if expected is None or actual is None or expected.name != actual.name:
            name = (actual or expected).name if (actual or expected) else None
            return DivergenceLocation(
                path=name,
                component="intermediate",
                message="intermediate record sequence differs",
                details={
                    "index": index,
                    "reference_name": expected.name if expected else None,
                    "candidate_name": actual.name if actual else None,
                },
            )
        comparison: OutputComparison = compare_outputs(expected.value, actual.value, tol)
        if not comparison.equal:
            return DivergenceLocation(
                path=expected.name,
                component="intermediate",
                message="intermediate output diverged",
                details={
                    "index": index,
                    "phase": expected.phase,
                    "mismatch_paths": list(comparison.mismatch_paths),
                    "max_absolute_diff": comparison.max_absolute_diff,
                    "max_relative_diff": comparison.max_relative_diff,
                },
            )
    return None


localize_intermediate_divergence = first_intermediate_divergence


__all__ = [
    "BoundedIntermediateRecorder",
    "CallableIntermediateCapture",
    "CallableIntermediateHook",
    "InstrumentedCallable",
    "IntermediateRecord",
    "IntermediateRecorder",
    "TorchHookHandles",
    "UnsupportedIntermediateLocalization",
    "attach_torch_hooks",
    "attach_torch_module_hooks",
    "attach_callable_hook",
    "capture_callable_intermediates",
    "first_intermediate_divergence",
    "instrument_callable",
    "localize_intermediate_divergence",
]
