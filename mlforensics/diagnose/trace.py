"""Bounded tensor provenance and non-finite-value tracing."""

from __future__ import annotations

import inspect
import math
import weakref
from collections import OrderedDict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..core import Incident, RunCapsule
from ..core import TraceEvent as CoreTraceEvent


class TraceEvent(CoreTraceEvent):
    """Core trace record with convenient tensor-diagnosis accessors."""

    @property
    def is_abnormal(self) -> bool:
        return bool(self.data.get("abnormal", False))

    @property
    def tensor_id(self) -> str | None:
        value = self.data.get("tensor_id")
        return str(value) if value is not None else None

    @property
    def operation(self) -> str:
        return self.kind


def _flatten(value: Any, limit: int = 200_000) -> list[float]:
    if hasattr(value, "detach"):
        try:
            flatten = getattr(value, "flatten", None)
            sampled = flatten() if callable(flatten) else value
            numel = int(getattr(sampled, "numel", lambda: limit)())
            if numel > limit:
                step = max(1, numel // limit)
                sampled = sampled.reshape(-1)[::step][:limit]
            sampled = sampled.detach()
            cpu = getattr(sampled, "cpu", None)
            if callable(cpu):
                sampled = cpu()
            value = sampled
        except Exception:
            try:
                value = value.detach()
            except Exception:
                pass
    flatten = getattr(value, "flatten", None)
    if callable(flatten):
        try:
            value = flatten()[:limit]
        except Exception:
            pass
    elif hasattr(value, "ravel"):
        try:
            value = value.ravel()[:limit]
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    result: list[float] = []

    def visit(item: Any) -> None:
        if len(result) >= limit:
            return
        if hasattr(item, "detach"):
            try:
                item = item.detach().cpu()
            except Exception:
                pass
        item_flatten = getattr(item, "flatten", None)
        if callable(item_flatten):
            try:
                item = item_flatten()[: max(0, limit - len(result))]
            except Exception:
                pass
        item_tolist = getattr(item, "tolist", None)
        if callable(item_tolist):
            try:
                item = item_tolist()
            except Exception:
                pass
        if isinstance(item, Mapping):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                pass

    visit(value)
    return result


def tensor_event(
    operation: str,
    value: Any,
    *,
    source: str | None = None,
    tensor_id: str | None = None,
    parents: Iterable[str] = (),
    event_type: str = "tensor",
    step: int | None = None,
    message: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    **extra: Any,
) -> TraceEvent:
    """Create a portable event with finite statistics and provenance.

    ``step`` records the training step the tensor belongs to, which is the key
    needed to correlate an anomaly with the rest of the run. Any further keyword
    arguments are retained as event metadata.
    """
    values = _flatten(value)
    finite = [item for item in values if math.isfinite(item)]
    minimum = min(finite) if finite else None
    maximum = max(finite) if finite else None
    average = sum(finite) / len(finite) if finite else None
    variance = sum((item - average) ** 2 for item in finite) / len(finite) if finite else None
    standard_deviation = math.sqrt(variance) if variance is not None else None
    shape = list(getattr(value, "shape", ())) or None
    element_count = None
    if shape is not None:
        element_count = math.prod(shape)
    dtype = str(getattr(value, "dtype", "")) or None
    device = str(getattr(value, "device", "")) or None
    gradient_norm = extra.pop("gradient_norm", None)
    if gradient_norm is None:
        gradient = getattr(value, "grad", None) if getattr(value, "is_leaf", True) else None
        if gradient is not None:
            try:
                gradient_values = _flatten(gradient)
                gradient_norm = math.sqrt(sum(item * item for item in gradient_values))
            except (TypeError, ValueError, OverflowError):
                gradient_norm = None
        elif event_type == "backward":
            try:
                gradient_norm = math.sqrt(sum(item * item for item in values))
            except (TypeError, ValueError, OverflowError):
                gradient_norm = None
    data = {
        "event_type": event_type,
        "shape": shape,
        "dtype": dtype,
        "device": device,
        "minimum": minimum,
        "min": minimum,
        "maximum": maximum,
        "max": maximum,
        "mean": average,
        "std": standard_deviation,
        "finite_fraction": len(finite) / len(values) if values else None,
        "finite_percentage": (100.0 * len(finite) / len(values)) if values else None,
        "gradient_norm": gradient_norm,
        "element_count": element_count,
        "inspected_count": len(values),
        "truncated": element_count is not None and element_count > len(values),
        "source": source,
        **({"tensor_id": tensor_id} if tensor_id is not None else {}),
        "parents": list(parents),
        "abnormal": bool(values) and len(finite) != len(values),
        **extra,
        **dict(metadata or {}),
    }
    return TraceEvent(kind=operation, message=message or "", step=step, data=data)


def _is_abnormal(event: TraceEvent) -> bool:
    return event.is_abnormal


class TraceBuffer:
    """A fixed-size ring buffer whose event count is bounded."""

    def __init__(self, max_events: int = 2_048) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self.max_events = max_events
        self._events: deque[TraceEvent] = deque(maxlen=max_events)
        self._next_id = 0
        self._dropped = 0
        self._evicted_ids: deque[str] = deque(maxlen=max_events * 2)
        self._first_abnormal: TraceEvent | None = None

    def record(self, event: TraceEvent | Mapping[str, Any]) -> TraceEvent:
        if isinstance(event, CoreTraceEvent) and not isinstance(event, TraceEvent):
            event = TraceEvent(
                kind=event.kind,
                message=event.message,
                timestamp=event.timestamp,
                step=event.step,
                data=event.data,
            )
        elif not isinstance(event, TraceEvent):
            data = dict(event)
            if "kind" in data:
                data.pop("schema_version", None)
                data.pop("type", None)
                allowed = {"kind", "message", "timestamp", "step", "data"}
                event = TraceEvent(**{key: value for key, value in data.items() if key in allowed})
            else:
                event = tensor_event(
                    data.pop("operation", "tensor"), data.pop("value", None), **data
                )
        event = replace(
            event,
            data={**event.data, "tensor_id": event.data.get("tensor_id", f"t{self._next_id}")},
        )
        self._next_id += 1
        if len(self._events) == self.max_events:
            self._dropped += 1
            evicted = self._events[0]
            if evicted.tensor_id is not None:
                self._evicted_ids.append(evicted.tensor_id)
        self._events.append(event)
        if event.is_abnormal and self._first_abnormal is None:
            self._first_abnormal = event
        return event

    def record_tensor(self, operation: str, value: Any, **kwargs: Any) -> TraceEvent:
        return self.record(tensor_event(operation, value, **kwargs))

    def events(self) -> list[TraceEvent]:
        return list(self._events)

    def first_abnormal(self) -> TraceEvent | None:
        return self._first_abnormal or next(
            (event for event in self._events if _is_abnormal(event)), None
        )

    first_abnormal_event = first_abnormal

    def around(self, event: TraceEvent | None = None, radius: int = 8) -> list[TraceEvent]:
        if radius < 0:
            raise ValueError("radius must be non-negative")
        records = list(self._events)
        event = event or self.first_abnormal()
        if event is None:
            return records
        try:
            index = records.index(event)
        except ValueError:
            # The first abnormal event is deliberately retained as a
            # diagnostic anchor even after ring eviction.  Its neighbours are
            # necessarily only the newest retained records, but returning the
            # anchor makes that limitation explicit to consumers.
            if event is self._first_abnormal:
                return [event, *records[-radius:]] if radius else [event]
            return records
        return records[max(0, index - radius) : index + radius + 1]

    @property
    def dropped_events(self) -> int:
        return self._dropped

    def ancestry(self, event: TraceEvent | str | None = None) -> list[TraceEvent]:
        """Return retained causal ancestors in topological order."""
        selected = event or self.first_abnormal()
        tensor_id = (
            selected
            if isinstance(selected, str)
            else (selected.tensor_id if selected is not None else None)
        )
        if tensor_id is None:
            return []
        by_id = {item.tensor_id: item for item in self._events if item.tensor_id is not None}
        if self._first_abnormal is not None and self._first_abnormal.tensor_id is not None:
            by_id.setdefault(self._first_abnormal.tensor_id, self._first_abnormal)
        ordered: list[TraceEvent] = []
        visiting: set[str] = set()
        seen: set[str] = set()

        def visit(identifier: str) -> None:
            if identifier in seen or identifier in visiting:
                return
            visiting.add(identifier)
            current = by_id.get(identifier)
            if current is not None:
                for parent in current.data.get("parents", ()):
                    visit(str(parent))
                ordered.append(current)
            visiting.discard(identifier)
            seen.add(identifier)

        visit(tensor_id)
        return ordered

    def analyze(self, *, radius: int = 8) -> dict[str, Any]:
        """Summarize the first anomaly and its retained causal path."""
        first = self.first_abnormal()
        ancestry = self.ancestry(first)
        known = {item.tensor_id for item in ancestry}
        missing = sorted(
            {
                str(parent)
                for item in ancestry
                for parent in item.data.get("parents", ())
                if str(parent) not in known
            }
        )
        roots = [
            item
            for item in ancestry
            if not item.data.get("parents")
            or all(str(parent) not in known for parent in item.data.get("parents", ()))
        ]
        missing_explanations = [
            {
                "parent_id": parent,
                "reason": (
                    "evicted_from_ring_buffer"
                    if parent in self._evicted_ids
                    else "not_recorded_or_outside_trace_window"
                ),
                "explanation": (
                    "the parent was observed before the bounded trace buffer window"
                    if parent in self._evicted_ids
                    else "no retained event with this parent id is available"
                ),
            }
            for parent in missing
        ]
        window = self.around(first, radius=radius)
        return {
            "first_abnormal": first.to_dict() if first else None,
            "causal_path": [item.to_dict() for item in ancestry],
            "root_candidates": [item.to_dict() for item in roots],
            "missing_parents": missing,
            "missing_parent_explanations": missing_explanations,
            "causal_window": [item.to_dict() for item in window],
            "window_radius": radius,
            "event_count": len(self._events),
            "dropped_events": self._dropped,
            "bounded": True,
        }

    def to_dict(self) -> dict[str, Any]:
        first = self.first_abnormal()
        return {
            "max_events": self.max_events,
            "events": [item.to_dict() for item in self._events],
            "first_abnormal": first.to_dict() if first else None,
            "dropped_events": self._dropped,
            "analysis": self.analyze(),
        }


class TensorTracer:
    def __init__(self, *, max_events: int = 2_048, enabled: bool = True) -> None:
        self.buffer, self.enabled = TraceBuffer(max_events), enabled
        self._tensor_ids: OrderedDict[int, tuple[weakref.ReferenceType[Any], str]] = OrderedDict()
        self._identity_limit = max_events * 4

    def __enter__(self) -> TensorTracer:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        return False

    @staticmethod
    def _tensor_objects(value: Any) -> list[Any]:
        found: list[Any] = []

        def visit(item: Any) -> None:
            if isinstance(item, Mapping):
                for child in item.values():
                    visit(child)
            elif isinstance(item, (tuple, list)):
                for child in item:
                    visit(child)
            elif hasattr(item, "shape") or hasattr(item, "detach"):
                found.append(item)

        visit(value)
        return found

    def record(
        self, operation: str, value: Any, *, inputs: Any = None, **kwargs: Any
    ) -> TraceEvent | None:
        if not self.enabled:
            return None
        parents = list(kwargs.pop("parents", ()))
        for item in self._tensor_objects(inputs):
            entry = self._tensor_ids.get(id(item))
            identifier = entry[1] if entry is not None and entry[0]() is item else None
            if identifier is not None and identifier not in parents:
                parents.append(identifier)
        event = self.buffer.record_tensor(operation, value, parents=parents, **kwargs)
        identifier = event.tensor_id
        if identifier is not None:
            for item in self._tensor_objects(value):
                key = id(item)
                try:
                    reference = weakref.ref(item)
                except TypeError:
                    continue
                self._tensor_ids[key] = (reference, identifier)
                self._tensor_ids.move_to_end(key)
            while len(self._tensor_ids) > self._identity_limit:
                self._tensor_ids.popitem(last=False)
        return event

    def events(self) -> list[TraceEvent]:
        """Return the retained events, matching :meth:`TraceBuffer.events`."""
        return self.buffer.events()

    def analyze(self) -> dict[str, Any]:
        """Summarize the first anomaly and its retained causal path."""
        return self.buffer.analyze()

    def to_dict(self) -> dict[str, Any]:
        """Return portable trace evidence suitable for ``record_evidence``."""
        return self.buffer.to_dict()


def attach_torch_hooks(
    module: Any,
    tracer: TensorTracer,
    *,
    backward: bool = False,
    operators: Sequence[str] | None = None,
) -> list[Any]:
    try:
        named_modules = module.named_modules()
    except AttributeError as exc:
        raise TypeError("module must expose named_modules()") from exc
    allowed = set(operators) if operators is not None else None
    handles = []
    for name, child in named_modules:
        if not name:
            continue
        if allowed is not None and name not in allowed and type(child).__name__ not in allowed:
            continue
        source = f"{type(child).__module__}.{type(child).__qualname__}"
        source_location = None
        try:
            source_location = inspect.getfile(type(child))
            source = f"{source} ({source_location})"
        except (OSError, TypeError):
            pass

        def forward_hook(
            _child: Any,
            _inputs: Any,
            output: Any,
            operation: str = name,
            origin: str = source,
        ) -> None:
            tracer.record(
                operation,
                output,
                inputs=_inputs,
                source=origin,
                module=operation,
                source_location=source_location,
                phase="forward",
            )

        try:
            handles.append(child.register_forward_hook(forward_hook))
        except AttributeError:
            continue
        if backward:

            def backward_hook(
                _child: Any,
                _grad_input: Any,
                grad_output: Any,
                operation: str = f"{name}.backward",
                origin: str = source,
            ) -> None:
                value = (
                    grad_output[0]
                    if isinstance(grad_output, tuple) and grad_output
                    else grad_output
                )
                tracer.record(
                    operation,
                    value,
                    source=origin,
                    module=operation.rsplit(".", 1)[0],
                    source_location=source_location,
                    event_type="backward",
                    phase="backward",
                )

            register = getattr(child, "register_full_backward_hook", None) or getattr(
                child, "register_backward_hook", None
            )
            if callable(register):
                try:
                    handles.append(register(backward_hook))
                except Exception:
                    pass
    return handles


def persist_trace(
    capsule: RunCapsule, tracer: TensorTracer, *, output: str | Any | None = None
) -> dict[str, Any]:
    """Attach trace analysis to capsule evidence and return the report."""
    report = tracer.analyze()
    report["missing_history"] = bool(report.get("missing_parents") or report.get("dropped_events"))
    evidence = dict(capsule.evidence)
    evidence["tensor_trace"] = {**tracer.to_dict(), "report": report}
    object.__setattr__(capsule, "evidence", evidence)
    if output is not None:
        capsule.save(output, overwrite=True)
    return report


def auto_trace_module(
    module: Any, *, max_events: int = 2_048, backward: bool = False
) -> tuple[TensorTracer, list[Any]]:
    tracer = TensorTracer(max_events=max_events)
    handles = attach_torch_hooks(module, tracer, backward=backward)
    return tracer, handles


def _tensor_trace_evidence(incident: Incident | RunCapsule) -> tuple[list[Any], Mapping[str, Any]]:
    """Return recorded tensor-trace events, if the capsule carries any."""
    evidence = getattr(incident, "evidence", None)
    if not isinstance(evidence, Mapping):
        return ([], {})
    recorded = evidence.get("tensor_trace", ())
    if isinstance(recorded, Mapping):
        return (
            list(recorded.get("events", ()))
            if isinstance(recorded.get("events", ()), (list, tuple))
            else [],
            recorded,
        )
    return (list(recorded) if isinstance(recorded, (list, tuple)) else [], {})


def _event_identity(event: Any) -> tuple[Any, ...]:
    """Return a comparison key that survives the dict/record round trip."""
    if isinstance(event, Mapping):
        kind = event.get("kind")
        message = event.get("message")
        step = event.get("step")
        timestamp = event.get("timestamp")
        data = event.get("data") or {}
    else:
        kind = getattr(event, "kind", None)
        message = getattr(event, "message", None)
        step = getattr(event, "step", None)
        timestamp = getattr(event, "timestamp", None)
        data = getattr(event, "data", None) or {}
    tensor_id = data.get("tensor_id") if isinstance(data, Mapping) else None
    return (kind, message, step, timestamp, tensor_id)


def trace_incident(
    incident: Incident | RunCapsule,
    events: Iterable[TraceEvent] | TraceBuffer | None = None,
    *,
    radius: int = 8,
) -> dict[str, Any]:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    trace_metadata: Mapping[str, Any] = {}
    if events is None and isinstance(incident, RunCapsule):
        # Tensor-trace evidence and run events are complementary: a run normally
        # records coarse markers *and* detailed tensor provenance, so both have to
        # be considered or the recorded provenance becomes unreachable. The traced
        # sequence is kept contiguous so a window around an anomaly shows the
        # neighbouring operations rather than unrelated run markers.
        merged, trace_metadata = _tensor_trace_evidence(incident)
        recorded_first = trace_metadata.get("first_abnormal")
        if isinstance(recorded_first, Mapping):
            first_identity = _event_identity(recorded_first)
            if first_identity not in {_event_identity(item) for item in merged}:
                # TraceBuffer keeps the first abnormal anchor separately from
                # the ring. Re-introduce it for causal analysis after a
                # capsule round trip, while retaining the truncation marker.
                merged.insert(0, recorded_first)
        seen = {_event_identity(item) for item in merged}
        for item in incident.run.events:
            identity = _event_identity(item)
            if identity not in seen:
                seen.add(identity)
                merged.append(item)
        events = merged
    if events is None:
        events = ()
    if isinstance(events, TraceBuffer):
        retained = events.events()
        anchor = events.first_abnormal()
        trace_metadata = {
            "max_events": events.max_events,
            "dropped_events": events.dropped_events,
            "first_abnormal": anchor.to_dict() if anchor is not None else None,
        }
        records = list(retained)
        if anchor is not None and _event_identity(anchor) not in {
            _event_identity(item) for item in records
        }:
            records.insert(0, anchor)
    else:
        records = list(events)
    records = [
        item
        if isinstance(item, TraceEvent)
        else TraceEvent.from_dict(item)
        if isinstance(item, Mapping)
        else TraceEvent(
            kind=item.kind,
            message=item.message,
            timestamp=item.timestamp,
            step=item.step,
            data=item.data,
        )
        for item in records
    ]
    try:
        incident.traces = records  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        pass
    configured_capacity = trace_metadata.get("max_events")
    capacity = (
        int(configured_capacity)
        if isinstance(configured_capacity, int) and configured_capacity > 0
        else max(1, len(records))
    )
    analysis_buffer = TraceBuffer(capacity)
    for item in records:
        analysis_buffer.record(item)
    recorded_drops = trace_metadata.get("dropped_events", 0)
    if isinstance(recorded_drops, int) and recorded_drops > 0:
        # The source ring may have evicted more history than the merged run
        # markers did; retain the larger, already-bounded count.
        analysis_buffer._dropped = max(analysis_buffer._dropped, recorded_drops)
    analysis = analysis_buffer.analyze(radius=radius)
    return {
        "first_abnormal": analysis["first_abnormal"],
        "events": [event.to_dict() for event in records],
        "incident_id": getattr(incident, "incident_id", None),
        "run_id": incident.run.run_id if isinstance(incident, RunCapsule) else incident.run_id,
        "causal_path": analysis["causal_path"],
        "root_candidates": analysis["root_candidates"],
        "missing_parents": analysis["missing_parents"],
        "missing_parent_explanations": analysis["missing_parent_explanations"],
        "causal_window": analysis["causal_window"],
        "window_radius": radius,
        "dropped_events": analysis["dropped_events"],
        "bounded": True,
    }


@dataclass
class _NamedTensorEvent:
    value: Any
    step: int | float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    finite: bool = True


@dataclass
class _NamedTraceRecord:
    name: str
    event: _NamedTensorEvent


def detect_non_finite(value: Any) -> tuple[bool, list[str]]:
    """Return whether a nested value contains NaN/Inf and their paths."""
    paths: list[str] = []

    def visit(item: Any, path: str) -> None:
        if hasattr(item, "detach"):
            try:
                item = item.detach().cpu()
            except Exception:
                pass
        if hasattr(item, "tolist"):
            try:
                item = item.tolist()
            except Exception:
                pass
        if isinstance(item, Mapping):
            for key, child in item.items():
                visit(child, f"{path}[{key!r}]")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        else:
            try:
                if not math.isfinite(float(item)):
                    paths.append(path)
            except (TypeError, ValueError):
                pass

    visit(value, "$")
    return bool(paths), paths


class RingBufferTrace:
    """Compatibility ring buffer for named tensor diagnostics."""

    def __init__(self, capacity: int = 2_048) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._events: deque[_NamedTraceRecord] = deque(maxlen=capacity)
        self._first_abnormal: _NamedTraceRecord | None = None

    def add(
        self,
        name: str,
        value: Any,
        *,
        step: int | float | None = None,
        provenance: Mapping[str, Any] | None = None,
    ) -> _NamedTraceRecord:
        abnormal, _paths = detect_non_finite(value)
        record = _NamedTraceRecord(
            name, _NamedTensorEvent(value, step, dict(provenance or {}), not abnormal)
        )
        self._events.append(record)
        if abnormal and self._first_abnormal is None:
            self._first_abnormal = record
        return record

    def events(self) -> list[_NamedTraceRecord]:
        return list(self._events)

    def first_abnormal_event(self) -> _NamedTraceRecord | None:
        if any(item is self._first_abnormal for item in self._events):
            return self._first_abnormal
        return next((item for item in self._events if not item.event.finite), self._first_abnormal)

    def __len__(self) -> int:
        return len(self._events)

    def to_dict(self) -> dict[str, Any]:
        first = self.first_abnormal_event()
        return {
            "capacity": self.capacity,
            "events": [
                {
                    "name": item.name,
                    "step": item.event.step,
                    "provenance": dict(item.event.provenance),
                    "finite": item.event.finite,
                }
                for item in self._events
            ],
            "first_abnormal": first.name if first is not None else None,
        }

    def clear(self) -> None:
        self._events.clear()
        self._first_abnormal = None
