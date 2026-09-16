"""Framework-neutral capture context and extension hooks."""

from __future__ import annotations

import json
import math
import os
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .capsule import RunCapsule
from .codecs import collect_artifact_digests, encode_state_tree
from .models import (
    ArtifactRef,
    CheckpointRef,
    DatasetRef,
    FailureSignature,
    LineageEdge,
    LineageNode,
    MetricSeries,
    ModelRef,
    Observation,
    ReplayPlan,
    ResourceSeries,
    RNGState,
    Run,
    StateSnapshot,
    TraceEvent,
    _now,
)
from .store import LocalArtifactStore


@runtime_checkable
class CaptureHook(Protocol):
    """Optional callbacks for framework integrations.

    Hooks are deliberately structural and receive only mlforensics records or
    the capture context. An integration can implement any subset of callbacks;
    absent callbacks are simply ignored.
    """

    def on_start(self, capture: CaptureContext) -> None: ...
    def on_event(self, event: TraceEvent) -> None: ...
    def on_artifact(self, ref: ArtifactRef) -> None: ...
    def on_finish(self, run: Run) -> None: ...
    def on_failure(self, failure: FailureSignature) -> None: ...


def _call(hooks: Iterable[CaptureHook], method: str, value: Any) -> None:
    for hook in hooks:
        callback = getattr(hook, method, None)
        if callback is not None:
            callback(value)


def _json_default(value: Any) -> Any:
    # RNG snapshots are compact JSON strings.  A 1-D uint8 tensor may be stored
    # as a list of integers; that is not advertised as application replay state.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return tolist()
        except Exception:
            pass
    if isinstance(value, bytes):
        return list(value)
    return repr(value)


def _encode_state(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def _portable_state(value: Any) -> Any:
    """Convert metadata containers to finite JSON values.

    Application replay state must use :func:`encode_state_tree` instead.  This
    helper is only for structured evidence that is not advertised as replayable.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError("state contains a non-finite float")
        return value
    if isinstance(value, bytes):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _portable_state(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _portable_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_state(item) for item in value]
    raise TypeError(
        f"state value {type(value).__module__}.{type(value).__qualname__} is not portable; "
        "return JSON-compatible values or bytes from the snapshot hook"
    )


def _take_snapshot(provider: Any) -> Any:
    snapshot = getattr(provider, "snapshot", None)
    if callable(snapshot):
        return snapshot()
    state_dict = getattr(provider, "state_dict", None)
    if callable(state_dict):
        return state_dict()
    return provider() if callable(provider) else provider


def _replay_value(context: CaptureContext, value: Any, *, name: str) -> Any:
    """Encode replay values recursively, embedding binary arrays/tensors."""

    def store_payload(payload: bytes, metadata: Mapping[str, Any]) -> str:
        ref = context.artifact(
            payload,
            name=f"{name}.bin",
            media_type="application/octet-stream",
            metadata={"role": "replay_state", **dict(metadata)},
        )
        return ref.sha256

    return encode_state_tree(value, store_payload, name=name)


def _snapshot_rng_state() -> RNGState:
    """Capture loaded RNG providers as JSON strings with no hard dependencies."""
    frameworks: dict[str, str] = {}
    python_state = _encode_state(random.getstate())
    numpy_state = None
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        try:
            numpy_state = _encode_state(numpy.random.get_state())
        except Exception:
            pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            frameworks["torch_cpu"] = _encode_state(torch.random.get_rng_state().tolist())
            if torch.cuda.is_available():
                frameworks["torch_cuda"] = _encode_state(
                    [state.tolist() for state in torch.cuda.get_rng_state_all()]
                )
        except Exception:
            pass
    return RNGState(python=python_state, numpy=numpy_state, frameworks=frameworks)


class CaptureContext:
    """Collect evidence in-process and expose it as a :class:`RunCapsule`.

    The context does not inspect or import a framework. Adapters can translate
    framework callbacks into ``event``, ``metric``, ``artifact``, and lineage
    calls, or observe those calls through ``CaptureHook`` implementations.
    """

    def __init__(
        self,
        name: str = "",
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        hooks: Iterable[CaptureHook] = (),
        artifact_store: LocalArtifactStore | None = None,
        root: str | os.PathLike[str] | None = None,
        state_providers: Mapping[str, Any] | None = None,
        replay_input: Any = None,
        replay_seed: int | None = None,
        evidence: Mapping[str, Any] | None = None,
        checkpoint_limit: int = 3,
    ) -> None:
        if root is not None and artifact_store is not None:
            raise ValueError("pass either root or artifact_store, not both")
        if root is not None:
            artifact_store = LocalArtifactStore(Path(root) / "artifacts")
        if run_id is None:
            inherited = os.environ.get("MLFORENSICS_RUN_ID")
            run_id = inherited or None
        self._run = Run(run_id=run_id) if run_id else Run()
        self._run = replace(
            self._run, name=name, metadata=dict(metadata or {}), rng_state=_snapshot_rng_state()
        )
        self._hooks = tuple(hooks)
        self._artifact_store = artifact_store
        self._artifacts: list[ArtifactRef] = []
        self._payloads: dict[str, bytes] = {}
        self._metrics: list[MetricSeries] = []
        self._resources: list[ResourceSeries] = []
        self._events: list[TraceEvent] = []
        self._observations: list[Observation] = []
        self._observation_counters: dict[tuple[str, str], int] = {}
        self._nodes: list[LineageNode] = []
        self._edges: list[LineageEdge] = []
        self._state_providers = dict(state_providers or {})
        self._evidence: dict[str, Any] = dict(evidence or {})
        if (
            isinstance(checkpoint_limit, bool)
            or not isinstance(checkpoint_limit, int)
            or checkpoint_limit < 0
        ):
            raise ValueError("checkpoint_limit must be a non-negative integer")
        self._checkpoint_limit = checkpoint_limit
        self._closed = False
        if replay_input is not None:
            self.record_replay_input(replay_input)
        if replay_seed is not None:
            self.record_replay_seed(replay_seed)
        self._capsule: RunCapsule | None = None

    @property
    def run(self) -> Run:
        return self._current_run()

    @property
    def run_id(self) -> str:
        return self._run.run_id

    @property
    def capsule(self) -> RunCapsule:
        if self._capsule is None:
            raise RuntimeError("capture is still active; use capsule after exiting the context")
        return self._capsule

    def __enter__(self) -> CaptureContext:
        _call(self._hooks, "on_start", self)
        # Capture pre-step application state before user code mutates it.
        if self._state_providers:
            self._snapshot_registered_state(role="replay")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: Any
    ) -> bool:
        if exc is not None:
            self.fail(exc)
        else:
            self.finish()
        return False

    def _current_run(self) -> Run:
        return replace(
            self._run,
            metrics=tuple(self._metrics),
            resources=tuple(self._resources),
            events=tuple(self._events),
            lineage_nodes=tuple(self._nodes),
            lineage_edges=tuple(self._edges),
            observations=tuple(self._observations),
        )

    def _next_observation_identity(
        self,
        kind: str,
        name: str,
        step: int | float | None,
        metadata: Mapping[str, Any] | None,
    ) -> Any:
        metadata = metadata or {}
        for key in ("identity", "observation_id", "sample_id", "case_id", "seed"):
            if key in metadata and metadata[key] is not None:
                identity = metadata[key]
                break
        else:
            identity = step
        if identity is None:
            counter_key = (kind, name)
            identity = self._observation_counters.get(counter_key, 0)
        counter_key = (kind, name)
        self._observation_counters[counter_key] = self._observation_counters.get(counter_key, 0) + 1
        return identity

    @staticmethod
    def _series_identities(series: MetricSeries | ResourceSeries) -> tuple[Any, ...]:
        if getattr(series, "identities", ()):
            return tuple(series.identities)
        return tuple(series.observation_ids)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("capture is already closed")

    def event(
        self,
        kind: str | TraceEvent,
        *,
        message: str = "",
        step: int | float | None = None,
        data: dict[str, Any] | None = None,
    ) -> TraceEvent:
        self._ensure_open()
        event = (
            kind
            if isinstance(kind, TraceEvent)
            else TraceEvent(kind=kind, message=message, step=step, data=data or {})
        )
        self._events.append(event)
        _call(self._hooks, "on_event", event)
        return event

    def record_event(
        self, event: TraceEvent | Mapping[str, Any] | str, **kwargs: Any
    ) -> TraceEvent:
        """Record an event from a record, mapping, or event kind."""
        if isinstance(event, TraceEvent):
            return self.event(event)
        if isinstance(event, Mapping):
            data = dict(event)
            data.pop("schema_version", None)
            data.pop("type", None)
            if "kind" not in data:
                data["kind"] = str(data.pop("name", "event"))
            if "data" not in data and "payload" in data:
                data["data"] = data.pop("payload")
            allowed = {"kind", "message", "timestamp", "step", "data"}
            return self.event(
                TraceEvent(**{key: value for key, value in data.items() if key in allowed})
            )
        return self.event(event, **kwargs)

    def metric(self, series: MetricSeries) -> MetricSeries:
        self._ensure_open()
        self._metrics.append(series)
        return series

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        step: int | float | None = None,
        timestamp: str | None = None,
        **metadata: Any,
    ) -> MetricSeries:
        """Append a value to a named metric series."""
        self._ensure_open()
        if not isinstance(name, str) or not name.strip():
            raise ValueError("metric name must be a non-empty string")
        identity = self._next_observation_identity("metric", name, step, metadata)
        if not self._is_finite_number(value):
            return self._record_nonfinite_observation(
                name,
                value,
                kind="metric",
                step=step,
                timestamp=timestamp,
                metadata=metadata,
                identity=identity,
            )
        value = float(value)
        index = next(
            (
                position
                for position, series in enumerate(self._metrics)
                if series.name == name
                and all(
                    series.metadata.get(key) == value
                    for key, value in metadata.items()
                    if key not in {"identity", "observation_id", "sample_id", "case_id", "seed"}
                )
            ),
            None,
        )
        if index is None:
            series = MetricSeries(
                name=name,
                values=(value,),
                steps=(step,) if step is not None else (),
                timestamps=(timestamp,) if timestamp is not None else (),
                metadata=metadata,
                identities=(identity,),
            )
            self._metrics.append(series)
            return series
        current = self._metrics[index]
        steps = tuple(current.steps)
        if steps or step is not None:
            steps = (steps if steps else tuple(range(len(current.values)))) + (
                step if step is not None else len(current.values),
            )
        timestamps = tuple(current.timestamps)
        if timestamps or timestamp is not None:
            timestamps = (timestamps if timestamps else tuple(_now() for _ in current.values)) + (
                timestamp if timestamp is not None else _now(),
            )
        identities = self._series_identities(current) + (identity,)
        series = MetricSeries(
            name=name,
            values=tuple(current.values) + (value,),
            steps=steps,
            timestamps=timestamps,
            metadata={**dict(current.metadata), **metadata},
            identities=identities,
        )
        self._metrics[index] = series
        return series

    metric_value = record_metric

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        try:
            return not isinstance(value, bool) and math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False

    def _record_nonfinite_observation(
        self,
        name: str,
        value: Any,
        *,
        kind: str,
        step: int | float | None,
        timestamp: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        identity: Any = None,
    ) -> MetricSeries | ResourceSeries:
        observation = Observation.from_value(
            name,
            value,
            identity=identity if identity is not None else step,
            step=step,
            timestamp=timestamp,
            metadata={"kind": kind, **dict(metadata or {})},
        )
        self._observations.append(observation)
        self._evidence.setdefault("observations", []).append(observation.to_dict())
        # Keep the return contract useful to callers that append values without
        # ever exposing an invalid float through MetricSeries/ResourceSeries.
        if kind == "metric":
            existing = next((item for item in self._metrics if item.name == name), None)
            return existing or MetricSeries(name=name, values=(), metadata=dict(metadata or {}))
        existing_resource = next((item for item in self._resources if item.name == name), None)
        return existing_resource or ResourceSeries(
            name=name, values=(), metadata=dict(metadata or {})
        )

    def resource(self, series: ResourceSeries) -> ResourceSeries:
        self._ensure_open()
        self._resources.append(series)
        return series

    def record_resource(
        self,
        name: str,
        value: float,
        *,
        step: int | float | None = None,
        units: str | None = None,
        **metadata: Any,
    ) -> ResourceSeries:
        """Append a value to a named resource series."""
        self._ensure_open()
        identity = self._next_observation_identity("resource", name, step, metadata)
        if not self._is_finite_number(value):
            return self._record_nonfinite_observation(
                name,
                value,
                kind="resource",
                step=step,
                metadata={"units": units, **metadata},
                identity=identity,
            )
        value = float(value)
        index = next(
            (position for position, series in enumerate(self._resources) if series.name == name),
            None,
        )
        if index is None:
            series = ResourceSeries(
                name=name,
                values=(value,),
                steps=(step,) if step is not None else (),
                units=units,
                metadata=metadata,
                identities=(identity,),
            )
            self._resources.append(series)
            return series
        current = self._resources[index]
        steps = tuple(current.steps)
        if steps or step is not None:
            steps = (steps if steps else tuple(range(len(current.values)))) + (
                step if step is not None else len(current.values),
            )
        series = ResourceSeries(
            name=name,
            values=tuple(current.values) + (value,),
            steps=steps,
            units=units or current.units,
            metadata={**dict(current.metadata), **metadata},
            identities=self._series_identities(current) + (identity,),
        )
        self._resources[index] = series
        return series

    resource_value = record_resource

    def artifact(
        self,
        content: bytes | str | os.PathLike[str],
        *,
        name: str | None = None,
        media_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        self._ensure_open()
        source_name = name
        if isinstance(content, (str, os.PathLike)):
            path = Path(content)
            source_name = source_name or path.name
            blob = path.read_bytes()
            if self._artifact_store is not None:
                ref = self._artifact_store.put_file(
                    path, name=source_name, media_type=media_type, metadata=metadata
                )
            else:
                ref = ArtifactRef.from_bytes(
                    source_name, blob, media_type=media_type, metadata=metadata
                )
        else:
            blob = content
            source_name = source_name or "artifact"
            if self._artifact_store is not None:
                ref = self._artifact_store.put_bytes(
                    blob, name=source_name, media_type=media_type, metadata=metadata
                )
            else:
                ref = ArtifactRef.from_bytes(
                    source_name, blob, media_type=media_type, metadata=metadata
                )
        # Capsules are immutable evidence.  Keep the payload embedded even when
        # a configured artifact store also receives a copy.
        self._payloads[ref.sha256] = blob
        self._artifacts.append(ref)
        _call(self._hooks, "on_artifact", ref)
        return ref

    add_artifact = artifact

    def register_state(self, name: str, provider: Any) -> None:
        """Register application/framework state to snapshot when capture closes.

        Providers may expose ``snapshot()``, ``state_dict()``, be callables, or
        be plain state values.  Bytes are embedded as content-addressed
        artifacts; common tensor/array values are converted to portable JSON.
        """
        self._ensure_open()
        if not isinstance(name, str) or not name.strip():
            raise ValueError("state name must be a non-empty string")
        self._state_providers[name] = provider

    def record_state(self, name: str, state: Any) -> Any:
        """Snapshot one named state immediately and return its portable value."""
        self._ensure_open()
        if not isinstance(name, str) or not name.strip():
            raise ValueError("state name must be a non-empty string")
        value = _replay_value(self, _take_snapshot(state), name=f"state-{name}")
        replay = self._evidence.setdefault("replay", {})
        states = replay.setdefault("state", {})
        states[name] = value
        return value

    snapshot_state = record_state

    def record_replay_input(self, value: Any, *, name: str = "input") -> Any:
        """Attach the exact input used by a later capsule-based replay."""
        self._ensure_open()
        portable = _replay_value(self, value, name=f"replay-{name}")
        replay = self._evidence.setdefault("replay", {})
        replay["input"] = portable
        return portable

    def record_replay_seed(self, seed: int) -> int:
        """Record the logical seed in addition to exact RNG snapshots."""
        self._ensure_open()
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("replay seed must be an integer")
        self._evidence.setdefault("replay", {})["seed"] = seed
        return seed

    def record_evidence(self, section: str, value: Any) -> None:
        """Record a named structured evidence section on the capsule."""
        self._ensure_open()
        if not isinstance(section, str) or not section.strip():
            raise ValueError("evidence section must be a non-empty string")
        portable = _portable_state(value)
        if isinstance(portable, bytes):
            raise TypeError("structured evidence must be JSON-compatible, not bytes")
        self._evidence[section] = portable

    def _snapshot_registered_state(self, *, role: str = "replay") -> None:
        errors: dict[str, str] = {}
        states: dict[str, Any] = {}
        for name, provider in tuple(self._state_providers.items()):
            try:
                states[str(name)] = _replay_value(
                    self, _take_snapshot(provider), name=f"{role}-{name}"
                )
            except BaseException as exc:
                errors[name] = f"{type(exc).__name__}: {exc}"
        for hook in self._hooks:
            snapshot = getattr(hook, "snapshot_state", None)
            if not callable(snapshot):
                continue
            try:
                values = snapshot(self)
                if not isinstance(values, Mapping):
                    raise TypeError("snapshot_state() must return a mapping")
                for name, value in values.items():
                    states[str(name)] = _replay_value(
                        self, _take_snapshot(value), name=f"{role}-hook-{name}"
                    )
            except BaseException as exc:
                errors[f"hook:{type(hook).__qualname__}"] = f"{type(exc).__name__}: {exc}"
        replay = self._evidence.setdefault("replay", {})
        if role == "diagnostic":
            replay["diagnostic_state"] = states
            replay["diagnostic_captured_after_failure"] = True
            if errors:
                replay["diagnostic_state_capture_errors"] = errors
            return
        replay.setdefault("state", {}).update(states)
        if errors:
            replay["state_capture_errors"] = {
                **dict(replay.get("state_capture_errors", {}) or {}),
                **errors,
            }

    def record_checkpoint(
        self,
        step: int | float,
        *,
        state_providers: Mapping[str, Any] | None = None,
        batch: Any = None,
        epoch: int | None = None,
        sampler_position: int | None = None,
        sample_ids: Sequence[Any] | None = None,
        before_step: bool = True,
    ) -> Mapping[str, Any]:
        """Record a bounded replay checkpoint for a training step.

        Checkpoints are metadata/state evidence rather than a framework
        checkpoint format. Providers may be registered ahead of time or
        supplied for this particular step. The newest ``checkpoint_limit``
        entries are retained, and the most recent checkpoint is also exposed
        as ``replay.last_checkpoint`` for simple replay clients.
        """
        self._ensure_open()
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, float))
            or not math.isfinite(float(step))
        ):
            raise ValueError("checkpoint step must be a finite number")
        if self._checkpoint_limit == 0:
            return {"step": step, "stored": False, "reason": "checkpoint_limit=0"}
        providers = dict(self._state_providers)
        providers.update(dict(state_providers or {}))
        state: dict[str, Any] = {}
        errors: dict[str, str] = {}
        for name, provider in providers.items():
            try:
                state[str(name)] = _replay_value(
                    self, _take_snapshot(provider), name=f"checkpoint-{step}-{name}"
                )
            except BaseException as exc:
                errors[str(name)] = f"{type(exc).__name__}: {exc}"
        try:
            checkpoint_rng = _snapshot_rng_state().to_dict()
        except BaseException as exc:
            checkpoint_rng = None
            errors["__rng__"] = f"{type(exc).__name__}: {exc}"
        portable_batch = None
        if batch is not None:
            portable_batch = _replay_value(self, batch, name=f"checkpoint-{step}-batch")
        elif sample_ids is not None:
            portable_batch = {
                "sample_ids": _replay_value(self, list(sample_ids), name=f"checkpoint-{step}-ids")
            }
        checkpoint = {
            "step": step,
            "before_step": bool(before_step),
            "rng_state": checkpoint_rng,
            "state": state,
            "batch": portable_batch,
            "epoch": epoch,
            "sampler_position": sampler_position,
            "sample_ids": (
                _replay_value(self, list(sample_ids), name=f"checkpoint-{step}-sample-ids")
                if sample_ids is not None
                else None
            ),
            "state_capture_errors": errors,
        }
        replay = self._evidence.setdefault("replay", {})
        checkpoints = replay.setdefault("checkpoints", [])
        if not isinstance(checkpoints, list):
            checkpoints = []
            replay["checkpoints"] = checkpoints
        checkpoints.append(checkpoint)
        replay["last_checkpoint"] = checkpoint
        if len(checkpoints) > self._checkpoint_limit:
            del checkpoints[: len(checkpoints) - self._checkpoint_limit]
            self._evict_unreferenced_payloads()
        if errors:
            replay.setdefault("checkpoint_capture_errors", {}).update(errors)
        return checkpoint

    def _evict_unreferenced_payloads(self) -> None:
        """Drop checkpoint payloads that are no longer referenced by evidence."""
        live = collect_artifact_digests(self._evidence)
        for ref in self._artifacts:
            if ref.metadata.get("role") != "replay_state":
                live.add(ref.sha256)
        self._payloads = {
            digest: payload for digest, payload in self._payloads.items() if digest in live
        }
        self._artifacts = [ref for ref in self._artifacts if ref.sha256 in live]

    checkpoint_before_step = record_checkpoint

    def record_offending_batch(
        self,
        batch: Any,
        *,
        step: int | float | None = None,
        sample_ids: Sequence[Any] | None = None,
    ) -> Any:
        """Attach the batch associated with a failure to replay evidence."""
        self._ensure_open()
        # Use the same codec path as named replay inputs so arrays/tensors are
        # retained as verified binary artifacts rather than silently expanded
        # into an unbounded JSON list.
        portable = _replay_value(self, batch, name="offending-batch")
        replay = self._evidence.setdefault("replay", {})
        replay["input"] = portable
        if step is not None:
            replay["step"] = step
        if sample_ids is not None:
            replay["sample_ids"] = _portable_state(list(sample_ids))
        return portable

    def add_dataset(
        self,
        dataset: Any,
        *,
        name: str | None = None,
        split: str | None = None,
        format: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DatasetRef:
        """Attach a dataset reference without importing a data framework."""
        self._ensure_open()
        if isinstance(dataset, DatasetRef):
            ref = dataset
        elif isinstance(dataset, ArtifactRef):
            ref = DatasetRef(
                name=name or dataset.name,
                artifact=dataset,
                split=split,
                format=format,
                metadata=metadata or {},
            )
        else:
            ref = DatasetRef(
                name=name or str(dataset),
                split=split,
                format=format,
                metadata={"source": str(dataset), **dict(metadata or {})},
            )
        self._run = replace(self._run, datasets=tuple(self._run.datasets) + (ref,))
        return ref

    def set_model(
        self,
        model: ModelRef | Mapping[str, Any] | Any,
        *,
        name: str = "model",
        framework: str | None = None,
    ) -> ModelRef:
        """Record a model reference or a lightweight architecture description."""
        self._ensure_open()
        if isinstance(model, ModelRef):
            ref = model
        elif isinstance(model, Mapping):
            ref = ModelRef.from_dict(model)
        else:
            ref = ModelRef(
                name=name,
                framework=framework,
                metadata={
                    "type": f"{type(model).__module__}.{type(model).__qualname__}",
                    "repr": repr(model)[:2_000],
                },
            )
        self._run = replace(self._run, models=tuple(self._run.models) + (ref,))
        return ref

    def lineage_node(self, node: LineageNode) -> LineageNode:
        self._ensure_open()
        self._nodes.append(node)
        return node

    def lineage_edge(self, edge: LineageEdge) -> LineageEdge:
        self._ensure_open()
        self._edges.append(edge)
        return edge

    def _make_replay_plan(self, failure: FailureSignature | None = None) -> ReplayPlan | None:
        replay = self._evidence.get("replay")
        if not isinstance(replay, Mapping):
            return None

        def snapshot(name: str, value: Any) -> StateSnapshot:
            artifact = None
            codec = "json"
            metadata: dict[str, Any] = {}
            if isinstance(value, Mapping):
                codec = str(value.get("codec", codec))
                metadata = {
                    str(key): item
                    for key, item in value.items()
                    if key not in {"artifact_sha256", "codec"}
                }
                digest = value.get("artifact_sha256")
                if isinstance(digest, str):
                    artifact = next((ref for ref in self._artifacts if ref.sha256 == digest), None)
            return StateSnapshot(
                name=name,
                codec=codec,
                value=value,
                artifact=artifact,
                dtype=metadata.get("dtype"),
                shape=tuple(metadata.get("shape", ())) if metadata.get("shape") else (),
                device=metadata.get("device"),
                metadata=metadata,
            )

        named_state = replay.get("state", {})
        state_records = (
            tuple(snapshot(str(name), value) for name, value in named_state.items())
            if isinstance(named_state, Mapping)
            else ()
        )
        checkpoint_records: list[CheckpointRef] = []
        raw_checkpoints = replay.get("checkpoints", ())
        if isinstance(raw_checkpoints, Sequence) and not isinstance(raw_checkpoints, (str, bytes)):
            for index, raw in enumerate(raw_checkpoints):
                if not isinstance(raw, Mapping):
                    continue
                raw_step = raw.get("step")
                if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
                    continue
                checkpoint_state = raw.get("state", {})
                snapshots = (
                    tuple(snapshot(str(name), value) for name, value in checkpoint_state.items())
                    if isinstance(checkpoint_state, Mapping)
                    else ()
                )
                checkpoint_records.append(
                    CheckpointRef(
                        checkpoint_id=str(
                            raw.get("checkpoint_id", f"{self.run_id}:{raw_step}:{index}")
                        ),
                        step=raw_step,
                        state=snapshots,
                        batch=raw.get("batch"),
                        epoch=raw.get("epoch"),
                        sampler_position=raw.get("sampler_position"),
                        metadata={
                            "before_step": bool(raw.get("before_step", True)),
                            "sample_ids": raw.get("sample_ids"),
                            "state_capture_errors": raw.get("state_capture_errors", {}),
                        },
                        rng_state=(
                            RNGState.from_dict(raw["rng_state"])
                            if isinstance(raw.get("rng_state"), Mapping)
                            else None
                        ),
                    )
                )
        limitations: list[str] = []
        for key in ("state_capture_errors", "checkpoint_capture_errors"):
            if replay.get(key):
                limitations.append(f"{key} present")
        if not state_records:
            limitations.append("no named application state was captured")
        return ReplayPlan(
            input=replay.get("input"),
            state=state_records,
            expected_failure=failure or self._run.failure_signature,
            determinism="best_effort",
            restore_order=tuple(item.name for item in state_records),
            checkpoints=tuple(checkpoint_records),
            limitations=tuple(dict.fromkeys(limitations)),
            metadata={"checkpoint_limit": self._checkpoint_limit},
        )

    def _minimal_capsule(
        self, failure: FailureSignature | None, error: BaseException | None = None
    ) -> RunCapsule:
        """Build the smallest valid capsule after normal assembly failed."""
        metadata: dict[str, Any] = {"capture_degraded": True}
        if error is not None:
            metadata["capture_error"] = f"{type(error).__name__}: {error}"
        try:
            minimal_run = Run(
                run_id=self._run.run_id,
                name=self._run.name,
                status="failed" if failure is not None else "completed",
                started_at=self._run.started_at,
                ended_at=_now(),
                metadata=metadata,
                failure_signature=failure,
                metrics=tuple(self._metrics),
                resources=tuple(self._resources),
                events=tuple(self._events),
                observations=tuple(self._observations),
            )
            return RunCapsule(
                minimal_run,
                artifacts=tuple(self._artifacts),
                payloads=self._payloads,
                evidence={
                    "capture_errors": metadata,
                    "observations": [item.to_dict() for item in self._observations],
                },
            )
        except BaseException:
            # This path is intentionally dependency-free and should only be
            # reachable if the model contract itself is damaged.
            fallback = Run(
                run_id=str(self._run.run_id), status="failed", started_at=str(self._run.started_at)
            )
            return RunCapsule(fallback)

    def _assemble_capsule(self, *, failure: FailureSignature | None = None) -> RunCapsule:
        try:
            return RunCapsule(
                self._run,
                tuple(self._artifacts),
                self._payloads,
                evidence=self._evidence,
            )
        except BaseException as exc:
            self._evidence = {"capture_errors": {"assembly": f"{type(exc).__name__}: {exc}"}}
            return self._minimal_capsule(failure, exc)

    def finish(self) -> RunCapsule:
        self._ensure_open()
        try:
            self._snapshot_registered_state(role="replay")
        except BaseException as exc:
            self._evidence.setdefault("capture_errors", {})["state"] = (
                f"{type(exc).__name__}: {exc}"
            )
        try:
            replay_plan = self._make_replay_plan()
        except BaseException as exc:
            self._evidence.setdefault("capture_errors", {})["replay_plan"] = (
                f"{type(exc).__name__}: {exc}"
            )
            replay_plan = None
        self._run = replace(
            self._current_run(), status="completed", ended_at=_now(), replay_plan=replay_plan
        )
        self._closed = True
        self._capsule = self._assemble_capsule()
        try:
            _call(self._hooks, "on_finish", self._run)
        except BaseException as exc:
            self._capsule = self._minimal_capsule(None, exc)
        self._persist_child_handoff(None)
        return self._capsule

    def fail(self, exc: BaseException) -> RunCapsule:
        self._ensure_open()
        try:
            # Post-failure snapshots are diagnostic.  Replay must use a
            # checkpoint captured at or before the requested step.
            self._snapshot_registered_state(role="diagnostic")
        except BaseException as capture_exc:
            self._evidence.setdefault("capture_errors", {})["state"] = (
                f"{type(capture_exc).__name__}: {capture_exc}"
            )
        try:
            failure = FailureSignature.from_exception(exc)
        except BaseException as signature_exc:
            failure = FailureSignature(
                error_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                message=str(exc),
                normalized_message=str(exc),
                traceback_hash="",
                exception_chain=(f"{type(exc).__module__}.{type(exc).__qualname__}",),
                kind="exception",
                details={
                    "signature_capture_error": f"{type(signature_exc).__name__}: {signature_exc}"
                },
            )
        try:
            replay_plan = self._make_replay_plan(failure)
        except BaseException as plan_exc:
            self._evidence.setdefault("capture_errors", {})["replay_plan"] = (
                f"{type(plan_exc).__name__}: {plan_exc}"
            )
            replay_plan = None
        self._run = replace(
            self._current_run(),
            status="failed",
            ended_at=_now(),
            failure_signature=failure,
            replay_plan=replay_plan,
        )
        self._closed = True
        self._capsule = self._assemble_capsule(failure=failure)
        try:
            _call(self._hooks, "on_failure", failure)
            _call(self._hooks, "on_finish", self._run)
        except BaseException as hook_exc:
            self._evidence.setdefault("capture_errors", {})["hook"] = (
                f"{type(hook_exc).__name__}: {hook_exc}"
            )
        self._persist_child_handoff(failure)
        return self._capsule

    def _persist_child_handoff(self, failure: FailureSignature | None) -> None:
        self._write_child_result(failure)
        self._persist_child_capsule()

    def _persist_child_capsule(self) -> None:
        path = os.environ.get("MLFORENSICS_CHILD_CAPSULE")
        if not path or self._capsule is None:
            return
        try:
            self._capsule.save(path, overwrite=True)
        except Exception as exc:
            self._evidence.setdefault("capture_errors", {})["child_capsule"] = (
                f"{type(exc).__name__}: {exc}"
            )

    def _write_child_result(self, failure: FailureSignature | None) -> None:
        path = os.environ.get("MLFORENSICS_CHILD_RESULT")
        if not path:
            return
        try:
            from .contracts import ExecutionResult, write_child_result

            write_child_result(
                path,
                ExecutionResult(
                    status="fail" if failure is not None else "pass",
                    failure=failure,
                    capsule=str(os.environ.get("MLFORENSICS_CHILD_CAPSULE") or ""),
                ),
            )
        except Exception:
            return

    def record_exception(self, exc: BaseException) -> RunCapsule:
        """Close the context as failed and return its capsule."""
        return self.fail(exc)
