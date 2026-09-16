"""Framework-neutral execution capture and subprocess runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

# The core records are a hard requirement, not an optional extra: every other
# subsystem imports them unguarded, so degrading to the compatibility shadow
# model here would only hide a broken install behind weaker evidence.
from ..core import (
    ArtifactRef,
    DatasetRef,
    FailureSignature,
    LocalArtifactStore,
    MetricSeries,
    ModelRef,
    ResourceSeries,
    RNGState,
    Run,
    RunCapsule,
    TraceEvent,
    exception_signature,
    utc_now,
)
from .data import fingerprint_dataset
from .dependencies import capture_dependencies as inventory_dependencies
from .environment import (
    DEFAULT_ENV_ALLOWLIST,
    capture_dependencies,
    capture_environment,
    capture_environment_variables,
    capture_hardware,
    capture_python_environment,
)
from .fingerprint import fingerprint_dataset as content_fingerprint
from .git import capture_git
from .models import (
    RunCapsule as PortableRunCapsule,
)
from .models import (
    json_safe,
    record_exception_on_capsule,
    record_on_capsule,
)
from .system import capture_system_metadata

RUN_ID_ENV = "MLFORENSICS_RUN_ID"
CHILD_CAPSULE_ENV = "MLFORENSICS_CHILD_CAPSULE"


def child_capture_environment(
    capsule_path: str | Path,
    *,
    run_id: str,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build an environment for an instrumented subprocess handoff.

    A child using ``mlforensics.capture()`` automatically adopts the run id and
    writes its canonical capsule to ``capsule_path``.  The explicit path makes
    handoff local, deterministic, and independent of a server.
    """
    if not run_id:
        raise ValueError("run_id must be non-empty")
    result = dict(os.environ if environ is None else environ)
    result[RUN_ID_ENV] = run_id
    result[CHILD_CAPSULE_ENV] = str(Path(capsule_path).absolute())
    return result


def load_child_capsule(path: str | Path) -> Any:
    """Load and integrity-check a capsule written through the child protocol."""
    return RunCapsule.load(path)


def capture_rng_state() -> RNGState:
    states: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np  # type: ignore

        states["numpy"] = np.random.get_state()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        states["torch_cpu"] = torch.get_rng_state().tolist()
        if torch.cuda.is_available():
            states["torch_cuda"] = [state.tolist() for state in torch.cuda.get_rng_state_all()]
    except Exception:
        pass
    try:
        return RNGState(states=states)
    except TypeError:
        # The core RNGState intentionally stores compact, serialized fields;
        # the legacy session keeps richer Python objects for its own fallback
        # capsule and therefore uses the local compatibility model.
        from ._fallback import RNGState as FallbackRNGState

        return FallbackRNGState(states=states)  # type: ignore[return-value]


class CaptureSession(AbstractContextManager):
    """Collect evidence while user code executes.

    The session intentionally exposes simple recording methods so integrations
    can instrument their own training loops without inheriting from a base
    trainer class.
    """

    def __init__(
        self,
        *,
        root: str | Path = ".mlforensics",
        command: Sequence[str] | None = None,
        replay: bool = False,
        run_id: str | None = None,
        env_allowlist: Iterable[str] | None = None,
        include_git_diff: bool = True,
        metadata: Mapping[str, Any] | None = None,
        autosave: bool = True,
        name: str | None = None,
        replay_input: Any = None,
        replay_seed: int | None = None,
        checkpoint_limit: int = 3,
    ) -> None:
        self.root = Path(root)
        self.autosave = autosave
        self.rng = capture_rng_state()
        # ``CaptureSession`` predates the immutable core records and keeps a
        # mutable compatibility model for its append-style API.  Keep the
        # selected record family together; otherwise a core ``MetricSeries``
        # can accidentally be inserted into the fallback ``Run`` and fail
        # only when the first metric is recorded.
        self._ArtifactRef = ArtifactRef
        self._DatasetRef = DatasetRef
        self._FailureSignature = FailureSignature
        self._LocalArtifactStore = LocalArtifactStore
        self._MetricSeries = MetricSeries
        self._ModelRef = ModelRef
        self._ResourceSeries = ResourceSeries
        self._TraceEvent = TraceEvent
        self._exception_signature = exception_signature
        self._utc_now = utc_now
        self._legacy_model = False
        inherited_run_id = os.environ.get(RUN_ID_ENV)
        run_id = run_id or inherited_run_id or f"run-{uuid.uuid4().hex[:8]}"
        child_target = os.environ.get(CHILD_CAPSULE_ENV)
        self._child_capsule_path = Path(child_target) if child_target else None
        if (
            isinstance(checkpoint_limit, bool)
            or not isinstance(checkpoint_limit, int)
            or checkpoint_limit < 0
        ):
            raise ValueError("checkpoint_limit must be a non-negative integer")
        self._checkpoint_limit = checkpoint_limit
        self._replay: dict[str, Any] = {}
        self._captured_payloads: dict[str, bytes] = {}
        self._observation_counter: dict[tuple[str, str], int] = {}
        run_metadata = dict(metadata or {})
        if name is not None:
            run_metadata["name"] = name
        environment = capture_environment(allowlist=env_allowlist)
        hardware = capture_hardware()
        dependencies = capture_dependencies()
        try:
            self.run = Run(
                run_id=run_id,
                command=list(command or ()),
                git=capture_git(include_diff=include_git_diff),
                environment=environment,
                hardware=hardware,
                dependencies=dependencies,
                metadata={**run_metadata, "replay_enabled": bool(replay)},
            )
        except TypeError:
            from ._fallback import (
                ArtifactRef as FallbackArtifactRef,
            )
            from ._fallback import (
                DatasetRef as FallbackDatasetRef,
            )
            from ._fallback import (
                FailureSignature as FallbackFailureSignature,
            )
            from ._fallback import (
                LocalArtifactStore as FallbackLocalArtifactStore,
            )
            from ._fallback import (
                MetricSeries as FallbackMetricSeries,
            )
            from ._fallback import (
                ModelRef as FallbackModelRef,
            )
            from ._fallback import (
                ResourceSeries as FallbackResourceSeries,
            )
            from ._fallback import (
                Run as FallbackRun,
            )
            from ._fallback import (
                TraceEvent as FallbackTraceEvent,
            )
            from ._fallback import (
                exception_signature as fallback_exception_signature,
            )
            from ._fallback import (
                utc_now as fallback_utc_now,
            )

            self.run = FallbackRun(
                run_id=run_id,
                command=list(command or ()),
                git=capture_git(include_diff=include_git_diff),
                environment=environment,
                hardware=hardware,
                dependencies=dependencies,
                metadata={**run_metadata, "replay_enabled": bool(replay)},
            )  # type: ignore[assignment]
            self._ArtifactRef = FallbackArtifactRef
            self._DatasetRef = FallbackDatasetRef
            self._FailureSignature = FallbackFailureSignature
            self._LocalArtifactStore = FallbackLocalArtifactStore
            self._MetricSeries = FallbackMetricSeries
            self._ModelRef = FallbackModelRef
            self._ResourceSeries = FallbackResourceSeries
            self._TraceEvent = FallbackTraceEvent
            self._exception_signature = fallback_exception_signature
            self._utc_now = fallback_utc_now
            self._legacy_model = True
        try:
            self._working = RunCapsule(
                run=self.run,
                code={"git": self.run.git},
                environment=environment,
                hardware=hardware,
                dependencies=dependencies,
                randomness=self.rng.to_dict(),
                training={"metrics": [], "events": []},
                system={"resource_trace": []},
            )
            # Core capsules may intentionally be immutable and only contain a
            # Run plus artifact payloads.  The legacy session needs sections
            # it can update during execution, so use the local adapter there.
            if not all(
                hasattr(self._working, name)
                for name in ("randomness", "training", "system", "extra")
            ):
                raise TypeError("core capsule does not expose mutable session sections")
        except (TypeError, AttributeError):
            from ._fallback import RunCapsule as FallbackRunCapsule

            self._working = FallbackRunCapsule(
                run=self.run,
                code={"git": self.run.git},
                environment=environment,
                hardware=hardware,
                dependencies=dependencies,
                randomness=self.rng.to_dict(),
                training={"metrics": [], "events": []},
                system={"resource_trace": []},
            )
        self._artifact_store: LocalArtifactStore | None = None
        self._canonical: Any | None = None
        self._core_failure: Any | None = None
        self._closed = False
        if replay_input is not None:
            self.record_replay_input(replay_input)
        if replay_seed is not None:
            self.record_replay_seed(replay_seed)

    @property
    def run_id(self) -> str:
        return self.run.run_id

    @property
    def capsule(self) -> Any:
        """Return the canonical capsule, in the format ``RunCapsule.load`` reads.

        The session keeps a mutable working record while user code runs, but
        that internal shape is not the portable capsule format. Exposing it here
        would let ``session.capsule.save(...)`` write a file that no other
        command in the library can open.
        """
        if self._canonical is not None:
            return self._canonical
        portable = self._portable_capsule()
        canonical = portable if portable is not None else self._working
        if self._closed:
            self._canonical = canonical
        return canonical

    def __enter__(self) -> CaptureSession:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_value is not None:
            self.run.status = "failed"
            self.run.failure = self._exception_signature(exc_value)
            self._working.failure = self.run.failure.to_dict()
            # Build the canonical signature straight from the live exception so
            # the originating frame and real traceback survive; reconstructing it
            # later from the compatibility record can only lose that detail.
            self._core_failure = exception_signature(exc_value)
        elif self.run.status == "running":
            self.run.status = "succeeded"
        self.run.ended_at = self._utc_now()
        self._working.randomness.setdefault("end", capture_rng_state().to_dict())
        self._working.training["metrics"] = [metric.to_dict() for metric in self.run.metrics]
        self._working.training["events"] = [
            event for event in self._working.training.get("events", [])
        ]
        self._closed = True
        if self.autosave:
            self.save(self._child_capsule_path)
        return False

    def save(self, path: str | Path | None = None, *, overwrite: bool = False) -> Path:
        target_root = Path(path) if path is not None else self.root
        if target_root.suffix in {".mlcap", ".mlcapdir", ".zip"}:
            target = target_root
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target_root.mkdir(parents=True, exist_ok=True)
            target = target_root / f"{self.run_id}.mlcap"
        capsule = self.capsule
        try:
            return capsule.save(target, overwrite=overwrite)
        except TypeError:
            return capsule.save(target)

    def _portable_capsule(self) -> Any | None:
        """Translate the mutable compatibility session to a core capsule.

        The legacy session remains mutable for callers that use its original
        append-style API, but persistence uses the canonical core format so a
        capsule produced by either capture entry point can be opened by
        ``mlforensics.RunCapsule.load``.
        """
        # The working record only counts as canonical when it is a core capsule.
        # Checking the run type instead would accept the compatibility capsule,
        # which serialises to a shape the rest of the library cannot read.
        if isinstance(self._working, RunCapsule) and not self._legacy_model:
            return self._working
        try:
            from ..core import (
                ArtifactRef as CoreArtifactRef,
            )
            from ..core import (
                CheckpointRef as CoreCheckpointRef,
            )
            from ..core import (
                DatasetRef as CoreDatasetRef,
            )
            from ..core import (
                FailureSignature as CoreFailureSignature,
            )
            from ..core import (
                MetricSeries as CoreMetricSeries,
            )
            from ..core import (
                ModelRef as CoreModelRef,
            )
            from ..core import (
                ReplayPlan as CoreReplayPlan,
            )
            from ..core import (
                ResourceSeries as CoreResourceSeries,
            )
            from ..core import (
                RNGState as CoreRNGState,
            )
            from ..core import (
                Run as CoreRun,
            )
            from ..core import (
                RunCapsule as CoreRunCapsule,
            )
            from ..core import (
                StateSnapshot as CoreStateSnapshot,
            )
            from ..core import (
                TraceEvent as CoreTraceEvent,
            )
        except Exception:
            return None

        legacy_run = self.run
        metrics = []
        observations = []
        for metric in legacy_run.metrics:
            raw_values = tuple(metric.values)
            raw_steps = tuple(metric.steps)
            raw_timestamps = tuple(metric.timestamps)
            values: list[float] = []
            steps: list[int | float] = []
            timestamps: list[str] = []
            identities: list[int | float] = []
            for position, value in enumerate(raw_values):
                identity = raw_steps[position] if len(raw_steps) == len(raw_values) else position
                timestamp = (
                    raw_timestamps[position] if len(raw_timestamps) == len(raw_values) else None
                )
                try:
                    finite = not isinstance(value, bool) and math.isfinite(float(value))
                except (TypeError, ValueError, OverflowError):
                    finite = False
                if finite:
                    values.append(float(value))
                    identities.append(identity)
                    if len(raw_steps) == len(raw_values):
                        steps.append(identity)
                    if timestamp is not None:
                        timestamps.append(timestamp)
                else:
                    from ..core import Observation as CoreObservation

                    observations.append(
                        CoreObservation.from_value(
                            metric.name,
                            value,
                            identity=identity,
                            step=identity,
                            timestamp=timestamp,
                            metadata={
                                **dict(metric.metadata),
                                "kind": "metric",
                                **({"split": metric.split} if metric.split else {}),
                            },
                        )
                    )
            if values or not raw_values:
                metrics.append(
                    CoreMetricSeries(
                        name=metric.name,
                        values=tuple(values),
                        steps=tuple(steps) if len(steps) == len(values) and steps else (),
                        timestamps=(
                            tuple(timestamps)
                            if len(timestamps) == len(values) and timestamps
                            else ()
                        ),
                        metadata={
                            **dict(metric.metadata),
                            **({"split": metric.split} if metric.split else {}),
                        },
                        identities=tuple(identities),
                    )
                )
        resources = []
        for resource in legacy_run.resources:
            raw_values = tuple(resource.values)
            raw_steps = tuple(resource.steps)
            values: list[float] = []
            steps: list[int | float] = []
            identities: list[int | float] = []
            for position, value in enumerate(raw_values):
                identity = raw_steps[position] if len(raw_steps) == len(raw_values) else position
                try:
                    finite = not isinstance(value, bool) and math.isfinite(float(value))
                except (TypeError, ValueError, OverflowError):
                    finite = False
                if finite:
                    values.append(float(value))
                    identities.append(identity)
                    if len(raw_steps) == len(raw_values):
                        steps.append(identity)
                else:
                    from ..core import Observation as CoreObservation

                    observations.append(
                        CoreObservation.from_value(
                            resource.name,
                            value,
                            identity=identity,
                            step=identity,
                            metadata={"kind": "resource", **dict(resource.metadata)},
                        )
                    )
            if values or not raw_values:
                resources.append(
                    CoreResourceSeries(
                        name=resource.name,
                        values=tuple(values),
                        steps=tuple(steps) if len(steps) == len(values) and steps else (),
                        units=resource.units,
                        metadata=dict(resource.metadata),
                        identities=tuple(identities),
                    )
                )
        datasets = []
        for dataset in legacy_run.datasets:
            fingerprint = dataset.fingerprint or ""
            artifact = None
            if len(fingerprint) == 64:
                artifact = CoreArtifactRef(
                    name=dataset.name,
                    uri=dataset.uri or "",
                    sha256=fingerprint,
                    metadata={"sample_count": dataset.sample_count, **dict(dataset.metadata)},
                )
            datasets.append(
                CoreDatasetRef(
                    name=dataset.name,
                    artifact=artifact,
                    format=dataset.metadata.get("format"),
                    metadata={
                        "sample_count": dataset.sample_count,
                        "schema": dataset.schema,
                        **dict(dataset.metadata),
                    },
                )
            )
        models = []
        if legacy_run.model is not None:
            models.append(
                CoreModelRef(
                    name=legacy_run.model.name,
                    framework=legacy_run.model.framework,
                    metadata={"architecture": legacy_run.model.architecture},
                )
            )
        events = []
        for event in getattr(self._working, "training", {}).get("events", []):
            if isinstance(event, Mapping):
                event_name = str(event.get("name", event.get("kind", "event")))
                payload = event.get("payload", event.get("data", {}))
                events.append(
                    CoreTraceEvent(
                        kind=event_name,
                        data=payload if isinstance(payload, Mapping) else {"value": payload},
                    )
                )
        rng_state = getattr(self.rng, "states", {})
        encoded_rng = {
            str(key): json.dumps(
                json_safe(value), sort_keys=True, separators=(",", ":"), default=repr
            )
            for key, value in rng_state.items()
            if key in {"python", "numpy", "torch_cpu", "torch_cuda"}
        }
        core_rng = CoreRNGState(
            python=encoded_rng.get("python"),
            numpy=encoded_rng.get("numpy"),
            frameworks={
                key: value for key, value in encoded_rng.items() if key not in {"python", "numpy"}
            },
        )
        failure = self._core_failure
        if failure is None and legacy_run.failure is not None:
            error_type = legacy_run.failure.exception_type or legacy_run.failure.kind
            message = legacy_run.failure.normalized_message or ""
            failure = CoreFailureSignature(
                error_type=error_type,
                message=message,
                normalized_message=message,
                traceback_hash=hashlib.sha256(message.encode("utf-8")).hexdigest(),
                top_frame=getattr(legacy_run.failure, "top_frame", None),
                exception_chain=(error_type,),
            )
        artifact_refs = []
        for artifact in getattr(legacy_run, "artifacts", []):
            digest = artifact.sha256 or ""
            if len(digest) != 64:
                continue
            artifact_refs.append(
                CoreArtifactRef(
                    name=Path(artifact.uri).name or artifact.kind,
                    uri=artifact.uri,
                    sha256=digest,
                    size_bytes=artifact.size_bytes,
                    media_type=artifact.media_type,
                    metadata=dict(artifact.metadata),
                )
            )

        def replay_snapshot(name: str, value: Any) -> Any:
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
                    artifact = next((ref for ref in artifact_refs if ref.sha256 == digest), None)
            return CoreStateSnapshot(
                name=name,
                codec=codec,
                value=value,
                artifact=artifact,
                dtype=metadata.get("dtype"),
                shape=tuple(metadata.get("shape", ())) if metadata.get("shape") else (),
                device=metadata.get("device"),
                metadata=metadata,
            )

        named_replay_state = self._replay.get("state", {})
        replay_state = (
            tuple(replay_snapshot(str(name), value) for name, value in named_replay_state.items())
            if isinstance(named_replay_state, Mapping)
            else ()
        )
        replay_checkpoints = []
        raw_checkpoints = self._replay.get("checkpoints", ())
        if isinstance(raw_checkpoints, Sequence) and not isinstance(raw_checkpoints, (str, bytes)):
            for index, raw in enumerate(raw_checkpoints):
                if not isinstance(raw, Mapping):
                    continue
                raw_step = raw.get("step")
                if isinstance(raw_step, bool) or not isinstance(raw_step, (int, float)):
                    continue
                checkpoint_state = raw.get("state", {})
                snapshots = (
                    tuple(
                        replay_snapshot(str(name), value)
                        for name, value in checkpoint_state.items()
                    )
                    if isinstance(checkpoint_state, Mapping)
                    else ()
                )
                replay_checkpoints.append(
                    CoreCheckpointRef(
                        checkpoint_id=str(
                            raw.get("checkpoint_id", f"{legacy_run.run_id}:{raw_step}:{index}")
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
                            CoreRNGState.from_dict(raw["rng_state"])
                            if isinstance(raw.get("rng_state"), Mapping)
                            else None
                        ),
                    )
                )
        limitations = []
        if self._replay.get("state_capture_errors"):
            limitations.append("state_capture_errors present")
        if self._replay.get("checkpoint_capture_errors"):
            limitations.append("checkpoint_capture_errors present")
        if not replay_state:
            limitations.append("no named application state was captured")
        replay_plan = None
        if self._replay:
            replay_plan = CoreReplayPlan(
                input=self._replay.get("input"),
                state=replay_state,
                expected_failure=failure,
                determinism="best_effort",
                restore_order=tuple(item.name for item in replay_state),
                checkpoints=tuple(replay_checkpoints),
                limitations=tuple(dict.fromkeys(limitations)),
                metadata={"checkpoint_limit": self._checkpoint_limit},
            )
        metadata = {
            **dict(legacy_run.metadata),
            "command": list(legacy_run.command),
            "git": legacy_run.git,
            "environment": legacy_run.environment,
            "hardware": legacy_run.hardware,
            "dependencies": legacy_run.dependencies,
        }
        status = "completed" if legacy_run.status == "succeeded" else legacy_run.status
        core_run = CoreRun(
            run_id=legacy_run.run_id,
            name=str(metadata.get("name", "")),
            status=status,
            started_at=legacy_run.started_at,
            ended_at=legacy_run.ended_at,
            metadata=metadata,
            datasets=tuple(datasets),
            models=tuple(models),
            metrics=tuple(metrics),
            resources=tuple(resources),
            events=tuple(events),
            rng_state=core_rng,
            failure_signature=failure,
            observations=tuple(observations),
            replay_plan=replay_plan,
        )
        evidence = {
            "replay": dict(self._replay),
            "observations": [item.to_dict() for item in observations],
        }
        evidence = {key: value for key, value in evidence.items() if value not in ({}, [])}
        return CoreRunCapsule(
            core_run,
            tuple(artifact_refs),
            {digest: payload for digest, payload in self._captured_payloads.items()},
            evidence=evidence,
        )

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        split: str | None = None,
        timestamp: str | None = None,
        **metadata: Any,
    ) -> MetricSeries:
        metric = self.run.metric(name, split)
        if metric is None:
            metric = self._MetricSeries(name=name, split=split, metadata=metadata)
            self.run.metrics.append(metric)
        elif step is not None and not metric.steps:
            metric.steps.extend(range(len(metric.values)))
        if metric is not None and timestamp is not None and not metric.timestamps:
            metric.timestamps.extend(self._utc_now() for _ in metric.values)
        metric.append(value, step=step, timestamp=timestamp)
        return metric

    def record_resource(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        units: str | None = None,
        **metadata: Any,
    ) -> ResourceSeries:
        series = next((item for item in self.run.resources if item.name == name), None)
        if series is None:
            series = self._ResourceSeries(name=name, units=units, metadata=metadata)
            self.run.resources.append(series)
        elif step is not None and not series.steps:
            series.steps.extend(range(len(series.values)))
        series.append(value, step=step)
        self._working.system["resource_trace"] = [item.to_dict() for item in self.run.resources]
        return series

    def add_dataset(
        self,
        source: str | Path | Iterable[Mapping[str, Any]],
        *,
        name: str | None = None,
        **metadata: Any,
    ) -> DatasetRef:
        record = fingerprint_dataset(source, name=name)
        ref = self._DatasetRef(
            name=record.pop("name"),
            fingerprint=record.get("sha256"),
            sample_count=record.get("row_count", record.get("sample_count")),
            schema=record.get("schema", {}),
            metadata={**record, **metadata},
        )
        self.run.datasets.append(ref)
        self._working.data.setdefault("fingerprints", []).append(ref.to_dict())
        return ref

    def set_model(
        self,
        model: ModelRef | Mapping[str, Any] | Any,
        *,
        name: str = "model",
        framework: str | None = None,
    ) -> ModelRef:
        if isinstance(model, self._ModelRef):
            ref = model
        elif isinstance(model, Mapping):
            try:
                ref = self._ModelRef.from_dict(model)
            except (KeyError, TypeError, ValueError):
                ref = self._ModelRef(
                    name=str(model.get("name", name)),
                    framework=model.get("framework", framework),
                    architecture=dict(model),
                )
        else:
            architecture = {
                "type": f"{type(model).__module__}.{type(model).__qualname__}",
                "repr": repr(model)[:2_000],
            }
            ref = self._ModelRef(name=name, framework=framework, architecture=architecture)
        self.run.model = ref
        self._working.model = ref.to_dict()
        return ref

    def record_event(self, event: TraceEvent | Mapping[str, Any]) -> TraceEvent:
        if isinstance(event, self._TraceEvent):
            record = event
        elif self._legacy_model:
            data = dict(event)
            if "kind" in data and "name" not in data:
                data["name"] = data["kind"]
            event_name = str(data.pop("name", data.pop("kind", "event")))
            payload = data.pop("payload", data.pop("data", data))
            record = self._TraceEvent(
                name=event_name,
                payload=dict(payload) if isinstance(payload, Mapping) else {"value": payload},
            )
        else:
            data = dict(event)
            data.pop("schema_version", None)
            data.pop("type", None)
            if "kind" not in data:
                data["kind"] = str(data.pop("name", "event"))
            if "data" not in data and "payload" in data:
                data["data"] = data.pop("payload")
            allowed = {"kind", "message", "timestamp", "step", "data"}
            record = self._TraceEvent(
                **{key: value for key, value in data.items() if key in allowed}
            )
        self._working.training.setdefault("events", []).append(record.to_dict())
        return record

    def add_artifact(
        self,
        source: str | Path | bytes | bytearray,
        *,
        kind: str = "file",
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRef:
        if isinstance(source, (bytes, bytearray)):
            payload = bytes(source)
        else:
            payload = Path(source).read_bytes()
        if self._artifact_store is None:
            self._artifact_store = self._LocalArtifactStore(self.root / "artifacts")
        if hasattr(self._artifact_store, "put"):
            try:
                ref = self._artifact_store.put(
                    source, kind=kind, media_type=media_type, metadata=dict(metadata or {})
                )
            except TypeError:
                ref = self._artifact_store.put(source, kind=kind, media_type=media_type)
        elif isinstance(source, (bytes, bytearray)):
            ref = self._artifact_store.put_bytes(bytes(source), name=kind, media_type=media_type)
        else:
            ref = self._artifact_store.put_file(source, media_type=media_type)
        self.run.artifacts.append(ref)
        digest = getattr(ref, "sha256", None)
        if digest:
            self._captured_payloads[str(digest)] = payload
        return ref

    def _replay_value(self, value: Any, *, name: str) -> Any:
        """Encode replay input/state and retain binary payloads in the capsule."""
        from ..core.codecs import encode_state_tree

        def store_payload(payload: bytes, metadata: Mapping[str, Any]) -> str:
            ref = self.add_artifact(
                payload,
                kind=f"{name}.bin",
                media_type="application/octet-stream",
                metadata={"role": "replay_state", **dict(metadata)},
            )
            return ref.sha256

        return encode_state_tree(value, store_payload, name=name)

    def record_replay_input(self, value: Any, *, name: str = "input") -> Any:
        self._replay["input"] = self._replay_value(value, name=f"replay-{name}")
        return self._replay["input"]

    def record_replay_seed(self, seed: int) -> int:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("replay seed must be an integer")
        self._replay["seed"] = seed
        return seed

    def record_state(self, name: str, state: Any) -> Any:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("state name must be a non-empty string")
        states = self._replay.setdefault("state", {})
        value = self._replay_value(state, name=f"state-{name}")
        states[name] = value
        return value

    snapshot_state = record_state

    def record_checkpoint(
        self,
        step: int,
        *,
        state_providers: Mapping[str, Any] | None = None,
        batch: Any = None,
        epoch: int | None = None,
        sampler_position: int | None = None,
        sample_ids: Sequence[Any] | None = None,
        before_step: bool = True,
    ) -> Mapping[str, Any]:
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError("checkpoint step must be an integer")
        checkpoints = self._replay.setdefault("checkpoints", [])
        if self._checkpoint_limit == 0:
            return {"step": step, "stored": False, "reason": "checkpoint_limit=0"}
        state: dict[str, Any] = {}
        for name, provider in dict(state_providers or {}).items():
            snapshot = getattr(provider, "snapshot", None)
            state[str(name)] = self._replay_value(
                snapshot() if callable(snapshot) else provider,
                name=f"checkpoint-{step}-{name}",
            )
        checkpoint = {
            "checkpoint_id": f"{self.run_id}:{step}:{len(checkpoints)}",
            "step": step,
            "before_step": bool(before_step),
            "state": state,
            "batch": self._replay_value(batch, name=f"checkpoint-{step}-batch")
            if batch is not None
            else None,
            "epoch": epoch,
            "sampler_position": sampler_position,
            "sample_ids": self._replay_value(list(sample_ids), name=f"checkpoint-{step}-ids")
            if sample_ids is not None
            else None,
        }
        checkpoints.append(checkpoint)
        self._replay["last_checkpoint"] = checkpoint
        del checkpoints[: max(0, len(checkpoints) - self._checkpoint_limit)]
        self._evict_unreferenced_replay_payloads()
        return checkpoint

    def _evict_unreferenced_replay_payloads(self) -> None:
        try:
            from ..core.codecs import collect_artifact_digests
        except Exception:
            return
        live = collect_artifact_digests(self._replay)
        kept: dict[str, bytes] = {}
        for digest, payload in self._captured_payloads.items():
            artifact = next(
                (item for item in self.run.artifacts if getattr(item, "sha256", None) == digest),
                None,
            )
            metadata = getattr(artifact, "metadata", {}) if artifact is not None else {}
            role = metadata.get("role") if isinstance(metadata, Mapping) else None
            if role != "replay_state" or digest in live:
                kept[digest] = payload
        self._captured_payloads = kept

    checkpoint_before_step = record_checkpoint

    def record_offending_batch(
        self,
        batch: Any,
        *,
        step: int | None = None,
        sample_ids: Sequence[Any] | None = None,
    ) -> Any:
        value = self.record_replay_input(batch, name="offending-batch")
        if step is not None:
            self._replay["step"] = step
        if sample_ids is not None:
            self._replay["sample_ids"] = self._replay_value(list(sample_ids), name="sample-ids")
        return value


def capture(**kwargs: Any) -> CaptureSession:
    return CaptureSession(**kwargs)


def run_command(
    command: Sequence[str],
    *,
    root: str | Path = ".mlforensics",
    capture_output: bool = True,
    check: bool = False,
    **capture_kwargs: Any,
) -> tuple[int, CaptureSession, subprocess.CompletedProcess[str]]:
    """Run a command under capture and return its exit code, session, and process result."""

    if not command:
        raise ValueError("command cannot be empty")
    with CaptureSession(root=root, command=command, **capture_kwargs) as session:
        child_path = session.root / f"{session.run_id}.child.mlcap"
        result = subprocess.run(
            list(command),
            text=True,
            capture_output=capture_output,
            check=False,
            env=child_capture_environment(child_path, run_id=session.run_id),
        )
        session._working.extra["process"] = {
            "returncode": result.returncode,
            "stdout": result.stdout[-100_000:] if capture_output else None,
            "stderr": result.stderr[-100_000:] if capture_output else None,
            "child_capsule": str(child_path) if child_path.exists() else None,
        }
        if result.returncode and not check:
            failure = session._exception_signature(
                subprocess.CalledProcessError(
                    result.returncode, list(command), result.stdout, result.stderr
                )
            )
            session.run.status = "failed"
            session.run.failure = failure
            session._working.failure = failure.to_dict()
        if check and result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, list(command), result.stdout, result.stderr
            )
    return result.returncode, session, result


def _portable_rng_snapshot() -> dict[str, Any]:
    """Capture loaded RNG providers without making optional frameworks mandatory."""

    states: dict[str, Any] = {"python": random.getstate()}
    numpy = sys.modules.get("numpy")
    if numpy is not None:
        try:
            states["numpy"] = numpy.random.get_state()
        except Exception:
            pass
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            states["torch_cpu"] = torch.random.get_rng_state().tolist()
        except Exception:
            pass
    return json_safe(states)


class CaptureContext(AbstractContextManager):
    """Framework-neutral metadata context backed by a portable RunCapsule."""

    def __init__(
        self,
        repo_path: str | Path | None = ".",
        data_paths: Mapping[str, Any] | Iterable[str | Path] | str | Path | None = None,
        env_allowlist: Sequence[str] | None = None,
        environ: Mapping[str, str] | None = None,
        capsule: Any | None = None,
        use_core_capsule: bool = True,
    ) -> None:
        self.repo_path = repo_path
        self.data_paths = data_paths
        self.env_allowlist = env_allowlist
        self.environ = environ
        self.capsule = capsule if capsule is not None else PortableRunCapsule()
        self._captured = False

    def _section(self, name: str, producer: Any) -> None:
        try:
            record_on_capsule(self.capsule, name, producer())
        except Exception as exc:
            record_exception_on_capsule(self.capsule, exc, where=name)

    def _capture_data(self) -> dict[str, Any]:
        if self.data_paths is None:
            return {}
        items = (
            self.data_paths.items()
            if isinstance(self.data_paths, Mapping)
            else [("data", self.data_paths)]
        )
        return {str(name): content_fingerprint(paths) for name, paths in items}

    def capture(self) -> Any:
        if self._captured:
            return self.capsule
        manifest = {
            "schema_version": getattr(self.capsule, "schema_version", "1"),
            "capture_package": "mlforensics.capture",
            "repo_path": str(self.repo_path) if self.repo_path is not None else None,
            "data_labels": sorted(str(key) for key in self.data_paths)
            if isinstance(self.data_paths, Mapping)
            else (["data"] if self.data_paths is not None else []),
        }
        record_on_capsule(self.capsule, "manifest", manifest)
        self._section(
            "environment",
            lambda: {
                "python": capture_python_environment(),
                "variables": capture_environment_variables(self.env_allowlist, self.environ),
                "allowlist": list(
                    DEFAULT_ENV_ALLOWLIST if self.env_allowlist is None else self.env_allowlist
                ),
            },
        )
        self._section(
            "code",
            lambda: (
                capture_git(self.repo_path) if self.repo_path is not None else {"available": False}
            ),
        )
        self._section("hardware", capture_system_metadata)
        self._section("dependencies", inventory_dependencies)
        self._section("data_fingerprints", self._capture_data)
        self._section("rng_snapshots", _portable_rng_snapshot)
        self._captured = True
        return self.capsule

    collect = capture

    def record_metric(self, name: str, value: Any) -> None:
        recorder = getattr(self.capsule, "record_metric", None)
        if callable(recorder):
            recorder(name, value)
        elif isinstance(getattr(self.capsule, "metrics", None), dict):
            self.capsule.metrics[str(name)] = json_safe(value)

    metric = record_metric

    def record_exception(self, exception: BaseException, where: str | None = None) -> None:
        record_exception_on_capsule(self.capsule, exception, where)

    def fingerprint(self, name: str, path: Any) -> Any:
        result = content_fingerprint(path)
        if isinstance(getattr(self.capsule, "data_fingerprints", None), dict):
            self.capsule.data_fingerprints[str(name)] = result
        return result

    def __enter__(self) -> CaptureContext:
        self.capture()
        return self

    def __exit__(self, exc_type: Any, exc_value: BaseException | None, traceback: Any) -> bool:
        if exc_value is not None:
            self.record_exception(exc_value, where="workload")
        snapshots = getattr(self.capsule, "rng_snapshots", None)
        if isinstance(snapshots, dict):
            snapshots["end"] = _portable_rng_snapshot()
        return False


class CaptureRunner:
    """Execute a callable under :class:`CaptureContext` and return its capsule."""

    def __init__(
        self,
        context: CaptureContext | None = None,
        raise_exceptions: bool = True,
        **context_kwargs: Any,
    ) -> None:
        self.context = context or CaptureContext(**context_kwargs)
        self.raise_exceptions = raise_exceptions
        self.result: Any = None

    @property
    def capsule(self) -> Any:
        return self.context.capsule

    def run(
        self,
        function: Any,
        *args: Any,
        metrics: Mapping[str, Any] | Any | None = None,
        **kwargs: Any,
    ) -> Any:
        try:
            with self.context as active:
                if isinstance(metrics, Mapping):
                    for name, value in metrics.items():
                        active.record_metric(name, value)
                self.result = function(*args, **kwargs)
                if callable(metrics):
                    for name, value in metrics(self.result).items():
                        active.record_metric(name, value)
        except BaseException:
            if self.raise_exceptions:
                raise
        return self.context.capsule

    execute = run
