"""State restoration and evidence-backed incident replay."""

from __future__ import annotations

import inspect
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from ..core import FailureSignature, RNGState, RunCapsule, decode_state_tree, digest_state_tree
from ..core import Incident as CoreIncident
from ..core.contracts import evaluate_predicate
from ..core.errors import CheckpointUnavailable, IncompleteReplay
from ..core.failure import exception_signature


@dataclass(slots=True)
class ReplayResult:
    reproduced: bool
    failure: FailureSignature | None = None
    result: Any = None
    restored: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def verified(self) -> bool:
        """Whether reproduction was checked against explicit evidence."""
        return bool(self.metadata.get("verified", False))

    def to_dict(self) -> dict[str, Any]:
        return {
            "reproduced": self.reproduced,
            "verified": self.verified,
            "failure": self.failure.to_dict() if self.failure else None,
            "result": repr(self.result),
            "restored": list(self.restored),
            "error": self.error,
            "metadata": dict(self.metadata),
        }


class StateRestorer(Protocol):
    def __call__(self, state: Any) -> None: ...


def _value_digest(value: Any) -> str | None:
    return digest_state_tree(value)


def _tupleize(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleize(item) for item in value)
    if isinstance(value, dict):
        return {key: _tupleize(item) for key, item in value.items()}
    return value


def _decode_state(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def restore_rng_state(state: RNGState | Mapping[str, Any]) -> list[str]:
    """Best-effort restore of captured Python, NumPy, and torch RNG state."""
    if isinstance(state, RNGState):
        states: Mapping[str, Any] = {**state.frameworks}
        if state.python is not None:
            states = {**states, "python": state.python}
        if state.numpy is not None:
            states = {**states, "numpy": state.numpy}
    else:
        candidate = state.get("states", state)
        states = candidate if isinstance(candidate, Mapping) else {}
    restored: list[str] = []
    if "python" in states:
        try:
            random.setstate(_tupleize(_decode_state(states["python"])))
            restored.append("Python RNG")
        except (TypeError, ValueError):
            pass
    if "numpy" in states:
        try:
            import numpy as np  # type: ignore

            value = _decode_state(states["numpy"])
            if isinstance(value, list) and len(value) >= 5:
                value = (
                    value[0],
                    np.asarray(value[1], dtype="uint32"),
                    int(value[2]),
                    int(value[3]),
                    float(value[4]),
                )
            np.random.set_state(value)
            restored.append("NumPy RNG")
        except Exception:
            pass
    restored_torch, _failures = _restore_rng_state_detailed(states)
    restored.extend(item for item in restored_torch if item not in restored)
    return restored


def _torch_state_tensor(torch: Any, value: Any) -> Any:
    """Turn a portable torch RNG value into the runtime's uint8 tensor."""
    value = _decode_state(value)
    if hasattr(value, "dtype") and hasattr(value, "tolist"):
        return value
    constructor = getattr(torch, "as_tensor", None) or getattr(torch, "tensor", None)
    if not callable(constructor):
        return value
    uint8 = getattr(torch, "uint8", None)
    return constructor(value, dtype=uint8) if uint8 is not None else constructor(value)


def _restore_rng_state_detailed(states: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Restore each RNG independently and retain failures for replay reports."""
    restored: list[str] = []
    failures: list[str] = []
    if "python" in states:
        try:
            random.setstate(_tupleize(_decode_state(states["python"])))
            restored.append("Python RNG")
        except Exception as exc:
            failures.append(f"Python RNG: {type(exc).__name__}: {exc}")
    if "numpy" in states:
        try:
            import numpy as np  # type: ignore

            value = _decode_state(states["numpy"])
            if isinstance(value, list) and len(value) >= 5:
                value = (
                    value[0],
                    np.asarray(value[1], dtype="uint32"),
                    int(value[2]),
                    int(value[3]),
                    float(value[4]),
                )
            np.random.set_state(value)
            restored.append("NumPy RNG")
        except Exception as exc:
            failures.append(f"NumPy RNG: {type(exc).__name__}: {exc}")
    if "torch_cpu" not in states and "torch_cuda" not in states:
        return restored, failures
    try:
        import torch  # type: ignore
    except ImportError:
        failures.extend(
            label
            for key, label in (("torch_cpu", "Torch CPU RNG"), ("torch_cuda", "Torch CUDA RNG"))
            if key in states
        )
        return restored, failures

    if "torch_cpu" in states:
        try:
            torch.set_rng_state(_torch_state_tensor(torch, states["torch_cpu"]))
            restored.append("Torch CPU RNG")
        except Exception as exc:
            failures.append(f"Torch CPU RNG: {type(exc).__name__}: {exc}")
    if "torch_cuda" in states:
        try:
            cuda = getattr(torch, "cuda", None)
            if (
                cuda is None
                or not callable(getattr(cuda, "is_available", None))
                or not cuda.is_available()
            ):
                raise RuntimeError("CUDA is unavailable")
            raw_states = _decode_state(states["torch_cuda"])
            if not isinstance(raw_states, Sequence) or isinstance(raw_states, (str, bytes)):
                raise TypeError("CUDA RNG state must be a sequence of device states")
            cuda.set_rng_state_all([_torch_state_tensor(torch, item) for item in raw_states])
            restored.append("Torch CUDA RNG")
        except Exception as exc:
            failures.append(f"Torch CUDA RNG: {type(exc).__name__}: {exc}")
    return restored, failures


class TorchStateAdapter:
    """Optional torch adapter exposed without importing torch at module load."""

    def __init__(self, torch_module: Any | None = None) -> None:
        if torch_module is None:
            try:
                import torch as torch_module  # type: ignore
            except ImportError as exc:
                raise RuntimeError("PyTorch is not installed") from exc
        self.torch = torch_module

    def snapshot(self) -> dict[str, Any]:
        result = {"torch_cpu": self.torch.get_rng_state()}
        if getattr(self.torch, "cuda", None) is not None and self.torch.cuda.is_available():
            result["torch_cuda"] = self.torch.cuda.get_rng_state_all()
        return result

    def restore(self, state: Mapping[str, Any]) -> None:
        if "torch_cpu" in state:
            self.torch.set_rng_state(_torch_state_tensor(self.torch, state["torch_cpu"]))
        cuda_state = state.get("torch_cuda")
        cuda = getattr(self.torch, "cuda", None)
        if cuda_state is not None and cuda is not None and cuda.is_available():
            raw_states = _decode_state(cuda_state)
            if isinstance(raw_states, Sequence) and not isinstance(raw_states, (str, bytes)):
                raw_states = [_torch_state_tensor(self.torch, item) for item in raw_states]
            cuda.set_rng_state_all(raw_states)

    def seed(self, seed: int) -> None:
        self.torch.manual_seed(seed)
        cuda = getattr(self.torch, "cuda", None)
        if cuda is not None and hasattr(cuda, "manual_seed_all"):
            cuda.manual_seed_all(seed)


@dataclass
class ReplayIncident:
    input: Any = None
    state: Any = None
    seed: int | None = None
    expected: FailureSignature | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


IncidentInput = ReplayIncident
Incident = ReplayIncident


@dataclass(frozen=True)
class _ReplayEvidence:
    input: Any
    has_input: bool
    state: Mapping[str, Any]
    seed: int | None
    expected: FailureSignature | None
    source: str
    capture_errors: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_id: str | None = None
    checkpoint_step: int | float | None = None
    requested_step: int | float | None = None
    executed_step: int | float | None = None
    limitations: tuple[str, ...] = ()
    restore_order: tuple[str, ...] = ()
    execution_steps: tuple[tuple[int | float, Any], ...] = ()
    required_state: tuple[str, ...] = ()
    supported_state: Mapping[str, Any] = field(default_factory=dict)
    determinism: str | None = None
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_context: Mapping[str, Any] = field(default_factory=dict)


_RNG_KEYS = frozenset({"python", "numpy", "torch_cpu", "torch_cuda"})


def _decode_named_state(
    capsule: RunCapsule,
    values: Any,
    *,
    limitations: list[str],
    label: str,
) -> dict[str, Any]:
    """Decode named state without turning one incompatible component into bad evidence."""
    decoded: dict[str, Any] = {}
    if isinstance(values, Mapping):
        items = values.items()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        items = (
            (item.get("name", f"state-{index}"), item)
            for index, item in enumerate(values)
            if isinstance(item, Mapping)
        )
    else:
        return decoded
    for name, value in items:
        name = str(name)
        try:
            decoded[name] = _snapshot_value(capsule, value)
        except Exception as exc:
            limitations.append(
                f"state {name!r} from {label} is incompatible: {type(exc).__name__}: {exc}"
            )
    return decoded


def _finite_step(value: Any) -> int | float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise TypeError("replay step must be a finite number or None")
    return value


def _steps_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return float(left) == float(right)


def _checkpoint_is_before_step(item: Mapping[str, Any]) -> bool:
    if item.get("before_step") is False:
        return False
    metadata = item.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("before_step") is False:
        return False
    return True


def _integerish(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value) == int(value)
    )


def _required_execution_steps(start: int | float, target: int | float) -> list[int | float]:
    if _integerish(start) and _integerish(target):
        begin, end = int(start), int(target)
        if end < begin:
            return [target]
        return list(range(begin, end + 1))
    return [start, target] if float(start) != float(target) else [target]


def _decode_step_input(capsule: RunCapsule, encoded: Any, *, plan_input: Any) -> Any:
    if encoded is plan_input:
        return _snapshot_value(capsule, encoded)
    return _artifact_value(capsule, encoded)


def _application_state_names(state: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(name) for name in state if name not in _RNG_KEYS)


def _state_restoration_verified(
    required: Sequence[str],
    restored: Sequence[str],
    omitted: Sequence[str],
    failed: Sequence[str],
) -> bool:
    restored_set = set(restored)
    return not omitted and not failed and all(name in restored_set for name in required)


def _report_metadata(
    evidence: _ReplayEvidence, *, input_digest: str | None = None, **extra: Any
) -> dict[str, Any]:
    payload = {
        "source": evidence.source,
        "checkpoint_id": evidence.checkpoint_id,
        "checkpoint_step": evidence.checkpoint_step,
        "requested_step": evidence.requested_step,
        "executed_step": evidence.executed_step,
        "limitations": list(evidence.limitations),
        "determinism": evidence.determinism,
        "supported_state": dict(evidence.supported_state),
        "checkpoint_context": dict(evidence.checkpoint_context),
        "executed_input_digest": input_digest,
        "planned_steps": [step for step, _value in evidence.execution_steps],
    }
    payload.update(extra)
    executed_steps = payload.get("executed_steps")
    if isinstance(executed_steps, Sequence) and executed_steps:
        payload["executed_step"] = executed_steps[-1]
    return payload


def _failure(value: Any) -> FailureSignature | None:
    if value is None or isinstance(value, FailureSignature):
        return value
    if isinstance(value, Mapping):
        return FailureSignature.from_dict(value)
    raise TypeError("expected failure must be a FailureSignature or mapping")


def _artifact_value(capsule: RunCapsule, value: Any) -> Any:
    """Decode replay markers recursively and verify embedded payloads."""

    def load_payload(digest: str) -> bytes:
        return capsule.artifact_payload(digest)

    return decode_state_tree(value, load_payload)


def _rng_mapping(state: RNGState | None) -> Mapping[str, Any]:
    if state is None:
        return {}
    result = {**state.frameworks}
    if state.python is not None:
        result["python"] = state.python
    if state.numpy is not None:
        result["numpy"] = state.numpy
    return result


def _snapshot_value(capsule: RunCapsule, value: Any) -> Any:
    if hasattr(value, "value") and hasattr(value, "codec"):
        value = getattr(value, "value")
    elif isinstance(value, Mapping) and value.get("type") == "state_snapshot":
        # ReplayPlan serializes snapshots as records.  Most captures put the
        # artifact marker in ``value``; reconstruct the marker from the
        # separate ArtifactRef as a compatibility path for producers that
        # stored only the reference.
        marker = value.get("value")
        if marker is None:
            artifact = value.get("artifact")
            digest = artifact.get("sha256") if isinstance(artifact, Mapping) else None
            if digest:
                marker = {
                    "artifact_sha256": digest,
                    "codec": value.get("codec", "bytes"),
                    **{
                        key: value[key]
                        for key in ("dtype", "shape", "device")
                        if value.get(key) is not None
                    },
                }
        value = marker
    elif isinstance(value, Mapping) and "value" in value and "codec" in value:
        value = value.get("value")
    return _artifact_value(capsule, value)


def _capsule_evidence(
    capsule: RunCapsule,
    expected: FailureSignature | None,
    requested_step: int | float | None = None,
) -> _ReplayEvidence:
    requested = _finite_step(requested_step)
    replay = capsule.evidence.get("replay", {})
    replay = replay if isinstance(replay, Mapping) else {}
    plan = capsule.run.replay_plan
    plan_input = getattr(plan, "input", None) if plan is not None else None
    named = replay.get("state", {})
    if not isinstance(named, Mapping) and plan is not None:
        named = {
            getattr(snapshot, "name", f"state-{index}"): snapshot
            for index, snapshot in enumerate(getattr(plan, "state", ()))
        }
    named = named if isinstance(named, Mapping) else {}
    limitations = list(getattr(plan, "limitations", ())) if plan is not None else []
    raw_limitations = replay.get("limitations", ())
    if isinstance(raw_limitations, Sequence) and not isinstance(raw_limitations, (str, bytes)):
        limitations.extend(str(item) for item in raw_limitations)
    state: dict[str, Any] = dict(_rng_mapping(capsule.run.rng_state))
    captured_state_names = [str(name) for name in named]
    state.update(_decode_named_state(capsule, named, limitations=limitations, label="replay state"))
    raw_checkpoints: list[Any] = []
    candidate_checkpoints = replay.get("checkpoints", ())
    if isinstance(candidate_checkpoints, Sequence) and not isinstance(
        candidate_checkpoints, (str, bytes)
    ):
        raw_checkpoints.extend(candidate_checkpoints)
    if not raw_checkpoints and plan is not None:
        raw_checkpoints.extend(getattr(plan, "checkpoints", ()))
    incident_step = _finite_step(replay.get("step"))
    selected_checkpoint: Any = None
    normalized_checkpoints: list[Mapping[str, Any]] = []
    for item in raw_checkpoints:
        if hasattr(item, "to_dict"):
            item = item.to_dict()
        if isinstance(item, Mapping):
            normalized_checkpoints.append(item)
    inputs_by_step: dict[float, Any] = {}
    for item in normalized_checkpoints:
        step = item.get("step")
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, float))
            or not math.isfinite(float(step))
        ):
            continue
        batch = item.get("batch")
        if batch is None:
            batch = item.get("historical_batch")
        if batch is None:
            continue
        key = float(step)
        previous = inputs_by_step.get(key)
        if previous is not None and previous != batch:
            raise ValueError(f"ambiguous replay input for step {step}")
        inputs_by_step[key] = batch
    history = replay.get("input_history", ())
    if isinstance(history, Sequence) and not isinstance(history, (str, bytes)):
        for item in history:
            if not isinstance(item, Mapping):
                continue
            step = item.get("step")
            batch = item.get("batch", item.get("input"))
            if (
                batch is None
                or isinstance(step, bool)
                or not isinstance(step, (int, float))
                or not math.isfinite(float(step))
            ):
                continue
            key = float(step)
            if key not in inputs_by_step:
                inputs_by_step[key] = batch
    override_encoded = replay.get("execution_input") if "execution_input" in replay else None
    override_step = _finite_step(replay.get("execution_input_step"))
    if override_encoded is not None and override_step is None:
        override_step = incident_step
    selection_step = requested if requested is not None else incident_step
    if normalized_checkpoints:
        eligible = [
            (index, item)
            for index, item in enumerate(normalized_checkpoints)
            if isinstance(item.get("step"), (int, float))
            and not isinstance(item.get("step"), bool)
            and math.isfinite(float(item.get("step")))
            and (selection_step is None or float(item.get("step")) <= float(selection_step))
        ]
        if selection_step is not None and not eligible:
            raise CheckpointUnavailable(selection_step)
        pre_step = [
            pair
            for pair in eligible
            if pair[1].get("before_step") is not False
            and not (
                isinstance(pair[1].get("metadata"), Mapping)
                and pair[1]["metadata"].get("before_step") is False
            )
        ]
        chosen = pre_step or eligible
        selected_checkpoint = (
            max(chosen, key=lambda pair: (float(pair[1]["step"]), pair[0]))[1] if chosen else None
        )
        if selected_checkpoint is not None:
            checkpoint_state = selected_checkpoint.get("state", {})
            if isinstance(checkpoint_state, Mapping):
                captured_state_names.extend(str(name) for name in checkpoint_state)
            elif isinstance(checkpoint_state, Sequence) and not isinstance(
                checkpoint_state, (str, bytes)
            ):
                captured_state_names.extend(
                    str(item.get("name", f"state-{index}"))
                    for index, item in enumerate(checkpoint_state)
                    if isinstance(item, Mapping)
                )
            state.update(
                _decode_named_state(
                    capsule,
                    checkpoint_state,
                    limitations=limitations,
                    label=f"checkpoint {selected_checkpoint.get('step')}",
                )
            )
            checkpoint_rng = selected_checkpoint.get("rng_state")
            if checkpoint_rng is not None:
                if isinstance(checkpoint_rng, RNGState):
                    state.update(_rng_mapping(checkpoint_rng))
                elif isinstance(checkpoint_rng, Mapping):
                    try:
                        state.update(_rng_mapping(RNGState.from_dict(checkpoint_rng)))
                    except (TypeError, ValueError):
                        limitations.append("checkpoint RNG state could not be decoded")
    if incident_step is not None and "input" in replay:
        key = float(incident_step)
        if key not in inputs_by_step:
            inputs_by_step[key] = replay.get("input")
    checkpoint_step = (
        selected_checkpoint.get("step") if isinstance(selected_checkpoint, Mapping) else None
    )
    checkpoint_context = {}
    if isinstance(selected_checkpoint, Mapping):
        for key in ("epoch", "sampler_position", "sample_ids"):
            if selected_checkpoint.get(key) is not None:
                checkpoint_context[key] = selected_checkpoint[key]
        selected_metadata = selected_checkpoint.get("metadata", {})
        if isinstance(selected_metadata, Mapping):
            checkpoint_context.update(
                {
                    str(key): value
                    for key, value in selected_metadata.items()
                    if key in {"epoch", "sampler_position", "sample_ids"} and value is not None
                }
            )
    execute_step = requested
    if execute_step is None:
        execute_step = override_step if override_encoded is not None else incident_step
    if execute_step is None and checkpoint_step is not None:
        execute_step = checkpoint_step
    if requested is None:
        use_override = override_encoded is not None
    else:
        use_override = override_encoded is not None and _steps_equal(requested, override_step)
    encoded_input: Any = None
    has_input = False
    executed_step = execute_step
    if use_override:
        encoded_input = override_encoded
        has_input = True
        executed_step = override_step if override_step is not None else execute_step
    elif executed_step is not None and float(executed_step) in inputs_by_step:
        encoded_input = inputs_by_step[float(executed_step)]
        has_input = True
    elif requested is None and "input" in replay:
        encoded_input = replay.get("input")
        has_input = True
        executed_step = incident_step
    elif requested is None and plan_input is not None:
        encoded_input = plan_input
        has_input = True
        executed_step = incident_step
    elif requested is not None:
        raise IncompleteReplay(requested, restored_step=checkpoint_step)
    input_value = None
    if has_input:
        if encoded_input is plan_input:
            input_value = _snapshot_value(capsule, encoded_input)
        else:
            input_value = _artifact_value(capsule, encoded_input)
    seed = replay.get("seed")
    if seed is None and isinstance(getattr(plan, "metadata", None), Mapping):
        seed = plan.metadata.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise TypeError("replay seed must be an integer")
    checkpoint_metadata = (
        selected_checkpoint.get("metadata", {}) if isinstance(selected_checkpoint, Mapping) else {}
    )
    checkpoint_errors: dict[str, Any] = {}
    if isinstance(selected_checkpoint, Mapping):
        raw_errors = selected_checkpoint.get("state_capture_errors")
        if isinstance(raw_errors, Mapping):
            checkpoint_errors.update(raw_errors)
    if isinstance(checkpoint_metadata, Mapping):
        raw_errors = checkpoint_metadata.get("state_capture_errors")
        if isinstance(raw_errors, Mapping):
            checkpoint_errors.update(raw_errors)
    global_checkpoint_errors = replay.get("checkpoint_capture_errors", {})
    if isinstance(global_checkpoint_errors, Mapping):
        checkpoint_errors.update(global_checkpoint_errors)
    if isinstance(selected_checkpoint, Mapping):
        if selected_checkpoint.get("before_step") is False or (
            isinstance(checkpoint_metadata, Mapping)
            and checkpoint_metadata.get("before_step") is False
        ):
            limitations.append("selected checkpoint was captured after the step")
        raw_limitations = selected_checkpoint.get("limitations")
        if isinstance(raw_limitations, Sequence) and not isinstance(raw_limitations, (str, bytes)):
            limitations.extend(str(item) for item in raw_limitations)
    execution_steps: tuple[tuple[int | float, Any], ...] = ()
    if execute_step is not None:
        step_encoded: dict[float, Any] = dict(inputs_by_step)
        if has_input:
            step_encoded[float(execute_step)] = encoded_input
        if checkpoint_step is not None:
            before_step = (
                _checkpoint_is_before_step(selected_checkpoint)
                if isinstance(selected_checkpoint, Mapping)
                else True
            )
            start = (
                checkpoint_step
                if before_step
                else (int(checkpoint_step) + 1 if _integerish(checkpoint_step) else checkpoint_step)
            )
            needed = _required_execution_steps(start, execute_step)
        else:
            needed = [execute_step]
        missing = [step for step in needed if float(step) not in step_encoded]
        if missing:
            raise IncompleteReplay(
                execute_step,
                restored_step=checkpoint_step,
                message=(
                    f"replay from checkpoint {checkpoint_step} to step {execute_step} is missing "
                    f"intervening input(s) at {missing}; refusing to skip ahead"
                ),
            )
        decoded_steps = []
        for step in needed:
            encoded = step_encoded[float(step)]
            decoded_steps.append(
                (step, _decode_step_input(capsule, encoded, plan_input=plan_input))
            )
        execution_steps = tuple(decoded_steps)
        if decoded_steps:
            input_value = decoded_steps[-1][1]
            has_input = True
            executed_step = decoded_steps[-1][0]
    required_state = tuple(
        str(name)
        for name in (
            tuple(getattr(plan, "restore_order", ()))
            if plan is not None and getattr(plan, "restore_order", ())
            else tuple(dict.fromkeys(captured_state_names or (str(key) for key in state)))
        )
        if name not in _RNG_KEYS
    )
    return _ReplayEvidence(
        input=input_value,
        has_input=has_input,
        state=state,
        seed=seed,
        expected=expected
        or capsule.run.failure_signature
        or (getattr(plan, "expected_failure", None) if plan is not None else None),
        source="capsule",
        capture_errors=(
            {
                **(
                    replay.get("state_capture_errors", {})
                    if isinstance(replay.get("state_capture_errors", {}), Mapping)
                    else {"state": replay.get("state_capture_errors")}
                ),
                **checkpoint_errors,
            }
        ),
        checkpoint_id=(
            str(selected_checkpoint.get("checkpoint_id"))
            if isinstance(selected_checkpoint, Mapping)
            and selected_checkpoint.get("checkpoint_id") is not None
            else None
        ),
        checkpoint_step=checkpoint_step,
        requested_step=requested,
        executed_step=executed_step,
        limitations=limitations,
        restore_order=(
            tuple(getattr(plan, "restore_order", ()))
            if plan is not None
            else tuple(str(name) for name in state if name not in _RNG_KEYS)
        ),
        execution_steps=execution_steps,
        required_state=required_state,
        supported_state=(
            replay.get("supported_state", {})
            if isinstance(replay.get("supported_state", {}), Mapping)
            else {}
        ),
        determinism=(
            str(replay.get("determinism"))
            if replay.get("determinism") is not None
            else (getattr(plan, "determinism", None) if plan is not None else None)
        ),
        runtime_metadata=(
            replay.get("metadata", {}) if isinstance(replay.get("metadata", {}), Mapping) else {}
        ),
        checkpoint_context=checkpoint_context,
    )


def _extract_evidence(
    incident: ReplayIncident | CoreIncident | RunCapsule | Mapping[str, Any],
    expected: FailureSignature | None,
    requested_step: int | float | None = None,
) -> _ReplayEvidence:
    if isinstance(incident, RunCapsule):
        return _capsule_evidence(incident, expected, requested_step)
    if isinstance(incident, CoreIncident):
        replay = incident.metadata.get("replay", {})
        replay = replay if isinstance(replay, Mapping) else {}
        state = replay.get("state", {})
        state = state if isinstance(state, Mapping) else {}
        metadata = replay.get("metadata", {})
        return _ReplayEvidence(
            replay.get("input"),
            "input" in replay,
            state,
            replay.get("seed"),
            expected or _failure(replay.get("failure")),
            "incident",
            required_state=_application_state_names(state),
            supported_state=(
                replay.get("supported_state", {})
                if isinstance(replay.get("supported_state", {}), Mapping)
                else {}
            ),
            determinism=(
                str(replay["determinism"]) if replay.get("determinism") is not None else None
            ),
            runtime_metadata=metadata if isinstance(metadata, Mapping) else {},
        )
    if isinstance(incident, ReplayIncident):
        state = incident.state if isinstance(incident.state, Mapping) else {}
        return _ReplayEvidence(
            incident.input,
            incident.input is not None,
            state,
            incident.seed,
            expected or incident.expected,
            "replay_incident",
            required_state=_application_state_names(state),
            runtime_metadata=incident.metadata,
        )
    state = incident.get("state", incident.get("randomness", {}))
    state = state if isinstance(state, Mapping) else {}
    metadata = incident.get("metadata", {})
    return _ReplayEvidence(
        incident.get("input"),
        "input" in incident,
        state,
        incident.get("seed"),
        expected or _failure(incident.get("failure", incident.get("expected"))),
        "mapping",
        required_state=_application_state_names(state),
        supported_state=(
            incident.get("supported_state", {})
            if isinstance(incident.get("supported_state", {}), Mapping)
            else {}
        ),
        determinism=(
            str(incident["determinism"]) if incident.get("determinism") is not None else None
        ),
        runtime_metadata=metadata if isinstance(metadata, Mapping) else {},
    )


def _target(runner: Callable[..., Any]) -> Callable[..., Any]:
    return runner.run if hasattr(runner, "run") else runner


def _invoke_runner(
    runner: Callable[..., Any], argument: Any, has_argument: bool, seed: int | None
) -> Any:
    target = _target(runner)
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return target(argument)

    positional = [
        item
        for item in parameters.values()
        if item.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    variadic = any(item.kind is inspect.Parameter.VAR_POSITIONAL for item in parameters.values())
    if seed is not None and "seed" in parameters and positional:
        return target(argument, seed=seed)
    if seed is not None and (variadic or len(positional) >= 2):
        return target(argument, seed)
    if has_argument and (variadic or positional):
        return target(argument)
    if seed is not None and "seed" in parameters:
        return target(seed=seed)
    if not positional and not variadic:
        return target()
    return target(argument)


def _restore_component(
    restorer: Any, state: Any, *, name: str | None = None
) -> tuple[bool, str | None]:
    """Restore a component using the common hook shapes used by ML objects."""
    callback = getattr(restorer, "restore", None)
    if callable(callback):
        callback(state)
        return True, None
    callback = getattr(restorer, "load_state_dict", None)
    if callable(callback):
        result = callback(state)
        missing = getattr(result, "missing_keys", ()) if result is not None else ()
        unexpected = getattr(result, "unexpected_keys", ()) if result is not None else ()
        missing = tuple(missing or ())
        unexpected = tuple(unexpected or ())
        if missing or unexpected:
            return False, (
                "incompatible state "
                f"(missing_keys={list(missing)}, unexpected_keys={list(unexpected)})"
            )
        return True, None
    callback = getattr(restorer, "set_state", None)
    if callable(callback):
        callback(state)
        return True, None
    if name == "dataloader" and isinstance(state, Mapping):
        restored_nested = False
        sampler = getattr(restorer, "sampler", None)
        sampler_state = state.get("sampler")
        if sampler is not None and sampler_state is not None:
            restored_ok, issue = _restore_component(sampler, sampler_state, name="sampler")
            if not restored_ok:
                return False, issue
            restored_nested = True
        generator = getattr(restorer, "generator", None)
        generator_state = state.get("generator")
        if generator is not None and generator_state is not None:
            setter = getattr(generator, "set_state", None)
            if not callable(setter):
                return False, "dataloader generator has no set_state() hook"
            setter(generator_state)
            restored_nested = True
        if restored_nested:
            return True, None
    if callable(restorer):
        restorer(state)
        return True, None
    raise TypeError("component has no restore(), load_state_dict(), set_state(), or callable hook")


def _restore_checkpoint_context(name: str, restorer: Any, context: Mapping[str, Any]) -> str | None:
    """Restore sampler epoch/position when a provider exposes those hooks."""
    if name not in {"sampler", "batch_sampler", "dataloader"} or not context:
        return None
    sampler = getattr(restorer, "sampler", None) if name == "dataloader" else restorer
    if sampler is None:
        return None
    epoch = context.get("epoch")
    if epoch is not None:
        setter = getattr(sampler, "set_epoch", None)
        if callable(setter):
            setter(epoch)
        elif not hasattr(sampler, "epoch"):
            return "sampler does not expose set_epoch() or an epoch attribute"
    position = context.get("sampler_position")
    if position is not None:
        setter = getattr(sampler, "set_position", None)
        if callable(setter):
            setter(position)
        elif hasattr(sampler, "position"):
            try:
                sampler.position = position
            except Exception as exc:
                return f"sampler position could not be restored: {type(exc).__name__}: {exc}"
        else:
            return "sampler does not expose set_position() or a position attribute"
    return None


def _failure_from_result(result: Any) -> FailureSignature | None:
    if isinstance(result, FailureSignature):
        return result
    candidate = getattr(result, "failure", None)
    if candidate is not None:
        try:
            return _failure(candidate)
        except (TypeError, ValueError):
            return None
    if isinstance(result, Mapping):
        for key in ("failure", "failure_signature", "signature"):
            candidate = result.get(key)
            if candidate is None:
                continue
            try:
                return _failure(candidate)
            except (TypeError, ValueError):
                return None
        status = str(result.get("status", "")).casefold()
        if status in {"timeout", "timed_out", "hang", "out_of_memory", "oom"}:
            kind = (
                "out_of_memory"
                if status == "oom"
                else "timeout"
                if status in {"timeout", "timed_out"}
                else status
            )
            default_message = "out of memory" if kind == "out_of_memory" else "operation timed out"
            return FailureSignature.structured(
                kind,
                message=str(result.get("message") or result.get("error") or default_message),
                details=(
                    result.get("details", {})
                    if isinstance(result.get("details", {}), Mapping)
                    else {}
                ),
            )
    return None


class _ReplayTimeout(TimeoutError):
    pass


def _invoke_with_timeout(
    runner: Callable[..., Any],
    argument: Any,
    has_argument: bool,
    seed: int | None,
    timeout: float | None,
) -> Any:
    if timeout is None:
        return _invoke_runner(runner, argument, has_argument, seed)
    import threading

    completed = threading.Event()
    outcome: list[Any] = []
    error: list[BaseException] = []

    def invoke() -> None:
        try:
            outcome.append(_invoke_runner(runner, argument, has_argument, seed))
        except BaseException as exc:
            error.append(exc)
        finally:
            completed.set()

    thread = threading.Thread(target=invoke, name="mlforensics-replay", daemon=True)
    thread.start()
    if not completed.wait(timeout):
        raise _ReplayTimeout(f"replay timed out after {timeout:g}s")
    if error:
        raise error[0]
    return outcome[0] if outcome else None


@contextmanager
def _replay_runtime_context(evidence: _ReplayEvidence, limitations: list[str]) -> Any:
    """Apply captured deterministic settings without importing torch eagerly."""
    metadata = dict(evidence.runtime_metadata)
    supported = evidence.supported_state
    autocast_value = metadata.get("autocast", supported.get("autocast"))
    deterministic = metadata.get("deterministic_algorithms")
    if deterministic is None:
        deterministic = metadata.get("torch_deterministic")
    cudnn_deterministic = metadata.get("cudnn_deterministic")
    cudnn_benchmark = metadata.get("cudnn_benchmark")
    needs_torch = (
        bool(autocast_value)
        or deterministic is not None
        or cudnn_deterministic is not None
        or cudnn_benchmark is not None
    )
    if not needs_torch:
        yield
        return
    try:
        import torch  # type: ignore
    except ImportError:
        limitations.append(
            "captured torch runtime settings could not be applied: PyTorch is unavailable"
        )
        yield
        return

    old_deterministic = None
    old_cudnn = getattr(torch, "backends", None)
    old_cudnn_deterministic = None
    old_cudnn_benchmark = None
    autocast = None
    try:
        if deterministic is not None and callable(
            getattr(torch, "are_deterministic_algorithms_enabled", None)
        ):
            old_deterministic = torch.are_deterministic_algorithms_enabled()
            torch.use_deterministic_algorithms(bool(deterministic))
        cudnn = getattr(old_cudnn, "cudnn", None)
        if cudnn is not None:
            if cudnn_deterministic is not None:
                old_cudnn_deterministic = cudnn.deterministic
                cudnn.deterministic = bool(cudnn_deterministic)
            if cudnn_benchmark is not None:
                old_cudnn_benchmark = cudnn.benchmark
                cudnn.benchmark = bool(cudnn_benchmark)
        autocast_enabled = (
            bool(autocast_value)
            if not isinstance(autocast_value, Mapping)
            else bool(autocast_value.get("enabled", False))
        )
        if autocast_enabled:
            config = autocast_value if isinstance(autocast_value, Mapping) else {}
            cuda = getattr(torch, "cuda", None)
            cuda_available = callable(getattr(cuda, "is_available", None)) and cuda.is_available()
            device_type = str(config.get("device_type", "cuda" if cuda_available else "cpu"))
            options: dict[str, Any] = {}
            if config.get("dtype") is not None:
                dtype = config["dtype"]
                options["dtype"] = getattr(torch, str(dtype).split(".")[-1], dtype)
            if config.get("cache_enabled") is not None:
                options["cache_enabled"] = bool(config["cache_enabled"])
            factory = getattr(torch, "autocast", None)
            if not callable(factory):
                raise RuntimeError("torch.autocast is unavailable")
            autocast = factory(device_type, **options)
            autocast.__enter__()
    except Exception as exc:
        limitations.append(
            f"captured torch runtime settings could not be applied: {type(exc).__name__}: {exc}"
        )
    try:
        yield
    finally:
        if "autocast" in locals() and autocast is not None:
            try:
                autocast.__exit__(None, None, None)
            except Exception:
                pass
        cudnn = getattr(old_cudnn, "cudnn", None)
        if cudnn is not None:
            if old_cudnn_deterministic is not None:
                cudnn.deterministic = old_cudnn_deterministic
            if old_cudnn_benchmark is not None:
                cudnn.benchmark = old_cudnn_benchmark
        if old_deterministic is not None:
            try:
                torch.use_deterministic_algorithms(old_deterministic)
            except Exception:
                pass


class ReplayEngine:
    """Restore recorded state and verify that the recorded failure recurs."""

    def __init__(
        self,
        *,
        state_restorers: Mapping[str, StateRestorer] | None = None,
        strict_state: bool = True,
        reseed: bool = False,
        timeout: float | None = None,
    ) -> None:
        self.state_restorers = dict(state_restorers or {})
        self.strict_state = strict_state
        self.reseed = reseed
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number or None")
        self.timeout = timeout

    def replay(
        self,
        incident: Incident | CoreIncident | RunCapsule | Mapping[str, Any],
        runner: Callable[..., Any],
        *,
        expected: FailureSignature | None = None,
        predicate: Callable[[Any], bool] | None = None,
        step: int | float | None = None,
        timeout: float | None = None,
    ) -> ReplayResult:
        effective_timeout = self.timeout if timeout is None else timeout
        if effective_timeout is not None and (
            isinstance(effective_timeout, bool)
            or not isinstance(effective_timeout, (int, float))
            or not math.isfinite(float(effective_timeout))
            or effective_timeout <= 0
        ):
            return ReplayResult(
                False,
                error="timeout must be a positive finite number or None",
                metadata={"verified": False, "status": "unavailable"},
            )
        try:
            evidence = _extract_evidence(incident, expected, step)
        except CheckpointUnavailable as exc:
            return ReplayResult(
                False,
                error=str(exc),
                metadata={
                    "verified": False,
                    "status": "unavailable-checkpoint",
                    "replay_status": "unavailable",
                    "reason": "checkpoint unavailable",
                    "requested_step": exc.step,
                },
            )
        except IncompleteReplay as exc:
            return ReplayResult(
                False,
                error=str(exc),
                metadata={
                    "verified": False,
                    "status": "incomplete-replay",
                    "replay_status": "incomplete",
                    "requested_step": exc.step,
                    "checkpoint_step": exc.restored_step,
                    "executed_step": None,
                    "executed_input_digest": None,
                },
            )
        except Exception as exc:
            return ReplayResult(
                False,
                error=f"invalid replay evidence: {exc}",
                metadata={"verified": False, "status": "unavailable"},
            )
        if evidence.capture_errors:
            return ReplayResult(
                False,
                error="capture did not produce complete replay state",
                metadata=_report_metadata(
                    evidence,
                    verified=False,
                    status="incomplete",
                    replay_status="incomplete",
                    state_capture_errors=dict(evidence.capture_errors),
                ),
            )
        restored: list[str] = []
        omitted: list[str] = []
        failed: list[str] = []
        limitations = list(evidence.limitations)
        required_state = evidence.required_state or _application_state_names(evidence.state)
        ordered_names = list(evidence.restore_order)
        ordered_names.extend(name for name in evidence.state if name not in ordered_names)

        def report_evidence() -> _ReplayEvidence:
            return replace(evidence, limitations=tuple(dict.fromkeys(limitations)))

        def restoration_fields(*, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
            payload = {
                "required_state": list(required_state),
                "restored_state": list(restored),
                "omitted_state": list(omitted),
                "failed_state": list(failed),
                "state_restoration_verified": _state_restoration_verified(
                    required_state, restored, omitted, failed
                ),
            }
            if extra:
                payload.update(extra)
            return payload

        for name in ordered_names:
            if name not in evidence.state:
                if name in required_state and name not in omitted:
                    omitted.append(name)
                    limitations.append(f"state {name!r} was required but omitted from evidence")
                continue
            state = evidence.state[name]
            if name in _RNG_KEYS:
                continue
            restorer = self.state_restorers.get(name)
            if restorer is None:
                omitted.append(name)
                limitations.append(
                    f"state {name!r} was omitted: no compatible restorer was supplied"
                )
                continue
            try:
                has_restore_hook = any(
                    callable(getattr(restorer, hook, None))
                    for hook in ("restore", "load_state_dict", "set_state")
                ) or callable(restorer)
                context_only = (
                    name in {"sampler", "batch_sampler", "dataloader"}
                    and bool(evidence.checkpoint_context)
                    and not has_restore_hook
                )
                restored_ok, issue = (
                    (True, None) if context_only else _restore_component(restorer, state, name=name)
                )
                if not restored_ok:
                    failed.append(name)
                    limitations.append(f"state {name!r} is incompatible: {issue}")
                    return ReplayResult(
                        False,
                        restored=restored,
                        error=f"failed to restore state {name!r}: {issue}",
                        metadata=_report_metadata(
                            report_evidence(),
                            verified=False,
                            status="incomplete",
                            replay_status="incomplete",
                            **restoration_fields(),
                        ),
                    )
                restored.append(name)
                context_issue = _restore_checkpoint_context(
                    name, restorer, evidence.checkpoint_context
                )
                if context_issue:
                    # Sampler epoch/position metadata is supplementary to the
                    # component state.  A callback-only restorer can restore
                    # its state dictionary but cannot expose an attribute or
                    # setter for every piece of loader context; retain that
                    # limitation without discarding the successful state
                    # restore or preventing the replay from executing.
                    limitations.append(f"state {name!r} is incompatible: {context_issue}")
            except Exception as exc:
                failed.append(name)
                limitations.append(f"state {name!r} is incompatible: {type(exc).__name__}: {exc}")
                return ReplayResult(
                    False,
                    restored=restored,
                    error=f"failed to restore state {name!r}: {type(exc).__name__}: {exc}",
                    metadata=_report_metadata(
                        report_evidence(),
                        verified=False,
                        status="incomplete",
                        replay_status="incomplete",
                        **restoration_fields(),
                    ),
                )
        if omitted and self.strict_state:
            return ReplayResult(
                False,
                restored=restored,
                error="missing state restorer(s): " + ", ".join(sorted(omitted)),
                metadata=_report_metadata(
                    report_evidence(),
                    verified=False,
                    status="incomplete",
                    replay_status="incomplete",
                    missing_state=list(omitted),
                    **restoration_fields(),
                ),
            )
        rng_state = {name: value for name, value in evidence.state.items() if name in _RNG_KEYS}
        if rng_state:
            restored_rng_values, rng_failures = _restore_rng_state_detailed(rng_state)
            restored.extend(item for item in restored_rng_values if item not in restored)
            restored_rng = {
                "Python RNG"
                if name == "python"
                else "NumPy RNG"
                if name == "numpy"
                else "Torch CPU RNG"
                if name == "torch_cpu"
                else "Torch CUDA RNG"
                for name in rng_state
            }
            missing_rng = sorted(restored_rng.difference(restored))
            if rng_failures:
                limitations.extend(
                    f"RNG state restoration incomplete: {failure}" for failure in rng_failures
                )
            if missing_rng:
                omitted.extend(missing_rng)
                if self.strict_state:
                    return ReplayResult(
                        False,
                        restored=restored,
                        error="failed to restore RNG state: " + ", ".join(missing_rng),
                        metadata=_report_metadata(
                            report_evidence(),
                            verified=False,
                            status="incomplete",
                            replay_status="incomplete",
                            missing_rng_state=missing_rng,
                            **restoration_fields(),
                        ),
                    )
        # A logical seed is only a fallback.  Reseeding after an exact RNG
        # restore would silently destroy the captured random stream.
        restored_rng_labels = {
            "Python RNG",
            "NumPy RNG",
            "Torch CPU RNG",
            "Torch CUDA RNG",
        }
        exact_rng_restored = bool(restored_rng_labels.intersection(restored))
        if evidence.seed is not None and (self.reseed or not exact_rng_restored):
            random.seed(evidence.seed)
            restored.append("Python RNG seed")

        planned = list(evidence.execution_steps)
        if not planned:
            argument = evidence.input if evidence.has_input else incident
            planned = ((evidence.executed_step, argument),)
        target_argument = planned[-1][1]
        input_digest = _value_digest(target_argument) if evidence.has_input else None
        executed_steps: list[int | float | None] = []
        result: Any = None

        def outcome_from_exception(exc: BaseException) -> ReplayResult:
            if isinstance(exc, _ReplayTimeout):
                failure = FailureSignature.structured("timeout", message="operation timed out")
                status = "timeout"
            else:
                failure = exception_signature(exc)
                status = "failure"
            if evidence.expected is not None:
                reproduced = evidence.expected.matches(failure)
                # ``verified`` answers whether the observed outcome was
                # compared with explicit failure evidence.  State fidelity is
                # reported independently as ``state_restoration_verified``;
                # an opted-out/partial replay can therefore verify a matching
                # failure without claiming that every provider was restored.
                verified = True
                error = None if reproduced else "replay raised a different failure"
            else:
                # Historical mapping/ReplayIncident calls are retained, but a
                # capsule cannot claim reproduction without a stored signature.
                reproduced = evidence.source != "capsule"
                verified = False
                error = None if reproduced else "capsule has no expected failure signature"
            return ReplayResult(
                reproduced,
                failure,
                restored=restored,
                error=error,
                metadata=_report_metadata(
                    report_evidence(),
                    input_digest=input_digest,
                    verified=verified,
                    status=(
                        status
                        if restoration_fields()["state_restoration_verified"]
                        else "incomplete-replay"
                    ),
                    replay_status=(
                        status
                        if restoration_fields()["state_restoration_verified"]
                        else "incomplete"
                    ),
                    expected=evidence.expected.to_dict() if evidence.expected else None,
                    executed_steps=executed_steps,
                    **restoration_fields(),
                ),
            )

        try:
            with _replay_runtime_context(evidence, limitations):
                for step, argument in planned:
                    has_argument = evidence.has_input or argument is not None
                    result = _invoke_with_timeout(
                        runner, argument, has_argument, evidence.seed, effective_timeout
                    )
                    executed_steps.append(step)
        except BaseException as exc:
            if len(executed_steps) < len(planned):
                executed_steps.append(planned[len(executed_steps)][0])
            return outcome_from_exception(exc)

        reported_failure = _failure_from_result(result)
        if reported_failure is not None and evidence.expected is not None:
            reproduced = evidence.expected.matches(reported_failure)
            state_verified = bool(restoration_fields()["state_restoration_verified"])
            return ReplayResult(
                reproduced,
                failure=reported_failure,
                result=result,
                restored=restored,
                error=None if reproduced else "replay reported a different failure",
                metadata=_report_metadata(
                    report_evidence(),
                    input_digest=input_digest,
                    verified=True,
                    status="failure" if state_verified else "incomplete-replay",
                    replay_status="failure" if state_verified else "incomplete",
                    expected=evidence.expected.to_dict(),
                    structured_result=True,
                    executed_steps=executed_steps,
                    **restoration_fields(),
                ),
            )
        if evidence.expected is not None:
            state_verified = bool(restoration_fields()["state_restoration_verified"])
            return ReplayResult(
                False,
                result=result,
                restored=restored,
                error="expected failure did not occur",
                metadata=_report_metadata(
                    report_evidence(),
                    input_digest=input_digest,
                    verified=True,
                    status="success" if state_verified else "incomplete-replay",
                    replay_status="success" if state_verified else "incomplete",
                    expected=evidence.expected.to_dict(),
                    executed_steps=executed_steps,
                    **restoration_fields(),
                ),
            )
        try:
            if predicate is not None:
                judged = evaluate_predicate(predicate, result)
                reproduced = judged.preserved and not judged.unresolved
                verified = not judged.unresolved
            elif isinstance(result, Mapping):
                reproduced = bool(result.get("reproduced", result.get("failed", False)))
                verified = "reproduced" in result or "failed" in result
            else:
                reproduced = False
                verified = False
            status = "failure" if reproduced else "success"
            if str(getattr(result, "status", "")).casefold() in {"timeout", "timed_out"}:
                status = "timeout"
            if isinstance(result, Mapping) and str(result.get("status", "")).casefold() in {
                "timeout",
                "timed_out",
            }:
                status = "timeout"
            if not restoration_fields()["state_restoration_verified"]:
                verified = False
                status = "incomplete-replay"
        except Exception as exc:
            return ReplayResult(
                False,
                result=result,
                restored=restored,
                error=f"replay predicate failed: {type(exc).__name__}: {exc}",
                metadata=_report_metadata(
                    report_evidence(),
                    input_digest=input_digest,
                    verified=False,
                    status="incomplete",
                    replay_status="incomplete",
                    executed_steps=executed_steps,
                    **restoration_fields(),
                ),
            )
        return ReplayResult(
            reproduced,
            result=result,
            restored=restored,
            metadata=_report_metadata(
                report_evidence(),
                input_digest=input_digest,
                verified=verified,
                status=status,
                replay_status=status.removesuffix("-replay"),
                expected=None,
                executed_steps=executed_steps,
                **restoration_fields(),
            ),
        )


def replay(
    incident: Incident | CoreIncident | RunCapsule | Mapping[str, Any],
    runner: Callable[..., Any],
    **kwargs: Any,
) -> ReplayResult:
    return ReplayEngine().replay(incident, runner, **kwargs)


def snapshot_state(adapters: Mapping[str, Any]) -> dict[str, Any]:
    """Snapshot named providers using ``snapshot``/``state_dict`` hooks."""
    result: dict[str, Any] = {}
    for name, provider in adapters.items():
        snapshot = getattr(provider, "snapshot", None)
        state_dict = getattr(provider, "state_dict", None)
        if callable(snapshot):
            result[name] = snapshot()
        elif callable(state_dict):
            result[name] = state_dict()
        else:
            result[name] = provider() if callable(provider) else provider
    return result


def replay_incident(
    incident: Any,
    runner: Callable[..., Any],
    *,
    hooks: Any | None = None,
    expected: FailureSignature | None = None,
) -> ReplayResult:
    """Compatibility replay helper using a single aggregate state hook."""
    if isinstance(incident, RunCapsule):
        restorers = {}
        if hooks is not None:
            replay_state = incident.evidence.get("replay", {})
            replay_state = (
                replay_state.get("state", {}) if isinstance(replay_state, Mapping) else {}
            )
            if isinstance(replay_state, Mapping):
                restorers = {str(name): hooks for name in replay_state}
        return ReplayEngine(state_restorers=restorers).replay(incident, runner, expected=expected)

    if isinstance(incident, Mapping):
        input_value = incident.get("input")
        state = incident.get("state", incident.get("randomness"))
        seed = incident.get("seed")
        expected = expected or _failure(incident.get("failure", incident.get("expected")))
    else:
        input_value = getattr(incident, "input", None)
        state = getattr(incident, "state", None)
        seed = getattr(incident, "seed", None)
        expected = expected or getattr(incident, "expected", None)
    restored: list[str] = []
    if state is not None:
        restore = getattr(hooks, "restore", None) if hooks is not None else None
        if not callable(restore):
            return ReplayResult(
                False,
                restored=restored,
                error="state restoration hook is required to replay this incident",
                metadata={"verified": False},
            )
        try:
            restore(state)
        except Exception as exc:
            return ReplayResult(
                False,
                error=f"failed to restore state: {type(exc).__name__}: {exc}",
                metadata={"verified": False},
            )
        restored.append("state")
    if seed is not None:
        seed_hook = getattr(hooks, "seed", None) if hooks is not None else None
        if callable(seed_hook):
            seed_hook(seed)
            restored.append("seed")
        else:
            random.seed(seed)
            restored.append("Python RNG seed")
    try:
        result = _invoke_runner(runner, input_value, True, seed)
    except BaseException as exc:
        failure = exception_signature(exc)
        reproduced = expected.matches(failure) if expected else True
        return ReplayResult(
            reproduced,
            failure,
            restored=restored,
            error=str(exc),
            metadata={"verified": expected is not None},
        )
    if expected is not None:
        return ReplayResult(
            False,
            result=result,
            restored=restored,
            error="expected failure did not occur",
            metadata={"verified": True},
        )
    if isinstance(result, Mapping):
        reproduced = bool(result.get("reproduced", result.get("failed", False)))
        verified = "reproduced" in result or "failed" in result
    else:
        reproduced = bool(result) if isinstance(result, bool) else False
        verified = isinstance(result, bool)
    return ReplayResult(
        reproduced, result=result, restored=restored, metadata={"verified": verified}
    )
