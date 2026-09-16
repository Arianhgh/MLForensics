"""State restoration and evidence-backed incident replay."""

from __future__ import annotations

import inspect
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
    if "torch_cpu" in states:
        try:
            import torch  # type: ignore

            cpu_state = _decode_state(states["torch_cpu"])
            torch.set_rng_state(torch.tensor(cpu_state, dtype=torch.uint8))
            restored.append("Torch CPU RNG")
            if "torch_cuda" in states and torch.cuda.is_available():
                cuda_states = _decode_state(states["torch_cuda"])
                torch.cuda.set_rng_state_all(
                    [torch.tensor(item, dtype=torch.uint8) for item in cuda_states]
                )
                restored.append("Torch CUDA RNG")
        except Exception:
            pass
    return restored


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
        self.torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and self.torch.cuda.is_available():
            self.torch.cuda.set_rng_state_all(state["torch_cuda"])

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


_RNG_KEYS = frozenset({"python", "numpy", "torch_cpu", "torch_cuda"})


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
        "executed_input_digest": input_digest,
    }
    payload.update(extra)
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
    state: dict[str, Any] = dict(_rng_mapping(capsule.run.rng_state))
    state.update({str(name): _snapshot_value(capsule, value) for name, value in named.items()})
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
    limitations = list(getattr(plan, "limitations", ())) if plan is not None else []
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
                state.update(
                    {
                        str(name): _snapshot_value(capsule, value)
                        for name, value in checkpoint_state.items()
                    }
                )
            elif isinstance(checkpoint_state, Sequence) and not isinstance(
                checkpoint_state, (str, bytes)
            ):
                for index, snapshot in enumerate(checkpoint_state):
                    if not isinstance(snapshot, Mapping):
                        continue
                    name = str(snapshot.get("name", f"state-{index}"))
                    state[name] = _snapshot_value(capsule, snapshot)
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
        return _ReplayEvidence(
            replay.get("input"),
            "input" in replay,
            state if isinstance(state, Mapping) else {},
            replay.get("seed"),
            expected or _failure(replay.get("failure")),
            "incident",
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
        )
    state = incident.get("state", incident.get("randomness", {}))
    return _ReplayEvidence(
        incident.get("input"),
        "input" in incident,
        state if isinstance(state, Mapping) else {},
        incident.get("seed"),
        expected or _failure(incident.get("failure", incident.get("expected"))),
        "mapping",
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
    if seed is not None and (variadic or len(positional) >= 2):
        return target(argument, seed)
    if has_argument and (variadic or positional):
        return target(argument)
    if seed is not None and "seed" in parameters:
        return target(seed=seed)
    if not positional and not variadic:
        return target()
    return target(argument)


class ReplayEngine:
    """Restore recorded state and verify that the recorded failure recurs."""

    def __init__(
        self,
        *,
        state_restorers: Mapping[str, StateRestorer] | None = None,
        strict_state: bool = True,
        reseed: bool = False,
    ) -> None:
        self.state_restorers = dict(state_restorers or {})
        self.strict_state = strict_state
        self.reseed = reseed

    def replay(
        self,
        incident: Incident | CoreIncident | RunCapsule | Mapping[str, Any],
        runner: Callable[..., Any],
        *,
        expected: FailureSignature | None = None,
        predicate: Callable[[Any], bool] | None = None,
        step: int | float | None = None,
    ) -> ReplayResult:
        try:
            evidence = _extract_evidence(incident, expected, step)
        except CheckpointUnavailable as exc:
            return ReplayResult(
                False,
                error=str(exc),
                metadata={
                    "verified": False,
                    "status": "unavailable-checkpoint",
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
                    "requested_step": exc.step,
                    "checkpoint_step": exc.restored_step,
                    "executed_step": None,
                    "executed_input_digest": None,
                },
            )
        except Exception as exc:
            return ReplayResult(False, error=f"invalid replay evidence: {exc}")
        if evidence.capture_errors:
            return ReplayResult(
                False,
                error="capture did not produce complete replay state",
                metadata=_report_metadata(
                    evidence,
                    verified=False,
                    state_capture_errors=dict(evidence.capture_errors),
                ),
            )
        restored: list[str] = []
        missing: list[str] = []
        ordered_names = list(evidence.restore_order)
        ordered_names.extend(name for name in evidence.state if name not in ordered_names)
        for name in ordered_names:
            if name not in evidence.state:
                continue
            state = evidence.state[name]
            if name in _RNG_KEYS:
                continue
            restorer = self.state_restorers.get(name)
            if restorer is None:
                missing.append(name)
                continue
            try:
                callback = getattr(restorer, "restore", restorer)
                callback(state)
                restored.append(name)
            except Exception as exc:
                return ReplayResult(
                    False,
                    restored=restored,
                    error=f"failed to restore state {name!r}: {type(exc).__name__}: {exc}",
                    metadata={
                        "verified": False,
                        "source": evidence.source,
                        "checkpoint_id": evidence.checkpoint_id,
                        "checkpoint_step": evidence.checkpoint_step,
                        "limitations": list(evidence.limitations),
                    },
                )
        if missing and self.strict_state:
            return ReplayResult(
                False,
                restored=restored,
                error="missing state restorer(s): " + ", ".join(sorted(missing)),
                metadata={
                    "verified": False,
                    "source": evidence.source,
                    "missing_state": missing,
                    "checkpoint_id": evidence.checkpoint_id,
                    "checkpoint_step": evidence.checkpoint_step,
                    "limitations": list(evidence.limitations),
                },
            )
        rng_state = {name: value for name, value in evidence.state.items() if name in _RNG_KEYS}
        if rng_state:
            restored.extend(restore_rng_state(rng_state))
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
            if missing_rng and self.strict_state:
                return ReplayResult(
                    False,
                    restored=restored,
                    error="failed to restore RNG state: " + ", ".join(missing_rng),
                    metadata={
                        "verified": False,
                        "source": evidence.source,
                        "missing_rng_state": missing_rng,
                        "checkpoint_id": evidence.checkpoint_id,
                        "checkpoint_step": evidence.checkpoint_step,
                        "limitations": list(evidence.limitations),
                    },
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

        argument = evidence.input if evidence.has_input else incident
        input_digest = _value_digest(argument) if evidence.has_input else None
        try:
            result = _invoke_runner(
                runner, argument, evidence.has_input or argument is not None, evidence.seed
            )
        except BaseException as exc:
            failure = exception_signature(exc)
            if evidence.expected is not None:
                reproduced = evidence.expected.matches(failure)
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
                    evidence,
                    input_digest=input_digest,
                    verified=verified,
                    expected=evidence.expected.to_dict() if evidence.expected else None,
                    state_restoration_verified=True,
                ),
            )

        reported_failure: FailureSignature | None = None
        if isinstance(result, FailureSignature):
            reported_failure = result
        elif isinstance(result, Mapping):
            for key in ("failure", "failure_signature", "signature"):
                candidate = result.get(key)
                if candidate is None:
                    continue
                try:
                    reported_failure = _failure(candidate)
                except (TypeError, ValueError):
                    reported_failure = None
                if reported_failure is not None:
                    break
        if reported_failure is not None and evidence.expected is not None:
            reproduced = evidence.expected.matches(reported_failure)
            return ReplayResult(
                reproduced,
                failure=reported_failure,
                result=result,
                restored=restored,
                error=None if reproduced else "replay reported a different failure",
                metadata=_report_metadata(
                    evidence,
                    input_digest=input_digest,
                    verified=True,
                    expected=evidence.expected.to_dict(),
                    structured_result=True,
                    state_restoration_verified=True,
                ),
            )
        if evidence.expected is not None:
            return ReplayResult(
                False,
                result=result,
                restored=restored,
                error="expected failure did not occur",
                metadata=_report_metadata(
                    evidence,
                    input_digest=input_digest,
                    verified=True,
                    expected=evidence.expected.to_dict(),
                    state_restoration_verified=True,
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
        except Exception as exc:
            return ReplayResult(
                False,
                result=result,
                restored=restored,
                error=f"replay predicate failed: {type(exc).__name__}: {exc}",
                metadata=_report_metadata(
                    evidence,
                    input_digest=input_digest,
                    verified=False,
                    state_restoration_verified=True,
                ),
            )
        return ReplayResult(
            reproduced,
            result=result,
            restored=restored,
            metadata=_report_metadata(
                evidence,
                input_digest=input_digest,
                verified=verified,
                expected=None,
                state_restoration_verified=True,
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
