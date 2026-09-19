"""In-process training capture for the single-process eager PyTorch demo."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from ..core.capture import CaptureContext


def _state_method(provider: Any) -> str | None:
    """Return the portable snapshot protocol implemented by *provider*."""
    if callable(getattr(provider, "snapshot", None)):
        return "snapshot"
    if callable(getattr(provider, "state_dict", None)):
        return "state_dict"
    return None


def _runtime_metadata() -> dict[str, Any]:
    """Read useful torch runtime flags without importing torch."""
    torch = sys.modules.get("torch")
    if torch is None:
        return {"torch_loaded": False, "autocast_enabled": False}

    def call(name: str, default: Any = None) -> Any:
        function = getattr(torch, name, None)
        if not callable(function):
            return default
        try:
            return function()
        except Exception:
            return default

    cuda = getattr(torch, "cuda", None)
    cuda_available = None
    if cuda is not None and callable(getattr(cuda, "is_available", None)):
        try:
            cuda_available = bool(cuda.is_available())
        except Exception:
            pass
    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None) if backends is not None else None
    autocast_enabled = call("is_autocast_enabled", False)
    amp = getattr(torch, "amp", None)
    if amp is not None and callable(getattr(amp, "is_autocast_enabled", None)):
        try:
            autocast_enabled = bool(amp.is_autocast_enabled())
        except Exception:
            pass
    result: dict[str, Any] = {
        "torch_loaded": True,
        "torch_version": str(getattr(torch, "__version__", "")) or None,
        "cuda_available": cuda_available,
        "autocast_enabled": bool(autocast_enabled),
        "deterministic_algorithms": call("are_deterministic_algorithms_enabled"),
        "cudnn_deterministic": getattr(cudnn, "deterministic", None),
        "cudnn_benchmark": getattr(cudnn, "benchmark", None),
    }
    get_autocast_dtype = getattr(torch, "get_autocast_dtype", None)
    if callable(get_autocast_dtype):
        for device in ("cpu", "cuda"):
            try:
                result[f"autocast_{device}_dtype"] = str(get_autocast_dtype(device))
            except Exception:
                pass
    else:
        for name in ("get_autocast_cpu_dtype", "get_autocast_gpu_dtype"):
            value = call(name)
            if value is not None:
                result[name.removeprefix("get_")] = str(value)
    return result


def _sampler_position(sampler: Any) -> int | None:
    """Best-effort position lookup for common sampler implementations."""
    if sampler is None:
        return None
    for name in ("position", "_position", "num_yielded", "_num_yielded", "index", "_index"):
        value = getattr(sampler, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    state_dict = getattr(sampler, "state_dict", None)
    if callable(state_dict):
        try:
            state = state_dict()
        except Exception:
            return None
        if isinstance(state, Mapping):
            for name in ("position", "num_yielded", "index"):
                value = state.get(name)
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                    return value
    return None


def _limitation(component: str, reason: str, message: str) -> dict[str, str]:
    return {"component": component, "reason": reason, "message": message}


class TrainingRestoreReport(list[str]):
    """List-compatible restoration result with structured diagnostics."""

    def __init__(
        self,
        restored: Sequence[str] = (),
        *,
        limitations: Sequence[Mapping[str, Any]] = (),
        status: str = "completed",
    ) -> None:
        super().__init__(str(item) for item in restored)
        self.limitations = tuple(dict(item) for item in limitations)
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "restored": list(self),
            "limitations": [dict(item) for item in self.limitations],
        }


class TrainingCapture:
    """Capture pre-step checkpoints for a model/optimizer training loop."""

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        *,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        dataloader: Any | None = None,
        sampler: Any | None = None,
        replay_factory: str | None = None,
        timeout: float | None = None,
        capture: CaptureContext | None = None,
        **capture_kwargs: Any,
    ) -> None:
        providers = {"model": model, "optimizer": optimizer}
        if scheduler is not None:
            providers["scheduler"] = scheduler
        if scaler is not None:
            providers["scaler"] = scaler
        if dataloader is not None and _state_method(dataloader) is not None:
            providers["dataloader"] = dataloader
        self.dataloader = dataloader
        self.sampler = sampler if sampler is not None else getattr(dataloader, "sampler", None)
        if self.sampler is not None and _state_method(self.sampler) is not None:
            providers["sampler"] = self.sampler
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError("timeout must be a positive finite number or None")
        metadata = dict(capture_kwargs.pop("metadata", {}) or {})
        if replay_factory:
            metadata["replay_factory"] = replay_factory
        metadata.setdefault("training_capture", True)
        metadata.setdefault("distributed", False)
        self._component_objects = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "dataloader": dataloader,
            "sampler": self.sampler,
        }
        self._limitations: list[dict[str, str]] = []
        for name, component in self._component_objects.items():
            if component is None:
                if name in {"scheduler", "scaler", "dataloader", "sampler"}:
                    self._add_limitation(name, "omitted", f"{name} was not provided")
            elif _state_method(component) is None:
                self._add_limitation(
                    name,
                    "incompatible",
                    f"{type(component).__module__}.{type(component).__qualname__} "
                    "does not expose snapshot() or state_dict()",
                )
        self._timeout = float(timeout) if timeout is not None else None
        metadata.setdefault(
            "training",
            {
                "components": {
                    name: {
                        "provided": component is not None,
                        "snapshot_protocol": _state_method(component),
                    }
                    for name, component in self._component_objects.items()
                },
                "limitations": [dict(item) for item in self._limitations],
                "checkpoint_window": {
                    "limit": capture_kwargs.get("checkpoint_limit", 3),
                    "interval": capture_kwargs.get("checkpoint_interval", 1),
                },
                "timeout_seconds": self._timeout,
            },
        )
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.replay_factory = replay_factory
        self.capture = capture or CaptureContext(
            state_providers=providers, metadata=metadata, **capture_kwargs
        )
        self._owns_capture = capture is None
        self._entered = False

    def _add_limitation(self, component: str, reason: str, message: str) -> None:
        item = _limitation(component, reason, message)
        if not any(
            existing["component"] == component and existing["reason"] == reason
            for existing in self._limitations
        ):
            self._limitations.append(item)

    def _training_evidence(
        self, *, status: str = "running", error: BaseException | None = None
    ) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "status": status,
            "components": {
                name: {
                    "provided": component is not None,
                    "snapshot_protocol": _state_method(component),
                    "restorable": _state_method(component) is not None,
                }
                for name, component in self._component_objects.items()
            },
            "limitations": [dict(item) for item in self._limitations],
            "checkpoint_window": {
                "limit": getattr(self.capture, "_checkpoint_limit", None),
                "interval": getattr(self.capture, "_checkpoint_interval", None),
            },
            "runtime": _runtime_metadata(),
            "rng": {
                "captured": self.capture.run.rng_state is not None,
                "frameworks": sorted(getattr(self.capture.run.rng_state, "frameworks", {}).keys()),
            },
            "timeout_seconds": self._timeout,
        }
        if error is not None:
            evidence["error"] = {
                "type": f"{type(error).__module__}.{type(error).__qualname__}",
                "message": str(error),
                "kind": "timeout" if isinstance(error, TimeoutError) else "exception",
            }
        return evidence

    def _record_training_evidence(
        self, *, status: str = "running", error: BaseException | None = None
    ) -> None:
        try:
            self.capture.record_evidence(
                "training", self._training_evidence(status=status, error=error)
            )
        except BaseException as capture_exc:
            self.capture.record_capture_error("training_metadata", capture_exc)

    def __enter__(self) -> TrainingCapture:
        if self._owns_capture:
            self.capture.__enter__()
        self._entered = True
        self._record_training_evidence()
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        self._record_training_evidence(
            status=("timeout" if isinstance(exc, TimeoutError) else "failed")
            if exc
            else "completed",
            error=exc,
        )
        if self._owns_capture:
            result = self.capture.__exit__(exc_type, exc, tb)
            self._entered = False
            return result
        self._entered = False
        return False

    @property
    def capsule(self) -> Any:
        return self.capture.capsule

    def _providers(self) -> dict[str, Any]:
        providers = {"model": self.model, "optimizer": self.optimizer}
        if self.scheduler is not None:
            providers["scheduler"] = self.scheduler
        if self.scaler is not None:
            providers["scaler"] = self.scaler
        if self.dataloader is not None and _state_method(self.dataloader) is not None:
            providers["dataloader"] = self.dataloader
        if self.sampler is not None and _state_method(self.sampler) is not None:
            providers["sampler"] = self.sampler
        return providers

    @contextmanager
    def step(
        self,
        step: int,
        batch: Any = None,
        sample_ids: Sequence[Any] | None = None,
        *,
        epoch: int | None = None,
        sampler_position: int | None = None,
    ) -> Iterator[CaptureContext]:
        """Capture state before a training step executes."""
        if sampler_position is None:
            sampler_position = _sampler_position(self.sampler)
        if sample_ids is None and isinstance(batch, Mapping):
            candidate = batch.get("sample_ids")
            if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
                sample_ids = candidate
        if sample_ids is None:
            self._add_limitation(
                "sample_ids", "omitted", f"sample IDs were not supplied for step {step}"
            )
        if batch is not None:
            try:
                self.capture.record_replay_input(batch, name="batch")
            except BaseException as exc:
                self.capture.record_capture_error("replay_input", exc)
        self.capture.record_checkpoint(
            step,
            state_providers=self._providers(),
            batch=batch,
            epoch=epoch,
            sampler_position=sampler_position,
            sample_ids=sample_ids,
            before_step=True,
        )
        self._record_training_evidence()
        self.capture.record_replay_support(self._providers())
        try:
            yield self.capture
        except BaseException:
            if batch is not None:
                try:
                    self.capture.record_offending_batch(batch, step=step, sample_ids=sample_ids)
                except BaseException as capture_exc:
                    self.capture.record_capture_error("offending_batch", capture_exc)
            raise


def capture_training(
    model: Any,
    optimizer: Any,
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    dataloader: Any | None = None,
    sampler: Any | None = None,
    replay_factory: str | None = None,
    timeout: float | None = None,
    **kwargs: Any,
) -> TrainingCapture:
    """Start a training capture session for single-process CPU training."""
    return TrainingCapture(
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        dataloader=dataloader,
        sampler=sampler,
        replay_factory=replay_factory,
        timeout=timeout,
        **kwargs,
    )


def restore_training_state(
    model: Any,
    optimizer: Any,
    state: Mapping[str, Any],
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    dataloader: Any | None = None,
    sampler: Any | None = None,
    rng_state: Any | None = None,
    sampler_position: int | None = None,
    report: dict[str, Any] | None = None,
    strict: bool = False,
) -> list[str]:
    """Best-effort restore of captured training state.

    The list result remains backward compatible. It is a
    :class:`TrainingRestoreReport` carrying structured ``limitations`` and a
    status; pass ``report`` to receive the same JSON-compatible information.
    """
    if not isinstance(state, Mapping):
        raise TypeError("state must be a mapping")
    source = state
    nested = state.get("state")
    component_names = ("model", "optimizer", "scheduler", "scaler", "dataloader", "sampler")
    if isinstance(nested, Mapping) and not any(name in state for name in component_names):
        nested_has_component = any(name in nested for name in component_names)
    else:
        nested_has_component = False
    if nested_has_component:
        source = nested
    targets = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "dataloader": dataloader,
        "sampler": sampler,
    }
    restored: list[str] = []
    limitations: list[dict[str, str]] = []

    def add(component: str, reason: str, message: str) -> None:
        item = _limitation(component, reason, message)
        limitations.append(item)
        if strict:
            raise RuntimeError(message)

    for name, target in targets.items():
        present = name in source
        if target is None:
            if present:
                add(name, "target_omitted", f"captured {name} state has no restore target")
            elif name in {"scheduler", "scaler", "dataloader", "sampler"}:
                add(name, "omitted", f"{name} state was omitted")
            continue
        if not present:
            add(name, "state_omitted", f"captured state does not contain {name}")
            continue
        loader = getattr(target, "load_state_dict", None)
        if not callable(loader):
            add(name, "incompatible", f"restore target for {name} lacks load_state_dict()")
            continue
        try:
            loader(source[name])
        except BaseException as exc:
            add(name, "incompatible", f"could not restore {name}: {type(exc).__name__}: {exc}")
            continue
        restored.append(name)

    if sampler_position is not None:
        if sampler is None:
            add(
                "sampler_position",
                "target_omitted",
                "sampler position was captured without a sampler target",
            )
        else:
            setter = getattr(sampler, "set_position", None)
            if callable(setter):
                try:
                    setter(sampler_position)
                    restored.append("sampler_position")
                except BaseException as exc:
                    add(
                        "sampler_position",
                        "incompatible",
                        f"could not restore sampler position: {exc}",
                    )
            elif hasattr(sampler, "position"):
                try:
                    sampler.position = sampler_position
                    restored.append("sampler_position")
                except BaseException as exc:
                    add(
                        "sampler_position",
                        "incompatible",
                        f"could not restore sampler position: {exc}",
                    )
            else:
                add(
                    "sampler_position",
                    "incompatible",
                    "sampler has no set_position() or position attribute",
                )

    captured_rng = (
        rng_state if rng_state is not None else source.get("rng_state", state.get("rng_state"))
    )
    if captured_rng is not None:
        try:
            from ..diagnose.replay import restore_rng_state

            normalized_rng = captured_rng
            if isinstance(captured_rng, Mapping):
                rng_mapping = captured_rng.get("states", captured_rng)
                if isinstance(rng_mapping, Mapping):
                    frameworks = rng_mapping.get("frameworks")
                    if isinstance(frameworks, Mapping):
                        normalized_rng = {
                            key: value
                            for key, value in rng_mapping.items()
                            if key in {"python", "numpy"}
                        }
                        normalized_rng.update(frameworks)
                    expected_rng = {
                        label
                        for key, label in (
                            ("python", "Python RNG"),
                            ("numpy", "NumPy RNG"),
                            ("torch_cpu", "Torch CPU RNG"),
                            ("torch_cuda", "Torch CUDA RNG"),
                        )
                        if key
                        in (normalized_rng if isinstance(normalized_rng, Mapping) else rng_mapping)
                    }
                else:
                    expected_rng = set()
            else:
                expected_rng = {"Python RNG"}
            if hasattr(captured_rng, "python"):
                expected_rng = set()
                if getattr(captured_rng, "python", None) is not None:
                    expected_rng.add("Python RNG")
                if getattr(captured_rng, "numpy", None) is not None:
                    expected_rng.add("NumPy RNG")
                expected_rng.update(
                    "Torch CPU RNG" if name == "torch_cpu" else "Torch CUDA RNG"
                    for name in getattr(captured_rng, "frameworks", {})
                    if name in {"torch_cpu", "torch_cuda"}
                )
            restored_rng = restore_rng_state(normalized_rng)
            restored.extend(restored_rng)
            missing_rng = expected_rng.difference(restored_rng)
            if missing_rng:
                add(
                    "rng",
                    "incompatible",
                    "could not restore " + ", ".join(sorted(missing_rng)),
                )
        except BaseException as exc:
            add("rng", "incompatible", f"could not restore RNG state: {type(exc).__name__}: {exc}")
    else:
        add("rng", "omitted", "captured RNG state was omitted")

    result = TrainingRestoreReport(
        restored,
        limitations=limitations,
        status="completed" if not limitations else "partial",
    )
    if report is not None:
        report.clear()
        report.update(result.to_dict())
    return result
