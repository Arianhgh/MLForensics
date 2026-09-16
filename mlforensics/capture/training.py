"""In-process training capture for the single-process eager PyTorch demo."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

from ..core.capture import CaptureContext


class TrainingCapture:
    """Capture pre-step checkpoints for a model/optimizer training loop."""

    def __init__(
        self,
        model: Any,
        optimizer: Any,
        *,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        replay_factory: str | None = None,
        capture: CaptureContext | None = None,
        **capture_kwargs: Any,
    ) -> None:
        providers = {"model": model, "optimizer": optimizer}
        if scheduler is not None:
            providers["scheduler"] = scheduler
        if scaler is not None:
            providers["scaler"] = scaler
        metadata = dict(capture_kwargs.pop("metadata", {}) or {})
        if replay_factory:
            metadata["replay_factory"] = replay_factory
        metadata.setdefault("training_capture", True)
        metadata.setdefault("distributed", False)
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

    def __enter__(self) -> TrainingCapture:
        if self._owns_capture:
            self.capture.__enter__()
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        if self._owns_capture:
            return self.capture.__exit__(exc_type, exc, tb)
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
        if batch is not None:
            self.capture.record_replay_input(batch, name="batch")
        self.capture.record_checkpoint(
            step,
            state_providers=self._providers(),
            batch=batch,
            epoch=epoch,
            sampler_position=sampler_position,
            sample_ids=sample_ids,
            before_step=True,
        )
        try:
            yield self.capture
        except BaseException:
            if batch is not None:
                self.capture.record_offending_batch(batch, step=step, sample_ids=sample_ids)
            raise


def capture_training(
    model: Any,
    optimizer: Any,
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    replay_factory: str | None = None,
    **kwargs: Any,
) -> TrainingCapture:
    """Start a training capture session for single-process CPU training."""
    return TrainingCapture(
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        replay_factory=replay_factory,
        **kwargs,
    )


def restore_training_state(
    model: Any,
    optimizer: Any,
    state: Mapping[str, Any],
    *,
    scheduler: Any | None = None,
    scaler: Any | None = None,
) -> list[str]:
    """Restore captured training objects. Optimizer keys remain integers."""
    restored: list[str] = []
    if "model" in state and hasattr(model, "load_state_dict"):
        model.load_state_dict(state["model"])
        restored.append("model")
    if "optimizer" in state and hasattr(optimizer, "load_state_dict"):
        optimizer.load_state_dict(state["optimizer"])
        restored.append("optimizer")
    if scheduler is not None and "scheduler" in state and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(state["scheduler"])
        restored.append("scheduler")
    if scaler is not None and "scaler" in state and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(state["scaler"])
        restored.append("scaler")
    return restored
