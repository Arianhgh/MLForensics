"""Regression coverage for optional training state and replay edge cases."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from mlforensics import (
    CaptureContext,
    ExecutionService,
    ExecutionSpec,
    ReplayEngine,
    capture_training,
    restore_training_state,
)
from mlforensics.core.codecs import decode_state_tree


class _StatefulModel:
    def __init__(self) -> None:
        self.weight = 3
        self.training = True

    def state_dict(self) -> dict[str, int]:
        return {"weight": self.weight}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.weight = state["weight"]


class _StatefulOptimizer:
    def __init__(self) -> None:
        self.step_count = 2

    def state_dict(self) -> dict[str, int]:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.step_count = state["step_count"]


class _StatefulComponent:
    def __init__(self, value: int) -> None:
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = state["value"]


class _StatefulSampler(_StatefulComponent):
    def __init__(self, value: int, position: int) -> None:
        super().__init__(value)
        self.position = position

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value, "position": self.position}


def test_training_capture_round_trips_duck_typed_optional_state_and_sampler_metadata() -> None:
    model = _StatefulModel()
    optimizer = _StatefulOptimizer()
    scheduler = _StatefulComponent(4)
    scaler = _StatefulComponent(5)
    dataloader = _StatefulComponent(8)
    sampler = _StatefulSampler(6, position=7)

    training = capture_training(
        model,
        optimizer,
        scheduler=scheduler,
        scaler=scaler,
        dataloader=dataloader,
        sampler=sampler,
        metadata={"seed": 19},
    )
    with training:
        with training.step(
            0,
            batch={"features": [1, 2]},
            sample_ids=["sample-a", "sample-b"],
            epoch=2,
        ):
            model.weight = 99
            optimizer.step_count = 100
            scheduler.value = 101
            scaler.value = 102

    capsule = training.capsule
    replay = capsule.evidence["replay"]
    checkpoint = replay["checkpoints"][0]
    assert set(checkpoint["state"]) == {
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "dataloader",
        "sampler",
    }
    assert checkpoint["epoch"] == 2
    assert checkpoint["sampler_position"] == 7
    assert decode_state_tree(checkpoint["sample_ids"], capsule.payloads.get) == [
        "sample-a",
        "sample-b",
    ]
    assert decode_state_tree(replay["state"]["model"], capsule.payloads.get) == {"weight": 99}
    assert replay["supported_state"]["sampler_state"] is True
    assert replay["supported_state"]["accumulation_state"] is True

    checkpoint_state = {
        name: decode_state_tree(value, capsule.payloads.get)
        for name, value in checkpoint["state"].items()
    }
    restored_model = _StatefulModel()
    restored_optimizer = _StatefulOptimizer()
    restored_scheduler = _StatefulComponent(0)
    restored_scaler = _StatefulComponent(0)
    restored_loader = _StatefulComponent(0)
    restored_sampler = _StatefulSampler(0, position=0)
    restored = restore_training_state(
        restored_model,
        restored_optimizer,
        checkpoint_state,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        dataloader=restored_loader,
        sampler=restored_sampler,
        sampler_position=checkpoint["sampler_position"],
    )
    assert restored == [
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "dataloader",
        "sampler",
        "sampler_position",
    ]
    assert (restored_model.weight, restored_optimizer.step_count) == (3, 2)
    assert (restored_scheduler.value, restored_scaler.value) == (4, 5)
    assert (restored_loader.value, restored_sampler.value, restored_sampler.position) == (8, 6, 7)


def test_torch_capture_is_optional_and_records_tensor_state_rng_and_determinism() -> None:
    torch = pytest.importorskip("torch")

    torch.manual_seed(123)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler("cpu", enabled=False)
        except TypeError:
            scaler = torch.amp.GradScaler(enabled=False)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    before_weight = model.weight.detach().clone()

    with capture_training(model, optimizer, scheduler=scheduler, scaler=scaler) as training:
        with training.step(
            0,
            batch=(torch.ones(1, 2), torch.ones(1, 1)),
            sample_ids=[0],
        ):
            pass

    capsule = training.capsule
    checkpoint = capsule.evidence["replay"]["checkpoints"][0]
    checkpoint_model = decode_state_tree(checkpoint["state"]["model"], capsule.payloads.get)
    assert torch.equal(checkpoint_model["weight"], before_weight)
    assert set(checkpoint["state"]) >= {"model", "optimizer", "scheduler", "scaler"}

    rng_state = capsule.run.rng_state
    assert rng_state is not None
    assert "torch_cpu" in rng_state.frameworks
    assert isinstance(capsule.evidence["replay"]["supported_state"]["autocast"], bool)
    runtime = capsule.evidence["training"]["runtime"]
    assert runtime["torch_loaded"] is True
    assert isinstance(runtime["autocast_enabled"], bool)
    assert isinstance(runtime["deterministic_algorithms"], bool)
    assert capsule.run.replay_plan is not None
    assert capsule.run.replay_plan.determinism == "best_effort"
    checkpoint_rng = checkpoint["rng_state"]
    assert "torch_cpu" in checkpoint_rng["frameworks"]
    if torch.cuda.is_available():
        assert "torch_cuda" in rng_state.frameworks
        assert "torch_cuda" in checkpoint_rng["frameworks"]


def test_replay_reports_omitted_and_incompatible_state() -> None:
    with pytest.raises(ValueError, match="incident"):
        with CaptureContext(
            replay_input="input", state_providers={"model": lambda: {"x": 1}}
        ) as capture:
            raise ValueError("incident")

    omitted = ReplayEngine().replay(
        capture.capsule,
        lambda _input: (_ for _ in ()).throw(ValueError("incident")),
    )
    assert omitted.reproduced is False
    assert omitted.metadata["omitted_state"] == ["model"]
    assert omitted.metadata["state_restoration_verified"] is False

    incompatible = ReplayEngine(
        state_restorers={"model": lambda _state: (_ for _ in ()).throw(TypeError("wrong shape"))}
    ).replay(
        capture.capsule,
        lambda _input: (_ for _ in ()).throw(ValueError("incident")),
    )
    assert incompatible.reproduced is False
    assert incompatible.metadata["failed_state"] == ["model"]
    assert incompatible.metadata["state_restoration_verified"] is False


def test_checkpoint_window_truncation_drops_old_payloads_and_rejects_future_request() -> None:
    with CaptureContext(checkpoint_limit=2, input_history_limit=8) as capture:
        for step in range(4):
            capture.record_checkpoint(
                step,
                state_providers={"model": lambda step=step: {"step": step, "blob": f"blob-{step}"}},
                batch={"step": step},
            )

    replay = capture.capsule.evidence["replay"]
    assert [item["step"] for item in replay["checkpoints"]] == [2, 3]
    assert b"blob-0" not in capture.capsule.payloads.values()
    unavailable = ReplayEngine().replay(capture.capsule, lambda _batch: None, step=1)
    assert unavailable.reproduced is False
    assert unavailable.metadata["status"] == "unavailable-checkpoint"


def test_exception_and_timeout_statuses_remain_distinct(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="capture failure"):
        with CaptureContext(root=tmp_path / "capture") as capture:
            raise RuntimeError("capture failure")
    assert capture.capsule.run.status == "failed"
    assert capture.capsule.run.failure_signature is not None
    assert capture.capsule.run.failure_signature.kind == "exception"

    result = ExecutionService().run(
        ExecutionSpec(
            command=[sys.executable, "-c", "import time; time.sleep(1)"],
            timeout=0.05,
        )
    )
    assert result.status == "timeout"
    assert result.error == "execution timed out"
    assert result.metadata["timed_out"] is True


def test_execution_timeout_drains_verbose_child_output() -> None:
    result = ExecutionService(log_limit=100).run(
        ExecutionSpec(
            command=[
                sys.executable,
                "-c",
                'import sys; sys.stdout.write("x" * 200000); sys.stdout.flush()',
            ],
            timeout=2,
        )
    )
    assert result.status == "ok"
    assert result.returncode == 0
    assert result.metadata["stdout"] == "x" * 100
