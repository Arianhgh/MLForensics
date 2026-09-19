"""Regression coverage for replay and fresh-process runner edge cases."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import pytest

from mlforensics import CaptureContext
from mlforensics.core import ExecutionService, ExecutionSpec, FailureSignature
from mlforensics.core.codecs import decode_state_tree
from mlforensics.diagnose.factory import replay_in_worker
from mlforensics.diagnose.replay import ReplayEngine
from mlforensics.diagnose.runner import run_many


class FakeSampler:
    def __init__(self) -> None:
        self.position = 99

    def snapshot(self) -> dict[str, int]:
        return {"position": self.position}


class FakeDataLoader:
    def __init__(self) -> None:
        self.cursor = 12

    def snapshot(self) -> dict[str, int]:
        return {"cursor": self.cursor}


def failing_seeded_runner(seed: int) -> None:
    raise KeyError(f"seed {seed} failed")


def _failed_capture(
    *,
    replay_input: Any,
    state_providers: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> CaptureContext:
    capture = CaptureContext(
        replay_input=replay_input,
        state_providers=state_providers,
        metadata=metadata,
    )
    failure = error or ValueError("target failure")
    with pytest.raises(type(failure), match=str(failure)):
        with capture:
            raise failure
    return capture


def test_in_process_replay_serializes_exception_and_reports_verified_fields() -> None:
    capture = _failed_capture(replay_input={"batch": 3})

    def runner(value: Any) -> None:
        assert value == {"batch": 3}
        raise ValueError("target failure")

    result = ReplayEngine().replay(capture.capsule, runner)

    assert result.reproduced is True
    assert result.verified is True
    assert result.failure is not None
    assert result.failure.error_type == "builtins.ValueError"
    serialized = result.to_dict()
    assert serialized["reproduced"] is True
    assert serialized["verified"] is True
    assert FailureSignature.from_dict(serialized["failure"]).message == "target failure"


def test_fresh_process_replay_restores_state_and_serializes_exception(tmp_path: Path) -> None:
    capture = _failed_capture(
        replay_input="boom",
        state_providers={"fixture": lambda: {"version": 1}},
        metadata={"replay_factory": "mlforensics.examples.dummy:DummyReplayFixture"},
    )
    path = capture.capsule.save(tmp_path / "fresh-process.mlcap")

    result = replay_in_worker(path, timeout=10, working_directory=str(tmp_path))

    assert result.reproduced is True
    assert result.verified is True
    assert result.failure is not None
    assert result.failure.error_type == "builtins.ValueError"
    assert result.metadata["worker"] is True
    assert result.metadata["state_restoration_verified"] is True
    assert result.metadata["restored_state"] == ["fixture"]
    assert result.metadata["execution_record"]["result"]["status"] == "fail"


def test_replay_reports_unavailable_checkpoint_without_running_runner() -> None:
    capture = CaptureContext(replay_input="target", checkpoint_limit=3)
    with capture:
        capture.record_checkpoint(4, state_providers={"model": lambda: {"step": 4}})
        capture.record_checkpoint(8, state_providers={"model": lambda: {"step": 8}})

    called: list[Any] = []
    result = ReplayEngine(state_restorers={"model": lambda _state: None}).replay(
        capture.capsule, lambda value: called.append(value), step=2
    )

    assert called == []
    assert result.reproduced is False
    assert result.verified is False
    assert result.metadata["status"] == "unavailable-checkpoint"
    assert result.metadata["requested_step"] == 2


def test_replay_reports_incomplete_checkpoint_gap_without_skipping_steps() -> None:
    capture = CaptureContext(checkpoint_limit=3)
    with capture:
        capture.record_checkpoint(
            0,
            state_providers={"model": lambda: {"step": 0}},
            batch="step-0",
        )
        capture.record_offending_batch("target", step=2)

    called: list[Any] = []
    result = ReplayEngine(state_restorers={"model": lambda _state: None}).replay(
        capture.capsule, lambda value: called.append(value), step=2
    )

    assert called == []
    assert result.reproduced is False
    assert result.verified is False
    assert result.metadata["status"] == "incomplete-replay"
    assert result.metadata["checkpoint_step"] == 0
    assert "intervening input" in (result.error or "")


def test_omitted_state_cannot_claim_verified_restoration_even_when_failure_matches() -> None:
    capture = _failed_capture(
        replay_input="target failure",
        state_providers={"model": lambda: {"weights": [1, 2]}},
    )

    result = ReplayEngine(strict_state=False).replay(
        capture.capsule,
        lambda _value: (_ for _ in ()).throw(ValueError("target failure")),
    )

    assert result.reproduced is True
    assert result.verified is True
    assert result.metadata["state_restoration_verified"] is False
    assert result.metadata["omitted_state"] == ["model"]


def test_incompatible_state_is_a_failed_unverified_replay() -> None:
    capture = _failed_capture(
        replay_input="target failure",
        state_providers={"model": lambda: {"weights": [1, 2]}},
    )

    def incompatible(_state: Any) -> None:
        raise TypeError("incompatible checkpoint")

    result = ReplayEngine(state_restorers={"model": incompatible}).replay(
        capture.capsule,
        lambda _value: (_ for _ in ()).throw(ValueError("target failure")),
    )

    assert result.reproduced is False
    assert result.verified is False
    assert result.metadata["state_restoration_verified"] is False
    assert result.metadata["failed_state"] == ["model"]
    assert "incompatible checkpoint" in (result.error or "")


def test_sampler_checkpoint_restoration_replays_every_intervening_batch_after_failure() -> None:
    sampler = FakeSampler()
    dataloader = FakeDataLoader()
    capture = CaptureContext(replay_input="step-2", checkpoint_limit=3)
    with pytest.raises(ValueError, match="target failure"):
        with capture:
            support = capture.record_replay_support({"dataloader": dataloader, "sampler": sampler})
            checkpoint = capture.record_checkpoint(
                0,
                state_providers={"dataloader": dataloader, "sampler": sampler},
                batch="step-0",
                epoch=4,
                sampler_position=7,
                sample_ids=["a"],
            )
            capture.record_input_history(1, "step-1")
            capture.record_input_history(2, "step-2")
            raise ValueError("target failure")

    restored: list[Any] = []
    executed: list[Any] = []

    def restore(state: Any) -> None:
        restored.append(state)

    def runner(value: Any) -> None:
        executed.append(value)
        if value == "step-2":
            raise ValueError("target failure")

    result = ReplayEngine(state_restorers={"dataloader": restore, "sampler": restore}).replay(
        capture.capsule, runner, step=2
    )

    assert checkpoint["epoch"] == 4
    assert checkpoint["sampler_position"] == 7
    assert support["sampler_state"] is True
    assert decode_state_tree(checkpoint["sample_ids"], capture.capsule.artifact_payload) == ["a"]
    assert restored == [{"cursor": 12}, {"position": 99}]
    assert executed == ["step-0", "step-1", "step-2"]
    assert result.reproduced is True
    assert result.verified is True
    assert result.metadata["planned_steps"] == [0, 1, 2]
    assert result.metadata["executed_steps"] == [0, 1, 2]


def test_torch_rng_metadata_is_captured_and_restored_when_available() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    capture = _failed_capture(replay_input="rng")
    frameworks = capture.capsule.run.rng_state.frameworks

    assert "torch_cpu" in frameworks
    assert "torch_cuda" in frameworks or not torch.cuda.is_available()

    result = ReplayEngine().replay(
        capture.capsule,
        lambda _value: (_ for _ in ()).throw(ValueError("target failure")),
    )

    assert result.reproduced is True
    assert result.verified is True
    assert "Torch CPU RNG" in result.restored
    if torch.cuda.is_available():
        assert "Torch CUDA RNG" in result.restored


def test_execution_service_serializes_timeout_status_and_record() -> None:
    service = ExecutionService()
    result = service.run(
        ExecutionSpec(
            command=[sys.executable, "-c", "import time; time.sleep(5)"],
            timeout=0.1,
        )
    )

    assert result.status == "timeout"
    assert result.returncode is not None
    assert result.error == "execution timed out"
    assert result.metadata["timed_out"] is True
    assert len(service.records) == 1
    assert service.records[0].timed_out is True
    assert service.records[0].result["status"] == "timeout"


def test_seeded_runner_fresh_process_serializes_exception_and_status() -> None:
    result = run_many(failing_seeded_runner, [7], fresh_process=True)[0].result

    assert result.ok is False
    assert result.status == "error"
    assert result.error == "KeyError: 'seed 7 failed'"
    assert result.exception is not None
    assert result.exception["qualified_type"] == "builtins.KeyError"
    assert result.exception["phase"] == "runner"
    assert result.to_dict()["status"] == "error"


def test_rng_seed_does_not_overwrite_an_exact_python_rng_snapshot() -> None:
    random.seed(91)
    random.random()
    expected = random.random()
    random.seed(91)
    random.random()
    capture = _failed_capture(replay_input=expected)

    result = ReplayEngine().replay(
        capture.capsule,
        lambda value: (
            (_ for _ in ()).throw(ValueError("target failure"))
            if random.random() == value
            else None
        ),
    )

    assert result.reproduced is True
    assert "Python RNG seed" not in result.restored
