import pytest

from mlforensics import ReplayEngine, capture_training, restore_training_state
from mlforensics.core.codecs import decode_state_tree


class FakeModel:
    def __init__(self, weight: int = 1) -> None:
        self.weight = weight

    def state_dict(self) -> dict[str, int]:
        return {"weight": self.weight}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.weight = state["weight"]


class FakeOptimizer:
    def __init__(self) -> None:
        self.state = {0: {"step": 0}}
        self.param_groups = [{"lr": 0.1}]

    def state_dict(self) -> dict[str, object]:
        return {"state": dict(self.state), "param_groups": list(self.param_groups)}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.state = dict(state["state"])  # type: ignore[arg-type]
        self.param_groups = list(state["param_groups"])  # type: ignore[arg-type]


class FakeStateful:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.value = state["value"]


class FakeSampler(FakeStateful):
    def __init__(self) -> None:
        super().__init__()
        self.position = 0

    def state_dict(self) -> dict[str, int]:
        return {"value": self.value, "position": self.position}


class Incompatible:
    pass


def test_training_capture_records_pre_step_state_and_batch() -> None:
    model = FakeModel()
    optimizer = FakeOptimizer()
    batch = {"x": [1.0, 2.0, 3.0], "sample_ids": ["a"]}

    with pytest.raises(ValueError, match="target failure"):
        with capture_training(model, optimizer, replay_factory="demo.restore") as training:
            with training.step(0, batch=batch, sample_ids=["a"], epoch=1, sampler_position=0):
                model.weight = 99
                optimizer.state[0]["step"] = 7
                raise ValueError("target failure")

    capsule = training.capsule
    assert capsule.run.failure_signature is not None
    replay = capsule.evidence["replay"]
    restored_model = decode_state_tree(replay["state"]["model"], capsule.payloads.get)
    restored_optimizer = decode_state_tree(replay["state"]["optimizer"], capsule.payloads.get)
    assert restored_model["weight"] == 1
    assert list(restored_optimizer["state"]) == [0]
    assert replay["checkpoints"][0]["before_step"] is True
    decoded_batch = decode_state_tree(replay["input"], capsule.payloads.get)
    assert decoded_batch["x"] == [1.0, 2.0, 3.0]

    clone = FakeModel(weight=0)
    clone_opt = FakeOptimizer()
    restored = restore_training_state(
        clone, clone_opt, {"model": restored_model, "optimizer": restored_optimizer}
    )
    assert restored == ["model", "optimizer"]
    assert clone.weight == 1
    assert clone_opt.state[0]["step"] == 0

    replayed = ReplayEngine(
        state_restorers={
            "model": clone.load_state_dict,
            "optimizer": clone_opt.load_state_dict,
        }
    ).replay(capsule, lambda _batch: (_ for _ in ()).throw(ValueError("target failure")))
    assert replayed.reproduced
    assert clone.weight == 1


def test_training_capture_records_bounded_window_runtime_and_timeout_status() -> None:
    scheduler = FakeStateful(2)
    scaler = FakeStateful(3)
    sampler = FakeSampler()
    loader = FakeStateful(4)

    with pytest.raises(TimeoutError, match="deadline"):
        with capture_training(
            FakeModel(),
            FakeOptimizer(),
            scheduler=scheduler,
            scaler=scaler,
            dataloader=loader,
            sampler=sampler,
            checkpoint_limit=2,
        ) as training:
            for step in range(3):
                sampler.position = step
                with training.step(step, batch={"x": [step]}, sample_ids=[step]):
                    if step == 2:
                        raise TimeoutError("deadline")

    replay = training.capsule.evidence["replay"]
    assert [item["step"] for item in replay["checkpoints"]] == [1, 2]
    assert replay["checkpoints"][-1]["sampler_position"] == 2
    assert decode_state_tree(
        replay["checkpoints"][-1]["sample_ids"], training.capsule.payloads.get
    ) == [2]
    details = training.capsule.evidence["training"]
    assert details["status"] == "timeout"
    assert details["checkpoint_window"]["limit"] == 2
    assert details["components"]["scheduler"]["restorable"] is True
    assert details["components"]["sampler"]["restorable"] is True

    checkpoint = replay["checkpoints"][-1]
    state = {
        name: decode_state_tree(checkpoint["state"][name], training.capsule.payloads.get)
        for name in ("model", "optimizer", "scheduler", "scaler", "dataloader", "sampler")
    }
    restored_scheduler = FakeStateful()
    restored_scaler = FakeStateful()
    restored_loader = FakeStateful()
    restored_sampler = FakeSampler()
    restored = restore_training_state(
        FakeModel(),
        FakeOptimizer(),
        state,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        dataloader=restored_loader,
        sampler=restored_sampler,
        sampler_position=checkpoint["sampler_position"],
    )
    assert {"model", "optimizer", "scheduler", "scaler", "dataloader", "sampler"}.issubset(restored)
    assert restored_scheduler.value == 2
    assert restored_scaler.value == 3
    assert restored_loader.value == 4
    assert restored_sampler.position == 2


def test_restore_training_state_reports_omitted_and_incompatible_state() -> None:
    report: dict[str, object] = {}
    restored = restore_training_state(
        FakeModel(),
        FakeOptimizer(),
        {"model": {"weight": 8}, "scheduler": {"value": 1}},
        scheduler=Incompatible(),
        report=report,
    )

    assert restored == ["model"]
    assert report["status"] == "partial"
    limitations = report["limitations"]
    assert isinstance(limitations, list)
    assert any(item["component"] == "scheduler" for item in limitations)  # type: ignore[index]
    assert any(item["component"] == "rng" for item in limitations)  # type: ignore[index]
    assert any(item["component"] == "scaler" for item in limitations)  # type: ignore[index]
