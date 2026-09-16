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
