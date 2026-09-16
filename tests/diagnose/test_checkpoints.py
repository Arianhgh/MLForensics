import pytest

from mlforensics import CaptureContext, ReplayEngine
from mlforensics.core.codecs import collect_artifact_digests, decode_state_tree


def test_replay_uses_pre_step_state_not_post_failure_mutation():
    state = {"step": 0}

    def step(_input):
        state["step"] += 1
        if state["step"] == 1:
            raise ValueError("target failure")

    capture = CaptureContext(replay_input=1, state_providers={"model": lambda: dict(state)})
    with pytest.raises(ValueError, match="target failure"):
        with capture:
            step(1)

    replay_state = decode_state_tree(capture.capsule.evidence["replay"]["state"]["model"])
    diagnostic = decode_state_tree(capture.capsule.evidence["replay"]["diagnostic_state"]["model"])
    assert replay_state["step"] == 0
    assert diagnostic["step"] == 1

    restored = {}

    def restore(value):
        restored.clear()
        restored.update(value)
        state.clear()
        state.update(value)

    result = ReplayEngine(state_restorers={"model": restore}).replay(capture.capsule, step)
    assert result.reproduced
    assert restored["step"] == 0


def test_replay_never_selects_a_future_checkpoint():
    with CaptureContext(replay_input=0, checkpoint_limit=3) as capture:
        capture.record_checkpoint(4, state_providers={"model": lambda: {"n": 4}})
        capture.record_checkpoint(8, state_providers={"model": lambda: {"n": 8}})

    result = ReplayEngine(state_restorers={"model": lambda _state: None}).replay(
        capture.capsule, lambda _value: None, step=2
    )
    assert result.reproduced is False
    assert result.metadata.get("status") == "unavailable-checkpoint"
    assert result.error is not None


def test_checkpoint_retention_evicts_unreferenced_payloads():
    with CaptureContext(replay_input=b"keep", checkpoint_limit=1) as capture:
        capture.record_checkpoint(0, state_providers={"model": lambda: {"blob": b"old-payload-0"}})
        capture.record_checkpoint(1, state_providers={"model": lambda: {"blob": b"new-payload-1"}})

    live = collect_artifact_digests(capture.capsule.evidence)
    live.update(
        ref.sha256
        for ref in capture.capsule.artifacts
        if ref.metadata.get("role") != "replay_state"
    )
    assert set(capture.capsule.payloads) <= live
    encoded_checkpoints = capture.capsule.evidence["replay"]["checkpoints"]
    assert len(encoded_checkpoints) == 1
    assert encoded_checkpoints[0]["step"] == 1
    old = b"old-payload-0"
    assert old not in capture.capsule.payloads.values()
