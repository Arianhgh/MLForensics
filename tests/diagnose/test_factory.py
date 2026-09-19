from pathlib import Path

from mlforensics import CaptureContext
from mlforensics.diagnose.factory import load_replay_factory, replay_in_worker, replay_with_factory
from mlforensics.examples.dummy import DummyReplayFixture


def test_factory_replays_without_live_model_or_manual_restorers():
    capture = CaptureContext(
        replay_input="boom",
        state_providers={"model": lambda: {"n": 1}},
        metadata={"replay_factory": "mlforensics.examples.dummy:DummyReplayFixture"},
    )
    with __import__("pytest").raises(ValueError, match="target failure"):
        with capture:
            raise ValueError("target failure")

    result = replay_with_factory(capture.capsule)
    assert result.reproduced
    assert result.metadata["factory_replay"] is True


def test_worker_replays_in_a_fresh_process(tmp_path: Path):
    capture = CaptureContext(
        replay_input="boom",
        state_providers={"model": lambda: {"n": 1}},
        metadata={"replay_factory": "mlforensics.examples.dummy:DummyReplayFixture"},
    )
    with __import__("pytest").raises(ValueError, match="target failure"):
        with capture:
            raise ValueError("target failure")
    path = capture.capsule.save(tmp_path / "failure.mlcap")
    # A worker commonly runs from an isolated worktree, not the source tree
    # that imported mlforensics in the parent process.
    result = replay_in_worker(path, timeout=30, working_directory=str(tmp_path))
    assert result.metadata["worker"] is True
    assert result.reproduced
    assert result.metadata["state_restoration_verified"] is True
    assert "No module named 'mlforensics'" not in (result.metadata["stderr"] or "")


def test_load_replay_factory_constructs_a_fresh_fixture():
    fixture = load_replay_factory("mlforensics.examples.dummy:DummyReplayFixture")
    assert isinstance(fixture, DummyReplayFixture)
    constructed = fixture.construct()
    fixture.restore({"model": {"n": 9}})
    assert constructed is fixture.state or fixture.state["model"] == {"n": 9}
    fixture.close()
