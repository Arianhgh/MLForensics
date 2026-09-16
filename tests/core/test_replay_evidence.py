import zipfile
from pathlib import Path

import pytest

from mlforensics import CaptureContext, Run, RunCapsule
from mlforensics.diagnose.replay import ReplayEngine


class Restorer:
    def __init__(self) -> None:
        self.values = []

    def restore(self, value) -> None:
        self.values.append(value)


def test_capsule_structured_layout_and_replay_round_trip(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bad 42"):
        with CaptureContext(
            run_id="evidence-1",
            replay_input={"batch": [1, 2]},
            state_providers={"optimizer": lambda: {"step": 7}},
            evidence={"code": {"commit": "abc"}},
        ) as capture:
            raise ValueError("bad 42")

    archive = capture.capsule.save(tmp_path / "failure.mlcap.zip")
    with zipfile.ZipFile(archive) as contents:
        names = set(contents.namelist())
    assert {
        "code.json",
        "failure/exception.json",
        "randomness/python.rng",
        "replay/inputs.json",
        "replay/state.json",
    } <= names

    loaded = RunCapsule.load(archive)
    restored = Restorer()

    def fail(value) -> None:
        assert value == {"batch": [1, 2]}
        raise ValueError("bad 999")

    result = ReplayEngine(state_restorers={"optimizer": restored}).replay(loaded, fail)
    assert result.reproduced and result.verified
    assert restored.values == [{"step": 7}]


def test_capsule_replay_rejects_false_positives() -> None:
    with pytest.raises(ValueError):
        with CaptureContext(run_id="failed") as capture:
            raise ValueError("expected")

    successful = ReplayEngine().replay(capture.capsule, lambda: None)
    different = ReplayEngine().replay(
        capture.capsule, lambda: (_ for _ in ()).throw(RuntimeError("different"))
    )
    assert not successful.reproduced and successful.verified
    assert not different.reproduced and different.verified

    no_signature = RunCapsule(Run(run_id="ok", status="completed", started_at="t"))
    unverified = ReplayEngine().replay(
        no_signature, lambda: (_ for _ in ()).throw(RuntimeError("unexpected"))
    )
    assert not unverified.reproduced and not unverified.verified


def test_binary_replay_state_is_embedded_and_requires_a_restorer(tmp_path: Path) -> None:
    with CaptureContext(run_id="binary", replay_input=b"input") as capture:
        capture.record_state("framework", b"state")
    loaded = RunCapsule.load(capture.capsule.save(tmp_path / "binary.mlcap"))

    missing = ReplayEngine().replay(loaded, lambda value: {"reproduced": value == b"input"})
    assert not missing.reproduced
    assert "missing state restorer" in (missing.error or "")

    restorer = Restorer()
    replayed = ReplayEngine(state_restorers={"framework": restorer}).replay(
        loaded, lambda value: {"reproduced": value == b"input"}
    )
    assert replayed.reproduced and replayed.verified
    assert restorer.values == [b"state"]
