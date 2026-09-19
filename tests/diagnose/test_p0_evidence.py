import json
import random
import shlex
import sys
from pathlib import Path

import pytest

from mlforensics import CaptureContext, RunCapsule, dumps, loads
from mlforensics.cli.main import main
from mlforensics.core.contracts import PredicateResult, RunGroup, evaluate_predicate
from mlforensics.core.errors import UnresolvedEvaluation
from mlforensics.diagnose.replay import ReplayEngine
from mlforensics.diagnose.shrink import shrink, shrink_capsule


def test_shrunk_capsule_replays_the_reduced_input_not_the_checkpoint_batch(tmp_path: Path):
    seen: list[object] = []

    def runner(value):
        seen.append(value)
        raise ValueError("target")

    capture = CaptureContext(replay_input=[1, 2], state_providers={"model": lambda: {"n": 0}})
    with pytest.raises(ValueError, match="target"):
        with capture:
            capture.record_checkpoint(0, batch=[1, 2], state_providers={"model": lambda: {"n": 0}})
            raise ValueError("target")

    result, child = shrink_capsule(
        capture.capsule, lambda value: 2 in value, output=tmp_path / "shrunk.mlcap"
    )
    assert result.value == [2]
    seen.clear()
    replayed = ReplayEngine(state_restorers={"model": lambda _state: None}).replay(child, runner)
    assert seen == [[2]]
    assert replayed.reproduced
    assert replayed.metadata["executed_input_digest"]
    assert (
        replayed.metadata["executed_input_digest"]
        != ReplayEngine(state_restorers={"model": lambda _state: None})
        .replay(capture.capsule, lambda value: (_ for _ in ()).throw(ValueError("target")))
        .metadata["executed_input_digest"]
    )


def _two_step_capsule():
    capture = CaptureContext(replay_input=["last"], state_providers={"model": lambda: {"n": 0}})
    with pytest.raises(ValueError, match="last"):
        with capture:
            capture.record_checkpoint(
                0, batch=["first"], state_providers={"model": lambda: {"n": 0}}
            )
            capture.record_checkpoint(
                1, batch=["last"], state_providers={"model": lambda: {"n": 0}}
            )
            raise ValueError("last")
    return capture.capsule


def test_earlier_step_replay_uses_the_earlier_checkpoint_batch():
    capsule = _two_step_capsule()
    seen: list[object] = []

    def runner(value):
        seen.append(value)
        if value == ["last"]:
            raise ValueError("last")

    engine = ReplayEngine(state_restorers={"model": lambda _state: None})
    earlier = engine.replay(capsule, runner, step=0)
    assert seen == [["first"]]
    assert earlier.metadata["requested_step"] == 0
    assert earlier.metadata["checkpoint_step"] == 0
    assert earlier.metadata["executed_step"] == 0
    assert earlier.reproduced is False

    seen.clear()
    default = engine.replay(capsule, lambda value: (_ for _ in ()).throw(ValueError("last")))
    assert default.reproduced
    assert default.metadata["executed_step"] == 1
    assert default.metadata["executed_input_digest"] != earlier.metadata["executed_input_digest"]


def test_missing_intermediate_input_is_an_incomplete_replay():
    capture = CaptureContext(state_providers={"model": lambda: {"n": 0}})
    with pytest.raises(ValueError, match="later"):
        with capture:
            capture.record_checkpoint(
                0, batch=["first"], state_providers={"model": lambda: {"n": 0}}
            )
            capture.record_offending_batch(["later"], step=2)
            raise ValueError("later")

    seen: list[object] = []
    result = ReplayEngine(state_restorers={"model": lambda _state: None}).replay(
        capture.capsule, lambda value: seen.append(value), step=1
    )
    assert seen == []
    assert result.reproduced is False
    assert result.metadata.get("status") == "incomplete-replay"
    assert result.metadata["requested_step"] == 1


def test_shrunk_override_applies_only_at_the_failing_step(tmp_path: Path):
    capsule = _two_step_capsule()
    result, child = shrink_capsule(
        capsule, lambda value: value == ["last"], output=tmp_path / "shrunk.mlcap"
    )
    assert result.value == ["last"]
    seen: list[object] = []
    engine = ReplayEngine(state_restorers={"model": lambda _state: None})
    engine.replay(
        child,
        lambda value: seen.append(value) or (_ for _ in ()).throw(ValueError("last")),
    )
    assert seen == [["last"]]
    seen.clear()
    earlier = engine.replay(child, lambda value: seen.append(value), step=0)
    assert seen == [["first"]]
    assert earlier.metadata["executed_step"] == 0


def test_structured_child_mismatch_is_not_overridden_by_logs(tmp_path: Path, capsys):
    capture = CaptureContext(replay_input=[1])
    with pytest.raises(ValueError, match="boom"):
        with capture:
            raise ValueError("boom")
    path = capture.capsule.save(tmp_path / "failure.mlcap")
    child = (
        "import os, sys\n"
        "from mlforensics.core.contracts import ExecutionResult, write_child_result\n"
        "from mlforensics.core.models import FailureSignature\n"
        "print('ValueError: boom')\n"
        "write_child_result(os.environ['MLFORENSICS_CHILD_RESULT'], ExecutionResult(\n"
        "    status='fail', returncode=1,\n"
        "    failure=FailureSignature("
        "error_type='builtins.KeyError', message='x', kind='exception'),\n"
        "))\n"
        "raise SystemExit(1)\n"
    )
    script = tmp_path / "child.py"
    script.write_text(child, encoding="utf-8")
    command = shlex.quote(sys.executable) + " " + shlex.quote(str(script))
    assert main(["replay", str(path), "--command", command, "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["reproduced"] is False
    assert result["signature_match"] is False


def test_rng_is_restored_after_application_state():
    random.seed(7)
    random.random()
    probe = random.Random()
    probe.setstate(random.getstate())
    wanted = probe.random()

    def runner(value):
        if random.random() == value:
            raise ValueError("target failure")

    capture = CaptureContext(replay_input=wanted, state_providers={"model": lambda: {"n": 1}})
    with pytest.raises(ValueError, match="target failure"):
        with capture:
            runner(wanted)

    def restore_model(_state):
        random.random()

    result = ReplayEngine(state_restorers={"model": restore_model}).replay(capture.capsule, runner)
    assert result.reproduced
    assert "model" in result.restored
    assert "Python RNG" in result.restored


def test_typed_negative_predicate_cannot_accept_a_candidate():
    negative = PredicateResult(preserved=False, status="pass")
    with pytest.raises(Exception, match="does not satisfy"):
        shrink([1, 2], lambda _value: negative, kind="sequence")

    unresolved = shrink(
        [1, 2, 3],
        lambda value: (
            PredicateResult(preserved=True, status="fail")
            if 2 in value
            else (_ for _ in ()).throw(UnresolvedEvaluation("lost"))
        ),
        kind="sequence",
    )
    assert 2 in unresolved.value
    assert any(item.get("unresolved") for item in unresolved.history)
    assert unresolved.metadata["verified"] is False
    assert evaluate_predicate(lambda _value: negative, [1]).preserved is False


def test_run_group_round_trips_through_loads():
    group = RunGroup(group_id="g1", run_ids=("a", "b"), pairing_key="seed")
    restored = loads(dumps(group))
    assert restored == group
    assert isinstance(restored, RunGroup)


def test_core_capture_inherits_parent_run_and_writes_child_capsule(tmp_path, monkeypatch):
    child = tmp_path / "child.mlcap"
    monkeypatch.setenv("MLFORENSICS_CHILD_CAPSULE", str(child))
    monkeypatch.setenv("MLFORENSICS_RUN_ID", "parent-1")
    with CaptureContext() as capture:
        capture.record_metric("loss", 0.5, step=0)
    loaded = RunCapsule.load(child)
    assert loaded.run.run_id == "parent-1"
    assert [metric.name for metric in loaded.run.metrics] == ["loss"]


def test_sparse_replay_does_not_skip_intervening_steps():
    capture = CaptureContext(state_providers={"model": lambda: {"n": 0}})
    with pytest.raises(ValueError, match="later"):
        with capture:
            capture.record_checkpoint(
                0, batch=["first"], state_providers={"model": lambda: {"n": 0}}
            )
            capture.record_offending_batch(["later"], step=2)
            raise ValueError("later")

    seen: list[object] = []
    result = ReplayEngine(state_restorers={"model": lambda _state: None}).replay(
        capture.capsule, lambda value: seen.append(value)
    )
    assert seen == []
    assert result.reproduced is False
    assert result.metadata.get("status") == "incomplete-replay"
    assert result.metadata["requested_step"] == 2 or result.metadata["checkpoint_step"] == 0


def test_replay_executes_intervening_recorded_steps():
    capture = CaptureContext(state_providers={"model": lambda: {"n": 0}}, input_history_limit=8)
    with pytest.raises(ValueError, match="later"):
        with capture:
            capture.record_checkpoint(
                0, batch=["first"], state_providers={"model": lambda: {"n": 0}}
            )
            capture.record_input_history(1, ["middle"])
            capture.record_offending_batch(["later"], step=2)
            raise ValueError("later")

    seen: list[object] = []

    def runner(value):
        seen.append(value)
        if value == ["later"]:
            raise ValueError("later")

    result = ReplayEngine(state_restorers={"model": lambda _state: None}).replay(
        capture.capsule, runner
    )
    assert seen == [["first"], ["middle"], ["later"]]
    assert result.reproduced
    assert result.metadata.get("executed_steps") == [0, 1, 2]


def test_omitted_model_state_is_not_verified_restoration():
    capture = CaptureContext(replay_input=1, state_providers={"model": lambda: {"n": 0}})
    with pytest.raises(ValueError, match="target"):
        with capture:
            raise ValueError("target")

    result = ReplayEngine(strict_state=False).replay(
        capture.capsule, lambda _value: (_ for _ in ()).throw(ValueError("target"))
    )
    assert result.reproduced
    assert result.metadata["state_restoration_verified"] is False
    assert "model" in result.metadata["omitted_state"]
