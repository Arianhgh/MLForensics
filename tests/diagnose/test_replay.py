from types import SimpleNamespace

from mlforensics import FailureSignature
from mlforensics.diagnose.replay import ReplayEngine


class Hooks:
    def __init__(self):
        self.restored = []

    def restore(self, state):
        self.restored.append(state)


def test_replay_restores_state_and_seed():
    hooks = Hooks()
    report = ReplayEngine(state_restorers={"custom": hooks}).replay(
        {"state": {"custom": 4}}, lambda _incident: (_ for _ in ()).throw(ValueError("bad"))
    )
    assert report.reproduced and report.restored == ["custom"]
    assert hooks.restored == [4]


def test_replay_requires_hook_for_state():
    report = ReplayEngine().replay({"state": {}}, lambda _incident: False)
    assert not report.reproduced and report.error is None


def test_replay_accepts_zero_argument_runner():
    report = ReplayEngine().replay({}, lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert report.reproduced and report.failure is not None


def test_omitted_state_is_structured_as_incomplete_even_when_failure_matches():
    expected = FailureSignature.from_exception(ValueError("target"))
    report = ReplayEngine(strict_state=False).replay(
        {"input": 1, "state": {"model": {"weight": 1}}, "failure": expected.to_dict()},
        lambda _value: (_ for _ in ()).throw(ValueError("target")),
    )

    assert report.reproduced
    assert report.verified
    assert report.metadata["status"] == "incomplete-replay"
    assert report.metadata["replay_status"] == "incomplete"
    assert report.metadata["omitted_state"] == ["model"]
    assert report.metadata["state_restoration_verified"] is False


def test_load_state_dict_incompatibility_is_not_reported_as_restored():
    class Incompatible:
        def load_state_dict(self, _state):
            return SimpleNamespace(missing_keys=["bias"], unexpected_keys=[])

    report = ReplayEngine(state_restorers={"model": Incompatible()}).replay(
        {"input": 1, "state": {"model": {"weight": 1}}}, lambda _value: True
    )

    assert not report.reproduced
    assert report.metadata["status"] == "incomplete"
    assert report.metadata["failed_state"] == ["model"]
    assert "missing_keys" in report.metadata["limitations"][0]


def test_replay_status_distinguishes_success_and_matching_failure():
    expected = FailureSignature.from_exception(ValueError("target"))
    success = ReplayEngine().replay({"failure": expected.to_dict()}, lambda: None)
    failure = ReplayEngine().replay(
        {"failure": expected.to_dict()},
        lambda: (_ for _ in ()).throw(ValueError("target")),
    )

    assert success.metadata["status"] == "success"
    assert success.verified and not success.reproduced
    assert failure.metadata["status"] == "failure"
    assert failure.verified and failure.reproduced
