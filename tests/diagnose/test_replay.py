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
