from mlforensics.diagnose.replay import TorchStateAdapter


class FakeCuda:
    def __init__(self):
        self.states = ["cuda-state"]

    def is_available(self):
        return True

    def get_rng_state_all(self):
        return self.states

    def set_rng_state_all(self, state):
        self.states = state

    def manual_seed_all(self, seed):
        self.seed = seed


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()
        self.state = "cpu-state"

    def get_rng_state(self):
        return self.state

    def set_rng_state(self, state):
        self.state = state

    def manual_seed(self, seed):
        self.seed = seed


def test_torch_adapter_is_duck_typed_and_lazy():
    torch = FakeTorch()
    adapter = TorchStateAdapter(torch)
    snapshot = adapter.snapshot()
    adapter.seed(4)
    adapter.restore(snapshot)
    assert torch.state == "cpu-state"
    assert torch.cuda.states == ["cuda-state"]


def test_named_generic_snapshots_restore_shape():
    adapter = TorchStateAdapter(FakeTorch())
    assert set(adapter.snapshot()) == {"torch_cpu", "torch_cuda"}
