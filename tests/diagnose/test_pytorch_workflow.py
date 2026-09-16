import pytest

from mlforensics import ReplayEngine, capture_training, shrink
from mlforensics.diagnose.trace import TensorTracer, attach_torch_hooks


def test_pytorch_capture_replay_shrink_and_trace_nonfinite_attention():
    torch = pytest.importorskip("torch")

    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.query = torch.nn.Linear(4, 4, bias=False)
            self.softmax = torch.nn.Softmax(dim=-1)

        def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            scores = self.query(tokens) @ tokens.transpose(-2, -1)
            # Bug: padded positions receive +inf, so a fully padded row is NaN.
            scores = scores.masked_fill(~mask.unsqueeze(-2), float("inf"))
            return self.softmax(scores).sum()

    torch.manual_seed(0)
    model = Attention()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    tokens = torch.randn(1, 3, 4)
    failing = {"tokens": tokens, "mask": torch.zeros(1, 3, dtype=torch.bool)}

    def run_step(batch):
        loss = model(batch["tokens"], batch["mask"])
        if not torch.isfinite(loss):
            raise ValueError("non-finite attention loss")
        return loss

    with pytest.raises(ValueError, match="non-finite attention loss"):
        with capture_training(model, optimizer) as training:
            with training.step(0, batch=failing, sample_ids=["pad-only"]):
                run_step(failing)

    capsule = training.capsule
    replayed = ReplayEngine(
        state_restorers={
            "model": model.load_state_dict,
            "optimizer": optimizer.load_state_dict,
        }
    ).replay(capsule, run_step)
    assert replayed.reproduced

    def still_fails(candidate) -> bool:
        try:
            run_step(candidate)
        except ValueError as exc:
            return "non-finite attention loss" in str(exc)
        return False

    reduced = shrink(failing, still_fails, kind="auto")
    assert still_fails(reduced.value)

    tracer = TensorTracer()
    handles = attach_torch_hooks(model, tracer)
    try:
        with pytest.raises(ValueError, match="non-finite attention loss"):
            run_step(reduced.value)
    finally:
        for handle in handles:
            handle.remove()
    first = tracer.buffer.first_abnormal()
    assert first is not None
    assert first.data.get("abnormal")
