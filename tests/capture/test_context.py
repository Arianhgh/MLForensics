import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlforensics.capture import CaptureContext, CaptureRunner, RunCapsule, capture
from mlforensics.core import RunCapsule as CoreRunCapsule


def test_context_records_expected_sections_and_metrics(tmp_path: Path) -> None:
    data = tmp_path / "data.txt"
    data.write_text("dataset", encoding="utf-8")
    context = CaptureContext(
        repo_path=tmp_path,
        data_paths={"training": data},
        env_allowlist=("CAPTURE_TEST_VALUE",),
        environ={"CAPTURE_TEST_VALUE": "ok"},
        use_core_capsule=False,
    )
    with context as active:
        active.record_metric("accuracy", 0.875)

    capsule = context.capsule
    assert isinstance(capsule, RunCapsule)
    assert {
        "manifest",
        "environment",
        "code",
        "hardware",
        "dependencies",
        "data_fingerprints",
        "rng_snapshots",
    } <= capsule.to_dict().keys()
    assert capsule.metrics["accuracy"] == 0.875
    assert capsule.environment["variables"] == {"CAPTURE_TEST_VALUE": "ok"}
    assert capsule.data_fingerprints["training"]["digest"]
    assert "python" in capsule.rng_snapshots
    assert "end" in capsule.rng_snapshots


def test_runner_records_exception_and_can_continue() -> None:
    runner = CaptureRunner(use_core_capsule=False, raise_exceptions=False, repo_path=None)

    def fail() -> None:
        raise RuntimeError("expected failure")

    capsule = runner.run(fail, metrics={"attempted": 1})
    assert runner.result is None
    assert capsule.metrics["attempted"] == 1
    assert any(item["type"] == "RuntimeError" for item in capsule.exceptions)


def test_runner_reraises_workload_exception() -> None:
    runner = CaptureRunner(use_core_capsule=False, repo_path=None)
    with pytest.raises(ValueError, match="bad"):
        runner.run(lambda: (_ for _ in ()).throw(ValueError("bad")))
    assert runner.capsule.exceptions


def test_session_capsule_is_the_portable_format(tmp_path) -> None:
    """``session.capsule`` must be loadable by the rest of the library."""
    with capture(root=tmp_path, name="train", autosave=False) as session:
        session.record_metric("accuracy", 0.91, step=0)
        session.record_resource("latency_ms", 17.2, units="ms")

    target = session.capsule.save(tmp_path / "session.mlcap")
    loaded = CoreRunCapsule.load(target)

    assert loaded.run.run_id == session.run_id
    assert [item.name for item in loaded.run.metrics] == ["accuracy"]
    assert [item.name for item in loaded.run.resources] == ["latency_ms"]


def test_session_failure_signature_keeps_the_originating_frame(tmp_path) -> None:
    with pytest.raises(RuntimeError):
        with capture(root=tmp_path, autosave=False) as session:
            raise RuntimeError("mat1 and mat2 shapes cannot be multiplied")

    signature = session.capsule.run.failure_signature
    assert signature is not None
    assert signature.error_type == "builtins.RuntimeError"
    assert signature.top_frame is not None
    assert signature.top_frame.startswith(str(Path(__file__)))


def test_context_captures_loaded_torch_rng_and_determinism_metadata(monkeypatch) -> None:
    class State:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    fake_torch = SimpleNamespace(
        __version__="2.test",
        random=SimpleNamespace(get_rng_state=lambda: State([1, 2, 3])),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index: f"GPU-{index}",
            get_device_capability=lambda index: (9, 0),
            get_rng_state_all=lambda: [State([4, 5, 6])],
        ),
        are_deterministic_algorithms_enabled=lambda: True,
        is_autocast_enabled=lambda: True,
        get_autocast_cpu_dtype=lambda: "float16",
        get_autocast_gpu_dtype=lambda: "bfloat16",
        is_autocast_cache_enabled=lambda: False,
        backends=SimpleNamespace(cudnn=SimpleNamespace(deterministic=True, benchmark=False)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Parameter:
        device = "cuda:0"
        dtype = "float16"
        requires_grad = True

        def numel(self):
            return 7

    model = SimpleNamespace(training=True, parameters=lambda: iter([Parameter()]))
    with CaptureContext(repo_path=None, model=model) as capture:
        pass

    data = capture.capsule.to_dict()
    torch_data = data["hardware"]["pytorch"]
    assert torch_data["version"] == "2.test"
    assert torch_data["deterministic_algorithms"] is True
    assert torch_data["autocast"] == {
        "enabled": True,
        "cpu_dtype": "float16",
        "gpu_dtype": "bfloat16",
        "cache_enabled": False,
    }
    assert data["rng_snapshots"]["torch_cpu"] == [1, 2, 3]
    assert data["rng_snapshots"]["torch_cuda"] == [[4, 5, 6]]
    assert torch_data["model"]["parameter_count"] == 7


def test_context_records_omitted_and_incompatible_state_structurally() -> None:
    class BrokenProvider:
        def snapshot(self):
            raise RuntimeError("state unavailable")

    with CaptureContext(
        repo_path=None,
        state_providers={"optimizer": BrokenProvider()},
        checkpoint_limit=2,
    ) as capture:
        capture.record_checkpoint(1)

    limitations = capture.capsule.manifest["limitations"]
    assert any(
        item["component"] == "state:optimizer" and item["status"] == "incompatible"
        for item in limitations
    )
    assert any(
        item["component"] == "application_state" or item["component"] == "checkpoint:optimizer"
        for item in limitations
    )
    window = capture.capsule.manifest["checkpoint_window"]
    assert window["limit"] == 2
    assert window["retained"] == 1
    assert window["first_step"] == window["last_step"] == 1


def test_runner_timeout_preserves_structured_timeout_state() -> None:
    runner = CaptureRunner(repo_path=None, timeout=0.005, raise_exceptions=False)
    capsule = runner.run(lambda: time.sleep(0.05))

    assert capsule.manifest["status"] == "timeout"
    assert capsule.manifest["timeout"]["seconds"] == 0.005
    assert capsule.exceptions[0]["kind"] == "timeout"
