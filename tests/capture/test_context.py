from pathlib import Path

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
