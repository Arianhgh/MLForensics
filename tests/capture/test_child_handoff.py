from pathlib import Path

from mlforensics.capture import (
    CHILD_CAPSULE_ENV,
    RUN_ID_ENV,
    CaptureSession,
    child_capture_environment,
    load_child_capsule,
)


def test_child_capture_environment_and_automatic_handoff(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "child.mlcap"
    environment = child_capture_environment(target, run_id="shared-run", environ={})
    assert environment[RUN_ID_ENV] == "shared-run"
    assert environment[CHILD_CAPSULE_ENV] == str(target.absolute())

    monkeypatch.setenv(RUN_ID_ENV, "shared-run")
    monkeypatch.setenv(CHILD_CAPSULE_ENV, str(target))
    with CaptureSession(root=tmp_path, include_git_diff=False) as session:
        session.record_metric("loss", 1.0)

    loaded = load_child_capsule(target)
    assert session.run_id == "shared-run"
    assert loaded.run.run_id == "shared-run"
    assert loaded.run.metric("loss") is not None
