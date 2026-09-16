import hashlib

import pytest

from mlforensics import (
    ArtifactRef,
    CaptureContext,
    LocalArtifactStore,
    MetricSeries,
    ValidationError,
    loads,
)


def test_contract_validation_rejects_bad_values() -> None:
    with pytest.raises(ValidationError):
        ArtifactRef("bad", sha256="not-a-digest")
    with pytest.raises(ValidationError):
        MetricSeries("loss", values=(float("nan"),))
    with pytest.raises(ValidationError):
        MetricSeries("loss", values=(1.0,), steps=(0, 1))
    with pytest.raises(ValidationError):
        loads('{"schema_version": 99, "type": "run"}')


def test_local_artifact_store_is_content_addressed_and_verifies_integrity(tmp_path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    ref = store.put_bytes(b"hello", name="greeting.txt", media_type="text/plain")
    assert ref.sha256 == hashlib.sha256(b"hello").hexdigest()
    assert store.has(ref)
    assert store.get_bytes(ref) == b"hello"
    assert store.put_bytes(b"hello", name="renamed.txt").sha256 == ref.sha256

    store.path_for(ref.sha256).write_bytes(b"tampered")
    with pytest.raises(ValidationError):
        store.get_bytes(ref)


def test_capture_context_records_events_artifacts_and_failure() -> None:
    with CaptureContext(name="capture", run_id="capture-1") as capture:
        capture.event("start", step=0)
        capture.metric(MetricSeries("accuracy", values=(0.5, 0.75)))
        artifact = capture.artifact(b"evidence", name="evidence.txt")
    assert capture.capsule.run.status == "completed"
    assert capture.capsule.run.events[0].kind == "start"
    assert capture.capsule.payloads[artifact.sha256] == b"evidence"

    with pytest.raises(RuntimeError):
        with CaptureContext(name="broken") as broken:
            raise RuntimeError("boom 42")
    assert broken.capsule.run.status == "failed"
    assert broken.capsule.run.failure_signature is not None
    assert broken.capsule.run.failure_signature.normalized_message == "boom <number>"
