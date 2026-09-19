import json
from pathlib import Path

import pytest

from mlforensics import (
    ArtifactRef,
    CaptureContext,
    ExecutionRecord,
    ExecutionResult,
    ExecutionSpec,
    Run,
    RunCapsule,
    ValidationError,
    dumps,
    loads,
)


def test_canonical_json_rejects_nonfinite_values_duplicate_keys_and_key_coercion() -> None:
    with pytest.raises(ValueError):
        dumps({"value": float("nan")})
    with pytest.raises(ValidationError):
        loads('{"value": NaN}')
    with pytest.raises(ValidationError):
        loads('{"value": 1, "value": 2}')
    with pytest.raises(TypeError):
        dumps({1: "would be ambiguous"})


def test_capsule_rejects_unmatched_or_wrong_sized_embedded_payload() -> None:
    ref = ArtifactRef.from_bytes("blob", b"payload")
    with pytest.raises(ValidationError, match="unexpected size"):
        RunCapsule(
            Run(run_id="size", started_at="t"),
            artifacts=(ArtifactRef(ref.name, sha256=ref.sha256, size_bytes=99),),
            payloads={ref.sha256: b"payload"},
        )
    with pytest.raises(ValidationError, match="no matching artifact"):
        RunCapsule(
            Run(run_id="extra", started_at="t"),
            payloads={"unreferenced": b"payload"},
        )


def test_recovery_journal_cannot_delete_an_outside_path(tmp_path: Path) -> None:
    target = tmp_path / "run.mlcap"
    outside = tmp_path / "keep.me"
    outside.write_text("do not remove", encoding="utf-8")
    journal = tmp_path / "run.mlcap.publish.json"
    journal.write_text(
        json.dumps({"target": str(target), "backup": str(outside), "staging": None}),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        RunCapsule.load(target)
    assert outside.read_text(encoding="utf-8") == "do not remove"
    assert not journal.exists()


def test_execution_record_has_a_strict_serialization_round_trip() -> None:
    spec = ExecutionSpec(command=["python", "-c", "pass"], timeout=1)
    result = ExecutionResult(status="ok", returncode=0)
    assert loads(dumps(spec)) == spec
    record = ExecutionRecord(
        spec=spec.to_dict(),
        result=result.to_dict(),
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:00:01Z",
    )
    assert loads(dumps(record)) == record


def test_capture_keeps_incident_when_provider_or_hook_capture_fails() -> None:
    class BrokenProvider:
        def snapshot(self):
            raise RuntimeError("provider unavailable")

    class BrokenHook:
        def on_start(self, _capture):
            raise RuntimeError("hook unavailable")

    with pytest.raises(ValueError, match="original incident"):
        with CaptureContext(
            state_providers={"model": BrokenProvider()}, hooks=(BrokenHook(),)
        ) as capture:
            raise ValueError("original incident")

    assert capture.capsule.run.failure_signature is not None
    errors = capture.capsule.evidence["capture_errors"]
    assert any("hook unavailable" in value for value in errors.values())
    state_errors = capture.capsule.evidence["replay"]["state_capture_errors"]
    assert any("provider unavailable" in value for value in state_errors.values())
