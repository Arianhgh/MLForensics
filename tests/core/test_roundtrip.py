import json
import multiprocessing
import shutil
import sys
import time
import zipfile
from pathlib import Path

import pytest

from mlforensics import (
    ArtifactRef,
    FailureSignature,
    LineageEdge,
    LineageNode,
    MetricSeries,
    Run,
    RunCapsule,
    TraceEvent,
    ValidationError,
    dumps,
    loads,
)


def test_run_json_is_deterministic_and_round_trips() -> None:
    run = Run(
        run_id="run-1",
        name="demo",
        status="completed",
        started_at="2026-01-01T00:00:00Z",
        ended_at="2026-01-01T00:01:00Z",
        metadata={"z": 2, "a": 1},
        metrics=(MetricSeries("loss", values=(1.0, 0.5), steps=(0, 1)),),
        events=(TraceEvent("checkpoint", message="saved", timestamp="2026-01-01T00:00:30Z"),),
        lineage_nodes=(LineageNode("dataset", "dataset"), LineageNode("model", "model")),
        lineage_edges=(LineageEdge("dataset", "model", "used_by"),),
    )
    encoded = dumps(run)
    assert encoded == dumps(run)
    assert '"a":1' in encoded
    assert loads(encoded) == run
    assert loads(encoded, Run) == run


def test_directory_and_zip_capsule_round_trip(tmp_path: Path) -> None:
    blob = b"weights"
    ref = ArtifactRef.from_bytes("weights.bin", blob)
    run = Run(run_id="run-2", name="portable", status="completed", started_at="t")
    capsule = RunCapsule(run, artifacts=(ref,), payloads={ref.sha256: blob})

    directory = capsule.save(tmp_path / "portable.mlcap")
    loaded_directory = RunCapsule.load(directory)
    assert loaded_directory.run == run
    assert loaded_directory.payloads[ref.sha256] == blob

    archive = capsule.save(tmp_path / "portable.mlcap.zip")
    loaded_archive = RunCapsule.load(archive)
    assert loaded_archive.run == run
    assert loaded_archive.payloads == {ref.sha256: blob}
    assert archive.read_bytes() == capsule.save(tmp_path / "portable-again.mlcap.zip").read_bytes()


def test_failure_signature_is_stable_for_volatile_values() -> None:
    def fail(value: int) -> FailureSignature:
        try:
            raise ValueError(f"record {value} failed")
        except ValueError as exc:
            return FailureSignature.from_exception(exc, traceback_text="stable traceback")

    first = fail(101)
    second = fail(202)
    assert first.normalized_message == second.normalized_message == "record <number> failed"
    assert first.traceback_hash == second.traceback_hash
    assert first.grouping_key() == second.grouping_key()
    assert loads(dumps(first)) == first


def test_capsule_rejects_unmanifested_files(tmp_path: Path) -> None:
    capsule = RunCapsule(Run(run_id="run-manifest", status="completed", started_at="t"))
    directory = capsule.save(tmp_path / "manifest.mlcap")
    (directory / "untrusted.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValidationError, match="manifest file index"):
        RunCapsule.load(directory)


def test_zip_save_is_atomic_and_preserves_the_original(tmp_path: Path) -> None:
    target = tmp_path / "kept.mlcap.zip"
    target.write_bytes(b"original")
    capsule = RunCapsule(Run(run_id="atomic", status="completed", started_at="t"))
    with pytest.raises(FileExistsError):
        capsule.save(target)
    assert target.read_bytes() == b"original"
    assert not list(tmp_path.glob("*.tmp"))
    capsule.save(target, overwrite=True)
    loaded = RunCapsule.load(target)
    assert loaded.run.run_id == "atomic"
    with zipfile.ZipFile(target) as archive:
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["producer"]["name"] == "mlforensics"
    assert manifest["producer"]["version"]


def test_metadata_only_load_skips_artifact_payloads(tmp_path: Path) -> None:
    blob = b"weights-bytes"
    ref = ArtifactRef.from_bytes("weights.bin", blob)
    capsule = RunCapsule(
        Run(run_id="payloads", status="completed", started_at="t"),
        artifacts=(ref,),
        payloads={ref.sha256: blob},
    )
    directory = capsule.save(tmp_path / "payloads.mlcap")
    archive = capsule.save(tmp_path / "payloads.mlcap.zip")
    for source in (directory, archive):
        metadata = RunCapsule.load(source, include_artifacts=False)
        assert metadata.run.run_id == "payloads"
        assert metadata.payloads == {}
        full = RunCapsule.load(source)
        assert full.payloads[ref.sha256] == blob


def test_interrupted_directory_replacement_restores_the_previous_capsule(tmp_path: Path) -> None:
    capsule = RunCapsule(Run(run_id="old", status="completed", started_at="t"))
    target = capsule.save(tmp_path / "dir.mlcap")
    backup = tmp_path / "dir.mlcap.bak.interrupted"
    target.rename(backup)
    journal = tmp_path / "dir.mlcap.publish.json"
    journal.write_text(
        json.dumps(
            {
                "target": str(target),
                "backup": str(backup),
                "staging": str(tmp_path / "gone"),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = RunCapsule.load(target)
    assert loaded.run.run_id == "old"
    assert not journal.exists()
    assert not backup.exists()


def _hold_in_progress_publication(
    root: str, started: object, proceed: object, staging_ok: object
) -> None:
    from mlforensics.core.capsule import _capsule_publication_lock, _directory_journal

    target = Path(root) / "dir.mlcap"
    with _capsule_publication_lock(target):
        backup = target.with_name(target.name + ".bak.hold")
        staging = Path(root) / "dir.mlcap.hold.staging"
        staging.mkdir()
        (staging / "placeholder").write_text("in-progress", encoding="utf-8")
        target.rename(backup)
        _directory_journal(target).write_text(
            json.dumps(
                {
                    "target": str(target),
                    "backup": str(backup),
                    "staging": str(staging),
                    "validated": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        started.set()
        proceed.wait(timeout=15)
        staging_ok.put(staging.exists())
        if backup.exists() and not target.exists():
            backup.replace(target)
        shutil.rmtree(staging, ignore_errors=True)
        _directory_journal(target).unlink(missing_ok=True)
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def _load_capsule_run_id(root: str, result: object) -> None:
    loaded = RunCapsule.load(Path(root) / "dir.mlcap")
    result.put(loaded.run.run_id)


def test_reader_waits_for_active_publication_and_does_not_delete_staging(tmp_path: Path) -> None:
    RunCapsule(Run(run_id="old", status="completed", started_at="t")).save(tmp_path / "dir.mlcap")
    context = multiprocessing.get_context("fork") if sys.platform != "win32" else multiprocessing
    started = context.Event()
    proceed = context.Event()
    staging_ok = context.Queue()
    loaded_id = context.Queue()
    holder = context.Process(
        target=_hold_in_progress_publication,
        args=(str(tmp_path), started, proceed, staging_ok),
    )
    reader = context.Process(target=_load_capsule_run_id, args=(str(tmp_path), loaded_id))
    holder.start()
    assert started.wait(timeout=10)
    reader.start()
    time.sleep(0.4)
    assert reader.is_alive()
    proceed.set()
    reader.join(timeout=10)
    holder.join(timeout=10)
    assert holder.exitcode == 0
    assert reader.exitcode == 0
    assert staging_ok.get(timeout=5) is True
    assert loaded_id.get(timeout=5) == "old"
