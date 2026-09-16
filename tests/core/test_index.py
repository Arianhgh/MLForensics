import multiprocessing
import sys
from pathlib import Path

from mlforensics import Run, RunCapsule
from mlforensics.core.index import load_index, record_run, resolve_run


def _index_worker(root: str, run_id: str) -> None:
    capsule = Path(root) / f"{run_id}.txt"
    capsule.write_text(run_id, encoding="utf-8")
    record_run(root, run_id, capsule, status="completed")


def test_record_and_resolve_run_ids_and_aliases(tmp_path: Path) -> None:
    capsule = tmp_path / "runs" / "demo.mlcap"
    capsule.parent.mkdir()
    capsule.write_text("capsule", encoding="utf-8")

    record_run(tmp_path, "run-1", capsule, alias="latest", status="failed")
    payload = load_index(tmp_path)
    assert payload["runs"]["run-1"]["status"] == "failed"
    assert payload["aliases"]["latest"] == "run-1"
    assert Path(payload["runs"]["run-1"]["path"]) == capsule.resolve()

    assert resolve_run(tmp_path, "run-1") == capsule.resolve()
    assert resolve_run(tmp_path, "latest") == capsule.resolve()
    assert resolve_run(tmp_path, str(capsule)) == capsule
    assert resolve_run(tmp_path, "missing") is None


def test_corrupt_index_is_preserved_and_rebuilt(tmp_path: Path) -> None:
    capsule = RunCapsule(Run(run_id="kept", status="completed", started_at="t"))
    path = capsule.save(tmp_path / "kept.mlcap")
    record_run(tmp_path, "kept", path, status="completed")
    (tmp_path / "index.json").write_text("{not-json", encoding="utf-8")
    payload = load_index(tmp_path)
    assert "kept" in payload["runs"]
    assert list(tmp_path.glob("index.json.corrupt-*"))
    assert resolve_run(tmp_path, "kept") == path.resolve()


def test_concurrent_index_writers_retain_every_run(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork") if sys.platform != "win32" else multiprocessing
    processes = [
        context.Process(target=_index_worker, args=(str(tmp_path), f"run-{index}"))
        for index in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    payload = load_index(tmp_path)
    assert set(payload["runs"]) == {f"run-{index}" for index in range(4)}
