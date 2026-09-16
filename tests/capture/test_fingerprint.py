import shutil
from pathlib import Path

from mlforensics.capture import fingerprint_dataset, fingerprint_directory, fingerprint_file


def test_file_fingerprint_is_content_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "one.txt"
    source.write_text("same bytes\n", encoding="utf-8")
    first = fingerprint_file(source)
    source.touch()
    second = fingerprint_file(source)

    assert first["sha256"] == second["sha256"]
    assert first["size"] == len("same bytes\n")
    assert first["kind"] == "file"


def test_directory_fingerprint_is_sorted_and_copy_stable(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "nested").mkdir(parents=True)
    (source / "z.txt").write_text("z", encoding="utf-8")
    (source / "nested" / "a.txt").write_text("a", encoding="utf-8")
    copied = tmp_path / "copied"
    shutil.copytree(source, copied)

    original = fingerprint_directory(source)
    duplicate = fingerprint_directory(copied)
    assert original["digest"] == duplicate["digest"]
    assert [entry["path"] for entry in original["files"]] == ["nested/a.txt", "z.txt"]


def test_dataset_collection_order_does_not_change_digest(tmp_path: Path) -> None:
    left = tmp_path / "left.txt"
    right = tmp_path / "right.txt"
    left.write_text("left", encoding="utf-8")
    right.write_text("right", encoding="utf-8")

    first = fingerprint_dataset([left, right])
    second = fingerprint_dataset([right, left])
    assert first["digest"] == second["digest"]
