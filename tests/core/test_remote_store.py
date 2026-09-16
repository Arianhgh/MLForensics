import hashlib
import importlib.util

import pytest

from mlforensics import ArtifactStore, FsspecArtifactStore, ValidationError

# The base installation intentionally omits fsspec, so only the tests that actually
# reach a filesystem need the extra. URI validation happens before the fsspec import
# and therefore stays covered in a minimal install.
requires_fsspec = pytest.mark.skipif(
    importlib.util.find_spec("fsspec") is None,
    reason="FsspecArtifactStore requires the optional 'fsspec' extra",
)


@requires_fsspec
def test_fsspec_memory_store_round_trip_and_integrity(tmp_path) -> None:
    store = FsspecArtifactStore(f"memory://mlforensics-tests/{tmp_path.name}")
    ref = store.put_bytes(
        b"remote evidence",
        name="evidence.bin",
        media_type="application/octet-stream",
        metadata={"role": "evidence"},
    )

    assert isinstance(store, ArtifactStore)
    assert ref.sha256 == hashlib.sha256(b"remote evidence").hexdigest()
    assert ref.uri.endswith(f"/sha256/{ref.sha256[:2]}/{ref.sha256}")
    assert store.has(ref)
    assert store.get_bytes(ref) == b"remote evidence"
    with store.open(ref) as handle:
        assert handle.read() == b"remote evidence"

    store.fs.pipe(store.path_for(ref.sha256), b"tampered")
    with pytest.raises(ValidationError, match="integrity"):
        store.get_bytes(ref)


@requires_fsspec
def test_fsspec_store_cleans_temporary_upload_after_publish_failure(tmp_path, monkeypatch) -> None:
    store = FsspecArtifactStore(f"memory://mlforensics-tests/{tmp_path.name}-failure")
    digest = hashlib.sha256(b"not published").hexdigest()

    def fail_move(source, destination):
        raise OSError("publish failed")

    monkeypatch.setattr(store.fs, "mv", fail_move)
    with pytest.raises(OSError, match="publish failed"):
        store.put_bytes(b"not published")

    assert not store.fs.exists(store.path_for(digest))
    assert not any(".upload-" in path for path in store.fs.find(store.root))


@requires_fsspec
def test_fsspec_store_rejects_invalid_digest(tmp_path) -> None:
    store = FsspecArtifactStore(f"memory://mlforensics-tests/{tmp_path.name}-digest")
    with pytest.raises(ValidationError, match="sha256"):
        store.get_bytes("../unsafe")


def test_fsspec_store_rejects_credentials_in_uri() -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        FsspecArtifactStore("s3://access:secret@example-bucket/artifacts")
