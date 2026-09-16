"""Content-addressed artifact stores.

The base package always provides :class:`LocalArtifactStore`.  Remote object
stores are supported through :class:`FsspecArtifactStore`, which imports
``fsspec`` only when it is instantiated so importing :mod:`mlforensics` keeps
its zero-dependency behavior.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO, Protocol, runtime_checkable
from urllib.parse import urlsplit

from .errors import ValidationError
from .models import ArtifactRef


def _validate_digest(sha256: str) -> str:
    if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
        raise ValidationError("sha256 must be a lowercase 64-character hex digest")
    return sha256


@runtime_checkable
class ArtifactStore(Protocol):
    """Structural interface implemented by artifact stores.

    Implementations store immutable blobs by SHA-256 digest.  Callers can use
    the protocol for dependency injection without importing any storage SDK.
    """

    def has(self, ref_or_sha256: ArtifactRef | str) -> bool: ...

    def put_bytes(
        self,
        content: bytes,
        *,
        name: str = "artifact",
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRef: ...

    def put_file(
        self,
        path: str | os.PathLike[str],
        *,
        name: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRef: ...

    def get_bytes(self, ref_or_sha256: ArtifactRef | str) -> bytes: ...

    def open(self, ref_or_sha256: ArtifactRef | str) -> BinaryIO: ...


class LocalArtifactStore:
    """Store immutable blobs below ``root`` using SHA-256 content addresses."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, sha256: str) -> Path:
        _validate_digest(sha256)
        return self.root / "sha256" / sha256[:2] / sha256

    def has(self, ref_or_sha256: ArtifactRef | str) -> bool:
        digest = ref_or_sha256.sha256 if isinstance(ref_or_sha256, ArtifactRef) else ref_or_sha256
        return bool(digest) and self.path_for(digest).is_file()

    def put_bytes(
        self,
        content: bytes,
        *,
        name: str = "artifact",
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRef:
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        digest = hashlib.sha256(content).hexdigest()
        destination = self.path_for(digest)
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".artifact-", dir=destination.parent)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(content)
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return ArtifactRef(
            name=name,
            uri=f"artifact://sha256/{digest}",
            sha256=digest,
            size_bytes=len(content),
            media_type=media_type,
            metadata=metadata or {},
        )

    def put_file(
        self,
        path: str | os.PathLike[str],
        *,
        name: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRef:
        source = Path(path)
        return self.put_bytes(
            source.read_bytes(), name=name or source.name, media_type=media_type, metadata=metadata
        )

    def get_bytes(self, ref_or_sha256: ArtifactRef | str) -> bytes:
        digest = ref_or_sha256.sha256 if isinstance(ref_or_sha256, ArtifactRef) else ref_or_sha256
        path = self.path_for(digest)
        try:
            content = path.read_bytes()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"artifact {digest!r} is not present in {self.root}") from exc
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValidationError(f"artifact {digest!r} failed integrity check")
        return content

    def open(self, ref_or_sha256: ArtifactRef | str) -> BinaryIO:
        digest = ref_or_sha256.sha256 if isinstance(ref_or_sha256, ArtifactRef) else ref_or_sha256
        return self.path_for(digest).open("rb")


class FsspecArtifactStore:
    """Content-addressed storage for ``s3://``, ``gs://``, ``az://`` and more.

    ``storage_options`` are passed directly to ``fsspec.core.url_to_fs``.  Use
    the normal provider credential chain rather than putting secrets in a
    capsule or URI. Writes stage data under a temporary key before requesting
    a backend move and clean that key on error. Atomic publication depends on
    the selected filesystem because some object stores implement moves as a
    copy followed by a delete.
    """

    def __init__(
        self,
        base_uri: str,
        *,
        storage_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(base_uri, str) or not base_uri.strip():
            raise ValueError("base_uri must be a non-empty URI")
        parsed_uri = urlsplit(base_uri)
        if parsed_uri.username is not None or parsed_uri.password is not None:
            raise ValueError(
                "base_uri must not contain credentials; use storage_options or the provider's "
                "credential chain"
            )
        try:
            from fsspec.core import url_to_fs  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "FsspecArtifactStore requires fsspec; install 'mlforensics[fsspec]' "
                "and the provider extra for your URI (s3, gcs, or azure)"
            ) from exc
        self.base_uri = base_uri.rstrip("/")
        self.storage_options = dict(storage_options or {})
        try:
            self.fs, self.root = url_to_fs(self.base_uri, **self.storage_options)
        except ImportError as exc:
            scheme = self.base_uri.partition(":")[0] or "requested"
            extra = {
                "az": "azure",
                "abfs": "azure",
                "adl": "azure",
                "gcs": "gcs",
                "gs": "gcs",
                "s3": "s3",
            }.get(scheme, "fsspec")
            raise ImportError(
                f"the optional filesystem driver for {scheme!r} is not installed; "
                f"install the matching 'mlforensics[{extra}]' extra"
            ) from exc
        self.root = self.root.rstrip("/")

    def path_for(self, sha256: str) -> str:
        _validate_digest(sha256)
        relative = f"sha256/{sha256[:2]}/{sha256}"
        return f"{self.root}/{relative}" if self.root else relative

    def uri_for(self, sha256: str) -> str:
        _validate_digest(sha256)
        return f"{self.base_uri}/sha256/{sha256[:2]}/{sha256}"

    @staticmethod
    def _digest(ref_or_sha256: ArtifactRef | str) -> str:
        digest = ref_or_sha256.sha256 if isinstance(ref_or_sha256, ArtifactRef) else ref_or_sha256
        return _validate_digest(digest)

    def has(self, ref_or_sha256: ArtifactRef | str) -> bool:
        return bool(self.fs.isfile(self.path_for(self._digest(ref_or_sha256))))

    def put_bytes(
        self,
        content: bytes,
        *,
        name: str = "artifact",
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRef:
        if not isinstance(content, bytes):
            raise TypeError("content must be bytes")
        digest = hashlib.sha256(content).hexdigest()
        destination = self.path_for(digest)
        parent = destination.rpartition("/")[0]
        if not self.fs.isfile(destination):
            self.fs.makedirs(parent, exist_ok=True)
            temporary = f"{destination}.upload-{uuid.uuid4().hex}"
            try:
                with self.fs.open(temporary, "wb") as handle:
                    handle.write(content)
                # Concurrent writers publish identical content at this digest.
                if self.fs.isfile(destination):
                    self.fs.rm(temporary)
                else:
                    self.fs.mv(temporary, destination)
            finally:
                if self.fs.exists(temporary):
                    self.fs.rm(temporary)
        return ArtifactRef(
            name=name,
            uri=self.uri_for(digest),
            sha256=digest,
            size_bytes=len(content),
            media_type=media_type,
            metadata=metadata or {},
        )

    def put_file(
        self,
        path: str | os.PathLike[str],
        *,
        name: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ArtifactRef:
        source = Path(path)
        return self.put_bytes(
            source.read_bytes(), name=name or source.name, media_type=media_type, metadata=metadata
        )

    def get_bytes(self, ref_or_sha256: ArtifactRef | str) -> bytes:
        digest = self._digest(ref_or_sha256)
        path = self.path_for(digest)
        try:
            with self.fs.open(path, "rb") as handle:
                content = handle.read()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"artifact {digest!r} is not present in {self.base_uri}"
            ) from exc
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValidationError(f"artifact {digest!r} failed integrity check")
        return content

    def open(self, ref_or_sha256: ArtifactRef | str) -> BinaryIO:
        """Open a raw binary stream; use :meth:`get_bytes` to verify integrity."""
        digest = self._digest(ref_or_sha256)
        return self.fs.open(self.path_for(digest), "rb")
