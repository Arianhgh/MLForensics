"""Portable directory and zip run capsules.

Directory publication and recovery share an exclusive interprocess lock on
``<capsule>.lock``. Readers that encounter an in-progress journal wait for the
writer instead of treating the publication as abandoned. The lock uses
``fcntl`` and has no equivalent Windows implementation in this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..__about__ import __version__
from .errors import ValidationError
from .models import SCHEMA_VERSION, ArtifactRef, Run, _metadata
from .serialization import dump_bytes, loads

_CHUNK = 1024 * 1024

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


def _directory_journal(target: Path) -> Path:
    return target.with_name(target.name + ".publish.json")


def _capsule_lock_path(target: Path) -> Path:
    return target.with_name(target.name + ".lock")


def _fsync(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        return
    finally:
        os.close(descriptor)


def _write_journal(journal: Path, payload: Mapping[str, Any]) -> None:
    journal.write_text(json.dumps(dict(payload), sort_keys=True) + "\n", encoding="utf-8")
    _fsync(journal)


@contextmanager
def _capsule_publication_lock(target: Path) -> Iterator[None]:
    """Serialize directory publication and recovery for one capsule path.

    ``fcntl.flock`` is process-owned. Windows builds skip locking; overlapping
    readers and writers are not serialized there.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = _capsule_lock_path(target).open("a+b")
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def recover_directory_capsule(target: Path) -> None:
    """Restore a capsule directory after an interrupted publication.

    Callers must hold :func:`_capsule_publication_lock`. A present journal is
    treated as abandoned only after that lock is acquired.
    """
    journal = _directory_journal(target)
    if not journal.is_file():
        return
    try:
        info = json.loads(journal.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        journal.unlink(missing_ok=True)
        return
    backup = Path(str(info["backup"])) if info.get("backup") else None
    staging = Path(str(info["staging"])) if info.get("staging") else None
    complete = False
    if target.exists():
        try:
            RunCapsule._load_from_path(target, include_artifacts=False)
            complete = True
        except Exception:
            complete = False
    if complete:
        if backup is not None and backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        journal.unlink(missing_ok=True)
        return
    if backup is not None and backup.exists():
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        backup.replace(target)
    elif target.exists():
        shutil.rmtree(target, ignore_errors=True)
    if staging is not None and staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    journal.unlink(missing_ok=True)


def _sha256_stream(chunks) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def _iter_file_chunks(path: Path):
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            yield chunk


def _iter_zip_chunks(archive: zipfile.ZipFile, name: str):
    with archive.open(name) as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            yield chunk


@dataclass(frozen=True)
class RunCapsule:
    run: Run
    artifacts: tuple[ArtifactRef, ...] = field(default_factory=tuple)
    payloads: Mapping[str, bytes] = field(default_factory=dict, repr=False, compare=False)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.run, Run):
            raise ValidationError("run must be a Run")
        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ArtifactRef) for item in artifacts):
            raise ValidationError("artifacts must contain ArtifactRef records")
        payloads = dict(self.payloads)
        if any(
            not isinstance(key, str) or not isinstance(value, bytes)
            for key, value in payloads.items()
        ):
            raise ValidationError("payloads must map strings to bytes")
        for ref in artifacts:
            if (
                ref.sha256
                and ref.sha256 in payloads
                and hashlib.sha256(payloads[ref.sha256]).hexdigest() != ref.sha256
            ):
                raise ValidationError(f"payload for {ref.name!r} does not match its sha256")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "payloads", payloads)
        object.__setattr__(self, "evidence", _metadata(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "run_capsule",
            "run": self.run.to_dict(),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "evidence": dict(self.evidence),
        }

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    @property
    def run_id(self) -> str:
        return self.run.run_id

    @property
    def digest(self) -> str:
        """Return a stable digest for the manifest-independent capsule content."""
        hasher = hashlib.sha256(dump_bytes(self))
        for digest, payload in sorted(self.payloads.items()):
            hasher.update(digest.encode("ascii", "strict"))
            hasher.update(payload)
        return hasher.hexdigest()

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, payloads: Mapping[str, bytes] | None = None
    ) -> RunCapsule:
        if data.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValidationError(
                f"unsupported schema_version {data.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION}"
            )
        run_data = data.get("run")
        if not isinstance(run_data, Mapping):
            raise ValidationError("capsule is missing run")
        return cls(
            run=Run.from_dict(run_data),
            artifacts=tuple(ArtifactRef.from_dict(item) for item in data.get("artifacts", ())),
            payloads=payloads or {},
            evidence=data.get("evidence", {}),
        )

    def artifact_payload(self, ref: ArtifactRef | str) -> bytes:
        """Return a verified embedded artifact payload.

        ``ref`` may be an artifact record, digest, or unique artifact name.  A
        capsule never follows the artifact URI here: replay must not silently
        read mutable external state when the immutable payload is unavailable.
        """
        if isinstance(ref, ArtifactRef):
            selected = ref
        else:
            selected = next(
                (item for item in self.artifacts if item.sha256 == ref or item.name == ref),
                None,
            )
            if selected is None:
                raise KeyError(ref)
        try:
            payload = self.payloads[selected.sha256]
        except KeyError as exc:
            raise FileNotFoundError(
                f"artifact {selected.name!r} is referenced but not embedded in this capsule"
            ) from exc
        if hashlib.sha256(payload).hexdigest() != selected.sha256:
            raise ValidationError(f"payload for {selected.name!r} failed integrity validation")
        return payload

    def artifacts_with_role(self, role: str) -> tuple[ArtifactRef, ...]:
        """Select artifacts by their portable ``metadata['role']`` value."""
        return tuple(item for item in self.artifacts if item.metadata.get("role") == role)

    def _fingerprints(self) -> list[dict[str, Any]]:
        """Return one identifying record per dataset the run registered."""
        records = []
        for dataset in self.run.datasets:
            metadata = dict(dataset.metadata)
            records.append(
                {
                    "name": dataset.name,
                    "split": dataset.split,
                    "format": dataset.format,
                    "sha256": getattr(dataset.artifact, "sha256", None) or metadata.get("sha256"),
                    "uri": getattr(dataset.artifact, "uri", None) or metadata.get("uri"),
                    "sample_count": metadata.get("sample_count", metadata.get("row_count")),
                }
            )
        return records

    def _schemas(self) -> dict[str, Any]:
        """Return the recorded schema of each dataset, keyed by dataset name."""
        schemas = {}
        for dataset in self.run.datasets:
            schema = dict(dataset.metadata).get("schema")
            if schema:
                schemas[dataset.name] = schema
        return schemas

    def _structured_files(self) -> dict[str, bytes]:
        """Materialize stable, human-inspectable views of canonical evidence."""
        evidence = dict(self.evidence)
        metadata = dict(self.run.metadata)
        files: dict[str, bytes] = {}

        def add(path: str, value: Any) -> None:
            if value not in (None, {}, [], ()):
                files[path] = dump_bytes(value)

        add("code.json", evidence.get("code", metadata.get("git")))
        add("environment.json", evidence.get("environment", metadata.get("environment")))
        add("hardware.json", evidence.get("hardware", metadata.get("hardware")))
        add("dependencies.json", evidence.get("dependencies", metadata.get("dependencies")))
        data_evidence = evidence.get("data") if isinstance(evidence.get("data"), Mapping) else {}
        # Registered datasets already carry their digest and schema, so derive
        # the documented views from them rather than requiring every caller to
        # restate the same evidence under ``evidence["data"]``.
        add("data/fingerprints.json", data_evidence.get("fingerprints") or self._fingerprints())
        add("data/schema.json", data_evidence.get("schema") or self._schemas())
        add("training/metrics.json", [item.to_dict() for item in self.run.metrics])
        add("training/observations.json", [item.to_dict() for item in self.run.observations])
        if self.run.events:
            files["training/events.jsonl"] = b"".join(
                dump_bytes(item) + b"\n" for item in self.run.events
            )
        add("model/signature.json", [item.to_dict() for item in self.run.models])
        add("system/resource_trace.json", [item.to_dict() for item in self.run.resources])
        if self.run.failure_signature is not None:
            add("failure/exception.json", self.run.failure_signature.to_dict())
        trace = evidence.get("tensor_trace")
        if trace is not None:
            add("failure/tensor_trace.json", trace)
        replay = evidence.get("replay")
        if isinstance(replay, Mapping):
            add("replay/inputs.json", replay.get("input"))
            add("replay/state.json", replay.get("state"))
            add("replay/checkpoints.json", replay.get("checkpoints"))
            add("replay/last_checkpoint.json", replay.get("last_checkpoint"))
        if self.run.replay_plan is not None:
            add("replay/plan.json", self.run.replay_plan.to_dict())
        rng = self.run.rng_state
        if rng is not None:
            if rng.python is not None:
                files["randomness/python.rng"] = rng.python.encode("utf-8")
            if rng.numpy is not None:
                files["randomness/numpy.rng"] = rng.numpy.encode("utf-8")
            for name, value in sorted(rng.frameworks.items()):
                safe_name = "".join(
                    character if character.isalnum() or character in "._-" else "_"
                    for character in name
                )
                files[f"randomness/{safe_name}.rng"] = value.encode("utf-8")
        return files

    def save(
        self, path: str | os.PathLike[str], *, zipped: bool | None = None, overwrite: bool = False
    ) -> Path:
        """Save as a directory or deterministic zip and return the resulting path."""
        target = Path(path)
        use_zip = (target.suffix == ".zip") if zipped is None else zipped
        files: dict[str, bytes] = {
            "manifest.json": b"",
            "run.json": dump_bytes(self.run),
            "capsule.json": dump_bytes(self),
        }
        files.update(self._structured_files())
        for ref in self.artifacts:
            if ref.sha256 in self.payloads:
                files[f"artifacts/{ref.sha256}"] = self.payloads[ref.sha256]
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "format": "mlcap-1",
            "producer": {"name": "mlforensics", "version": __version__},
            "files": sorted(name for name in files if name != "manifest.json"),
            "sha256": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in files.items()
                if name != "manifest.json"
            },
        }
        files["manifest.json"] = dump_bytes(manifest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if use_zip:
            handle, temporary_name = tempfile.mkstemp(
                prefix=target.name + ".", suffix=".tmp", dir=target.parent
            )
            os.close(handle)
            temporary = Path(temporary_name)
            try:
                with zipfile.ZipFile(
                    temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
                ) as archive:
                    for name in sorted(files):
                        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                        info.compress_type = zipfile.ZIP_DEFLATED
                        info.external_attr = 0o644 << 16
                        archive.writestr(info, files[name])
                if target.exists() and not overwrite:
                    temporary.unlink(missing_ok=True)
                    raise FileExistsError(target)
                _fsync(temporary)
                temporary.replace(target)
                _fsync(target.parent)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        else:
            with _capsule_publication_lock(target):
                recover_directory_capsule(target)
                if target.exists() and not target.is_dir():
                    raise FileExistsError(f"capsule target is not a directory: {target}")
                if target.exists() and not overwrite and any(target.iterdir()):
                    raise FileExistsError(target)
                staging = Path(
                    tempfile.mkdtemp(prefix=target.name + ".", suffix=".staging", dir=target.parent)
                )
                backup: Path | None = None
                journal = _directory_journal(target)
                try:
                    for name, content in files.items():
                        destination = staging / name
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(content)
                        _fsync(destination)
                    _fsync(staging)
                    payload = {
                        "target": str(target),
                        "backup": None,
                        "staging": str(staging),
                        "validated": False,
                        "pid": os.getpid(),
                    }
                    if target.exists():
                        backup = target.with_name(target.name + ".bak." + uuid.uuid4().hex)
                        payload["backup"] = str(backup)
                        _write_journal(journal, payload)
                        target.rename(backup)
                    else:
                        _write_journal(journal, payload)
                    staging.replace(target)
                    _fsync(target.parent)
                    try:
                        type(self)._load_from_path(target, include_artifacts=True)
                    except Exception as exc:
                        recover_directory_capsule(target)
                        raise ValidationError("published capsule failed integrity checks") from exc
                    payload["validated"] = True
                    _write_journal(journal, payload)
                    if backup is not None and backup.exists():
                        shutil.rmtree(backup)
                    journal.unlink(missing_ok=True)
                except Exception:
                    recover_directory_capsule(target)
                    if staging.exists():
                        shutil.rmtree(staging, ignore_errors=True)
                    raise
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, include_artifacts: bool = True) -> RunCapsule:
        source = Path(path)
        journal = _directory_journal(source)
        if source.is_dir() or journal.is_file():
            with _capsule_publication_lock(source):
                recover_directory_capsule(source)
                return cls._load_from_path(source, include_artifacts=include_artifacts)
        return cls._load_from_path(source, include_artifacts=include_artifacts)

    @classmethod
    def _load_from_path(cls, source: Path, *, include_artifacts: bool = True) -> RunCapsule:
        names: set[str] = set()
        contents: dict[str, bytes] = {}
        hashes: dict[str, str] = {}

        def keep(name: str) -> bool:
            return include_artifacts or not name.startswith("artifacts/")

        def consume(name: str, data: bytes) -> None:
            names.add(name)
            hashes[name] = hashlib.sha256(data).hexdigest()
            if keep(name):
                contents[name] = data

        def consume_stream(name: str, chunks) -> None:
            if keep(name):
                consume(name, b"".join(chunks))
                return
            names.add(name)
            hashes[name] = _sha256_stream(chunks)

        if source.is_dir():
            for item in source.rglob("*"):
                if item.is_file() and not item.is_symlink():
                    relative = str(item.relative_to(source))
                    if keep(relative):
                        consume(relative, item.read_bytes())
                    else:
                        consume_stream(relative, _iter_file_chunks(item))
        elif source.is_file() and zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                zip_names = [name for name in archive.namelist() if not name.endswith("/")]
                if any(Path(name).is_absolute() or ".." in Path(name).parts for name in zip_names):
                    raise ValidationError("capsule contains an unsafe path")
                if len(zip_names) != len(set(zip_names)):
                    raise ValidationError("capsule contains duplicate file names")
                if len(zip_names) > 100_000:
                    raise ValidationError("capsule contains too many files")
                if sum(item.file_size for item in archive.infolist()) > 20 * 1024**3:
                    raise ValidationError("capsule expands beyond the safety limit")
                for name in zip_names:
                    if keep(name):
                        consume(name, archive.read(name))
                    else:
                        consume_stream(name, _iter_zip_chunks(archive, name))
        else:
            raise FileNotFoundError(f"not an mlcap directory or zip: {source}")
        try:
            manifest = loads(contents["manifest.json"])
            capsule_data = loads(contents.get("capsule.json", contents["run.json"]))
        except KeyError as exc:
            raise ValidationError(f"capsule is missing {exc.args[0]}") from exc
        if not isinstance(manifest, dict) or manifest.get("format") != "mlcap-1":
            raise ValidationError("unsupported or invalid capsule manifest")
        if isinstance(capsule_data, RunCapsule):
            capsule_data = capsule_data.to_dict()
        elif isinstance(capsule_data, Run):
            capsule_data = {
                "schema_version": SCHEMA_VERSION,
                "type": "run_capsule",
                "run": capsule_data.to_dict(),
                "artifacts": [],
            }
        declared = manifest.get("files", ())
        digests = manifest.get("sha256", {})
        if not isinstance(declared, list) or not isinstance(digests, dict):
            raise ValidationError("capsule manifest file index is invalid")
        if set(declared) != set(digests):
            raise ValidationError("capsule manifest file index and digests disagree")
        actual = names.difference({"manifest.json"})
        if set(declared) != actual:
            raise ValidationError("capsule contents do not match the manifest file index")
        for name, expected in digests.items():
            if hashes.get(name) != expected:
                raise ValidationError(f"capsule file integrity check failed for {name!r}")
        payloads = {}
        if include_artifacts and isinstance(capsule_data, dict):
            for ref in capsule_data.get("artifacts", ()):
                digest = ref.get("sha256") if isinstance(ref, dict) else None
                if digest and f"artifacts/{digest}" in contents:
                    payloads[digest] = contents[f"artifacts/{digest}"]
        if isinstance(capsule_data, dict) and capsule_data.get("type") == "run":
            capsule_data = {
                "schema_version": SCHEMA_VERSION,
                "type": "run_capsule",
                "run": capsule_data,
                "artifacts": [],
            }
        return cls.from_dict(capsule_data, payloads=payloads)
