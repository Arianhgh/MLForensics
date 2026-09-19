"""Portable directory and zip run capsules.

Directory publication and recovery share an exclusive interprocess lock on
``<capsule>.lock``. Readers that encounter an in-progress journal wait for the
writer instead of treating the publication as abandoned. The lock uses
``fcntl`` and has no equivalent Windows implementation in this module.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import stat
import tempfile
import uuid
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any

from ..__about__ import __version__
from .errors import ValidationError
from .models import SCHEMA_VERSION, ArtifactRef, Run, _metadata
from .serialization import dump_bytes

_CHUNK = 1024 * 1024
_MAX_ARCHIVE_FILES = 100_000
_MAX_ARCHIVE_BYTES = 20 * 1024**3
_MAX_FILE_BYTES = _MAX_ARCHIVE_BYTES
_SCHEMA_RESOURCE = "schemas/mlcap-1.schema.json"
_PROTOCOL_RESOURCE = "schemas/mlcap-1.protocol.md"

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _strict_document(data: bytes | str) -> dict[str, Any]:
    try:
        value = json.loads(
            data,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError("invalid capsule JSON") from exc
    if not isinstance(value, dict):
        raise ValidationError("capsule JSON documents must contain an object")
    return value


def _resource_text(name: str) -> str:
    try:
        return resource_files("mlforensics.core").joinpath(name).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise FileNotFoundError(f"packaged protocol resource {name!r} is unavailable") from exc


def protocol_schema() -> dict[str, Any]:
    """Return the packaged JSON Schema for the canonical ``capsule.json`` record.

    The schema is an inspectable resource rather than a runtime dependency on a
    JSON-Schema implementation. Applications may pass the returned document to
    their validator of choice.
    """
    return _strict_document(_resource_text(_SCHEMA_RESOURCE))


def protocol_document() -> str:
    """Return the packaged human-readable ``.mlcap`` protocol document."""
    return _resource_text(_PROTOCOL_RESOURCE)


def protocol_schema_resource() -> Any:
    """Return the importlib resource for the packaged protocol schema."""
    return resource_files("mlforensics.core").joinpath(_SCHEMA_RESOURCE)


def _schema_version(data: Mapping[str, Any], *, default: int = SCHEMA_VERSION) -> int:
    value = data.get("schema_version", default)
    if isinstance(value, bool):
        raise ValidationError("schema_version must be an integer")
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int):
        raise ValidationError("schema_version must be an integer")
    return value


def _migrate_record(record: Any) -> Any:
    """Make a legacy record acceptable to the current typed record loaders."""
    if not isinstance(record, Mapping):
        return record
    result = dict(record)
    result["schema_version"] = SCHEMA_VERSION
    return result


def _migrate_run_document(run: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(run)
    result["schema_version"] = SCHEMA_VERSION
    if "failure_signature" not in result and "failure" in result:
        result["failure_signature"] = result.pop("failure")
    if "rng_state" not in result and "rng" in result:
        result["rng_state"] = result.pop("rng")
    for key in (
        "datasets",
        "models",
        "metrics",
        "resources",
        "events",
        "lineage_nodes",
        "lineage_edges",
        "observations",
    ):
        values = result.get(key)
        if isinstance(values, list):
            result[key] = [_migrate_record(item) for item in values]
    for key in ("rng_state", "failure_signature", "replay_plan"):
        if isinstance(result.get(key), Mapping):
            result[key] = _migrate_record(result[key])
    for dataset in result.get("datasets", ()):
        if isinstance(dataset, Mapping) and isinstance(dataset.get("artifact"), Mapping):
            dataset["artifact"] = _migrate_record(dataset["artifact"])
    return result


def _migrate_capsule_document(
    data: Mapping[str, Any], *, allow_migrations: bool = True
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Normalize supported pre-v1 envelopes without weakening future-version checks."""
    result = dict(data)
    raw_version = result.get("schema_version")
    version = _schema_version(result, default=SCHEMA_VERSION if raw_version is None else 0)
    if version > SCHEMA_VERSION:
        raise ValidationError(
            f"unsupported schema_version {version!r}; expected at most {SCHEMA_VERSION}"
        )
    history: list[str] = []
    if version < SCHEMA_VERSION:
        if not allow_migrations:
            raise ValidationError(f"schema_version {version} requires a migration")
        history.append(f"schema-{version}-to-{SCHEMA_VERSION}")

    if result.get("type") == "run" or ("run_id" in result and "run" not in result):
        result = {
            "schema_version": SCHEMA_VERSION,
            "type": "run_capsule",
            "run": result,
            "artifacts": result.get("artifacts", []),
        }
        history.append("legacy-run-envelope")
    elif "run" in result and result.get("type") != "run_capsule":
        result["type"] = "run_capsule"
        history.append("legacy-capsule-envelope")
    run = result.get("run")
    if not isinstance(run, Mapping):
        raise ValidationError("capsule is missing run")
    result["run"] = _migrate_run_document(run) if version < SCHEMA_VERSION else dict(run)
    if isinstance(result.get("artifacts"), list) and version < SCHEMA_VERSION:
        result["artifacts"] = [_migrate_record(item) for item in result["artifacts"]]
    result["schema_version"] = SCHEMA_VERSION
    if history:
        previous = result.get("migration_history", ())
        if isinstance(previous, list):
            history = [str(item) for item in previous] + history
        result["migration_history"] = history
    return result, tuple(history)


def _migrate_manifest(
    data: Mapping[str, Any], *, allow_migrations: bool = True
) -> tuple[dict[str, Any], tuple[str, ...]]:
    result = dict(data)
    raw_version = result.get("schema_version")
    version = _schema_version(result, default=SCHEMA_VERSION if raw_version is None else 0)
    if version > SCHEMA_VERSION:
        raise ValidationError(
            f"unsupported manifest schema_version {version!r}; expected at most {SCHEMA_VERSION}"
        )
    history: list[str] = []
    if version < SCHEMA_VERSION:
        if not allow_migrations:
            raise ValidationError(f"manifest schema_version {version} requires a migration")
        history.append(f"manifest-schema-{version}-to-{SCHEMA_VERSION}")
        if result.get("format") in (None, "mlcap", "mlcap-0"):
            result["format"] = "mlcap-1"
    result["schema_version"] = SCHEMA_VERSION
    return result, tuple(history)


def _safe_member_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return bool(
        name
        and "\\" not in name
        and "\x00" not in name
        and not Path(normalized).is_absolute()
        and ".." not in Path(normalized).parts
    )


def _signature_payload(manifest: Mapping[str, Any]) -> bytes:
    unsigned = dict(manifest)
    unsigned.pop("signature", None)
    return dump_bytes(unsigned)


def _normalize_signature(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
        if not isinstance(result.get("value"), (str, bytes, bytearray)):
            raise ValidationError("signature must contain a string or byte value")
        raw = result["value"]
        if isinstance(raw, (bytes, bytearray)):
            result["value"] = base64.b64encode(bytes(raw)).decode("ascii")
            result.setdefault("encoding", "base64")
        result.setdefault("algorithm", "custom")
        return result
    if isinstance(value, (bytes, bytearray)):
        return {
            "algorithm": "custom",
            "encoding": "base64",
            "value": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, str):
        return {"algorithm": "custom", "encoding": "utf-8", "value": value}
    raise ValidationError("signer must return bytes, text, or a signature mapping")


def _invoke_signer(signer: Any, payload: bytes) -> dict[str, Any]:
    method = getattr(signer, "sign", None)
    if not callable(method):
        method = signer if callable(signer) else None
    if method is None:
        raise TypeError("signer must be callable or expose sign(payload)")
    return _normalize_signature(method(payload))


def _invoke_verifier(verifier: Any, payload: bytes, signature: Mapping[str, Any]) -> bool:
    method = getattr(verifier, "verify", None)
    if not callable(method):
        method = verifier if callable(verifier) else None
    if method is None:
        raise TypeError("verifier must be callable or expose verify(payload, signature)")
    try:
        return bool(method(payload, signature))
    except TypeError:
        encoded = signature.get("value")
        if not isinstance(encoded, str):
            raise
        return bool(method(payload, base64.b64decode(encoded)))


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
    temporary = journal.with_name(journal.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), sort_keys=True) + "\n", encoding="utf-8")
    _fsync(temporary)
    temporary.replace(journal)
    _fsync(journal)


def _recovery_path(target: Path, value: Any, *, kind: str) -> Path | None:
    """Accept only paths generated beside this target by the publisher."""
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if candidate.is_symlink():
        return None
    try:
        parent = target.parent.resolve()
        resolved = candidate.resolve(strict=False)
    except OSError:
        return None
    if resolved.parent != parent or resolved.is_symlink():
        return None
    if kind == "backup":
        valid_name = resolved.name.startswith(target.name + ".bak.")
    else:
        valid_name = resolved.name.startswith(target.name + ".") and resolved.name.endswith(
            ".staging"
        )
    return resolved if valid_name else None


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
        if not isinstance(info, Mapping):
            raise ValueError("journal is not an object")
        recorded_target = info.get("target")
        if recorded_target is not None and Path(str(recorded_target)).resolve() != target.resolve():
            raise ValueError("journal target does not match capsule target")
        backup = _recovery_path(target, info.get("backup"), kind="backup")
        staging = _recovery_path(target, info.get("staging"), kind="staging")
        # A stale/legacy staging hint must not prevent restoration of a valid
        # backup.  Invalid staging paths are ignored and are never removed;
        # backup paths remain strict because restoring or deleting one changes
        # the published capsule.
        if info.get("backup") and backup is None:
            raise ValueError("journal contains an unsafe backup path")
    except (OSError, TypeError, ValueError):
        journal.unlink(missing_ok=True)
        return
    complete = False
    if target.exists() and not target.is_symlink():
        try:
            RunCapsule._load_from_path(target, include_artifacts=False)
            complete = True
        except Exception:
            complete = False
    if complete:
        if backup is not None and backup.is_dir() and not backup.is_symlink():
            shutil.rmtree(backup, ignore_errors=True)
        if staging is not None and staging.is_dir() and not staging.is_symlink():
            shutil.rmtree(staging, ignore_errors=True)
        journal.unlink(missing_ok=True)
        return
    if backup is not None and backup.is_dir() and not backup.is_symlink():
        if target.exists() and target.is_dir() and not target.is_symlink():
            shutil.rmtree(target, ignore_errors=True)
        backup.replace(target)
    elif target.exists() and target.is_dir() and not target.is_symlink():
        shutil.rmtree(target, ignore_errors=True)
    if staging is not None and staging.is_dir() and not staging.is_symlink():
        shutil.rmtree(staging, ignore_errors=True)
    journal.unlink(missing_ok=True)


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
    features: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    migration_history: tuple[str, ...] = field(default_factory=tuple)
    file_payloads: Mapping[str, bytes] = field(default_factory=dict, repr=False, compare=False)
    signature: Mapping[str, Any] | None = field(default=None, repr=False, compare=False)

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
            if ref.sha256 and ref.sha256 in payloads and ref.size_bytes is not None:
                if len(payloads[ref.sha256]) != ref.size_bytes:
                    raise ValidationError(f"payload for {ref.name!r} has an unexpected size")
        declared_digests = {ref.sha256 for ref in artifacts if ref.sha256}
        extra_payloads = set(payloads).difference(declared_digests)
        if extra_payloads:
            raise ValidationError("payloads contain bytes with no matching artifact reference")
        file_payloads = dict(self.file_payloads)
        if any(
            not isinstance(key, str) or not _safe_member_name(key) or not isinstance(value, bytes)
            for key, value in file_payloads.items()
        ):
            raise ValidationError("file_payloads must map safe names to bytes")
        migration_history = tuple(self.migration_history)
        if any(not isinstance(item, str) or not item.strip() for item in migration_history):
            raise ValidationError("migration_history must contain non-empty strings")
        signature = None if self.signature is None else _normalize_signature(self.signature)
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "payloads", payloads)
        object.__setattr__(self, "evidence", _metadata(self.evidence))
        object.__setattr__(self, "features", _metadata(self.features))
        object.__setattr__(self, "provenance", _metadata(self.provenance))
        object.__setattr__(self, "migration_history", migration_history)
        object.__setattr__(self, "file_payloads", file_payloads)
        object.__setattr__(self, "signature", signature)

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": SCHEMA_VERSION,
            "type": "run_capsule",
            "run": self.run.to_dict(),
            "artifacts": [item.to_dict() for item in self.artifacts],
            "evidence": dict(self.evidence),
        }
        if self.features:
            result["features"] = dict(self.features)
        if self.provenance:
            result["provenance"] = dict(self.provenance)
        if self.migration_history:
            result["migration_history"] = list(self.migration_history)
        return result

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    @staticmethod
    def protocol_schema() -> dict[str, Any]:
        """Return the packaged JSON Schema for canonical capsules."""
        return protocol_schema()

    @staticmethod
    def protocol_document() -> str:
        """Return the packaged human-readable ``.mlcap`` protocol."""
        return protocol_document()

    @staticmethod
    def protocol_schema_resource() -> Any:
        """Return the packaged schema resource for callers needing a path-like object."""
        return protocol_schema_resource()

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
        cls,
        data: Mapping[str, Any],
        *,
        payloads: Mapping[str, bytes] | None = None,
        file_payloads: Mapping[str, bytes] | None = None,
        signature: Mapping[str, Any] | None = None,
        allow_migrations: bool = True,
    ) -> RunCapsule:
        if not isinstance(data, Mapping):
            raise ValidationError("capsule must be an object")
        data, history = _migrate_capsule_document(data, allow_migrations=allow_migrations)
        run_data = data.get("run")
        if not isinstance(run_data, Mapping):
            raise ValidationError("capsule is missing run")
        previous_history = data.get("migration_history", ())
        if not isinstance(previous_history, (list, tuple)):
            raise ValidationError("migration_history must be a list")
        return cls(
            run=Run.from_dict(run_data),
            artifacts=tuple(ArtifactRef.from_dict(item) for item in data.get("artifacts", ())),
            payloads=payloads or {},
            evidence=data.get("evidence", {}),
            features=data.get("features", {}),
            provenance=data.get("provenance", {}),
            migration_history=tuple(str(item) for item in previous_history) or history,
            file_payloads=file_payloads or {},
            signature=signature,
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
            selected = next((item for item in self.artifacts if item.sha256 == ref), None)
            if selected is None:
                matching = [item for item in self.artifacts if item.name == ref]
                if len(matching) > 1:
                    raise ValidationError(f"artifact name {ref!r} is ambiguous")
                selected = matching[0] if matching else None
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

    def _feature_metadata(
        self, *, signed: bool = False, provenance: bool = False
    ) -> dict[str, Any]:
        """Return explicit and automatically detected reader feature metadata."""
        result = dict(self.features)
        flags = result.get("flags", ())
        if not isinstance(flags, (list, tuple, set)):
            raise ValidationError("features.flags must be a sequence of strings")
        normalized_flags = {str(item) for item in flags}
        normalized_flags.update({"canonical-json", "sha256-manifest"})
        if self.artifacts:
            normalized_flags.add("content-addressed-artifacts")
        if self._structured_files():
            normalized_flags.add("structured-evidence")
        if self.run.replay_plan is not None or "replay" in self.evidence:
            normalized_flags.add("replay-state")
        if provenance or self.provenance:
            normalized_flags.add("provenance")
        if signed:
            normalized_flags.add("detached-signature")
        if any(not item.strip() for item in normalized_flags):
            raise ValidationError("features.flags must contain non-empty strings")
        result["flags"] = sorted(normalized_flags)
        return result

    def save(
        self,
        path: str | os.PathLike[str],
        *,
        zipped: bool | None = None,
        overwrite: bool = False,
        signer: Any | None = None,
        provenance: Mapping[str, Any] | None = None,
        provenance_hook: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> Path:
        """Save as a directory or deterministic zip and return the resulting path.

        ``signer`` is an optional callable (or object exposing ``sign``) that
        receives the canonical unsigned manifest bytes. Its return value may be
        bytes, text, or a mapping containing a signature value. ``provenance``
        and ``provenance_hook`` add attestations to the manifest without making
        a crypto or provenance dependency mandatory.
        """
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
        if provenance is not None and not isinstance(provenance, Mapping):
            raise ValidationError("provenance must be a mapping")
        manifest_provenance = dict(self.provenance)
        if provenance:
            manifest_provenance.update(provenance)
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
        manifest["features"] = self._feature_metadata(
            signed=signer is not None, provenance=bool(manifest_provenance)
        )
        required_capabilities = manifest["features"].get("required_reader_capabilities", ())
        if not isinstance(required_capabilities, (list, tuple, set)) or any(
            not isinstance(item, str) or not item.strip() for item in required_capabilities
        ):
            raise ValidationError("features.required_reader_capabilities must contain strings")
        manifest["required_reader_capabilities"] = sorted(set(required_capabilities))
        minimum_reader = manifest["features"].get("minimum_reader_version")
        if minimum_reader is not None and (
            not isinstance(minimum_reader, str) or not minimum_reader.strip()
        ):
            raise ValidationError("features.minimum_reader_version must be a non-empty string")
        if minimum_reader is not None:
            manifest["minimum_reader_version"] = minimum_reader
        if manifest_provenance:
            manifest["provenance"] = manifest_provenance
        if provenance_hook is not None:
            if not callable(provenance_hook):
                raise TypeError("provenance_hook must be callable")
            attestation = provenance_hook(manifest)
            if not isinstance(attestation, Mapping):
                raise ValidationError("provenance_hook must return a mapping")
            manifest["provenance"] = {**manifest.get("provenance", {}), **dict(attestation)}
        if signer is not None:
            manifest["signature"] = _invoke_signer(signer, _signature_payload(manifest))
        files["manifest.json"] = dump_bytes(manifest)
        target.parent.mkdir(parents=True, exist_ok=True)
        if use_zip:
            with _capsule_publication_lock(target):
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
                    if target.is_symlink():
                        raise ValidationError("capsule target must not be a symlink")
                    _fsync(temporary)
                    temporary.replace(target)
                    _fsync(target.parent)
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise
        else:
            with _capsule_publication_lock(target):
                recover_directory_capsule(target)
                if target.is_symlink():
                    raise ValidationError("capsule target must not be a symlink")
                if target.exists() and not target.is_dir():
                    raise FileExistsError(f"capsule target is not a directory: {target}")
                if target.exists() and not overwrite:
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
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        include_artifacts: bool = True,
        include_files: Iterable[str] | None = None,
        files: Iterable[str] | None = None,
        max_file_bytes: int | None = None,
        max_total_bytes: int | None = None,
        verify: bool = True,
        verifier: Any | None = None,
        require_signature: bool = False,
        allow_migrations: bool = True,
    ) -> RunCapsule:
        """Load a capsule with optional selective retention and safety limits.

        All members are still inspected and hashed by default so a caller can
        safely retain only metadata. ``include_files`` (or its short alias
        ``files``) controls which non-core member bytes remain on the returned
        capsule as ``file_payloads``. The manifest and canonical record are
        always retained. ``verify=False`` is an explicit opt-out for trusted,
        latency-sensitive metadata reads; path and size checks still apply.
        """
        selected: set[str] | None = None
        for requested in (include_files, files):
            if requested is None:
                continue
            values = {requested} if isinstance(requested, str) else set(requested)
            if any(not isinstance(name, str) or not _safe_member_name(name) for name in values):
                raise ValidationError("selected capsule file names must be safe strings")
            if selected is None:
                selected = values
            else:
                selected.update(values)
        source = Path(path)
        journal = _directory_journal(source)
        if source.is_dir() or journal.is_file():
            with _capsule_publication_lock(source):
                recover_directory_capsule(source)
                return cls._load_from_path(
                    source,
                    include_artifacts=include_artifacts,
                    selected=selected,
                    max_file_bytes=max_file_bytes,
                    max_total_bytes=max_total_bytes,
                    verify=verify,
                    verifier=verifier,
                    require_signature=require_signature,
                    allow_migrations=allow_migrations,
                )
        if source.is_file():
            with _capsule_publication_lock(source):
                return cls._load_from_path(
                    source,
                    include_artifacts=include_artifacts,
                    selected=selected,
                    max_file_bytes=max_file_bytes,
                    max_total_bytes=max_total_bytes,
                    verify=verify,
                    verifier=verifier,
                    require_signature=require_signature,
                    allow_migrations=allow_migrations,
                )
        return cls._load_from_path(
            source,
            include_artifacts=include_artifacts,
            selected=selected,
            max_file_bytes=max_file_bytes,
            max_total_bytes=max_total_bytes,
            verify=verify,
            verifier=verifier,
            require_signature=require_signature,
            allow_migrations=allow_migrations,
        )

    @classmethod
    def _load_from_path(
        cls,
        source: Path,
        *,
        include_artifacts: bool = True,
        selected: set[str] | None = None,
        max_file_bytes: int | None = None,
        max_total_bytes: int | None = None,
        verify: bool = True,
        verifier: Any | None = None,
        require_signature: bool = False,
        allow_migrations: bool = True,
    ) -> RunCapsule:
        if source.is_symlink():
            raise ValidationError("capsule path must not be a symlink")
        for value, label in (
            (max_file_bytes, "max_file_bytes"),
            (max_total_bytes, "max_total_bytes"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValidationError(f"{label} must be a non-negative integer or None")
        max_file = _MAX_FILE_BYTES if max_file_bytes is None else max_file_bytes
        max_total = _MAX_ARCHIVE_BYTES if max_total_bytes is None else max_total_bytes
        names: set[str] = set()
        contents: dict[str, bytes] = {}
        hashes: dict[str, str] = {}
        total_seen = 0
        core_names = {"manifest.json", "capsule.json", "run.json"}

        def keep(name: str) -> bool:
            if name.startswith("artifacts/"):
                return include_artifacts and (selected is None or name in selected)
            return selected is None or name in selected or name in core_names

        def consume_stream(name: str, chunks: Iterable[bytes]) -> None:
            nonlocal total_seen
            digest = hashlib.sha256()
            retained: list[bytes] | None = [] if keep(name) else None
            size = 0
            for chunk in chunks:
                if not isinstance(chunk, bytes):
                    chunk = bytes(chunk)
                size += len(chunk)
                if size > max_file:
                    raise ValidationError(f"capsule file {name!r} exceeds the safety limit")
                total_seen += len(chunk)
                if total_seen > max_total:
                    raise ValidationError("capsule expands beyond the safety limit")
                digest.update(chunk)
                if retained is not None:
                    retained.append(chunk)
            names.add(name)
            hashes[name] = digest.hexdigest()
            if retained is not None:
                contents[name] = b"".join(retained)

        if source.is_dir():
            for item in sorted(source.rglob("*"), key=lambda value: str(value)):
                if item.is_symlink():
                    raise ValidationError("capsule contains a symlink")
                if item.is_file():
                    relative = str(item.relative_to(source))
                    if not _safe_member_name(relative):
                        raise ValidationError("capsule contains an unsafe path")
                    consume_stream(relative, _iter_file_chunks(item))
        elif source.is_file() and zipfile.is_zipfile(source):
            with zipfile.ZipFile(source) as archive:
                infos = [item for item in archive.infolist() if not item.is_dir()]
                zip_names = [item.filename for item in infos]
                if len(zip_names) != len(set(zip_names)):
                    raise ValidationError("capsule contains duplicate file names")
                if len(zip_names) > _MAX_ARCHIVE_FILES:
                    raise ValidationError("capsule contains too many files")
                declared_total = 0
                for item in infos:
                    mode = (item.external_attr >> 16) & 0xFFFF
                    if not _safe_member_name(item.filename) or stat.S_ISLNK(mode):
                        raise ValidationError("capsule contains an unsafe path")
                    if item.file_size < 0 or item.file_size > max_file:
                        raise ValidationError(
                            f"capsule file {item.filename!r} exceeds the safety limit"
                        )
                    if item.flag_bits & 0x1:
                        raise ValidationError("encrypted capsule members are unsupported")
                    declared_total += item.file_size
                if declared_total > max_total:
                    raise ValidationError("capsule expands beyond the safety limit")
                for item in infos:
                    consume_stream(item.filename, _iter_zip_chunks(archive, item.filename))
        else:
            raise FileNotFoundError(f"not an mlcap directory or zip: {source}")
        try:
            manifest = _strict_document(contents["manifest.json"])
            capsule_payload = contents.get("capsule.json")
            if capsule_payload is None:
                capsule_payload = contents["run.json"]
            capsule_data = _strict_document(capsule_payload)
        except KeyError as exc:
            raise ValidationError(f"capsule is missing {exc.args[0]}") from exc
        manifest, manifest_history = _migrate_manifest(manifest, allow_migrations=allow_migrations)
        capsule_data, capsule_history = _migrate_capsule_document(
            capsule_data, allow_migrations=allow_migrations
        )
        if manifest.get("format") != "mlcap-1":
            raise ValidationError("unsupported or invalid capsule manifest")
        if not isinstance(capsule_data, dict):  # defensive for future migration hooks
            raise ValidationError("capsule canonical document must be an object")

        signature = manifest.get("signature")
        if signature is not None:
            signature = _normalize_signature(signature)
            if verifier is not None and not _invoke_verifier(
                verifier, _signature_payload(manifest), signature
            ):
                raise ValidationError("capsule signature verification failed")
        elif require_signature:
            raise ValidationError("capsule does not contain a signature")

        if verify:
            declared = manifest.get("files", ())
            digests = manifest.get("sha256", {})
            if (
                not isinstance(declared, list)
                or any(
                    not isinstance(name, str) or not _safe_member_name(name) for name in declared
                )
                or len(declared) != len(set(declared))
                or not isinstance(digests, dict)
            ):
                raise ValidationError("capsule manifest file index is invalid")
            if any(
                not _safe_member_name(name)
                or not isinstance(expected, str)
                or len(expected) != 64
                or any(character not in "0123456789abcdef" for character in expected)
                for name, expected in digests.items()
            ):
                raise ValidationError("capsule manifest contains an invalid file entry")
            if set(declared) != set(digests):
                raise ValidationError("capsule manifest file index and digests disagree")
            actual = names.difference({"manifest.json"})
            if set(declared) != actual:
                raise ValidationError("capsule contents do not match the manifest file index")
            for name, expected in digests.items():
                if hashes.get(name) != expected:
                    raise ValidationError(f"capsule file integrity check failed for {name!r}")

        manifest_features = manifest.get("features", {})
        if manifest_features is None:
            manifest_features = {}
        if not isinstance(manifest_features, Mapping):
            raise ValidationError("capsule manifest features must be an object")
        capsule_features = capsule_data.get("features", {})
        if capsule_features is None:
            capsule_features = {}
        if not isinstance(capsule_features, Mapping):
            raise ValidationError("capsule features must be an object")
        merged_features = {**dict(manifest_features), **dict(capsule_features)}
        manifest_flags = manifest_features.get("flags", ())
        capsule_flags = capsule_features.get("flags", ())
        if isinstance(manifest_flags, (list, tuple)) and isinstance(capsule_flags, (list, tuple)):
            merged_features["flags"] = sorted({*manifest_flags, *capsule_flags})
        capsule_data["features"] = merged_features
        if "provenance" not in capsule_data and isinstance(manifest.get("provenance"), Mapping):
            capsule_data["provenance"] = dict(manifest["provenance"])
        combined_history = list(capsule_data.get("migration_history", ()))
        for item in (*manifest_history, *capsule_history):
            if item not in combined_history:
                combined_history.append(item)
        if combined_history:
            capsule_data["migration_history"] = combined_history

        payloads: dict[str, bytes] = {}
        artifacts = capsule_data.get("artifacts", ())
        if not isinstance(artifacts, list):
            raise ValidationError("capsule artifacts must be a list")
        if include_artifacts:
            for ref in artifacts:
                digest = ref.get("sha256") if isinstance(ref, Mapping) else None
                if digest and f"artifacts/{digest}" in contents:
                    payloads[digest] = contents[f"artifacts/{digest}"]
        file_payloads = {
            name: data for name, data in contents.items() if not name.startswith("artifacts/")
        }
        return cls.from_dict(
            capsule_data,
            payloads=payloads,
            file_payloads=file_payloads,
            signature=signature,
            allow_migrations=allow_migrations,
        )
