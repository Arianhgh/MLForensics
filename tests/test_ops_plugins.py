from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlforensics.core import Run, RunCapsule
from mlforensics.ops import (
    AuditLogger,
    LegalHoldRegistry,
    OfflineQueue,
    RemoteOperations,
    RetentionPolicy,
    SidecarLegalHold,
    garbage_collect,
    iter_capsules,
    publish_remote,
    redact_mapping,
    redact_text,
    redact_value,
    restore_from_trash,
    scan_sensitive_data,
    sha256_bytes,
)
from mlforensics.plugins import (
    PLUGIN_PROTOCOL_VERSION,
    PluginCompatibilityError,
    PluginDependencyError,
    PluginLoadError,
    PluginMetadata,
    check_optional_dependencies,
    discover_plugins,
    find_plugin,
    load_plugin,
    negotiate_plugin,
    require_optional_dependencies,
)


def test_redaction_and_sensitive_scanning_cover_nested_values_and_files(tmp_path: Path) -> None:
    value = {
        "password": "do-not-store",
        "nested": ["Bearer abc123", ("postgres://alice:secret@db",), {"token": "hidden"}],
        "safe": {"set-value"},
    }
    redacted = redact_value(value)
    assert redacted["password"] == "<redacted>"
    assert "<redacted>" in redacted["nested"][0]
    assert "<redacted>" in redacted["nested"][1][0]
    assert redacted["nested"][2]["token"] == "<redacted>"
    assert isinstance(redacted["safe"], list)
    assert redact_mapping({"api_key": "secret"})["api_key"] == "<redacted>"
    assert "<redacted>" in redact_text("Authorization: Bearer abc")
    assert "<redacted>" in redact_text("secret=abc", patterns=[r"secret=([^ ]+)"])
    with pytest.raises(TypeError):
        redact_text(1)  # type: ignore[arg-type]

    report = scan_sensitive_data(
        {"password": "hidden", "message": "Bearer abc", "blob": b"token=xyz"}
    )
    assert report.has_findings and not report.clean
    assert all(item.value == "<redacted>" for item in report.findings)
    assert report.to_dict()["findings"]

    secret_file = tmp_path / "config.json"
    secret_file.write_text('{"client_secret":"hidden", "note":"Bearer abc"}', encoding="utf-8")
    clean_file = tmp_path / "clean.txt"
    clean_file.write_text("ordinary", encoding="utf-8")
    file_report = scan_sensitive_data(tmp_path, max_files=2)
    assert str(secret_file) in file_report.scanned_files
    assert file_report.findings
    assert scan_sensitive_data(clean_file).clean
    symlink = tmp_path / "link"
    symlink.symlink_to(secret_file)
    assert scan_sensitive_data(symlink).errors
    assert scan_sensitive_data(b"token=abc").findings
    assert scan_sensitive_data("api_key=abc").findings
    with pytest.raises(ValueError):
        scan_sensitive_data("x", max_bytes=0)


def test_audit_holds_and_recoverable_retention(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.jsonl"
    fixed_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    logger = AuditLogger(audit_path, actor="tester", clock=lambda: fixed_time, fsync=False)
    event = logger.record(
        "capture",
        tmp_path / "run.mlcap",
        details={"token": "must not be written"},
        error=ValueError("bad secret"),
    )
    assert event.actor == "tester"
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    encoded = json.loads(lines[0])
    assert encoded["details"]["token"] == "<redacted>"
    assert "bad secret" in encoded["error"]
    with pytest.raises(ValueError):
        logger.record("", "target")

    hold = LegalHoldRegistry()
    target = tmp_path / "nested" / "run.mlcap"
    target.parent.mkdir()
    hold.add(tmp_path / "nested")
    assert target in hold
    hold.remove(tmp_path / "nested")
    assert target not in hold
    hold.add(target)
    hold_path = hold.save(tmp_path / "holds.json")
    assert LegalHoldRegistry.load(hold_path).is_held(target)
    sidecar = SidecarLegalHold()
    marker = sidecar.hold(target, reason="legal review")
    assert marker.read_text(encoding="utf-8") == "legal review"
    assert sidecar.is_held(target)
    sidecar.release(target)
    assert not sidecar.is_held(target)
    with pytest.raises(ValueError):
        SidecarLegalHold("../unsafe")

    root = tmp_path / "capsules"
    old = RunCapsule(Run(run_id="old", status="completed", started_at="t")).save(root / "old.mlcap")
    newest = RunCapsule(Run(run_id="new", status="completed", started_at="t")).save(
        root / "new.mlcap"
    )
    old_time = 100.0
    os.utime(old, (old_time, old_time))
    os.utime(newest, (200.0, 200.0))
    assert set(iter_capsules(root)) == {old, newest}
    planned = garbage_collect(
        root,
        policy=RetentionPolicy(max_age=50, keep_recent=1),
        now=300.0,
        held=LegalHoldRegistry([old]),
        dry_run=True,
        audit_log=logger,
    )
    assert not planned.deleted and planned.skipped[str(old)] == "legal_hold"

    # Remove the hold and execute a recoverable move.  The newest capsule is
    # retained while the old one is moved outside the scanned root.
    hold.remove(old)
    collected = garbage_collect(
        root,
        policy=RetentionPolicy(max_age=50, keep_recent=1),
        now=300.0,
        dry_run=False,
        trash_dir=tmp_path / "trash",
        audit_log=logger,
    )
    assert collected.deleted == (old,)
    moved = next((path for path in (tmp_path / "trash").iterdir()), None)
    assert moved is not None
    restored = restore_from_trash(moved, old)
    assert restored == old and old.is_dir()
    with pytest.raises(FileNotFoundError):
        restore_from_trash(moved, old)

    with pytest.raises(ValueError):
        RetentionPolicy(max_age=-1)
    with pytest.raises(ValueError):
        garbage_collect(tmp_path / "missing")


class _Transport:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.fail = False

    def put_bytes(self, key: str, payload: bytes) -> None:
        if self.fail:
            raise OSError("offline")
        self.values[key] = payload

    def get_bytes(self, key: str) -> bytes:
        return self.values[key]

    def delete(self, key: str) -> None:
        self.values.pop(key, None)
        self.deleted.append(key)


def test_remote_operations_are_durable_and_explicitly_offline(tmp_path: Path) -> None:
    queue = OfflineQueue(tmp_path / "queue.jsonl")
    audit = AuditLogger(tmp_path / "remote-audit.jsonl", fsync=False)
    offline = RemoteOperations(offline=True, queue=queue, audit_log=audit)
    put = offline.put("capsule.bin", b"evidence")
    delete = offline.delete("old.bin")
    assert put.key == "capsule.bin" and delete.action == "delete"
    assert len(queue.pending()) == 2
    with pytest.raises(RuntimeError):
        offline.get("capsule.bin")

    transport = _Transport()
    assert queue.flush(transport, max_items=1)
    assert transport.values["capsule.bin"] == b"evidence"
    assert len(queue.pending()) == 1
    assert not queue.flush(transport)
    assert transport.deleted == ["old.bin"]

    online = RemoteOperations(transport, audit_log=audit)
    assert online.put("live.bin", b"live") is None
    assert online.get("live.bin") == b"live"
    online.delete("live.bin")
    assert "live.bin" not in transport.values
    assert publish_remote("queued.bin", b"queued", offline=True).action == "put"
    assert sha256_bytes(b"evidence")
    with pytest.raises(ValueError):
        online.put("../unsafe", b"x")
    with pytest.raises(TypeError):
        sha256_bytes("not-bytes")  # type: ignore[arg-type]

    transport.fail = True
    pending = OfflineQueue()
    pending.enqueue("put", "retry.bin", b"retry")
    assert pending.flush(transport)


class _EntryPoint:
    group = "mlforensics.plugins"

    def __init__(
        self,
        name: str,
        *,
        protocol_version: str = PLUGIN_PROTOCOL_VERSION,
        capabilities: tuple[str, ...] = ("capture",),
        dependencies: tuple[str, ...] = (),
        loaded: object | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.value = f"pkg:{name}"
        self.protocol_version = protocol_version
        self.capabilities = capabilities
        self.optional_dependencies = dependencies
        self.dist = SimpleNamespace(name="demo", version="1.2.3")
        self.loaded = loaded if loaded is not None else object()
        self.error = error

    def load(self) -> object:
        if self.error:
            raise self.error
        return self.loaded


def test_plugin_discovery_negotiation_and_lazy_loading() -> None:
    good = _EntryPoint("good", loaded="provider")
    old = _EntryPoint("old", protocol_version="2.0")
    missing_capability = _EntryPoint("limited", capabilities=("trace",))
    handles = discover_plugins(
        capabilities=["capture"], entry_points=[old, missing_capability, good]
    )
    assert [handle.name for handle in handles] == ["good"]
    assert handles[0].load() == "provider"
    assert find_plugin("good", entry_points=[good]).name == "good"
    assert load_plugin("good", entry_points=[good]) == "provider"
    assert check_optional_dependencies(["json", "module_that_does_not_exist_123"]) == (
        "module_that_does_not_exist_123",
    )

    metadata = PluginMetadata.from_value(
        {
            "name": "manual",
            "version": "1.0",
            "api_version": "1.0",
            "features": "capture trace",
            "requires": ["json"],
            "custom": True,
        }
    )
    assert metadata.supports(["capture"]) and metadata.requires == ("json",)
    assert metadata.to_dict()["metadata"]["custom"] is True
    assert negotiate_plugin(metadata).name == "manual"
    require_optional_dependencies(metadata)
    with pytest.raises(PluginCompatibilityError):
        negotiate_plugin(metadata, protocol_version="2.0")
    with pytest.raises(PluginCompatibilityError):
        negotiate_plugin(metadata, capabilities=["missing"])
    with pytest.raises(PluginLoadError):
        find_plugin("unknown", entry_points=[good])


def test_plugin_dependency_and_load_errors_are_explicit() -> None:
    missing = _EntryPoint("missing", error=ImportError("optional_pkg", name="optional_pkg"))
    with pytest.raises(PluginDependencyError, match="optional_pkg"):
        load_plugin("missing", entry_points=[missing])
    broken = _EntryPoint("broken", error=RuntimeError("provider failed"))
    with pytest.raises(PluginLoadError, match="provider failed"):
        load_plugin("broken", entry_points=[broken])
    metadata = PluginMetadata("bad", optional_dependencies=("module_that_does_not_exist_123",))
    with pytest.raises(PluginDependencyError):
        require_optional_dependencies(metadata)
