"""Operational safety helpers for local and offline MLForensics workflows.

This module intentionally uses the standard library and does not import the
capsule or cloud-storage implementations at module import time.  It provides
small, injectable building blocks for retention, sensitive-data review,
append-only audit trails, legal holds, and remote/offline queues.  None of the
helpers upload data or delete anything by default.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

REDACTED = "<redacted>"
DEFAULT_SECRET_KEY_PATTERN = re.compile(
    r"(?:pass(?:word|wd)?|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|"
    r"authorization|credential|connection[_-]?string|client[_-]?secret)",
    re.IGNORECASE,
)
DEFAULT_SECRET_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "bearer-token",
        re.compile(r"(?i)(\bbearer\s+)[^\s,;]+"),
    ),
    (
        "basic-credential",
        re.compile(r"(?i)(\bbasic\s+)[A-Za-z0-9+/=]+"),
    ),
    (
        "uri-credential",
        re.compile(r"(?i)(://[^/\s:@]+:)[^@\s]+(@)"),
    ),
    (
        "named-secret",
        re.compile(
            r"(?i)(\b(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|"
            r"authorization|client[_-]?secret)\b\s*[:=]\s*[\"']?)[^\s,;\"']+"
        ),
    ),
)


def _compile_patterns(
    patterns: Mapping[str, str | re.Pattern[str]] | Sequence[str | re.Pattern[str]] | None,
) -> tuple[tuple[str, re.Pattern[str]], ...]:
    if patterns is None:
        return DEFAULT_SECRET_VALUE_PATTERNS
    if isinstance(patterns, Mapping):
        values = patterns.items()
    else:
        values = ((f"custom-{index}", item) for index, item in enumerate(patterns))
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for name, pattern in values:
        compiled.append(
            (str(name), pattern if hasattr(pattern, "search") else re.compile(str(pattern)))
        )
    return tuple(compiled)


def _secret_key(key: Any, key_pattern: re.Pattern[str] = DEFAULT_SECRET_KEY_PATTERN) -> bool:
    return isinstance(key, str) and bool(key_pattern.search(key))


def redact_text(
    value: str,
    *,
    patterns: Mapping[str, str | re.Pattern[str]] | Sequence[str | re.Pattern[str]] | None = None,
) -> str:
    """Redact common credentials from text without exposing their values."""
    if not isinstance(value, str):
        raise TypeError("value must be a string")
    result = value
    for _, pattern in _compile_patterns(patterns):
        result = pattern.sub(lambda match: _redacted_match(match), result)
    return result


def _redacted_match(match: re.Match[str]) -> str:
    groups = match.groups()
    if not groups:
        return REDACTED
    if len(groups) == 1:
        return f"{groups[0]}{REDACTED}"
    return f"{groups[0]}{REDACTED}{groups[-1]}"


def redact_value(
    value: Any,
    *,
    key: str | None = None,
    patterns: Mapping[str, str | re.Pattern[str]] | Sequence[str | re.Pattern[str]] | None = None,
    key_pattern: re.Pattern[str] = DEFAULT_SECRET_KEY_PATTERN,
) -> Any:
    """Recursively redact mappings, sequences, and strings.

    Secret-bearing keys are replaced wholesale.  Non-secret text is filtered
    for bearer/basic tokens, URI credentials, and named key/value pairs.
    """
    if _secret_key(key, key_pattern):
        return REDACTED
    if isinstance(value, Mapping):
        return {
            str(item_key): redact_value(
                item_value,
                key=str(item_key),
                patterns=patterns,
                key_pattern=key_pattern,
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item, patterns=patterns, key_pattern=key_pattern) for item in value]
    if isinstance(value, tuple):
        return tuple(
            redact_value(item, patterns=patterns, key_pattern=key_pattern) for item in value
        )
    if isinstance(value, set):
        return [
            redact_value(item, patterns=patterns, key_pattern=key_pattern)
            for item in sorted(value, key=str)
        ]
    if isinstance(value, str):
        return redact_text(value, patterns=patterns)
    return value


def redact_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Convenience alias for recursively redacting a mapping."""
    result = redact_value(value)
    return result if isinstance(result, dict) else {"value": result}


@dataclass(frozen=True)
class SensitiveFinding:
    """A location and category of a possible sensitive value.

    ``value`` is always redacted; the scanner never stores the matched secret.
    """

    path: str
    kind: str
    value: str = REDACTED
    source: str = ""

    @property
    def redacted(self) -> str:
        return self.value

    @property
    def redacted_value(self) -> str:
        return self.value

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "kind": self.kind, "value": self.value, "source": self.source}


@dataclass(frozen=True)
class ScanReport:
    """Result of :func:`scan_sensitive_data`."""

    findings: tuple[SensitiveFinding, ...] = ()
    scanned_files: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.findings and not self.errors

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean": self.clean,
            "findings": [item.to_dict() for item in self.findings],
            "scanned_files": list(self.scanned_files),
            "errors": list(self.errors),
        }


def _scan_text(
    text: str,
    *,
    path: str,
    source: str,
    findings: list[SensitiveFinding],
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
    key_pattern: re.Pattern[str],
) -> None:
    for name, pattern in patterns:
        for match in pattern.finditer(text):
            findings.append(
                SensitiveFinding(
                    path=path,
                    kind=name,
                    value=REDACTED,
                    source=source,
                )
            )
    # A JSON/text scan can contain a secret key whose value does not match a
    # generic token pattern.  Record only the key location, never its value.
    for match in re.finditer(r"(?i)[\"']?([A-Za-z][A-Za-z0-9_.-]{1,80})[\"']?\s*[:=]", text):
        if _secret_key(match.group(1), key_pattern):
            findings.append(
                SensitiveFinding(path=f"{path}.{match.group(1)}", kind="secret-key", source=source)
            )


def _scan_object(
    value: Any,
    *,
    path: str,
    source: str,
    findings: list[SensitiveFinding],
    patterns: tuple[tuple[str, re.Pattern[str]], ...],
    key_pattern: re.Pattern[str],
    seen: set[int],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if _secret_key(key, key_pattern):
                findings.append(SensitiveFinding(child, "secret-key", REDACTED, source))
            else:
                _scan_object(
                    item,
                    path=child,
                    source=source,
                    findings=findings,
                    patterns=patterns,
                    key_pattern=key_pattern,
                    seen=seen,
                )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_object(
                item,
                path=f"{path}[{index}]",
                source=source,
                findings=findings,
                patterns=patterns,
                key_pattern=key_pattern,
                seen=seen,
            )
        return
    if isinstance(value, str):
        _scan_text(
            value,
            path=path,
            source=source,
            findings=findings,
            patterns=patterns,
            key_pattern=key_pattern,
        )
    elif isinstance(value, (bytes, bytearray)):
        _scan_text(
            value.decode("utf-8", "replace"),
            path=path,
            source=source,
            findings=findings,
            patterns=patterns,
            key_pattern=key_pattern,
        )


def _iter_scan_files(root: Path, *, follow_symlinks: bool, max_files: int) -> Iterator[Path]:
    count = 0
    for candidate in sorted(root.rglob("*")):
        if candidate.is_symlink():
            if follow_symlinks:
                continue
            continue
        if candidate.is_file():
            yield candidate
            count += 1
            if count >= max_files:
                return


def scan_sensitive_data(
    target: Any,
    *,
    patterns: Mapping[str, str | re.Pattern[str]] | Sequence[str | re.Pattern[str]] | None = None,
    max_bytes: int = 2_000_000,
    max_files: int = 10_000,
    follow_symlinks: bool = False,
    key_pattern: re.Pattern[str] = DEFAULT_SECRET_KEY_PATTERN,
) -> ScanReport:
    """Scan a path, mapping, JSON-like value, or text for sensitive data.

    File reads are bounded and symlinks are skipped by default.  The returned
    report contains finding locations and categories only.
    """
    if max_bytes <= 0 or max_files <= 0:
        raise ValueError("max_bytes and max_files must be positive")
    compiled = _compile_patterns(patterns)
    findings: list[SensitiveFinding] = []
    scanned: list[str] = []
    errors: list[str] = []
    if isinstance(target, Mapping) or isinstance(target, (list, tuple)):
        _scan_object(
            target,
            path="$",
            source="object",
            findings=findings,
            patterns=compiled,
            key_pattern=key_pattern,
            seen=set(),
        )
    elif isinstance(target, (bytes, bytearray)):
        _scan_text(
            target[:max_bytes].decode("utf-8", "replace"),
            path="$",
            source="bytes",
            findings=findings,
            patterns=compiled,
            key_pattern=key_pattern,
        )
    elif isinstance(target, Path) or (isinstance(target, str) and Path(target).exists()):
        root = Path(target)
        if root.is_symlink() and not follow_symlinks:
            errors.append(f"skipped symlink: {root}")
        elif root.is_dir():
            for file_path in _iter_scan_files(
                root, follow_symlinks=follow_symlinks, max_files=max_files
            ):
                try:
                    content = file_path.read_bytes()[:max_bytes]
                    scanned.append(str(file_path))
                    _scan_text(
                        content.decode("utf-8", "replace"),
                        path=str(file_path),
                        source="file",
                        findings=findings,
                        patterns=compiled,
                        key_pattern=key_pattern,
                    )
                except OSError as exc:
                    errors.append(f"{file_path}: {exc}")
        elif root.is_file():
            try:
                content = root.read_bytes()[:max_bytes]
                scanned.append(str(root))
                _scan_text(
                    content.decode("utf-8", "replace"),
                    path=str(root),
                    source="file",
                    findings=findings,
                    patterns=compiled,
                    key_pattern=key_pattern,
                )
            except OSError as exc:
                errors.append(f"{root}: {exc}")
        else:
            errors.append(f"unsupported scan target: {root}")
    elif isinstance(target, str):
        _scan_text(
            target[:max_bytes],
            path="$",
            source="text",
            findings=findings,
            patterns=compiled,
            key_pattern=key_pattern,
        )
    else:
        _scan_object(
            target,
            path="$",
            source="object",
            findings=findings,
            patterns=compiled,
            key_pattern=key_pattern,
            seen=set(),
        )
    # Deduplicate overlapping regex/key reports while preserving stable order.
    unique: dict[tuple[str, str, str], SensitiveFinding] = {}
    for finding in findings:
        unique.setdefault((finding.path, finding.kind, finding.source), finding)
    return ScanReport(tuple(unique.values()), tuple(scanned), tuple(errors))


scan_sensitive = scan_sensitive_data


@dataclass(frozen=True)
class AuditEvent:
    """Structured append-only audit event."""

    timestamp: str
    actor: str
    action: str
    target: str
    result: str = "success"
    details: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return redact_value(
            {
                "timestamp": self.timestamp,
                "actor": self.actor,
                "action": self.action,
                "target": self.target,
                "result": self.result,
                "details": dict(self.details),
                "error": self.error,
            }
        )


class AuditLogger:
    """Write redacted JSONL audit events using append-only file semantics."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        actor: str | None = None,
        clock: Callable[[], datetime] | None = None,
        fsync: bool = True,
    ) -> None:
        self.path = Path(path)
        self.actor = actor or os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.fsync = fsync

    def record(
        self,
        action: str,
        target: str | os.PathLike[str],
        *,
        result: str = "success",
        details: Mapping[str, Any] | None = None,
        error: BaseException | str | None = None,
        actor: str | None = None,
    ) -> AuditEvent:
        if not str(action).strip() or not str(target).strip():
            raise ValueError("audit action and target must be non-empty")
        event = AuditEvent(
            timestamp=self.clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            actor=actor or self.actor,
            action=str(action),
            target=str(target),
            result=str(result),
            details=details or {},
            error=str(error) if error is not None else None,
        )
        payload = (
            json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        fd = os.open(self.path, flags, 0o600)
        try:
            with os.fdopen(fd, "ab", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
        except Exception:
            # fdopen owns the descriptor after construction; this branch is
            # intentionally narrow and never truncates an existing log.
            raise
        return event

    append = record


AuditLog = AuditLogger


@runtime_checkable
class LegalHold(Protocol):
    """Protocol used by retention to decide whether an item is protected."""

    def is_held(self, target: str | os.PathLike[str]) -> bool: ...


class LegalHoldRegistry:
    """In-memory legal holds with optional JSON persistence."""

    def __init__(self, values: Iterable[str | os.PathLike[str]] = ()) -> None:
        self._values = {self._normalize(value) for value in values}

    @staticmethod
    def _normalize(value: str | os.PathLike[str]) -> str:
        return str(Path(value).expanduser().resolve())

    def add(self, target: str | os.PathLike[str]) -> None:
        self._values.add(self._normalize(target))

    def remove(self, target: str | os.PathLike[str]) -> None:
        self._values.discard(self._normalize(target))

    def is_held(self, target: str | os.PathLike[str]) -> bool:
        value = self._normalize(target)
        return value in self._values or any(
            value.startswith(item + os.sep) for item in self._values
        )

    def __contains__(self, target: object) -> bool:
        return isinstance(target, (str, os.PathLike)) and self.is_held(target)

    def save(self, path: str | os.PathLike[str]) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(sorted(self._values), indent=2) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> LegalHoldRegistry:
        source = Path(path)
        if not source.exists():
            return cls()
        values = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
            raise ValueError("legal hold file must contain a JSON list of paths")
        return cls(values)


class SidecarLegalHold:
    """Treat a sibling marker file as a hold, without reading capsule data."""

    def __init__(self, suffix: str = ".legal-hold") -> None:
        if not suffix or os.sep in suffix or (os.altsep and os.altsep in suffix):
            raise ValueError("suffix must be a simple marker suffix")
        self.suffix = suffix

    def marker_for(self, target: str | os.PathLike[str]) -> Path:
        path = Path(target)
        return path.with_name(path.name + self.suffix)

    def is_held(self, target: str | os.PathLike[str]) -> bool:
        return self.marker_for(target).is_file()

    def hold(self, target: str | os.PathLike[str], *, reason: str = "") -> Path:
        marker = self.marker_for(target)
        marker.write_text(reason, encoding="utf-8")
        return marker

    def release(self, target: str | os.PathLike[str]) -> None:
        self.marker_for(target).unlink(missing_ok=True)


def _as_age_seconds(value: timedelta | float | int | None, days: float | None) -> float | None:
    if days is not None:
        value = days * 86400
    if value is None:
        return None
    if isinstance(value, timedelta):
        value = value.total_seconds()
    if isinstance(value, bool) or float(value) < 0:
        raise ValueError("retention age must be non-negative")
    return float(value)


@dataclass(frozen=True)
class RetentionPolicy:
    """Retention limits used by :func:`garbage_collect`.

    ``max_age`` can be seconds, a :class:`datetime.timedelta`, or use the
    friendlier ``max_age_days`` field.  ``max_bytes``/``max_total_bytes`` keep
    the newest items until the root is under the requested budget.
    """

    max_age: timedelta | float | int | None = None
    max_age_days: float | None = None
    max_bytes: int | None = None
    max_total_bytes: int | None = None
    keep_recent: int = 0
    suffixes: tuple[str, ...] = (".mlcap", ".mlcap.zip")

    def __post_init__(self) -> None:
        _as_age_seconds(self.max_age, self.max_age_days)
        budget = self.max_total_bytes if self.max_total_bytes is not None else self.max_bytes
        if budget is not None and (isinstance(budget, bool) or int(budget) < 0):
            raise ValueError("max_bytes must be a non-negative integer")
        if isinstance(self.keep_recent, bool) or self.keep_recent < 0:
            raise ValueError("keep_recent must be a non-negative integer")
        if not self.suffixes or any(not item.startswith(".") for item in self.suffixes):
            raise ValueError("suffixes must contain dot-prefixed suffixes")

    @property
    def age_seconds(self) -> float | None:
        return _as_age_seconds(self.max_age, self.max_age_days)

    @property
    def byte_budget(self) -> int | None:
        return self.max_total_bytes if self.max_total_bytes is not None else self.max_bytes


@dataclass(frozen=True)
class RetentionCandidate:
    path: Path
    size_bytes: int
    modified_at: float
    reason: str
    held: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "reason": self.reason,
            "held": self.held,
        }


@dataclass(frozen=True)
class GCResult:
    """Summary of a retention run."""

    scanned: tuple[Path, ...] = ()
    eligible: tuple[RetentionCandidate, ...] = ()
    deleted: tuple[Path, ...] = ()
    skipped: Mapping[str, str] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    bytes_reclaimed: int = 0
    dry_run: bool = True
    trash_dir: Path | None = None

    @property
    def deleted_count(self) -> int:
        return len(self.deleted)

    @property
    def candidates(self) -> tuple[RetentionCandidate, ...]:
        return self.eligible

    def to_dict(self) -> dict[str, Any]:
        return {
            "scanned": [str(item) for item in self.scanned],
            "eligible": [item.to_dict() for item in self.eligible],
            "deleted": [str(item) for item in self.deleted],
            "skipped": dict(self.skipped),
            "errors": list(self.errors),
            "bytes_reclaimed": self.bytes_reclaimed,
            "dry_run": self.dry_run,
            "trash_dir": str(self.trash_dir) if self.trash_dir else None,
        }


def _capsule_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink():
            continue
        if item.is_file():
            total += item.stat().st_size
    return total


def _looks_like_capsule(path: Path, suffixes: Sequence[str]) -> bool:
    if path.is_symlink() or not any(path.name.endswith(suffix) for suffix in suffixes):
        return False
    if path.is_file():
        return path.name.endswith(".zip")
    if path.is_dir():
        return (path / "manifest.json").is_file() and (
            (path / "capsule.json").is_file() or (path / "run.json").is_file()
        )
    return False


def iter_capsules(
    root: str | os.PathLike[str], *, suffixes: Sequence[str] = (".mlcap", ".mlcap.zip")
) -> Iterator[Path]:
    """Yield recognized, non-symlink capsule paths below ``root``."""
    base = Path(root)
    if base.is_symlink():
        return
    if _looks_like_capsule(base, suffixes):
        yield base
        return
    if not base.is_dir():
        return
    for item in sorted(base.rglob("*")):
        if item.is_symlink():
            continue
        if item.is_dir() and _looks_like_capsule(item, suffixes):
            yield item
        elif item.is_file() and _looks_like_capsule(item, suffixes):
            yield item


def _hold_check(
    held: LegalHold | Callable[[Path], bool] | Iterable[str | os.PathLike[str]] | None, path: Path
) -> bool:
    if held is None:
        return False
    if callable(held) and not hasattr(held, "is_held"):
        return bool(held(path))
    if hasattr(held, "is_held"):
        return bool(held.is_held(path))
    values = {str(Path(item).resolve()) for item in held}
    return str(path.resolve()) in values


def _indexed_check(
    index: Iterable[str | os.PathLike[str]] | Mapping[Any, Any] | None, path: Path
) -> bool:
    if index is None:
        return False
    values = index.keys() if isinstance(index, Mapping) else index
    normalized = {
        str(Path(item).resolve()) for item in values if isinstance(item, (str, os.PathLike))
    }
    return str(path.resolve()) in normalized or path.name in normalized


def _move_to_trash(path: Path, root: Path, trash_dir: Path) -> Path:
    root_resolved = root.resolve()
    trash_resolved = trash_dir.resolve()
    if trash_resolved == root_resolved or root_resolved in trash_resolved.parents:
        # The trash must not be a child of a scanned root: it would be scanned
        # recursively and create an accidental feedback loop.
        raise ValueError("trash_dir must be outside the scanned root")
    trash_dir.mkdir(parents=True, exist_ok=True)
    relative = path.resolve().relative_to(root_resolved)
    destination = trash_dir / (str(relative).replace(os.sep, "__") + "." + uuid.uuid4().hex)
    os.replace(path, destination)
    return destination


def garbage_collect(
    root: str | os.PathLike[str],
    *,
    policy: RetentionPolicy | None = None,
    dry_run: bool = True,
    now: float | datetime | None = None,
    held: LegalHold | Callable[[Path], bool] | Iterable[str | os.PathLike[str]] | None = None,
    index: Iterable[str | os.PathLike[str]] | Mapping[Any, Any] | None = None,
    trash_dir: str | os.PathLike[str] | None = None,
    audit_log: AuditLogger | None = None,
) -> GCResult:
    """Plan or execute safe local capsule collection.

    Collection is a no-op unless a policy selects an item.  A real collection
    moves candidates to an external trash directory rather than unlinking
    them, so callers can restore them with :func:`restore_from_trash`.
    """
    base = Path(root)
    if base.is_symlink() or not base.exists() or not base.is_dir():
        raise ValueError("root must be an existing non-symlink directory")
    policy = policy or RetentionPolicy()
    current = (
        now.timestamp()
        if isinstance(now, datetime)
        else float(now if now is not None else time.time())
    )
    all_paths = tuple(iter_capsules(base, suffixes=policy.suffixes))
    skipped: dict[str, str] = {}
    errors: list[str] = []
    stats: dict[Path, tuple[int, float]] = {}
    candidates: list[RetentionCandidate] = []
    protected_recent = set(
        sorted(all_paths, key=lambda item: item.stat().st_mtime, reverse=True)[: policy.keep_recent]
    )
    for path in all_paths:
        try:
            if path.is_symlink():
                skipped[str(path)] = "symlink"
                continue
            stat = path.stat()
            size = _capsule_size(path)
            stats[path] = (size, stat.st_mtime)
            if path in protected_recent:
                skipped[str(path)] = "keep_recent"
                continue
            if _hold_check(held, path):
                skipped[str(path)] = "legal_hold"
                continue
            if _indexed_check(index, path):
                skipped[str(path)] = "indexed"
                continue
            reasons: list[str] = []
            if policy.age_seconds is not None and current - stat.st_mtime >= policy.age_seconds:
                reasons.append("max_age")
            if reasons:
                candidates.append(RetentionCandidate(path, size, stat.st_mtime, "+".join(reasons)))
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    budget = policy.byte_budget
    if budget is not None:
        total = sum(size for size, _ in stats.values())
        # Oldest first; held/indexed/recent items remain protected and are not
        # silently counted as candidates.
        for path, (size, mtime) in sorted(stats.items(), key=lambda item: item[1][1]):
            if (
                total <= budget
                or path in protected_recent
                or path in {item.path for item in candidates}
            ):
                continue
            if path in skipped:
                continue
            candidate = RetentionCandidate(path, size, mtime, "max_bytes")
            candidates.append(candidate)
            total -= size
    deduped = {item.path: item for item in candidates}
    selected = tuple(sorted(deduped.values(), key=lambda item: (item.modified_at, str(item.path))))
    deleted: list[Path] = []
    reclaimed = 0
    trash: Path | None = Path(trash_dir) if trash_dir is not None else None
    for item in selected:
        if audit_log is not None:
            audit_log.record(
                "retention.plan",
                item.path,
                result="planned" if dry_run else "selected",
                details=item.to_dict(),
            )
        if dry_run:
            continue
        try:
            trash = trash or (base.parent / (base.name + ".trash"))
            moved = _move_to_trash(item.path, base, trash)
            deleted.append(item.path)
            reclaimed += item.size_bytes
            if audit_log is not None:
                audit_log.record(
                    "retention.move_to_trash",
                    item.path,
                    details={"trash_path": str(moved), "size_bytes": item.size_bytes},
                )
        except (OSError, ValueError) as exc:
            errors.append(f"{item.path}: {exc}")
            skipped[str(item.path)] = "error"
            if audit_log is not None:
                audit_log.record("retention.move_to_trash", item.path, result="error", error=exc)
    return GCResult(
        all_paths, selected, tuple(deleted), skipped, tuple(errors), reclaimed, dry_run, trash
    )


retention_gc = garbage_collect


def restore_from_trash(
    trash_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    overwrite: bool = False,
) -> Path:
    """Restore one recoverable collection result to a validated destination."""
    source = Path(trash_path)
    target = Path(destination)
    if source.is_symlink() or not source.exists():
        raise FileNotFoundError(source)
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("restore destination must not be a symlink")
    os.replace(source, target)
    return target


@runtime_checkable
class RemoteTransport(Protocol):
    """Minimal injectable remote transport protocol."""

    def put_bytes(self, key: str, content: bytes) -> Any: ...

    def get_bytes(self, key: str) -> bytes: ...


def _safe_remote_key(key: str) -> str:
    if not isinstance(key, str) or not key.strip() or "\x00" in key:
        raise ValueError("remote key must be a non-empty string")
    normalized = key.replace("\\", "/")
    if normalized.startswith("/") or any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ValueError("remote key must be a relative normalized path")
    return normalized


@dataclass(frozen=True)
class OfflineOperation:
    operation_id: str
    action: str
    key: str
    payload: bytes = b""
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_id": self.operation_id,
            "action": self.action,
            "key": self.key,
            "payload": base64.b64encode(self.payload).decode("ascii"),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> OfflineOperation:
        try:
            payload = base64.b64decode(str(value.get("payload", "")), validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("offline queue contains invalid base64 payload") from exc
        return cls(
            operation_id=str(value["operation_id"]),
            action=str(value["action"]),
            key=_safe_remote_key(str(value["key"])),
            payload=payload,
            created_at=str(value.get("created_at", "")),
        )


class OfflineQueue:
    """Durable JSONL queue for explicitly deferred remote operations."""

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._memory: list[OfflineOperation] = []

    def _read(self) -> list[OfflineOperation]:
        if self.path is None or not self.path.exists():
            return list(self._memory)
        operations: list[OfflineOperation] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                operations.append(OfflineOperation.from_dict(json.loads(line)))
        return operations

    def _write(self, operations: Sequence[OfflineOperation]) -> None:
        if self.path is None:
            self._memory = list(operations)
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary_name = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                for operation in operations:
                    stream.write(
                        json.dumps(operation.to_dict(), sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def enqueue(self, action: str, key: str, payload: bytes = b"") -> OfflineOperation:
        if action not in {"put", "delete"}:
            raise ValueError("offline action must be put or delete")
        if action == "put" and not isinstance(payload, bytes):
            raise TypeError("offline payload must be bytes")
        operation = OfflineOperation(
            operation_id=uuid.uuid4().hex,
            action=action,
            key=_safe_remote_key(key),
            payload=payload,
            created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        operations = self._read()
        operations.append(operation)
        self._write(operations)
        return operation

    def pending(self) -> tuple[OfflineOperation, ...]:
        return tuple(self._read())

    def flush(
        self,
        transport: RemoteTransport,
        *,
        audit_log: AuditLogger | None = None,
        max_items: int | None = None,
    ) -> tuple[OfflineOperation, ...]:
        """Apply queued operations, retaining failures for a later retry."""
        pending = list(self._read())
        if max_items is not None and max_items < 0:
            raise ValueError("max_items must be non-negative or None")
        remaining: list[OfflineOperation] = []
        processed = 0
        for operation in pending:
            if max_items is not None and processed >= max_items:
                remaining.append(operation)
                continue
            try:
                if operation.action == "put":
                    _transport_put(transport, operation.key, operation.payload)
                else:
                    _transport_delete(transport, operation.key)
                processed += 1
                if audit_log is not None:
                    audit_log.record(
                        "offline.flush",
                        operation.key,
                        details={"operation_id": operation.operation_id},
                    )
            except Exception as exc:
                remaining.append(operation)
                if audit_log is not None:
                    audit_log.record(
                        "offline.flush",
                        operation.key,
                        result="error",
                        error=exc,
                        details={"operation_id": operation.operation_id},
                    )
        self._write(remaining)
        return tuple(remaining)


def _transport_put(transport: Any, key: str, payload: bytes) -> Any:
    method = getattr(transport, "put_bytes", None) or getattr(transport, "put", None)
    if method is None:
        raise TypeError("remote transport must expose put_bytes() or put()")
    return method(key, payload)


def _transport_get(transport: Any, key: str) -> bytes:
    method = getattr(transport, "get_bytes", None) or getattr(transport, "get", None)
    if method is None:
        raise TypeError("remote transport must expose get_bytes() or get()")
    value = method(key)
    if not isinstance(value, bytes):
        raise TypeError("remote transport get method must return bytes")
    return value


def _transport_delete(transport: Any, key: str) -> Any:
    method = getattr(transport, "delete", None) or getattr(transport, "remove", None)
    if method is None:
        raise TypeError("remote transport must expose delete() or remove()")
    return method(key)


class RemoteOperations:
    """Explicit remote/offline facade over an injected transport."""

    def __init__(
        self,
        transport: RemoteTransport | Any | None = None,
        *,
        offline: bool = False,
        queue: OfflineQueue | None = None,
        audit_log: AuditLogger | None = None,
    ) -> None:
        if not offline and transport is None:
            raise ValueError("transport is required unless offline=True")
        self.transport = transport
        self.offline = offline
        self.queue = queue or OfflineQueue()
        self.audit_log = audit_log

    def put(self, key: str, payload: bytes) -> OfflineOperation | Any:
        normalized = _safe_remote_key(key)
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if self.offline:
            operation = self.queue.enqueue("put", normalized, payload)
            self._audit("remote.queue_put", normalized, operation)
            return operation
        result = _transport_put(self.transport, normalized, payload)
        self._audit("remote.put", normalized)
        return result

    def get(self, key: str) -> bytes:
        normalized = _safe_remote_key(key)
        if self.offline:
            raise RuntimeError("remote reads are unavailable in offline mode")
        result = _transport_get(self.transport, normalized)
        self._audit("remote.get", normalized)
        return result

    def delete(self, key: str) -> OfflineOperation | Any:
        normalized = _safe_remote_key(key)
        if self.offline:
            operation = self.queue.enqueue("delete", normalized)
            self._audit("remote.queue_delete", normalized, operation)
            return operation
        result = _transport_delete(self.transport, normalized)
        self._audit("remote.delete", normalized)
        return result

    def flush(self, *, max_items: int | None = None) -> tuple[OfflineOperation, ...]:
        if self.transport is None:
            raise RuntimeError("a transport is required to flush an offline queue")
        return self.queue.flush(self.transport, audit_log=self.audit_log, max_items=max_items)

    def _audit(self, action: str, key: str, operation: OfflineOperation | None = None) -> None:
        if self.audit_log is not None:
            self.audit_log.record(
                action, key, details={"operation_id": operation.operation_id} if operation else {}
            )


OfflineRemoteStore = RemoteOperations


def publish_remote(
    key: str,
    payload: bytes,
    *,
    transport: RemoteTransport | Any | None = None,
    offline: bool = False,
    queue: OfflineQueue | None = None,
    audit_log: AuditLogger | None = None,
) -> OfflineOperation | Any:
    """Publish now or queue explicitly for later; never discovers credentials."""
    return RemoteOperations(transport, offline=offline, queue=queue, audit_log=audit_log).put(
        key, payload
    )


def sha256_bytes(payload: bytes) -> str:
    """Return a content digest for remote integrity checks."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "AuditEvent",
    "AuditLog",
    "AuditLogger",
    "DEFAULT_SECRET_KEY_PATTERN",
    "GCResult",
    "LegalHold",
    "LegalHoldRegistry",
    "OfflineOperation",
    "OfflineQueue",
    "OfflineRemoteStore",
    "REDACTED",
    "RemoteOperations",
    "RemoteTransport",
    "RetentionCandidate",
    "RetentionPolicy",
    "ScanReport",
    "SensitiveFinding",
    "SidecarLegalHold",
    "garbage_collect",
    "iter_capsules",
    "publish_remote",
    "redact_mapping",
    "redact_text",
    "redact_value",
    "restore_from_trash",
    "retention_gc",
    "scan_sensitive",
    "scan_sensitive_data",
    "sha256_bytes",
]
