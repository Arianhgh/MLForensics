"""Shared child-process execution for capture, replay, shrink, and bisect."""

from __future__ import annotations

import os
import signal
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from subprocess import PIPE, Popen, TimeoutExpired
from typing import Any

from .contracts import (
    CHILD_RESULT_ENV,
    ExecutionResult,
    ExecutionSpec,
    load_child_result,
)
from .errors import UnresolvedEvaluation, ValidationError
from .models import SCHEMA_VERSION, utc_now
from .serialization import dump_bytes

DEFAULT_LOG_LIMIT = 100_000


@dataclass
class ExecutionRecord:
    """Durable record of one child execution."""

    spec: Mapping[str, Any]
    result: Mapping[str, Any]
    started_at: str
    ended_at: str
    stdout: str = ""
    stderr: str = ""
    cancelled: bool = False
    timed_out: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, Mapping) or not isinstance(self.result, Mapping):
            raise ValidationError("execution record spec and result must be mappings")
        if not isinstance(self.started_at, str) or not isinstance(self.ended_at, str):
            raise ValidationError("execution record timestamps must be strings")
        if not isinstance(self.stdout, str) or not isinstance(self.stderr, str):
            raise ValidationError("execution record logs must be strings")
        if not isinstance(self.cancelled, bool) or not isinstance(self.timed_out, bool):
            raise ValidationError("execution record flags must be booleans")
        self.spec = dict(self.spec)
        self.result = dict(self.result)
        self.metadata = dict(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "type": "execution_record",
            "spec": dict(self.spec),
            "result": dict(self.result),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "cancelled": self.cancelled,
            "timed_out": self.timed_out,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionRecord:
        if data.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValidationError("unsupported execution record schema version")
        if data.get("type") not in {None, "execution_record"}:
            raise ValidationError("not an execution_record record")
        spec = data.get("spec")
        result = data.get("result")
        if not isinstance(spec, Mapping) or not isinstance(result, Mapping):
            raise ValidationError("execution record is missing spec or result")
        return cls(
            spec=spec,
            result=result,
            started_at=data.get("started_at"),
            ended_at=data.get("ended_at"),
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            cancelled=data.get("cancelled", False),
            timed_out=data.get("timed_out", False),
            metadata=data.get("metadata", {}),
        )


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


class ExecutionService:
    """Run a child command with bounded logs, timeout, and process cleanup."""

    def __init__(self, *, log_limit: int = DEFAULT_LOG_LIMIT) -> None:
        if isinstance(log_limit, bool) or log_limit < 1:
            raise ValueError("log_limit must be a positive integer")
        self.log_limit = log_limit
        self._active: list[Popen[str]] = []
        self.records: list[ExecutionRecord] = []

    def run(
        self,
        spec: ExecutionSpec | Sequence[str] | str,
        *,
        env: Mapping[str, str] | None = None,
        extra_env: Mapping[str, str] | None = None,
        result_path: str | os.PathLike[str] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> ExecutionResult:
        if not isinstance(spec, ExecutionSpec):
            spec = ExecutionSpec(command=spec)
        command: Sequence[str] | str = spec.command
        environment = dict(env if env is not None else os.environ)
        if extra_env:
            environment.update({str(key): str(value) for key, value in extra_env.items()})
        if result_path is not None:
            result_file = Path(result_path)
            if result_file.exists() and not result_file.is_file():
                raise ValidationError("result_path must name a file")
            # Never consume an envelope left by an earlier run using the same
            # path.  A structured result is authoritative only for this child.
            result_file.unlink(missing_ok=True)
            environment[CHILD_RESULT_ENV] = str(result_file)
        started = utc_now()
        timed_out = False
        cancelled = False
        popen_command: Any = command if isinstance(command, str) else list(command)
        process = Popen(
            popen_command,
            cwd=spec.working_directory,
            env=environment,
            text=True,
            errors="replace",
            stdout=PIPE,
            stderr=PIPE,
            shell=isinstance(command, str),
            start_new_session=True,
        )
        self._active.append(process)
        stdout = ""
        stderr = ""
        try:
            if cancel_event is None and spec.timeout is None:
                stdout, stderr = process.communicate()
            else:
                deadline = None if spec.timeout is None else time.monotonic() + float(spec.timeout)
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                        _terminate_group(process)
                        break
                    if deadline is not None and time.monotonic() >= deadline:
                        timed_out = True
                        _terminate_group(process)
                        break
                    wait = 0.05
                    if deadline is not None:
                        wait = max(0.001, min(wait, deadline - time.monotonic()))
                    try:
                        stdout, stderr = process.communicate(timeout=wait)
                        break
                    except TimeoutExpired:
                        # communicate() drains both pipes while it waits. Calling
                        # it again is supported and returns the complete output.
                        continue
                if timed_out or cancelled:
                    stdout, stderr = process.communicate()
            returncode = process.wait()
        finally:
            if process in self._active:
                self._active.remove(process)
            if process.poll() is None:
                _terminate_group(process)
        stdout = _trim(stdout or "", self.log_limit)
        stderr = _trim(stderr or "", self.log_limit)
        child_result = None
        child_path = environment.get(CHILD_RESULT_ENV)
        if child_path and Path(child_path).exists():
            try:
                child_result = load_child_result(child_path)
            except UnresolvedEvaluation:
                child_result = ExecutionResult(
                    status="inconclusive",
                    returncode=returncode,
                    unresolved=True,
                    error="malformed child result envelope",
                )
        if child_result is None:
            status = (
                "timeout"
                if timed_out
                else "cancelled"
                if cancelled
                else "ok"
                if returncode == 0
                else "fail"
            )
            child_result = ExecutionResult(
                status=status,
                returncode=returncode,
                error=(
                    "execution timed out"
                    if timed_out
                    else "execution cancelled"
                    if cancelled
                    else None
                ),
            )
        ended = utc_now()
        record = ExecutionRecord(
            spec=spec.to_dict(),
            result=child_result.to_dict(),
            started_at=started,
            ended_at=ended,
            stdout=stdout,
            stderr=stderr,
            cancelled=cancelled,
            timed_out=timed_out,
        )
        self.records.append(record)
        metadata = dict(child_result.metadata)
        metadata.update(
            {
                "stdout": stdout,
                "stderr": stderr,
                "timed_out": timed_out,
                "cancelled": cancelled,
                "execution_record": record.to_dict(),
            }
        )
        return ExecutionResult(
            status=child_result.status,
            returncode=child_result.returncode,
            failure=child_result.failure,
            error=child_result.error,
            resources=child_result.resources,
            capsule=child_result.capsule,
            unresolved=child_result.unresolved,
            metadata=metadata,
        )

    def cancel_all(self) -> None:
        for process in list(self._active):
            _terminate_group(process)


def _terminate_group(process: Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        try:
            process.terminate()
        except (OSError, ProcessLookupError):
            return
    try:
        process.wait(timeout=2)
    except Exception:
        try:
            if sys.platform == "win32":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=1)
        except Exception:
            pass


def persist_execution_record(path: str | os.PathLike[str], record: ExecutionRecord) -> Path:
    if not isinstance(record, ExecutionRecord):
        raise TypeError("record must be an ExecutionRecord")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(dump_bytes(record))
            handle.write(b"\n")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


__all__ = ["ExecutionRecord", "ExecutionService", "persist_execution_record"]
