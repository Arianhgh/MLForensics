"""Load ReplayFixture factories and execute them in-process or in a worker."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..core import FailureSignature, RunCapsule
from ..core.contracts import (
    CHILD_RESULT_ENV,
    REPLAY_CAPSULE_ENV,
    REPLAY_STEP_ENV,
    ExecutionSpec,
    ReplayFixture,
)
from ..core.execution import ExecutionService
from .replay import ReplayEngine, ReplayResult, _extract_evidence

WORKER_MODULE = "mlforensics.diagnose.worker"


def _loaded_package_import_root() -> str | None:
    """Return the import root for the currently loaded ``mlforensics`` package.

    Worker commands may run with a working directory that is unrelated to the
    parent process' checkout (for example, a detached Git worktree).  In that
    case Python does not inherit the parent's implicit ``sys.path[0]``.  The
    package location is a reliable import root for both source-tree and
    installed-wheel execution.
    """
    package = sys.modules.get("mlforensics")
    package_file = getattr(package, "__file__", None)
    if not package_file:
        return None
    package_directory = Path(package_file).resolve().parent
    return str(package_directory.parent)


def _worker_pythonpath() -> str | None:
    package_root = _loaded_package_import_root()
    inherited = os.environ.get("PYTHONPATH")
    if package_root is None:
        return inherited
    entries = inherited.split(os.pathsep) if inherited else []
    if package_root not in entries:
        entries.insert(0, package_root)
    return os.pathsep.join(entries)


def load_replay_factory(spec: str, **kwargs: Any) -> ReplayFixture:
    """Import ``module:attr`` or ``module.attr`` and return a ReplayFixture."""
    if not spec or not str(spec).strip():
        raise ValueError("replay factory spec must be a non-empty string")
    text = str(spec).strip()
    if ":" in text:
        module_name, attr = text.split(":", 1)
    else:
        module_name, _, attr = text.rpartition(".")
    if not module_name or not attr:
        raise ValueError(f"invalid replay factory spec: {spec!r}")
    module = importlib.import_module(module_name)
    target = module
    for part in attr.split("."):
        target = getattr(target, part)
    if isinstance(target, type):
        fixture = target(**kwargs)
    elif callable(target) and not hasattr(target, "construct"):
        fixture = target(**kwargs) if kwargs else target()
    else:
        fixture = target
    if not all(hasattr(fixture, name) for name in ("construct", "restore", "execute", "close")):
        raise TypeError(f"{spec!r} did not produce a ReplayFixture")
    return fixture


def factory_spec_from_capsule(capsule: RunCapsule) -> str | None:
    metadata = capsule.run.metadata if isinstance(capsule.run.metadata, Mapping) else {}
    spec = metadata.get("replay_factory")
    if spec:
        return str(spec)
    replay = capsule.evidence.get("replay", {})
    if isinstance(replay, Mapping) and replay.get("factory"):
        return str(replay["factory"])
    plan = capsule.run.replay_plan
    plan_meta = getattr(plan, "metadata", None)
    if isinstance(plan_meta, Mapping) and plan_meta.get("replay_factory"):
        return str(plan_meta["replay_factory"])
    return None


def replay_with_factory(
    capsule: RunCapsule | str | Path,
    *,
    factory: str | ReplayFixture | None = None,
    step: int | float | None = None,
    expected: FailureSignature | None = None,
    strict_state: bool = True,
) -> ReplayResult:
    """Replay a capsule by constructing a fresh fixture, not a live model."""
    source = RunCapsule.load(capsule) if isinstance(capsule, (str, Path)) else capsule
    fixture: ReplayFixture
    if isinstance(factory, str) or factory is None:
        spec = factory or factory_spec_from_capsule(source)
        if spec is None:
            raise ValueError("capsule does not register a replay_factory")
        fixture = load_replay_factory(spec)
    else:
        fixture = factory
    fixture.construct()
    try:
        evidence = _extract_evidence(source, expected, step)
        fixture.restore(dict(evidence.state))
        tracer = None
        handles: list[Any] = []
        model = getattr(fixture, "model", None)
        if model is not None:
            from .trace import TensorTracer, attach_torch_hooks, persist_trace

            try:
                tracer = TensorTracer()
                handles = attach_torch_hooks(model, tracer)
            except Exception:
                tracer = None
                handles = []
        restorers = {
            name: (lambda _state: None)
            for name in evidence.state
            if name not in {"python", "numpy", "torch_cpu", "torch_cuda"}
        }
        engine = ReplayEngine(state_restorers=restorers, strict_state=strict_state)
        result = engine.replay(source, fixture.execute, expected=expected, step=step)
        result.metadata["factory_replay"] = True
        if tracer is not None:
            report = persist_trace(source, tracer)
            result.metadata["tensor_trace"] = report
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        return result
    finally:
        fixture.close()


def replay_in_worker(
    capsule: RunCapsule | str | Path,
    *,
    factory: str | None = None,
    step: int | float | None = None,
    timeout: float | None = 120.0,
    working_directory: str | None = None,
    service: ExecutionService | None = None,
) -> ReplayResult:
    """Replay a capsule in a fresh process using the registered factory."""
    source = RunCapsule.load(capsule) if isinstance(capsule, (str, Path)) else capsule
    spec = factory or factory_spec_from_capsule(source)
    if spec is None:
        raise ValueError("capsule does not register a replay_factory")
    service = service or ExecutionService()
    with tempfile.TemporaryDirectory(prefix="mlforensics-worker-") as temporary:
        capsule_path = Path(temporary) / "capsule.mlcap"
        request_path = Path(temporary) / "request.json"
        result_path = Path(temporary) / "child.result.json"
        source.save(capsule_path, overwrite=True)
        request = {
            "capsule": str(capsule_path),
            "factory": spec,
            "step": step,
            "expected_failure": (
                source.run.failure_signature.to_dict() if source.run.failure_signature else None
            ),
        }
        request_path.write_text(json.dumps(request), encoding="utf-8")
        extra = {
            REPLAY_CAPSULE_ENV: str(capsule_path),
            CHILD_RESULT_ENV: str(result_path),
            "MLFORENSICS_WORKER_REQUEST": str(request_path),
        }
        pythonpath = _worker_pythonpath()
        if pythonpath:
            extra["PYTHONPATH"] = pythonpath
        if os.environ.get("MLFORENSICS_EXAMPLE_ROOT"):
            extra["MLFORENSICS_EXAMPLE_ROOT"] = os.environ["MLFORENSICS_EXAMPLE_ROOT"]
        if step is not None:
            extra[REPLAY_STEP_ENV] = str(step)
        completed = service.run(
            ExecutionSpec(
                command=[sys.executable, "-m", WORKER_MODULE],
                working_directory=working_directory,
                timeout=timeout,
                capsule=str(capsule_path),
                metadata={"factory": spec},
            ),
            extra_env=extra,
            result_path=result_path,
        )
        payload = completed.metadata.get("execution_record", {})
        reproduced = completed.status == "fail" and completed.failure is not None
        if completed.failure is not None and source.run.failure_signature is not None:
            reproduced = source.run.failure_signature.matches(completed.failure)
        restoration = {}
        if isinstance(completed.metadata, Mapping):
            restoration = completed.metadata.get("restoration", {}) or {}
        return ReplayResult(
            reproduced,
            failure=completed.failure,
            error=completed.error,
            restored=list(restoration.get("restored_state", ())),
            metadata={
                "verified": not completed.unresolved,
                "factory_replay": True,
                "worker": True,
                "state_restoration_verified": bool(restoration.get("state_restoration_verified")),
                "required_state": restoration.get("required_state", []),
                "restored_state": restoration.get("restored_state", []),
                "omitted_state": restoration.get("omitted_state", []),
                "failed_state": restoration.get("failed_state", []),
                "returncode": completed.returncode,
                "stdout": completed.metadata.get("stdout"),
                "stderr": completed.metadata.get("stderr"),
                "execution_record": payload,
            },
        )
