"""Fresh-process ReplayFixture worker.

Invoked as ``python -m mlforensics.diagnose.worker``. The parent process
supplies a JSON request describing the capsule, factory, and target step.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from mlforensics.core import RunCapsule
from mlforensics.core.contracts import CHILD_RESULT_ENV, ExecutionResult, write_child_result
from mlforensics.core.failure import exception_signature
from mlforensics.diagnose.factory import factory_spec_from_capsule, load_replay_factory
from mlforensics.diagnose.replay import _extract_evidence


def _load_request() -> dict[str, Any]:
    path = os.environ.get("MLFORENSICS_WORKER_REQUEST")
    if path:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    raw = sys.stdin.read()
    if not raw.strip():
        raise SystemExit("worker request missing")
    return json.loads(raw)


def main() -> int:
    request = _load_request()
    capsule = RunCapsule.load(request["capsule"])
    spec = request.get("factory") or factory_spec_from_capsule(capsule)
    if not spec:
        raise SystemExit("capsule does not register a replay_factory")
    step = request.get("step")
    if step is not None:
        step = float(step) if "." in str(step) else int(step)
    fixture = load_replay_factory(spec)
    evidence = _extract_evidence(capsule, None, step)
    restoration = {
        "required_state": list(evidence.required_state),
        "restored_state": [],
        "omitted_state": [],
        "failed_state": [],
        "state_restoration_verified": False,
    }
    result_path = os.environ.get(CHILD_RESULT_ENV)
    try:
        fixture.construct()
        try:
            fixture.restore(dict(evidence.state))
            restoration["restored_state"] = list(evidence.required_state)
            restoration["state_restoration_verified"] = True
            planned = list(evidence.execution_steps) or [
                (evidence.executed_step, evidence.input if evidence.has_input else None)
            ]
            last: Any = None
            for _step, argument in planned:
                last = fixture.execute(argument)
            payload = ExecutionResult(
                status="ok",
                returncode=0,
                metadata={"restoration": restoration, "result": repr(last)},
            )
            if result_path:
                write_child_result(result_path, payload)
            return 0
        except BaseException as exc:
            failure = exception_signature(exc)
            expected = evidence.expected or capsule.run.failure_signature
            matched = expected.matches(failure) if expected is not None else False
            payload = ExecutionResult(
                status="fail",
                returncode=1,
                failure=failure,
                metadata={
                    "restoration": restoration,
                    "reproduced": matched,
                    "traceback": traceback.format_exc(),
                },
            )
            if result_path:
                write_child_result(result_path, payload)
            return 1
        finally:
            fixture.close()
    except BaseException as exc:
        payload = ExecutionResult(
            status="error",
            returncode=2,
            error=f"{type(exc).__name__}: {exc}",
            unresolved=True,
            metadata={"restoration": restoration, "traceback": traceback.format_exc()},
        )
        if result_path:
            write_child_result(result_path, payload)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
