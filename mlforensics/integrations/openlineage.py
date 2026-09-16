"""OpenLineage-shaped event generation without a runtime dependency."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from ..core import RunCapsule


def lineage_event(
    capsule: RunCapsule,
    *,
    namespace: str = "mlforensics",
    producer: str = "mlforensics",
    inputs: Iterable[Mapping[str, Any]] | None = None,
    outputs: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    run_id = capsule.run_id
    status = str(getattr(capsule.run, "status", "unknown")).lower()
    command = getattr(capsule.run, "command", ())
    metadata = getattr(capsule.run, "metadata", {}) or {}
    job_name = command[0] if command else metadata.get("command", run_id)
    if isinstance(job_name, (list, tuple)):
        job_name = job_name[0] if job_name else run_id

    def dataset_input(dataset: Any) -> dict[str, Any]:
        artifact = getattr(dataset, "artifact", None)
        fingerprint = getattr(dataset, "fingerprint", None)
        if fingerprint is None and artifact is not None:
            fingerprint = getattr(artifact, "sha256", None)
        if fingerprint is None:
            fingerprint = (getattr(dataset, "metadata", {}) or {}).get("sha256")
        return {
            "namespace": namespace,
            "name": getattr(dataset, "name", str(dataset)),
            "facets": {"fingerprint": {"value": fingerprint} if fingerprint else {}},
        }

    digest = getattr(capsule, "digest", None)
    if digest is None:
        payload = capsule.to_dict() if hasattr(capsule, "to_dict") else repr(capsule)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()
    return {
        "eventType": "COMPLETE" if status in {"succeeded", "completed", "success"} else "FAIL",
        "eventTime": capsule.run.ended_at or capsule.run.started_at,
        "run": {"runId": run_id},
        "job": {
            "namespace": namespace,
            "name": str(job_name),
            "facets": {"mlforensics": {"schemaVersion": capsule.schema_version, "digest": digest}},
        },
        "producer": producer,
        "inputs": list(inputs)
        if inputs is not None
        else [dataset_input(dataset) for dataset in getattr(capsule.run, "datasets", ())],
        "outputs": list(outputs) if outputs is not None else [],
    }


class OpenLineageAdapter:
    def __init__(self, client: Any) -> None:
        self.client = client

    def emit(self, capsule: RunCapsule, **kwargs: Any) -> dict[str, Any]:
        event = lineage_event(capsule, **kwargs)
        if hasattr(self.client, "emit"):
            self.client.emit(event)
        elif hasattr(self.client, "emit_event"):
            self.client.emit_event(event)
        else:
            raise TypeError("OpenLineage client must expose emit() or emit_event()")
        return event
