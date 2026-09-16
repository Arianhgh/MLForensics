"""Optional OpenLineage-shaped dictionaries (no OpenLineage package required)."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .graph import DependencyGraph, Node


def _dataset(node: Node, namespace: str) -> dict[str, Any]:
    return {
        "namespace": namespace,
        "name": node.id,
        "facets": {"mlforensics": {"kind": node.kind, **dict(node.metadata)}},
    }


def openlineage_event(
    job_name: str,
    *,
    run_id: str | None = None,
    event_type: str = "COMPLETE",
    inputs: Iterable[Mapping[str, Any]] = (),
    outputs: Iterable[Mapping[str, Any]] = (),
    namespace: str = "mlforensics",
    event_time: str | None = None,
    facets: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a serializable OpenLineage-style run event."""
    return {
        "eventType": event_type,
        "eventTime": event_time or datetime.now(timezone.utc).isoformat(),
        "run": {"runId": run_id or str(uuid4())},
        "job": {"namespace": namespace, "name": job_name, "facets": dict(facets or {})},
        "inputs": [dict(item) for item in inputs],
        "outputs": [dict(item) for item in outputs],
        "producer": "mlforensics",
        "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json#/$defs/RunEvent",
    }


def export_openlineage_events(
    graph: DependencyGraph,
    node_ids: Iterable[str] | None = None,
    *,
    run_id: str | None = None,
    event_type: str = "COMPLETE",
    namespace: str = "mlforensics",
) -> list[dict[str, Any]]:
    """Create one plain-dictionary event per selected graph node.

    Dependencies become inputs and downstream-produced nodes become outputs;
    arbitrary code nodes are still represented as jobs, making the export safe
    when metadata is incomplete.
    """
    selected = list(node_ids) if node_ids is not None else sorted(graph.nodes)
    events = []
    for node_id in selected:
        node = graph.get_node(node_id)
        if node is None:
            continue
        inputs = [_dataset(dep, namespace) for dep in graph.successors(node.id)]
        outputs = [_dataset(dep, namespace) for dep in graph.predecessors(node.id)]
        events.append(
            openlineage_event(
                node.name or node.id,
                run_id=run_id,
                event_type=event_type,
                inputs=inputs,
                outputs=outputs,
                namespace=namespace,
                facets={"mlforensics": {"nodeId": node.id, "kind": node.kind}},
            )
        )
    return events


export_openlineage_event = openlineage_event
