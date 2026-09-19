"""Dependency-graph bridges for DVC and OpenLineage-shaped data.

These helpers intentionally operate on plain dictionaries and the local
``DependencyGraph``.  Neither the DVC CLI nor the OpenLineage client is
required, and importing data never starts a service or follows a referenced
path.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from .graph import DependencyGraph, Node
from .readers import load_structured

_DEFAULT_EVENT_TIME = "1970-01-01T00:00:00+00:00"


def _json_safe(value: Any) -> Any:
    """Convert metadata to deterministic JSON-compatible values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(value[key]) for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        values = [_json_safe(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted(values, key=lambda item: json.dumps(item, sort_keys=True, default=str))
        return values
    return str(value)


def _records(value: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if value is None:
        return []
    source: Iterable[Any] = [value] if isinstance(value, Mapping) else value
    result = [_json_safe(dict(item)) for item in source if isinstance(item, Mapping)]
    return sorted(result, key=lambda item: json.dumps(item, sort_keys=True, default=str))


def _dataset(node: Node, namespace: str) -> dict[str, Any]:
    facets = _json_safe(dict(node.metadata))
    if not isinstance(facets, dict):
        facets = {}
    facets.update({"kind": node.kind, "nodeId": node.id})
    return {
        "namespace": namespace,
        "name": node.id,
        "facets": {"mlforensics": facets},
    }


def _stable_run_id(
    job_name: str,
    *,
    event_type: str,
    namespace: str,
    inputs: Iterable[Mapping[str, Any]],
    outputs: Iterable[Mapping[str, Any]],
) -> str:
    payload = {
        "eventType": event_type,
        "job": {"namespace": namespace, "name": job_name},
        "inputs": _records(inputs),
        "outputs": _records(outputs),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return str(uuid5(NAMESPACE_URL, "mlforensics:" + encoded))


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
    """Return a serializable OpenLineage-style run event.

    Defaults are stable so repeated exports of the same graph produce the
    same event.  Callers that need wall-clock event time or a run identity can
    continue to provide ``event_time`` and ``run_id`` explicitly.
    """

    input_records = _records(inputs)
    output_records = _records(outputs)
    normalized_name = str(job_name)
    normalized_type = str(event_type)
    normalized_namespace = str(namespace)
    stable_id = run_id or _stable_run_id(
        normalized_name,
        event_type=normalized_type,
        namespace=normalized_namespace,
        inputs=input_records,
        outputs=output_records,
    )
    job_facets = _json_safe(dict(facets or {}))
    if not isinstance(job_facets, dict):
        job_facets = {}
    return {
        "eventType": normalized_type,
        "eventTime": event_time or _DEFAULT_EVENT_TIME,
        "run": {"runId": str(stable_id)},
        "job": {
            "namespace": normalized_namespace,
            "name": normalized_name,
            "facets": job_facets,
        },
        "inputs": input_records,
        "outputs": output_records,
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
    """Create deterministic OpenLineage-shaped events from graph nodes.

    The graph convention is ``consumer -> dependency``.  Therefore outgoing
    graph neighbors are event inputs and incoming neighbors are event outputs.
    Missing selected node IDs are ignored, matching the original API.
    """

    selected = sorted(set(node_ids)) if node_ids is not None else sorted(graph.nodes)
    events: list[dict[str, Any]] = []
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


def _lineage_value(value: Any, fallback: str = "") -> str:
    if isinstance(value, (str, int, float)):
        return str(value).strip()
    return fallback


def _lineage_dataset_id(dataset: Mapping[str, Any], default_namespace: str) -> tuple[str, str, str]:
    facets = dataset.get("facets", {})
    ml_facets = facets.get("mlforensics", {}) if isinstance(facets, Mapping) else {}
    original_id = ml_facets.get("nodeId") if isinstance(ml_facets, Mapping) else None
    namespace = _lineage_value(dataset.get("namespace"), default_namespace) or default_namespace
    name = _lineage_value(dataset.get("name"))
    node_id = _lineage_value(original_id) or f"dataset:{namespace}:{name}"
    kind = (
        _lineage_value(ml_facets.get("kind"), "dataset")
        if isinstance(ml_facets, Mapping)
        else "dataset"
    )
    return node_id, kind, name or node_id


def _lineage_job_id(job: Mapping[str, Any], default_namespace: str) -> tuple[str, str, str]:
    facets = job.get("facets", {})
    ml_facets = facets.get("mlforensics", {}) if isinstance(facets, Mapping) else {}
    original_id = ml_facets.get("nodeId") if isinstance(ml_facets, Mapping) else None
    namespace = _lineage_value(job.get("namespace"), default_namespace) or default_namespace
    name = _lineage_value(job.get("name"))
    node_id = _lineage_value(original_id) or f"job:{namespace}:{name}"
    kind = _lineage_value(ml_facets.get("kind"), "job") if isinstance(ml_facets, Mapping) else "job"
    return node_id, kind, name or node_id


def _event_sequence(
    events: Any,
    *,
    root: str | os.PathLike[str] | None = None,
) -> list[Mapping[str, Any]]:
    if isinstance(events, Mapping):
        if isinstance(events.get("events"), list):
            return [item for item in events["events"] if isinstance(item, Mapping)]
        return [events]
    if isinstance(events, (str, os.PathLike)):
        value = str(events)
        if value.lstrip().startswith(("{", "[")):
            # JSON text is data, even when a root boundary is supplied.  Do
            # not reinterpret braces or dataset names as a filesystem path.
            text = value
        else:
            try:
                candidate = Path(value).expanduser().resolve()
            except (OSError, ValueError):
                candidate = None
            if candidate is not None and root is not None:
                root_path = Path(root).expanduser().resolve()
                if candidate != root_path and root_path not in candidate.parents:
                    return []
            try:
                text = candidate.read_text(encoding="utf-8") if candidate is not None else value
            except (OSError, UnicodeError, ValueError):
                # A JSON string is also a useful input when it is not a path.
                text = value
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            # OpenLineage exports are often JSONL.  Invalid lines are simply
            # ignored so one bad event cannot erase valid evidence.
            decoded_lines: list[Mapping[str, Any]] = []
            for line in text.splitlines():
                try:
                    item = json.loads(line)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                if isinstance(item, Mapping):
                    decoded_lines.append(item)
            return decoded_lines
        return _event_sequence(decoded, root=root)
    if isinstance(events, Iterable) and not isinstance(events, (bytes, bytearray)):
        return [item for item in events if isinstance(item, Mapping)]
    return []


def import_openlineage_events(
    events: Any,
    graph: DependencyGraph | None = None,
    *,
    namespace: str = "mlforensics",
    relation_prefix: str = "openlineage",
    root: str | os.PathLike[str] | None = None,
) -> DependencyGraph:
    """Import OpenLineage-shaped events without an OpenLineage dependency.

    Inputs become ``job -> dataset`` edges and outputs become
    ``dataset -> job`` edges, preserving the dependency direction used by
    ``DependencyGraph``.  Malformed events and duplicate relationships are
    ignored safely; graph edge keys provide an additional deterministic
    de-duplication boundary.
    """

    graph = graph if graph is not None else DependencyGraph()
    parsed = _event_sequence(events, root=root)
    canonical_events = sorted(
        parsed,
        key=lambda item: json.dumps(_json_safe(dict(item)), sort_keys=True, default=str),
    )
    for event in canonical_events:
        job = event.get("job")
        if not isinstance(job, Mapping):
            continue
        job_id, job_kind, job_name = _lineage_job_id(job, namespace)
        graph.add_node(
            Node(
                job_id,
                job_kind,
                job_name,
                None,
                {
                    "source_type": "openlineage",
                    "event_type": _lineage_value(event.get("eventType")),
                },
            )
        )
        run = event.get("run")
        run_id = _lineage_value(run.get("runId")) if isinstance(run, Mapping) else ""
        for direction, relation in (
            ("inputs", f"{relation_prefix}_input"),
            ("outputs", f"{relation_prefix}_output"),
        ):
            values = event.get(direction)
            if not isinstance(values, list):
                continue
            datasets = sorted(
                (item for item in values if isinstance(item, Mapping)),
                key=lambda item: json.dumps(_json_safe(dict(item)), sort_keys=True, default=str),
            )
            for dataset in datasets:
                dataset_id, dataset_kind, dataset_name = _lineage_dataset_id(dataset, namespace)
                facets = dataset.get("facets", {})
                metadata = {
                    "source_type": "openlineage",
                    "namespace": _lineage_value(dataset.get("namespace"), namespace),
                    "run_id": run_id,
                }
                if isinstance(facets, Mapping):
                    metadata["facets"] = _json_safe(facets)
                graph.add_node(Node(dataset_id, dataset_kind, dataset_name, None, metadata))
                source, target = (
                    (job_id, dataset_id) if direction == "inputs" else (dataset_id, job_id)
                )
                graph.add_edge(
                    source,
                    target,
                    relation,
                    {
                        "confidence": "high",
                        "explanation": f"OpenLineage event {direction[:-1]}",
                        "event_type": _lineage_value(event.get("eventType")),
                        "run_id": run_id,
                    },
                )
    return graph


import_openlineage_relationships = import_openlineage_events
read_openlineage_events = import_openlineage_events
read_openlineage = import_openlineage_events


# ---------------------------------------------------------------------------
# DVC-shaped graph import/export


def _dvc_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in ("path", "uri", "name"):
            if value.get(key) is not None:
                return [str(value[key])]
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(_dvc_values(item))
        return result
    if isinstance(value, (str, int, float)):
        return [str(value)]
    return []


def _dvc_stage_mapping(data: Mapping[str, Any]) -> Mapping[str, Any]:
    stages = data.get("stages")
    return stages if isinstance(stages, Mapping) else {}


def import_dvc_graph(
    source: Any,
    graph: DependencyGraph | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    path: str | os.PathLike[str] | None = None,
) -> DependencyGraph:
    """Import a DVC ``stages`` graph from YAML/JSON-shaped data."""

    graph = graph if graph is not None else DependencyGraph()
    if isinstance(source, Mapping):
        data, diagnostic = source, None
        display_path = str(path or "<dvc>")
    else:
        data, diagnostic = load_structured(source, path=path, root=root)
        if diagnostic is not None:
            display_path = diagnostic.path
        elif path is not None:
            display_path = str(path)
        elif isinstance(source, os.PathLike):
            display_path = str(source)
        elif isinstance(source, str) and "\n" not in source and Path(source).suffix:
            display_path = source
        else:
            display_path = "<dvc>"
    normalized_path = display_path.replace("\\", "/")
    root_id = f"file:{normalized_path}"
    root_metadata: dict[str, Any] = {"source_type": "dvc", "parseable": diagnostic is None}
    if diagnostic is not None:
        root_metadata["read_error"] = diagnostic.error
    graph.add_node(
        Node(root_id, "dvc", Path(display_path).name or display_path, display_path, root_metadata)
    )
    if not isinstance(data, Mapping):
        return graph
    stages = _dvc_stage_mapping(data)
    for name in sorted(stages, key=str):
        config = stages[name]
        if not isinstance(config, Mapping):
            continue
        stage_name = str(name)
        stage_id = f"pipeline:{stage_name}"
        stage_metadata = {"source_type": "dvc", **_json_safe(dict(config))}
        graph.add_node(Node(stage_id, "pipeline_stage", stage_name, display_path, stage_metadata))
        graph.add_edge(
            root_id,
            stage_id,
            "contains",
            {"confidence": "high", "explanation": "DVC declares a stage"},
        )
        for field in ("deps", "params"):
            for value in sorted(set(_dvc_values(config.get(field)))):
                normalized = value.replace("\\", "/")
                dataset_id = f"dataset:{normalized}"
                graph.add_node(
                    Node(
                        dataset_id,
                        "dataset",
                        normalized,
                        value,
                        {"source_type": "dvc", "field": field},
                    )
                )
                graph.add_edge(
                    stage_id,
                    dataset_id,
                    "dvc_dep",
                    {"field": field, "confidence": "high", "explanation": f"DVC {field} path"},
                )
        for field in ("outs", "metrics", "plots"):
            for value in sorted(set(_dvc_values(config.get(field)))):
                normalized = value.replace("\\", "/")
                dataset_id = f"dataset:{normalized}"
                graph.add_node(
                    Node(
                        dataset_id,
                        "dataset",
                        normalized,
                        value,
                        {"source_type": "dvc", "field": field},
                    )
                )
                graph.add_edge(
                    dataset_id,
                    stage_id,
                    "dvc_out",
                    {"field": field, "confidence": "high", "explanation": f"DVC {field} path"},
                )
    return graph


def _dvc_node_path(node: Node) -> str | None:
    if node.path and not node.path.startswith("<"):
        return str(node.path).replace("\\", "/")
    if node.name and not node.name.startswith(("<", "pipeline:")):
        return str(node.name).replace("\\", "/")
    for prefix in ("dataset:", "file:"):
        if node.id.startswith(prefix):
            value = node.id[len(prefix) :]
            return value if value and not value.startswith("<") else None
    return None


def _dvc_stage_nodes(graph: DependencyGraph, node_ids: Iterable[str] | None) -> list[Node]:
    selected = set(node_ids) if node_ids is not None else None
    return sorted(
        (
            node
            for node in graph
            if (selected is None or node.id in selected)
            and (
                node.kind in {"pipeline_stage", "dvc_stage", "stage"}
                or node.id.startswith("pipeline:")
            )
        ),
        key=lambda node: node.id,
    )


def export_dvc_graph(
    graph: DependencyGraph,
    path: str | os.PathLike[str] | None = None,
    *,
    node_ids: Iterable[str] | None = None,
) -> dict[str, Any] | Path:
    """Export pipeline-stage relationships in a DVC-shaped mapping.

    With no ``path`` this returns a plain dictionary.  With a path ending in
    ``.json`` JSON is written; all other suffixes use a small dependency-free
    YAML emitter and the written ``Path`` is returned.
    """

    stages: dict[str, dict[str, Any]] = {}
    for stage in _dvc_stage_nodes(graph, node_ids):
        deps: list[str] = []
        outs: list[str] = []
        for dependency in graph.successors(stage.id):
            if dependency.kind in {
                "pipeline_stage",
                "dvc_stage",
                "stage",
            } or dependency.id.startswith("pipeline:"):
                continue
            value = _dvc_node_path(dependency)
            if value is not None:
                deps.append(value)
        for output in graph.predecessors(stage.id):
            if output.kind in {"pipeline_stage", "dvc_stage", "stage"} or output.id.startswith(
                "pipeline:"
            ):
                continue
            if output.kind in {"dvc", "pipeline", "config"}:
                # The declaration file contains the stage but is not a DVC
                # output.  Only artifact-like predecessors belong in ``outs``.
                continue
            value = _dvc_node_path(output)
            if value is not None:
                outs.append(value)
        payload: dict[str, Any] = {}
        configured_command = stage.metadata.get("cmd")
        if isinstance(configured_command, str) and configured_command.strip():
            payload["cmd"] = configured_command.strip()
        if deps:
            payload["deps"] = sorted(set(deps))
        if outs:
            payload["outs"] = sorted(set(outs))
        stages[stage.name or stage.id] = payload
    result: dict[str, Any] = {"stages": {name: stages[name] for name in sorted(stages)}}
    if path is None:
        return result
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".json":
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        lines = ["stages:"]
        for name in sorted(stages):
            lines.append(f"  {name}:")
            payload = stages[name]
            for key in ("cmd", "deps", "outs"):
                value = payload.get(key)
                if value is None:
                    continue
                if isinstance(value, list):
                    lines.append(f"    {key}:")
                    for item in value:
                        escaped = str(item).replace("'", "''")
                        lines.append(f"      - '{escaped}'")
                else:
                    escaped = str(value).replace("'", "''")
                    lines.append(f"    {key}: '{escaped}'")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_dvc_graph(graph: DependencyGraph, path: str | os.PathLike[str]) -> Path:
    result = export_dvc_graph(graph, path)
    return result if isinstance(result, Path) else Path(path)


import_dvc_relationships = import_dvc_graph
read_dvc_graph = import_dvc_graph
read_dvc_dependencies = import_dvc_graph
import_dvc_dependencies = import_dvc_graph
export_dvc_relationships = export_dvc_graph
export_dvc_dependencies = export_dvc_graph
import_dvc = import_dvc_graph
export_dvc = export_dvc_graph
import_openlineage = import_openlineage_events
export_openlineage = export_openlineage_events


export_openlineage_event = openlineage_event


__all__ = [
    "export_dvc_graph",
    "export_dvc",
    "export_dvc_dependencies",
    "export_dvc_relationships",
    "export_openlineage_event",
    "export_openlineage_events",
    "export_openlineage",
    "import_dvc",
    "import_dvc_dependencies",
    "import_dvc_graph",
    "import_dvc_relationships",
    "import_openlineage_events",
    "import_openlineage",
    "import_openlineage_relationships",
    "openlineage_event",
    "read_dvc_graph",
    "read_dvc_dependencies",
    "read_openlineage",
    "read_openlineage_events",
    "write_dvc_graph",
]
