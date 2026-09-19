"""Build dataset/feature/model edges from lightweight configuration mappings."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .graph import DependencyGraph, Node


def _read_config(value: Mapping[str, Any] | str | os.PathLike[str]) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        path = Path(value)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".toml":
            try:
                import tomllib
            except ImportError:  # pragma: no cover - Python 3.10 compatibility
                import tomli as tomllib
            loaded = tomllib.loads(text)
        else:
            loaded = json.loads(text)
        return loaded if isinstance(loaded, Mapping) else {}
    except (OSError, UnicodeError, ValueError, TypeError):
        return {}


def _entries(section: Any) -> list[tuple[str, Mapping[str, Any]]]:
    if isinstance(section, Mapping):
        result = []
        for name, value in section.items():
            result.append((str(name), value if isinstance(value, Mapping) else {}))
        return result
    if isinstance(section, list):
        result = []
        for value in section:
            if isinstance(value, Mapping):
                name = value.get("name") or value.get("id")
                if name is not None:
                    result.append((str(name), value))
        return result
    return []


def _names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return (
            [str(value.get("name") or value.get("id"))]
            if value.get("name") or value.get("id")
            else []
        )
    if isinstance(value, (list, tuple, set)):
        return [
            str(item.get("name") or item.get("id"))
            if isinstance(item, Mapping) and (item.get("name") or item.get("id"))
            else str(item)
            for item in value
            if isinstance(item, (str, int, float, Mapping))
        ]
    return []


def load_relationships(
    config: Mapping[str, Any] | str | os.PathLike[str], graph: DependencyGraph | None = None
) -> DependencyGraph:
    """Load relationships, ignoring malformed or absent metadata.

    Configuration sections may be mappings or lists.  ``inputs``, ``features``,
    ``datasets``, ``depends_on``, ``uses``, and ``produces`` are accepted as
    conventional relationship fields.
    """
    data = _read_config(config)
    graph = graph or DependencyGraph()
    sections = (("datasets", "dataset"), ("features", "feature"), ("models", "model"))
    for section, kind in sections:
        for name, metadata in _entries(data.get(section)):
            configured_path = metadata.get("path") or metadata.get("source")
            graph.add_node(
                Node(
                    f"{kind}:{name}",
                    kind,
                    name,
                    str(configured_path) if configured_path else None,
                    dict(metadata),
                )
            )

    def ensure(name: str, kind: str) -> str:
        node_id = name if ":" in name else f"{kind}:{name}"
        graph.add_node(Node(node_id, kind, name.split(":", 1)[-1], None))
        return node_id

    # A consumer points at what it needs, matching the code graph direction.
    for section, kind in sections:
        for name, metadata in _entries(data.get(section)):
            consumer = ensure(name, kind)
            implementation = metadata.get("implemented_by") or metadata.get("path")
            if implementation:
                values = _names(implementation)
                for value in values:
                    target = value if value.startswith(("file:", "module:")) else None
                    if target is None:
                        normalized = value.replace("\\", "/").lstrip("./")
                        target = next(
                            (
                                node.id
                                for node in graph
                                if node.kind in {"file", "module"}
                                and node.path
                                and (
                                    node.path.replace("\\", "/").lstrip("./") == normalized
                                    or node.path.replace("\\", "/").endswith("/" + normalized)
                                )
                            ),
                            f"file:{value}",
                        )
                    graph.add_edge(
                        consumer,
                        target,
                        "implemented_by",
                        {
                            "confidence": "high",
                            "explanation": "configured implementation path",
                        },
                    )
            fields = (
                "inputs",
                "depends_on",
                "dependencies",
                "uses",
                "features",
                "uses_features",
                "feature_inputs",
                "datasets",
                "dataset_inputs",
            )
            for field in fields:
                for dependency in _names(metadata.get(field)):
                    if dependency.partition(":")[0] in {"dataset", "feature", "model"}:
                        dependency_kind = dependency.partition(":")[0]
                    elif field in {"features", "uses_features", "feature_inputs"}:
                        dependency_kind = "feature"
                    elif field in {"datasets", "dataset_inputs"} or kind == "feature":
                        dependency_kind = "dataset"
                    else:
                        dependency_kind = "feature"
                    graph.add_edge(
                        consumer,
                        ensure(dependency, dependency_kind),
                        "configured",
                        {
                            "field": field,
                            "confidence": "high",
                            "explanation": f"configured {field} relationship",
                        },
                    )
            for produced in _names(metadata.get("produces")):
                # A producer declaration on a dataset/feature means the named
                # artifact depends on this entry.
                produced_kind = "feature" if kind == "dataset" else "model"
                graph.add_edge(
                    ensure(produced, produced_kind),
                    consumer,
                    "configured",
                    {
                        "field": "produces",
                        "confidence": "high",
                        "explanation": "configured producer relationship",
                    },
                )

    for relationship in (
        data.get("relationships", []) if isinstance(data.get("relationships"), list) else []
    ):
        if not isinstance(relationship, Mapping):
            continue
        source, target = relationship.get("source"), relationship.get("target")
        if source is None or target is None:
            continue

        def resolve_reference(value: Any) -> str:
            text = str(value)
            if text in graph.nodes:
                return text
            for kind in ("dataset", "feature", "model"):
                candidate = f"{kind}:{text}"
                if candidate in graph.nodes:
                    return candidate
            return text

        source_id = resolve_reference(source)
        target_id = resolve_reference(target)
        graph.add_edge(
            source_id,
            target_id,
            str(relationship.get("kind", "configured")),
            {
                **{k: v for k, v in relationship.items() if k not in {"source", "target", "kind"}},
                "confidence": relationship.get("confidence", "medium"),
                "explanation": relationship.get("explanation", "configured relationship"),
            },
        )
    return graph


build_configured_graph = load_relationships
