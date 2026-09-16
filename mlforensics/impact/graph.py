"""Dependency and lineage graph primitives with no external dependencies."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    id: str
    kind: str = "unknown"
    name: str | None = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def node_id(self) -> str:
        return self.id

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "node_id": self.id,
            "kind": self.kind,
            "name": self.name,
            "path": self.path,
            "metadata": dict(self.metadata),
        }

    to_dict = as_dict


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: str = "depends_on"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def relation(self) -> str:
        return self.kind

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
            "relation": self.kind,
            "metadata": dict(self.metadata),
        }

    to_dict = as_dict


class EdgeCollection(dict[tuple[str, str, str], Edge]):
    """Dict-like edge store that also iterates over edge objects."""

    def __iter__(self) -> Iterator[Edge]:
        return iter(self.values())


class DependencyGraph:
    """A directed graph whose consumer nodes point to their dependencies."""

    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.edges: EdgeCollection = EdgeCollection()

    def add_node(
        self,
        node: Node | str,
        kind: str = "unknown",
        name: str | None = None,
        path: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> Node:
        if isinstance(node, str):
            node = Node(node, kind, name, path, {**dict(metadata or {}), **extra})
        existing = self.nodes.get(node.id)
        if existing:
            if node.kind != "unknown":
                existing.kind = node.kind
            if node.name is not None:
                existing.name = node.name
            if node.path is not None:
                existing.path = node.path
            existing.metadata.update(node.metadata)
            return existing
        self.nodes[node.id] = node
        return node

    def add_edge(
        self,
        source: str | Node,
        target: str | Node,
        kind: str = "depends_on",
        metadata: Mapping[str, Any] | None = None,
        **extra: Any,
    ) -> Edge:
        source_id = source.id if isinstance(source, Node) else str(source)
        target_id = target.id if isinstance(target, Node) else str(target)
        self.add_node(source_id)
        self.add_node(target_id)
        edge = Edge(source_id, target_id, kind, {**dict(metadata or {}), **extra})
        self.edges[(source_id, target_id, kind)] = edge
        return edge

    add_dependency = add_edge

    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def successors(self, node_id: str, kinds: Iterable[str] | None = None) -> list[Node]:
        accepted = set(kinds) if kinds is not None else None
        ids = {
            edge.target
            for edge in self.edges.values()
            if edge.source == node_id and (accepted is None or edge.kind in accepted)
        }
        return [self.nodes[node] for node in sorted(ids)]

    def predecessors(self, node_id: str, kinds: Iterable[str] | None = None) -> list[Node]:
        accepted = set(kinds) if kinds is not None else None
        ids = {
            edge.source
            for edge in self.edges.values()
            if edge.target == node_id and (accepted is None or edge.kind in accepted)
        }
        return [self.nodes[node] for node in sorted(ids)]

    def outgoing_edges(self, node_id: str) -> list[Edge]:
        return sorted(
            (edge for edge in self.edges.values() if edge.source == node_id),
            key=lambda edge: (edge.target, edge.kind),
        )

    def incoming_edges(self, node_id: str) -> list[Edge]:
        return sorted(
            (edge for edge in self.edges.values() if edge.target == node_id),
            key=lambda edge: (edge.source, edge.kind),
        )

    def downstream(self, roots: Iterable[str], include_roots: bool = True) -> set[str]:
        """Return roots and every transitive consumer of those roots."""
        seen = set(roots) if include_roots else set()
        queue = deque(roots)
        while queue:
            current = queue.popleft()
            for node in self.predecessors(current):
                if node.id not in seen:
                    seen.add(node.id)
                    queue.append(node.id)
        return seen

    def upstream(self, roots: Iterable[str], include_roots: bool = True) -> set[str]:
        seen = set(roots) if include_roots else set()
        queue = deque(roots)
        while queue:
            current = queue.popleft()
            for node in self.successors(current):
                if node.id not in seen:
                    seen.add(node.id)
                    queue.append(node.id)
        return seen

    def affected(self, roots: Iterable[str], include_roots: bool = True) -> set[str]:
        return self.downstream(roots, include_roots)

    # Compatibility names retained for the earlier file/symbol API.
    def dependents(self, node_id: str, *, transitive: bool = True) -> set[str]:
        return self._walk(node_id, outgoing=True, transitive=transitive)

    def dependencies(self, node_id: str, *, transitive: bool = True) -> set[str]:
        return self._walk(node_id, outgoing=False, transitive=transitive)

    def _walk(self, node_id: str, *, outgoing: bool, transitive: bool) -> set[str]:
        found: set[str] = set()
        queue = deque([node_id])
        while queue:
            current = queue.popleft()
            neighbors = self.successors(current) if outgoing else self.predecessors(current)
            for node in neighbors:
                if node.id not in found:
                    found.add(node.id)
                    if transitive:
                        queue.append(node.id)
        return found

    def subgraph(self, node_ids: Iterable[str]) -> DependencyGraph:
        selected = set(node_ids)
        graph = DependencyGraph()
        for node_id in selected:
            if node_id in self.nodes:
                graph.add_node(self.nodes[node_id])
        for edge in self.edges.values():
            if edge.source in selected and edge.target in selected:
                graph.add_edge(edge.source, edge.target, edge.kind, edge.metadata)
        return graph

    def __len__(self) -> int:
        return len(self.nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": [self.nodes[node_id].as_dict() for node_id in sorted(self.nodes)],
            "edges": [
                edge.as_dict()
                for edge in sorted(
                    self.edges.values(), key=lambda item: (item.source, item.target, item.kind)
                )
            ],
        }

    to_dict = as_dict

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DependencyGraph:
        graph = cls()
        for value in data.get("nodes", []):
            if not isinstance(value, Mapping):
                continue
            node_id = value.get("id", value.get("node_id"))
            if node_id is None:
                continue
            graph.add_node(
                Node(
                    str(node_id),
                    str(value.get("kind", "unknown")),
                    value.get("name"),
                    value.get("path"),
                    dict(value.get("metadata", {})),
                )
            )
        for value in data.get("edges", []):
            if (
                not isinstance(value, Mapping)
                or value.get("source") is None
                or value.get("target") is None
            ):
                continue
            graph.add_edge(
                str(value["source"]),
                str(value["target"]),
                str(value.get("kind", value.get("relation", "depends_on"))),
                value.get("metadata", {}),
            )
        return graph


Graph = DependencyGraph
