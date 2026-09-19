"""Downstream impact planning and validation recommendations."""

from __future__ import annotations

import fnmatch
import subprocess
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import load_relationships
from .diff import DiffChange, GitDiffImpactExtractor, parse_git_diff
from .graph import DependencyGraph, Node
from .python import analyze_python as analyze_python_tree


@dataclass
class ValidationRecommendation:
    check: str
    reason: str
    priority: str = "medium"
    node_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "reason": self.reason,
            "priority": self.priority,
            "node_ids": list(self.node_ids),
        }


@dataclass
class AffectedNodePlan:
    changed_nodes: list[str]
    affected_nodes: list[str]
    distances: dict[str, int] = field(default_factory=dict)
    recommendations: list[ValidationRecommendation] = field(default_factory=list)

    @property
    def validation(self) -> list[ValidationRecommendation]:
        return self.recommendations

    def as_dict(self) -> dict[str, Any]:
        return {
            "changed_nodes": list(self.changed_nodes),
            "affected_nodes": list(self.affected_nodes),
            "distances": dict(self.distances),
            "recommendations": [item.as_dict() for item in self.recommendations],
        }


@dataclass
class ImpactReport(AffectedNodePlan):
    """Compatibility report returned by the repository-level helpers."""

    changed_files: list[str] = field(default_factory=list)
    direct_dependents: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    recommended_validation: list[str] = field(default_factory=list)
    skipped_validation: list[str] = field(default_factory=list)
    graph: dict[str, Any] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = self.as_dict()
        data.update(
            {
                "changed_files": self.changed_files,
                "direct_dependents": self.direct_dependents,
                "datasets": self.datasets,
                "models": self.models,
                "recommended_validation": self.recommended_validation,
                "skipped_validation": self.skipped_validation,
                "graph": self.graph,
                "errors": self.errors,
            }
        )
        return data


def _recommendations(
    graph: DependencyGraph, node_ids: Iterable[str]
) -> list[ValidationRecommendation]:
    groups: dict[str, list[str]] = {}
    for node_id in node_ids:
        node = graph.get_node(node_id)
        groups.setdefault(node.kind if node else "unknown", []).append(node_id)
    rules = {
        "dataset": (
            "data_quality_and_schema",
            "Check schema, null/range constraints, and representative data.",
            "high",
        ),
        "feature": (
            "feature_recompute_and_distribution",
            "Recompute affected features and compare distributions or reference values.",
            "high",
        ),
        "model": (
            "model_regression_and_slices",
            "Run evaluation, drift, and important slice/regression checks.",
            "high",
        ),
        "module": (
            "import_and_integration_smoke",
            "Run import/compile checks and integration tests covering this module.",
            "medium",
        ),
        "file": (
            "import_and_integration_smoke",
            "Run import/compile checks and integration tests covering this file.",
            "medium",
        ),
        "function": (
            "focused_unit_tests",
            "Run focused unit tests and callers of the changed function.",
            "medium",
        ),
        "class": (
            "focused_unit_tests",
            "Run focused unit tests and integration checks for this class.",
            "medium",
        ),
        "import": (
            "import_and_integration_smoke",
            "Verify the dependency is importable and exercise its boundary.",
            "medium",
        ),
    }
    result = []
    for kind in sorted(groups):
        result.append(
            ValidationRecommendation(
                *rules.get(
                    kind,
                    (
                        "smoke_and_regression_tests",
                        "Run smoke and nearest regression tests.",
                        "low",
                    ),
                ),
                sorted(groups[kind]),
            )
        )
    return result


def recommend_validation(
    graph: DependencyGraph, node_ids: Iterable[str]
) -> list[ValidationRecommendation]:
    return _recommendations(graph, node_ids)


class ImpactPlanner:
    def __init__(self, graph: DependencyGraph, mappings: Mapping[str, Any] | None = None) -> None:
        self.graph, self.mappings = graph, mappings or {}

    def _roots(self, changed: Iterable[str | Node | DiffChange]) -> set[str]:
        roots: set[str] = set()

        def normalized_path(value: str) -> str:
            return value.replace("\\", "/").lstrip("./")

        def module_aliases(value: str) -> tuple[str, ...]:
            path = normalized_path(value)
            if not path.endswith(".py"):
                return ()
            module = path[:-3].replace("/", ".")
            if module.endswith(".__init__"):
                module = module[:-9]
            return tuple(
                dict.fromkeys(
                    (
                        f"module:{path}",
                        f"module:{module}",
                    )
                )
            )

        def add_unknown_change(value: str) -> str:
            path = normalized_path(value)
            kind = (
                "notebook"
                if path.endswith(".ipynb")
                else "configuration"
                if path.endswith((".toml", ".yaml", ".yml", ".json", ".ini", ".cfg"))
                else "data"
                if path.endswith((".csv", ".parquet", ".sql", ".dvc"))
                else "generated"
                if path.endswith((".lock", ".generated"))
                else "unknown"
            )
            node_id = f"unknown:change:{path}"
            self.graph.add_node(
                Node(
                    node_id,
                    "unknown",
                    path,
                    path,
                    {"change_kind": kind, "conservative": True},
                )
            )
            # Runtime/configuration/data changes can affect code that is not
            # statically connected.  Preserve that uncertainty explicitly so
            # the planner cannot report an unjustified empty impact set.
            for node in list(self.graph):
                if node.id == node_id or node.kind == "unknown":
                    continue
                if node.kind in {"file", "module", "dataset", "feature", "model"}:
                    self.graph.add_edge(
                        node.id,
                        node_id,
                        "conservative",
                        {
                            "confidence": "low",
                            "explanation": f"possible dependency on changed {kind} {path}",
                        },
                    )
            return node_id

        def add_root(value: str) -> None:
            if value not in self.graph.nodes:
                return
            roots.add(value)
            # A changed module/file also changes its locally defined symbols.
            # Import edges often terminate at ``module:path:symbol`` nodes,
            # so retaining those as roots is what lets impact propagation reach
            # downstream model consumers.
            if value.startswith("module:"):
                prefix = value + ":"
                roots.update(node_id for node_id in self.graph.nodes if node_id.startswith(prefix))
            if value.startswith("file:"):
                # File nodes contain symbols, but the graph points from a
                # consumer to its dependency.  Expand a whole-file change to
                # its contained symbols before walking predecessors so callers
                # of imported functions are included in the impact set.
                roots.update(
                    edge.target
                    for edge in self.graph.edges.values()
                    if edge.source == value and edge.kind == "contains"
                )

        for item in changed:
            if isinstance(item, DiffChange):
                matched_node_ids = GitDiffImpactExtractor.changed_node_ids([item], self.graph)
                for node_id in matched_node_ids:
                    add_root(node_id)
                all_paths = [item.path, *([item.old_path] if item.old_path else [])]
                for path in all_paths:
                    missing_source = item.is_deleted or (
                        item.old_path == path and item.status.upper().startswith(("R", "C"))
                    )
                    aliases = tuple(
                        alias for alias in module_aliases(path) if alias in self.graph.nodes
                    )
                    if aliases:
                        roots.update(aliases)
                    elif path.endswith(".py") and missing_source:
                        # Deleted files are absent from the current tree, but
                        # unresolved imports still use dotted module IDs.
                        for alias in module_aliases(path):
                            self.graph.add_node(
                                Node(
                                    alias,
                                    "module",
                                    alias.removeprefix("module:"),
                                    normalized_path(path),
                                    {"deleted_or_renamed": True, "conservative": True},
                                )
                            )
                            roots.add(alias)
                    elif not matched_node_ids:
                        roots.add(add_unknown_change(path))
                    if path.endswith(".py") and missing_source:
                        module_name = normalized_path(path)[:-3].replace("/", ".")
                        if module_name.endswith(".__init__"):
                            module_name = module_name[:-9]
                        unresolved_id = f"unknown:module:{module_name}"
                        self.graph.add_node(
                            Node(
                                unresolved_id,
                                "unknown",
                                module_name,
                                normalized_path(path),
                                {"deleted_or_renamed": True, "conservative": True},
                            )
                        )
                        roots.add(unresolved_id)
                        roots.update(
                            node.id
                            for node in self.graph
                            if node.id.startswith(f"{module_name}:")
                            or node.id.startswith(f"module:{module_name}:")
                        )
                continue
            value = item.id if isinstance(item, Node) else str(item)
            if value in self.graph.nodes:
                add_root(value)
                continue
            matched = False
            normalized = value.replace("\\", "/")
            for node in self.graph:
                if not node.path:
                    continue
                node_path = node.path.replace("\\", "/")
                absolute_value = str(Path(value).resolve()).replace("\\", "/")
                if node_path in {normalized, absolute_value} or node_path.endswith(
                    "/" + normalized.lstrip("/")
                ):
                    add_root(node.id)
                    matched = True
            if not matched and value.startswith("module:"):
                roots.add(value)
            elif not matched:
                roots.add(add_unknown_change(value))
        return roots

    def plan(
        self, changed: Iterable[str | Node | DiffChange] | str | Node | DiffChange
    ) -> AffectedNodePlan:
        if isinstance(changed, (str, Node, DiffChange)):
            changed = [changed]
        roots = self._roots(changed)
        distances = {node_id: 0 for node_id in roots}
        queue = deque(roots)
        while queue:
            current = queue.popleft()
            for node in self.graph.predecessors(current):
                if node.id not in distances:
                    distances[node.id] = distances[current] + 1
                    queue.append(node.id)
        affected = sorted(distances, key=lambda item: (distances[item], item))
        return AffectedNodePlan(
            sorted(roots), affected, distances, _recommendations(self.graph, affected)
        )

    def report(self, changed_files: Iterable[str | DiffChange]) -> ImpactReport:
        changes = list(changed_files)
        paths = sorted(
            {
                path
                for item in changes
                for path in (
                    (item.path, item.old_path) if isinstance(item, DiffChange) else (str(item),)
                )
                if path
            }
        )
        roots = self._roots(changes)
        plan = self.plan(changes)
        direct = sorted({node.id for root in roots for node in self.graph.predecessors(root)})
        datasets, models = self._configured_targets(set(plan.affected_nodes), paths)
        names = [item.check for item in plan.recommendations]
        validations = (
            self.mappings.get("validations", {}) if isinstance(self.mappings, Mapping) else {}
        )
        for name, patterns in validations.items() if isinstance(validations, Mapping) else []:
            patterns = patterns if isinstance(patterns, list) else [patterns]
            if any(fnmatch.fnmatch(path, pattern) for path in paths for pattern in patterns):
                names.append(str(name))
        all_models = self.mappings.get("models", {}) if isinstance(self.mappings, Mapping) else {}
        all_models = set(all_models) if isinstance(all_models, Mapping) else set()
        return ImpactReport(
            plan.changed_nodes,
            plan.affected_nodes,
            plan.distances,
            plan.recommendations,
            paths,
            direct,
            datasets,
            models,
            sorted(set(names)),
            sorted(all_models - set(models)),
            self.graph.to_dict(),
            [],
        )

    def _configured_targets(
        self, affected: set[str], changed: list[str]
    ) -> tuple[list[str], list[str]]:
        datasets: set[str] = {
            node.name or node.id.split(":", 1)[-1]
            for node_id in affected
            if (node := self.graph.get_node(node_id)) is not None and node.kind == "dataset"
        }
        models: set[str] = {
            node.name or node.id.split(":", 1)[-1]
            for node_id in affected
            if (node := self.graph.get_node(node_id)) is not None and node.kind == "model"
        }
        for section, output in (("datasets", datasets), ("models", models)):
            entries = self.mappings.get(section, {}) if isinstance(self.mappings, Mapping) else {}
            if not isinstance(entries, Mapping):
                continue
            for name, config in entries.items():
                values = (
                    config
                    if isinstance(config, list)
                    else config.get("depends_on", [])
                    if isinstance(config, Mapping)
                    else [config]
                )
                if any(
                    str(dep) in affected or any(str(dep) in path for path in changed)
                    for dep in values
                ):
                    output.add(str(name))
        return sorted(datasets), sorted(models)


def changed_files(base: str, head: str = "HEAD", *, root: str | Path = ".") -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}..{head}"],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return [line for line in result.stdout.splitlines() if line]


def analyze_impact(
    root: str | Path,
    changed: Iterable[str | DiffChange],
    *,
    mappings: Mapping[str, Any] | None = None,
) -> ImpactReport:
    graph, errors = analyze_python_tree(root)
    if mappings:
        load_relationships(mappings, graph)
    report = ImpactPlanner(graph, mappings).report(changed)
    report.errors.extend(errors)
    return report


def impact_from_git(
    root: str | Path, base: str, head: str = "HEAD", *, mappings: Mapping[str, Any] | None = None
) -> ImpactReport:
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--find-renames", f"{base}..{head}"],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    changes = parse_git_diff(result.stdout)
    return analyze_impact(root, changes, mappings=mappings)
