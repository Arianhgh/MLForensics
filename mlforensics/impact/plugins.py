"""Small, optional-dependency-free plugin registry for impact readers.

Plugins are ordinary Python objects.  Built-ins are intentionally thin
wrappers around :mod:`mlforensics.impact.readers` and :mod:`.lineage`; custom
plugins can be discovered through Python entry points when a caller opts in.
Discovery failures are diagnostics, not fatal analysis errors.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .graph import DependencyGraph
from .lineage import import_dvc_graph, import_openlineage_events
from .readers import (
    read_notebook_dependencies,
    read_pipeline_dependencies,
    read_shell_dependencies,
    read_sql_dependencies,
    reader_name,
)

PLUGIN_PROTOCOL_VERSION = "1"
ENTRY_POINT_GROUPS = (
    "mlforensics.impact",
    "mlforensics.impact_readers",
    "mlforensics.impact.plugins",
)


@runtime_checkable
class ImpactPlugin(Protocol):
    """Protocol implemented by a dependency reader plugin."""

    name: str
    kind: str
    priority: int

    def can_read(self, source: Any, *, format: str | None = None) -> bool: ...

    def read(
        self,
        source: Any,
        graph: DependencyGraph | None = None,
        *,
        root: str | os.PathLike[str] | None = None,
        path: str | os.PathLike[str] | None = None,
    ) -> DependencyGraph: ...


@dataclass(frozen=True)
class PluginInfo:
    """Stable metadata exposed by the registry without importing core APIs."""

    name: str
    kind: str = "impact_reader"
    protocol_version: str = PLUGIN_PROTOCOL_VERSION
    capabilities: tuple[str, ...] = ()
    optional_dependencies: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "protocol_version": self.protocol_version,
            "capabilities": list(self.capabilities),
            "optional_dependencies": list(self.optional_dependencies),
        }

    to_dict = as_dict


class _BuiltinPlugin:
    name = "builtin"
    kind = "impact_reader"
    priority = 100
    formats: tuple[str, ...] = ()

    @property
    def descriptor(self) -> PluginInfo:
        return PluginInfo(self.name, self.kind, PLUGIN_PROTOCOL_VERSION, self.formats)

    def can_read(self, source: Any, *, format: str | None = None) -> bool:
        if format is not None:
            return format.lower() in self.formats
        if isinstance(source, (str, os.PathLike)):
            return reader_name(source) in self.formats
        return False


class SQLDependencyPlugin(_BuiltinPlugin):
    name = "sql"
    formats = ("sql", "duckdb")
    priority = 80

    def read(
        self,
        source: Any,
        graph: DependencyGraph | None = None,
        *,
        root: str | os.PathLike[str] | None = None,
        path: str | os.PathLike[str] | None = None,
    ) -> DependencyGraph:
        return read_sql_dependencies(
            source,
            graph,
            root=root,
            path=path,
            dialect="duckdb" if path and Path(path).suffix.lower() == ".duckdb" else "sql",
        )


class DuckDBDependencyPlugin(SQLDependencyPlugin):
    name = "duckdb"
    formats = ("duckdb", "sql")
    priority = 79


class NotebookDependencyPlugin(_BuiltinPlugin):
    name = "notebook"
    formats = ("notebook", "ipynb")
    priority = 80

    def read(
        self,
        source: Any,
        graph: DependencyGraph | None = None,
        *,
        root: str | os.PathLike[str] | None = None,
        path: str | os.PathLike[str] | None = None,
    ) -> DependencyGraph:
        return read_notebook_dependencies(source, graph, root=root, path=path)


class ShellDependencyPlugin(_BuiltinPlugin):
    name = "shell"
    formats = ("shell", "sh", "bash")
    priority = 80

    def read(
        self,
        source: Any,
        graph: DependencyGraph | None = None,
        *,
        root: str | os.PathLike[str] | None = None,
        path: str | os.PathLike[str] | None = None,
    ) -> DependencyGraph:
        return read_shell_dependencies(source, graph, root=root, path=path)


class PipelineDependencyPlugin(_BuiltinPlugin):
    name = "pipeline"
    formats = ("pipeline", "config", "yaml", "toml", "json")
    priority = 50

    def read(
        self,
        source: Any,
        graph: DependencyGraph | None = None,
        *,
        root: str | os.PathLike[str] | None = None,
        path: str | os.PathLike[str] | None = None,
    ) -> DependencyGraph:
        return read_pipeline_dependencies(source, graph, root=root, path=path)


class DVCDependencyPlugin(PipelineDependencyPlugin):
    name = "dvc"
    formats = ("dvc", "yaml", "json")
    priority = 90

    def can_read(self, source: Any, *, format: str | None = None) -> bool:
        if format is not None:
            return format.lower() == "dvc"
        if isinstance(source, (str, os.PathLike)):
            return Path(source).name.lower() in {"dvc.yaml", "dvc.yml", "dvc.lock"}
        return isinstance(source, Mapping) and isinstance(source.get("stages"), Mapping)

    def read(
        self,
        source: Any,
        graph: DependencyGraph | None = None,
        *,
        root: str | os.PathLike[str] | None = None,
        path: str | os.PathLike[str] | None = None,
    ) -> DependencyGraph:
        return import_dvc_graph(source, graph, root=root, path=path)


class OpenLineageDependencyPlugin(_BuiltinPlugin):
    name = "openlineage"
    formats = ("openlineage", "lineage", "json", "jsonl")
    priority = 70

    def can_read(self, source: Any, *, format: str | None = None) -> bool:
        if format is not None:
            return format.lower() in {"openlineage", "lineage", "json", "jsonl"}
        if isinstance(source, Mapping):
            return "job" in source and ("inputs" in source or "outputs" in source)
        if isinstance(source, (str, os.PathLike)):
            return Path(source).name.lower().startswith(("lineage", "openlineage"))
        return False

    def read(
        self,
        source: Any,
        graph: DependencyGraph | None = None,
        *,
        root: str | os.PathLike[str] | None = None,
        path: str | os.PathLike[str] | None = None,
    ) -> DependencyGraph:
        return import_openlineage_events(source, graph, root=root)


# Short aliases are useful for entry-point declarations and keep the public
# spelling unsurprising for users who call these adapters directly.
SQLPlugin = SQLDependencyPlugin
DuckDBPlugin = DuckDBDependencyPlugin
NotebookPlugin = NotebookDependencyPlugin
ShellPlugin = ShellDependencyPlugin
PipelinePlugin = PipelineDependencyPlugin
DVCPlugin = DVCDependencyPlugin
OpenLineagePlugin = OpenLineageDependencyPlugin


def _plugin_name(plugin: Any) -> str:
    return str(getattr(plugin, "name", plugin.__class__.__name__))


def _plugin_priority(plugin: Any) -> int:
    try:
        return int(getattr(plugin, "priority", 0))
    except (TypeError, ValueError):
        return 0


class PluginRegistry:
    """Deterministic registry with explicit registration and safe dispatch."""

    def __init__(self, plugins: Iterable[ImpactPlugin] = ()) -> None:
        self._plugins: dict[str, ImpactPlugin] = {}
        self.errors: list[dict[str, str]] = []
        for plugin in plugins:
            self.register(plugin)

    @property
    def plugins(self) -> tuple[ImpactPlugin, ...]:
        return tuple(
            sorted(
                self._plugins.values(),
                key=lambda item: (-_plugin_priority(item), _plugin_name(item)),
            )
        )

    def register(self, plugin: ImpactPlugin, *, replace: bool = False) -> ImpactPlugin:
        name = _plugin_name(plugin)
        if not callable(getattr(plugin, "read", None)) or not callable(
            getattr(plugin, "can_read", None)
        ):
            raise TypeError("impact plugin must provide can_read() and read()")
        if name in self._plugins and not replace:
            raise ValueError(f"impact plugin already registered: {name}")
        self._plugins[name] = plugin
        return plugin

    def unregister(self, name: str) -> None:
        self._plugins.pop(name, None)

    def get(self, name: str) -> ImpactPlugin | None:
        return self._plugins.get(name)

    def resolve(self, source: Any, *, format: str | None = None) -> ImpactPlugin | None:
        candidates: list[ImpactPlugin] = []
        for plugin in self.plugins:
            try:
                if plugin.can_read(source, format=format):
                    candidates.append(plugin)
            except (AttributeError, TypeError, ValueError, OSError) as exc:
                self.errors.append({"plugin": _plugin_name(plugin), "error": str(exc)})
        return candidates[0] if candidates else None

    def read(
        self,
        source: Any,
        graph: DependencyGraph | None = None,
        *,
        format: str | None = None,
        root: str | os.PathLike[str] | None = None,
        path: str | os.PathLike[str] | None = None,
    ) -> DependencyGraph:
        plugin = self.resolve(source, format=format)
        if plugin is None:
            # The default pipeline reader is intentionally the final fallback
            # for mappings and structured text; it only inspects known keys.
            return read_pipeline_dependencies(source, graph, root=root, path=path)
        try:
            return plugin.read(source, graph, root=root, path=path)
        except (OSError, UnicodeError, TypeError, ValueError) as exc:
            self.errors.append({"plugin": _plugin_name(plugin), "error": str(exc)})
            return graph if graph is not None else DependencyGraph()

    def descriptors(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for plugin in self.plugins:
            descriptor = getattr(plugin, "descriptor", None)
            if isinstance(descriptor, PluginInfo):
                result.append(descriptor.as_dict())
            elif isinstance(descriptor, Mapping):
                result.append({str(key): value for key, value in descriptor.items()})
            else:
                result.append(PluginInfo(_plugin_name(plugin)).as_dict())
        return result


def builtin_plugins() -> tuple[ImpactPlugin, ...]:
    """Return fresh built-ins in deterministic dispatch order."""

    return (
        DVCDependencyPlugin(),
        SQLDependencyPlugin(),
        DuckDBDependencyPlugin(),
        NotebookDependencyPlugin(),
        ShellDependencyPlugin(),
        OpenLineageDependencyPlugin(),
        PipelineDependencyPlugin(),
    )


def default_registry(*, discover: bool = False) -> PluginRegistry:
    registry = PluginRegistry(builtin_plugins())
    if discover:
        discover_plugins(registry=registry)
    return registry


def _entry_points(group: str) -> list[Any]:
    try:
        points = importlib_metadata.entry_points()
        if hasattr(points, "select"):
            return list(points.select(group=group))
        return list(points.get(group, ()))
    except (AttributeError, KeyError, TypeError, ValueError):
        return []


def discover_plugins(
    registry: PluginRegistry | None = None,
    *,
    groups: Iterable[str] = ENTRY_POINT_GROUPS,
) -> PluginRegistry:
    """Load opt-in entry-point plugins without making discovery mandatory."""

    registry = registry or default_registry()
    seen: set[str] = set()
    for group in groups:
        for entry in sorted(_entry_points(group), key=lambda item: str(getattr(item, "name", ""))):
            entry_name = str(getattr(entry, "name", ""))
            if entry_name in seen:
                continue
            seen.add(entry_name)
            try:
                loaded = entry.load()
                value = loaded() if isinstance(loaded, type) else loaded
                values = value if isinstance(value, (list, tuple, set)) else (value,)
                for plugin in values:
                    registry.register(plugin, replace=True)
            except (ImportError, AttributeError, TypeError, ValueError, OSError) as exc:
                registry.errors.append({"plugin": entry_name or str(entry), "error": str(exc)})
    return registry


load_plugins = discover_plugins
get_default_registry = default_registry


__all__ = [
    "DVCDependencyPlugin",
    "DVCPlugin",
    "DuckDBDependencyPlugin",
    "DuckDBPlugin",
    "ENTRY_POINT_GROUPS",
    "ImpactPlugin",
    "NotebookDependencyPlugin",
    "NotebookPlugin",
    "OpenLineageDependencyPlugin",
    "OpenLineagePlugin",
    "PLUGIN_PROTOCOL_VERSION",
    "PipelineDependencyPlugin",
    "PipelinePlugin",
    "PluginInfo",
    "PluginRegistry",
    "SQLDependencyPlugin",
    "SQLPlugin",
    "ShellDependencyPlugin",
    "ShellPlugin",
    "builtin_plugins",
    "default_registry",
    "discover_plugins",
    "get_default_registry",
    "load_plugins",
]
