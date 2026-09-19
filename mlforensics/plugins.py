"""Optional plugin discovery and compatibility negotiation.

The plugin layer deliberately depends only on the Python standard library.
Providers advertise themselves with the ``mlforensics.plugins`` entry-point
group and are imported only when :meth:`PluginHandle.load` (or
:func:`load_plugin`) is called.  This keeps the base installation usable in a
minimal, offline environment and makes optional-dependency failures explicit.
"""

from __future__ import annotations

import importlib
import importlib.metadata as metadata_api
import importlib.util
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import total_ordering
from typing import Any, Protocol, runtime_checkable

ENTRY_POINT_GROUP = "mlforensics.plugins"
PLUGIN_PROTOCOL_VERSION = "1.0"


class PluginError(RuntimeError):
    """Base class for discovery, loading, and negotiation errors."""


class PluginCompatibilityError(PluginError):
    """Raised when a plugin does not satisfy the requested protocol/capability."""


class PluginDependencyError(PluginError, ImportError):
    """Raised when a selected plugin cannot import an optional dependency."""


class PluginLoadError(PluginError):
    """Raised when an entry point or plugin descriptor cannot be loaded."""


@total_ordering
class _Version:
    """Small dependency-free version value for protocol negotiation."""

    def __init__(self, value: str | int | float) -> None:
        text = str(value).strip()
        match = re.fullmatch(r"v?(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[-+].*)?", text)
        if match is None:
            raise ValueError(f"invalid version {value!r}")
        self.text = text
        self.parts = tuple(int(item or 0) for item in match.groups())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Version) and self.parts == other.parts

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _Version):
            return NotImplemented
        return self.parts < other.parts


def _version(value: str | int | float | None, *, default: str = "0.0") -> str:
    text = default if value is None else str(value).strip()
    _Version(text)
    return text


def _split_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in re.split(r"[,\s]+", value) if item.strip())
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)


def _metadata_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _descriptor_from_object(value: Any, *, entry_point: Any | None = None) -> Mapping[str, Any]:
    """Extract a descriptor from common plugin conventions without importing modules."""
    if isinstance(value, Mapping):
        return value
    for attribute in ("plugin_metadata", "metadata", "descriptor", "PLUGIN_METADATA"):
        candidate = getattr(value, attribute, None)
        if candidate is not None:
            if callable(candidate):
                candidate = candidate()
            if isinstance(candidate, Mapping):
                return candidate
            return {
                name: getattr(candidate, name)
                for name in (
                    "name",
                    "version",
                    "protocol_version",
                    "capabilities",
                    "optional_dependencies",
                    "requires",
                    "description",
                )
                if hasattr(candidate, name)
            }
    describe = getattr(value, "describe", None)
    if callable(describe):
        described = describe()
        if isinstance(described, Mapping):
            return described
    return {
        "name": _metadata_value(entry_point, "name", ""),
        "version": _metadata_value(_metadata_value(entry_point, "dist"), "version", "0.0"),
    }


@dataclass(frozen=True)
class PluginMetadata:
    """Normalized metadata advertised by an entry-point plugin."""

    name: str
    version: str = "0.0"
    protocol_version: str = PLUGIN_PROTOCOL_VERSION
    capabilities: tuple[str, ...] = field(default_factory=tuple)
    optional_dependencies: tuple[str, ...] = field(default_factory=tuple)
    description: str | None = None
    distribution: str | None = None
    entry_point: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise PluginLoadError("plugin metadata requires a non-empty name")
        protocol = _version(self.protocol_version, default=PLUGIN_PROTOCOL_VERSION)
        version = _version(self.version)
        capabilities = tuple(sorted(set(_split_values(self.capabilities))))
        dependencies = tuple(sorted(set(_split_values(self.optional_dependencies))))
        if self.description is not None and not isinstance(self.description, str):
            raise PluginLoadError("plugin description must be a string or None")
        if not isinstance(self.metadata, Mapping):
            raise PluginLoadError("plugin metadata extension fields must be a mapping")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "protocol_version", protocol)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "optional_dependencies", dependencies)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def api_version(self) -> str:
        """Compatibility alias for ``protocol_version``."""
        return self.protocol_version

    @property
    def requires(self) -> tuple[str, ...]:
        """Compatibility alias for optional dependency names."""
        return self.optional_dependencies

    def supports(self, capabilities: Iterable[str] = ()) -> bool:
        """Return whether every requested capability is advertised."""
        wanted = set(_split_values(capabilities))
        return wanted.issubset(self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "protocol_version": self.protocol_version,
            "capabilities": list(self.capabilities),
            "optional_dependencies": list(self.optional_dependencies),
            "description": self.description,
            "distribution": self.distribution,
            "entry_point": self.entry_point,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_value(cls, value: Any, *, entry_point: Any | None = None) -> PluginMetadata:
        raw = dict(_descriptor_from_object(value, entry_point=entry_point))
        ep_name = _metadata_value(entry_point, "name")
        distribution = _metadata_value(_metadata_value(entry_point, "dist"), "name")
        if distribution is None:
            distribution = _metadata_value(_metadata_value(entry_point, "dist"), "project_name")
        name = raw.pop("name", None) or ep_name
        if not name:
            raise PluginLoadError("plugin entry point has no name")
        protocol = raw.pop("protocol_version", raw.pop("api_version", None))
        capabilities = raw.pop("capabilities", raw.pop("features", ()))
        dependencies = raw.pop(
            "optional_dependencies", raw.pop("requires", raw.pop("dependencies", ()))
        )
        known = {
            "version",
            "description",
            "distribution",
            "entry_point",
            "metadata",
        }
        extensions = raw.pop("metadata", {})
        if not isinstance(extensions, Mapping):
            extensions = {"metadata": extensions}
        extensions = {**raw, **dict(extensions)}
        return cls(
            name=str(name),
            version=_version(raw.pop("version", _metadata_value(entry_point, "version", None))),
            protocol_version=_version(protocol, default=PLUGIN_PROTOCOL_VERSION),
            capabilities=_split_values(capabilities),
            optional_dependencies=_split_values(dependencies),
            description=raw.pop("description", None),
            distribution=str(raw.pop("distribution", distribution)) if distribution else None,
            entry_point=str(raw.pop("entry_point", ep_name)) if ep_name else None,
            metadata={key: value for key, value in extensions.items() if key not in known},
        )


@runtime_checkable
class PluginProtocol(Protocol):
    """Structural protocol implemented by loaded plugins."""

    @property
    def plugin_metadata(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class PluginHandle:
    """Lazy entry-point handle; creating one never imports the provider."""

    metadata: PluginMetadata
    entry_point: Any = field(repr=False, compare=False)

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def capabilities(self) -> tuple[str, ...]:
        return self.metadata.capabilities

    def compatible_with(
        self,
        *,
        protocol_version: str = PLUGIN_PROTOCOL_VERSION,
        capabilities: Iterable[str] = (),
    ) -> bool:
        try:
            _negotiate_protocol(self.metadata.protocol_version, protocol_version)
        except PluginCompatibilityError:
            return False
        return self.metadata.supports(capabilities)

    def load(
        self,
        *,
        protocol_version: str = PLUGIN_PROTOCOL_VERSION,
        capabilities: Iterable[str] = (),
    ) -> Any:
        """Import and validate the selected plugin, translating optional errors."""
        negotiate_plugin(
            self.metadata, protocol_version=protocol_version, capabilities=capabilities
        )
        try:
            loaded = self.entry_point.load()
        except ImportError as exc:
            missing = getattr(exc, "name", None) or str(exc)
            hint = (
                f" Install the provider's optional dependency ({missing!s}) or the plugin extra."
                if missing
                else " Install the plugin's optional dependencies."
            )
            raise PluginDependencyError(
                f"plugin {self.name!r} could not be imported because an optional dependency is "
                f"missing.{hint}"
            ) from exc
        except Exception as exc:
            raise PluginLoadError(f"plugin {self.name!r} failed to load: {exc}") from exc
        if loaded is None:
            raise PluginLoadError(f"plugin {self.name!r} entry point returned None")
        return loaded


def _entry_points(group: str) -> tuple[Any, ...]:
    """Return entry points across Python 3.10--3.13 metadata APIs."""
    try:
        selected = metadata_api.entry_points()
    except Exception as exc:
        raise PluginError(f"could not enumerate plugin entry points: {exc}") from exc
    if hasattr(selected, "select"):
        return tuple(selected.select(group=group))
    if isinstance(selected, Mapping):
        return tuple(selected.get(group, ()))
    return tuple(item for item in selected if _metadata_value(item, "group") == group)


def _metadata_for_entry_point(entry_point: Any) -> PluginMetadata:
    dist = _metadata_value(entry_point, "dist")
    version = _metadata_value(dist, "version", None)
    raw = {
        "name": _metadata_value(entry_point, "name", ""),
        "version": version or _metadata_value(entry_point, "version", "0.0"),
        "protocol_version": _metadata_value(
            entry_point, "protocol_version", PLUGIN_PROTOCOL_VERSION
        ),
        "capabilities": _metadata_value(entry_point, "capabilities", ()),
        "optional_dependencies": _metadata_value(entry_point, "optional_dependencies", ()),
        "description": _metadata_value(entry_point, "description", None),
    }
    return PluginMetadata.from_value(raw, entry_point=entry_point)


def discover_plugins(
    *,
    group: str = ENTRY_POINT_GROUP,
    capabilities: Iterable[str] = (),
    protocol_version: str | None = PLUGIN_PROTOCOL_VERSION,
    entry_points: Sequence[Any] | None = None,
) -> tuple[PluginHandle, ...]:
    """Discover compatible plugins without importing them.

    ``entry_points`` is an injection hook for tests and embedded applications.
    Results are deterministic by normalized plugin name, distribution, and
    entry-point value. Incompatible plugins are omitted; use
    :func:`negotiate_plugin` to obtain a detailed error for a selected plugin.
    """
    handles: list[PluginHandle] = []
    requested = tuple(_split_values(capabilities))
    for item in entry_points if entry_points is not None else _entry_points(group):
        try:
            descriptor = _metadata_for_entry_point(item)
            if protocol_version is not None:
                _negotiate_protocol(descriptor.protocol_version, protocol_version)
            if not descriptor.supports(requested):
                continue
            handles.append(PluginHandle(metadata=descriptor, entry_point=item))
        except (PluginCompatibilityError, PluginLoadError, ValueError):
            continue
    handles.sort(
        key=lambda handle: (
            handle.name.casefold(),
            (handle.metadata.distribution or "").casefold(),
            str(_metadata_value(handle.entry_point, "value", "")),
        )
    )
    return tuple(handles)


def find_plugin(
    name: str,
    *,
    group: str = ENTRY_POINT_GROUP,
    entry_points: Sequence[Any] | None = None,
) -> PluginHandle:
    """Find one plugin by entry-point or advertised name."""
    matches = [
        handle
        for handle in discover_plugins(
            group=group, protocol_version=None, entry_points=entry_points
        )
        if handle.name == name
        or (handle.metadata.entry_point and handle.metadata.entry_point == name)
    ]
    if not matches:
        raise PluginLoadError(f"no plugin named {name!r} was found in entry-point group {group!r}")
    if len(matches) > 1:
        raise PluginLoadError(f"multiple plugins named {name!r} were found")
    return matches[0]


def load_plugin(
    name: str,
    *,
    group: str = ENTRY_POINT_GROUP,
    protocol_version: str = PLUGIN_PROTOCOL_VERSION,
    capabilities: Iterable[str] = (),
    entry_points: Sequence[Any] | None = None,
) -> Any:
    """Find, negotiate, and lazily load a plugin by name."""
    return find_plugin(name, group=group, entry_points=entry_points).load(
        protocol_version=protocol_version, capabilities=capabilities
    )


def _negotiate_protocol(plugin_version: str, requested_version: str) -> None:
    plugin = _Version(plugin_version)
    requested = _Version(requested_version)
    if plugin.parts[0] != requested.parts[0]:
        raise PluginCompatibilityError(
            f"plugin protocol {plugin_version!r} is incompatible with requested "
            f"protocol {requested_version!r}; install a plugin with the same major version"
        )


def negotiate_plugin(
    plugin: PluginMetadata | PluginHandle,
    *,
    protocol_version: str = PLUGIN_PROTOCOL_VERSION,
    capabilities: Iterable[str] = (),
) -> PluginMetadata:
    """Validate protocol and capability requirements and return metadata."""
    descriptor = plugin.metadata if isinstance(plugin, PluginHandle) else plugin
    _negotiate_protocol(descriptor.protocol_version, protocol_version)
    missing = sorted(set(_split_values(capabilities)).difference(descriptor.capabilities))
    if missing:
        raise PluginCompatibilityError(
            f"plugin {descriptor.name!r} lacks required capabilities: {', '.join(missing)}"
        )
    return descriptor


def check_optional_dependencies(modules: Iterable[str]) -> tuple[str, ...]:
    """Return missing import names without importing optional providers."""
    missing: list[str] = []
    for module in _split_values(modules):
        try:
            importlib.util.find_spec(module)
        except (ImportError, ModuleNotFoundError, ValueError):
            missing.append(module)
        else:
            if importlib.util.find_spec(module) is None:
                missing.append(module)
    return tuple(missing)


def require_optional_dependencies(plugin: PluginMetadata) -> None:
    """Raise a clear dependency error for dependencies explicitly advertised by a plugin."""
    missing = check_optional_dependencies(plugin.optional_dependencies)
    if missing:
        raise PluginDependencyError(
            f"plugin {plugin.name!r} requires optional dependencies {', '.join(missing)}; "
            "install the plugin extra before loading it"
        )


__all__ = [
    "ENTRY_POINT_GROUP",
    "PLUGIN_PROTOCOL_VERSION",
    "PluginCompatibilityError",
    "PluginDependencyError",
    "PluginError",
    "PluginHandle",
    "PluginLoadError",
    "PluginMetadata",
    "PluginProtocol",
    "check_optional_dependencies",
    "discover_plugins",
    "find_plugin",
    "load_plugin",
    "negotiate_plugin",
    "require_optional_dependencies",
]
