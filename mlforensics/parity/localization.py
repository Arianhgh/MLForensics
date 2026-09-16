"""Optional hooks for locating where backend divergence begins."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DivergenceLocation:
    """A normalized, report-friendly localization result."""

    path: str | None = None
    component: str | None = None
    message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "component": self.component,
            "message": self.message,
            "details": self.details,
        }


LocalizationHook = Callable[..., Any]


class DivergenceLocalizer:
    """Compose one or more localization hooks.

    Hooks may return any JSON-friendly value, a :class:`DivergenceLocation`,
    or ``None``.  Returning a list is supported for layer-by-layer probes.
    """

    def __init__(self, hooks: Iterable[LocalizationHook] = ()) -> None:
        self.hooks = tuple(hooks)

    def __call__(self, **context: Any) -> list[Any]:
        results: list[Any] = []
        for hook in self.hooks:
            results.append(hook(**context))
        return results


def first_divergent_path(localization: Any) -> str | None:
    """Extract a useful path from common hook return values."""

    if isinstance(localization, DivergenceLocation):
        return localization.path
    if isinstance(localization, Mapping):
        value = localization.get("path")
        return str(value) if value is not None else None
    if isinstance(localization, (list, tuple)):
        for item in localization:
            path = first_divergent_path(item)
            if path:
                return path
    return None
