"""Lightweight installed dependency inventory."""

from __future__ import annotations

import importlib.metadata as metadata
import sys
from typing import Any


def capture_dependencies() -> dict[str, Any]:
    """Return sorted installed distributions using the Python standard library."""

    packages: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        distributions = metadata.distributions()
    except Exception as exc:  # pragma: no cover - unusual interpreter-specific failure
        distributions = ()
        errors.append(str(exc))

    for distribution in distributions:
        try:
            name = distribution.metadata.get("Name") or distribution.name
            version = distribution.version
            item: dict[str, Any] = {"name": name, "version": version}
            direct_url = distribution.read_text("direct_url.json")
            if direct_url:
                item["direct_url"] = direct_url
            packages.append(item)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    packages.sort(
        key=lambda item: (str(item.get("name", "")).lower(), str(item.get("version", "")))
    )
    result: dict[str, Any] = {
        "available": True,
        "python": sys.version.split()[0],
        "packages": packages,
        "count": len(packages),
    }
    if errors:
        result["errors"] = errors
    return result


dependency_inventory = capture_dependencies
capture_dependency_inventory = capture_dependencies
