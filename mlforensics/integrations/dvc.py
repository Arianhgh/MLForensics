"""Minimal DVC dependency import/export helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def read_dvc_dependencies(path: str | Path = "dvc.yaml") -> dict[str, dict[str, list[str]]]:
    source = Path(path)
    if not source.exists():
        return {}
    text = source.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        parsed = yaml.safe_load(text) or {}
        stages = parsed.get("stages", {})
        result = {}
        for name, config in stages.items():
            result[str(name)] = {
                "deps": [str(item) for item in config.get("deps", [])],
                "outs": [str(item) for item in config.get("outs", [])],
            }
        return result
    except ImportError:
        # A conservative fallback is enough for impact analysis in minimal installs.
        result: dict[str, dict[str, list[str]]] = {}
        current = None
        section = None
        for line in text.splitlines():
            stage = re.match(r"^  ([^:#]+):\s*$", line)
            if stage:
                current = stage.group(1).strip()
                result[current] = {"deps": [], "outs": []}
                section = None
                continue
            heading = re.match(r"^    (deps|outs):\s*$", line)
            if heading and current:
                section = heading.group(1)
                continue
            item = re.match(r"^      -\s+(.+?)\s*$", line)
            if item and current and section:
                result[current][section].append(item.group(1).strip("'\""))
        return result


def export_dvc_dependencies(
    mappings: Mapping[str, Any], path: str | Path = "mlforensics-dvc.json"
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(mappings, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
