"""Configuration loading for ``mlforensics.toml``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


_KNOWN_SECTIONS = {
    "storage": {"root", "backend", "uri"},
    "tracking": {"backend", "uri", "experiment", "project"},
    "lineage": {"backend", "uri"},
    "impact": None,
    "ci": {
        "confidence",
        "fail_on_regression",
        "fail_on_failed_run",
        "fail_on_nonfinite",
        "fail_on_missing_evidence",
        "fail_on_behavior_regression",
        "fail_on_parity_failure",
        "thresholds",
        "required_metrics",
        "required_resources",
        "min_sample_count",
        "min_slice_support",
        "behavior_threshold",
        "calibration_threshold",
        "higher_is_better",
        "noninferiority_margins",
        "required_evidence",
    },
}


def _reject_unknown_keys(value: Mapping[str, Any], *, path: Path | None) -> None:
    unknown_sections = [str(key) for key in value if key not in _KNOWN_SECTIONS]
    if unknown_sections:
        location = f" in {path}" if path else ""
        raise ValueError(
            f"unsupported configuration section{location}: {', '.join(sorted(unknown_sections))}"
        )
    for section, allowed in _KNOWN_SECTIONS.items():
        if allowed is None or section not in value:
            continue
        raw = value.get(section) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"[{section}] must be a table")
        unknown = [str(key) for key in raw if key not in allowed]
        if unknown:
            raise ValueError(
                f"unsupported [{section}] key{'s' if len(unknown) > 1 else ''}: "
                + ", ".join(sorted(unknown))
            )


@dataclass(frozen=True)
class Config:
    storage: Mapping[str, Any] = field(default_factory=dict)
    tracking: Mapping[str, Any] = field(default_factory=dict)
    lineage: Mapping[str, Any] = field(default_factory=dict)
    impact: Mapping[str, Any] = field(default_factory=dict)
    ci: Mapping[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @property
    def storage_root(self) -> Path:
        value = self.storage.get("root", ".mlforensics")
        root = Path(str(value))
        if not root.is_absolute() and self.path is not None:
            return self.path.parent / root
        return root

    @property
    def confidence(self) -> float:
        try:
            return float(self.ci.get("confidence", 0.95))
        except (TypeError, ValueError):
            return 0.95

    def get(self, section: str, key: str, default: Any = None) -> Any:
        value = getattr(self, section, {})
        return value.get(key, default) if isinstance(value, Mapping) else default

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, path: Path | None = None) -> Config:
        _reject_unknown_keys(value, path=path)
        return cls(
            storage=dict(value.get("storage", {}) or {}),
            tracking=dict(value.get("tracking", {}) or {}),
            lineage=dict(value.get("lineage", {}) or {}),
            impact=dict(value.get("impact", {}) or {}),
            ci=dict(value.get("ci", {}) or {}),
            path=path,
        )


def load_config(path: str | Path | None = None, *, start: str | Path = ".") -> Config:
    """Load a TOML configuration, returning safe defaults when absent."""
    candidate = Path(path) if path is not None else Path(start) / "mlforensics.toml"
    if not candidate.exists():
        return Config(path=None)
    try:
        data = tomllib.loads(candidate.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid MLForensics configuration: {candidate}") from exc
    return Config.from_dict(data, path=candidate)


read_config = load_config
