"""Statistical, behavioral, and performance analysis."""

from .behavior import BehavioralDiff, SliceResult, behavioral_diff, discover_slices
from .compare import compare_capsules, compare_runs, format_comparison, render_comparison
from .performance import (
    PerformanceDiff,
    PerformanceMetric,
    compare_performance,
    performance_delta,
    performance_diff,
)
from .statistics import (
    BootstrapResult,
    bootstrap,
    compare_samples,
    effect_size,
    noninferiority_decision,
    paired_bootstrap,
    paired_differences,
    regression_decision,
    relative_delta,
)

__all__ = [
    "BehavioralDiff",
    "BootstrapResult",
    "PerformanceDiff",
    "PerformanceMetric",
    "SliceResult",
    "behavioral_diff",
    "bootstrap",
    "compare_capsules",
    "compare_performance",
    "compare_runs",
    "compare_samples",
    "discover_slices",
    "effect_size",
    "format_comparison",
    "noninferiority_decision",
    "paired_bootstrap",
    "paired_differences",
    "performance_delta",
    "performance_diff",
    "regression_decision",
    "relative_delta",
    "render_comparison",
]
