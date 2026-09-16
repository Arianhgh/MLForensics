"""Compatibility import surface for parity comparison.

The implementation is kept in :mod:`mlforensics.parity.compare`; this module
remains available for callers of the initial parity API.
"""

from .compare import (
    ComparisonResult,
    OutputComparison,
    ParityCaseResult,
    ParityChecker,
    ParityComparator,
    ParityDivergence,
    ParityReport,
    ParityResult,
    Tolerance,
    TolerancePolicy,
    Tolerances,
    compare_batch,
    compare_models,
    compare_outputs,
    parity,
)

__all__ = [
    "ComparisonResult",
    "OutputComparison",
    "ParityCaseResult",
    "ParityChecker",
    "ParityComparator",
    "ParityDivergence",
    "ParityReport",
    "ParityResult",
    "Tolerance",
    "TolerancePolicy",
    "Tolerances",
    "compare_batch",
    "compare_models",
    "compare_outputs",
    "parity",
]
