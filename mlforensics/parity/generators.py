"""Reusable configuration for representative and edge input generation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .inputs import InputCase, generate_input_cases, generate_representative_inputs


def generate_edge_inputs(
    shape: Sequence[int],
    *,
    count: int = 32,
    seed: int = 0,
    low: float = -1.0,
    high: float = 1.0,
) -> list[Any]:
    """Backward-compatible generator returning representative shaped values."""

    return generate_representative_inputs(
        shape=shape,
        count=count,
        seed=seed,
        low=low,
        high=high,
    )


@dataclass(slots=True)
class RepresentativeInputGenerator:
    """Generate deterministic model inputs and labeled parity cases."""

    shape: tuple[int, ...]
    count: int = 32
    seed: int = 0
    low: float = -1.0
    high: float = 1.0
    dtype: Any = float
    include_edge: bool = True
    include_adversarial: bool = True
    include_non_finite: bool = False

    def generate(self) -> list[Any]:
        return generate_representative_inputs(
            self.shape,
            dtype=self.dtype,
            count=self.count,
            seed=self.seed,
            low=self.low,
            high=self.high,
        )

    def generate_cases(self, *, examples: Sequence[Any] | None = None) -> list[InputCase]:
        return generate_input_cases(
            examples=examples,
            shape=self.shape,
            dtype=self.dtype,
            seed=self.seed,
            representative_count=self.count,
            include_edge=self.include_edge,
            include_adversarial=self.include_adversarial,
            include_non_finite=self.include_non_finite,
        )
