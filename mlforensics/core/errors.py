"""Errors raised by the portable mlforensics contracts."""


class ValidationError(ValueError):
    """Raised when a record is malformed or has an unsupported schema."""


class CheckpointUnavailable(LookupError):
    """Raised when replay needs a checkpoint at or before a step and none exists."""

    def __init__(self, step: int | float | None = None, message: str | None = None) -> None:
        self.step = step
        detail = message or (
            f"no replay checkpoint is available at or before step {step}"
            if step is not None
            else "no replay checkpoint is available for the requested step"
        )
        super().__init__(detail)


class IncompleteReplay(LookupError):
    """Raised when replay cannot bind an input to the requested step."""

    def __init__(
        self,
        step: int | float | None = None,
        restored_step: int | float | None = None,
        message: str | None = None,
    ) -> None:
        self.step = step
        self.restored_step = restored_step
        detail = message or (
            f"replay input for step {step} is missing"
            if step is not None
            else "replay input is missing for the requested step"
        )
        super().__init__(detail)


class UnresolvedEvaluation(ValueError):
    """Raised when a child result envelope is malformed or infrastructure failed."""
