"""Failure-signature helpers for integrations and diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import FailureSignature, normalize_failure_message


def normalize_exception_message(message: str, *, max_length: int = 500) -> str:
    return normalize_failure_message(str(message))[:max_length]


def exception_signature(
    exc: BaseException, *, top_frame: str | None = None, phase: str | None = None
) -> FailureSignature:
    """Build a portable signature; legacy context arguments are accepted."""
    return FailureSignature.from_exception(exc, top_frame=top_frame, phase=phase)


def signature_from_record(
    record: FailureSignature | Mapping[str, Any] | BaseException,
) -> FailureSignature:
    if isinstance(record, FailureSignature):
        return record
    if isinstance(record, BaseException):
        return exception_signature(record)
    return FailureSignature.from_dict(record)


__all__ = ["exception_signature", "normalize_exception_message", "signature_from_record"]
