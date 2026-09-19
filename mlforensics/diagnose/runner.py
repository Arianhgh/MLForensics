"""Protocols and deterministic execution helpers."""

from __future__ import annotations

import inspect
import multiprocessing
import random
import signal
import threading
import traceback
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RunResult:
    """Normalised result of one seeded experiment run."""

    seed: int | None = None
    ok: bool = True
    metric: float | None = None
    value: Any = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    status: str | None = None

    def __post_init__(self) -> None:
        status = _status_text(self.status)
        if status is None and isinstance(self.metadata, Mapping):
            status = _status_text(self.metadata.get("status"))
        object.__setattr__(self, "status", status or ("ok" if self.ok else "error"))

    @property
    def success(self) -> bool:
        return self.ok

    @property
    def timed_out(self) -> bool:
        return self.status == "timeout" or bool(self.metadata.get("timed_out", False))

    @property
    def exception(self) -> Mapping[str, Any] | None:
        """Structured exception information, when the run failed."""
        value = self.metadata.get("exception")
        return value if isinstance(value, Mapping) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "ok": self.ok,
            "metric": self.metric,
            "value": self.value,
            "error": self.error,
            "metadata": dict(self.metadata),
            "status": self.status,
        }

    as_dict = to_dict

    @classmethod
    def from_value(cls, value: Any, seed: int | None = None) -> RunResult:
        if isinstance(value, cls):
            return (
                value
                if value.seed is not None or seed is None
                else cls(
                    seed=seed,
                    ok=value.ok,
                    metric=value.metric,
                    value=value.value,
                    error=value.error,
                    metadata=value.metadata,
                    status=value.status,
                )
            )
        if isinstance(value, Mapping):
            status = _status_text(value.get("status"))
            default_ok = not value.get("error")
            if status is not None:
                default_ok = status in _SUCCESS_STATUSES
            ok = value.get("ok", value.get("success", default_ok))
            return cls(
                seed=seed if value.get("seed") is None else value.get("seed"),
                ok=bool(ok),
                metric=_number(value.get("metric")),
                value=value.get("value", value),
                error=_error_text(value.get("error")),
                metadata={
                    **(
                        dict(value.get("metadata", {}))
                        if isinstance(value.get("metadata", {}), Mapping)
                        else {}
                    ),
                    **({"status": status} if status is not None else {}),
                },
                status=status,
            )
        if isinstance(value, bool):
            return cls(seed=seed, ok=value, value=value)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return cls(seed=seed, ok=True, metric=float(value), value=value)
        return cls(seed=seed, ok=True, value=value)


@dataclass(frozen=True)
class SeedRun:
    seed: int
    result: RunResult


@runtime_checkable
class ExperimentRunner(Protocol):
    """A runner can accept a seed as a keyword or positional argument.

    Returning ``RunResult``, a mapping, a boolean, or a numeric metric is
    supported by :func:`run_many`; this keeps adapters small for user code.
    """

    def run(self, seed: int, **kwargs: Any) -> Any: ...


def run_many(
    runner: ExperimentRunner | Callable[..., Any],
    seeds: Iterable[int],
    *,
    kwargs: Mapping[str, Any] | None = None,
    fail_fast: bool = False,
    fresh_process: bool = False,
    timeout: float | None = None,
    torch_metadata: bool = False,
) -> tuple[SeedRun, ...]:
    """Run each seed exactly once, in the supplied order.

    Runs are deterministic with respect to Python's RNG and, when installed,
    NumPy and PyTorch CPU/CUDA RNGs.  By default the callable runs in-process;
    ``fresh_process=True`` gives every seed a separate child process, which
    also makes timeout enforcement safe for arbitrary user code.

    ``torch_metadata`` is deliberately opt-in.  PyTorch is never imported at
    module import time, and its version/device/RNG details are only attached
    to a result when this option is requested.
    """
    _validate_timeout(timeout)
    if not isinstance(fresh_process, bool):
        raise TypeError("fresh_process must be a boolean")
    if not isinstance(torch_metadata, bool):
        raise TypeError("torch_metadata must be a boolean")
    options = dict(kwargs or {})
    output: list[SeedRun] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seeds must contain integers")
        try:
            if fresh_process:
                result = _run_fresh_process(
                    runner, seed, options, timeout=timeout, include_torch_metadata=torch_metadata
                )
            else:
                result = _run_in_process(
                    runner,
                    seed,
                    options,
                    timeout=timeout,
                    include_torch_metadata=torch_metadata,
                    capture_errors=not fail_fast,
                )
        except Exception as exc:  # runners are experiments; capture failures as data
            result = _exception_result(seed, exc)
            if fail_fast:
                raise
        output.append(SeedRun(seed, result))
    return tuple(output)


def evaluate_runner(*args: Any, **kwargs: Any) -> tuple[SeedRun, ...]:
    """Alias kept as the readable name for callers building reports."""
    return run_many(*args, **kwargs)


def _call_seeded(
    runner: ExperimentRunner | Callable[..., Any], seed: int, options: Mapping[str, Any]
) -> Any:
    """Call positional or keyword-only seed runners without catching body errors."""
    target = runner.run if hasattr(runner, "run") else runner
    call_options = dict(options)
    # The seed supplied by run_many is authoritative even if a caller copied
    # one into the generic kwargs mapping.
    call_options.pop("seed", None)
    try:
        parameters = inspect.signature(target).parameters
        seed_parameter = parameters.get("seed")
        if seed_parameter is not None and seed_parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            return target(seed=seed, **call_options)
    except (TypeError, ValueError):
        pass
    return target(seed, **call_options)


_SUCCESS_STATUSES = frozenset({"ok", "success", "passed", "pass"})


def _status_text(value: Any) -> str | None:
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def _validate_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
        or timeout != timeout
        or timeout in (float("inf"), float("-inf"))
    ):
        raise ValueError("timeout must be a positive finite number or None")


class _RunnerTimeout(TimeoutError):
    pass


def _run_in_process(
    runner: ExperimentRunner | Callable[..., Any],
    seed: int,
    options: Mapping[str, Any],
    *,
    timeout: float | None,
    include_torch_metadata: bool,
    capture_errors: bool,
) -> RunResult:
    states = _capture_rng_states()
    try:
        seeded_metadata = _seed_rngs(seed, include_torch_metadata=include_torch_metadata)
        raw = _call_with_timeout(runner, seed, options, timeout)
        result = RunResult.from_value(raw, seed)
        return _merge_metadata(result, seeded_metadata)
    except _RunnerTimeout as exc:
        if not capture_errors:
            raise
        return _timeout_result(seed, timeout, exc)
    except Exception as exc:
        if not capture_errors:
            raise
        return _exception_result(seed, exc)
    finally:
        _restore_rng_states(states)


def _run_fresh_process(
    runner: ExperimentRunner | Callable[..., Any],
    seed: int,
    options: Mapping[str, Any],
    *,
    timeout: float | None,
    include_torch_metadata: bool,
) -> RunResult:
    # ``fork`` preserves compatibility with closures and callable instances,
    # while still giving each run a genuinely fresh interpreter state.  Spawn
    # is the portable fallback for platforms without fork support.
    methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in methods else methods[0])
    queue = context.Queue(maxsize=1)
    process = context.Process(
        target=_fresh_process_entry,
        args=(queue, runner, seed, dict(options), include_torch_metadata),
    )
    try:
        process.start()
    except BaseException as exc:
        queue.close()
        return _exception_result(seed, exc, status="error", phase="process_start")
    try:
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(1.0)
            return _timeout_result(seed, timeout, _RunnerTimeout("runner timed out"))
        try:
            result = queue.get(timeout=0.2)
        except Exception:
            return _exception_result(
                seed,
                RuntimeError(
                    f"runner process exited without a result (exit code {process.exitcode})"
                ),
                status="error",
                phase="process_result",
            )
        if not isinstance(result, RunResult):
            return _exception_result(
                seed,
                TypeError("runner process returned an invalid result"),
                status="error",
                phase="process_result",
            )
        return result
    finally:
        queue.close()
        queue.join_thread()


def _fresh_process_entry(
    queue: Any,
    runner: ExperimentRunner | Callable[..., Any],
    seed: int,
    options: Mapping[str, Any],
    include_torch_metadata: bool,
) -> None:
    try:
        seeded_metadata = _seed_rngs(seed, include_torch_metadata=include_torch_metadata)
        result = RunResult.from_value(_call_seeded(runner, seed, options), seed)
        queue.put(_merge_metadata(result, seeded_metadata))
    except BaseException as exc:
        queue.put(_exception_result(seed, exc))


def _call_with_timeout(
    runner: ExperimentRunner | Callable[..., Any],
    seed: int,
    options: Mapping[str, Any],
    timeout: float | None,
) -> Any:
    if timeout is None or threading.current_thread() is not threading.main_thread():
        return _call_seeded(runner, seed, options)
    previous_handler = signal.getsignal(signal.SIGALRM)

    def alarm(_signum: int, _frame: Any) -> None:
        raise _RunnerTimeout(f"runner timed out after {timeout} seconds")

    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, float(timeout))
    try:
        return _call_seeded(runner, seed, options)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def _capture_rng_states() -> dict[str, Any]:
    states: dict[str, Any] = {"python": random.getstate()}
    try:
        import numpy as np  # type: ignore

        states["numpy"] = np.random.get_state()
    except Exception:
        pass
    try:
        import torch  # type: ignore

        states["torch_cpu"] = torch.get_rng_state().clone()
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            states["torch_cuda"] = [item.clone() for item in cuda.get_rng_state_all()]
    except Exception:
        pass
    return states


def _restore_rng_states(states: Mapping[str, Any]) -> None:
    random.setstate(states["python"])
    if "numpy" in states:
        try:
            import numpy as np  # type: ignore

            np.random.set_state(states["numpy"])
        except Exception:
            pass
    if "torch_cpu" in states:
        try:
            import torch  # type: ignore

            torch.set_rng_state(states["torch_cpu"])
            if "torch_cuda" in states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(states["torch_cuda"])
        except Exception:
            pass


def _seed_rngs(seed: int, *, include_torch_metadata: bool) -> dict[str, Any]:
    random.seed(seed)
    metadata: dict[str, Any] = {
        "rng": {"python": True, "numpy": False, "torch_cpu": False, "torch_cuda": False}
    }
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
        metadata["rng"]["numpy"] = True
    except Exception:
        pass
    torch = None
    try:
        import torch as torch_module  # type: ignore

        torch = torch_module
        torch.manual_seed(seed)
        metadata["rng"]["torch_cpu"] = True
        cuda = getattr(torch, "cuda", None)
        cuda_available = bool(cuda is not None and cuda.is_available())
        if cuda_available and hasattr(cuda, "manual_seed_all"):
            cuda.manual_seed_all(seed)
            metadata["rng"]["torch_cuda"] = True
    except Exception:
        pass
    if include_torch_metadata:
        metadata["torch"] = _torch_metadata(torch)
    return metadata


def _torch_metadata(torch: Any) -> dict[str, Any]:
    if torch is None:
        return {"available": False}
    cuda = getattr(torch, "cuda", None)
    cuda_available = bool(cuda is not None and cuda.is_available())
    result: dict[str, Any] = {
        "available": True,
        "version": str(getattr(torch, "__version__", "unknown")),
        "cuda_available": cuda_available,
    }
    if cuda_available:
        try:
            result["cuda_device_count"] = int(cuda.device_count())
        except Exception:
            result["cuda_device_count"] = None
    try:
        result["cpu_rng_state"] = _tensor_to_list(torch.get_rng_state())
        if cuda_available:
            result["cuda_rng_state"] = [_tensor_to_list(item) for item in cuda.get_rng_state_all()]
    except Exception:
        result["rng_state_available"] = False
    return result


def _tensor_to_list(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _merge_metadata(result: RunResult, metadata: Mapping[str, Any]) -> RunResult:
    merged = dict(result.metadata)
    merged.update(metadata)
    merged.setdefault("status", result.status)
    return replace(result, metadata=merged)


def _exception_result(
    seed: int,
    exc: BaseException,
    *,
    status: str = "error",
    phase: str = "runner",
) -> RunResult:
    exception_type = "TimeoutError" if isinstance(exc, _RunnerTimeout) else type(exc).__name__
    qualified_type = (
        "builtins.TimeoutError"
        if isinstance(exc, _RunnerTimeout)
        else f"{type(exc).__module__}.{type(exc).__qualname__}"
    )
    exception = {
        "type": exception_type,
        "qualified_type": qualified_type,
        "message": str(exc),
        "phase": phase,
        "traceback": traceback.format_exc(),
    }
    return RunResult(
        seed=seed,
        ok=False,
        error=f"{type(exc).__name__}: {exc}",
        metadata={"status": status, "exception": exception},
    )


def _timeout_result(seed: int, timeout: float | None, exc: BaseException) -> RunResult:
    result = _exception_result(seed, exc, status="timeout", phase="timeout")
    metadata = dict(result.metadata)
    metadata["timed_out"] = True
    metadata["timeout"] = timeout
    return replace(
        result,
        error=f"TimeoutError: runner timed out after {timeout} seconds",
        metadata=metadata,
    )


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _error_text(value: Any) -> str | None:
    return None if value is None else str(value)
