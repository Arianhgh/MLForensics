"""Runtime, dependency, and host metadata capture."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from collections.abc import Iterable, Mapping
from typing import Any

DEFAULT_ENV_ALLOWLIST = {
    "PATH",
    "PYTHONPATH",
    "VIRTUAL_ENV",
    "CONDA_DEFAULT_ENV",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "CUDNN_VERSION",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
}


def capture_environment(
    *,
    allowlist: Iterable[str] | None = None,
    include_env: bool = True,
    extra: dict[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    names = set(DEFAULT_ENV_ALLOWLIST if allowlist is None else allowlist)
    source = os.environ if environ is None else environ
    env = {name: source[name] for name in sorted(names) if name in source} if include_env else {}
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return {
        "python": sys.version,
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "environment": env,
        "variables": env,
        "allowlist": sorted(names),
    }


def capture_dependencies(
    *, packages: Iterable[str] | None = None, max_packages: int = 2_000
) -> dict[str, Any]:
    requested = set(packages or ())
    records: dict[str, str] = {}
    try:
        distributions = importlib.metadata.distributions()
        for index, distribution in enumerate(distributions):
            if index >= max_packages:
                break
            name = distribution.metadata.get("Name") or distribution.name
            if (
                not requested
                or name in requested
                or name.lower() in {item.lower() for item in requested}
            ):
                records[name] = distribution.version
    except Exception as exc:  # pragma: no cover - platform-specific metadata failures
        return {"available": False, "error": str(exc), "packages": records}
    return {
        "available": True,
        "packages": dict(sorted(records.items(), key=lambda item: item[0].lower())),
    }


def capture_hardware() -> dict[str, Any]:
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore

        result["memory_bytes"] = psutil.virtual_memory().total
    except Exception:
        pass
    try:
        import torch  # type: ignore

        result["torch"] = {
            "version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if torch.cuda.is_available():
            result["gpus"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": torch.cuda.get_device_capability(index),
                }
                for index in range(torch.cuda.device_count())
            ]
    except Exception:
        pass
    return result


def capture_python_environment() -> dict[str, Any]:
    """Return interpreter identity separately from the host environment."""

    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "version_info": list(sys.version_info[:5]),
        "executable": os.path.abspath(sys.executable),
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "cache_tag": getattr(sys.implementation, "cache_tag", None),
        "argv": list(sys.argv),
    }


def capture_environment_variables(
    allowlist: Iterable[str] | None = None, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Return only explicitly allowlisted environment variables."""

    source = os.environ if environ is None else environ
    names = DEFAULT_ENV_ALLOWLIST if allowlist is None else allowlist
    return {name: source[name] for name in sorted(set(names)) if name in source}


capture_runtime = capture_python_environment
capture_env = capture_environment
capture_python_info = capture_python_environment
