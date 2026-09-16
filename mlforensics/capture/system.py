"""System and hardware metadata using standard-library APIs where possible."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any


def _nvidia_metadata() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode:
        return []
    gpus: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == 3:
            gpus.append({"name": values[0], "driver_version": values[1], "memory_mb": values[2]})
    return gpus


def capture_system_metadata() -> dict[str, Any]:
    """Capture portable OS/CPU/memory metadata and optional NVIDIA GPU details."""

    uname = platform.uname()
    result: dict[str, Any] = {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "os": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": platform.processor(),
            "python_platform": platform.platform(),
        },
        "cpu": {
            "logical_count": os.cpu_count(),
            "physical_count": None,
        },
        "memory": {"total_bytes": None},
        "gpu": _nvidia_metadata(),
    }
    # Avoid a psutil dependency, while still reporting common Linux/macOS data.
    try:
        import resource

        result["process"] = {"max_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    except (ImportError, OSError):
        pass
    if shutil.which("sysctl"):
        try:
            physical = subprocess.run(
                ["sysctl", "-n", "hw.physicalcpu"],
                text=True,
                capture_output=True,
                check=False,
            )
            if physical.returncode == 0 and physical.stdout.strip().isdigit():
                result["cpu"]["physical_count"] = int(physical.stdout.strip())
            memory = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                text=True,
                capture_output=True,
                check=False,
            )
            if memory.returncode == 0 and memory.stdout.strip().isdigit():
                result["memory"]["total_bytes"] = int(memory.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    return result


capture_hardware_metadata = capture_system_metadata
capture_hardware = capture_system_metadata
