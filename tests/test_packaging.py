from __future__ import annotations

import re
import subprocess
import sys
from importlib.resources import files

import mlforensics


def test_public_exports_are_resolvable_and_unique() -> None:
    assert len(mlforensics.__all__) == len(set(mlforensics.__all__))
    assert all(hasattr(mlforensics, name) for name in mlforensics.__all__)


def test_version_is_a_release_version() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[abrc]\d+)?", mlforensics.__version__)


def test_typing_marker_is_packaged() -> None:
    marker = files("mlforensics").joinpath("py.typed")
    assert marker.is_file()


def test_module_entry_point_is_available() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mlforensics", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Forensic reliability tools" in result.stdout
