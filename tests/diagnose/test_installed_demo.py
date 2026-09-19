from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mlforensics.examples.attention import run_complete_demo, write_fixture_repository


def test_worktree_demo_finds_bug_reduces_input_and_traces(tmp_path: Path):
    pytest.importorskip("torch")
    repo = write_fixture_repository(tmp_path / "fixture")
    result = run_complete_demo(repo["root"], tmp_path / "artifacts")
    assert result["first_bad"] == repo["buggy"]
    assert result["replayed"]
    assert result["child_replayed"]
    assert result["reduced"] is not None and result["original"] is not None
    assert result["reduced"] < result["original"]
    assert result["trace"]


def test_installed_wheel_demo_runs_outside_the_source_tree(tmp_path: Path):
    pytest.importorskip("torch")
    workspace = Path(__file__).resolve().parents[2]
    dist = tmp_path / "dist"
    dist.mkdir()
    built = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist)],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if built.returncode != 0:
        pytest.skip(built.stderr[-1000:] or "python -m build is required")
    wheels = list(dist.glob("mlforensics-*.whl"))
    assert wheels, built.stdout
    site = tmp_path / "site"
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", str(wheels[0]), "-t", str(site)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert install.returncode == 0, install.stderr
    repo = write_fixture_repository(tmp_path / "fixture")
    artifacts = tmp_path / "artifacts"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(site)
    env["PYTHONNOUSERSITE"] = "1"
    env["MLFORENSICS_EXAMPLE_ROOT"] = repo["root"]
    script = tmp_path / "run_demo.py"
    script.write_text(
        "from pathlib import Path\n"
        "from mlforensics.examples.attention import run_complete_demo\n"
        f"result = run_complete_demo({repo['root']!r}, {str(artifacts)!r})\n"
        "assert result['first_bad']\n"
        "assert result['replayed'] and result['child_replayed']\n"
        "assert result['reduced'] < result['original']\n"
        "assert result['trace']\n"
        "Path('demo.json').write_text(repr(result))\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert (tmp_path / "demo.json").exists()
