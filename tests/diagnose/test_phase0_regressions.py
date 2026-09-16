import json
import random
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from mlforensics import (
    BisectCache,
    CaptureContext,
    Run,
    RunCapsule,
    behavioral_diff,
    bisect_commits,
    ci_gate,
)
from mlforensics.cli.main import main
from mlforensics.diagnose.bisect import SubprocessGit, bytecode_isolation_env
from mlforensics.diagnose.replay import ReplayEngine
from mlforensics.impact import analyze_impact


def test_ci_missing_required_metric_is_not_a_pass():
    gate = ci_gate(
        Run(status="completed"), Run(status="completed"), practical_thresholds={"accuracy": 0.01}
    )
    assert gate.passed is False
    assert any("missing" in check.name or "evidence" in check.name for check in gate.checks)


def test_bisect_pairs_shared_seeds_and_does_not_reuse_stale_runners():
    class Git:
        def __init__(self):
            self.current = None

        def commits(self, *args):
            return ["good", "bad"]

        def checkout(self, revision):
            self.current = revision

    git = Git()
    failed = {"error": "failed"}
    # Only seed 29 produces a usable result on both sides, so the single shared
    # pair must stay inconclusive rather than being classified from thin evidence.
    values = {
        "good": {11: 0, 29: 100, 37: failed, 53: failed, 71: failed},
        "bad": {11: failed, 29: 100, 37: 200, 53: 200, 71: 200},
    }
    report = bisect_commits(
        git,
        lambda seed: values[git.current][seed],
        [11, 29, 37, 53, 71],
        tolerance=1,
        n_resamples=100,
    )
    assert report.first_bad is None
    assert report.inconclusive


def test_bisect_rejects_fewer_seeds_than_the_evidence_requirement():
    class Git:
        def commits(self, *args):
            return ["good", "bad"]

        def checkout(self, revision):
            self.current = revision

    with pytest.raises(ValueError, match="at least 5 seeds"):
        bisect_commits(Git(), lambda seed: 1.0, [11, 29, 37])

    cache = BisectCache()
    git = Git()
    first = bisect_commits(
        git, lambda seed: 0 if git.current == "good" else 10, [11, 29, 37, 53, 71], cache=cache
    )
    second = bisect_commits(git, lambda seed: 0, [11, 29, 37, 53, 71], cache=cache)
    assert first.first_bad == "bad"
    assert second.metadata["executed_runs"] > 0


def test_replay_prefers_rng_snapshot_over_logical_seed():
    random.seed(123)
    random.random()
    probe = random.Random()
    probe.setstate(random.getstate())
    wanted = probe.random()

    def rng_runner(value):
        if random.random() == value:
            raise ValueError("target failure")

    capture = CaptureContext(replay_input=wanted, replay_seed=123)
    with pytest.raises(ValueError, match="target failure"):
        with capture:
            rng_runner(wanted)
    result = ReplayEngine().replay(capture.capsule, rng_runner)
    assert result.reproduced


def test_cli_replay_matches_unqualified_python_exception_names(tmp_path: Path, capsys):
    capture = CaptureContext(replay_input=[1])
    with pytest.raises(ValueError, match="target failure"):
        with capture:
            raise ValueError("target failure")
    path = capture.capsule.save(tmp_path / "failure.mlcap")
    command = (
        shlex.quote(sys.executable) + " -c " + shlex.quote("raise ValueError('target failure')")
    )
    assert main(["replay", str(path), "--command", command, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["reproduced"] is True
    assert result["signature_match"] is True


def test_cli_shrink_requires_original_failure_signature(tmp_path: Path, capsys):
    predicate = tmp_path / "predicate.py"
    predicate.write_text(
        "import json, sys\n"
        "value = json.load(sys.stdin)\n"
        "if len(value) == 3:\n"
        '    raise RuntimeError("original failure")\n'
        'raise KeyError("different failure")\n',
        encoding="utf-8",
    )
    command = shlex.quote(sys.executable) + " " + shlex.quote(str(predicate))
    assert main(["shrink", "[1,2,3]", "--command", command, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["value"] == [1, 2, 3]
    assert result["metadata"]["predicate_kind"] == "failure_signature"


def test_cli_contains_predicate_is_labeled(capsys):
    assert main(["shrink", '["noise", "needle"]', "--contains", "needle", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["value"] == ["needle"]
    assert result["metadata"]["predicate_kind"] == "contains"


def test_relative_imports_are_impact_dependencies(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "features.py").write_text(
        "def feature():\n    return 1\n", encoding="utf-8"
    )
    (tmp_path / "pkg" / "model.py").write_text(
        "from .features import feature\n\ndef predict():\n    return feature()\n",
        encoding="utf-8",
    )
    report = analyze_impact(tmp_path, ["pkg/features.py"])
    assert any("model.py" in node for node in report.affected_nodes)


def test_confidence_only_change_is_reported():
    diff = behavioral_diff([[0.9, 0.1]], [[0.6, 0.4]], labels=[0], n_resamples=100)
    assert diff.confidence_only_changes == 1
    assert diff.class_flips == 0


def test_nan_metric_still_produces_a_capsule(tmp_path: Path):
    import mlforensics

    output = tmp_path / "nan.mlcap"
    try:
        with mlforensics.capture(root=output) as run:
            run.record_metric("loss", float("nan"))
            raise RuntimeError("original training failure")
    except RuntimeError:
        pass
    capsule = RunCapsule.load(output)
    assert capsule.run.failure_signature is not None
    assert any(item.name == "loss" for item in capsule.run.observations)


def test_bytecode_isolation_differs_per_revision(tmp_path):
    """Two revisions must never share a bytecode cache directory.

    CPython treats a cached .pyc as current when the source size and whole-second
    mtime match, which is exactly what happens when a bisect checks out several
    revisions in the same second.
    """
    first = bytecode_isolation_env("a8271ce", tmp_path)
    second = bytecode_isolation_env("8ba3c18", tmp_path)

    assert first["PYTHONPYCACHEPREFIX"] != second["PYTHONPYCACHEPREFIX"]
    assert bytecode_isolation_env("a8271ce", tmp_path) == first
    # Branch names contain path separators; they must not escape the cache root.
    nested = bytecode_isolation_env("feature/new-attention", tmp_path)
    assert Path(nested["PYTHONPYCACHEPREFIX"]).parent == Path(tmp_path)


def test_bisect_checkout_records_the_revision_for_cache_isolation(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    for arguments in (["init", "-q", "-b", "main"], ["commit", "-q", "--allow-empty", "-m", "one"]):
        subprocess.run(["git", *arguments], cwd=repository, check=True)

    git = SubprocessGit(str(repository))
    assert git.revision is None
    git.checkout("main")
    assert git.revision == "main"
