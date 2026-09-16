import json
import subprocess
import sys
from pathlib import Path

import pytest

from mlforensics.cli.main import main
from mlforensics.core import MetricSeries, Run, RunCapsule, TraceEvent


def test_run_captures_portable_evidence_and_resources(tmp_path, capsys):
    data = tmp_path / "data.json"
    data.write_text('[{"x": 1}]', encoding="utf-8")
    output = tmp_path / "run.mlcap"

    exit_code = main(
        [
            "run",
            "--repo",
            str(tmp_path),
            "--output",
            str(output),
            "--data",
            f"training={data}",
            "--json",
            sys.executable,
            "-c",
            "print('ok')",
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert summary["status"] == "completed"
    capsule = RunCapsule.load(output)
    assert capsule.run.metadata["environment"]["python_version"]
    assert capsule.run.metadata["dependencies"]["packages"]
    assert capsule.run.metadata["data_fingerprints"]["training"]["sha256"]
    assert {item.name for item in capsule.run.resources} >= {"wall_time_s", "cpu_time_s"}


def test_run_merges_evidence_from_an_instrumented_child(tmp_path, capsys):
    output = tmp_path / "instrumented.mlcap"
    repository = Path(__file__).parents[2]
    child_code = (
        "import mlforensics; "
        "session = mlforensics.capture(); "
        "session.__enter__(); "
        "session.record_metric('accuracy', 0.91, step=1); "
        "session.__exit__(None, None, None)"
    )

    exit_code = main(
        [
            "run",
            "--repo",
            str(repository),
            "--output",
            str(output),
            "--no-output",
            "--json",
            "--",
            sys.executable,
            "-c",
            child_code,
        ]
    )

    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["child_instrumented"] is True
    capsule = RunCapsule.load(output)
    assert capsule.run.run_id
    assert [(metric.name, metric.values) for metric in capsule.run.metrics] == [
        ("accuracy", (0.91,))
    ]


def test_run_captures_training_metrics_from_child(tmp_path, capsys):
    output = tmp_path / "training.mlcap"
    repository = Path(__file__).parents[2]
    child_code = (
        "from mlforensics import capture_training\n"
        "class M:\n"
        "    def state_dict(self): return {'w': 1}\n"
        "class O:\n"
        "    def state_dict(self): return {'p': 1}\n"
        "with capture_training(M(), O()) as training:\n"
        "    training.capture.record_metric('loss', 0.25, step=0)\n"
    )
    exit_code = main(
        [
            "run",
            "--repo",
            str(repository),
            "--output",
            str(output),
            "--no-output",
            "--json",
            "--",
            sys.executable,
            "-c",
            child_code,
        ]
    )
    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["child_instrumented"] is True
    capsule = RunCapsule.load(output)
    assert [(metric.name, metric.values) for metric in capsule.run.metrics] == [("loss", (0.25,))]


def test_replay_requires_a_failure_and_never_treats_success_as_reproduction(tmp_path, capsys):
    output = tmp_path / "failed.mlcap"
    command = [sys.executable, "-c", "raise ValueError('boom')"]
    assert main(["run", "--repo", str(tmp_path), "--output", str(output), *command]) == 1
    capsys.readouterr()

    assert main(["replay", str(output), "--command", "true", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["reproduced"] is False

    different_failure = f"{sys.executable} -c \"raise RuntimeError('different')\""
    assert main(["replay", str(output), "--command", different_failure, "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["exit_code_match"] is True
    assert result["process_output_match"] is False
    assert result["reproduced"] is False

    replay_command = f"{sys.executable} -c \"raise ValueError('boom')\""
    assert main(["replay", str(output), "--command", replay_command, "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["reproduced"] is True
    assert result["exit_code_match"] is True
    assert result["process_output_match"] is True


def test_shrink_requires_a_real_predicate_and_parity_can_generate_inputs(capsys):
    with pytest.raises(SystemExit, match="2"):
        main(["shrink", "[1, 2, 3]"])
    capsys.readouterr()

    assert main(["shrink", '["noise", "needle", "more"]', "--contains", "needle", "--json"]) == 0
    shrunk = json.loads(capsys.readouterr().out)
    assert shrunk["value"] == ["needle"]

    assert main(["parity", "operator:neg", "operator:neg", "--count", "5", "--json"]) == 0
    parity = json.loads(capsys.readouterr().out)
    assert parity["passed"] is True
    assert parity["sample_count"] == 5


def test_trace_and_ci_are_structured_workflows(tmp_path, capsys):
    trace_capsule = RunCapsule(
        Run(
            run_id="trace-run",
            status="failed",
            started_at="t",
            events=(
                TraceEvent("linear", data={"tensor_id": "t0", "parents": []}),
                TraceEvent(
                    "softmax",
                    data={
                        "tensor_id": "t1",
                        "parents": ["t0"],
                        "abnormal": True,
                        "finite_fraction": 0.5,
                    },
                ),
            ),
        )
    )
    trace_path = trace_capsule.save(tmp_path / "trace.mlcap")
    assert main(["trace", str(trace_path), "--json"]) == 0
    trace = json.loads(capsys.readouterr().out)
    assert trace["first_abnormal"]["kind"] == "softmax"
    assert trace["ancestry"][0]["kind"] == "linear"

    def _seeded(label, accuracy, seed):
        return str(
            RunCapsule(
                Run(
                    run_id=f"{label}-{seed}",
                    status="completed",
                    started_at="t",
                    metrics=(MetricSeries("accuracy", (accuracy,)),),
                    metadata={"seed": seed},
                )
            ).save(tmp_path / f"{label}-{seed}.mlcap")
        )

    # One capsule per side cannot support a run-level claim, so the gate needs a
    # seeded run per repetition on each side.
    seeds = (11, 29, 37, 53, 71)
    baseline = [_seeded("baseline", 0.90, seed) for seed in seeds]
    candidate = [_seeded("candidate", 0.70, seed) for seed in seeds]
    sides: list[str] = []
    for path in baseline:
        sides.extend(["--baseline", path])
    for path in candidate:
        sides.extend(["--candidate", path])

    assert main(["ci", *sides]) == 1
    assert "FAIL" in capsys.readouterr().out

    config = tmp_path / "mlforensics.toml"
    config.write_text("[ci]\nfail_on_regression = false\n", encoding="utf-8")
    assert main(["--config", str(config), "ci", *sides]) == 0
    assert "PASS" in capsys.readouterr().out


def test_bisect_restores_original_branch(tmp_path, capsys):
    def git(*arguments):
        return subprocess.run(
            ["git", *arguments],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            check=True,
        )

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    branch = git("branch", "--show-current").stdout.strip() or "master"
    revisions = []
    for value in (0, 0, 1):
        (tmp_path / "score.txt").write_text(str(value), encoding="utf-8")
        git("add", "score.txt")
        git("commit", "--allow-empty", "-qm", f"score {value}")
        revisions.append(git("rev-parse", "HEAD").stdout.strip())

    command = f"{sys.executable} -c \"print(open('score.txt').read())\""
    exit_code = main(
        [
            "bisect",
            "--repo",
            str(tmp_path),
            "--good",
            revisions[0],
            "--bad",
            revisions[-1],
            "--command",
            command,
            "--seed",
            "1",
            "--seed",
            "2",
            "--seed",
            "3",
            "--seed",
            "4",
            "--seed",
            "5",
            "--json",
        ]
    )
    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["first_bad"] == revisions[-1]
    assert git("branch", "--show-current").stdout.strip() == branch
