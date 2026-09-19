"""Adversarial regression probes for the local development checkout."""

# The script intentionally changes into the checkout before importing the
# package so it can probe an uninstalled source tree.
# ruff: noqa: E402

import json
import os
import random
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.chdir(ROOT)
from mlforensics import BisectCache, CaptureContext, Run, behavioral_diff, bisect_commits, ci_gate
from mlforensics.diagnose.replay import ReplayEngine
from mlforensics.impact import analyze_impact

out = {}

# CI silently ignores a configured metric absent from both capsules.
gate = ci_gate(
    Run(status="completed"), Run(status="completed"), practical_thresholds={"accuracy": 0.01}
)
out["missing_required_metric"] = {"passed": gate.passed, "checks": [c.name for c in gate.checks]}


class Git:
    def __init__(self):
        self.current = None

    def commits(self, *args):
        return ["good", "bad"]

    def checkout(self, revision):
        self.current = revision


git = Git()
values = {
    "good": {11: 0, 29: 100, 37: {"error": "failed"}},
    "bad": {11: {"error": "failed"}, 29: 100, 37: 200},
}
r = bisect_commits(
    git,
    lambda seed: values[git.current][seed],
    [11, 29, 37],
    tolerance=1,
    n_resamples=100,
    min_observations=3,
)
out["bisect_seed_pairing"] = {
    "first_bad": r.first_bad,
    "decision": r.decisions.get("bad").to_dict() if "bad" in r.decisions else None,
    "inconclusive": r.inconclusive,
    "true_shared_seed_delta": 0,
}

cache = BisectCache()
git = Git()
a = bisect_commits(
    git,
    lambda seed: 0 if git.current == "good" else 10,
    [11, 29],
    cache=cache,
    min_observations=2,
)
b = bisect_commits(git, lambda seed: 0, [11, 29], cache=cache, min_observations=2)
out["bisect_stale_cache"] = {
    "second_first_bad": b.first_bad,
    "second_executed_runs": b.metadata["executed_runs"],
}

random.seed(123)
random.random()
probe = random.Random()
probe.setstate(random.getstate())
wanted = probe.random()


def rng_runner(input):
    if random.random() == input:
        raise ValueError("target failure")


cap = CaptureContext(replay_input=wanted, replay_seed=123)
try:
    with cap:
        rng_runner(wanted)
except ValueError:
    pass
r = ReplayEngine().replay(cap.capsule, rng_runner)
out["replay_seed_overwrites_snapshot"] = {
    "reproduced": r.reproduced,
    "error": r.error,
    "restored": r.restored,
}

state = {"step": 0}


def step(input):
    state["step"] += 1
    if state["step"] == 1:
        raise ValueError("target failure")


cap = CaptureContext(replay_input=1, state_providers={"model": lambda: state})
try:
    with cap:
        step(1)
except ValueError:
    pass
r = ReplayEngine(state_restorers={"model": lambda s: state.update(s)}).replay(cap.capsule, step)
out["replay_post_failure_state"] = {
    "stored": cap.capsule.evidence["replay"]["state"],
    "reproduced": r.reproduced,
    "error": r.error,
}

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    capsule = root / "failure.mlcap"
    cap = CaptureContext(replay_input=[1, 2, 3])
    try:
        with cap:
            raise ValueError("original failure")
    except ValueError:
        pass
    cap.capsule.save(capsule)
    r = subprocess.run(
        [sys.executable, "-m", "mlforensics", "shrink", str(capsule), "--contains", "2"],
        capture_output=True,
        text=True,
    )
    out["shrink_cannot_read_replay_input"] = {
        "returncode": r.returncode,
        "stderr": r.stderr.strip(),
    }
    predicate = root / "predicate.py"
    predicate.write_text(
        "import json,sys\n"
        "x=json.load(sys.stdin)\n"
        'if len(x)==3: raise RuntimeError("original failure")\n'
        'raise KeyError("different failure")\n'
    )
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlforensics",
            "shrink",
            "[1,2,3]",
            "--command",
            shlex.quote(sys.executable) + " " + shlex.quote(str(predicate)),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    out["shrink_changes_failure"] = {"returncode": r.returncode, "result": json.loads(r.stdout)}
    (root / "pkg").mkdir()
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "features.py").write_text("def feature():\n return 1\n")
    (root / "pkg" / "model.py").write_text(
        "from .features import feature\ndef predict():\n return feature()\n"
    )
    r = analyze_impact(root, ["pkg/features.py"])
    out["impact_relative_import"] = {
        "affected_nodes": r.affected_nodes,
        "missed_model": not any("model.py" in x for x in r.affected_nodes),
    }

diff = behavioral_diff([[0.9, 0.1]], [[0.6, 0.4]], labels=[0], n_resamples=100)
out["confidence_only_change"] = {
    "changed_predictions": diff.changed_predictions,
    "change_rate": diff.change_rate,
    "confidence_only_changes": diff.confidence_only_changes,
}
# The append-style capture must preserve failures even when the metric is NaN.
import mlforensics

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "nan.mlcap"
    try:
        with mlforensics.capture(root=p) as run:
            run.record_metric("loss", float("nan"))
            raise RuntimeError("original training failure")
    except Exception as exc:
        out["nan_destroys_capsule"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "capsule_exists": p.exists(),
        }

# Ordinary Python tracebacks do not contain the stored fully qualified type name.
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "failure.mlcap"
    c = CaptureContext(replay_input=[1])
    try:
        with c:
            raise ValueError("target failure")
    except ValueError:
        pass
    c.capsule.save(p)
    command = (
        shlex.quote(sys.executable) + " -c " + shlex.quote("raise ValueError('target failure')")
    )
    r = subprocess.run(
        [sys.executable, "-m", "mlforensics", "replay", str(p), "--command", command, "--json"],
        capture_output=True,
        text=True,
    )
    d = json.loads(r.stdout)
    out["cli_replay_exception_name"] = {
        "returncode": r.returncode,
        "reproduced": d["reproduced"],
        "signature_match": d["signature_match"],
        "expected_type": d["expected_failure"]["error_type"],
        "stderr_final_line": d["stderr"].strip().splitlines()[-1],
    }

# Use the real optional framework, not a duck-typed test double.
try:
    import torch
except ImportError:
    out["torch_checks"] = {"skipped": "PyTorch not installed"}
else:
    model = torch.nn.Linear(2, 1)
    cap = CaptureContext(replay_input=[1.0, 2.0], state_providers={"model": model})
    try:
        with cap:
            raise RuntimeError("original failure")
    except RuntimeError:
        pass
    result = ReplayEngine(state_restorers={"model": model.load_state_dict}).replay(
        cap.capsule, lambda x: (_ for _ in ()).throw(RuntimeError("original failure"))
    )
    out["torch_state_codec"] = {
        "torch_version": torch.__version__,
        "reproduced": result.reproduced,
        "error": result.error,
    }
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "linear.pt"
        torch.jit.trace(model, torch.ones(1, 2)).save(str(p))
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "mlforensics",
                "parity",
                f"pytorch:{p}",
                f"pytorch:{p}",
                "--shape",
                "1,2",
                "--count",
                "1",
                "--json",
            ],
            text=True,
            capture_output=True,
        )
        d = json.loads(r.stdout)
        case_error = d["cases"][0].get("error")
        out["parity_generated_dtype"] = {
            "returncode": r.returncode,
            "passed": d["passed"],
            "error": case_error.strip().splitlines()[-1] if case_error else None,
        }
print(json.dumps(out, indent=2))
