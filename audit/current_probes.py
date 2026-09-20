"""Reproduce audit findings using temporary files and an in-memory transport.

Run from the checkout with: python audit/current_probes.py
This script does not modify library code or contact remote services.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlforensics import MetricSeries, Run, RunCapsule, ci_gate, compare_runs
from mlforensics.core.contracts import ExecutionSpec
from mlforensics.core.execution import ExecutionService
from mlforensics.ops import OfflineQueue, RetentionPolicy, garbage_collect


def metric_run(run_id, values, identities=("a", "b", "c")):
    return Run(
        run_id=run_id,
        status="completed",
        metrics=(MetricSeries("accuracy", values, identities=identities),),
    )


def main():
    results = {}
    baseline = metric_run("baseline", (0.90, 0.90, 0.90))
    candidate = metric_run("candidate", (0.88, 0.88, 0.88))
    gate = ci_gate(
        baseline,
        candidate,
        practical_thresholds={"accuracy": 0.1},
        noninferiority_margins={"accuracy": 0.01},
        n_resamples=100,
    )
    results["noninferiority_gate"] = {
        "comparison_status": gate.comparison.status,
        "regressions": gate.comparison.regressions,
        "ci_passed": gate.passed,
        "checks": {check.name: check.status for check in gate.checks},
    }

    def seeded(side, seed, value):
        return Run(
            run_id=f"{side}-{seed}",
            status="completed",
            metadata={"seed": seed},
            metrics=(MetricSeries("accuracy", (value, value), identities=("a", "b")),),
        )

    grouped_baseline = [seeded("baseline", 11, 0.9), seeded("baseline", 29, 0.1)]
    grouped_candidate = [seeded("candidate", 11, 0.85), seeded("candidate", 29, 0.05)]
    permutation_results = []
    for group in (grouped_candidate, grouped_candidate[::-1]):
        gate = ci_gate(
            grouped_baseline,
            group,
            practical_thresholds={"accuracy": 0.01},
            n_resamples=1000,
        )
        permutation_results.append(
            {
                "candidate_seed_order": [run.metadata["seed"] for run in group],
                "ci_passed": gate.passed,
                "interval": gate.comparison.evidence["accuracy"]["confidence_interval"],
            }
        )
    results["group_pairing_changes_with_order"] = permutation_results

    numeric = compare_runs(
        metric_run("baseline", (0.9, 0.9, 0.9), (0, 1, 2)),
        metric_run("candidate", (0.1, 0.1, 0.1), (0, 1, 2)),
        n_resamples=100,
    )
    strings = compare_runs(
        metric_run("baseline", (0.9, 0.9, 0.9), ("0", "1", "2")),
        metric_run("candidate", (0.1, 0.1, 0.1), ("0", "1", "2")),
        n_resamples=100,
    )
    results["explicit_integer_identities"] = {
        "integer_status": numeric.status,
        "string_status": strings.status,
        "integer_quality": numeric.evidence["_data_quality"]["metrics"]["accuracy"],
    }

    with tempfile.TemporaryDirectory(prefix="mlforensics-audit-") as directory:
        root = Path(directory)
        capsule = RunCapsule(baseline)
        try:
            capsule.save(root / "run.mlcap")
            directory_result = "saved"
        except Exception as exc:
            directory_result = f"{type(exc).__name__}: {exc}"
        capsule.save(root / "run.zip")
        results["directory_vs_zip"] = {
            "directory": directory_result,
            "zip_round_trip": RunCapsule.load(root / "run.zip").run_id == baseline.run_id,
        }

        retention_root = root / "retention"
        retention_root.mkdir()
        for name, modified in (("old.mlcap.zip", 100), ("new.mlcap.zip", 995)):
            target = retention_root / name
            target.write_bytes(b"x" * 100)
            os.utime(target, (modified, modified))
        plan = garbage_collect(
            retention_root,
            policy=RetentionPolicy(max_age=500, max_bytes=100),
            now=1000,
        )
        results["retention_age_and_budget"] = {
            "selected": [item.path.name for item in plan.candidates],
            "expected_selected": ["old.mlcap.zip"],
            "dry_run": plan.dry_run,
        }

    class Transport:
        def __init__(self):
            self.data = {}
            self.fail_once = True

        def put_bytes(self, key, payload):
            if self.fail_once:
                self.fail_once = False
                raise OSError("temporary outage")
            self.data[key] = payload

        def delete(self, key):
            self.data.pop(key, None)

    queue = OfflineQueue()
    queue.enqueue("put", "model", b"old")
    queue.enqueue("put", "model", b"new")
    transport = Transport()
    queue.flush(transport)
    after_first = transport.data.get("model")
    queue.flush(transport)
    results["offline_retry_order"] = {
        "after_first_flush": after_first.decode() if after_first is not None else None,
        "after_retry": transport.data["model"].decode(),
        "expected_after_retry": "new",
    }
    child_results = {}
    for size in (10, 200000):
        command = [
            sys.executable,
            "-c",
            f'import sys; sys.stdout.write("x" * {size}); sys.stdout.flush()',
        ]
        result = ExecutionService(log_limit=100).run(ExecutionSpec(command=command, timeout=2))
        child_results[str(size)] = {
            "status": result.status,
            "returncode": result.returncode,
        }
    results["verbose_child_timeout"] = child_results
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
