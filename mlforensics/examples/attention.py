"""Installed-package attention debugging fixture and factory."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


def _load_worktree_model() -> Any:
    root = Path(os.environ.get("MLFORENSICS_EXAMPLE_ROOT", os.getcwd()))
    path = root / "model.py"
    if not path.exists():
        raise FileNotFoundError(f"example model.py not found at {path}")
    sys.modules.pop("mlforensics_example_model", None)
    spec = importlib.util.spec_from_file_location("mlforensics_example_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AttentionReplayFixture:
    """Construct, restore, execute, and close a fresh attention model."""

    def __init__(self) -> None:
        self.model: Any = None
        self.optimizer: Any = None
        self._module: Any = None

    def construct(self) -> Any:
        module = _load_worktree_model()
        self._module = module
        torch = __import__("torch")
        torch.manual_seed(0)
        self.model = module.Attention()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        return self.model

    def restore(self, checkpoint: Any) -> None:
        from mlforensics.capture.training import restore_training_state

        if self.model is None:
            self.construct()
        restore_training_state(self.model, self.optimizer, checkpoint or {})

    def restore_component(self, name: str, state: Any) -> None:
        self.restore({name: state})

    def execute(self, value: Any) -> Any:
        if self.model is None:
            self.construct()
        batch = value if isinstance(value, dict) else {"tokens": value[0], "mask": value[1]}
        loss = self.model(batch["tokens"], batch["mask"])
        torch = __import__("torch")
        if not torch.isfinite(loss):
            raise ValueError("non-finite attention loss")
        return loss

    def close(self) -> None:
        self.model = None
        self.optimizer = None
        self._module = None


def failing_batch(torch_module: Any | None = None) -> dict[str, Any]:
    torch = torch_module or __import__("torch")
    torch.manual_seed(0)
    tokens = torch.randn(1, 3, 4)
    return {"tokens": tokens, "mask": torch.zeros(1, 3, dtype=torch.bool)}


HEALTHY_MODEL = """
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(4, 4, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tokens, mask):
        scores = self.query(tokens) @ tokens.transpose(-2, -1)
        scores = scores.masked_fill(~mask.unsqueeze(-2), 0.0)
        return self.softmax(scores).sum()
"""

BUGGY_MODEL = """
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(4, 4, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tokens, mask):
        scores = self.query(tokens) @ tokens.transpose(-2, -1)
        scores = scores.masked_fill(~mask.unsqueeze(-2), float("inf"))
        return self.softmax(scores).sum()
"""


def write_fixture_repository(path: str | Path) -> dict[str, str]:
    """Create a tiny Git history with a healthy revision and a known bug."""
    import subprocess

    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.py").write_text(HEALTHY_MODEL, encoding="utf-8")
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "demo@example.com"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Demo"], cwd=root, check=True)
    subprocess.run(["git", "add", "model.py"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "healthy attention"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    healthy = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    (root / "model.py").write_text(BUGGY_MODEL, encoding="utf-8")
    subprocess.run(["git", "add", "model.py"], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "introduce inf padding mask"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    buggy = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {"healthy": healthy, "buggy": buggy, "root": str(root)}


def _run_attention(batch: dict[str, Any], *, example_root: str) -> Any:
    os.environ["MLFORENSICS_EXAMPLE_ROOT"] = example_root
    fixture = AttentionReplayFixture()
    fixture.construct()
    try:
        return fixture.execute(batch)
    finally:
        fixture.close()


def run_complete_demo(repository: str | Path, artifacts: str | Path) -> dict[str, Any]:
    """Execute capture → compare → bisect → replay → shrink → replay → trace."""
    import subprocess

    from mlforensics.analysis import compare_runs
    from mlforensics.capture import capture_training
    from mlforensics.diagnose.bisect import WorktreeGit, bisect_commits
    from mlforensics.diagnose.factory import replay_in_worker, replay_with_factory
    from mlforensics.diagnose.shrink import shrink_capsule

    torch = __import__("torch")
    repo = Path(repository)
    output = Path(artifacts)
    output.mkdir(parents=True, exist_ok=True)
    git = WorktreeGit(str(repo), root=output / "worktrees")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
        parent = subprocess.run(
            ["git", "rev-parse", "HEAD^"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
        healthy, buggy = parent, head
        batch = failing_batch(torch)

        def capture_revision(revision: str, expect_fail: bool):
            git.checkout(revision)
            os.environ["MLFORENSICS_EXAMPLE_ROOT"] = git.working_directory or str(repo)
            module = _load_worktree_model()
            torch.manual_seed(0)
            model = module.Attention()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            try:
                with capture_training(
                    model,
                    optimizer,
                    replay_factory="mlforensics.examples.attention:AttentionReplayFixture",
                    metadata={"seed": 0, "revision": revision},
                ) as training:
                    with training.step(0, batch=batch, sample_ids=["pad-only"]):
                        loss = model(batch["tokens"], batch["mask"])
                        metric = float(loss.detach()) if torch.isfinite(loss) else float("nan")
                        training.capture.record_metric("loss", metric)
                        if not torch.isfinite(loss):
                            raise ValueError("non-finite attention loss")
            except ValueError:
                if not expect_fail:
                    raise
            return training.capsule

        healthy_capsule = capture_revision(healthy, expect_fail=False)
        buggy_capsule = capture_revision(buggy, expect_fail=True)
        healthy_path = healthy_capsule.save(output / "healthy.mlcap", overwrite=True)
        buggy_path = buggy_capsule.save(output / "buggy.mlcap", overwrite=True)
        comparison = compare_runs(healthy_capsule, buggy_capsule, n_resamples=20)

        def revision_fails(seed: int) -> bool:
            cwd = git.working_directory or str(repo)
            os.environ["MLFORENSICS_EXAMPLE_ROOT"] = cwd
            try:
                _run_attention(batch, example_root=cwd)
            except ValueError as exc:
                return "non-finite attention loss" in str(exc)
            return False

        report = bisect_commits(
            git,
            revision_fails,
            seeds=[0],
            good=healthy,
            bad=buggy,
            failure_predicate=lambda value: bool(value) is True,
            min_observations=1,
        )
        git.checkout(buggy)
        os.environ["MLFORENSICS_EXAMPLE_ROOT"] = git.working_directory or str(repo)
        replayed = replay_with_factory(buggy_capsule)
        reduced, child = shrink_capsule(
            buggy_capsule,
            lambda candidate: _fails(candidate, git.working_directory or str(repo)),
            output=output / "shrunk.mlcap",
            fixture=AttentionReplayFixture(),
            verify=False,
        )
        child_replay = replay_in_worker(
            output / "shrunk.mlcap",
            working_directory=git.working_directory,
            timeout=60,
        )
        return {
            "healthy_revision": healthy,
            "buggy_revision": buggy,
            "first_bad": report.first_bad,
            "comparison_status": comparison.status,
            "replayed": replayed.reproduced,
            "reduced": reduced.metadata.get("final_nested_size"),
            "original": reduced.metadata.get("original_nested_size"),
            "child_replayed": child_replay.reproduced,
            "trace": replayed.metadata.get("tensor_trace") or child.evidence.get("tensor_trace"),
            "healthy_capsule": str(healthy_path),
            "buggy_capsule": str(buggy_path),
            "shrunk_capsule": str(output / "shrunk.mlcap"),
        }
    finally:
        git.cleanup()


def _fails(candidate: Any, example_root: str) -> bool:
    try:
        _run_attention(candidate, example_root=example_root)
    except ValueError as exc:
        return "non-finite attention loss" in str(exc)
    return False
