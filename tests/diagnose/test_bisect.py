import subprocess
from pathlib import Path

from mlforensics.core import FailureSignature
from mlforensics.core.contracts import PredicateResult
from mlforensics.core.errors import UnresolvedEvaluation
from mlforensics.diagnose.bisect import (
    BisectCache,
    RegressionStatus,
    StochasticBisector,
    WorktreeGit,
    bisect_commits,
    decide_regression,
)


def test_decision_is_uncertain_for_crossing_interval():
    bisector = StochasticBisector(
        "good",
        "bad",
        lambda target, seed: 1 if target == "good" else seed,
        seeds=[1, 2, 3, 4, 5],
        higher_is_better=False,
    )
    outcome = bisector.evaluate("good")
    candidate = bisector.evaluate("bad")
    assert outcome.passed and candidate.confidence_interval
    assert candidate.inconclusive is False


class FakeGit:
    def __init__(self):
        self.revisions = ["a", "b", "c", "d", "e"]
        self.current = None
        self.checkouts = []


def test_bisect_uses_matched_seeds_and_cache():
    calls = []
    seeds = [11, 12, 13, 14, 15]

    def runner(target, seed):
        calls.append((target, seed))
        return 0.0 if target in {"a", "b"} else 10.0

    cache = {}
    report = StochasticBisector(
        "a", "e", runner, seeds=seeds, higher_is_better=False, cache=cache
    ).run(["a", "b", "c", "d", "e"])
    assert report.first_bad == "c"
    expected = [(rev, seed) for rev in ("a", "e", "c", "b") for seed in seeds]
    assert calls == expected
    second = StochasticBisector(
        "a", "e", runner, seeds=seeds, higher_is_better=False, cache=cache
    ).run(["a", "b", "c", "d", "e"])
    assert second.first_bad == "c" and len(calls) == 20


def test_commit_bisect_stops_on_inconclusive_and_restores_revision():
    class Git:
        current = "main"

        def commits(self, good, bad):
            return ["a", "b", "c"]

        def checkout(self, revision):
            self.current = revision

        def current_revision(self):
            return self.current

        def restore(self, revision):
            self.current = revision

    git = Git()
    mixed = {1: -1.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 1.0}

    def runner(seed):
        return {"a": 0.0, "b": mixed[seed], "c": 1.0}[git.current]

    report = bisect_commits(
        git,
        runner,
        [1, 2, 3, 4, 5],
        good="a",
        bad="c",
        higher_is_better=False,
        n_resamples=100,
    )
    assert report.first_bad is None
    assert report.inconclusive == ["b"]
    assert report.decisions["b"].status is RegressionStatus.INCONCLUSIVE
    assert git.current == "main"


def test_bisect_cache_round_trips_to_disk(tmp_path):
    path = tmp_path / "bisect-cache.json"
    cache = BisectCache(path=path)
    cache.set("abc", 11, {"score": 0.5})

    loaded = BisectCache(path=path)
    assert loaded.get("abc", 11) == {"score": 0.5}


def test_one_seed_stochastic_search_is_inconclusive():
    report = StochasticBisector(
        "good",
        "bad",
        lambda target, seed: 0 if target == "good" else 10,
        seeds=[11],
        higher_is_better=False,
    ).run(["good", "bad"])
    assert report.first_bad is None
    assert report.evaluations[0].inconclusive


class _EndpointGit:
    def __init__(self):
        self.current = None

    def commits(self, good, bad):
        return ["good", "mid", "bad"]

    def checkout(self, revision):
        self.current = revision


def test_typed_and_boolean_predicates_classify_the_same_revisions():
    def runner(seed):
        return "boom" if git.current != "good" else "ok"

    def as_bool(value):
        return value == "boom"

    def as_typed(value):
        preserved = value == "boom"
        return PredicateResult(preserved=preserved, status="fail" if preserved else "pass")

    git = _EndpointGit()
    boolean = bisect_commits(git, runner, [11], good="good", bad="bad", failure_predicate=as_bool)
    git = _EndpointGit()
    typed = bisect_commits(git, runner, [11], good="good", bad="bad", failure_predicate=as_typed)
    assert boolean.first_bad == typed.first_bad == "mid"
    assert boolean.evaluations[0].passed is True
    assert typed.evaluations[0].passed is True
    assert boolean.evaluations[0].inconclusive is False


def test_unresolved_predicate_cannot_establish_a_good_endpoint_or_culprit():
    git = _EndpointGit()

    def runner(seed):
        return "boom" if git.current != "good" else "ok"

    report = bisect_commits(
        git,
        runner,
        [11],
        good="good",
        bad="bad",
        failure_predicate=lambda _value: (_ for _ in ()).throw(UnresolvedEvaluation("lost")),
    )
    assert report.first_bad is None
    assert report.evaluations[0].inconclusive
    assert report.evaluations[0].passed is True


def test_decide_regression_rejects_a_single_pair_and_serializes():
    import json

    decision = decide_regression([0.0], [1.0], higher_is_better=False)
    assert decision.status is RegressionStatus.INCONCLUSIVE
    encoded = json.dumps(decision.to_dict(), allow_nan=False)
    assert "NaN" not in encoded
    assert json.loads(encoded)["lower"] is None


def test_bisect_cache_restores_failure_signatures_and_rejects_changed_identity(tmp_path):
    path = tmp_path / "cache.json"
    signature = FailureSignature.from_exception(ValueError("target"))
    first = BisectCache(path=path)
    first.set("rev", 11, signature, identity={"command": "one"})

    loaded = BisectCache(path=path)
    restored = loaded.get("rev", 11, identity={"command": "one"})
    assert isinstance(restored, FailureSignature)
    assert loaded.get("rev", 11, identity={"command": "two"}) is None


def test_git_bisect_resumes_from_matching_cached_evidence(tmp_path):
    class Git:
        current = "good"

        def commits(self, *args):
            return ["good", "bad"]

        def checkout(self, revision):
            self.current = revision

    cache = BisectCache(path=tmp_path / "cache.json")
    git = Git()
    first = bisect_commits(
        git,
        lambda seed: 0.0 if git.current == "good" else 10.0,
        [11, 29, 37, 53, 71],
        cache=cache,
        command=["stable-harness"],
    )

    def fail_if_called(seed):
        raise AssertionError("resume executed a cached run")

    second = bisect_commits(
        git,
        fail_if_called,
        [11, 29, 37, 53, 71],
        cache=BisectCache(path=tmp_path / "cache.json"),
        command=["stable-harness"],
    )
    assert first.first_bad == second.first_bad == "bad"
    assert second.metadata["resumed"] is True
    assert second.metadata["executed_runs"] == 0


def test_git_bisect_requires_a_healthy_numeric_good_endpoint():
    class Git:
        current = "good"

        def commits(self, *args):
            return ["good", "bad"]

        def checkout(self, revision):
            self.current = revision

    git = Git()
    values = {11: 0.0, 29: None, 37: None, 53: None, 71: None}
    report = bisect_commits(
        git,
        lambda seed: values[seed] if git.current == "good" else 10.0,
        [11, 29, 37, 53, 71],
    )
    assert report.first_bad is None
    assert report.evaluations[0].inconclusive
    assert report.evaluations[0].metadata["reason"] == "insufficient observations"


def test_git_bisect_budget_exhaustion_is_safe_and_reported():
    class Git:
        current = "good"

        def commits(self, *args):
            return ["good", "bad"]

        def checkout(self, revision):
            self.current = revision

    git = Git()
    report = bisect_commits(
        git,
        lambda seed: 0.0 if git.current == "good" else 10.0,
        [11, 29, 37, 53, 71],
        budget={"run_count": 2},
    )
    assert report.first_bad is None
    assert report.metadata["budget_exhausted"] is True
    assert report.metadata["executed_runs"] == 2


def test_git_bisect_detects_nonmonotonic_history():
    class Git:
        current = "good"

        def commits(self, *args):
            return ["good", "bad-early", "good-again", "bad"]

        def checkout(self, revision):
            self.current = revision

    git = Git()
    scores = {"good": 0.0, "bad-early": 10.0, "good-again": 0.0, "bad": 10.0}
    report = bisect_commits(
        git,
        lambda seed: scores[git.current],
        [11, 29, 37, 53, 71],
        detect_nonmonotonic=True,
    )
    assert report.first_bad is None
    assert report.metadata["nonmonotonic"] is True
    assert report.inconclusive == ["bad-early", "bad"]


def test_worktree_bisect_isolated_runner_and_cleanup(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()

    def git(*args):
        subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    git("init", "-q", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    (repository / "value.txt").write_text("0")
    git("add", "value.txt")
    git("commit", "-qm", "good")
    good = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    (repository / "value.txt").write_text("1")
    git("commit", "-qam", "bad")
    bad = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    (repository / "local.txt").write_text("keep")

    root = tmp_path / "worktrees"
    worktree = WorktreeGit(repository, root=root)
    report = bisect_commits(
        worktree,
        lambda seed, cwd: float((Path(cwd) / "value.txt").read_text()),
        [11, 29, 37, 53, 71],
        good=good,
        bad=bad,
        command=["read-value"],
    )
    assert report.first_bad == bad
    assert (repository / "local.txt").read_text() == "keep"
    assert worktree.working_directory == str(repository.resolve())
    assert root.exists() and not list(root.iterdir())
