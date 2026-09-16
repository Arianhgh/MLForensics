from mlforensics.core.contracts import PredicateResult
from mlforensics.core.errors import UnresolvedEvaluation
from mlforensics.diagnose.bisect import (
    BisectCache,
    RegressionStatus,
    StochasticBisector,
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
