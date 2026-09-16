from mlforensics.diagnose.runner import RunResult, run_many


def test_runner_is_seed_ordered_and_normalises_results():
    seen = []
    result = run_many(lambda seed: seen.append(seed) or {"metric": seed / 10}, [3, 1, 3])
    assert seen == [3, 1, 3]
    assert [x.result.metric for x in result] == [0.3, 0.1, 0.3]


def test_runner_captures_exceptions():
    result = run_many(lambda seed: 1 / 0, [7])[0].result
    assert result == RunResult(seed=7, ok=False, error="ZeroDivisionError: division by zero")


def test_runner_supports_keyword_only_seed():
    result = run_many(lambda *, seed: seed + 1, [4])[0].result
    assert result.metric == 5.0
