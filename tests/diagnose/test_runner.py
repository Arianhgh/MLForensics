import os
import random
import time

from mlforensics.diagnose.runner import run_many


def test_runner_is_seed_ordered_and_normalises_results():
    seen = []
    result = run_many(lambda seed: seen.append(seed) or {"metric": seed / 10}, [3, 1, 3])
    assert seen == [3, 1, 3]
    assert [x.result.metric for x in result] == [0.3, 0.1, 0.3]


def test_runner_captures_exceptions():
    result = run_many(lambda seed: 1 / 0, [7])[0].result
    assert result.seed == 7
    assert result.ok is False
    assert result.status == "error"
    assert result.error == "ZeroDivisionError: division by zero"
    assert result.exception == {
        "type": "ZeroDivisionError",
        "qualified_type": "builtins.ZeroDivisionError",
        "message": "division by zero",
        "phase": "runner",
        "traceback": result.exception["traceback"],
    }


def test_runner_supports_keyword_only_seed():
    result = run_many(lambda *, seed: seed + 1, [4])[0].result
    assert result.metric == 5.0


def test_runner_seeds_in_process_deterministically_and_restores_parent_rng():
    random.seed(99)
    expected = random.random()
    random.seed(99)

    def runner(seed):
        return random.random()

    first = run_many(runner, [12])[0].result
    second = run_many(runner, [12])[0].result

    assert first.value == second.value
    assert random.random() == expected
    assert first.metadata["rng"]["python"] is True


def test_runner_supports_fresh_process_and_returns_status():
    parent_pid = os.getpid()
    result = run_many(lambda seed: {"value": os.getpid(), "ok": True}, [3], fresh_process=True)[0]

    assert result.result.ok is True
    assert result.result.status == "ok"
    assert result.result.value != parent_pid


def test_runner_timeout_is_structured_in_a_fresh_process():
    def slow_runner(seed):
        time.sleep(1.0)
        return seed

    result = run_many(slow_runner, [5], fresh_process=True, timeout=0.05)[0].result

    assert result.ok is False
    assert result.status == "timeout"
    assert result.timed_out is True
    assert result.exception["type"] == "TimeoutError"
    assert result.metadata["timeout"] == 0.05


def test_runner_normalises_failure_status_from_mapping():
    result = run_many(lambda seed: {"status": "timeout", "value": None}, [1])[0].result

    assert result.ok is False
    assert result.status == "timeout"
