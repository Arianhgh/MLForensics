import pytest

from mlforensics.diagnose.shrink import InitialPredicateError, shrink, shrink_columns


def test_shrinker_finds_minimal_failing_rows_and_columns():
    report = shrink(
        [[0, 0], [0, 0], [9, 0], [0, 0]],
        lambda x: any(9 in row for row in x),
        kind="rows",
    )
    assert report.value == [[9, 0]]
    assert report.predicate_calls > 0


def test_token_shrink_is_deterministic():
    def predicate(value):
        return "needle" in value

    one = shrink("remove lots needle words".split(), predicate, kind="tokens").value
    two = shrink("remove lots needle words".split(), predicate, kind="tokens").value
    assert one == two == ["needle"]


def test_numpy_tensor_shrink_removes_rows_and_columns():
    np = pytest.importorskip("numpy")
    value = np.asarray([[0, 0, 0], [0, 7, 0], [0, 0, 0]])
    report = shrink(value, lambda x: 7 in x, kind="tensor")
    assert report.final_size == 1
    assert report.value.shape == value.shape
    assert report.value.dtype == value.dtype
    assert report.metadata["shape_preserved"] is True


def test_column_shrinker_keeps_only_failing_column():
    rows = [{"noise": 0, "signal": 1}, {"noise": 0, "signal": 1}]
    report = shrink_columns(
        rows,
        lambda candidate: all(row.get("signal") == 1 for row in candidate),
    )
    assert list(report.value[0]) == ["signal"]


def test_shrinker_refuses_a_non_failing_or_raising_initial_value():
    with pytest.raises(InitialPredicateError, match="does not satisfy"):
        shrink([1, 2], lambda value: 9 in value, kind="rows")

    with pytest.raises(InitialPredicateError, match="predicate raised"):
        shrink([1, 2], lambda value: 1 / 0, kind="rows")


def test_text_and_tuple_shrinking_preserve_the_input_type():
    text = shrink("noise keep more", lambda value: "keep" in value, kind="tokens")
    assert text.value == "keep"
    assert isinstance(text.value, str)

    values = shrink((0, 7, 0), lambda value: 7 in value, kind="sequence")
    assert values.value == (7,)
    assert isinstance(values.value, tuple)


def test_structured_shrinking_removes_noise_and_simplifies_values():
    value = {"noise": [1, 2, 3], "request": {"token": "bad", "padding": 99}}
    report = shrink(
        value,
        lambda candidate: candidate.get("request", {}).get("token") == "bad",
        kind="structured",
    )
    assert report.value == {"request": {"token": "bad"}}
    assert report.metadata["verified"] is True
