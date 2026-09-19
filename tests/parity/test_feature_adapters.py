"""Feature parity adapter contracts and optional dependency behavior."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mlforensics.parity.data import as_table
from mlforensics.parity.feature_parity import (
    DuckDBFeatureAdapter,
    FeatureAdapterError,
    FeatureDependencyError,
    FeatureParityPolicy,
    PandasFeatureAdapter,
    PythonFeatureAdapter,
    compare_feature_tables,
    compare_features,
)


def test_python_row_and_batch_adapters_have_the_same_table_contract() -> None:
    rows = [{"x": 1, "kind": "a"}, {"x": 2, "kind": "b"}]
    row_adapter = PythonFeatureAdapter(lambda row: {"score": row["x"] * 2})
    batch_adapter = PythonFeatureAdapter(
        lambda values: [{"score": row["x"] * 2} for row in values], mode="batch"
    )

    assert compare_features(row_adapter, batch_adapter, rows).passed


def test_feature_comparison_reports_null_type_order_and_precision_mismatches() -> None:
    reference = [{"score": 1.0, "label": None}, {"score": 2.0, "label": "ok"}]
    candidate = [{"score": 1.1, "label": ""}, {"score": 2, "label": "ok"}]
    result = compare_feature_tables(
        reference,
        candidate,
        policy=FeatureParityPolicy(absolute=1e-8, relative=0),
    )

    assert not result.passed
    assert {item.kind for item in result.mismatches} >= {"null", "precision"}
    assert result.mismatches[0].path.startswith("$")

    reordered = compare_feature_tables(
        [{"x": 1}, {"x": 2}],
        [{"x": 2}, {"x": 1}],
    )
    assert any(item.kind == "order" for item in reordered.mismatches)


def test_column_order_is_part_of_the_schema_contract() -> None:
    result = compare_feature_tables(
        [{"first": 1, "second": 2}],
        [{"second": 2, "first": 1}],
    )
    assert not result.passed
    assert result.mismatches[0].kind == "schema"


def test_order_can_be_ignored_without_disabling_precision_checks() -> None:
    result = compare_feature_tables(
        [{"x": 1.0}, {"x": 2.0}],
        [{"x": 2.0000001}, {"x": 1.0}],
        policy=FeatureParityPolicy(absolute=1e-5, relative=0, check_order=False),
    )
    assert result.passed

    strict = compare_feature_tables(
        [{"x": None}],
        [{"x": 1}],
        policy=FeatureParityPolicy(check_nulls=False),
    )
    assert strict.passed


def test_pandas_adapter_imports_dependency_only_on_execution() -> None:
    adapter = PandasFeatureAdapter(lambda frame: frame)
    try:
        import pandas  # noqa: F401
    except ImportError:
        with pytest.raises(FeatureDependencyError, match=r"install pandas"):
            adapter.transform([{"x": 1}])


def test_duckdb_adapter_accepts_existing_connection_without_importing_duckdb() -> None:
    class Result:
        description = [("score", "DOUBLE")]

        def fetchall(self):
            return [(1.0,)]

    class Connection:
        def __init__(self):
            self.queries = []

        def register(self, name, value):
            self.queries.append(("register", name, value))

        def execute(self, query):
            self.queries.append(("execute", query))
            return Result()

    connection = Connection()
    adapter = DuckDBFeatureAdapter("SELECT score FROM features", connection=connection)
    output = adapter.transform(
        SimpleNamespace(columns=("score",), to_dict=lambda orient: [{"score": 1.0}])
    )
    assert as_table(output).rows == ((1.0,),)
    assert connection.queries[0][0] == "register"


def test_duckdb_missing_dependency_has_a_clear_error() -> None:
    try:
        import duckdb  # noqa: F401
    except ImportError:
        with pytest.raises(FeatureDependencyError, match=r"install duckdb"):
            DuckDBFeatureAdapter("SELECT 1").transform([{"x": 1}])


def test_duckdb_query_failures_are_actionable() -> None:
    class Connection:
        def execute(self, query):
            raise ValueError("bad SQL")

    with pytest.raises(FeatureAdapterError, match="feature query failed"):
        DuckDBFeatureAdapter("SELECT 1", connection=Connection()).transform([{"x": 1}])
