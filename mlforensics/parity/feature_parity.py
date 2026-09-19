"""Feature parity adapters and dependency-free tabular comparison.

``PythonFeatureAdapter`` handles row or batch feature functions,
``PandasFeatureAdapter`` handles DataFrame-oriented functions, and
``DuckDBFeatureAdapter`` executes SQL against an input table.  Pandas and
DuckDB are optional: neither is imported until its adapter is executed.
"""

from __future__ import annotations

import datetime as _datetime
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from numbers import Number
from typing import Any, Literal

from .backends import OptionalDependencyError
from .data import MISSING, TableData, as_table, is_null, row_records, type_family


class FeatureDependencyError(OptionalDependencyError):
    """Raised when an adapter needs an uninstalled optional dependency."""


class FeatureAdapterError(RuntimeError):
    """Raised when a feature adapter cannot execute its transformation."""


@dataclass(frozen=True)
class FeatureParityPolicy:
    """Checks applied to feature outputs."""

    absolute: float = 1e-6
    relative: float = 1e-5
    check_nulls: bool = True
    check_types: bool = True
    check_order: bool = True
    check_precision: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.absolute, bool)
            or isinstance(self.relative, bool)
            or not math.isfinite(float(self.absolute))
            or not math.isfinite(float(self.relative))
            or self.absolute < 0
            or self.relative < 0
        ):
            raise ValueError("absolute and relative tolerances must be non-negative")

    @property
    def atol(self) -> float:
        return self.absolute

    @property
    def rtol(self) -> float:
        return self.relative


FeatureTolerance = FeatureParityPolicy


@dataclass(frozen=True)
class FeatureMismatch:
    """One schema, row, null, type, order, or precision difference."""

    kind: str
    row: int | None = None
    column: str | None = None
    reference: Any = None
    candidate: Any = None
    message: str = ""

    @property
    def path(self) -> str:
        if self.row is None and self.column is None:
            return "$"
        if self.row is None:
            return f"$.{self.column}"
        if self.column is None:
            return f"$[{self.row}]"
        return f"$[{self.row}].{self.column}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "row": self.row,
            "column": self.column,
            "path": self.path,
            "reference": _json_value(self.reference),
            "candidate": _json_value(self.candidate),
            "message": self.message,
        }


@dataclass
class FeatureParityResult:
    """Result of comparing two feature transformations."""

    passed: bool
    mismatches: tuple[FeatureMismatch, ...] = ()
    reference: TableData | None = field(default=None, repr=False)
    candidate: TableData | None = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def equal(self) -> bool:
        return self.passed

    @property
    def mismatch_count(self) -> int:
        return len(self.mismatches)

    def __bool__(self) -> bool:
        return self.passed

    def summary(self) -> str:
        if self.passed:
            return "PASS: feature outputs matched"
        return f"FAIL: {self.mismatch_count} feature mismatch(es)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "equal": self.passed,
            "mismatch_count": self.mismatch_count,
            "mismatches": [mismatch.to_dict() for mismatch in self.mismatches],
            "metadata": dict(self.metadata),
        }


def _json_value(value: Any) -> Any:
    if value is MISSING:
        return "<missing>"
    if is_null(value):
        return None
    if isinstance(value, (_datetime.datetime, _datetime.date, _datetime.time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _numeric(value: Any) -> bool:
    return isinstance(value, (Number, Decimal)) and not isinstance(value, bool)


def _values_equal(left: Any, right: Any, policy: FeatureParityPolicy) -> bool:
    if left is MISSING or right is MISSING:
        return left is right
    left_null, right_null = is_null(left), is_null(right)
    if left_null or right_null:
        return left_null and right_null
    if _numeric(left) and _numeric(right):
        if not policy.check_precision:
            return left == right
        try:
            difference = abs(float(left) - float(right))
            scale = max(abs(float(left)), abs(float(right)))
            return difference <= policy.absolute + policy.relative * scale
        except (OverflowError, TypeError, ValueError):
            return left == right
    try:
        result = left == right
        return result if isinstance(result, bool) else bool(result)
    except (TypeError, ValueError):
        return repr(left) == repr(right)


def _canonical_row(row: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(
        ("<null>",) if is_null(value) else (type_family(value), repr(value)) for value in row
    )


def compare_feature_tables(
    reference: Any,
    candidate: Any,
    *,
    policy: FeatureParityPolicy | None = None,
    reference_columns: Sequence[str] | None = None,
    candidate_columns: Sequence[str] | None = None,
    max_mismatches: int | None = 100,
) -> FeatureParityResult:
    """Compare two tabular feature outputs with explicit schema semantics."""

    policy = policy or FeatureParityPolicy()
    left, right = (
        as_table(reference, columns=reference_columns),
        as_table(candidate, columns=candidate_columns),
    )
    mismatches: list[FeatureMismatch] = []

    def add(kind: str, **kwargs: Any) -> None:
        if max_mismatches is None or len(mismatches) < max_mismatches:
            mismatches.append(FeatureMismatch(kind, **kwargs))

    if left.columns != right.columns:
        add(
            "schema",
            message=(
                f"column order/name differs: reference={left.columns!r}, "
                f"candidate={right.columns!r}"
            ),
        )
    if left.row_count != right.row_count:
        add(
            "shape",
            message=f"row count differs: reference={left.row_count}, candidate={right.row_count}",
        )
    columns = tuple(dict.fromkeys((*left.columns, *right.columns)))
    left_index = {name: index for index, name in enumerate(left.columns)}
    right_index = {name: index for index, name in enumerate(right.columns)}

    for column in columns:
        if column not in left_index or column not in right_index:
            continue
        left_type = left.dtypes[left_index[column]] if left.dtypes else None
        right_type = right.dtypes[right_index[column]] if right.dtypes else None
        if (
            policy.check_types
            and left_type
            and right_type
            and type_family(None, left_type) != type_family(None, right_type)
        ):
            add(
                "type",
                column=column,
                reference=left_type,
                candidate=right_type,
                message="column types differ",
            )

    row_pairs = zip(left.rows, right.rows) if policy.check_order else ()
    for row_index, (left_row, right_row) in enumerate(row_pairs):
        for column in columns:
            if column not in left_index or column not in right_index:
                continue
            left_value, right_value = left_row[left_index[column]], right_row[right_index[column]]
            left_declared = left.dtypes[left_index[column]] if left.dtypes else None
            right_declared = right.dtypes[right_index[column]] if right.dtypes else None
            left_null, right_null = is_null(left_value), is_null(right_value)
            if left_null or right_null:
                if policy.check_nulls and left_null != right_null:
                    add(
                        "null",
                        row=row_index,
                        column=column,
                        reference=left_value,
                        candidate=right_value,
                        message="null behavior differs",
                    )
                continue
            if policy.check_types and type_family(left_value, left_declared) != type_family(
                right_value, right_declared
            ):
                add(
                    "type",
                    row=row_index,
                    column=column,
                    reference=left_value,
                    candidate=right_value,
                    message="value types differ",
                )
                continue
            if not _values_equal(left_value, right_value, policy):
                kind = "precision" if _numeric(left_value) and _numeric(right_value) else "value"
                add(
                    kind,
                    row=row_index,
                    column=column,
                    reference=left_value,
                    candidate=right_value,
                    message="feature values differ",
                )

    if (
        not policy.check_order
        and left.columns == right.columns
        and left.row_count == right.row_count
    ):
        unmatched = list(right.rows)
        for left_row in left.rows:
            match = next(
                (
                    index
                    for index, right_row in enumerate(unmatched)
                    if _rows_equal(left_row, right_row, left, right, policy)
                ),
                None,
            )
            if match is None:
                add("value", message="row multisets differ when row ordering is ignored")
                break
            unmatched.pop(match)

    if (
        policy.check_order
        and left.row_count == right.row_count
        and (
            tuple(_canonical_row(row) for row in left.rows)
            != tuple(_canonical_row(row) for row in right.rows)
        )
    ):
        left_multiset = Counter(_canonical_row(row) for row in left.rows)
        right_multiset = Counter(_canonical_row(row) for row in right.rows)
        if left_multiset == right_multiset:
            add("order", message="rows contain the same values in a different order")
    return FeatureParityResult(
        not mismatches, tuple(mismatches), left, right, {"columns": list(columns)}
    )


def _rows_equal(
    left_row: tuple[Any, ...],
    right_row: tuple[Any, ...],
    left: TableData,
    right: TableData,
    policy: FeatureParityPolicy,
) -> bool:
    for index, (left_value, right_value) in enumerate(zip(left_row, right_row)):
        left_declared = left.dtypes[index] if left.dtypes else None
        right_declared = right.dtypes[index] if right.dtypes else None
        left_null, right_null = is_null(left_value), is_null(right_value)
        if left_null or right_null:
            if policy.check_nulls and left_null != right_null:
                return False
            continue
        if policy.check_types and type_family(left_value, left_declared) != type_family(
            right_value, right_declared
        ):
            return False
        if not _values_equal(left_value, right_value, policy):
            return False
    return True


@dataclass
class PythonFeatureAdapter:
    """Adapter for a Python row or batch feature function."""

    function: Callable[[Any], Any]
    mode: Literal["row", "batch"] = "row"
    name: str = "python"
    columns: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("feature function must be callable")
        if self.mode not in {"row", "batch"}:
            raise ValueError("mode must be 'row' or 'batch'")

    def transform(self, data: Any) -> Any:
        if self.mode == "batch":
            return self.function(data)
        records = row_records(data, columns=self.columns)
        return [self.function(record) for record in records]

    def predict(self, data: Any) -> Any:
        return self.transform(data)


@dataclass
class PandasFeatureAdapter:
    """Adapter for a function whose input and output are Pandas objects."""

    function: Callable[[Any], Any]
    name: str = "pandas"
    columns: Sequence[str] | None = None

    def _pandas(self) -> Any:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise FeatureDependencyError(
                "PandasFeatureAdapter requires the optional 'pandas' dependency; "
                "install pandas (for example, `pip install pandas`)."
            ) from exc
        return pd

    def transform(self, data: Any) -> Any:
        pd = self._pandas()
        if getattr(data, "__class__", None).__module__.split(".")[0] == "pandas":
            frame = data
        else:
            frame = pd.DataFrame(row_records(data, columns=self.columns))
        return self.function(frame)

    def predict(self, data: Any) -> Any:
        return self.transform(data)


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class DuckDBFeatureAdapter:
    """Adapter executing a SQL feature query against a table named ``table``."""

    query: str
    connection: Any | None = None
    table: str = "features"
    name: str = "duckdb"
    parameters: Sequence[Any] | Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("DuckDB feature query cannot be empty")
        if not _IDENTIFIER.fullmatch(self.table):
            raise ValueError("table must be a simple SQL identifier")

    def _duckdb(self) -> Any:
        try:
            import duckdb  # type: ignore
        except ImportError as exc:
            raise FeatureDependencyError(
                "DuckDBFeatureAdapter requires the optional 'duckdb' dependency; "
                "install duckdb (for example, `pip install duckdb`)."
            ) from exc
        return duckdb

    def transform(self, data: Any) -> TableData:
        connection = self.connection
        owned = connection is None
        if owned:
            connection = self._duckdb().connect()
        try:
            self._register(connection, data)
            result = (
                connection.execute(self.query, self.parameters)
                if self.parameters is not None
                else connection.execute(self.query)
            )
            description = getattr(result, "description", None) or ()
            columns = tuple(str(item[0]) for item in description)
            rows = tuple(tuple(row) for row in result.fetchall())
            dtypes = tuple(_duckdb_type(item) for item in description)
            return TableData(columns, rows, dtypes)
        except FeatureDependencyError:
            raise
        except Exception as exc:
            raise FeatureAdapterError(f"DuckDB feature query failed: {exc}") from exc
        finally:
            if owned:
                connection.close()

    def _register(self, connection: Any, data: Any) -> None:
        if hasattr(data, "columns") and callable(getattr(connection, "register", None)):
            connection.register(self.table, data)
            return
        table_data = as_table(data)
        columns = table_data.columns or ("feature",)
        definitions = ", ".join(
            f"{_quote_identifier(column)} {_sql_type(table_data, index)}"
            for index, column in enumerate(columns)
        )
        connection.execute(
            f"CREATE OR REPLACE TEMP TABLE {_quote_identifier(self.table)} ({definitions})"
        )
        if table_data.rows:
            placeholders = ", ".join("?" for _ in columns)
            connection.executemany(
                f"INSERT INTO {_quote_identifier(self.table)} VALUES ({placeholders})",
                table_data.rows,
            )

    def predict(self, data: Any) -> Any:
        return self.transform(data)


def _sql_type(table: TableData, index: int) -> str:
    declared = table.dtypes[index] if table.dtypes else None
    family = (
        type_family(None, declared)
        if declared
        else next(
            (
                type_family(value)
                for value in (row[index] for row in table.rows)
                if not is_null(value)
            ),
            None,
        )
    )
    return {
        "integer": "BIGINT",
        "number": "DOUBLE",
        "boolean": "BOOLEAN",
        "datetime": "TIMESTAMP",
        "string": "VARCHAR",
    }.get(family or "", "VARCHAR")


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _duckdb_type(description: Any) -> str | None:
    if len(description) < 2 or description[1] is None:
        return None
    return str(description[1])


class FeatureParityComparator:
    """Run two feature adapters and compare their tabular outputs."""

    def __init__(self, policy: FeatureParityPolicy | None = None) -> None:
        self.policy = policy or FeatureParityPolicy()

    def compare(self, reference: Any, candidate: Any, data: Any) -> FeatureParityResult:
        reference_output = _transform(reference, data)
        candidate_output = _transform(candidate, data)
        return compare_feature_tables(reference_output, candidate_output, policy=self.policy)

    __call__ = compare


def _transform(adapter: Any, data: Any) -> Any:
    if callable(adapter):
        return PythonFeatureAdapter(adapter).transform(data)
    method = getattr(adapter, "transform", None) or getattr(adapter, "predict", None)
    if not callable(method):
        raise TypeError("feature adapter must expose transform() or predict()")
    return method(data)


def compare_features(
    reference: Any,
    candidate: Any,
    data: Any,
    *,
    policy: FeatureParityPolicy | None = None,
) -> FeatureParityResult:
    """Convenience wrapper for :class:`FeatureParityComparator`."""

    return FeatureParityComparator(policy).compare(reference, candidate, data)


# Short names make the adapters convenient to discover while retaining the
# explicit names in documentation and error messages.
PythonAdapter = PythonFeatureAdapter
PandasAdapter = PandasFeatureAdapter
DuckDBAdapter = DuckDBFeatureAdapter
SQLFeatureAdapter = DuckDBFeatureAdapter
FeatureParity = FeatureParityComparator

__all__ = [
    "DuckDBAdapter",
    "DuckDBFeatureAdapter",
    "FeatureAdapterError",
    "FeatureDependencyError",
    "FeatureMismatch",
    "FeatureParity",
    "FeatureParityComparator",
    "FeatureParityPolicy",
    "FeatureParityResult",
    "FeatureTolerance",
    "PandasAdapter",
    "PandasFeatureAdapter",
    "PythonAdapter",
    "PythonFeatureAdapter",
    "SQLFeatureAdapter",
    "compare_feature_tables",
    "compare_features",
]
