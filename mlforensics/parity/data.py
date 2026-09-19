"""Small dependency-free tabular helpers for feature parity.

The public adapters accept ordinary Python records as well as DataFrame-like
objects.  Keeping the comparison representation here deliberately small means
that importing feature parity never imports Pandas, NumPy, or DuckDB.
"""

from __future__ import annotations

import datetime as _datetime
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from numbers import Number
from typing import Any

MISSING = object()


@dataclass(frozen=True)
class TableData:
    """A normalized, ordered table used by feature parity comparisons."""

    columns: tuple[str, ...]
    rows: tuple[tuple[Any, ...], ...]
    dtypes: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        if self.dtypes and len(self.dtypes) != len(self.columns):
            raise ValueError("dtypes must contain one entry per column")
        for row in self.rows:
            if len(row) != len(self.columns):
                raise ValueError("every table row must contain one value per column")

    @property
    def row_count(self) -> int:
        return len(self.rows)


def is_null(value: Any) -> bool:
    """Return whether *value* is a scalar null, without importing Pandas."""

    if value is None:
        return True
    type_name = type(value).__name__
    if type_name in {"NAType", "NaTType"}:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, Decimal):
        return value.is_nan()
    # NumPy scalar NaN values and similar objects support ``!=``.  Avoid
    # calling bool() on vector-like results, notably pandas.NA.
    try:
        different = value != value
    except Exception:
        return False
    try:
        return bool(different)
    except (TypeError, ValueError):
        return False


def type_family(value: Any, declared: str | None = None) -> str | None:
    """Map runtime/declared types to stable cross-backend families."""

    if declared:
        name = declared.lower().strip()
        if name in {"object", "any"}:
            declared = None
        else:
            if "categor" in name:
                return "category"
            if any(token in name for token in ("datetime", "timestamp", "date", "time")):
                return "datetime"
            if any(token in name for token in ("bool",)):
                return "boolean"
            if any(token in name for token in ("int", "uint")):
                return "integer"
            if any(token in name for token in ("float", "double", "real", "decimal", "numeric")):
                return "number"
            if any(token in name for token in ("str", "string", "varchar", "text")):
                return "string"

    if is_null(value):
        return None
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, Number) or isinstance(value, Decimal):
        return "number"
    if isinstance(value, (_datetime.date, _datetime.datetime, _datetime.time)):
        return "datetime"
    if isinstance(value, str):
        return "string"
    return type(value).__name__


def _dataframe_table(value: Any) -> TableData | None:
    """Extract a DataFrame-like value by protocol, not by importing Pandas."""

    to_frame = getattr(value, "to_frame", None)
    if not hasattr(value, "columns") and callable(to_frame):
        return _dataframe_table(to_frame())
    columns = getattr(value, "columns", None)
    to_dict = getattr(value, "to_dict", None)
    if columns is None or not callable(to_dict):
        return None
    try:
        column_names = tuple(str(column) for column in columns)
        records = to_dict(orient="records")
    except (TypeError, AttributeError, ValueError):
        return None
    rows = tuple(tuple(record.get(column, MISSING) for column in columns) for record in records)
    dtypes_value = getattr(value, "dtypes", None)
    try:
        dtypes = tuple(str(dtype) for dtype in dtypes_value) if dtypes_value is not None else ()
    except TypeError:
        dtypes = ()
    return TableData(column_names, rows, dtypes)


def _records_table(records: Sequence[Any], columns: Sequence[str] | None) -> TableData:
    values = list(records)
    if not values:
        names = tuple(str(column) for column in (columns or ()))
        return TableData(names, ())
    if all(isinstance(item, Mapping) for item in values):
        discovered = list(str(column) for column in (columns or ()))
        for item in values:
            for key in item:
                key = str(key)
                if key not in discovered:
                    discovered.append(key)
        names = tuple(discovered)
        rows = tuple(tuple(item.get(name, MISSING) for name in names) for item in values)
        return TableData(names, rows)
    if any(isinstance(item, Mapping) for item in values):
        raise TypeError("table rows must all be mappings or all be sequences")
    if columns is None:
        width = (
            len(values[0])
            if isinstance(values[0], Sequence) and not isinstance(values[0], (str, bytes))
            else 1
        )
        names = tuple(f"feature_{index}" for index in range(width))
    else:
        names = tuple(str(column) for column in columns)
    rows = []
    for item in values:
        row = item if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) else (item,)
        if len(row) != len(names):
            raise ValueError("table row width differs from the declared columns")
        rows.append(tuple(row))
    return TableData(names, tuple(rows))


def as_table(value: Any, *, columns: Sequence[str] | None = None) -> TableData:
    """Normalize a DataFrame, column mapping, record sequence, or scalar."""

    if isinstance(value, TableData):
        return value
    dataframe = _dataframe_table(value)
    if dataframe is not None:
        return dataframe
    if isinstance(value, Mapping):
        values = list(value.values())
        if values and all(
            isinstance(item, Sequence) and not isinstance(item, (str, bytes)) for item in values
        ):
            lengths = {len(item) for item in values}
            if len(lengths) != 1:
                raise ValueError("column mapping values must have equal lengths")
            names = tuple(str(column) for column in value)
            return TableData(names, tuple(zip(*values)))
        names = tuple(str(column) for column in (columns or value.keys()))
        return TableData(names, (tuple(value.get(column, MISSING) for column in names),))
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return as_table(tolist(), columns=columns)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _records_table(value, columns)
    name = str(columns[0]) if columns and len(columns) == 1 else "feature"
    return TableData((name,), ((value,),))


def row_records(value: Any, *, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """Return normalized rows as mappings, retaining column order."""

    table = as_table(value, columns=columns)
    return [dict(zip(table.columns, row)) for row in table.rows]
