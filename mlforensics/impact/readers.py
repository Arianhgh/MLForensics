"""Dependency readers for text, notebooks, shell files, and pipelines.

The readers in this module are deliberately lexical and conservative.  They
never execute a command, import a notebook, connect to a database, or follow a
path mentioned by an input file.  A reference that cannot be named safely is
represented by a low-confidence ``unknown:...`` node rather than being
silently treated as independent.
"""

from __future__ import annotations

import ast
import json
import os
import posixpath
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - Python 3.10 uses the optional backport
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

from .graph import DependencyGraph, Node

SQL_SUFFIXES = frozenset({".sql", ".ddl", ".dml", ".duckdb", ".dbsql"})
NOTEBOOK_SUFFIXES = frozenset({".ipynb"})
SHELL_SUFFIXES = frozenset({".sh", ".bash", ".zsh", ".ksh", ".fish"})
STRUCTURED_SUFFIXES = frozenset({".json", ".jsonl", ".toml", ".yaml", ".yml"})
SCRIPT_SUFFIXES = frozenset({".py", ".pyw", ".sh", ".bash", ".zsh", ".r", ".rb", ".pl"})
DATA_SUFFIXES = frozenset(
    {
        ".avro",
        ".csv",
        ".db",
        ".duckdb",
        ".feather",
        ".json",
        ".jsonl",
        ".ndjson",
        ".npy",
        ".npz",
        ".parquet",
        ".pkl",
        ".pickle",
        ".pq",
        ".pt",
        ".sqlite",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }
)


@dataclass(frozen=True)
class ReaderDiagnostic:
    """A non-fatal reader problem suitable for logs or a report."""

    path: str
    error: str
    reader: str = "dependency"

    def as_dict(self) -> dict[str, str]:
        return {"path": self.path, "error": self.error, "reader": self.reader}


@dataclass(frozen=True)
class ReaderResult:
    """Graph plus diagnostics for callers that want both pieces of output."""

    graph: DependencyGraph
    errors: tuple[dict[str, str], ...] = ()

    @property
    def diagnostics(self) -> tuple[dict[str, str], ...]:
        return self.errors


@dataclass(frozen=True)
class _LoadedText:
    text: str | None
    display_path: str
    actual_path: Path | None = None
    error: str | None = None


def _is_within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _display_path(value: str | os.PathLike[str] | None) -> str:
    if value is None:
        return "<string>"
    return os.fspath(value).replace("\\", "/")


def _looks_like_file(value: str | os.PathLike[str]) -> bool:
    if isinstance(value, os.PathLike):
        return True
    text = str(value)
    if "\n" in text or "\r" in text:
        return False
    try:
        return Path(text).exists()
    except (OSError, ValueError):
        return False


def _load_text(
    source: str | os.PathLike[str],
    *,
    path: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
    reader: str = "dependency",
) -> _LoadedText:
    """Load an explicitly supplied file or treat ``source`` as text.

    ``root`` is a read boundary for the input file itself.  Referenced paths
    are never opened by any reader, so a ``../`` value in a config cannot
    escape this boundary or cause an unexpected file read.
    """

    path_exists = False
    if path is not None:
        try:
            path_exists = Path(path).expanduser().exists()
        except (OSError, ValueError):
            path_exists = False
    source_is_text_with_label = (
        path is not None
        and isinstance(source, str)
        and not _looks_like_file(source)
        and not path_exists
    )
    explicit = not source_is_text_with_label and (path is not None or _looks_like_file(source))
    if not explicit:
        return _LoadedText(str(source), _display_path(path))
    candidate_value = source if path is None or isinstance(source, os.PathLike) else path
    candidate = Path(candidate_value).expanduser().resolve()
    root_path = Path(root).expanduser().resolve() if root is not None else None
    if root_path is not None and not _is_within(candidate, root_path):
        return _LoadedText(
            None,
            str(candidate),
            candidate,
            f"path is outside the configured root for {reader}",
        )
    try:
        return _LoadedText(candidate.read_text(encoding="utf-8"), str(candidate), candidate)
    except (OSError, UnicodeError, ValueError) as exc:
        return _LoadedText(None, str(candidate), candidate, str(exc))


def _root_node(
    graph: DependencyGraph,
    node_id: str,
    *,
    kind: str,
    name: str,
    path: str,
    reader: str,
    error: str | None = None,
) -> Node:
    metadata: dict[str, Any] = {"reader": reader, "parseable": error is None}
    if error is not None:
        metadata["read_error"] = error
    return graph.add_node(Node(node_id, kind, name, path, metadata))


def _text_node_id(path: str) -> str:
    normalized = path.replace("\\", "/")
    return f"file:{normalized}"


def _stable_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def _strip_matching_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"`":
        return value[1:-1]
    return value


def _reference_label(value: Any) -> str:
    text = _strip_matching_quotes(_stable_text(value))
    if not text:
        return ""
    return text.replace("\\", "/")


def _looks_dynamic(value: str) -> bool:
    text = value.strip()
    return (
        not text
        or text in {"?", ":", "%s", "%d"}
        or "$" in text
        or "{" in text
        or "}" in text
        or "<" in text
        or ">" in text
        or "*" in text
        or text.startswith(("@", ":param"))
    )


def _is_uri(value: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", value))


def _is_path_reference(value: str) -> bool:
    lower = value.lower().split("?", 1)[0]
    return (
        _is_uri(value)
        or lower.startswith(("/", "./", "../", "~/", "file:"))
        or "/" in lower
        or Path(lower).suffix in DATA_SUFFIXES | SCRIPT_SUFFIXES | STRUCTURED_SUFFIXES
    )


def _safe_reference_node(
    graph: DependencyGraph,
    value: Any,
    *,
    kind: str = "dataset",
    root: str | os.PathLike[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str | None:
    """Add a named reference without resolving or reading it."""

    label = _reference_label(value)
    if not label or _looks_dynamic(label):
        return None
    # Normalize dot segments for stable graph IDs, while retaining a marker
    # for a path that escaped a caller-provided root.  The referenced path is
    # still opaque and is never opened.
    if not _is_uri(label) and not label.startswith("file:"):
        normalized = posixpath.normpath(label)
    else:
        normalized = label
    escaped = False
    if root is not None and not _is_uri(normalized) and not Path(normalized).is_absolute():
        root_path = Path(root).expanduser().resolve()
        escaped = not _is_within((root_path / normalized).resolve(), root_path)
    node_label = normalized
    node_id = f"{kind}:{node_label}"
    node_metadata = dict(metadata or {})
    if escaped:
        node_metadata["outside_root"] = True
        node_metadata.setdefault("conservative", True)
    graph.add_node(Node(node_id, kind, node_label, node_label, node_metadata))
    return node_id


def _unknown_dependency(
    graph: DependencyGraph,
    source_id: str,
    *,
    node_id: str,
    relation: str,
    explanation: str,
) -> None:
    graph.add_node(
        Node(
            node_id,
            "unknown",
            node_id.split(":", 1)[-1],
            None,
            {"conservative": True},
        )
    )
    graph.add_edge(
        source_id,
        node_id,
        relation,
        confidence="low",
        explanation=explanation,
    )


def _edge(
    graph: DependencyGraph,
    source_id: str,
    target_id: str,
    kind: str,
    *,
    confidence: str = "high",
    explanation: str,
    **metadata: Any,
) -> None:
    graph.add_edge(
        source_id,
        target_id,
        kind,
        confidence=confidence,
        explanation=explanation,
        **metadata,
    )


# ---------------------------------------------------------------------------
# Lightweight structured data support


def _yaml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith(("[", "{")) and value.endswith(("]", "}")):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            try:
                return ast.literal_eval(value)
            except (SyntaxError, ValueError, TypeError):
                return value
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return _strip_matching_quotes(value)
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)(?:[eE][-+]?\d+)?", value):
            return float(value)
    except ValueError:
        pass
    return value


def _yaml_uncomment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in "'\"":
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
        elif char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.rstrip()


def _yaml_subset(text: str) -> Any:
    """Parse the small YAML subset used by common pipeline declarations.

    This fallback is intentionally not a YAML implementation.  It handles
    indentation, scalar values, and the usual list-of-mapping form.  A full
    parser, when installed, is used through ``safe_load`` instead.
    """

    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if raw.strip() in {"---", "..."}:
            continue
        content = _yaml_uncomment(raw.lstrip(" \t"))
        if not content:
            continue
        lines.append((len(raw) - len(raw.lstrip(" \t")), content))

    def split_key(value: str) -> tuple[str, str] | None:
        quote: str | None = None
        for index, char in enumerate(value):
            if char in "'\"":
                if quote == char:
                    quote = None
                elif quote is None:
                    quote = char
            elif char == ":" and quote is None:
                return value[:index].strip(), value[index + 1 :].strip()
        return None

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        is_list = lines[index][0] == indent and lines[index][1].startswith("-")
        result: Any = [] if is_list else {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                # A malformed indentation is ignored rather than allowed to
                # change the shape of a previously parsed relationship.
                index += 1
                continue
            if is_list:
                if not content.startswith("-"):
                    break
                item = content[1:].strip()
                if not item:
                    if index + 1 < len(lines) and lines[index + 1][0] > indent:
                        child, index = parse_block(index + 1, lines[index + 1][0])
                        result.append(child)
                    else:
                        result.append(None)
                        index += 1
                    continue
                pair = split_key(item)
                if pair is None:
                    result.append(_yaml_scalar(item))
                    index += 1
                    continue
                key, value = pair
                mapping: dict[str, Any] = {str(_strip_matching_quotes(key)): _yaml_scalar(value)}
                index += 1
                if index < len(lines) and lines[index][0] > indent:
                    child, index = parse_block(index, lines[index][0])
                    if isinstance(child, Mapping):
                        mapping.update(child)
                result.append(mapping)
                continue
            pair = split_key(content)
            if pair is None:
                index += 1
                continue
            key, value = pair
            key = str(_strip_matching_quotes(key))
            index += 1
            if value:
                result[key] = _yaml_scalar(value)
            elif index < len(lines) and lines[index][0] > indent:
                child, index = parse_block(index, lines[index][0])
                result[key] = child
            else:
                result[key] = {}
        return result, index

    if not lines:
        return {}
    parsed, _ = parse_block(0, lines[0][0])
    return parsed


def load_structured(
    source: str | os.PathLike[str] | Mapping[str, Any] | Sequence[Any],
    *,
    path: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
) -> tuple[Any, ReaderDiagnostic | None]:
    """Load JSON/TOML/YAML-shaped data without following includes."""

    if isinstance(source, (Mapping, list, tuple)):
        return source, None
    loaded = _load_text(source, path=path, root=root, reader="structured")
    if loaded.error is not None or loaded.text is None:
        return {}, ReaderDiagnostic(loaded.display_path, loaded.error or "unable to read input")
    suffix = Path(loaded.display_path).suffix.lower()
    try:
        if suffix == ".toml":
            parsed = tomllib.loads(loaded.text)
        elif suffix in {".json", ".ipynb"} or loaded.text.lstrip().startswith(("{", "[")):
            parsed = json.loads(loaded.text)
        else:
            try:
                import yaml  # type: ignore[import-not-found]

                parsed = yaml.safe_load(loaded.text)
            except ImportError:
                parsed = _yaml_subset(loaded.text)
            except Exception:
                # PyYAML's safe loader rejects unknown tags and malformed
                # documents with library-specific exception types.  The
                # indentation fallback is non-executing and lets analysis
                # continue conservatively in those cases.
                parsed = _yaml_subset(loaded.text)
        return parsed if parsed is not None else {}, None
    except (OSError, UnicodeError, TypeError, ValueError, SyntaxError) as exc:
        return {}, ReaderDiagnostic(loaded.display_path, str(exc), "structured")


# ---------------------------------------------------------------------------
# SQL and DuckDB text


def strip_sql_comments(source: str) -> str:
    """Return SQL with line/block comments replaced by whitespace.

    Quoted strings and quoted identifiers are preserved so DuckDB file scans
    such as ``read_parquet('data/x.parquet')`` remain discoverable.
    """

    output: list[str] = []
    index = 0
    quote: str | None = None
    while index < len(source):
        char = source[index]
        if quote is not None:
            output.append(char)
            if char == "\\" and quote == "'" and index + 1 < len(source):
                output.append(source[index + 1])
                index += 2
                continue
            if char == quote:
                if index + 1 < len(source) and source[index + 1] == quote:
                    output.append(source[index + 1])
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if source.startswith("--", index):
            end = source.find("\n", index + 2)
            if end < 0:
                output.extend(" " * (len(source) - index))
                break
            output.extend(" " * (end - index))
            output.append("\n")
            index = end + 1
            continue
        if source.startswith("/*", index):
            end = source.find("*/", index + 2)
            if end < 0:
                output.extend(" " * (len(source) - index))
                break
            output.extend(" " * (end + 2 - index))
            index = end + 2
            continue
        # DuckDB accepts # comments in a few CLI contexts.  Only recognize a
        # hash at the beginning of a line or after whitespace to avoid
        # corrupting quoted/unusual identifiers.
        if char == "#" and (index == 0 or source[index - 1].isspace()):
            end = source.find("\n", index + 1)
            if end < 0:
                output.extend(" " * (len(source) - index))
                break
            output.extend(" " * (end - index))
            output.append("\n")
            index = end + 1
            continue
        if char in "'\"`[":
            quote = "]" if char == "[" else char
        output.append(char)
        index += 1
    return "".join(output)


def _mask_sql_literals(source: str) -> str:
    """Replace quoted SQL literals/identifiers with spaces of equal length."""

    output = list(source)
    index = 0
    quote: str | None = None
    while index < len(source):
        char = source[index]
        if quote is not None:
            if char == quote:
                if index + 1 < len(source) and source[index + 1] == quote:
                    output[index] = " "
                    output[index + 1] = " "
                    index += 2
                    continue
                output[index] = " "
                quote = None
            else:
                output[index] = "\n" if char == "\n" else " "
            index += 1
            continue
        if char in "'\"`[":
            quote = "]" if char == "[" else char
            output[index] = " "
        index += 1
    return "".join(output)


def _sql_identifier(source: str, start: int) -> tuple[str, int, bool]:
    while start < len(source) and source[start].isspace():
        start += 1
    if start >= len(source):
        return "", start, False
    if source[start] in "'\"`[":
        opening = source[start]
        closing = "]" if opening == "[" else opening
        index = start + 1
        value: list[str] = []
        while index < len(source):
            if source[index] == closing:
                if index + 1 < len(source) and source[index + 1] == closing:
                    value.append(closing)
                    index += 2
                    continue
                return "".join(value), index + 1, opening == "'"
            value.append(source[index])
            index += 1
        return "".join(value), index, opening == "'"
    if source[start] in "(?{":
        return source[start], start + 1, False
    index = start
    while index < len(source) and source[index] not in " \t\r\n,;()":
        index += 1
    return source[start:index], index, False


def _sql_ctes(source: str) -> set[str]:
    return {
        _strip_matching_quotes(match.group(1)).lower()
        for match in re.finditer(
            r"(?:\bwith\b|,)\s*([A-Za-z_][\w$]*|\"[^\"]+\")\s+as\s*\(",
            source,
            flags=re.IGNORECASE,
        )
    }


def _sql_function_arguments(source: str, start: int) -> tuple[str | None, bool]:
    index = start
    while index < len(source) and source[index].isspace():
        index += 1
    if index >= len(source) or source[index] != "(":
        return None, True
    index += 1
    while index < len(source) and source[index].isspace():
        index += 1
    if index >= len(source):
        return None, True
    if source[index] in "'\"`":
        value, _, is_string = _sql_identifier(source, index)
        return value if is_string else None, not is_string
    return None, True


def _sql_table_node(
    graph: DependencyGraph,
    reference: str,
    *,
    quoted_string: bool,
    root: str | os.PathLike[str] | None,
) -> str | None:
    label = _reference_label(reference)
    if not label or _looks_dynamic(label):
        return None
    kind_metadata = {
        "source_type": "sql_file" if quoted_string and _is_path_reference(label) else "sql_table"
    }
    return _safe_reference_node(graph, label, kind="dataset", root=root, metadata=kind_metadata)


def _read_sql_text(
    source: str,
    graph: DependencyGraph,
    *,
    source_id: str,
    source_path: str,
    root: str | os.PathLike[str] | None = None,
    dialect: str = "sql",
) -> DependencyGraph:
    node = graph.get_node(source_id)
    if node is None:
        _root_node(
            graph,
            source_id,
            kind="sql",
            name=Path(source_path).name or source_path,
            path=source_path,
            reader="sql",
        )
    elif node.metadata.get("reader") is None:
        node.metadata.update({"reader": "sql", "parseable": True})
    cleaned = strip_sql_comments(source)
    code = _mask_sql_literals(cleaned)
    ctes = _sql_ctes(code)
    references: list[tuple[str, bool, str]] = []
    for match in re.finditer(r"\b(from|join|update|into|using)\b", code, re.IGNORECASE):
        reference, reference_end, quoted_string = _sql_identifier(cleaned, match.end())
        if not reference or reference in {"(", "select", "values"}:
            continue
        # Table-valued functions are handled below, where their string
        # arguments can be classified as DuckDB file scans.  Keeping the
        # function name out of the table set avoids a spurious dependency such
        # as ``dataset:read_parquet``.
        if not quoted_string and re.match(r"\s*\(", code[reference_end:]):
            continue
        bare = reference.split(".")[-1].strip('"`[]').lower()
        if not quoted_string and bare in ctes:
            continue
        references.append((reference, quoted_string, match.group(1).lower()))

    dynamic = False
    for reference, quoted_string, keyword in references:
        if reference.startswith(("(", "{", "?")) or _looks_dynamic(reference):
            dynamic = True
            continue
        target = _sql_table_node(
            graph,
            reference,
            quoted_string=quoted_string,
            root=root,
        )
        if target is not None:
            _edge(
                graph,
                source_id,
                target,
                "sql_reads",
                explanation=f"SQL {keyword} reference",
                dialect=dialect,
            )

    functions = re.finditer(
        r"\b(read_csv_auto|read_csv|read_parquet|parquet_scan|read_json_auto|read_json|"
        r"read_ndjson|read_text|glob)\s*\(",
        code,
        re.IGNORECASE,
    )
    for match in functions:
        value, is_dynamic = _sql_function_arguments(cleaned, match.end() - 1)
        if value is None or is_dynamic or _looks_dynamic(value):
            dynamic = True
            continue
        target = _safe_reference_node(
            graph,
            value,
            kind="dataset",
            root=root,
            metadata={"source_type": "duckdb_scan", "function": match.group(1).lower()},
        )
        if target is not None:
            _edge(
                graph,
                source_id,
                target,
                "sql_reads",
                explanation=f"DuckDB {match.group(1)} scan",
                dialect=dialect,
            )

    outputs: list[tuple[str, str]] = []
    output_patterns = (
        r"\bcreate\s+(?:or\s+replace\s+)?(?:temporary\s+|temp\s+)?"
        r"(?:table|view|materialized\s+view)\s+(?:if\s+not\s+exists\s+)?",
        r"\binsert\s+into\s+",
        r"\bcopy\s+.+?\s+to\s+",
    )
    for pattern in output_patterns:
        for match in re.finditer(pattern, code, re.IGNORECASE | re.DOTALL):
            value, end, quoted_string = _sql_identifier(cleaned, match.end())
            if not value or value.startswith(("(", "?", "{")):
                dynamic = True
                continue
            if (
                pattern.startswith(r"\bcopy")
                and not quoted_string
                and not _is_path_reference(value)
            ):
                # COPY ... TO table is valid in some dialects; retain it as a
                # dataset but do not pretend that it is a filesystem output.
                pass
            outputs.append((value, "sql_file" if quoted_string else "sql_table"))
            del end
    output_nodes: list[str] = []
    for value, output_type in outputs:
        target = _safe_reference_node(
            graph,
            value,
            kind="dataset",
            root=root,
            metadata={"source_type": output_type, "output": True},
        )
        if target is None:
            dynamic = True
            continue
        output_nodes.append(target)
        _edge(
            graph,
            target,
            source_id,
            "sql_produces",
            explanation="SQL statement produces or writes this dataset",
            dialect=dialect,
        )
    for output_id in sorted(set(output_nodes)):
        for input_id in sorted(
            {
                edge.target
                for edge in graph.outgoing_edges(source_id)
                if edge.kind == "sql_reads" and edge.target in graph.nodes
            }
        ):
            _edge(
                graph,
                output_id,
                input_id,
                "sql_derives",
                explanation="SQL output depends on a referenced input",
                dialect=dialect,
            )
    if dynamic:
        _unknown_dependency(
            graph,
            source_id,
            node_id="unknown:sql-dynamic",
            relation="sql_reads",
            explanation="SQL contains a dynamic or unsupported dependency expression",
        )
    if node is not None:
        node.metadata.update({"dialect": dialect, "parseable": True})
    return graph


def read_sql_dependencies(
    source: str | os.PathLike[str],
    graph: DependencyGraph | None = None,
    *,
    path: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
    dialect: str = "sql",
) -> DependencyGraph:
    """Read static table and DuckDB file-scan dependencies from SQL text."""

    graph = graph if graph is not None else DependencyGraph()
    loaded = _load_text(source, path=path, root=root, reader="sql")
    source_id = _text_node_id(loaded.display_path)
    if loaded.error is not None or loaded.text is None:
        _root_node(
            graph,
            source_id,
            kind="sql",
            name=Path(loaded.display_path).name or loaded.display_path,
            path=loaded.display_path,
            reader="sql",
            error=loaded.error or "unable to read SQL",
        )
        return graph
    return _read_sql_text(
        loaded.text,
        graph,
        source_id=source_id,
        source_path=loaded.display_path,
        root=root,
        dialect=dialect,
    )


# Friendly names used by callers that think of SQL as an analyzer rather than
# a file reader.
read_sql = read_sql_dependencies
analyze_sql = read_sql_dependencies
read_duckdb_dependencies = read_sql_dependencies
read_duckdb = read_sql_dependencies


class SQLReader:
    """Object-shaped adapter for SQL/DuckDB dependency extraction."""

    name = "sql"

    def read(
        self,
        source: str | os.PathLike[str],
        graph: DependencyGraph | None = None,
        *,
        path: str | os.PathLike[str] | None = None,
        root: str | os.PathLike[str] | None = None,
        dialect: str = "sql",
    ) -> DependencyGraph:
        return read_sql_dependencies(
            source,
            graph,
            path=path,
            root=root,
            dialect=dialect,
        )

    __call__ = read


# ---------------------------------------------------------------------------
# Shell text


def _shell_is_dynamic(value: str) -> bool:
    return bool(re.search(r"\$(?:\(|\{|[A-Za-z_])|`|\*|\?|\$\([^)]+\)", value))


def _shell_pathish(value: str) -> bool:
    value = value.strip()
    return (
        bool(value)
        and not value.startswith("-")
        and not _shell_is_dynamic(value)
        and (_is_path_reference(value) or value in {"Makefile", "Dockerfile"})
    )


def _shell_reference(
    graph: DependencyGraph,
    value: str,
    *,
    root: str | os.PathLike[str] | None,
    source_id: str,
    relation: str,
    output: bool = False,
    explanation: str,
) -> None:
    if not _shell_pathish(value):
        return
    suffix = Path(value).suffix.lower()
    kind = "file" if suffix in SCRIPT_SUFFIXES or suffix in STRUCTURED_SUFFIXES else "dataset"
    target = _safe_reference_node(
        graph,
        value,
        kind=kind,
        root=root,
        metadata={"source_type": "shell_path"},
    )
    if target is None:
        return
    if output:
        _edge(graph, target, source_id, relation, explanation=explanation)
    else:
        _edge(graph, source_id, target, relation, explanation=explanation)


def _shell_tokens(source: str) -> tuple[list[list[str]], bool]:
    try:
        lexer = shlex.shlex(source, posix=True, punctuation_chars=";&|<>")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        commands: list[list[str]] = [[]]
        for token in lexer:
            if token in {";", "&&", "||", "&", "|"}:
                if commands[-1]:
                    commands.append([])
            else:
                commands[-1].append(token)
        return [command for command in commands if command], True
    except (ValueError, shlex.Error):
        # An unmatched quote should not make impact analysis fail.  The
        # fallback sacrifices shell grammar but still finds explicit paths.
        rough = re.sub(r"(^|\s)#.*", r"\1", source)
        commands = [line.split() for line in rough.splitlines() if line.split()]
        return commands, False


def _read_shell_text(
    source: str,
    graph: DependencyGraph,
    *,
    source_id: str,
    source_path: str,
    root: str | os.PathLike[str] | None = None,
) -> DependencyGraph:
    node = graph.get_node(source_id)
    if node is None:
        node = _root_node(
            graph,
            source_id,
            kind="shell",
            name=Path(source_path).name or source_path,
            path=source_path,
            reader="shell",
        )
    commands, parseable = _shell_tokens(source)
    dynamic = False
    option_names = {
        "-i",
        "--input",
        "--inputs",
        "-o",
        "--output",
        "--outputs",
        "--config",
        "--file",
        "--deps",
        "--outs",
        "-f",
    }
    executable_names = {"python", "python3", "python3.10", "bash", "sh", "zsh", "rscript"}
    for tokens in commands:
        if not tokens:
            continue
        lowered = [Path(token).name.lower() for token in tokens]
        command = lowered[0]
        output_values: set[str] = set()
        if any(_shell_is_dynamic(token) for token in tokens):
            dynamic = True
        # Redirections are punctuation tokens with shlex's configured lexer.
        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token in {"<", ">", ">>", "<<<"} and index + 1 < len(tokens):
                if token.startswith(">"):
                    output_values.add(tokens[index + 1])
                _shell_reference(
                    graph,
                    tokens[index + 1],
                    root=root,
                    source_id=source_id,
                    relation="shell_writes" if token.startswith(">") else "shell_reads",
                    output=token.startswith(">"),
                    explanation="shell redirection path",
                )
                index += 2
                continue
            if token.endswith((">", ">>")) and len(token) > 1:
                _shell_reference(
                    graph,
                    token[1:],
                    root=root,
                    source_id=source_id,
                    relation="shell_writes",
                    output=True,
                    explanation="shell output redirection",
                )
            index += 1

        # Explicit option values are the safest shell dependencies to infer.
        index = 0
        while index < len(tokens):
            token = tokens[index]
            option = token.split("=", 1)[0]
            if option in option_names:
                if "=" in token:
                    value = token.split("=", 1)[1]
                elif index + 1 < len(tokens):
                    index += 1
                    value = tokens[index]
                else:
                    value = ""
                if option in {"-o", "--output", "--outputs"}:
                    output_values.add(value)
                _shell_reference(
                    graph,
                    value,
                    root=root,
                    source_id=source_id,
                    relation="shell_writes"
                    if option in {"-o", "--output", "--outputs"}
                    else "shell_reads",
                    output=option in {"-o", "--output", "--outputs"},
                    explanation="explicit shell input/output option",
                )
            index += 1

        if command in {"source", "."} and len(tokens) > 1:
            _shell_reference(
                graph,
                tokens[1],
                root=root,
                source_id=source_id,
                relation="shell_executes",
                explanation="shell sources another script",
            )
        if command in executable_names:
            for token in tokens[1:]:
                if not token.startswith("-") and _shell_pathish(token):
                    _shell_reference(
                        graph,
                        token,
                        root=root,
                        source_id=source_id,
                        relation="shell_executes",
                        explanation="shell invokes an explicit script",
                    )
                    break
        if command == "dvc" and len(tokens) > 1 and lowered[1] in {"repro", "exp"}:
            for stage in tokens[2:]:
                if stage.startswith("-") or _shell_is_dynamic(stage):
                    continue
                target = f"pipeline:{stage}"
                graph.add_node(Node(target, "pipeline_stage", stage, None, {"source_type": "dvc"}))
                _edge(
                    graph,
                    source_id,
                    target,
                    "shell_invokes",
                    explanation="shell invokes a DVC pipeline stage",
                )
        # Known data/script extensions elsewhere in a command are explicit
        # enough to retain.  Environment variables, flags, and bare words are
        # intentionally ignored.
        for token in tokens[1:]:
            if token in output_values:
                continue
            if _shell_pathish(token):
                _shell_reference(
                    graph,
                    token,
                    root=root,
                    source_id=source_id,
                    relation="shell_reads",
                    explanation="explicit path argument in shell command",
                )
            elif token.startswith(("$", "`")):
                dynamic = True
    node.metadata.update({"parseable": parseable, "dynamic": dynamic})
    if dynamic:
        _unknown_dependency(
            graph,
            source_id,
            node_id="unknown:shell-dynamic",
            relation="shell_reads",
            explanation="shell command contains a dynamic path or command expression",
        )
    return graph


def read_shell_dependencies(
    source: str | os.PathLike[str],
    graph: DependencyGraph | None = None,
    *,
    path: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
) -> DependencyGraph:
    """Read explicit shell scripts and command-line file dependencies."""

    graph = graph if graph is not None else DependencyGraph()
    loaded = _load_text(source, path=path, root=root, reader="shell")
    source_id = _text_node_id(loaded.display_path)
    if loaded.error is not None or loaded.text is None:
        _root_node(
            graph,
            source_id,
            kind="shell",
            name=Path(loaded.display_path).name or loaded.display_path,
            path=loaded.display_path,
            reader="shell",
            error=loaded.error or "unable to read shell input",
        )
        return graph
    return _read_shell_text(
        loaded.text,
        graph,
        source_id=source_id,
        source_path=loaded.display_path,
        root=root,
    )


read_shell = read_shell_dependencies
analyze_shell = read_shell_dependencies


class ShellReader:
    """Object-shaped adapter for shell dependency extraction."""

    name = "shell"

    def read(
        self,
        source: str | os.PathLike[str],
        graph: DependencyGraph | None = None,
        *,
        path: str | os.PathLike[str] | None = None,
        root: str | os.PathLike[str] | None = None,
    ) -> DependencyGraph:
        return read_shell_dependencies(source, graph, path=path, root=root)

    __call__ = read


# ---------------------------------------------------------------------------
# Notebook cells


def _call_name(node: ast.Call) -> str:
    value: ast.AST = node.func
    parts: list[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _constant_string(value: ast.AST | None) -> str | None:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _notebook_file_call(
    graph: DependencyGraph,
    cell_id: str,
    call: ast.Call,
    *,
    root: str | os.PathLike[str] | None,
) -> tuple[bool, bool]:
    name = _call_name(call).lower()
    leaf = name.rsplit(".", 1)[-1]
    input_calls = {
        "open",
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_jsonl",
        "read_ndjson",
        "read_table",
        "read_pickle",
        "load",
        "load_workbook",
        "read_sql",
        "read_sql_query",
        "read_sql_table",
    }
    output_calls = {
        "to_csv",
        "to_parquet",
        "to_json",
        "to_pickle",
        "to_sql",
        "dump",
        "save",
        "write_text",
    }
    if leaf not in input_calls | output_calls:
        return False, False
    value = _constant_string(call.args[0] if call.args else None)
    if value is None or _looks_dynamic(value):
        return True, True
    target = _safe_reference_node(
        graph,
        value,
        kind="dataset",
        root=root,
        metadata={"source_type": "notebook_call", "function": name},
    )
    if target is None:
        return True, True
    is_output = leaf in output_calls
    _edge(
        graph,
        target if is_output else cell_id,
        cell_id if is_output else target,
        "notebook_writes" if is_output else "notebook_reads",
        explanation=f"notebook calls {name}",
    )
    return True, False


def _read_notebook_python(
    source: str,
    graph: DependencyGraph,
    *,
    cell_id: str,
    root: str | os.PathLike[str] | None,
) -> bool:
    code_lines = [
        line for line in source.splitlines() if not line.lstrip().startswith(("%", "!", "%%"))
    ]
    has_magic = any(line.strip().startswith(("%", "!")) for line in source.splitlines())
    try:
        tree = ast.parse("\n".join(code_lines), mode="exec")
    except (SyntaxError, ValueError, TypeError):
        # SQL and shell magics are handled by the sibling magic reader.  A
        # cell containing only such a magic is not a Python syntax error.
        return has_magic and not any(line.strip() for line in code_lines)
    dynamic = False
    for item in ast.walk(tree):
        if isinstance(item, ast.Import):
            for alias in item.names:
                target = f"module:{alias.name}"
                graph.add_node(
                    Node(target, "module", alias.name, None, {"source_type": "notebook_import"})
                )
                _edge(
                    graph, cell_id, target, "imports", explanation=f"notebook imports {alias.name}"
                )
        elif isinstance(item, ast.ImportFrom):
            module = "." * item.level + (item.module or "")
            target = f"module:{module or '<package>'}"
            graph.add_node(
                Node(
                    target,
                    "module",
                    module or "<package>",
                    None,
                    {"source_type": "notebook_import"},
                )
            )
            _edge(
                graph,
                cell_id,
                target,
                "imports",
                explanation=f"notebook imports from {module or '<package>'}",
            )
        elif isinstance(item, ast.Call):
            handled, call_dynamic = _notebook_file_call(graph, cell_id, item, root=root)
            dynamic = dynamic or call_dynamic
            call_name = _call_name(item)
            if call_name in {"__import__", "importlib.import_module"}:
                module = _constant_string(item.args[0] if item.args else None)
                if module and not _looks_dynamic(module):
                    target = f"module:{module}"
                    graph.add_node(Node(target, "module", module, None, {"dynamic": True}))
                    _edge(
                        graph,
                        cell_id,
                        target,
                        "imports",
                        confidence="low",
                        explanation="notebook uses a dynamic import with a static name",
                    )
                else:
                    dynamic = True
            if call_name.rsplit(".", 1)[-1].lower() == "sql" and item.args:
                query = _constant_string(item.args[0])
                if query:
                    _read_sql_text(
                        query,
                        graph,
                        source_id=cell_id,
                        source_path=cell_id,
                        root=root,
                        dialect="duckdb",
                    )
    if dynamic:
        _unknown_dependency(
            graph,
            cell_id,
            node_id="unknown:notebook-dynamic",
            relation="notebook_reads",
            explanation="notebook contains a dynamic file or import expression",
        )
    return not dynamic


def _read_notebook_magic(
    source: str,
    graph: DependencyGraph,
    *,
    cell_id: str,
    root: str | os.PathLike[str] | None,
) -> bool:
    lines = source.splitlines()
    dynamic = False
    sql_body: list[str] = []
    if lines and lines[0].strip().lower().startswith("%%sql"):
        sql_body = lines[1:]
    elif lines and lines[0].strip().lower().startswith("%sql"):
        sql_body = [lines[0].strip()[4:].lstrip()]
    if sql_body:
        _read_sql_text(
            "\n".join(sql_body),
            graph,
            source_id=cell_id,
            source_path=cell_id,
            root=root,
            dialect="duckdb",
        )
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(("%run", "%load")):
            parts = stripped.split(maxsplit=1)
            value = parts[1].split()[0] if len(parts) > 1 and parts[1].split() else ""
            if value.startswith(("-", "--")) and len(parts) > 1:
                remainder = parts[1].split()
                value = remainder[-1] if remainder else ""
            if _shell_is_dynamic(value) or not value:
                dynamic = True
            else:
                _shell_reference(
                    graph,
                    value,
                    root=root,
                    source_id=cell_id,
                    relation="notebook_executes"
                    if stripped.startswith("%run")
                    else "notebook_reads",
                    explanation="notebook file magic",
                )
        elif stripped.startswith("!"):
            _read_shell_text(
                stripped[1:],
                graph,
                source_id=cell_id,
                source_path=cell_id,
                root=root,
            )
    if dynamic:
        _unknown_dependency(
            graph,
            cell_id,
            node_id="unknown:notebook-dynamic",
            relation="notebook_reads",
            explanation="notebook magic contains a dynamic path",
        )
    return not dynamic


def read_notebook_dependencies(
    source: str | os.PathLike[str],
    graph: DependencyGraph | None = None,
    *,
    path: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
) -> DependencyGraph:
    """Read imports, file calls, SQL cells, and explicit notebook magics."""

    graph = graph if graph is not None else DependencyGraph()
    loaded = _load_text(source, path=path, root=root, reader="notebook")
    source_id = _text_node_id(loaded.display_path)
    if loaded.error is not None or loaded.text is None:
        _root_node(
            graph,
            source_id,
            kind="notebook",
            name=Path(loaded.display_path).name or loaded.display_path,
            path=loaded.display_path,
            reader="notebook",
            error=loaded.error or "unable to read notebook",
        )
        return graph
    root_node = _root_node(
        graph,
        source_id,
        kind="notebook",
        name=Path(loaded.display_path).name or loaded.display_path,
        path=loaded.display_path,
        reader="notebook",
    )
    try:
        parsed = json.loads(loaded.text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        root_node.metadata.update({"parseable": False, "parse_error": str(exc)})
        return graph
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("cells"), list):
        root_node.metadata.update({"parseable": False, "parse_error": "not a notebook cell list"})
        return graph
    for index, cell in enumerate(parsed["cells"]):
        if not isinstance(cell, Mapping):
            continue
        cell_id = f"{source_id}#cell-{index}"
        cell_source = cell.get("source", "")
        if isinstance(cell_source, list):
            text = "".join(str(part) for part in cell_source if isinstance(part, (str, int, float)))
        elif isinstance(cell_source, str):
            text = cell_source
        else:
            text = ""
        cell_node = graph.add_node(
            Node(
                cell_id,
                "notebook_cell",
                f"cell-{index}",
                loaded.display_path,
                {"index": index, "cell_type": cell.get("cell_type", "code")},
            )
        )
        _edge(graph, source_id, cell_id, "contains", explanation="notebook contains cell")
        if cell_node.metadata.get("cell_type") == "code":
            parseable = _read_notebook_python(text, graph, cell_id=cell_id, root=root)
            magic_parseable = _read_notebook_magic(text, graph, cell_id=cell_id, root=root)
            cell_node.metadata["parseable"] = parseable and magic_parseable
    return graph


read_notebook = read_notebook_dependencies
analyze_notebook = read_notebook_dependencies


class NotebookReader:
    """Object-shaped adapter for callers that prefer reader instances."""

    name = "notebook"

    def read(
        self,
        source: str | os.PathLike[str],
        graph: DependencyGraph | None = None,
        *,
        path: str | os.PathLike[str] | None = None,
        root: str | os.PathLike[str] | None = None,
    ) -> DependencyGraph:
        return read_notebook_dependencies(source, graph, path=path, root=root)

    __call__ = read


# ---------------------------------------------------------------------------
# Pipeline/config declarations


def _entry_name(value: Any, fallback: str) -> str:
    if isinstance(value, Mapping):
        for key in ("name", "id", "key", "stage", "job"):
            if value.get(key) is not None:
                return str(value[key])
    return str(value) if isinstance(value, (str, int, float)) else fallback


def _entry_values(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in ("path", "uri", "file", "name", "id", "value"):
            if value.get(key) is not None:
                return [value[key]]
        return []
    if isinstance(value, (list, tuple, set)):
        values: list[Any] = []
        for item in value:
            values.extend(_entry_values(item) if isinstance(item, Mapping) else [item])
        return values
    return [value]


def _pipeline_units(data: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    candidates: list[tuple[str, Any]] = []
    for key in ("stages", "steps", "jobs", "tasks", "nodes", "pipelines"):
        value = data.get(key)
        if isinstance(value, Mapping):
            candidates.extend((str(name), item) for name, item in value.items())
        elif isinstance(value, list):
            candidates.extend(
                (_entry_name(item, f"{key}-{index}"), item) for index, item in enumerate(value)
            )
    result: list[tuple[str, Mapping[str, Any]]] = []
    for name, value in candidates:
        if isinstance(value, Mapping):
            result.append((name, value))
    return sorted({name: value for name, value in result}.items(), key=lambda item: item[0])


def _config_relationships(data: Mapping[str, Any], graph: DependencyGraph) -> None:
    """Reuse the existing config vocabulary when the input is not a pipeline."""

    try:
        from .config import load_relationships

        load_relationships(data, graph)
    except (ImportError, TypeError, ValueError, KeyError):
        # Reader inputs are untrusted.  The direct pipeline reader below still
        # handles the well-known step fields if the compatibility loader is
        # unavailable or sees an unusual mapping.
        return


def _pipeline_reference(
    graph: DependencyGraph,
    value: Any,
    *,
    source_id: str,
    root: str | os.PathLike[str] | None,
    output: bool,
    field: str,
    units: set[str],
) -> None:
    label = _reference_label(value)
    if not label or _looks_dynamic(label):
        _unknown_dependency(
            graph,
            source_id,
            node_id="unknown:pipeline-dynamic",
            relation="pipeline_reads",
            explanation=f"pipeline field {field} contains a dynamic reference",
        )
        return
    if label in units:
        target = f"pipeline:{label}"
    elif label.partition(":")[0] in {"dataset", "feature", "model", "file", "module", "pipeline"}:
        target = label
        graph.add_node(Node(target, target.partition(":")[0], label.partition(":")[2], None))
    else:
        suffix = Path(label).suffix.lower()
        kind = "file" if suffix in SCRIPT_SUFFIXES or suffix in STRUCTURED_SUFFIXES else "dataset"
        target = (
            _safe_reference_node(
                graph,
                label,
                kind=kind,
                root=root,
                metadata={"source_type": "pipeline_field", "field": field},
            )
            or ""
        )
    if not target:
        return
    _edge(
        graph,
        target if output else source_id,
        source_id if output else target,
        "pipeline_writes" if output else "pipeline_reads",
        explanation=f"pipeline {field} reference",
        field=field,
    )


def _read_pipeline_mapping(
    data: Mapping[str, Any],
    graph: DependencyGraph,
    *,
    source_id: str,
    source_path: str,
    root: str | os.PathLike[str] | None = None,
) -> DependencyGraph:
    _config_relationships(data, graph)
    units = _pipeline_units(data)
    unit_names = {name for name, _ in units}
    for name, metadata in units:
        stage_id = f"pipeline:{name}"
        stage = graph.add_node(
            Node(
                stage_id,
                "pipeline_stage",
                name,
                source_path,
                {"source_type": "pipeline_config", **dict(metadata)},
            )
        )
        _edge(graph, source_id, stage_id, "contains", explanation="pipeline declares a stage")
        for field in ("depends_on", "needs", "dependencies"):
            for value in _entry_values(metadata.get(field)):
                _pipeline_reference(
                    graph,
                    value,
                    source_id=stage_id,
                    root=root,
                    output=False,
                    field=field,
                    units=unit_names,
                )
        for field in (
            "inputs",
            "input",
            "deps",
            "sources",
            "params",
            "data",
            "datasets",
            "features",
        ):
            for value in _entry_values(metadata.get(field)):
                _pipeline_reference(
                    graph,
                    value,
                    source_id=stage_id,
                    root=root,
                    output=False,
                    field=field,
                    units=unit_names,
                )
        for field in ("outputs", "output", "outs", "artifacts", "products", "targets"):
            for value in _entry_values(metadata.get(field)):
                _pipeline_reference(
                    graph,
                    value,
                    source_id=stage_id,
                    root=root,
                    output=True,
                    field=field,
                    units=unit_names,
                )
        for field in ("script", "entrypoint", "implemented_by"):
            for value in _entry_values(metadata.get(field)):
                _pipeline_reference(
                    graph,
                    value,
                    source_id=stage_id,
                    root=root,
                    output=False,
                    field=field,
                    units=unit_names,
                )
        command = metadata.get("cmd") or metadata.get("command") or metadata.get("run")
        if isinstance(command, str) and command.strip():
            _read_shell_text(
                command,
                graph,
                source_id=stage_id,
                source_path=source_path,
                root=root,
            )
        elif command is not None:
            _unknown_dependency(
                graph,
                stage_id,
                node_id="unknown:pipeline-command",
                relation="pipeline_reads",
                explanation="pipeline command is not a static string",
            )
        stage.metadata.setdefault("parseable", True)
    return graph


def read_pipeline_dependencies(
    source: str | os.PathLike[str] | Mapping[str, Any],
    graph: DependencyGraph | None = None,
    *,
    path: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
) -> DependencyGraph:
    """Read DVC-shaped and common step/stage pipeline declarations."""

    graph = graph if graph is not None else DependencyGraph()
    if isinstance(source, Mapping):
        data, diagnostic = source, None
        display_path = _display_path(path)
    else:
        data, diagnostic = load_structured(source, path=path, root=root)
        display_path = (
            diagnostic.path
            if diagnostic is not None
            else _load_text(source, path=path, root=root).display_path
        )
    source_id = _text_node_id(display_path)
    error = diagnostic.error if diagnostic is not None else None
    root_node = _root_node(
        graph,
        source_id,
        kind="pipeline",
        name=Path(display_path).name or display_path,
        path=display_path,
        reader="pipeline",
        error=error,
    )
    if not isinstance(data, Mapping):
        root_node.metadata.update(
            {"parseable": False, "parse_error": "structured input is not a mapping"}
        )
        return graph
    _read_pipeline_mapping(data, graph, source_id=source_id, source_path=display_path, root=root)
    return graph


read_pipeline = read_pipeline_dependencies
read_config_dependencies = read_pipeline_dependencies
read_configuration = read_pipeline_dependencies
analyze_pipeline = read_pipeline_dependencies


class PipelineReader:
    """Object-shaped adapter for pipeline/configuration dependencies."""

    name = "pipeline"

    def read(
        self,
        source: str | os.PathLike[str] | Mapping[str, Any],
        graph: DependencyGraph | None = None,
        *,
        path: str | os.PathLike[str] | None = None,
        root: str | os.PathLike[str] | None = None,
    ) -> DependencyGraph:
        return read_pipeline_dependencies(source, graph, path=path, root=root)

    __call__ = read


SQLDependencyReader = SQLReader
NotebookDependencyReader = NotebookReader
ShellDependencyReader = ShellReader
PipelineDependencyReader = PipelineReader


# ---------------------------------------------------------------------------
# Dispatch helpers


def reader_name(path: str | os.PathLike[str]) -> str:
    text = str(path)
    upper = text.lstrip().upper()
    if upper.startswith(("SELECT ", "WITH ", "INSERT ", "UPDATE ", "DELETE ", "CREATE ", "COPY ")):
        return "sql"
    if text.lstrip().startswith("#!") and any(
        value in text.splitlines()[0].lower() for value in ("/sh", "/bash", "/zsh")
    ):
        return "shell"
    name = Path(path).name.lower()
    suffix = Path(name).suffix.lower()
    if name in {"dvc.yaml", "dvc.yml", "dvc.lock"}:
        return "dvc"
    if suffix in SQL_SUFFIXES:
        return "sql"
    if suffix in NOTEBOOK_SUFFIXES:
        return "notebook"
    if suffix in SHELL_SUFFIXES:
        return "shell"
    if suffix in STRUCTURED_SUFFIXES:
        return "pipeline"
    return "text"


def read_dependency_file(
    path: str | os.PathLike[str],
    graph: DependencyGraph | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
    format: str | None = None,
) -> DependencyGraph:
    """Dispatch one file to a dependency reader using its suffix/name."""

    selected = (format or reader_name(path)).lower()
    if selected == "dvc":
        from .lineage import import_dvc_graph

        return import_dvc_graph(path, graph, root=root)
    if selected in {"sql", "duckdb"}:
        return read_sql_dependencies(
            path, graph, root=root, dialect="duckdb" if selected == "duckdb" else "sql"
        )
    if selected in {"notebook", "ipynb"}:
        return read_notebook_dependencies(path, graph, root=root)
    if selected in {"shell", "bash", "sh"}:
        return read_shell_dependencies(path, graph, root=root)
    return read_pipeline_dependencies(path, graph, root=root)


def read_dependencies(
    source: str | os.PathLike[str] | Mapping[str, Any],
    graph: DependencyGraph | None = None,
    *,
    path: str | os.PathLike[str] | None = None,
    root: str | os.PathLike[str] | None = None,
    format: str | None = None,
) -> DependencyGraph:
    """Read one dependency source, selecting a conservative built-in reader."""

    if isinstance(source, Mapping):
        return read_pipeline_dependencies(source, graph, path=path, root=root)
    selected = (format or reader_name(source)).lower()
    if selected == "text":
        selected = reader_name(path or "") if path is not None else "pipeline"
    if selected in {"sql", "duckdb"}:
        return read_sql_dependencies(source, graph, path=path, root=root, dialect=selected)
    if selected == "dvc":
        from .lineage import import_dvc_graph

        return import_dvc_graph(source, graph, path=path, root=root)
    if selected in {"notebook", "ipynb"}:
        return read_notebook_dependencies(source, graph, path=path, root=root)
    if selected in {"shell", "bash", "sh"}:
        return read_shell_dependencies(source, graph, path=path, root=root)
    return read_pipeline_dependencies(source, graph, path=path, root=root)


analyze_dependencies = read_dependencies
read_dependency_source = read_dependencies


__all__ = [
    "DATA_SUFFIXES",
    "NOTEBOOK_SUFFIXES",
    "ReaderDiagnostic",
    "ReaderResult",
    "NotebookDependencyReader",
    "NotebookReader",
    "PipelineDependencyReader",
    "PipelineReader",
    "SCRIPT_SUFFIXES",
    "SHELL_SUFFIXES",
    "ShellDependencyReader",
    "ShellReader",
    "SQL_SUFFIXES",
    "SQLDependencyReader",
    "SQLReader",
    "analyze_dependencies",
    "analyze_notebook",
    "analyze_pipeline",
    "analyze_shell",
    "analyze_sql",
    "load_structured",
    "read_config_dependencies",
    "read_configuration",
    "read_dependencies",
    "read_dependency_file",
    "read_dependency_source",
    "read_duckdb",
    "read_duckdb_dependencies",
    "read_notebook",
    "read_notebook_dependencies",
    "read_pipeline",
    "read_pipeline_dependencies",
    "read_shell",
    "read_shell_dependencies",
    "read_sql",
    "read_sql_dependencies",
    "reader_name",
    "strip_sql_comments",
]
