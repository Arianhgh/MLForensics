import json
from pathlib import Path

from mlforensics.impact.lineage import import_dvc_graph
from mlforensics.impact.python import PythonAnalyzer
from mlforensics.impact.readers import (
    read_notebook_dependencies,
    read_pipeline_dependencies,
    read_shell_dependencies,
    read_sql_dependencies,
)


def test_sql_reader_ignores_comments_and_keeps_duckdb_scans():
    graph = read_sql_dependencies(
        "-- FROM commented_out\n"
        "/* JOIN also_commented */\n"
        "SELECT * FROM raw JOIN read_parquet('data/events.parquet') ON true"
    )

    assert "dataset:raw" in graph.nodes
    assert "dataset:data/events.parquet" in graph.nodes
    assert "dataset:commented_out" not in graph.nodes
    assert "dataset:also_commented" not in graph.nodes
    assert "dataset:read_parquet" not in graph.nodes


def test_readers_are_tolerant_of_unknown_syntax_and_dynamic_paths():
    graph = read_sql_dependencies("SELECT * FROM {table_name /* unfinished")
    assert graph.get_node("file:<string>").metadata["parseable"] is True
    assert "unknown:sql-dynamic" in graph.nodes
    assert any(edge.metadata["confidence"] == "low" for edge in graph.edges.values())

    shell = read_shell_dependencies("python train.py --input '$DATA_FILE' && echo [")
    assert "file:train.py" in shell.nodes
    assert "unknown:shell-dynamic" in shell.nodes


def test_notebook_invalid_json_is_a_nonfatal_diagnostic(tmp_path: Path):
    notebook = tmp_path / "broken.ipynb"
    notebook.write_text("{not valid json", encoding="utf-8")

    graph = read_notebook_dependencies(notebook, root=tmp_path)
    node = graph.get_node(f"file:{notebook.resolve().as_posix()}")
    assert node is not None
    assert node.metadata["parseable"] is False
    assert "parse_error" in node.metadata


def test_pipeline_does_not_follow_parent_paths_and_preserves_the_boundary(tmp_path: Path):
    graph = read_pipeline_dependencies(
        {"steps": {"train": {"inputs": ["../secret.csv"], "outputs": ["model.pkl"]}}},
        root=tmp_path,
    )

    outside = graph.get_node("dataset:../secret.csv")
    assert outside is not None
    assert outside.metadata["outside_root"] is True
    assert not any(node.kind == "secret" for node in graph)


def test_python_reader_skips_paths_outside_root(tmp_path: Path):
    outside = tmp_path.parent / "outside-mlforensics.py"
    outside.write_text("value = 1\n", encoding="utf-8")
    try:
        analyzer = PythonAnalyzer(tmp_path)
        graph = analyzer.analyze([outside])
        assert graph.nodes == {}
        assert analyzer.errors[0]["error"] == "path is outside the analysis root"
    finally:
        outside.unlink()


def test_notebook_reads_code_and_sql_magic(tmp_path: Path):
    notebook = tmp_path / "features.ipynb"
    notebook.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "source": ["import pandas as pd\n", "pd.read_csv('raw.csv')\n"],
                    },
                    {"cell_type": "code", "source": ["%%sql\n", "SELECT * FROM clean\n"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    graph = read_notebook_dependencies(notebook, root=tmp_path)

    assert "module:pandas" in graph.nodes
    assert "dataset:raw.csv" in graph.nodes
    assert "dataset:clean" in graph.nodes


def test_dvc_duplicate_paths_produce_one_relationship():
    graph = import_dvc_graph(
        {
            "stages": {
                "train": {
                    "deps": ["data.csv", "data.csv"],
                    "outs": [{"path": "model.bin"}, {"path": "model.bin"}],
                }
            }
        }
    )

    assert len([edge for edge in graph.edges.values() if edge.kind in {"dvc_dep", "dvc_out"}]) == 2
