import json
from pathlib import Path

from mlforensics.impact.graph import DependencyGraph
from mlforensics.impact.lineage import (
    export_dvc_graph,
    export_openlineage_events,
    import_openlineage_events,
)
from mlforensics.impact.plugins import (
    DVCDependencyPlugin,
    PluginRegistry,
    SQLDependencyPlugin,
    default_registry,
)


def test_openlineage_export_is_repeatable_and_round_trips_graph_edges():
    graph = DependencyGraph()
    graph.add_edge("model:m", "dataset:raw", "configured")

    first = export_openlineage_events(graph, ["model:m"])
    second = export_openlineage_events(graph, ["model:m"])
    assert first == second

    imported = import_openlineage_events(first)
    assert "model:m" in imported.nodes
    assert "dataset:raw" in imported.nodes
    assert any(
        edge.source == "model:m" and edge.target == "dataset:raw"
        for edge in imported.edges.values()
    )


def test_openlineage_import_skips_malformed_events_and_deduplicates_relationships():
    event = {
        "eventType": "COMPLETE",
        "run": {"runId": "run-1"},
        "job": {"namespace": "demo", "name": "train"},
        "inputs": [{"namespace": "demo", "name": "raw"}],
        "outputs": [{"namespace": "demo", "name": "model"}],
    }
    graph = import_openlineage_events([event, event, {"job": None}, "bad"])

    assert len(graph.edges) == 2
    assert {edge.kind for edge in graph.edges.values()} == {
        "openlineage_input",
        "openlineage_output",
    }


def test_openlineage_invalid_jsonl_is_safe(tmp_path: Path):
    path = tmp_path / "events.jsonl"
    path.write_text("not json\n" + json.dumps({"job": {"name": "incomplete"}}), encoding="utf-8")

    graph = import_openlineage_events(path, root=tmp_path)
    assert set(graph.nodes) == {"job:mlforensics:incomplete"}


def test_dvc_export_is_sorted_and_registry_dispatches_without_optional_packages(tmp_path: Path):
    graph = DependencyGraph()
    graph.add_edge("pipeline:train", "dataset:z.csv", "dvc_dep")
    graph.add_edge("dataset:model.bin", "pipeline:train", "dvc_out")
    graph.nodes["pipeline:train"].kind = "pipeline_stage"
    graph.nodes["pipeline:train"].name = "train"

    assert export_dvc_graph(graph) == {
        "stages": {"train": {"deps": ["z.csv"], "outs": ["model.bin"]}}
    }

    registry = PluginRegistry([SQLDependencyPlugin(), DVCDependencyPlugin()])
    sql_graph = registry.read("SELECT * FROM raw", format="sql")
    dvc_graph = registry.read({"stages": {"train": {"deps": ["raw"]}}}, format="dvc")
    assert "dataset:raw" in sql_graph.nodes
    assert "pipeline:train" in dvc_graph.nodes
    assert DVCDependencyPlugin().name == "dvc"
    assert default_registry().get("sql") is not None
