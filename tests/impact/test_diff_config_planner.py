from mlforensics.impact import (
    DependencyGraph,
    ImpactPlanner,
    analyze_impact,
    build_configured_graph,
    export_openlineage_events,
    parse_git_diff,
)


def test_parse_unified_diff_and_name_status_are_tolerant():
    unified = """diff --git a/src/a.py b/src/a.py
--- a/src/a.py
+++ b/src/a.py
@@ -1,2 +1,3 @@
 old()
-removed()
+added()
+also_added()
"""
    changes = parse_git_diff(unified)
    assert changes[0].path == "src/a.py"
    assert changes[0].added_lines == {2, 3}
    assert changes[0].removed_lines == {2}

    renamed = parse_git_diff("R100\told.py\tnew.py\nA\tadded.py\n")
    assert [(item.status, item.path, item.old_path) for item in renamed] == [
        ("R100", "new.py", "old.py"),
        ("A", "added.py", None),
    ]


def test_configured_relationships_flow_dataset_to_feature_to_model():
    graph = build_configured_graph(
        {
            "datasets": {"raw": {"produces": ["clean"]}},
            "features": {"clean": {"inputs": ["raw"]}},
            "models": {"fraud": {"features": ["clean"]}},
        }
    )
    assert graph.downstream(["dataset:raw"]) == {"dataset:raw", "feature:clean", "model:fraud"}
    assert graph.get_node("dataset:raw").metadata["produces"] == ["clean"]


def test_missing_configuration_metadata_is_safe():
    graph = build_configured_graph({"datasets": [None, {"name": "raw"}], "models": None})
    assert set(graph.nodes) == {"dataset:raw"}
    assert build_configured_graph(None).nodes == {}


def test_planner_accepts_changed_paths_and_recommends_validation():
    graph = build_configured_graph(
        {
            "datasets": {"raw": {"path": "data/raw.csv", "produces": ["clean"]}},
            "features": {"clean": {"inputs": ["raw"]}},
            "models": {"fraud": {"features": ["clean"]}},
        }
    )
    # Config nodes do not require source paths; a synthetic module demonstrates
    # the same path-based resolution used for static-analysis nodes.
    graph.add_node("module:feature.py", "module", path="feature.py")
    graph.add_edge("feature:clean", "module:feature.py", "implemented_by")
    plan = ImpactPlanner(graph).plan("feature.py")

    assert plan.changed_nodes == ["module:feature.py"]
    assert plan.affected_nodes == ["module:feature.py", "feature:clean", "model:fraud"]
    assert {item.check for item in plan.recommendations} == {
        "feature_recompute_and_distribution",
        "model_regression_and_slices",
        "import_and_integration_smoke",
    }


def test_lineage_export_is_plain_dicts_and_missing_nodes_are_ignored():
    graph = DependencyGraph()
    graph.add_edge("model:m", "feature:f", "configured")
    events = export_openlineage_events(graph, ["model:m", "missing"], run_id="run-1")

    assert len(events) == 1
    assert isinstance(events[0], dict)
    assert events[0]["run"]["runId"] == "run-1"
    assert events[0]["job"]["name"] == "model:m"
    assert events[0]["inputs"][0]["name"] == "feature:f"


def test_diff_lines_select_changed_symbol_and_configured_model(tmp_path):
    source = tmp_path / "features.py"
    source.write_text(
        "def untouched():\n    return 1\n\ndef changed():\n    return 2\n",
        encoding="utf-8",
    )
    change = parse_git_diff(
        "diff --git a/features.py b/features.py\n"
        "--- a/features.py\n"
        "+++ b/features.py\n"
        "@@ -5,1 +5,1 @@\n"
        "-    return 1\n"
        "+    return 2\n"
    )
    report = analyze_impact(
        tmp_path,
        change,
        mappings={"models": {"classifier": {"path": "features.py"}}},
    )

    assert any(node.endswith(":changed") for node in report.changed_nodes)
    assert not any(node.endswith(":untouched") for node in report.changed_nodes)
    assert "classifier" in report.models
