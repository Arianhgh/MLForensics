from pathlib import Path

from mlforensics.impact import DependencyGraph, Node, PythonStaticAnalyzer, analyze_tree


def test_graph_tracks_downstream_consumers_and_is_deterministic():
    graph = DependencyGraph()
    graph.add_edge("model:m", "feature:f", "configured")
    graph.add_edge("feature:f", "dataset:d", "configured")
    graph.add_node(Node("dataset:d", "dataset", "d"))

    assert graph.downstream(["dataset:d"]) == {"dataset:d", "feature:f", "model:m"}
    assert [node.id for node in graph.predecessors("dataset:d")] == ["feature:f"]
    assert graph.as_dict()["edges"][0]["source"] == "feature:f"


def test_analyzer_finds_imports_definitions_inheritance_and_known_calls(tmp_path: Path):
    (tmp_path / "source.py").write_text(
        "from helpers import clean as tidy\n"
        "class Child(Base):\n"
        "    def run(self):\n"
        "        return tidy(1)\n"
        "def build():\n"
        "    return Child().run()\n",
        encoding="utf-8",
    )
    graph = PythonStaticAnalyzer(tmp_path).analyze_tree()
    ids = set(graph.nodes)

    assert "module:source.py" in ids
    assert "module:source.py:Child" in ids
    assert "module:source.py:Child.run" in ids
    assert "module:source.py:build" in ids
    assert any(edge.kind == "imports" and "helpers" in edge.target for edge in graph.edges.values())
    assert any(
        edge.kind == "inherits" and edge.target.endswith(":Base") for edge in graph.edges.values()
    )
    assert any(
        edge.kind == "calls" and edge.target.endswith(":clean") for edge in graph.edges.values()
    )
    assert any(
        edge.kind == "calls" and edge.target.endswith(".run") for edge in graph.edges.values()
    )


def test_analyzer_keeps_syntax_errors_as_metadata_and_continues_tree(tmp_path: Path):
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "good.py").write_text("def okay():\n    return 1\n", encoding="utf-8")

    graph = analyze_tree(tmp_path)
    assert graph.get_node("module:broken.py").metadata["parseable"] is False
    assert graph.get_node("module:good.py:okay").kind == "function"
