"""Git diff parsing without requiring GitPython."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .graph import DependencyGraph


@dataclass
class DiffChange:
    path: str
    status: str = "M"
    old_path: str | None = None
    added_lines: set[int] = field(default_factory=set)
    removed_lines: set[int] = field(default_factory=set)

    @property
    def is_deleted(self) -> bool:
        return self.status.upper().startswith("D")

    @property
    def line_count(self) -> int:
        return len(self.added_lines) + len(self.removed_lines)


_HEADER = re.compile(r"^diff --git a/(.+) b/(.+)$")
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_git_diff(diff_text: str | None) -> list[DiffChange]:
    """Parse unified or ``--name-status`` Git output, tolerating partial diffs."""
    if not diff_text:
        return []
    changes: list[DiffChange] = []
    by_path: dict[str, DiffChange] = {}
    current: DiffChange | None = None
    old_line = new_line = 0
    in_hunk = False
    for raw in diff_text.splitlines():
        line = raw.rstrip("\n")
        header = _HEADER.match(line)
        if header:
            path = header.group(2)
            current = DiffChange(path=path, status="M", old_path=header.group(1))
            changes.append(current)
            by_path[path] = current
            old_line = new_line = 0
            in_hunk = False
            continue
        # Handles output such as A\tsrc/new.py and R100\told.py\tnew.py.
        if "\t" in line and re.match(r"^[A-Z][0-9]*\t", line):
            fields = line.split("\t")
            status = fields[0]
            if status.startswith("R") or status.startswith("C"):
                if len(fields) >= 3:
                    current = DiffChange(fields[2], status, fields[1])
                else:
                    continue
            else:
                current = DiffChange(fields[1] if len(fields) > 1 else "", status)
            if current.path and current.path not in by_path:
                changes.append(current)
                by_path[current.path] = current
            elif current.path:
                existing = by_path[current.path]
                existing.status = current.status
                existing.old_path = current.old_path
                current = existing
            continue
        hunk = _HUNK.match(line)
        if hunk:
            old_line = int(hunk.group(1))
            new_line = int(hunk.group(3))
            in_hunk = True
            continue
        if current and in_hunk:
            if line.startswith("+") and not line.startswith("+++"):
                current.added_lines.add(new_line)
                new_line += 1
            elif line.startswith("-") and not line.startswith("---"):
                current.removed_lines.add(old_line)
                old_line += 1
            elif not line.startswith("\\"):
                old_line += 1
                new_line += 1
    return changes


class GitDiffImpactExtractor:
    def __init__(self, repository: str | Path | None = None) -> None:
        self.repository = str(repository or ".")

    def from_text(self, diff_text: str | None) -> list[DiffChange]:
        return parse_git_diff(diff_text)

    parse = from_text

    def extract(self, base: str = "HEAD", target: str | None = None) -> list[DiffChange]:
        command = ["git", "-C", self.repository, "diff", base]
        if target:
            command.append(target)
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError:
            return []
        return parse_git_diff(completed.stdout if completed.returncode == 0 else "")

    @staticmethod
    def changed_node_ids(changes: Iterable[DiffChange], graph: DependencyGraph) -> set[str]:
        result: set[str] = set()
        paths: set[str] = set()
        for change in changes:
            change_paths = {change.path.replace("\\", "/").lstrip("./")}
            if change.old_path:
                change_paths.add(change.old_path.replace("\\", "/").lstrip("./"))
            paths.update(change_paths)
            matching = [
                node
                for node in graph
                if node.path
                and any(
                    node.path.replace("\\", "/").lstrip("./") == path
                    or node.path.replace("\\", "/").endswith("/" + path)
                    for path in change_paths
                )
            ]
            changed_lines = change.added_lines | change.removed_lines
            changed_symbols = []
            if changed_lines:
                for node in matching:
                    if node.kind not in {"function", "class"}:
                        continue
                    start = node.metadata.get("line")
                    end = node.metadata.get("end_line", start)
                    if (
                        isinstance(start, int)
                        and isinstance(end, int)
                        and any(start <= line <= end for line in changed_lines)
                    ):
                        changed_symbols.append(node)
            if changed_symbols:
                result.update(node.id for node in changed_symbols)
            else:
                result.update(node.id for node in matching if node.kind in {"file", "module"})
        # Preserve a useful root even when the source was deleted or the graph
        # was built from a different checkout.
        for path in paths:
            if not any(
                graph.get_node(node_id)
                and graph.get_node(node_id).path
                and graph.get_node(node_id).path.replace("\\", "/") == path
                for node_id in result
            ):
                module_id = "module:" + path
                if module_id in graph.nodes:
                    result.add(module_id)
        return result

    changed_nodes = changed_node_ids


def extract_git_diff_impact(diff_text: str, graph: DependencyGraph | None = None):
    changes = parse_git_diff(diff_text)
    if graph is None:
        return changes
    return {
        "changes": changes,
        "changed_node_ids": GitDiffImpactExtractor.changed_node_ids(changes, graph),
    }


GitDiffParser = GitDiffImpactExtractor
