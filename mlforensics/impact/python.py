"""Conservative Python import and symbol analysis using the standard AST."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .graph import DependencyGraph


class _FileVisitor(ast.NodeVisitor):
    def __init__(
        self,
        graph: DependencyGraph,
        file_path: Path,
        module: str,
        imported_modules: Mapping[str, str] | None = None,
    ) -> None:
        self.graph = graph
        self.file_path = file_path
        self.module = module
        self.scope: list[str] = []
        self.symbols: dict[str, str] = {}
        self.imported_modules = dict(imported_modules or {})

    @property
    def file_node(self) -> str:
        return f"file:{self.file_path.as_posix()}"

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        symbol = f"{self.module}:{'.'.join([*self.scope, node.name])}"
        self.graph.add_node(
            symbol,
            kind="function",
            name=node.name,
            path=str(self.file_path),
            line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
        )
        self.graph.add_edge(self.file_node, symbol, "contains")
        self.symbols[node.name] = symbol
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        symbol = f"{self.module}:{'.'.join([*self.scope, node.name])}"
        self.graph.add_node(
            symbol,
            kind="class",
            name=node.name,
            path=str(self.file_path),
            line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
        )
        self.graph.add_edge(self.file_node, symbol, "contains")
        self.symbols[node.name] = symbol
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        def dotted(value: ast.AST) -> str | None:
            if isinstance(value, ast.Name):
                return value.id
            if isinstance(value, ast.Attribute):
                prefix = dotted(value.value)
                return f"{prefix}.{value.attr}" if prefix else value.attr
            return None

        name = dotted(node.func)
        target = self.symbols.get(name or "")
        if target is None and name:
            for alias, imported_module in sorted(
                self.imported_modules.items(), key=lambda item: len(item[1]), reverse=True
            ):
                if name == alias:
                    target = imported_module
                    break
                prefix = alias + "."
                if name.startswith(prefix):
                    suffix = name[len(prefix) :]
                    module_suffix = (
                        imported_module[len(alias) + 1 :]
                        if imported_module.startswith(prefix)
                        else ""
                    )
                    if module_suffix and suffix.startswith(module_suffix + "."):
                        suffix = suffix[len(module_suffix) + 1 :]
                    target = f"{imported_module}:{suffix}"
                    break
        if target:
            caller = f"{self.module}:{'.'.join(self.scope)}" if self.scope else self.file_node
            self.graph.add_edge(
                caller,
                target,
                "calls",
                confidence="high" if target in self.graph.nodes else "medium",
                explanation="statically resolved call through imported or local symbol",
            )
        self.generic_visit(node)


def module_name(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or root.name


class PythonAnalyzer:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.errors: list[dict[str, Any]] = []

    def analyze(self, paths: Iterable[str | Path] | None = None) -> DependencyGraph:
        graph = DependencyGraph()
        files = (
            [Path(path).resolve() for path in paths]
            if paths is not None
            else sorted(self.root.rglob("*.py"))
        )
        module_paths = {module_name(path, self.root): path for path in files if path.exists()}
        for path in files:
            if not path.is_file():
                continue
            module = module_name(path, self.root)
            file_node = f"file:{path.as_posix()}"
            graph.add_node(file_node, kind="file", name=module, path=str(path))
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except (OSError, SyntaxError) as exc:
                self.errors.append({"path": str(path), "error": str(exc)})
                continue
            # Keep imported aliases in the same symbol table as local
            # definitions.  This lets a call such as ``clean(...)`` after
            # ``from .features import clean`` point at the imported symbol.
            imported_symbols: dict[str, str] = {}
            imported_modules: dict[str, str] = {}
            for node in ast.walk(tree):
                imported: str | None = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = alias.name
                        target = module_paths.get(imported) or module_paths.get(
                            imported.split(".")[0]
                        )
                        if target:
                            graph.add_edge(
                                file_node,
                                f"file:{target.as_posix()}",
                                "imports",
                                module=imported,
                                confidence="high",
                                explanation=f"absolute import {imported}",
                            )
                        bound_name = alias.asname or alias.name.split(".")[0]
                        imported_modules[bound_name] = imported
                elif isinstance(node, ast.ImportFrom):
                    imported = node.module or ""
                    if node.level:
                        current_parts = module.split(".")[:-1]
                        if node.level > 1:
                            current_parts = current_parts[
                                : max(0, len(current_parts) - node.level + 1)
                            ]
                        imported = ".".join((*current_parts, imported))
                    target = module_paths.get(imported) or module_paths.get(imported.split(".")[0])
                    if target:
                        graph.add_edge(
                            file_node,
                            f"file:{target.as_posix()}",
                            "imports",
                            module=imported,
                            confidence="high" if node.level else "medium",
                            explanation=(
                                f"relative import level {node.level}"
                                if node.level
                                else f"absolute from-import {imported}"
                            ),
                        )
                        # A from-import is a symbol dependency as well as a
                        # module dependency.  Emit the symbol edge when the
                        # target module is local; its node is merged with the
                        # real definition when that module is visited.
                        for alias in node.names:
                            if alias.name == "*":
                                continue
                            symbol_id = f"{imported}:{alias.name}"
                            imported_symbols[alias.asname or alias.name] = symbol_id
                            if imported in module_paths:
                                graph.add_node(
                                    symbol_id,
                                    kind="imported_symbol",
                                    name=alias.name,
                                    path=str(target),
                                )
                                graph.add_edge(
                                    file_node,
                                    symbol_id,
                                    "imports",
                                    module=imported,
                                    symbol=alias.name,
                                    confidence="high",
                                    explanation=f"from-import symbol {imported}.{alias.name}",
                                )
                    else:
                        graph.add_edge(
                            file_node,
                            f"unknown:module:{imported or '<package>'}",
                            "imports",
                            module=imported,
                            confidence="low",
                            explanation="unresolved import retained conservatively",
                        )
                        for alias in node.names:
                            if alias.name != "*":
                                imported_symbols[alias.asname or alias.name] = (
                                    f"{imported}:{alias.name}"
                                )
            visitor = _FileVisitor(graph, path, module, imported_modules)
            visitor.symbols.update(imported_symbols)
            visitor.visit(tree)
        return graph


def analyze_python(
    root: str | Path, paths: Iterable[str | Path] | None = None
) -> tuple[DependencyGraph, list[dict[str, Any]]]:
    analyzer = PythonAnalyzer(root)
    graph = analyzer.analyze(paths)
    return graph, analyzer.errors
