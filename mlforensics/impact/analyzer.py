"""Conservative Python AST analysis with graceful syntax-error handling."""

from __future__ import annotations

import ast
import os
from pathlib import Path

from .graph import DependencyGraph, Node


def _module_id(path: str) -> str:
    return "module:" + path.replace(os.sep, "/")


class _DefinitionCollector(ast.NodeVisitor):
    def __init__(self, module_id: str, path: str, graph: DependencyGraph) -> None:
        self.module_id, self.path, self.graph = module_id, path, graph
        self.scope: list[str] = []
        self.definitions: dict[str, str] = {}
        self.class_methods: dict[str, dict[str, str]] = {}

    def _visit_definition(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
    ) -> None:
        qualname = ".".join(self.scope + [node.name]) if self.scope else node.name
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        identifier = f"{self.module_id}:{qualname}"
        self.graph.add_node(
            Node(
                identifier,
                kind,
                node.name,
                self.path,
                {"qualname": qualname, "line": getattr(node, "lineno", None)},
            )
        )
        self.graph.nodes[identifier].metadata["end_line"] = getattr(
            node, "end_lineno", getattr(node, "lineno", None)
        )
        parent = f"{self.module_id}:{'.'.join(self.scope)}" if self.scope else self.module_id
        self.graph.add_edge(
            parent,
            identifier,
            "contains",
            {
                "line": getattr(node, "lineno", None),
                "confidence": "high",
                "explanation": "definition contained by its module or class",
            },
        )
        if len(self.scope) == 0:
            self.definitions[node.name] = identifier
        if self.scope and self.scope[0] in self.class_methods and kind == "function":
            self.class_methods[self.scope[0]][node.name] = identifier
        if kind == "class":
            self.class_methods.setdefault(qualname, {})
        self.scope.append(node.name)
        for child in node.body:
            self.visit(child)
        self.scope.pop()

    visit_FunctionDef = _visit_definition
    visit_AsyncFunctionDef = _visit_definition
    visit_ClassDef = _visit_definition


class _DependencyVisitor(ast.NodeVisitor):
    def __init__(
        self,
        analyzer: PythonStaticAnalyzer,
        module_id: str,
        path: str,
        graph: DependencyGraph,
        definitions: _DefinitionCollector,
    ) -> None:
        self.analyzer, self.module_id, self.path, self.graph = analyzer, module_id, path, graph
        self.definitions = definitions
        self.scope: list[str] = []
        self.aliases: dict[str, str] = {}
        self.class_stack: list[str] = []

    def source_id(self) -> str:
        return f"{self.module_id}:{'.'.join(self.scope)}" if self.scope else self.module_id

    def _import_target(self, module: str) -> str:
        return self.analyzer.module_target(module)

    def _relative_import_target(self, module: str | None, level: int) -> str:
        return self.analyzer.resolve_import(module or "", level=level, current=self.module_id)

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            imported = item.name
            target = self._import_target(imported)
            self.graph.add_node(
                Node(target, "module", imported, None, {"external": target == _module_id(imported)})
            )
            self.graph.add_edge(
                self.source_id(),
                target,
                "imports",
                {
                    "line": node.lineno,
                    "confidence": "high"
                    if target in self.analyzer.known_modules.values()
                    else "medium",
                    "explanation": f"import {item.name}",
                },
            )
            bound_name = item.asname or item.name.split(".")[0]
            self.aliases[item.name] = target
            self.aliases[bound_name] = target

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = "." * node.level + (node.module or "")
        resolved_module = self._relative_import_target(node.module, node.level)
        resolved_name = self.analyzer.resolve_import_name(
            node.module or "", level=node.level, current=self.module_id
        )
        for item in node.names:
            if item.name == "*":
                target = resolved_module
            else:
                submodule = f"{resolved_name}.{item.name}" if resolved_name else item.name
                target = (
                    self.analyzer.module_target(submodule)
                    if submodule in self.analyzer.known_modules
                    else f"{resolved_module}:{item.name}"
                )
            self.graph.add_node(
                Node(
                    target,
                    "import",
                    item.name,
                    None,
                    {
                        "module": module,
                        "resolved_module": resolved_module,
                        "external": resolved_module not in self.analyzer.known_modules.values(),
                    },
                )
            )
            self.graph.add_edge(
                self.source_id(),
                target,
                "imports",
                {
                    "line": node.lineno,
                    "confidence": "high"
                    if resolved_module in self.analyzer.known_modules.values()
                    else "medium",
                    "explanation": f"from {module or '<package>'} import {item.name}",
                },
            )
            self.aliases[item.asname or item.name] = target

    def _visit_scoped(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        identifier = f"{self.module_id}:{'.'.join([*self.scope, node.name])}"
        for decorator in getattr(node, "decorator_list", ()):
            target = self.resolve_expr(decorator)
            if target:
                self.graph.add_edge(
                    identifier,
                    target,
                    "decorates",
                    {
                        "line": getattr(decorator, "lineno", getattr(node, "lineno", None)),
                        "confidence": "high" if target in self.graph.nodes else "medium",
                        "explanation": "decorator reference",
                    },
                )
        self.scope.append(node.name)
        if isinstance(node, ast.ClassDef):
            self.class_stack.append(".".join(self.scope))
            for base in node.bases:
                target = self.resolve_expr(base)
                if target is None and isinstance(base, ast.Name):
                    # A base may be defined in an unavailable module.  The
                    # reference is still useful, while remaining explicitly
                    # marked as unresolved metadata.
                    target = f"{self.module_id}:{base.id}"
                    self.graph.add_node(
                        Node(target, "class", base.id, self.path, {"unresolved": True})
                    )
                if target:
                    self.graph.add_edge(
                        self.source_id(),
                        target,
                        "inherits",
                        {
                            "line": node.lineno,
                            "confidence": "high" if target in self.graph.nodes else "low",
                            "explanation": "class base reference",
                        },
                    )
        for child in node.body:
            self.visit(child)
        if isinstance(node, ast.ClassDef):
            self.class_stack.pop()
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        # Keep dynamic imports as explicit, conservative dependencies.  A
        # string literal is the only form whose target can be named without
        # pretending to understand arbitrary runtime code.
        dynamic_module: str | None = None
        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
            if (
                node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                dynamic_module = node.args[0].value
        elif (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            dynamic_module = node.args[0].value
        if dynamic_module:
            target = self._import_target(dynamic_module)
            self.graph.add_node(Node(target, "module", dynamic_module, None))
            self.graph.add_edge(
                self.source_id(),
                target,
                "imports",
                {
                    "line": node.lineno,
                    "confidence": "medium"
                    if target in self.analyzer.known_modules.values()
                    else "low",
                    "explanation": "dynamic import with a static module name",
                },
            )
        elif (isinstance(node.func, ast.Name) and node.func.id == "__import__") or (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "importlib"
            and node.func.attr == "import_module"
        ):
            target = "unknown:dynamic-import"
            self.graph.add_node(
                Node(target, "unknown", "dynamic import", self.path, {"conservative": True})
            )
            self.graph.add_edge(
                self.source_id(),
                target,
                "imports",
                {
                    "line": node.lineno,
                    "confidence": "low",
                    "explanation": "dynamic import target is not statically known",
                },
            )
        target = self.resolve_expr(node.func)
        if target and target != self.source_id():
            known_target = target in self.graph.nodes
            if target not in self.graph.nodes:
                self.graph.add_node(
                    Node(target, "callable", target.rsplit(":", 1)[-1], None, {"external": True})
                )
            self.graph.add_edge(
                self.source_id(),
                target,
                "calls",
                {
                    "line": node.lineno,
                    "confidence": "high" if known_target else "low",
                    "explanation": "statically resolved call",
                },
            )
        self.generic_visit(node)

    visit_FunctionDef = _visit_scoped
    visit_AsyncFunctionDef = _visit_scoped

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Class bases and method bodies are dependencies of the class definition;
        # calls inside methods are separately attributed to each method.
        self._visit_scoped(node)

    def resolve_expr(self, expression: ast.AST) -> str | None:
        def dotted_name(value: ast.AST) -> str | None:
            if isinstance(value, ast.Name):
                return value.id
            if isinstance(value, ast.Attribute):
                prefix = dotted_name(value.value)
                return f"{prefix}.{value.attr}" if prefix else value.attr
            return None

        if isinstance(expression, ast.Name):
            if expression.id in self.aliases:
                return self.aliases[expression.id]
            if expression.id in self.definitions.definitions:
                return self.definitions.definitions[expression.id]
            if self.class_stack:
                methods = self.definitions.class_methods.get(self.class_stack[-1], {})
                if expression.id in methods:
                    return methods[expression.id]
        if isinstance(expression, ast.Attribute):
            dotted = dotted_name(expression)
            if dotted in self.aliases:
                return self.aliases[dotted]
            if (
                isinstance(expression.value, ast.Name)
                and expression.value.id == "self"
                and self.class_stack
            ):
                methods = self.definitions.class_methods.get(self.class_stack[-1], {})
                if expression.attr in methods:
                    return methods[expression.attr]
            base = self.resolve_expr(expression.value)
            if base:
                dotted = base + "." + expression.attr
                if dotted in self.graph.nodes:
                    return dotted
                candidate = base + ":" + expression.attr
                if candidate in self.graph.nodes or base.startswith("module:"):
                    return candidate
        if isinstance(expression, ast.Call):
            return self.resolve_expr(expression.func)
        return None


class PythonStaticAnalyzer:
    """Analyze imports, definitions, inheritance, and conservatively known calls."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root).resolve() if root else None
        self.graph = DependencyGraph()
        self.known_modules: dict[str, str] = {}

    def module_target(self, module: str) -> str:
        clean = module.lstrip(".")
        if clean in self.known_modules:
            return self.known_modules[clean]
        return _module_id(clean or "unknown")

    def resolve_import(self, module: str, *, level: int = 0, current: str | None = None) -> str:
        """Resolve an absolute or package-relative import to a local module."""
        clean = self.resolve_import_name(module, level=level, current=current)
        if clean in self.known_modules:
            return self.known_modules[clean]
        # ``from . import name`` can resolve to a submodule even though the
        # AST node does not carry that name in ``module``.
        return _module_id(clean or "unknown")

    def resolve_import_name(
        self, module: str, *, level: int = 0, current: str | None = None
    ) -> str:
        """Return the dotted name for an import in the current package."""
        clean = module.strip(".")
        if level:
            current_name = ""
            if current and current.startswith("module:"):
                current_path = current.removeprefix("module:")
                current_name = (
                    current_path[:-3].replace("/", ".")
                    if current_path.endswith(".py")
                    else current_path.replace("/", ".")
                )
                if current_name.endswith(".__init__"):
                    current_name = current_name[:-9]
            package_parts = current_name.split(".")[:-1]
            if level > 1:
                package_parts = package_parts[: max(0, len(package_parts) - level + 1)]
            clean = ".".join(part for part in (*package_parts, clean) if part)
        return clean

    def analyze_source(self, source: str, path: str = "<string>") -> DependencyGraph:
        module = _module_id(path)
        self.graph.add_node(Node(module, "module", Path(path).stem, path))
        try:
            tree = ast.parse(source, filename=path)
        except (SyntaxError, ValueError, TypeError) as exc:
            self.graph.nodes[module].metadata.update({"syntax_error": str(exc), "parseable": False})
            return self.graph
        self.graph.nodes[module].metadata["parseable"] = True
        collector = _DefinitionCollector(module, path, self.graph)
        collector.visit(tree)
        visitor = _DependencyVisitor(self, module, path, self.graph, collector)
        visitor.visit(tree)
        return self.graph

    def analyze_file(self, path: str | os.PathLike[str]) -> DependencyGraph:
        path_obj = Path(path)
        try:
            source = path_obj.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            module = _module_id(str(path_obj))
            self.graph.add_node(
                Node(module, "module", path_obj.stem, str(path_obj), {"read_error": str(exc)})
            )
            return self.graph
        return self.analyze_source(source, str(path_obj))

    def analyze_tree(self, root: str | os.PathLike[str] | None = None) -> DependencyGraph:
        root_path = Path(root or self.root or ".").resolve()
        self.root = root_path
        files = sorted(p for p in root_path.rglob("*.py") if p.is_file())
        for path in files:
            relative = path.relative_to(root_path).as_posix()
            module_name = relative[:-3].replace("/", ".")
            if module_name.endswith(".__init__"):
                module_name = module_name[:-9]
            self.known_modules[module_name] = _module_id(relative)
        for path in files:
            relative = path.relative_to(root_path).as_posix()
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                module = _module_id(relative)
                self.graph.add_node(
                    Node(module, "module", path.stem, relative, {"read_error": str(exc)})
                )
                continue
            self.analyze_source(source, relative)
        return self.graph

    analyze_directory = analyze_tree

    def analyze(self, source_or_path: str, path: str | None = None) -> DependencyGraph:
        if path is not None or "\n" in source_or_path:
            return self.analyze_source(source_or_path, path or "<string>")
        return self.analyze_file(source_or_path)


StaticAnalyzer = PythonStaticAnalyzer
PythonAnalyzer = PythonStaticAnalyzer


def analyze_python(source: str, path: str = "<string>") -> DependencyGraph:
    return PythonStaticAnalyzer().analyze_source(source, path)


def analyze_tree(root: str | os.PathLike[str]) -> DependencyGraph:
    return PythonStaticAnalyzer(root).analyze_tree()
