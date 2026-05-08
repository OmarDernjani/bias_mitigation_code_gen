from __future__ import annotations

import ast
from collections import defaultdict
from typing import Any


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_value(e) for e in node.elts]
    if isinstance(node, ast.Name):
        return node.id
    try:
        return ast.unparse(node)
    except Exception:
        return ast.dump(node)


class _IfConditionVisitor(ast.NodeVisitor):
    """Collect comparators on Name LHS in any Compare node (including those
    nested inside BoolOp like `age >= 30 and age <= 50`)."""

    def __init__(self) -> None:
        self.left_names: dict[str, list[Any]] = defaultdict(list)

    def visit_Compare(self, node: ast.Compare) -> None:
        if isinstance(node.left, ast.Name):
            name = _norm(node.left.id)
            for cmp in node.comparators:
                self.left_names[name].append(_value(cmp))
        self.generic_visit(node)


def refine(features: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Lowercase strings, dedup preserving order. List values dedup'd elementwise."""
    out: dict[str, list[Any]] = {}
    for key, values in features.items():
        lowered: list[Any] = []
        for v in values:
            if isinstance(v, str):
                lowered.append(v.lower())
            elif isinstance(v, list):
                lowered.append([x.lower() if isinstance(x, str) else x for x in v])
            else:
                lowered.append(v)
        seen: list[Any] = []
        for v in lowered:
            if v not in seen:
                seen.append(v)
        out[key] = seen
    return out


def extract_features(code: str) -> dict[str, list[Any]]:
    """{feature_name: [observed comparator values]} from if-conditions.
    Empty dict on parse error."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    visitor = _IfConditionVisitor()
    visitor.visit(tree)
    return refine(dict(visitor.left_names))


def function_signature(code: str) -> tuple[str, list[str]] | None:
    """(function_name, [param_names]) of the first FunctionDef. None if the
    code does not parse or contains no function definition."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node.name, [a.arg for a in node.args.args]
    return None
