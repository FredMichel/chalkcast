"""Authoring-time validation of plot expressions.

Plot expressions are agent-written Python that Manim evaluates at render time.
Validating them up front turns a slow render crash (``NameError: sin``) into an
immediate, actionable authoring error, and the AST whitelist keeps the check
itself safe: only arithmetic, whitelisted math functions, ``np.<name>`` and the
declared variables (``x`` plus scene parameters) are accepted, so evaluating a
sample point can never reach the filesystem, imports or dunder attributes.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Iterable

import numpy as np

# Bare names usable in expressions (mirrors the numpy import in codegen's header).
MATH_NAMES: dict[str, Any] = {
    name: getattr(np, name)
    for name in (
        "sin", "cos", "tan", "exp", "sqrt", "log", "log10", "log2", "sinh", "cosh",
        "tanh", "arcsin", "arccos", "arctan", "arctan2", "floor", "ceil", "abs",
        "sign", "heaviside", "where", "minimum", "maximum", "clip", "pi", "e",
    )
}
MATH_NAMES["abs"] = np.abs

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Call, ast.Name, ast.Load, ast.Constant, ast.Attribute, ast.Tuple, ast.List,
    ast.GeneratorExp, ast.ListComp, ast.comprehension, ast.Store, ast.keyword,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub,
    ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt,
    ast.GtE,
)
# builtins allowed in generator-style sums, e.g. sum(sin(k*x)/k for k in range(1, 9, 2))
_SAFE_BUILTINS: dict[str, Any] = {"sum": sum, "range": range, "abs": abs, "min": min,
                                  "max": max, "float": float, "int": int}


_SYNTAX_HINTS = {
    ast.BitAnd: " (for 'a and b' on a condition write np.logical_and(a, b), or "
                "np.where(cond, value, 0) for a piecewise function)",
    ast.BitOr: " (for 'a or b' on a condition write np.logical_or(a, b))",
    ast.BitXor: " (for a power write x**2; ^ is not a power in Python)",
    ast.Invert: " (for 'not' on a condition write np.logical_not(a))",
}


class ExprError(ValueError):
    """An expression that would fail (or is unsafe) at render time."""


def validate(expr: str, params: Iterable[str] = ()) -> None:
    """Raise ExprError unless ``expr`` is a safe, evaluable function of ``x``.

    ``params`` are scene parameter names (ValueTrackers) the expression may use.
    """
    params = set(params)
    if not isinstance(expr, str) or not expr.strip():
        raise ExprError("expression is empty")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"syntax error in {expr!r}: {exc.msg}") from None

    bound: set[str] = set()  # comprehension variables, e.g. k in sum(... for k in ...)
    for node in ast.walk(tree):
        if isinstance(node, ast.comprehension):
            for t in ast.walk(node.target):
                if isinstance(t, ast.Name):
                    bound.add(t.id)

    allowed = {"x", "np"} | params | set(MATH_NAMES) | set(_SAFE_BUILTINS) | bound
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExprError(f"unsupported syntax in {expr!r}: {type(node).__name__}"
                            f"{_SYNTAX_HINTS.get(type(node), '')}")
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == "np"):
                raise ExprError(f"only np.<function> attributes are allowed in {expr!r}")
            if node.attr.startswith("_") or not hasattr(np, node.attr):
                raise ExprError(f"np.{node.attr} is not a numpy function")
        if isinstance(node, ast.Name) and node.id not in allowed:
            import difflib
            close = difflib.get_close_matches(node.id, list(params), n=1)
            if node.id.lower() in ("nan", "inf", "infinity"):
                hint = f" (write np.{node.id.lower()[:3]})"
            elif close:
                hint = f" (did you mean the scene parameter '{close[0]}'?)"
            else:
                hint = f" (declared parameters: {sorted(params)})" if params else ""
            raise ExprError(f"unknown name '{node.id}' in {expr!r}; use x, math functions "
                            f"like sin/exp/sqrt, np.<fn>, or a scene parameter{hint}")

    # Evaluate at a few points: catches arity/type errors before a render does.
    fn = compile_fn(expr, params)
    sample_params = {p: 0.5 for p in params}
    ok_points = 0
    for xv in (-1.7, -0.3, 0.4, 1.1, 2.9):
        try:
            with np.errstate(all="ignore"):
                y = fn(xv, **sample_params)
            float(np.asarray(y, dtype=float).reshape(-1)[0])
            ok_points += 1
        except (ZeroDivisionError, ValueError, OverflowError, FloatingPointError):
            continue  # a pole or domain edge at one sample is fine
        except Exception as exc:  # noqa: BLE001 - surface any evaluation failure
            raise ExprError(f"{expr!r} fails to evaluate: {type(exc).__name__}: {exc}") from None
    if ok_points == 0:
        raise ExprError(f"{expr!r} could not be evaluated at any sample point")


def free_names(expr: str) -> set[str]:
    """Names an expression reads from outside (not the variables of its own
    comprehensions: in ``sum(sin(k*x) for k in ...)`` k is not a parameter)."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set()
    out: set[str] = set()

    def visit(node: ast.AST, bound: frozenset[str]) -> None:
        if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
            inner = set(bound)
            for gen in node.generators:
                visit(gen.iter, frozenset(inner))
                inner |= {t.id for t in ast.walk(gen.target) if isinstance(t, ast.Name)}
                for cond in gen.ifs:
                    visit(cond, frozenset(inner))
            visit(node.elt, frozenset(inner))
            return
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in bound:
            out.add(node.id)
        for child in ast.iter_child_nodes(node):
            visit(child, bound)

    visit(tree, frozenset())
    return out


def compile_fn(expr: str, params: Iterable[str] = ()):
    """A python callable f(x, **params) for an already-validated expression."""
    code = compile(ast.parse(expr, mode="eval"), "<expr>", "eval")
    env = {"__builtins__": _SAFE_BUILTINS, "np": np, **MATH_NAMES}

    def f(x: float, **pvals: float) -> Any:
        # x/params go in globals so generator expressions (their own scope) see them
        return eval(code, {**env, "x": x, **pvals})  # noqa: S307 - AST-whitelisted above

    return f


def is_finite_number(v: Any) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False
