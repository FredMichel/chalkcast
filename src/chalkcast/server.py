"""Chalkcast MCP server.

Exposes the roadmap tools, grouped by capability with ``verb_object`` names,
each returning the common evidence-bearing envelope (SPEC §5).

Design rules the tools follow:

* **fail fast, at authoring time** -- expressions, colours, ranges, step targets
  and parameters are validated when added, so mistakes surface in the reply to
  the call that made them, not as a render crash later;
* **no silent defaults** -- e.g. ``add_plot`` without functions draws bare axes;
* **measure before you render** -- ``inspect_render`` builds the real Manim
  objects (no video) and measures them; ``render_scene`` does the same first and
  skips the slow render when the scene cannot compile;
* **edit, don't rebuild** -- ``update_object`` / ``remove_object`` /
  ``remove_step`` / ``fix_scene`` change a scene in place.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:  # mcp >= 2.0: FastMCP was renamed to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ModuleNotFoundError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import render as rnd
from . import session
from .codegen import _KNOWN_COLORS
from .envelope import err, ok
from .expr import ExprError, validate
from .fix import apply_fixes
from .model import STEP_KINDS, Obj, Project, Scene
from .verify import obj_params, overlay_error, GEOMETRIC_LANES, beat_conflicts, inspect_scene, is_dynamic, unused_tweens

mcp = _Server("chalkcast")

SHAPES = ("circle", "square", "rectangle", "triangle", "arrow", "dot", "line")
_HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED = {"x", "np", "P", "ns", "self", "e", "pi"}
_DEFAULT_FPS = {"low": 15, "medium": 30, "high": 60}
# builtins usable in expressions; a parameter must not shadow them
_SAFE_NAMES = {"sum", "range", "abs", "min", "max", "float", "int"}
_STEP_ARGS: dict[str, set[str]] = {
    "write": {"targets"}, "highlight": {"targets"}, "fade": {"targets"},
    "transform": {"targets", "to"}, "camera_move": {"zoom", "focus_x", "focus_y"},
    "wait": set(), "tween": {"param", "value"},
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _require_scene(project_id: str | None, scene_id: str | None) -> tuple[Project, Scene] | dict[str, Any]:
    project = session.get_project(project_id)
    if project is None:
        return err("no open project; call open_project first")
    scene = project.get_scene(scene_id)
    if scene is None:
        return err("no such scene; call add_scene first")
    return project, scene


_measured_at: dict[str, str] = {}  # "project/scene" -> fingerprint of the last inspection


def _register_obligations(project: Project, scene_id: str, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace this scene's open obligations with a freshly measured set."""
    sc = project.get_scene(scene_id)
    if sc is not None:
        _measured_at[f"{project.id}/{scene_id}"] = _fingerprint(sc)
    project.obligations = [
        o for o in project.obligations if not (o["scene"] == scene_id and o["status"] == "open")
    ]
    registered = []
    for o in raw:
        o = {**o, "id": project.next_obligation_id()}
        project.obligations.append(o)
        registered.append(o)
    return registered


def _color_error(color: str | None) -> str | None:
    if color is None or _HEX.match(color) or color.upper() in _KNOWN_COLORS:
        return None
    return (f"unknown color {color!r}; use a hex code like '#5a8fd6' or one of "
            f"{', '.join(sorted(_KNOWN_COLORS))}")


def _range_error(r: list[float] | None, name: str) -> str | None:
    if r is None:
        return None
    if len(r) not in (2, 3) or not all(isinstance(v, (int, float)) for v in r):
        return f"{name} must be [min, max] or [min, max, tick_step]"
    if r[1] <= r[0]:
        return f"{name} must have min < max (got {list(r)})"
    if len(r) == 3 and r[2] <= 0:
        return f"{name} tick step must be > 0"
    return None


def _with_ticks(r: list[float] | None, default: list[float]) -> list[float]:
    """[min, max] -> [min, max, step] with a readable step (about 5-10 ticks)."""
    if r is None:
        return list(default)
    if len(r) == 3:
        return list(r)
    span = float(r[1]) - float(r[0])
    raw = span / 8
    mag = 10 ** __import__("math").floor(__import__("math").log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    return [r[0], r[1], step]


def _normalize_functions(functions: list[Any] | None, params: dict[str, float]) -> list[dict[str, Any]] | str:
    out = []
    for i, fn in enumerate(functions or [], start=1):
        if isinstance(fn, str):
            fn = {"expr": fn}
        if not isinstance(fn, dict) or "expr" not in fn:
            return f"function {i} must be {{'expr': '...'}} (optionally color, smooth, dx)"
        unknown = set(fn) - {"expr", "color", "smooth", "dx"}
        if unknown:
            return f"function {i}: unknown keys {sorted(unknown)} (allowed: expr, color, smooth, dx)"
        try:
            validate(str(fn["expr"]), params)
        except ExprError as exc:
            return f"function {i}: {exc}"
        if (ce := _color_error(fn.get("color"))):
            return f"function {i}: {ce}"
        if fn.get("smooth") is not None and not isinstance(fn["smooth"], bool):
            return f"function {i}: smooth must be true or false (got {fn['smooth']!r})"
        if fn.get("dx") is not None:
            if isinstance(fn["dx"], bool) or not isinstance(fn["dx"], (int, float)):
                return f"function {i}: dx must be a number (got {fn['dx']!r})"
            if fn["dx"] <= 0:
                return f"function {i}: dx must be > 0"
        out.append(dict(fn))
    return out


def _interval_error(plot: Obj, x_min: float, x_max: float) -> str | None:
    if x_max <= x_min:
        return "x_min must be < x_max"
    lo, hi = (float(v) for v in plot.payload.get("x_range", [-5, 5])[:2])
    if x_min < lo - 1e-9 or x_max > hi + 1e-9:
        return f"[{x_min}, {x_max}] is outside the x_range [{lo}, {hi}] of '{plot.id}'"
    return None


def _overlay_error(scene: Scene, plot: Obj, fn_index: int, x_min: float, x_max: float,
                   kind: str) -> str | None:
    """At the parameters' current values -- unless the curve follows a tweened
    parameter: then it is shown at other values, which verification checks at
    the moments the overlay is on screen."""
    tweened = {st.param for st in scene.steps if st.kind == "tween"}
    return overlay_error(scene, plot, fn_index, x_min, x_max, kind, scene.params,
                         axis_only=bool(obj_params(plot, scene) & tweened))


def _anchor_error(expr: str, axis: str) -> str | None:
    import ast
    if any(isinstance(n, ast.Name) and n.id == "x" for n in ast.walk(ast.parse(expr, mode="eval"))):
        return (f"at_{axis} cannot use x (anchors are numbers or scene parameters); "
                "write the value, e.g. at_x='2', at_y='2**2', or use a parameter")
    return None


def _fingerprint(scene: Scene) -> str:
    """Content hash of everything that affects the rendered video."""
    import hashlib
    import json
    d = scene.to_dict()
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def _measure(project: Project, scene: Scene) -> dict[str, Any] | None:
    """Probe the scene (exact geometry) when Manim is available, else None."""
    probe = rnd.probe_scene(project, scene, session.workdir(project.id, scene.id))
    if probe.get("ok") is None and not probe.get("timeout"):
        return None  # no Manim: estimated geometry
    session.set_last_probe(project.id, scene.id, probe)
    return probe


def _coverage(insp: dict[str, Any], scene: Scene, **extra: Any) -> dict[str, Any]:
    return {"lanes_run": insp["lanes_run"], "lanes_skipped": insp["lanes_skipped"],
            "measured": insp["measured"], "scope": scene.id, **extra}


# ---------------------------------------------------------------------------
# 6.1 Session & project
# ---------------------------------------------------------------------------

@mcp.tool()
def open_project(quality: str = "low", fps: int | None = None,
                 theme: str = "chalkboard") -> dict[str, Any]:
    """Start a project session and make it the current one (later calls without
    project_id target it). quality: low (480p) | medium (720p) | high (1080p).
    fps: frame rate, default 15/30/60 for low/medium/high. theme: chalkboard|whiteboard|dark."""
    if quality not in _DEFAULT_FPS:
        return err(f"unknown quality {quality!r}; use low|medium|high")
    if theme not in ("chalkboard", "whiteboard", "dark"):
        return err(f"unknown theme {theme!r}; use chalkboard|whiteboard|dark")
    if fps is not None and not 1 <= fps <= 120:
        return err("fps must be between 1 and 120")
    fps = fps or _DEFAULT_FPS[quality]
    project = session.create_project(quality=quality, fps=fps, theme=theme)
    return ok({"project_id": project.id, "quality": quality, "fps": fps, "theme": theme},
              coverage={"scope": project.id})


@mcp.tool()
def inspect_project(project_id: str | None = None) -> dict[str, Any]:
    """Current project state: scenes, objects, steps, parameters and open obligations."""
    project = session.get_project(project_id)
    if project is None:
        return err("no open project")
    # an obligation measured before later edits may no longer hold: say so
    changed = {sc.id for sc in project.scenes
               if _measured_at.get(f"{project.id}/{sc.id}") not in (None, _fingerprint(sc))}
    obls = [{**o, "stale": True} if o["scene"] in changed else o
            for o in project.obligations if o["status"] == "open"]
    d = project.to_dict()
    if "open_obligations" in d:
        d["open_obligations"] = obls
    return ok(d, obligations=obls, coverage={"scope": project.id},
              limitations=[f"scene(s) {sorted(changed)} changed since their last inspection: their "
                           "obligations are marked stale; call inspect_render again"] if changed else [])


@mcp.tool()
def close_project(project_id: str | None = None) -> dict[str, Any]:
    """Release a project session (default: the current one). If others are open, the
    most recently opened becomes current; the result names it."""
    if not session.close_project(project_id):
        return err(f"no open project {project_id!r}" if project_id else "no open project")
    current = session.get_project()
    return ok({"closed": True, "current_project": current.id if current else None})


# ---------------------------------------------------------------------------
# 6.2 Authoring
# ---------------------------------------------------------------------------

@mcp.tool()
def add_scene(title: str = "", project_id: str | None = None) -> dict[str, Any]:
    """Append a scene to the project; it becomes the default target of later calls
    without scene_id. The title is a label for you (it is not drawn: use add_text)."""
    project = session.get_project(project_id)
    if project is None:
        return err("no open project; call open_project first")
    scene = project.new_scene(title=title)
    return ok({"scene_id": scene.id, "title": title}, coverage={"scope": scene.id})


@mcp.tool()
def add_parameter(name: str, value: float = 0.0, scene_id: str | None = None,
                  project_id: str | None = None) -> dict[str, Any]:
    """Declare a continuous scene parameter (e.g. time t). Use it by name in plot
    expressions or shape anchors (expr='sin(x - t)'), then animate it smoothly with
    add_step(kind='tween', param=name, value=...). Curves, areas and anchored shapes
    that use it are redrawn every frame."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    import keyword
    if not _IDENT.match(name) or name in _RESERVED or keyword.iskeyword(name) \
            or keyword.issoftkeyword(name) or name in _SAFE_NAMES:
        return err(f"invalid parameter name {name!r} (an identifier, not one of {sorted(_RESERVED)})")
    from .expr import MATH_NAMES
    if name in MATH_NAMES:
        return err(f"parameter name {name!r} shadows a math function; pick another")
    previous = scene.params.get(name)
    scene.params[name] = float(value)
    res = {"scene_id": scene.id, "param": name, "value": float(value), "params": dict(scene.params)}
    if previous is not None:
        res["redeclared_from"] = previous  # every curve/anchor using it starts elsewhere now
    return ok(res)


@mcp.tool()
def add_equation(tex: str, x: float = 0, y: float = 0, color: str | None = None,
                 scale: float = 1.0, scene_id: str | None = None,
                 project_id: str | None = None) -> dict[str, Any]:
    """Add a LaTeX equation (math mode, e.g. tex=r'e^{i\\pi}+1=0'). Dirac notation
    (\\ket{n}, \\bra{n}, \\braket{a}{b}, \\expval{A}) and physics macros work."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    if (ce := _color_error(color)):
        return err(ce)
    if not tex.strip():
        return err("tex is empty")
    if scale <= 0:
        return err("scale must be > 0")
    oid = scene.new_obj_id("equation")
    scene.objects.append(Obj(id=oid, kind="equation", position=(x, y), color=color,
                             scale=scale, payload={"tex": tex}))
    return ok({"scene_id": scene.id, "object_id": oid},
              limitations=["LaTeX validity is checked by inspect_render / render_scene"],
              untrusted=["tex"])


@mcp.tool()
def add_plot(functions: list[dict[str, Any]] | None = None,
             x_range: list[float] | None = None, y_range: list[float] | None = None,
             x: float = 0, y: float = 0, width: float = 8.0, height: float = 5.0,
             x_label: str | None = None, y_label: str | None = None, show_numbers: bool = False,
             scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Add axes with functions of x. Each function: {'expr': 'sin(x)', 'color': 'YELLOW'}.
    Expressions use x, bare math functions (sin, exp, sqrt, sign, ...), np.<fn>, and
    declared scene parameters; they are validated now. Sampling is adaptive (dense,
    no spline overshoot on jumps, curves split at poles and clipped to the axes);
    optional overrides: 'smooth': bool, 'dx': step. x_label/y_label are LaTeX.
    Defaults: x_range [-5, 5, 1], y_range [-3, 3, 1]; a [min, max] range gets a
    readable tick step (4-8 ticks, starting at min, or at 0 if 0 is inside the range);
    the result echoes the ranges used. Curves
    without a color get distinct colors suited to the theme. The axes are centred at
    (x, y); labels hang off them. Without functions you get bare axes. The axes end
    in arrow tips at x_max / y_max, so those two values get no tick or number
    (mark a bound there with add_text/add_equation if it matters)."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    for r, name in ((x_range, "x_range"), (y_range, "y_range")):
        if (re_ := _range_error(r, name)):
            return err(re_)
    if width <= 0 or height <= 0:
        return err("width and height must be > 0")
    funcs = _normalize_functions(functions, scene.params)
    if isinstance(funcs, str):
        return err(funcs)
    oid = scene.new_obj_id("plot")
    payload: dict[str, Any] = {
        "functions": funcs,
        "x_range": _with_ticks(x_range, [-5, 5, 1]),
        "y_range": _with_ticks(y_range, [-3, 3, 1]),
        "width": width, "height": height,
    }
    if x_label:
        payload["x_label"] = x_label
    if y_label:
        payload["y_label"] = y_label
    if show_numbers:
        payload["show_numbers"] = True
    scene.objects.append(Obj(id=oid, kind="plot", position=(x, y), payload=payload))
    return ok({"scene_id": scene.id, "object_id": oid, "functions": len(funcs),
               "x_range": payload["x_range"], "y_range": payload["y_range"]}, untrusted=["expr"],
              limitations=["no tick or number at x_max / y_max: the axis tips sit there"]
              if show_numbers else [])


@mcp.tool()
def add_text(text: str, x: float = 0, y: float = 0, font_size: float = 36,
             color: str | None = None, scale: float = 1.0,
             scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Add a title/label/callout (plain text, any language). The rendered size is
    font_size x scale."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    if (ce := _color_error(color)):
        return err(ce)
    if font_size <= 0 or scale <= 0:
        return err("font_size and scale must be > 0")
    if not text.strip():
        return err("text is empty")
    oid = scene.new_obj_id("text")
    scene.objects.append(Obj(id=oid, kind="text", position=(x, y), color=color,
                             scale=scale, payload={"text": text, "font_size": font_size}))
    return ok({"scene_id": scene.id, "object_id": oid})


@mcp.tool()
def add_shape(shape: str = "circle", size: float | None = None, x: float | None = None,
              y: float | None = None, color: str | None = None,
              width: float | None = None, height: float | None = None,
              x1: float | None = None, y1: float | None = None,
              x2: float | None = None, y2: float | None = None,
              plot_id: str | None = None, at_x: str | None = None, at_y: str | None = None,
              scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Add primitive geometry: circle|square|rectangle|triangle|arrow|dot|line, centred
    at (x, y) (default 0, 0). size (default 1): circle = diameter, square = side,
    triangle = circumradius, dot = 10 x radius (minimum radius 0.05), line/arrow =
    length (horizontal).
    rectangle: width x height (default 1 x 0.5) instead of size.
    line/arrow: give x1,y1,x2,y2 for an explicit segment (any direction) instead of
    size/x/y. Anchor in a plot's coordinates with plot_id + at_x + at_y expressions
    instead of x/y (they may use scene parameters, so a dot can track <x>(t) while a
    tween runs). Arguments that do not apply to the chosen shape are rejected."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    if shape not in SHAPES:
        return err(f"unknown shape {shape!r}; use one of {', '.join(SHAPES)}")
    if (ce := _color_error(color)):
        return err(ce)
    seg = [x1, y1, x2, y2]
    has_seg = any(v is not None for v in seg)
    if shape == "rectangle":
        if size is not None:
            return err("rectangle takes width and height, not size")
        width = 1.0 if width is None else width
        height = 0.5 if height is None else height
        if width <= 0 or height <= 0:
            return err("width and height must be > 0")
        payload: dict[str, Any] = {"shape": shape, "width": width, "height": height}
    else:
        if width is not None or height is not None:
            return err(f"width/height apply to rectangle only; a {shape} uses size")
        if has_seg and size is not None:
            return err("give either size or x1,y1,x2,y2 for a line/arrow, not both")
        size = 1.0 if size is None else size
        if size <= 0:
            return err("size must be > 0")
        payload = {"shape": shape, "size": size}
    if has_seg:
        if shape not in ("line", "arrow") or any(v is None for v in seg):
            return err("x1,y1,x2,y2 must all be given, and only for line/arrow")
        if x is not None or y is not None or plot_id is not None:
            return err("an explicit segment (x1..y2) is placed by its endpoints; drop x/y/plot_id")
        payload.update(x1=x1, y1=y1, x2=x2, y2=y2)
    if plot_id is not None:
        plot = scene.get_obj(plot_id)
        if plot is None or plot.kind != "plot":
            return err(f"plot_id '{plot_id}' is not a plot in this scene")
        if at_x is None or at_y is None:
            return err("an anchored shape needs both at_x and at_y (plot coordinates)")
        if x is not None or y is not None:
            return err("an anchored shape is placed by at_x/at_y; drop x/y")
        anchor = {"plot": plot_id, "x": str(at_x), "y": str(at_y)}
        for k in ("x", "y"):
            try:
                validate(anchor[k], scene.params)
            except ExprError as exc:
                return err(f"at_{k}: {exc}")
            if (e := _anchor_error(anchor[k], k)):
                return err(e)
        payload["anchor"] = anchor
    elif at_x is not None or at_y is not None:
        return err("at_x/at_y need plot_id (they are plot coordinates)")
    oid = scene.new_obj_id("shape")
    scene.objects.append(Obj(id=oid, kind="shape", position=(x or 0.0, y or 0.0), color=color,
                             payload=payload))
    return ok({"scene_id": scene.id, "object_id": oid})


@mcp.tool()
def add_area(plot_id: str, x_min: float = 0.0, x_max: float = 1.0, function_index: int = 1,
             color: str | None = "BLUE", opacity: float = 0.5,
             scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Shade the area between a plotted function and the x-axis on [x_min, x_max] (the
    integral). [x_min, x_max] must lie within the plot's x_range, y = 0 must be on the
    axes, and the function must be defined and inside y_range there (otherwise an
    error says what to widen). It follows its plot (move the plot, not it)."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    plot = scene.get_obj(plot_id)
    if plot is None or plot.kind != "plot":
        return err(f"add_area needs a valid plot_id; '{plot_id}' is not a plot")
    nf = len(plot.payload.get("functions", []))
    if not 1 <= function_index <= nf:
        return err(f"function_index must be in 1..{nf} for '{plot_id}'")
    if (e := _interval_error(plot, x_min, x_max)) or \
            (e := _overlay_error(scene, plot, function_index, x_min, x_max, "area")):
        return err(e)
    if not 0 <= opacity <= 1:
        return err("opacity must be in [0, 1]")
    if (ce := _color_error(color)):
        return err(ce)
    color = color or "BLUE"
    oid = scene.new_obj_id("area")
    scene.objects.append(Obj(id=oid, kind="area", position=plot.position, color=color, payload={
        "plot": plot_id, "fn": function_index, "x_min": x_min, "x_max": x_max, "opacity": opacity,
        "width": plot.payload.get("width", 6), "height": plot.payload.get("height", 4),
    }))
    return ok({"scene_id": scene.id, "object_id": oid})


@mcp.tool()
def add_riemann(plot_id: str, x_min: float = 0.0, x_max: float = 1.0, dx: float = 0.1,
                function_index: int = 1, color: str | None = "BLUE",
                scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Riemann rectangles (left endpoints, fill opacity 0.6, one color) under a plotted
    function on [x_min, x_max] within the plot's x_range; (x_max - x_min)/dx must be a
    whole number, y = 0 must be on the axes and the function inside y_range there.
    Smaller dx = finer; transform coarse->fine to show convergence. They follow
    their plot."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    plot = scene.get_obj(plot_id)
    if plot is None or plot.kind != "plot":
        return err(f"add_riemann needs a valid plot_id; '{plot_id}' is not a plot")
    nf = len(plot.payload.get("functions", []))
    if not 1 <= function_index <= nf:
        return err(f"function_index must be in 1..{nf} for '{plot_id}'")
    if dx <= 0:
        return err("dx must be > 0")
    if (e := _interval_error(plot, x_min, x_max)) or \
            (e := _overlay_error(scene, plot, function_index, x_min, x_max, "Riemann sum")):
        return err(e)
    n = (x_max - x_min) / dx
    if abs(n - round(n)) > 1e-6:
        return err(f"(x_max - x_min) / dx = {n:.4g} is not a whole number of rectangles; "
                   f"pick dx = {(x_max - x_min) / max(1, round(n)):.6g} or adjust the interval")
    if (ce := _color_error(color)):
        return err(ce)
    color = color or "BLUE"
    oid = scene.new_obj_id("riemann")
    scene.objects.append(Obj(id=oid, kind="riemann", position=plot.position, color=color, payload={
        "plot": plot_id, "fn": function_index, "x_min": x_min, "x_max": x_max, "dx": dx,
        "width": plot.payload.get("width", 6), "height": plot.payload.get("height", 4),
    }))
    return ok({"scene_id": scene.id, "object_id": oid})


@mcp.tool()
def add_step(kind: str, targets: list[str] | None = None, to: str | None = None,
             duration: float = 1.0, note: str = "", zoom: float | None = None,
             focus_x: float | None = None, focus_y: float | None = None,
             with_previous: bool = False, param: str | None = None, value: float | None = None,
             scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Append a timed beat. kind: write|transform|highlight|fade|camera_move|wait|tween.
    write/highlight/fade: all targets animate together, over `duration` (a plot's axes
    and curves are drawn together; highlight is a brief 1.2x pulse in the theme's
    accent color). An object driven by a parameter cannot be written, highlighted
    or faded in the same beat as a tween of that parameter (Manim freezes it).
    transform: one target morphs into `to` (the target keeps its id afterwards).
    camera_move: zoom is absolute (1 = the full frame, 2 = twice as close); focus_x/
    focus_y recentre the camera, omitted = keep the current centre.
    tween: animate scene parameter `param` to `value` continuously (linear in time).
    with_previous=true plays this step at the same time as the previous one (e.g. move
    a curve and a dot together); not after a wait. `note` is a label for you (not
    drawn). Arguments that do not apply to `kind` are rejected."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    if kind not in STEP_KINDS:
        return err(f"unknown step kind {kind!r}; use one of {', '.join(STEP_KINDS)}")
    if duration <= 0:
        return err("duration must be > 0")
    given = {"targets": bool(targets), "to": to is not None, "zoom": zoom is not None,
             "focus_x": focus_x is not None, "focus_y": focus_y is not None,
             "param": param is not None, "value": value is not None}
    allowed = _STEP_ARGS[kind]
    extra = [k for k, v in given.items() if v and k not in allowed]
    if extra:
        return err(f"{kind} does not take {extra} (it takes {sorted(allowed) or 'only duration'})")
    targets = targets or []
    if len(set(targets)) != len(targets):
        return err(f"duplicate targets in {targets}; list each object once")
    defined = {o.id for o in scene.objects}
    unknown = [t for t in targets if t not in defined]
    if unknown:
        return err(f"unknown target(s) {unknown}; existing objects: {sorted(defined)}")
    if kind in ("write", "highlight", "fade") and not targets:
        return err(f"{kind} needs at least one target")
    if kind == "transform":
        if len(targets) != 1 or not to:
            return err("transform needs exactly one target and a destination `to`")
        if to not in defined:
            return err(f"transform destination '{to}' does not exist; create it first")
        moving = [o for o in (targets[0], to) if is_dynamic(scene.get_obj(o), scene)]
        if moving:
            return err(f"{moving} follow a scene parameter and would snap back after a "
                       "transform; fade one out and write the other instead")
    if kind == "camera_move":
        if not zoom or zoom <= 0:
            return err("camera_move needs zoom > 0 (absolute: 1 = the full frame)")
        if (focus_x is None) != (focus_y is None):
            return err("give both focus_x and focus_y, or neither (keep the current centre)")
        fx = "" if focus_x is None else focus_x
        fy = "" if focus_y is None else focus_y
        to = f"{zoom}@{fx},{fy}"
    if kind == "tween":
        if param not in scene.params:
            return err(f"tween needs a declared parameter; got {param!r}, declared: {sorted(scene.params)}")
        if value is None:
            return err("tween needs a target value")
    if with_previous:
        if not scene.steps:
            return err("with_previous needs a previous step")
        if kind == "wait" or scene.steps[-1].kind == "wait":
            return err("with_previous cannot join a wait (a wait is a pause, not an animation)")
    step = scene.add_step(kind, targets=targets, to=to, duration=duration, note=note,
                          with_previous=with_previous, param=param, value=value)
    conflicts = beat_conflicts(scene, only_step=step.index)
    if conflicts:
        scene.remove_step(step.index)
        return err(conflicts[0])
    res = {"scene_id": scene.id, "step_index": step.index, "kind": kind,
           "with_previous": with_previous, "scene_duration": scene.duration()}
    if kind == "tween" and step.index in unused_tweens(scene):
        res["warning"] = f"no object uses parameter '{param}' yet: this tween changes nothing on screen"
    return ok(res)


# ---------------------------------------------------------------------------
# Editing (fast correction loop)
# ---------------------------------------------------------------------------

@mcp.tool()
def update_object(object_id: str, x: float | None = None, y: float | None = None,
                  scale: float | None = None, color: str | None = None,
                  text: str | None = None, tex: str | None = None, font_size: float | None = None,
                  functions: list[dict[str, Any]] | None = None,
                  x_range: list[float] | None = None, y_range: list[float] | None = None,
                  width: float | None = None, height: float | None = None,
                  x_label: str | None = None, y_label: str | None = None,
                  show_numbers: bool | None = None, size: float | None = None,
                  at_x: str | None = None, at_y: str | None = None,
                  x_min: float | None = None, x_max: float | None = None,
                  dx: float | None = None, opacity: float | None = None,
                  x1: float | None = None, y1: float | None = None,
                  x2: float | None = None, y2: float | None = None,
                  scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Change an existing object in place (only the fields you pass). Use it to
    reposition, rescale, recolor, retype text/LaTeX, replace plot functions or
    ranges, or adjust an area/Riemann interval -- no need to rebuild the scene.
    Explicit line/arrow segments move by their endpoints x1, y1, x2, y2.
    Everything is validated first: on error nothing changes. A plot's color is the
    default color of its curves without one (axes keep the theme ink). Areas, Riemann
    sums and anchored shapes follow their plot: move the plot instead."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    obj = scene.get_obj(object_id)
    if obj is None:
        return err(f"unknown object '{object_id}'; existing: {sorted(o.id for o in scene.objects)}")
    if (ce := _color_error(color)):
        return err(ce)
    p = dict(obj.payload)
    changed: list[str] = []
    shape = p.get("shape")
    fields = (
        ("text", text, ("text",), None), ("tex", tex, ("equation",), None),
        ("font_size", font_size, ("text",), "pos"),
        ("width", width, ("plot",) + (("shape",) if shape == "rectangle" else ()), "pos"),
        ("height", height, ("plot",) + (("shape",) if shape == "rectangle" else ()), "pos"),
        ("x_label", x_label, ("plot",), None), ("y_label", y_label, ("plot",), None),
        ("show_numbers", show_numbers, ("plot",), None),
        ("size", size, ("shape",) if shape != "rectangle" and "x2" not in p else (), "pos"),
        ("x_min", x_min, ("area", "riemann"), None), ("x_max", x_max, ("area", "riemann"), None),
        ("dx", dx, ("riemann",), "pos"), ("opacity", opacity, ("area",), "unit"),
    )
    for field, val, kinds, rule in fields:
        if val is None:
            continue
        if obj.kind not in kinds:
            what = f"a {shape}" if obj.kind == "shape" else f"a {obj.kind}"
            return err(f"'{field}' does not apply to {what}")
        if field in ("text", "tex") and not str(val).strip():
            return err(f"{field} is empty")
        if rule == "pos" and val <= 0:
            return err(f"{field} must be > 0")
        if rule == "unit" and not 0 <= val <= 1:
            return err(f"{field} must be in [0, 1]")
        p[field] = val
        changed.append(field)
    for k, v in (("x1", x1), ("y1", y1), ("x2", x2), ("y2", y2)):
        if v is not None:
            if "x2" not in p:
                return err(f"'{k}' applies to a line/arrow created with x1, y1, x2, y2")
            p[k] = v
            changed.append(k)
    for r, name in ((x_range, "x_range"), (y_range, "y_range")):
        if r is not None:
            if obj.kind != "plot":
                return err(f"'{name}' does not apply to a {obj.kind}")
            if (re_ := _range_error(r, name)):
                return err(re_)
            p[name] = _with_ticks(r, [])
            changed.append(name)
    if functions is not None:
        if obj.kind != "plot":
            return err(f"'functions' does not apply to a {obj.kind}")
        funcs = _normalize_functions(functions, scene.params)
        if isinstance(funcs, str):
            return err(funcs)
        p["functions"] = funcs
        changed.append("functions")
    if at_x is not None or at_y is not None:
        if not p.get("anchor"):
            return err("at_x/at_y apply to a shape anchored with plot_id")
        for k, v in (("x", at_x), ("y", at_y)):
            if v is not None:
                try:
                    validate(v, scene.params)
                except ExprError as exc:
                    return err(f"at_{k}: {exc}")
                if (e := _anchor_error(v, k)):
                    return err(e)
                p["anchor"] = {**p["anchor"], k: str(v)}
        changed.append("anchor")
    follows = obj.kind in ("area", "riemann") or bool(p.get("anchor")) or "x2" in p
    if (x is not None or y is not None) and follows:
        return err(f"'{object_id}' is placed by its plot/anchor/endpoints (x1..y2), not by x/y"
                   + ("; move the plot instead" if obj.kind in ("area", "riemann") else ""))
    if scale is not None:
        if obj.kind in ("area", "riemann"):
            return err(f"'{object_id}' follows its plot; scale the plot instead")
        if "x2" in p:
            return err(f"'{object_id}' is an explicit segment; change x1, y1, x2, y2 instead")
        if scale <= 0:
            return err("scale must be > 0")
    if obj.kind == "riemann":
        n = (float(p["x_max"]) - float(p["x_min"])) / float(p.get("dx", 0.1))
        if abs(n - round(n)) > 1e-6:
            return err(f"(x_max - x_min) / dx = {n:.4g} is not a whole number of rectangles")
    if obj.kind in ("area", "riemann"):
        plot = scene.get_obj(p["plot"])
        if plot is not None and ((e := _interval_error(plot, float(p["x_min"]), float(p["x_max"])))
                                 or (e := _overlay_error(scene, plot, int(p.get("fn", 1)),
                                                         float(p["x_min"]), float(p["x_max"]),
                                                         obj.kind))):
            return err(e)
    if obj.kind == "plot":
        nf = len(p.get("functions", []))
        lo, hi = (float(v) for v in p.get("x_range", [-5, 5])[:2])
        for dep in scene.objects:
            if dep.payload.get("plot") != obj.id:
                continue
            if int(dep.payload.get("fn", 1)) > nf:
                return err(f"'{dep.id}' uses function {dep.payload.get('fn', 1)} of '{obj.id}', "
                           f"which would only have {nf}; update or remove '{dep.id}' first")
            if dep.payload.get("x_min", lo) < lo or dep.payload.get("x_max", hi) > hi:
                return err(f"'{dep.id}' covers [{dep.payload['x_min']}, {dep.payload['x_max']}], "
                           f"outside the new x_range; adjust it first")
            if dep.kind in ("area", "riemann"):
                trial = Obj(id=obj.id, kind="plot", payload=p)
                if (e := _overlay_error(scene, trial, int(dep.payload.get("fn", 1)),
                                        float(dep.payload["x_min"]), float(dep.payload["x_max"]),
                                        dep.kind)):
                    return err(f"'{dep.id}' would break: {e}")
    if not changed and x is None and y is None and scale is None and color is None:
        return err("nothing to change; pass at least one field")
    # all checks passed: commit
    obj.payload = p
    if x is not None or y is not None:
        ox, oy = obj.position
        obj.position = (ox if x is None else x, oy if y is None else y)
        changed.append("position")
        if obj.kind == "plot":  # overlays follow their plot
            for dep in scene.objects:
                if dep.payload.get("plot") == obj.id:
                    dep.position = obj.position
    if scale is not None:
        obj.scale = scale
        changed.append("scale")
    if color is not None:
        obj.color = color
        changed.append("color")
    return ok({"scene_id": scene.id, "object_id": object_id, "changed": changed,
               "object": obj.to_dict()})


@mcp.tool()
def remove_object(object_id: str, cascade: bool = False, scene_id: str | None = None,
                  project_id: str | None = None) -> dict[str, Any]:
    """Delete an object. If steps or other objects (areas, anchored shapes) depend on
    it, nothing is removed unless cascade=true, which also removes the dependent
    objects and takes the removed ids out of steps (a step left with nothing to
    animate is deleted; other targets of a shared step stay). The reply lists exactly
    what was deleted or changed, and any object left without a step."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    if scene.get_obj(object_id) is None:
        return err(f"unknown object '{object_id}'")
    to_remove = [object_id]
    i = 0
    while i < len(to_remove):  # transitive dependents
        for d in scene.dependents(to_remove[i]):
            if d not in to_remove:
                to_remove.append(d)
        i += 1
    steps = sorted({s for oid in to_remove for s in scene.steps_referencing(oid)})
    if (len(to_remove) > 1 or steps) and not cascade:
        return err(f"'{object_id}' is used by objects {to_remove[1:]} and steps {steps}; "
                   "pass cascade=true to remove them too, or update them first",
                   payload={"dependent_objects": to_remove[1:], "dependent_steps": steps})
    gone = set(to_remove)
    removed_steps, edited_steps = [], []
    for st in list(scene.steps):
        if st.index not in steps:
            continue
        if st.kind == "transform" or not [t for t in st.targets if t not in gone]:
            removed_steps.append(st.index)
        else:
            st.targets = [t for t in st.targets if t not in gone]
            edited_steps.append(st.index)
    for oid in to_remove:
        scene.remove_obj(oid)
    regrouped = _remove_steps(scene, removed_steps)
    shown = {t for st in scene.steps for t in st.targets} | {st.to for st in scene.steps if st.to}
    orphaned = [o.id for o in scene.objects if o.id not in shown] if scene.steps else []
    return ok({"scene_id": scene.id, "removed_objects": to_remove, "removed_steps": removed_steps,
               "edited_steps": edited_steps, "regrouped_steps": regrouped,
               "objects_without_step": orphaned, "remaining_steps": len(scene.steps)})


def _remove_steps(scene: Scene, indices: list[int]) -> list[int]:
    """Remove steps (original indices); returns the new indices of steps whose
    with_previous grouping changed (they now start their own beat)."""
    doomed = set(indices)
    # a removed step that began a beat leaves the steps joined to it without a partner
    starters = {s.index for s in scene.steps if s.index in doomed and not s.with_previous}
    regrouped = []
    kept = [s for s in scene.steps if s.index not in doomed]
    for i, st in enumerate(kept):
        prev = kept[i - 1] if i else None
        lo = prev.index if prev else 0
        if st.with_previous and (prev is None or prev.kind == "wait" or
                                 any(lo < d < st.index for d in starters)):
            st.with_previous = False
            regrouped.append(st)
    scene.steps = kept
    for i, st in enumerate(scene.steps, start=1):
        st.index = i
    return [st.index for st in regrouped]


@mcp.tool()
def remove_step(step_index: int, scene_id: str | None = None,
                project_id: str | None = None) -> dict[str, Any]:
    """Delete one step of the timeline (later steps are renumbered). A with_previous
    step that was joined to the removed one starts its own beat; it is listed in
    regrouped_steps."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    if not any(s.index == step_index for s in scene.steps):
        return err(f"no step {step_index}; the scene has {len(scene.steps)} steps")
    regrouped = _remove_steps(scene, [step_index])
    return ok({"scene_id": scene.id, "removed_step": step_index, "regrouped_steps": regrouped,
               "steps": [s.to_dict() for s in scene.steps]})


# ---------------------------------------------------------------------------
# 6.3 Rendering + 6.4 Verification
# ---------------------------------------------------------------------------

@mcp.tool()
def inspect_render(scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Verify a scene WITHOUT rendering video (~1-2 s): builds the real Manim objects
    and measures them -- LaTeX/expression compile errors, off-screen objects, label
    collisions, curve fidelity (drawn vs true function), legibility, timing.
    Call it after authoring and fix what it reports before render_scene."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    project, scene = got
    probe = _measure(project, scene)
    insp = inspect_scene(scene, measurement=probe, theme_name=project.theme)
    obls = _register_obligations(project, scene.id, insp["obligations"])
    return ok({"scene_id": scene.id},
              timing=(probe or {}).get("timing", {}),
              coverage=_coverage(insp, scene, available_lanes=GEOMETRIC_LANES),
              verdict=insp["verdict"], obligations=obls, limitations=insp["limitations"],
              untrusted=["latex_log"])


@mcp.tool()
def fix_scene(scene_id: str | None = None, max_rounds: int = 3,
              project_id: str | None = None) -> dict[str, Any]:
    """Automatically repair measured problems that have a mechanical fix: objects
    pushed back inside the frame, labels moved to the nearest free spot, curve
    sampling corrected, too-small text enlarged. Repeats measure->fix up to
    max_rounds, reports every change, and lists what still needs the author."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    project, scene = got
    if max_rounds < 1:
        return err("max_rounds must be >= 1")
    all_changes: list[dict[str, Any]] = []
    probe = _measure(project, scene)
    if probe is None or probe.get("ok") is not True:
        return err("fix_scene needs measured geometry (Manim, and a probe that completes); "
                   "estimated findings are not auto-fixed")
    insp = inspect_scene(scene, measurement=probe, theme_name=project.theme)
    attempts: dict[str, int] = {}  # repairs tried per curve, so none is retried in a loop
    for _ in range(max_rounds):
        changes, _unresolved = apply_fixes(scene, insp["obligations"], probe, attempts)
        if not changes:
            break
        all_changes += changes
        probe = _measure(project, scene)
        insp = inspect_scene(scene, measurement=probe, theme_name=project.theme)
        if probe is None or probe.get("ok") is not True:
            break
    obls = _register_obligations(project, scene.id, insp["obligations"])
    remaining = [o for o in obls if o["severity"] in ("error", "warning")]
    return ok({"scene_id": scene.id, "changes": all_changes, "needs_author": remaining},
              coverage=_coverage(insp, scene), verdict=insp["verdict"], obligations=obls,
              limitations=insp["limitations"])


@mcp.tool()
def render_scene(scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Render one scene to video + sampled frames, then run verification lanes.
    The scene is measured first; if it cannot compile, no render is attempted."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    project, scene = got
    wd = session.workdir(project.id, scene.id)
    probe = _measure(project, scene)
    if probe is not None and probe.get("ok") is False:
        insp = inspect_scene(scene, measurement=probe, theme_name=project.theme)
        obls = _register_obligations(project, scene.id, insp["obligations"])
        return err("scene construction failed; render skipped", payload={"scene_id": scene.id},
                   timing=probe.get("timing", {}), coverage=_coverage(insp, scene),
                   verdict=insp["verdict"], obligations=obls, limitations=insp["limitations"],
                   untrusted=["manim_stderr", "latex_log"])
    report = rnd.render_scene(project, scene, wd)
    report["fingerprint"] = _fingerprint(scene)

    insp = inspect_scene(scene, render_report=report, measurement=probe,
                         theme_name=project.theme)
    obls = _register_obligations(project, scene.id, insp["obligations"])
    # export is judged on this render's own verdict, not on a later inspection
    report["blocking"] = [o for o in obls if o["severity"] == "error"]
    session.set_last_render(project.id, scene.id, report)
    timing = {**report.get("timing", {}), **(probe or {}).get("timing", {})}
    if not report.get("ok"):
        return err("render failed", payload={"scene_id": scene.id}, timing=timing,
                   coverage=_coverage(insp, scene), verdict=insp["verdict"], obligations=obls,
                   limitations=insp["limitations"] + [str(report.get("error", "")),
                                                       report.get("stderr_tail", "")],
                   untrusted=["manim_stderr", "latex_log"])
    return ok({"scene_id": scene.id, "artifact": report.get("artifact")},
              artifacts=[report["artifact"]] if report.get("artifact") else [],
              frames=report.get("frames", []), timing=timing,
              coverage=_coverage(insp, scene), verdict=insp["verdict"], obligations=obls,
              limitations=insp["limitations"], untrusted=["manim_stderr", "latex_log"])


@mcp.tool()
def render_frame(scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Fast single-still preview of a scene (last frame)."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    project, scene = got
    wd = session.workdir(project.id, scene.id)
    report = rnd.render_scene(project, scene, wd, single_frame=True)
    if not report.get("ok"):
        return err("preview failed", limitations=[str(report.get("error", "")),
                                                   report.get("stderr_tail", "")],
                   untrusted=["manim_stderr"])
    return ok({"scene_id": scene.id}, frames=report.get("frames", []),
              timing=report.get("timing", {}))


# ---------------------------------------------------------------------------
# 6.5 Export
# ---------------------------------------------------------------------------

@mcp.tool()
def export_gif(width: int = 640, fps: int = 15, scene_id: str | None = None,
               project_id: str | None = None) -> dict[str, Any]:
    """Export the latest render of a scene as a shareable GIF, resampled to `width`
    pixels wide at `fps` (defaults keep GIFs small). Blocked by open error
    obligations of that render and by edits made since it."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    project, scene = got
    report = session.get_last_render(project.id, scene.id)
    if not report or not report.get("ok") or not report.get("artifact"):
        return err("no rendered video for this scene; call render_scene first")
    if report.get("fingerprint") != _fingerprint(scene):
        return err("the scene changed since its last render; call render_scene again so the "
                   "GIF (and its verification) matches the current scene")
    # the video is exactly the rendered scene: its render-time verdict decides
    blocking = report.get("blocking", [])
    if blocking:
        return err("export blocked by error obligations found when this video was rendered; "
                   "fix them (fix_scene can repair mechanical ones) and re-render",
                   obligations=blocking)
    if not 64 <= width <= 3840 or not 1 <= fps <= 60:
        return err("width must be in 64..3840 and fps in 1..60")
    res = rnd.export_gif(Path(report["artifact"]), session.workdir(project.id, scene.id),
                         width=width, fps=fps)
    if not res.get("ok"):
        return err(str(res.get("error")), limitations=[res.get("stderr_tail", "")])
    return ok({"scene_id": scene.id, "gif": res["artifact"], "width": width, "fps": fps},
              artifacts=[res["artifact"]])


# ---------------------------------------------------------------------------
# 6.6 Discovery
# ---------------------------------------------------------------------------

@mcp.tool()
def inspect_environment() -> dict[str, Any]:
    """Which providers are present (Manim, LaTeX, ffmpeg, vision) and what degrades without each."""
    env = rnd.environment()
    limits = []
    if not env["manim"]:
        limits.append("manim missing: no video output (authoring + estimated verification still work)")
    if not env["latex"]:
        limits.append("latex missing: equations unavailable")
    if not env["ffmpeg"]:
        limits.append("ffmpeg missing: GIF/thumbnail export disabled")
    return ok(env, limitations=limits)


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
