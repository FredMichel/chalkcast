"""Compile a Chalkcast scene graph into runnable Manim Community code.

The generated module has three parts:

* a runtime prelude (math names, a LaTeX template with Dirac-notation macros,
  and ``_cc_plot`` -- adaptive, discontinuity-aware, axes-clipped function plots);
* ``_cc_build()`` which constructs every Mobject and returns them by id. Both the
  scene and the geometry probe (``chalkcast.probe``) call it, so what verification
  measures is exactly what gets rendered;
* one ``MovingCameraScene`` subclass that unpacks those objects and plays the
  timeline (steps flagged ``with_previous`` share a single ``self.play``).

Object ids double as Python variable names (valid identifiers by construction).
"""

from __future__ import annotations


from .expr import free_names
from .model import Obj, Scene, Step
from .themes import theme

_KNOWN_COLORS = {
    "WHITE", "BLACK", "RED", "GREEN", "BLUE", "YELLOW", "ORANGE", "PURPLE",
    "PINK", "TEAL", "GRAY", "GREY", "GOLD", "MAROON", "LIGHT_GRAY", "DARK_GRAY",
    "BLUE_A", "BLUE_B", "BLUE_C", "BLUE_D", "BLUE_E", "GREEN_A", "GREEN_E",
    "RED_A", "RED_E", "YELLOW_A", "YELLOW_E",
}

# Executed at import of every generated module (and by the probe).
_PRELUDE = r'''
from manim import *
import numpy as np
# make common math usable bare in plot expressions, e.g. add_plot(expr='sin(x)')
from numpy import (sin, cos, tan, exp, sqrt, log, log10, log2, sinh, cosh,
                   tanh, arcsin, arccos, arctan, arctan2, floor, ceil, sign,
                   heaviside, where, minimum, maximum, clip, pi, e)
from numpy import abs as _np_abs
abs = _np_abs

# Dirac notation works out of the box: the physics package when installed,
# otherwise equivalent fallback macros.
_cc_tex = TexTemplate()
_cc_tex.add_to_preamble(r"""
\IfFileExists{physics.sty}{\usepackage{physics}}{%
\providecommand{\ket}[1]{\left|#1\right\rangle}%
\providecommand{\bra}[1]{\left\langle#1\right|}%
\providecommand{\braket}[2]{\left\langle#1\middle|#2\right\rangle}%
\providecommand{\expval}[1]{\left\langle#1\right\rangle}%
\providecommand{\abs}[1]{\left|#1\right|}%
\providecommand{\dv}[2]{\frac{d#1}{d#2}}%
\providecommand{\pdv}[2]{\frac{\partial#1}{\partial#2}}}
""")
config.tex_template = _cc_tex


# Math failures at a point (pole, domain edge) become NaN; anything else is a bug in
# the expression and must surface instead of silently drawing nothing.
_CC_MATH_ERRORS = (ZeroDivisionError, ValueError, OverflowError, FloatingPointError)


def _cc_eval(f, xs):
    """Vectorized evaluation with a scalar fallback; math failures become NaN."""
    with np.errstate(all="ignore"):
        try:
            ys = np.asarray(f(xs), dtype=float)
            if ys.shape != xs.shape:
                raise ValueError
            return ys
        except _CC_MATH_ERRORS + (TypeError,):
            out = []
            for v in xs:
                try:
                    out.append(float(f(v)))
                except _CC_MATH_ERRORS:
                    out.append(np.nan)
            return np.array(out, dtype=float)


def _cc_plot(ax, f, color, smooth=None, dx=None, clip=True):
    """Plot f on ax, faithfully.

    * dense sampling by default (600 samples over the x range), so high
      frequencies are not aliased by Manim's tick-based default;
    * jump discontinuities split the curve (separate steps, no invented riser or
      overshoot) -- unless smooth=True is forced, which keeps it whole;
    * poles / NaNs / values far outside y_range split the curve and the graph is
      clipped to the axes box instead of shooting off-screen.
    """
    x0, x1 = float(ax.x_range[0]), float(ax.x_range[1])
    y0, y1 = float(ax.y_range[0]), float(ax.y_range[1])
    span = y1 - y0
    xs = np.linspace(x0, x1, 4001)
    ys = _cc_eval(f, xs)
    fin = np.isfinite(ys)
    ok = fin.copy()
    if clip:
        margin = 0.02 * span
        ok &= (ys >= y0 - margin) & (ys <= y1 + margin)
    d = np.abs(np.diff(ys))
    d = np.where(np.isfinite(d), d, 0.0)
    # a jump: a big step, or a step far larger than its neighbours (floor(x) on a
    # wide y-range jumps by only a few % of the span, but from a flat line)
    # (immediate neighbours skipped: a jump can land on a sample, e.g. sign(0) = 0,
    # and then takes two half-steps)
    nb = np.maximum.reduce([np.roll(d, k) for k in (-4, -3, -2, 2, 3, 4)])
    jumps = fin[:-1] & fin[1:] & ((d > 0.15 * span) | ((d > 0.01 * span) & (d > 8 * nb)))
    split = smooth is not True  # only forcing smooth=True keeps a curve whole
    if smooth is None:
        smooth = not bool(jumps.any())
    if not split:
        jumps = np.zeros_like(jumps)
    if dx is None:
        dx = (x1 - x0) / 600.0
    pieces = []
    i, n = 0, len(xs)
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1] and not jumps[j]:
            j += 1
        if j > i:
            a, b = float(xs[i]), float(xs[j])
            # next to an undefined sample (e.g. sqrt(1-x**2) past x=1) the curve would
            # stop one grid step short: bisect to the edge of the domain
            if i > 0 and not fin[i - 1]:
                a = _cc_edge(f, float(xs[i - 1]), a)
            if j + 1 < n and not fin[j + 1]:
                b = _cc_edge(f, float(xs[j + 1]), b)
            step = min(dx, (b - a) / 2.0)
            pieces.append(ax.plot(f, x_range=[a, b, step], color=color,
                                  use_smoothing=bool(smooth)))
        i = j + 1
    if len(pieces) == 1:
        return pieces[0]
    return VGroup(*pieces)


def _cc_edge(f, bad, good):
    """The point nearest ``bad`` (undefined) where f is still defined, from ``good``."""
    for _ in range(40):
        mid = (bad + good) / 2.0
        with np.errstate(all="ignore"):
            try:
                v = float(f(mid))
            except Exception:
                v = float("nan")
        if np.isfinite(v):
            good = mid
        else:
            bad = mid
    return good


def _cc_helper_graph(ax, f, a, b):
    """Unsmoothed graph on [a, b], clipped to the y-range like the drawn curve,
    used to build areas / Riemann sums (so a fill never leaves its axes)."""
    y0, y1 = float(ax.y_range[0]), float(ax.y_range[1])

    def g(x):
        with np.errstate(all="ignore"):
            v = float(f(x))
        return float(np.clip(v, y0, y1)) if np.isfinite(v) else 0.0  # undefined: no fill

    return ax.plot(g, x_range=[a, b, (b - a) / 300.0], use_smoothing=False)
'''


# default colour of the i-th curve of a plot, so two curves are never identical
_COLOR_HEX = {  # Manim CE values of the names above
    'BLACK': '#000000', 'BLUE': '#58c4dd', 'BLUE_A': '#c7e9f1', 'BLUE_B': '#9cdceb',
    'BLUE_C': '#58c4dd', 'BLUE_D': '#29abca', 'BLUE_E': '#236b8e', 'DARK_GRAY': '#444444',
    'GOLD': '#f0ac5f', 'GRAY': '#888888', 'GREEN': '#83c167', 'GREEN_A': '#c9e2ae',
    'GREEN_E': '#699c52', 'GREY': '#888888', 'LIGHT_GRAY': '#bbbbbb', 'MAROON': '#c55f73',
    'ORANGE': '#ff862f', 'PINK': '#d147bd', 'PURPLE': '#9a72ac', 'RED': '#fc6255',
    'RED_A': '#f7a1a3', 'RED_E': '#cf5044', 'TEAL': '#5cd0b3', 'WHITE': '#ffffff',
    'YELLOW': '#f7d96f', 'YELLOW_A': '#fff1b6', 'YELLOW_E': '#e8c11c',
}

CURVE_COLORS = ("#f6c76b", "#6bc5f6", "#f6866b", "#8ee07a", "#c99bf0")  # dark boards
LIGHT_CURVE_COLORS = ("#c2410c", "#1d4ed8", "#b91c1c", "#15803d", "#7e22ce")


def curve_colors(ink: str) -> tuple[str, ...]:
    """Default curve palette that contrasts with the theme (dark ink = light board)."""
    return LIGHT_CURVE_COLORS if ink.lstrip("#")[:1] in "0123" else CURVE_COLORS


def _color(value: str | None, default: str) -> str:
    """Return a Manim color expression (a constant token or a quoted hex/name)."""
    if not value:
        return repr(default)
    if value.startswith("#"):
        return repr(value)
    up = value.upper()
    if up in _KNOWN_COLORS:
        return up
    return repr(value)


def _pos(obj: Obj) -> str:
    x, y = obj.position
    return f".move_to([{x}, {y}, 0])"


def _scale(obj: Obj) -> str:
    return f".scale({obj.scale})" if obj.scale != 1.0 else ""


def uses_params(expr: str, params: dict[str, float]) -> bool:
    names = free_names(expr)
    return any(p in names for p in params)


def _fn_def(name: str, expr: str, params: dict[str, float]) -> list[str]:
    """A named function of x that reads the current scene-parameter values."""
    lines = [f"def {name}(x):"]
    for p in params:
        if uses_params(expr, {p: 0.0}):
            lines.append(f"    {p} = P[{p!r}].get_value()")
    lines.append(f"    return {expr}")
    return lines


def _define_object(obj: Obj, ink: str, scene: Scene) -> list[str]:
    """Python statements that construct the Mobject bound to ``obj.id``."""
    var = obj.id
    p = obj.payload

    if obj.kind == "equation":
        tex = p.get("tex", "")
        col = _color(obj.color, ink)
        # repr() already emits a valid Python literal with backslashes escaped;
        # adding an r-prefix on top would DOUBLE the backslashes (\int -> \\int,
        # a LaTeX line break), so pass repr() alone.
        return [f"{var} = MathTex({tex!r}, color={col}){_scale(obj)}{_pos(obj)}"]
    if obj.kind == "text":
        text = p.get("text", "")
        fs = float(p.get("font_size", 36))
        col = _color(obj.color, ink)
        return [f"{var} = Text({text!r}, font_size={fs}, color={col}){_scale(obj)}{_pos(obj)}"]
    if obj.kind == "shape":
        return _define_shape(obj, ink, scene)
    if obj.kind == "plot":
        return _define_plot(obj, ink, scene)
    if obj.kind in ("area", "riemann"):
        return _define_overlay(obj, scene)
    raise ValueError(f"unknown object kind: {obj.kind}")  # pragma: no cover


def _overlay_body(obj: Obj) -> str:
    p = obj.payload
    plot, fn = p["plot"], int(p.get("fn", 1))
    a, b = float(p.get("x_min", 0)), float(p.get("x_max", 1))
    col = _color(obj.color, "#5a8fd6")
    helper = f"_cc_helper_graph({plot}_ax, {plot}_f{fn}, {a}, {b})"
    if obj.kind == "area":
        op = float(p.get("opacity", 0.5))
        return f"{plot}_ax.get_area({helper}, x_range=[{a}, {b}], color={col}, opacity={op})"
    dx = float(p.get("dx", 0.1))
    # Manim uses np.arange(a, b, dx), which float rounding can extend by one
    # rectangle past b; stopping half a step early keeps exactly (b - a) / dx
    return (f"{plot}_ax.get_riemann_rectangles({helper}, x_range=[{a}, {b - dx / 2}], dx={dx}, "
            f"color={col}, fill_opacity=0.6, stroke_width=0.5, show_signed_area=False)")


def plot_is_dynamic(plot: Obj, params: dict[str, float]) -> bool:
    for fn in plot.payload.get("functions", []):
        expr = fn.get("expr", "x") if isinstance(fn, dict) else str(fn)
        if uses_params(expr, params):
            return True
    return False


def _define_overlay(obj: Obj, scene: Scene) -> list[str]:
    plot = scene.get_obj(obj.payload["plot"])
    body = _overlay_body(obj)
    if plot is not None and plot_is_dynamic(plot, scene.params):
        return [f"{obj.id} = always_redraw(lambda: {body})"]
    return [f"{obj.id} = {body}"]


def _define_shape(obj: Obj, ink: str, scene: Scene) -> list[str]:
    var = obj.id
    p = obj.payload
    shape = p.get("shape", "circle")
    size = float(p.get("size", 1.0))
    col = _color(obj.color, ink)
    if shape in ("line", "arrow") and "x2" in p:
        # explicit segment endpoints (any direction, e.g. a vertical jump)
        cls, buff = ("Line", "") if shape == "line" else ("Arrow", ", buff=0")
        return [f"{var} = {cls}([{p['x1']}, {p['y1']}, 0], [{p['x2']}, {p['y2']}, 0]"
                f"{buff}, color={col})"]
    if shape == "circle":
        body = f"Circle(radius={size / 2}, color={col})"
    elif shape == "square":
        body = f"Square(side_length={size}, color={col})"
    elif shape == "rectangle":
        w = float(p.get("width", size))
        h = float(p.get("height", size / 2))
        body = f"Rectangle(width={w}, height={h}, color={col})"
    elif shape == "triangle":
        body = f"Triangle(color={col}).scale({size})"
    elif shape == "dot":
        body = f"Dot(radius={max(size / 10, 0.05)}, color={col})"
    elif shape == "line":
        body = f"Line([{-size / 2}, 0, 0], [{size / 2}, 0, 0], color={col})"
    elif shape == "arrow":
        body = f"Arrow([{-size / 2}, 0, 0], [{size / 2}, 0, 0], buff=0, color={col})"
    else:
        body = f"Circle(radius={size / 2}, color={col})"

    body += _scale(obj)
    anchor = p.get("anchor")
    if not anchor:
        return [f"{var} = {body}{_pos(obj)}"]
    # Anchored in a plot's coordinates; follows scene parameters if it uses them.
    plot = anchor["plot"]
    xe, ye = str(anchor.get("x", "0")), str(anchor.get("y", "0"))
    lines = _fn_def(f"{var}_at_x", xe, scene.params) + _fn_def(f"{var}_at_y", ye, scene.params)
    where = f"{plot}_ax.c2p(float({var}_at_x(0)), float({var}_at_y(0)))"
    lines.append(f"{var} = {body}.move_to({where})")
    if uses_params(xe, scene.params) or uses_params(ye, scene.params):
        lines.append(f"{var}.add_updater(lambda m: m.move_to({where}))")
    return lines


def _define_plot(obj: Obj, ink: str, scene: Scene) -> list[str]:
    var = obj.id
    p = obj.payload
    ax = f"{var}_ax"
    x_range = list(p.get("x_range", [-5, 5, 1]))
    y_range = list(p.get("y_range", [-3, 3, 1]))
    width = float(p.get("width", 8.0))
    height = float(p.get("height", 5.0))
    extra = ""
    if p.get("show_numbers"):
        cfg = []
        for r in (x_range, y_range):
            step = float(r[2]) if len(r) > 2 else 1.0
            d = next((k for k in range(4) if abs(step * 10 ** k - round(step * 10 ** k)) < 1e-9), 3)
            # no thousands separators (2,000 reads as a decimal in many locales)
            cfg.append("{'include_numbers': True, 'font_size': 24, 'decimal_number_config': "
                       f"{{'num_decimal_places': {d}, 'group_with_commas': False}}}}")
        extra = f", x_axis_config={cfg[0]}, y_axis_config={cfg[1]}"
    lines = [
        f"{ax} = Axes(x_range={x_range}, y_range={y_range}, x_length={width}, "
        f"y_length={height}, axis_config={{'color': {_color(None, ink)}}}{extra})",
    ]
    members = [ax]
    for i, fn in enumerate(p.get("functions", []), start=1):
        is_d = isinstance(fn, dict)
        expr = fn.get("expr", "x") if is_d else str(fn)
        palette = curve_colors(ink)
        fcol = _color(fn.get("color") if is_d else None,
                      obj.color or palette[(i - 1) % len(palette)])
        smooth = fn.get("smooth") if is_d else None
        dx = fn.get("dx") if is_d else None
        fname, gvar = f"{var}_f{i}", f"{var}_g{i}"
        lines.extend(_fn_def(fname, expr, scene.params))
        call = f"_cc_plot({ax}, {fname}, {fcol}, smooth={smooth!r}, dx={dx!r})"
        if uses_params(expr, scene.params):
            lines.append(f"{gvar} = always_redraw(lambda: {call})")
        else:
            lines.append(f"{gvar} = {call}")
        members.append(gvar)
    xl, yl = p.get("x_label"), p.get("y_label")
    if xl or yl:
        ink_c = _color(None, ink)
        lines.append(f"{var}_labels = {ax}.get_axis_labels("
                     f"x_label=MathTex({(xl or '')!r}, color={ink_c}), "
                     f"y_label=MathTex({(yl or '')!r}, color={ink_c}))")
        members.append(f"{var}_labels")
    # the axes (not the axes + labels) are centred at the plot's position, so a
    # labelled and an unlabelled plot at the same x/y line up
    x, y = obj.position
    lines.append(f"{var} = VGroup({', '.join(members)}){_scale(obj)}")
    lines.append(f"{var}.shift(np.array([{x}, {y}, 0]) - {ax}.get_center())")
    return lines


def _introduce_anim(obj: Obj | None) -> str:
    """Pick a natural 'appear' animation for a target object."""
    if obj is not None and obj.kind == "plot":
        return "Create"  # with lag_ratio=0 below: axes and curves drawn together
    if obj is not None and obj.kind in ("shape", "area", "riemann"):
        return "Create"
    return "Write"


def _step_anims(scene: Scene, step: Step, accent: str = "#ff7eb6") -> list[str]:
    """The animation expressions of one step (each carries its own run_time)."""
    dur = step.duration
    if step.kind == "write":
        out = []
        for t in step.targets:
            o = scene.get_obj(t)
            lag = ", lag_ratio=0" if o is not None and o.kind == "plot" else ""
            out.append(f"{_introduce_anim(o)}({t}, run_time={dur}{lag})")
        return out
    if step.kind == "transform":
        if step.targets and step.to:
            return [f"Transform({step.targets[0]}, {step.to}, run_time={dur})"]
        return []
    if step.kind == "highlight":
        return [f"Indicate({t}, color={accent!r}, run_time={dur})" for t in step.targets]
    if step.kind == "fade":
        return [f"FadeOut({t}, run_time={dur})" for t in step.targets]
    if step.kind == "camera_move":
        # camera params live on the step's 'to' field encoded as "zoom@x,y"; zoom is
        # absolute (1 = the full frame) and an empty focus keeps the current centre.
        spec = step.to or "1@,"
        zoom, _, xy = spec.partition("@")
        x, _, y = xy.partition(",")
        z = float(zoom or 1)
        move = f".move_to([{float(x)}, {float(y)}, 0])" if x != "" and y != "" else ""
        return [f"self.camera.frame.animate(run_time={dur}).set_width(config.frame_width / {z})"
                f"{move}"]
    if step.kind == "tween" and step.param is not None and step.value is not None:
        return [f"P[{step.param!r}].animate(run_time={dur}, rate_func=linear)"
                f".set_value({float(step.value)})"]
    return []


END_HOLD = 0.5  # seconds the final state is held when the last step is an animation


def _step_lines(scene: Scene, accent: str = "#ff7eb6") -> list[str]:
    # Each group becomes one self.play(...): a step's targets animate together, and a
    # with_previous step joins the previous step's play.
    groups: list[list[str] | float] = []
    for step in scene.steps:
        if step.kind == "wait":
            groups.append(float(step.duration))
            continue
        anims = _step_anims(scene, step, accent)
        if not anims:
            continue
        if step.with_previous and groups and isinstance(groups[-1], list):
            groups[-1].extend(anims)
        else:
            groups.append(list(anims))
    lines = [f"self.wait({g})" if isinstance(g, float) else f"self.play({', '.join(g)})"
             for g in groups]
    if groups and not isinstance(groups[-1], float):
        # Manim draws an animation's frames up to (not including) its end state: a
        # scene ending on a play would never show it (half-written text, a tween
        # 1/fps short). Hold the final state briefly (Scene.duration counts it).
        lines.append(f"self.wait({END_HOLD})")
    return lines or ["self.wait(1)"]


def build_function_code(scene: Scene, theme_name: str) -> str:
    """``_cc_build()``: constructs every object and returns them keyed by name."""
    ink = theme(theme_name)["ink"]
    body: list[str] = ["P = {}"]
    for name, val in scene.params.items():
        body.append(f"P[{name!r}] = ValueTracker({float(val)})")
    for obj in scene.objects:
        body.extend(_define_object(obj, ink, scene))
    body.append("return dict(locals())")
    return "def _cc_build():\n" + "\n".join("    " + ln for ln in body) + "\n"


def generate_scene_code(scene: Scene, theme_name: str) -> str:
    body = ["ns = _cc_build()", "P = ns['P']"]
    body += [f"{o.id} = ns[{o.id!r}]" for o in scene.objects]
    body += _step_lines(scene, theme(theme_name)["accent"])
    indented = "\n".join("        " + line for line in body)
    return (
        f"class {scene.class_name()}(MovingCameraScene):\n"
        f"    def construct(self):\n"
        f"{indented}\n"
    )


def generate_module(scene: Scene, theme_name: str) -> str:
    """A complete, runnable Manim module for a single scene."""
    t = theme(theme_name)
    return (
        _PRELUDE.lstrip("\n")
        + f'\nconfig.background_color = "{t["background_color"]}"\n\n\n'
        + build_function_code(scene, theme_name)
        + "\n\n"
        + generate_scene_code(scene, theme_name)
    )
