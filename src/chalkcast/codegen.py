"""Compile a Chalkcast scene graph into runnable Manim Community code.

The generated module defines one ``MovingCameraScene`` subclass per scene so that
``camera_move`` steps work uniformly. Object ids double as Python variable names
(they are valid identifiers by construction in ``model.py``).
"""

from __future__ import annotations

from .model import Obj, Scene
from .themes import theme

_KNOWN_COLORS = {
    "WHITE", "BLACK", "RED", "GREEN", "BLUE", "YELLOW", "ORANGE", "PURPLE",
    "PINK", "TEAL", "GRAY", "GREY", "GOLD", "MAROON", "LIGHT_GRAY", "DARK_GRAY",
    "BLUE_A", "BLUE_B", "BLUE_C", "BLUE_D", "BLUE_E", "GREEN_A", "GREEN_E",
    "RED_A", "RED_E", "YELLOW_A", "YELLOW_E",
}


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


def _define_object(obj: Obj, ink: str) -> list[str]:
    """Python statements that construct the Mobject bound to ``obj.id``."""
    var = obj.id
    p = obj.payload
    lines: list[str] = []

    if obj.kind == "equation":
        tex = p.get("tex", "")
        col = _color(obj.color, ink)
        # repr() already emits a valid Python literal with backslashes escaped;
        # adding an r-prefix on top would DOUBLE the backslashes (\int -> \\int,
        # a LaTeX line break), so pass repr() alone.
        lines.append(f"{var} = MathTex({tex!r}, color={col}){_scale(obj)}{_pos(obj)}")

    elif obj.kind == "text":
        text = p.get("text", "")
        fs = float(p.get("font_size", 36))
        col = _color(obj.color, ink)
        lines.append(
            f"{var} = Text({text!r}, font_size={fs}, color={col}){_scale(obj)}{_pos(obj)}"
        )

    elif obj.kind == "shape":
        lines.append(_define_shape(obj, ink))

    elif obj.kind == "plot":
        lines.extend(_define_plot(obj, ink))

    elif obj.kind == "area":
        lines.append(_define_area(obj))

    elif obj.kind == "riemann":
        lines.append(_define_riemann(obj))

    else:  # pragma: no cover - guarded by the authoring layer
        raise ValueError(f"unknown object kind: {obj.kind}")

    return lines


def _define_area(obj: Obj) -> str:
    p = obj.payload
    plot = p["plot"]
    fn = int(p.get("fn", 1))
    a, b = p.get("x_min", 0), p.get("x_max", 1)
    col = _color(obj.color, "#5a8fd6")
    op = float(p.get("opacity", 0.5))
    return (
        f"{obj.id} = {plot}_ax.get_area({plot}_g{fn}, x_range=[{a}, {b}], "
        f"color={col}, opacity={op})"
    )


def _define_riemann(obj: Obj) -> str:
    p = obj.payload
    plot = p["plot"]
    fn = int(p.get("fn", 1))
    a, b = p.get("x_min", 0), p.get("x_max", 1)
    dx = float(p.get("dx", 0.1))
    col = _color(obj.color, "#5a8fd6")
    return (
        f"{obj.id} = {plot}_ax.get_riemann_rectangles({plot}_g{fn}, "
        f"x_range=[{a}, {b}], dx={dx}, color={col}, fill_opacity=0.6, stroke_width=0.5)"
    )


def _define_shape(obj: Obj, ink: str) -> str:
    var = obj.id
    p = obj.payload
    shape = p.get("shape", "circle")
    size = float(p.get("size", 1.0))
    col = _color(obj.color, ink)
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
        body = f"Arrow([{-size / 2}, 0, 0], [{size / 2}, 0, 0], color={col})"
    else:
        body = f"Circle(radius={size / 2}, color={col})"
    return f"{var} = {body}{_pos(obj)}"


def _define_plot(obj: Obj, ink: str) -> list[str]:
    var = obj.id
    p = obj.payload
    ax = f"{var}_ax"
    x_range = p.get("x_range", [-5, 5, 1])
    y_range = p.get("y_range", [-3, 3, 1])
    width = float(p.get("width", 8.0))
    height = float(p.get("height", 5.0))
    lines = [
        f"{ax} = Axes(x_range={list(x_range)}, y_range={list(y_range)}, "
        f"x_length={width}, y_length={height}, axis_config={{'color': {_color(None, ink)}}})",
    ]
    members = [ax]
    for i, fn in enumerate(p.get("functions", []), start=1):
        expr = fn.get("expr", "x") if isinstance(fn, dict) else str(fn)
        fcol = _color(fn.get("color") if isinstance(fn, dict) else None, obj.color or "#f6c76b")
        gvar = f"{var}_g{i}"
        lines.append(f"{gvar} = {ax}.plot(lambda x: {expr}, color={fcol})")
        members.append(gvar)
    lines.append(f"{var} = VGroup({', '.join(members)}){_scale(obj)}{_pos(obj)}")
    return lines


def _introduce_anim(obj: Obj | None) -> str:
    """Pick a natural 'appear' animation for a target object."""
    if obj is not None and obj.kind in ("shape", "plot", "area", "riemann"):
        return "Create"
    return "Write"


def _step_lines(scene: Scene) -> list[str]:
    lines: list[str] = []
    for step in scene.steps:
        dur = step.duration
        if step.kind == "write":
            for tid in step.targets:
                anim = _introduce_anim(scene.get_obj(tid))
                lines.append(f"self.play({anim}({tid}), run_time={dur})")
        elif step.kind == "transform":
            if step.targets and step.to:
                lines.append(
                    f"self.play(Transform({step.targets[0]}, {step.to}), run_time={dur})"
                )
        elif step.kind == "highlight":
            for tid in step.targets:
                lines.append(f"self.play(Indicate({tid}), run_time={dur})")
        elif step.kind == "fade":
            for tid in step.targets:
                lines.append(f"self.play(FadeOut({tid}), run_time={dur})")
        elif step.kind == "camera_move":
            # payload carried on the note-less step via targets[0] being unused;
            # camera params live on the step's 'to' field encoded as "zoom@x,y".
            spec = step.to or "1@0,0"
            zoom, _, xy = spec.partition("@")
            x, _, y = xy.partition(",")
            z = float(zoom or 1)
            x = float(x or 0)
            y = float(y or 0)
            lines.append(
                f"self.play(self.camera.frame.animate.scale({1 / z}).move_to([{x}, {y}, 0]), run_time={dur})"
            )
        elif step.kind == "wait":
            lines.append(f"self.wait({dur})")
    if not lines:
        lines.append("self.wait(1)")
    return lines


def generate_scene_code(scene: Scene, theme_name: str) -> str:
    t = theme(theme_name)
    ink = t["ink"]
    body: list[str] = []
    for obj in scene.objects:
        body.extend(_define_object(obj, ink))
    body.extend(_step_lines(scene))

    indented = "\n".join("        " + line for line in body)
    return (
        f"class {scene.class_name()}(MovingCameraScene):\n"
        f"    def construct(self):\n"
        f"{indented}\n"
    )


def generate_module(scene: Scene, theme_name: str) -> str:
    """A complete, runnable Manim module for a single scene."""
    t = theme(theme_name)
    header = (
        "from manim import *\n\n"
        f'config.background_color = "{t["background_color"]}"\n\n'
    )
    return header + generate_scene_code(scene, theme_name)
