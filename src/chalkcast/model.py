"""The Chalkcast scene graph.

A ``Project`` holds ordered ``Scene`` objects. A ``Scene`` holds a bag of visual
``Obj`` nodes (equation / plot / text / shape) and an ordered ``Step`` timeline.
The agent mutates this graph through the authoring tools; rendering compiles it
to Manim and verification reasons over it.

Every object carries an *estimated* bounding box in Manim units so the geometric
verification lanes can run without rendering. The estimates are deliberately
rough (font metrics are unknown pre-render) and callers surface that in
``limitations``.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from .geometry import BBox

_LATEX_CMD = re.compile(r"\\[a-zA-Z]+")


def latex_visible_len(tex: str) -> int:
    """Rough count of *rendered* glyphs in a LaTeX string: drop \\commands and
    structural characters so width estimates don't balloon on \\frac, \\sin, {}."""
    s = _LATEX_CMD.sub("x", tex)          # each command ~ one glyph-ish
    s = re.sub(r"[{}$^_&~\\ ]", "", s)     # braces, scripts, spaces, stray backslashes
    return max(len(s), 1)

# ---------------------------------------------------------------------------
# Visual objects
# ---------------------------------------------------------------------------

ObjKind = Literal["equation", "plot", "text", "shape", "area", "riemann"]
ShapeKind = Literal["circle", "square", "rectangle", "triangle", "arrow", "dot", "line"]


@dataclass
class Obj:
    id: str
    kind: ObjKind
    position: tuple[float, float] = (0.0, 0.0)
    color: str | None = None
    scale: float = 1.0
    payload: dict[str, Any] = field(default_factory=dict)

    # ---- bounding-box estimation (Manim units) ----------------------------
    def estimate_bbox(self) -> BBox:
        cx, cy = self.position
        w, h = self._estimate_size()
        return BBox(cx=cx, cy=cy, width=w * self.scale, height=h * self.scale)

    def _estimate_size(self) -> tuple[float, float]:
        if self.kind == "equation":
            # LaTeX width is unknown until typeset; estimate from visible glyphs
            # (ignoring \commands and braces) so \frac/\sin don't inflate it.
            tex = str(self.payload.get("tex", ""))
            n = latex_visible_len(tex)
            return (min(0.33 * n, 12.0), 0.9)
        if self.kind == "text":
            text = str(self.payload.get("text", ""))
            font_size = float(self.payload.get("font_size", 36))
            # ~0.0075 units per pt per char (calibrated against rendered Text so
            # the framing lane doesn't false-positive); line height ~font_size*0.016.
            return (max(0.0075 * font_size * max(len(text), 1), 0.4), font_size * 0.016)
        if self.kind == "plot":
            return (
                float(self.payload.get("width", 8.0)),
                float(self.payload.get("height", 5.0)),
            )
        if self.kind in ("area", "riemann"):
            # overlays its parent plot; bbox copied from the plot at authoring time
            return (
                float(self.payload.get("width", 6.0)),
                float(self.payload.get("height", 4.0)),
            )
        if self.kind == "shape":
            shape = self.payload.get("shape", "circle")
            size = float(self.payload.get("size", 1.0))
            if shape in ("line", "arrow"):
                return (size, 0.2)
            return (size, size)
        return (1.0, 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "position": list(self.position),
            "color": self.color,
            "scale": self.scale,
            "payload": self.payload,
        }


# ---------------------------------------------------------------------------
# Timeline steps
# ---------------------------------------------------------------------------

StepKind = Literal["write", "transform", "highlight", "fade", "camera_move", "wait"]

# Steps that bring an object on screen for the first time.
INTRODUCING = {"write"}


@dataclass
class Step:
    index: int
    kind: StepKind
    targets: list[str] = field(default_factory=list)
    to: str | None = None  # for transform: target object id to morph into
    duration: float = 1.0
    note: str = ""  # declared intent, consumed by the (future) vision lane

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "targets": self.targets,
            "to": self.to,
            "duration": self.duration,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Scene / Project
# ---------------------------------------------------------------------------


@dataclass
class Scene:
    id: str
    title: str = ""
    objects: list[Obj] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    _obj_counter: Any = field(default_factory=lambda: itertools.count(1), repr=False)

    def class_name(self) -> str:
        # scene:0003 -> Scene0003 (a valid Python identifier for codegen).
        suffix = self.id.split(":")[-1]
        return "Scene" + "".join(c for c in suffix if c.isalnum()).zfill(4)

    def new_obj_id(self, kind: str) -> str:
        return f"{kind}_{next(self._obj_counter)}"

    def get_obj(self, obj_id: str) -> Obj | None:
        return next((o for o in self.objects if o.id == obj_id), None)

    def add_step(self, kind: StepKind, **kw: Any) -> Step:
        step = Step(index=len(self.steps) + 1, kind=kind, **kw)
        self.steps.append(step)
        return step

    def duration(self) -> float:
        return sum(s.duration for s in self.steps) or 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "class_name": self.class_name(),
            "objects": [o.to_dict() for o in self.objects],
            "steps": [s.to_dict() for s in self.steps],
            "duration": self.duration(),
        }


@dataclass
class Project:
    id: str
    quality: Literal["draft", "low", "medium", "high"] = "low"
    fps: int = 30
    theme: str = "chalkboard"
    scenes: list[Scene] = field(default_factory=list)
    obligations: list[dict[str, Any]] = field(default_factory=list)
    _scene_counter: Any = field(default_factory=lambda: itertools.count(1), repr=False)
    _obl_counter: Any = field(default_factory=lambda: itertools.count(1), repr=False)

    def new_scene(self, title: str = "") -> Scene:
        sid = f"scene:{next(self._scene_counter):04d}"
        scene = Scene(id=sid, title=title)
        self.scenes.append(scene)
        return scene

    def get_scene(self, scene_id: str | None) -> Scene | None:
        if scene_id is None:
            return self.scenes[-1] if self.scenes else None
        return next((s for s in self.scenes if s.id == scene_id), None)

    def next_obligation_id(self) -> str:
        return f"obl-{next(self._obl_counter):04d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "quality": self.quality,
            "fps": self.fps,
            "theme": self.theme,
            "scenes": [s.to_dict() for s in self.scenes],
            "open_obligations": [o for o in self.obligations if o["status"] == "open"],
        }
