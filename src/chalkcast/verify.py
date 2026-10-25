"""Verification lanes (Layer C).

Two evidence levels, always reported in ``coverage.measured``:

* **measured** -- a geometry probe (``chalkcast.probe``) built the real Manim
  objects: bounding boxes are exact (LaTeX and fonts typeset), curve points are
  the drawn ones, and the ``fidelity`` lane compares each drawn curve with its
  true function. Measured findings are reliable, so they can be ``error``s.
* **estimated** -- no render engine available: sizes are guessed from the scene
  graph, so estimate-only findings are capped at ``warning`` and never block
  export.

Collision and framing are *timeline-aware*: an object is only checked while it
is actually on screen (a ``transform`` destination is never shown by itself; the
source takes its shape), and intentional overlays (curves sharing axes,
annotations on a plot, area/Riemann fills) are not collisions.

Each problem becomes an *obligation* (SPEC §7): ``error`` blocks export.
"""

from __future__ import annotations

from typing import Any

from .geometry import FRAME_HEIGHT, FRAME_WIDTH, SAFE_MARGIN, BBox, within_safe_area
from .codegen import plot_is_dynamic, uses_params
from .model import Obj, Scene

COLLISION_THRESHOLD = 0.35   # estimated mode: overlap / smaller area
MEASURED_TEXT_OVERLAP = 0.10  # measured mode: text-on-text overlap that is a defect
MIN_FONT_SIZE = 20.0
FIDELITY_TOLERANCE = 0.02     # drawn-vs-true deviation, fraction of the y-axis span
OFFSCREEN_TOLERANCE = 0.02    # units beyond the frame edge before it is "off-screen"
MAX_CLIPPED = 0.5             # curve fraction outside y_range before it is a warning
TICK_FONT_SIZE = 24           # axis numbers (codegen's include_numbers font size)
MATHTEX_FONT_SIZE = 48        # Manim's MathTex default; scale multiplies it
MARGIN_TOLERANCE = 0.1        # measured intrusion into the safe margin worth noting (info)
TEXTLIKE = ("text", "equation")

GEOMETRIC_LANES = ["framing", "collision", "legibility", "timing", "fidelity", "range"]


def _obl(scene_id: str, lane: str, severity: str, summary: str, fix: str,
         step: int | None = None, target: str | None = None, **extra: Any) -> dict[str, Any]:
    o = {
        "scene": scene_id,
        "step": step,
        "lane": lane,
        "severity": severity,
        "summary": summary,
        "suggested_fix": fix,
        "status": "open",
    }
    if target is not None:
        o["target"] = target
    o.update(extra)
    return o


# ---------------------------------------------------------------------------
# geometry access
# ---------------------------------------------------------------------------

class Geometry:
    """Bounding boxes and ink for objects: measured when a probe ran, else estimated."""

    def __init__(self, scene: Scene, measurement: dict[str, Any] | None):
        self.scene = scene
        m = measurement if (measurement and measurement.get("ok") is True) else None
        self.measured = m is not None
        self._objs = (m or {}).get("objects", {})
        self._plots = (m or {}).get("plots", {})
        self._ink = (m or {}).get("ink", {})
        self._states = (m or {}).get("states", [])
        self.states = self._states

    @staticmethod
    def _box(lrbt: list[float] | None) -> BBox | None:
        if not lrbt:
            return None
        l, r, b, t = lrbt
        return BBox(cx=(l + r) / 2, cy=(b + t) / 2, width=r - l, height=t - b)

    def bbox(self, oid: str) -> BBox | None:
        if self.measured:
            rec = self._objs.get(oid)
            return self._box(rec["bbox"]) if rec else None
        obj = self.scene.get_obj(oid)
        return obj.estimate_bbox() if obj else None

    def all_boxes(self, oid: str) -> list[BBox]:
        """Initial box plus the box at every parameter state the timeline visits."""
        out = [b for b in [self.bbox(oid)] if b]
        for st in self._states:
            b = self._box(st["objects"].get(oid))
            if b:
                out.append(b)
        return out

    def curve_points(self, plot_id: str) -> list[list[float]]:
        """Everything a label must not sit on: the curves and the axes' own ink."""
        rec = self._plots.get(plot_id, {})
        pts: list[list[float]] = list(rec.get("axes_points", []))
        for c in rec.get("curves", []):
            pts.extend(c.get("points", []))
        return pts

    def axes_box(self, plot_id: str) -> BBox | None:
        return self._box(self._plots.get(plot_id, {}).get("axes_bbox"))

    def ink(self, oid: str) -> list[list[float]]:
        return self._ink.get(oid, [])

    def curves(self, plot_id: str) -> list[dict[str, Any]]:
        return self._plots.get(plot_id, {}).get("curves", [])


def timeline(scene: Scene) -> list[dict[str, str]]:
    """Visible objects after each step, as {object id: id of the shape it shows}."""
    defined = {o.id for o in scene.objects}
    visible: dict[str, str] = {}
    states: list[dict[str, str]] = []
    for st in scene.steps:
        if st.kind == "write":
            for t in st.targets:
                if t in defined:
                    visible[t] = t
        elif st.kind == "transform" and st.targets and st.to in defined:
            if st.targets[0] in visible:
                visible[st.targets[0]] = st.to  # Transform keeps the source, reshaped
        elif st.kind == "fade":
            for t in st.targets:
                visible.pop(t, None)
        states.append(dict(visible))
    return states


def _shown_geoms(scene: Scene) -> list[str]:
    seen: list[str] = []
    for st in timeline(scene):
        for g in st.values():
            if g not in seen:
                seen.append(g)
    return seen


def _inside(pt: list[float], box: BBox, shrink: float = 0.02) -> bool:
    return (box.left + shrink <= pt[0] <= box.right - shrink
            and box.bottom + shrink <= pt[1] <= box.top - shrink)


# ---------------------------------------------------------------------------
# lanes
# ---------------------------------------------------------------------------

def is_dynamic(obj: Obj, scene: Scene) -> bool:
    """Redrawn every frame from a scene parameter (curve, its fill, or an anchor)."""
    if obj.kind == "plot":
        return plot_is_dynamic(obj, scene.params)
    if obj.kind in ("area", "riemann"):
        parent = scene.get_obj(obj.payload.get("plot", ""))
        return parent is not None and plot_is_dynamic(parent, scene.params)
    a = obj.payload.get("anchor")
    return bool(a) and (uses_params(str(a.get("x", "")), scene.params) or
                        uses_params(str(a.get("y", "")), scene.params))


def obj_params(obj: Obj | None, scene: Scene) -> set[str]:
    """Scene parameters an object is redrawn from."""
    if obj is None:
        return set()
    if obj.kind in ("area", "riemann"):
        return obj_params(scene.get_obj(obj.payload.get("plot", "")), scene)
    exprs: list[str] = []
    if obj.kind == "plot":
        exprs = [str(f.get("expr", "") if isinstance(f, dict) else f)
                 for f in obj.payload.get("functions", [])]
    elif obj.payload.get("anchor"):
        a = obj.payload["anchor"]
        exprs = [str(a.get("x", "")), str(a.get("y", ""))]
    return {p for p in scene.params if any(uses_params(e, {p: 0.0}) for e in exprs)}


def _beats(scene: Scene) -> list[list]:
    beats: list[list] = []
    for st in scene.steps:
        if st.with_previous and beats:
            beats[-1].append(st)
        else:
            beats.append([st])
    return beats


def beat_conflicts(scene: Scene, only_step: int | None = None) -> list[str]:
    """A parameter-driven object written/highlighted/faded in the same beat as a tween
    of its parameter: Manim suspends its redraw during that animation, so it freezes
    and then jumps."""
    out = []
    for beat in _beats(scene):
        tweened = {st.param for st in beat if st.kind == "tween"}
        if not tweened or (only_step is not None and only_step not in [st.index for st in beat]):
            continue
        for st in beat:
            if st.kind not in ("write", "highlight", "fade"):
                continue
            for tid in st.targets:
                hit = obj_params(scene.get_obj(tid), scene) & tweened
                if hit:
                    out.append(f"step {st.index} ({st.kind}s '{tid}') shares a beat with a tween of "
                               f"{sorted(hit)}, which '{tid}' is drawn from: it would freeze during "
                               "the animation and then jump; put the tween in its own step")
    return out


def unused_tweens(scene: Scene) -> list[int]:
    used = set().union(*(obj_params(o, scene) for o in scene.objects)) if scene.objects else set()
    return [st.index for st in scene.steps if st.kind == "tween" and st.param in scene.params
            and st.param not in used]


def _camera_views(scene: Scene) -> dict[int, BBox]:
    """Camera frame (absolute zoom, kept centre) at each write step, when zoomed."""
    zoom, cx, cy, views = 1.0, 0.0, 0.0, {}
    for st in scene.steps:
        if st.kind == "camera_move" and st.to:
            z, _, xy = st.to.partition("@")
            fx, _, fy = xy.partition(",")
            zoom = float(z or 1)
            if fx != "" and fy != "":
                cx, cy = float(fx), float(fy)
        elif st.kind == "write" and (zoom != 1.0 or cx or cy):
            views[st.index] = BBox(cx=cx, cy=cy, width=FRAME_WIDTH / zoom, height=FRAME_HEIGHT / zoom)
    return views


def _lane_framing(scene: Scene, geo: Geometry) -> list[dict[str, Any]]:
    out = []
    for idx, view in _camera_views(scene).items():
        st = scene.steps[idx - 1]
        for tid in st.targets:
            b = geo.bbox(tid)
            if b and (b.left < view.left or b.right > view.right or
                      b.bottom < view.bottom or b.top > view.top):
                out.append(_obl(scene.id, "framing", "warning",
                                f"step {idx} writes '{tid}' outside the camera's current view",
                                "move the camera (camera_move) or the object", step=idx, target=tid))
    hw, hh = FRAME_WIDTH / 2, FRAME_HEIGHT / 2
    shown = _shown_geoms(scene) if scene.steps else [o.id for o in scene.objects]
    for oid in shown:
        obj = scene.get_obj(oid)
        if obj is None:
            continue
        if not geo.measured:
            cx, cy = obj.position
            # A center outside the frame is definitely wrong -> error.
            if abs(cx) > hw or abs(cy) > hh:
                out.append(_obl(scene.id, "framing", "error",
                                f"object '{oid}' is positioned off-screen (center at {cx:.1f},{cy:.1f})",
                                "move its position back inside the frame", target=oid))
            # bbox estimates (font/LaTeX width unknown) only ever warn.
            elif obj.kind not in ("area", "riemann") and not within_safe_area(obj.estimate_bbox()):
                out.append(_obl(scene.id, "framing", "warning",
                                f"object '{oid}' may extend outside the safe area (estimated)",
                                "nudge it toward the center or reduce its scale if the render "
                                "confirms it", target=oid))
            continue
        boxes = geo.all_boxes(oid)
        if not boxes:
            continue
        over = max(max(b.right - hw, -hw - b.left, b.top - hh, -hh - b.bottom) for b in boxes)
        if over > OFFSCREEN_TOLERANCE:
            out.append(_obl(scene.id, "framing", "error",
                            f"object '{oid}' extends {over:.2f} units off-screen (measured)",
                            "move it toward the center or reduce its scale (fix_scene can do this)",
                            target=oid, overflow=round(over, 3)))
        elif over > MARGIN_TOLERANCE - SAFE_MARGIN:
            # inside the frame but near its edge: worth knowing, not a defect of a GIF
            out.append(_obl(scene.id, "framing", "info",
                            f"object '{oid}' reaches {over + SAFE_MARGIN:.2f} units into the "
                            f"{SAFE_MARGIN}-unit safe margin (measured)",
                            "nudge it toward the center", target=oid))
    return out


def _pair_finding(scene: Scene, geo: Geometry, a: Obj, b: Obj) -> tuple[str, str] | None:
    """(severity, summary) if the two shown shapes genuinely collide, else None."""
    if a.kind in ("area", "riemann") or b.kind in ("area", "riemann"):
        return None  # fills sit on their plot by design
    ka, kb = a.kind, b.kind
    ba, bb = geo.bbox(a.id), geo.bbox(b.id)
    if ba is None or bb is None:
        return None
    if not geo.measured:
        if ka == "plot" and kb == "plot" and a.position == b.position and \
                a.payload.get("width") == b.payload.get("width") and \
                a.payload.get("height") == b.payload.get("height"):
            return None  # curves sharing the same axes
        if ka == "shape" and kb == "shape":
            return None  # composite drawings are made of touching shapes
        if "plot" in (ka, kb) and "shape" in (ka, kb):
            return None  # annotations on a plot
        ov = ba.overlap_area(bb)
        ratio = ov / min(ba.area, bb.area) if ov > 0 else 0
        if ratio >= COLLISION_THRESHOLD:
            return ("warning", f"'{a.id}' and '{b.id}' overlap (~{ratio * 100:.0f}% of the smaller, estimated)")
        return None

    if ka in TEXTLIKE and kb in TEXTLIKE:
        ov = ba.overlap_area(bb)
        ratio = ov / min(ba.area, bb.area) if ov > 0 else 0
        if ratio >= MEASURED_TEXT_OVERLAP:
            return ("error", f"'{a.id}' and '{b.id}' overlap (~{ratio * 100:.0f}% of the smaller, measured)")
        return None
    if ka in TEXTLIKE or kb in TEXTLIKE:
        txt, other = (a, b) if ka in TEXTLIKE else (b, a)
        tbox = geo.bbox(txt.id)
        pts = geo.curve_points(other.id) if other.kind == "plot" else geo.ink(other.id)
        hits = sum(1 for p in pts if _inside(p, tbox))
        if hits:
            what = "the curves/axes of" if other.kind == "plot" else "the outline of"
            return ("warning", f"'{txt.id}' is crossed by {what} '{other.id}' (measured)")
        return None
    if ka == "plot" and kb == "plot":
        xa, xb = geo.axes_box(a.id), geo.axes_box(b.id)
        if xa and xb and all(abs(u - v) < 0.05 for u, v in
                             ((xa.left, xb.left), (xa.right, xb.right), (xa.bottom, xb.bottom), (xa.top, xb.top))):
            return None  # curves sharing the same axes
        ov = ba.overlap_area(bb)
        ratio = ov / min(ba.area, bb.area) if ov > 0 else 0
        if ratio >= MEASURED_TEXT_OVERLAP:
            return ("warning", f"plots '{a.id}' and '{b.id}' overlap without sharing axes (measured)")
    return None  # shape/shape and shape/plot: annotations and composite drawings


def _lane_collision(scene: Scene, geo: Geometry) -> list[dict[str, Any]]:
    out, seen = [], set()
    states = timeline(scene) if scene.steps else [{o.id: o.id for o in scene.objects}]
    for st in states:
        geoms = list(dict.fromkeys(st.values()))
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                key = tuple(sorted((geoms[i], geoms[j])))
                if key in seen:
                    continue
                seen.add(key)
                a, b = scene.get_obj(key[0]), scene.get_obj(key[1])
                if a is None or b is None:
                    continue
                found = _pair_finding(scene, geo, a, b)
                if found:
                    sev, summary = found
                    out.append(_obl(scene.id, "collision", sev, summary,
                                    "move one of them, or stagger their steps so they are not "
                                    "visible together", target=key[1], other=key[0]))
    # moving curves/shapes: check every measured moment of every tween
    tl = timeline(scene) if scene.steps else []
    for st in geo.states:
        idx = int(st.get("step", 0)) - 1
        if not 0 <= idx < len(tl):
            continue
        shown = tl[idx]
        for mover, pts in st.get("ink", {}).items():
            if mover not in shown.values():
                continue
            for tid in shown:
                tobj = scene.get_obj(tid)
                if tobj is None or tobj.kind not in TEXTLIKE or (tid, mover) in seen:
                    continue
                tbox = geo.bbox(tid)
                if tbox and any(_inside(p, tbox) for p in pts):
                    seen.add((tid, mover))
                    out.append(_obl(scene.id, "collision", "warning",
                                    f"'{tid}' is crossed by '{mover}' while step {idx + 1} "
                                    "animates it (measured)",
                                    "move the label out of the path, or limit the tween's range",
                                    target=tid, other=mover, step=idx + 1))
    return out


def _zoom_at_write(scene: Scene) -> dict[str, float]:
    """Camera zoom when each object is first written (1 = full frame)."""
    zoom, out = 1.0, {}
    for st in scene.steps:
        if st.kind == "camera_move" and st.to:
            zoom = float(st.to.partition("@")[0] or 1)
        elif st.kind == "write":
            for t in st.targets:
                out.setdefault(t, zoom)
    return out


def _lane_legibility(scene: Scene, geo: Geometry) -> list[dict[str, Any]]:
    out = []
    zoom = _zoom_at_write(scene)  # a zoomed-out camera shrinks everything on screen
    for obj in scene.objects:
        z = zoom.get(obj.id, 1.0)
        if obj.kind == "text":
            fs = float(obj.payload.get("font_size", 36)) * obj.scale * z
            if fs < MIN_FONT_SIZE:
                out.append(_obl(scene.id, "legibility", "warning",
                                f"text '{obj.id}' renders at ~{fs:.0f}pt (below {MIN_FONT_SIZE:.0f}pt)",
                                "increase font_size or scale", target=obj.id))
        elif obj.kind == "equation":
            # glyph size is what matters: a tall fraction can hide tiny symbols
            # (not the box height: a lone '=' or '-' is short at any size)
            fs = MATHTEX_FONT_SIZE * obj.scale * z
            if fs < MIN_FONT_SIZE:
                out.append(_obl(scene.id, "legibility", "warning",
                                f"equation '{obj.id}' is too small to read (~{fs:.0f}pt)",
                                "increase its scale", target=obj.id))
        elif obj.kind == "plot" and obj.payload.get("show_numbers"):
            fs = TICK_FONT_SIZE * obj.scale * z
            if fs < MIN_FONT_SIZE:
                out.append(_obl(scene.id, "legibility", "warning",
                                f"tick numbers of '{obj.id}' render at ~{fs:.0f}pt (below "
                                f"{MIN_FONT_SIZE:.0f}pt)", "increase the plot's scale or drop "
                                "show_numbers", target=obj.id))
    return out


def _lane_timing(scene: Scene) -> list[dict[str, Any]]:
    out = []
    defined = {o.id for o in scene.objects}
    if scene.objects and not scene.steps:
        out.append(_obl(scene.id, "timing", "error",
                        "the scene has objects but no steps, so nothing will be shown",
                        "add write steps for the objects to display"))
        return out
    introduced: set[str] = set()
    on_screen: set[str] = set()
    for step in scene.steps:
        # Every referenced target must be a defined object, or codegen emits an
        # undefined variable and the render crashes with a NameError.
        for tid in step.targets:
            if tid not in defined:
                out.append(_obl(scene.id, "timing", "error",
                                f"step {step.index} ({step.kind}) targets unknown object '{tid}'",
                                "use an object id returned by an add_* call", step=step.index))
        if step.kind == "write":
            for tid in step.targets:
                if tid in on_screen:
                    out.append(_obl(scene.id, "timing", "warning",
                                    f"step {step.index} writes '{tid}', which is already on screen "
                                    "(it will be erased and redrawn)",
                                    "use highlight to draw attention to it, or drop the step",
                                    step=step.index, target=tid))
            introduced.update(t for t in step.targets if t in defined)
            on_screen.update(t for t in step.targets if t in defined)
            continue
        if step.kind == "transform":
            for oid in [*step.targets, step.to]:
                o = scene.get_obj(oid) if oid else None
                if o is not None and is_dynamic(o, scene):
                    out.append(_obl(scene.id, "timing", "error",
                                    f"step {step.index} transforms '{oid}', which follows a scene "
                                    "parameter: it snaps back after the transform",
                                    "fade one out and write the other instead", step=step.index))
        if step.kind == "tween" and step.param not in scene.params:
            out.append(_obl(scene.id, "timing", "error",
                            f"step {step.index} tweens undeclared parameter '{step.param}'",
                            "declare it with add_parameter first", step=step.index))
        # transform/highlight/fade must target an object already on screen.
        if step.kind in ("highlight", "fade", "transform"):
            for tid in step.targets:
                if tid in defined and tid not in introduced:
                    out.append(_obl(scene.id, "timing", "error",
                                    f"step {step.index} ({step.kind}) targets '{tid}' before it is introduced",
                                    f"add a 'write' step for '{tid}' first", step=step.index))
                elif tid in defined and tid not in on_screen and step.kind in ("highlight", "transform"):
                    out.append(_obl(scene.id, "timing", "error",
                                    f"step {step.index} ({step.kind}) targets '{tid}' after it was "
                                    "faded out; this brings it back on screen",
                                    f"write '{tid}' again first, or drop the step", step=step.index))
            if step.kind == "transform" and step.to in on_screen:
                out.append(_obl(scene.id, "timing", "warning",
                                f"step {step.index} transforms into '{step.to}', which is already on "
                                "screen: two copies of it will be visible",
                                f"fade '{step.to}' first, or transform into a fresh object",
                                step=step.index))
            if step.kind == "fade":
                on_screen.difference_update(step.targets)
            if step.kind == "transform" and step.to and step.to not in defined:
                out.append(_obl(scene.id, "timing", "error",
                                f"step {step.index} transforms into undefined object '{step.to}'",
                                f"create object '{step.to}' before the transform", step=step.index))
    for msg in beat_conflicts(scene):
        out.append(_obl(scene.id, "timing", "error", msg, "put the tween in its own step"))
    for idx in unused_tweens(scene):
        out.append(_obl(scene.id, "timing", "warning",
                        f"step {idx} tweens '{scene.steps[idx - 1].param}', which no object uses: "
                        "nothing moves", "use the parameter in an expression or anchor, or drop "
                        "the step", step=idx))
    # Objects that are defined but never appear are almost always a forgotten step.
    if scene.steps:
        used = set()
        for st in scene.steps:
            used.update(st.targets)
            if st.to:
                used.add(st.to)
        for o in scene.objects:
            if o.id not in used:
                out.append(_obl(scene.id, "timing", "warning",
                                f"object '{o.id}' is never shown (no step uses it)",
                                f"add a write step for '{o.id}', or remove_object it", target=o.id))
    return out


def _lane_fidelity(scene: Scene, geo: Geometry) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fid, rng = [], []
    for obj in scene.objects:
        if obj.kind != "plot":
            continue
        # the curve as first drawn, then where each tween leaves it
        states = [c for st in geo.states for c in st.get("curves", {}).get(obj.id, [])]
        worst: dict[int, dict[str, Any]] = {}
        for c in list(geo.curves(obj.id)) + states:
            i = c.get("fn")
            w = worst.setdefault(i, dict(c))
            for key in ("max_error", "clipped_fraction"):
                if float(c.get(key, 0.0)) > float(w.get(key, 0.0)):
                    w[key] = c[key]
                    if key == "max_error":
                        w["at_x"] = c.get("at_x")
        for c in worst.values():
            i = c.get("fn")
            err = float(c.get("max_error", 0.0))
            if err > FIDELITY_TOLERANCE:
                at = c.get("at_x")
                rng_txt = f" near x={at:.3g}" if isinstance(at, (int, float)) else ""
                fid.append(_obl(scene.id, "fidelity", "error",
                                f"curve {i} of '{obj.id}' deviates from its function by "
                                f"{err * 100:.1f}% of the y-span{rng_txt} (measured)",
                                "sample it more finely (smaller dx) and/or set smooth=false for "
                                "discontinuous functions (fix_scene can do this)",
                                target=obj.id, fn=i, error=round(err, 4)))
            clipped = float(c.get("clipped_fraction", 0.0))
            if clipped > 0.005:
                # mostly off the axes is almost never intended (e.g. a default y_range)
                rng.append(_obl(scene.id, "range", "warning" if clipped > MAX_CLIPPED else "info",
                                f"curve {i} of '{obj.id}' leaves y_range on {clipped * 100:.0f}% of "
                                "the x-range and is clipped to the axes (measured)",
                                "widen y_range if the clipped part matters", target=obj.id, fn=i))
    return fid, rng


def inspect_scene(scene: Scene, render_report: dict[str, Any] | None = None,
                  measurement: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the verification lanes.

    ``measurement`` is a probe report (``render.probe_scene``); without it, or if
    the probe could not run, geometry is estimated and capped at warnings.
    """
    obligations: list[dict[str, Any]] = []
    lanes_run: list[str] = []
    geo = Geometry(scene, measurement)

    compile_failed = measurement is not None and measurement.get("ok") is False
    if measurement is not None and measurement.get("timeout"):
        obligations.append(_obl(scene.id, "measure", "warning",
                                str(measurement.get("error")) + "; geometry is estimated",
                                "reduce sampling cost (fewer/shorter tweens, narrower ranges) "
                                "to get measured verification"))
    if render_report is not None or compile_failed:
        lanes_run.append("compile")
    if compile_failed:
        obligations.append(_obl(scene.id, "compile", "error",
                                "scene construction failed: " + str(measurement.get("error", "unknown")),
                                "fix the reported LaTeX/expression error (see traceback)",
                                traceback=measurement.get("traceback", "")[-1200:]))
    if render_report is not None and not render_report.get("ok", False):
        obligations.append(_obl(scene.id, "compile", "error",
                                "render did not complete: " + str(render_report.get("error", "unknown")),
                                "fix the reported Manim/LaTeX error"))

    # Until the scene compiles there is no geometry to measure, and estimates of a
    # scene that is about to change are noise: report the compile error alone.
    lanes = [("legibility", _lane_legibility)] if compile_failed else \
        [("framing", _lane_framing), ("collision", _lane_collision), ("legibility", _lane_legibility)]
    for lane, fn in lanes:
        lanes_run.append(lane)
        obligations.extend(fn(scene, geo))
    lanes_run.append("timing")
    obligations.extend(_lane_timing(scene))
    if geo.measured:
        lanes_run += ["fidelity", "range"]
        fid, rng = _lane_fidelity(scene, geo)
        obligations += fid + rng

    lanes_skipped = ["intent"] + ([] if geo.measured else ["fidelity", "range"]) + \
        (["framing", "collision"] if compile_failed else [])
    if geo.measured:
        limitations = ["geometry measured from the constructed Manim objects (exact boxes "
                       "and drawn curves; checked against the default camera frame)",
                       "intent lane (vision) is not available yet"]
    else:
        limitations = ["bounding boxes are estimated from the scene graph, not measured; "
                       "framing/collision are approximate and capped at warnings",
                       "fidelity/range lanes need Manim (geometry probe) and were skipped",
                       "intent lane (vision) is not available yet"]

    errors = [o for o in obligations if o["severity"] == "error"]
    warnings = [o for o in obligations if o["severity"] == "warning"]
    score = max(0.0, 1.0 - 0.25 * len(errors) - 0.1 * len(warnings))
    status = "needs_fix" if errors else ("pass_with_warnings" if warnings else "pass")
    return {
        "verdict": {"status": status, "score": round(score, 2),
                    "errors": len(errors), "warnings": len(warnings)},
        "obligations": obligations,
        "lanes_run": lanes_run,
        "lanes_skipped": lanes_skipped,
        "measured": geo.measured,
        "limitations": limitations,
    }
