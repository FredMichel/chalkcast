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

from .geometry import FRAME_HEIGHT, FRAME_WIDTH, SAFE_MARGIN, BBox
from .codegen import _COLOR_HEX, plot_is_dynamic, uses_params
from .expr import compile_fn
from .model import Obj, Scene
from .themes import DEFAULT_THEME, theme

COLLISION_THRESHOLD = 0.35   # estimated mode: overlap / smaller area
MEASURED_TEXT_OVERLAP = 0.10  # measured mode: text-on-text overlap that is a defect
MIN_FONT_SIZE = 20.0
FIDELITY_TOLERANCE = 0.02     # drawn-vs-true deviation, fraction of the y-axis span
OFFSCREEN_TOLERANCE = 0.02    # units beyond the frame edge before it is "off-screen"
MAX_CLIPPED = 0.5             # curve fraction outside y_range before it is a warning
TICK_FONT_SIZE = 24           # axis numbers (codegen's include_numbers font size)
MATHTEX_FONT_SIZE = 48        # Manim's MathTex default; scale multiplies it
MARGIN_TOLERANCE = 0.1        # measured intrusion into the safe margin worth noting (info)
MIN_CONTRAST = 1.6            # WCAG contrast ratio below which a color all but vanishes
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

    def states_while(self, oid: str, steps: set[int]) -> tuple[bool, list[dict[str, Any]]]:
        """Which measured states of ``oid`` are ever on screen, given the steps it is
        shown at: (its state at the declared parameter values?, tween states).

        The declared values hold until the first tween beat; a tween's in-between
        states during its beat; where it lands, until the next tween. A dot written
        only after a tween is never seen where it started, and a faded one may move
        off-screen unseen."""
        obj = self.scene.get_obj(oid)
        dynamic = obj is not None and is_dynamic(obj, self.scene)
        initial = not dynamic or not self.scene.steps or bool(steps & initial_window(self.scene))
        windows = _beat_windows(self.scene)
        seen = []
        for st in self._states:
            during, after = windows.get(int(st.get("step", 0)), ([], []))
            if steps & (set(during) | (set(after) if st.get("beat_end") else set())):
                seen.append(st)
        return initial, seen

    def undefined(self, oid: str, steps: set[int]) -> bool:
        """Measured with no finite point (e.g. a dot anchored at log(-2)) while shown."""
        if not self.measured:
            return False
        initial, states = self.states_while(oid, steps)
        rec = self._objs.get(oid)
        if initial and rec is not None and rec.get("bbox") is None:
            return True
        return any(oid in st["objects"] and st["objects"][oid] is None for st in states)

    def moving_points(self, oid: str) -> list[list[float]]:
        """Every position a moving curve/shape takes during the tweens (measured)."""
        return [p for st in self._states for p in st.get("ink", {}).get(oid, [])]

    def axes_points(self, plot_id: str) -> list[list[float]]:
        """The axes' own ink (lines, ticks, labels): it does not move with parameters."""
        return list(self._plots.get(plot_id, {}).get("axes_points", []))

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
    shape: dict[str, str] = {}  # a transformed object keeps its new shape, even re-written
    states: list[dict[str, str]] = []
    for st in scene.steps:
        if st.kind == "write":
            for t in st.targets:
                if t in defined:
                    visible[t] = shape.get(t, t)
        elif st.kind == "transform" and st.targets and st.to in defined:
            if st.targets[0] in visible:
                visible[st.targets[0]] = st.to  # Transform keeps the source, reshaped
                shape[st.targets[0]] = st.to
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


def _inside(pt: list[float], box: BBox, shrink: float = -0.02) -> bool:
    """Point within the box; by default the box is padded by about half a stroke
    width, since a curve drawn at that distance already touches the glyphs."""
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
        if only_step is not None and only_step not in [st.index for st in beat]:
            continue
        # Manim keeps one animation per mobject in a play(): the last one wins, so an
        # earlier one (a write, a transform's morph) silently never happens
        animated: dict[str, int] = {}
        for st in beat:
            if st.kind not in ("write", "highlight", "fade", "transform"):
                continue
            for tid in st.targets:
                if tid in animated:
                    out.append(f"steps {animated[tid]} and {st.index} both animate '{tid}' in the "
                               f"same beat: Manim keeps only the last (step {st.index}), so step "
                               f"{animated[tid]} would never show; make it a separate step")
                animated.setdefault(tid, st.index)
        cams = [st.index for st in beat if st.kind == "camera_move"]
        if len(cams) > 1:
            out.append(f"steps {cams} move the camera in the same beat: Manim keeps only the last; "
                       "combine them into one camera_move (zoom and focus together)")
        params = [st.param for st in beat if st.kind == "tween"]
        for dup in sorted({p for p in params if params.count(p) > 1}):
            out.append(f"parameter '{dup}' is tweened twice in the same beat (only the last "
                       "would act); chain the tweens as separate steps")
        tweened = set(params)
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


def _beat_windows(scene: Scene) -> dict[int, tuple[list[int], list[int]]]:
    """For each beat containing a tween, keyed by the beat's first tween step: the
    step indices of the beat, and those after it until the next tween beat."""
    beats = _beats(scene)
    out: dict[int, tuple[list[int], list[int]]] = {}
    for bi, beat in enumerate(beats):
        tweens = [st.index for st in beat if st.kind == "tween"]
        if not tweens:
            continue
        after: list[int] = []
        for later in beats[bi + 1:]:
            if any(st.kind == "tween" for st in later):
                break
            after += [st.index for st in later]
        out[tweens[0]] = ([st.index for st in beat], after)
    return out


def initial_window(scene: Scene) -> set[int]:
    """Steps played with every parameter at its declared value (before any tween)."""
    out: set[int] = set()
    for beat in _beats(scene):
        if any(st.kind == "tween" for st in beat):
            break
        out |= {st.index for st in beat}
    return out


def param_moments(scene: Scene, samples: int = 8) -> list[tuple[dict[str, float], set[int]]]:
    """Parameter values the timeline passes through, each with the steps during
    which they are on screen: the declared values, then samples of every tween."""
    vals = dict(scene.params)
    out = [(dict(vals), initial_window(scene) if scene.steps else {0})]
    windows = _beat_windows(scene)
    for beat in _beats(scene):
        tweens = [st for st in beat if st.kind == "tween"]
        if not tweens:
            continue
        during, after = windows[tweens[0].index]
        old = dict(vals)
        total = max(float(st.duration) for st in tweens)
        for k in range(1, samples + 1):
            v = dict(old)
            for st in tweens:  # each over its own duration, like the probe
                if st.param in v and st.value is not None:
                    a = min(1.0, k / samples * total / max(float(st.duration), 1e-9))
                    v[st.param] = old[st.param] + (float(st.value) - old[st.param]) * a
            out.append((v, set(during) | (set(after) if k == samples else set())))
            vals = v
    return out


def overlay_error(scene: Scene, plot: Obj, fn_index: int, x_min: float, x_max: float,
                  kind: str, params: dict[str, float], axis_only: bool = False) -> str | None:
    """An area/Riemann sum only shows its integral if the x-axis (y = 0) is on the
    axes and the function, at these parameter values, stays inside y_range over
    the interval."""
    import numpy as np
    y0, y1 = (float(v) for v in plot.payload.get("y_range", [-3, 3])[:2])
    if not y0 <= 0 <= y1:
        return (f"the {kind} is measured from y = 0, which is outside the y_range [{y0}, {y1}] "
                f"of '{plot.id}'; include 0 in y_range")
    fns = plot.payload.get("functions", [])
    if axis_only or not 1 <= fn_index <= len(fns):
        return None
    fn = fns[fn_index - 1]
    f = compile_fn(str(fn["expr"] if isinstance(fn, dict) else fn), params)
    xs = np.linspace(x_min, x_max, 401)
    with np.errstate(all="ignore"):
        ys = []
        for v in xs:
            try:
                ys.append(float(f(v, **params)))
            except Exception:  # noqa: BLE001 - outside the domain
                ys.append(float("nan"))
    ys = np.asarray(ys)
    fin = ys[np.isfinite(ys)]
    if fin.size == 0:
        return f"function {fn_index} of '{plot.id}' is undefined on [{x_min}, {x_max}]"
    if fin.size < ys.size:
        return (f"function {fn_index} of '{plot.id}' is undefined on part of [{x_min}, {x_max}]; "
                "restrict the interval to where it is defined")
    margin = 0.005 * (y1 - y0)
    if fin.min() < y0 - margin or fin.max() > y1 + margin:
        return (f"function {fn_index} of '{plot.id}' leaves y_range [{y0}, {y1}] on "
                f"[{x_min}, {x_max}] (it reaches {fin.min():.3g}..{fin.max():.3g}), so the {kind} "
                "would be clipped; widen y_range or narrow the interval")
    return None


def _lane_overlays(scene: Scene) -> list[dict[str, Any]]:
    """Areas/Riemann sums re-checked at every parameter value they are shown at:
    a redeclared parameter or a tween can push them into a state add_area refuses."""
    out = []
    vis = visible_views(scene)
    moments = param_moments(scene)
    for obj in scene.objects:
        if obj.kind not in ("area", "riemann") or obj.id not in vis:
            continue
        plot = scene.get_obj(obj.payload.get("plot", ""))
        if plot is None:
            continue
        steps = {i for i, _ in vis[obj.id]}
        kind = "area" if obj.kind == "area" else "Riemann sum"
        for vals, window in moments:
            if not (window & steps):
                continue
            e = overlay_error(scene, plot, int(obj.payload.get("fn", 1)), float(obj.payload["x_min"]),
                              float(obj.payload["x_max"]), kind, vals)
            if e:
                at = ", ".join(f"{k}={v:.3g}" for k, v in vals.items() if k in obj_params(obj, scene))
                out.append(_obl(scene.id, "range", "error",
                                f"{kind} '{obj.id}' no longer shows its integral"
                                f"{f' at {at}' if at else ''}: {e}",
                                "change the parameter's value/tween range, y_range or the interval",
                                target=obj.id))
                break
    for obj in scene.objects:
        a = obj.payload.get("anchor")
        plot = scene.get_obj(a.get("plot", "")) if a else None
        if plot is None or obj.id not in vis:
            continue
        steps = {i for i, _ in vis[obj.id]}
        (x0, x1), (y0, y1) = ((float(v) for v in plot.payload.get(k, d)[:2])
                              for k, d in (("x_range", [-5, 5]), ("y_range", [-3, 3])))
        import numpy as np
        fx, fy = compile_fn(str(a["x"]), scene.params), compile_fn(str(a["y"]), scene.params)
        for vals, window in moments:
            if not (window & steps):
                continue
            try:
                with np.errstate(all="ignore"):
                    px, py = float(fx(0.0, **vals)), float(fy(0.0, **vals))
            except Exception:  # noqa: BLE001 - undefined: reported by framing
                continue
            mx, my = 0.02 * (x1 - x0), 0.02 * (y1 - y0)
            if px == px and py == py and not (x0 - mx <= px <= x1 + mx and y0 - my <= py <= y1 + my):
                at = ", ".join(f"{k}={v:.3g}" for k, v in vals.items() if k in obj_params(obj, scene))
                out.append(_obl(scene.id, "range", "warning",
                                f"'{obj.id}' leaves the axes of '{plot.id}'{f' at {at}' if at else ''}: "
                                f"it is at ({px:.3g}, {py:.3g}), outside x_range [{x0}, {x1}] / "
                                f"y_range [{y0}, {y1}], detached from the drawn plot",
                                "widen the plot's ranges or limit the parameter", target=obj.id))
                break
    return out


def unused_tweens(scene: Scene) -> list[int]:
    used = set().union(*(obj_params(o, scene) for o in scene.objects)) if scene.objects else set()
    return [st.index for st in scene.steps if st.kind == "tween" and st.param in scene.params
            and st.param not in used]


DEFAULT_VIEW = BBox(cx=0.0, cy=0.0, width=FRAME_WIDTH, height=FRAME_HEIGHT)


def camera_by_step(scene: Scene) -> dict[int, tuple[BBox, BBox]]:
    """Camera frame before and after each step. camera_move zoom is absolute; a
    move without focus keeps the centre."""
    zoom, cx, cy = 1.0, 0.0, 0.0
    out: dict[int, tuple[BBox, BBox]] = {}
    for st in scene.steps:
        before = BBox(cx=cx, cy=cy, width=FRAME_WIDTH / zoom, height=FRAME_HEIGHT / zoom)
        if st.kind == "camera_move" and st.to:
            z, _, xy = st.to.partition("@")
            fx, _, fy = xy.partition(",")
            zoom = float(z or 1)
            if fx != "" and fy != "":
                cx, cy = float(fx), float(fy)
        out[st.index] = (before, BBox(cx=cx, cy=cy, width=FRAME_WIDTH / zoom, height=FRAME_HEIGHT / zoom))
    return out


def visible_views(scene: Scene) -> dict[str, list[tuple[int, BBox]]]:
    """For each shown geometry: (step index, camera frame after it) at every step
    it is on screen."""
    cams = camera_by_step(scene)
    out: dict[str, list[tuple[int, BBox]]] = {}
    for st, shown in zip(scene.steps, timeline(scene)):
        for g in dict.fromkeys(shown.values()):
            out.setdefault(g, []).append((st.index, cams[st.index][1]))
    return out


def smooth(t: float, inflection: float = 10.0) -> float:
    """Manim's default rate function (camera moves use it; tweens are linear)."""
    import math
    sig = lambda v: 1 / (1 + math.exp(-v))  # noqa: E731
    err = sig(-inflection / 2)
    return min(max((sig(inflection * (t - 0.5)) - err) / (1 - 2 * err), 0.0), 1.0)


def _lerp_view(a: BBox, b: BBox, f: float) -> BBox:
    return BBox(cx=a.cx + (b.cx - a.cx) * f, cy=a.cy + (b.cy - a.cy) * f,
                width=a.width + (b.width - a.width) * f, height=a.height + (b.height - a.height) * f)


def _framing_entries(scene: Scene, geo: "Geometry", oid: str,
                     seen: list[tuple[int, BBox]]) -> list[tuple[BBox, list[BBox]]]:
    """Each measured box of ``oid`` with the camera frames it is on screen under:
    its first state under the views of the steps before any tween (all its steps if
    it never moves); a tween frame under the camera at that moment (a camera_move
    in the same beat is interpolated); where a tween lands, under the views of
    the steps until the next tween."""
    steps = {i for i, _ in seen}
    end_view = dict(seen)
    obj = scene.get_obj(oid)
    dynamic = obj is not None and is_dynamic(obj, scene)
    initial, states = geo.states_while(oid, steps)
    out: list[tuple[BBox, list[BBox]]] = []
    box = geo.bbox(oid)
    if initial and box:
        init_steps = steps & initial_window(scene) if dynamic and scene.steps else steps
        out.append((box, [end_view[i] for i in sorted(init_steps)] or [DEFAULT_VIEW]))
    windows, cams = _beat_windows(scene), camera_by_step(scene)
    by_index = {s.index: s for s in scene.steps}
    for st in states:
        b = geo._box(st["objects"].get(oid))
        if not b:
            continue
        during, after = windows.get(int(st.get("step", 0)), ([], []))
        views: list[BBox] = []
        if any(i in steps for i in during):
            cam = [by_index[i] for i in during if by_index[i].kind == "camera_move" and by_index[i].to]
            if cam:
                tween_t = max(by_index[i].duration for i in during if by_index[i].kind == "tween")
                c = cam[-1]
                # frac is the time fraction of the beat; the camera eases over its own run_time
                f = smooth(min(1.0, float(st.get("frac", 1.0)) * tween_t / max(c.duration, 1e-9)))
                views.append(_lerp_view(cams[during[0]][0], cams[c.index][1], f))
            else:
                views += [end_view[i] for i in during if i in steps]
        if st.get("beat_end"):
            views += [end_view[i] for i in after if i in steps]
        if views:
            out.append((b, views))
    return out


def overflow(box: BBox, view: BBox) -> float:
    """How far ``box`` reaches beyond ``view`` (negative: clearance inside it)."""
    return max(box.right - view.right, view.left - box.left, box.top - view.top,
               view.bottom - box.bottom)


def view_margin(view: BBox) -> float:
    """The safe margin, scaled with the camera (it is a fraction of the screen)."""
    return SAFE_MARGIN * view.width / FRAME_WIDTH


def _where(view: BBox) -> str:
    if view == DEFAULT_VIEW:
        return "off-screen"
    return (f"outside the camera's view (zoom {FRAME_WIDTH / view.width:.3g}, centre "
            f"{view.cx:.3g},{view.cy:.3g})")


def _lane_framing(scene: Scene, geo: Geometry) -> list[dict[str, Any]]:
    """Each object must be fully inside the camera frame at some moment it is on
    screen: zooming in on a detail may crop others, and a zoom-out may reveal
    what was written outside the view, but something never fully seen is a defect."""
    out = []
    shown = _shown_geoms(scene) if scene.steps else [o.id for o in scene.objects]
    vis = visible_views(scene)
    for oid in shown:
        obj = scene.get_obj(oid)
        if obj is None:
            continue
        seen = vis.get(oid) or [(0, DEFAULT_VIEW)]
        steps = {i for i, _ in seen}
        views = list({(v.cx, v.cy, v.width, v.height): v for _, v in seen}.values())
        if not geo.measured:
            cx, cy = obj.position
            centre = BBox(cx=cx, cy=cy, width=0.0, height=0.0)
            best = min(views, key=lambda v: overflow(centre, v))
            # A center outside the frame is definitely wrong -> error.
            if overflow(centre, best) > 0:
                out.append(_obl(scene.id, "framing", "error",
                                f"object '{oid}' is positioned {_where(best)} (center at "
                                f"{cx:.1f},{cy:.1f})", "move it (or the camera) back inside the frame",
                                target=oid, view=[best.cx, best.cy, best.width, best.height]))
            # bbox estimates (font/LaTeX width unknown) only ever warn.
            elif obj.kind not in ("area", "riemann") and not any(
                    overflow(obj.estimate_bbox(), v) <= -view_margin(v) + 1e-6 for v in views):
                out.append(_obl(scene.id, "framing", "warning",
                                f"object '{oid}' may extend outside the safe area (estimated)",
                                "nudge it toward the center or reduce its scale if the render "
                                "confirms it", target=oid))
            continue
        if obj.kind == "shape" and geo.undefined(oid, steps):
            out.append(_obl(scene.id, "framing", "error",
                            f"shape '{oid}' has no finite position while shown (its anchor is "
                            "undefined there, e.g. log of a negative number) and is not drawn",
                            "change its anchor expression or the parameter range", target=oid))
            continue
        # every measured position must be fully seen at some moment it is shown
        over, best = None, DEFAULT_VIEW
        for b, vs in _framing_entries(scene, geo, oid, seen):
            v = min(vs, key=lambda v: overflow(b, v))
            if over is None or overflow(b, v) > over:
                over, best = overflow(b, v), v
        if over is None:
            continue
        if over > OFFSCREEN_TOLERANCE:
            out.append(_obl(scene.id, "framing", "error",
                            f"object '{oid}' extends {over:.2f} units {_where(best)} (measured)",
                            "move it toward the center or reduce its scale (fix_scene can do "
                            "this), or move the camera", target=oid, overflow=round(over, 3),
                            view=[best.cx, best.cy, best.width, best.height]))
        elif over > MARGIN_TOLERANCE - view_margin(best):
            # inside the frame but near its edge: worth knowing, not a defect of a GIF
            out.append(_obl(scene.id, "framing", "info",
                            f"object '{oid}' reaches {over + view_margin(best):.2f} units into the "
                            f"{view_margin(best):.2g}-unit safe margin (measured)",
                            "nudge it toward the center", target=oid))
    return out


def _pair_finding(scene: Scene, geo: Geometry, a: Obj, b: Obj,
                  initial: bool = True) -> tuple[str, str] | None:
    """(severity, summary) if the two shown shapes genuinely collide, else None.
    ``initial``: whether they are on screen together at the declared parameter
    values (geometry here is measured there; tween states are checked separately)."""
    if a.kind in ("area", "riemann") or b.kind in ("area", "riemann"):
        return None  # fills sit on their plot by design
    static_only = not initial and (is_dynamic(a, scene) or is_dynamic(b, scene))
    # never together where the moving one started: only what does not move (a
    # plot's axes and labels) is checked here; the moving parts per tween state
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
        if static_only and other.kind != "plot":
            return None
        pts = (geo.axes_points(other.id) if static_only else geo.curve_points(other.id)) \
            if other.kind == "plot" else geo.ink(other.id)
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
        if static_only:
            ba, bb = xa or ba, xb or bb  # the axes stay put; curves move
        ov = ba.overlap_area(bb)
        ratio = ov / min(ba.area, bb.area) if ov > 0 else 0
        if ratio >= MEASURED_TEXT_OVERLAP:
            return ("warning", f"plots '{a.id}' and '{b.id}' overlap without sharing axes (measured)")
    return None  # shape/shape and shape/plot: annotations and composite drawings


def _lane_collision(scene: Scene, geo: Geometry) -> list[dict[str, Any]]:
    out, seen = [], set()
    states = timeline(scene) if scene.steps else [{o.id: o.id for o in scene.objects}]
    init = initial_window(scene) if scene.steps else {1}
    together: dict[tuple[str, str], bool] = {}  # pair -> together at the declared values?
    shown_as: dict[str, set[str]] = {}  # transform destination -> the object showing it
    for st in states:
        for oid, g in st.items():
            if oid != g:
                shown_as.setdefault(g, set()).add(oid)
    for n, st in enumerate(states, 1):
        geoms = list(dict.fromkeys(st.values()))
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                key = tuple(sorted((geoms[i], geoms[j])))
                together[key] = together.get(key, False) or n in init
    for key, initial in together.items():
        seen.add(key)
        a, b = scene.get_obj(key[0]), scene.get_obj(key[1])
        if a is None or b is None:
            continue
        found = _pair_finding(scene, geo, a, b, initial)
        if found:
            sev, summary = found
            for g in key:
                if g in shown_as:
                    summary += f" ('{g}' is on screen as {', '.join(repr(s) for s in sorted(shown_as[g]))})"
            out.append(_obl(scene.id, "collision", sev, summary,
                            "move one of them, or stagger their steps so they are not "
                            "visible together", target=key[1], other=key[0]))
    # moving curves/shapes: check every measured moment of every tween against what
    # is on screen during that beat -- and, for where the beat leaves them, against
    # everything shown until the next tween moves them again
    tl = timeline(scene) if scene.steps else []
    windows = _beat_windows(scene)
    crossed: set[tuple[str, str]] = set()
    for st in geo.states:
        start = int(st.get("step", 0))
        if start not in windows:
            continue
        during, after = windows[start]
        steps = during + (after if st.get("beat_end") else [])
        for mover, pts in st.get("ink", {}).items():
            shown: dict[str, str] = {}
            for i in steps:
                if mover in tl[i - 1].values():
                    shown.update(tl[i - 1])
            for tid in dict.fromkeys(shown.values()):  # the shape each label shows
                tobj = scene.get_obj(tid)
                if tobj is None or tobj.kind not in TEXTLIKE or (tid, mover) in crossed:
                    continue
                tbox = geo.bbox(tid)
                if tbox and any(_inside(p, tbox) for p in pts):
                    crossed.add((tid, mover))
                    out.append(_obl(scene.id, "collision", "warning",
                                    f"'{tid}' is crossed by '{mover}' during or after the tween "
                                    f"of step {start} (measured)",
                                    "move the label out of the path, or limit the tween's range",
                                    target=tid, other=mover, step=start))
    return out


def zoom_while_visible(scene: Scene) -> dict[str, float]:
    """Smallest camera zoom (1 = full frame) under which each object is on screen."""
    out: dict[str, float] = {}
    cams = camera_by_step(scene)
    for st, shown in zip(scene.steps, timeline(scene)):
        zoom = FRAME_WIDTH / cams[st.index][1].width
        for g in shown.values():  # a transform destination is shown through its source
            out[g] = min(out.get(g, zoom), zoom)
    return out


def _luminance(color: str) -> float | None:
    h = _COLOR_HEX.get(color.upper(), color).lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        return None
    try:
        rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    except ValueError:
        return None
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float | None:
    """WCAG contrast ratio of two colors (1 = identical, 21 = black on white)."""
    la, lb = _luminance(a), _luminance(b)
    if la is None or lb is None:
        return None
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _explicit_colors(obj: Obj) -> list[tuple[str, str]]:
    """(what, color) for every color the author set on a stroked/text object."""
    if obj.kind in ("area", "riemann"):
        return []  # translucent fills over the board, judged by their curve
    out = [(f"'{obj.id}'", obj.color)] if obj.color else []
    if obj.kind == "plot":
        for i, f in enumerate(obj.payload.get("functions", []), 1):
            if isinstance(f, dict) and f.get("color"):
                out.append((f"curve {i} of '{obj.id}'", str(f["color"])))
    return out


def _lane_legibility(scene: Scene, geo: Geometry, theme_name: str = DEFAULT_THEME) -> list[dict[str, Any]]:
    out = []
    zoom = zoom_while_visible(scene)  # a zoomed-out camera shrinks everything on screen
    bg = theme(theme_name)["background_color"]
    for obj in scene.objects:
        for what, col in _explicit_colors(obj):
            c = contrast(col, bg)
            if c is not None and c < MIN_CONTRAST:
                out.append(_obl(scene.id, "legibility", "warning",
                                f"{what} is {col} on the {theme_name} background {bg} (contrast "
                                f"{c:.2f}:1): nearly invisible", "pick a color that contrasts with "
                                "the theme, or omit color to use the theme's palette", target=obj.id))
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
        if obj.kind == "plot" and (obj.payload.get("x_label") or obj.payload.get("y_label")):
            fs = MATHTEX_FONT_SIZE * obj.scale * z
            if fs < MIN_FONT_SIZE:
                out.append(_obl(scene.id, "legibility", "warning",
                                f"axis labels of '{obj.id}' render at ~{fs:.0f}pt (below "
                                f"{MIN_FONT_SIZE:.0f}pt)", "increase the plot's scale", target=obj.id))
        if obj.kind == "plot" and obj.payload.get("show_numbers"):
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
    shown_by: dict[str, str] = {}  # transform destination -> the object that took its shape
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
                if shown_by.get(tid) in on_screen:
                    out.append(_obl(scene.id, "timing", "warning",
                                    f"step {step.index} writes '{tid}', which is already on screen "
                                    f"as '{shown_by[tid]}' (transformed into it): a duplicate is drawn",
                                    f"drop the step, or fade '{shown_by[tid]}' first",
                                    step=step.index, target=tid))
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
                if tid in defined and tid not in on_screen and shown_by.get(tid) in on_screen:
                    out.append(_obl(scene.id, "timing", "error",
                                    f"step {step.index} ({step.kind}) targets '{tid}', which is on "
                                    f"screen only as '{shown_by[tid]}' (transformed into it)",
                                    f"target '{shown_by[tid]}' instead", step=step.index))
                elif tid in defined and tid not in introduced:
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
            if step.kind == "transform" and step.targets and step.targets[0] in on_screen and step.to:
                shown_by[step.to] = step.targets[0]
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
    # Areas, Riemann sums and anchored shapes are drawn from their plot's axes but
    # are separate objects: fading (or never writing) the plot leaves them floating.
    floating: set[str] = set()
    for st in timeline(scene):
        on = set(st) | set(st.values())
        for oid in st:
            o = scene.get_obj(st[oid])
            parent = (o.payload.get("plot") or (o.payload.get("anchor") or {}).get("plot")) if o else None
            if parent and parent not in on and oid not in floating:
                floating.add(oid)
                out.append(_obl(scene.id, "timing", "warning",
                                f"'{oid}' is on screen without its plot '{parent}' (faded or not "
                                "written yet): it floats without axes",
                                f"fade '{oid}' together with '{parent}', or write '{parent}' first",
                                target=oid))
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
    vis = visible_views(scene)
    for obj in scene.objects:
        if obj.kind != "plot":
            continue
        if scene.steps and obj.id not in vis:
            continue  # never on screen (the timing lane says so)
        # the curve as drawn while on screen: at the declared values and/or where tweens land
        initial, seen = geo.states_while(obj.id, {i for i, _ in vis.get(obj.id, [(0, None)])})
        states = [c for st in seen for c in st.get("curves", {}).get(obj.id, [])]
        drawn = (list(geo.curves(obj.id)) if initial else []) + states
        worst: dict[int, dict[str, Any]] = {}
        for c in drawn:
            i = c.get("fn")
            w = worst.setdefault(i, dict(c))
            for key in ("max_error", "clipped_fraction"):
                if float(c.get(key, 0.0)) > float(w.get(key, 0.0)):
                    w[key] = c[key]
                    if key == "max_error":
                        w["at_x"] = c.get("at_x")
        # where it draws nothing: at the declared values (None) or where a tween lands
        undefined: dict[int, list[dict[str, float] | None]] = {}
        for c in (geo.curves(obj.id) if initial else []):
            if c.get("defined_fraction") == 0.0:
                undefined.setdefault(c.get("fn"), []).append(None)
        for st in seen:
            for c in st.get("curves", {}).get(obj.id, []):
                if c.get("defined_fraction") == 0.0:
                    undefined.setdefault(c.get("fn"), []).append(st.get("params", {}))
        for i, where in sorted(undefined.items()):
            if None in where:
                rng.append(_obl(scene.id, "range", "error",
                                f"curve {i} of '{obj.id}' draws nothing: its function is undefined on "
                                "the whole x_range (measured)", "change x_range to where the function "
                                "is defined, or fix the expression", target=obj.id, fn=i))
                continue
            at = ", ".join(f"{k}={v:.3g}" for k, v in where[0].items() if k in obj_params(obj, scene))
            rng.append(_obl(scene.id, "range", "error",
                            f"curve {i} of '{obj.id}' draws nothing at {at}, where a tween lands: its "
                            "function is undefined on the whole x_range there (measured)",
                            "stop the tween short of that value (e.g. 0.01 instead of 0), or rewrite "
                            "the expression so it is defined there", target=obj.id, fn=i))
        for c in worst.values():
            i = c.get("fn")
            if i in undefined:
                continue
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
                  measurement: dict[str, Any] | None = None,
                  theme_name: str = DEFAULT_THEME) -> dict[str, Any]:
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
    def legibility(sc: Scene, g: Geometry) -> list[dict[str, Any]]:
        return _lane_legibility(sc, g, theme_name)
    lanes = [("legibility", legibility)] if compile_failed else \
        [("framing", _lane_framing), ("collision", _lane_collision), ("legibility", legibility)]
    for lane, fn in lanes:
        lanes_run.append(lane)
        obligations.extend(fn(scene, geo))
    lanes_run.append("timing")
    obligations.extend(_lane_timing(scene))
    lanes_run.append("range")
    obligations.extend(_lane_overlays(scene))
    if geo.measured:
        lanes_run += ["fidelity"]
        fid, rng = _lane_fidelity(scene, geo)
        obligations += fid + rng

    lanes_skipped = ["intent"] + ([] if geo.measured else ["fidelity"]) + \
        (["framing", "collision"] if compile_failed else [])
    if geo.measured:
        limitations = ["geometry measured from the constructed Manim objects (exact boxes "
                       "and drawn curves; framing checked against the camera view while each "
                       "object is shown)",
                       "intent lane (vision) is not available yet"]
    else:
        limitations = ["bounding boxes are estimated from the scene graph, not measured; "
                       "framing/collision are approximate and capped at warnings",
                       "fidelity lane and curve clipping need Manim (geometry probe) and were skipped",
                       "shapes anchored to a plot are not placed in estimated geometry",
                       "intent lane (vision) is not available yet"]

    # one cause, one finding: a dot that leaves the frame has left its axes too
    off = {o.get("target") for o in obligations if o["lane"] == "framing" and o["severity"] == "error"}
    obligations = [o for o in obligations
                   if not ("leaves the axes" in o["summary"] and o.get("target") in off)]
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
