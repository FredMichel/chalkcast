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

import math
from typing import Any

from .geometry import FRAME_HEIGHT, FRAME_WIDTH, SAFE_MARGIN, BBox
from .codegen import _COLOR_HEX, plot_is_dynamic, uses_params
from .expr import compile_fn
from .model import Obj, Scene, tween_fractions
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
MIN_TEXT_CONTRAST = 2.5       # text needs more than a stroke to be read (GREEN on white: 1.9)
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
        self._fills = (m or {}).get("fills", {})
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

    def number_boxes(self, plot_id: str) -> list[BBox]:
        return [b for b in (self._box(x) for x in self._plots.get(plot_id, {}).get("number_boxes", [])) if b]

    def glyph_boxes(self, oid: str) -> list[BBox]:
        return [b for b in (self._box(x) for x in self._objs.get(oid, {}).get("glyphs", [])) if b]

    def glyph_pt(self, oid: str) -> float | None:
        return self._objs.get(oid, {}).get("glyph_pt")

    def named_label_boxes(self, plot_id: str) -> list[tuple[str, BBox]]:
        rec = self._plots.get(plot_id, {})
        names = rec.get("label_names") or ["x_label", "y_label"]
        return [(n, b) for n, raw in zip(names, rec.get("label_boxes", [])) if (b := self._box(raw))]

    def label_boxes(self, plot_id: str) -> list[BBox]:
        return [b for b in (self._box(x) for x in self._plots.get(plot_id, {}).get("label_boxes", [])) if b]

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

    def fill_cover(self, fill_id: str, box: BBox) -> float | None:
        """Fraction of ``box`` inside the filled region (polygons, even-odd), or None
        when the fill's outline was not measured."""
        polys = self._fills.get(fill_id)
        if not polys:
            return None
        nx, ny = 9, 5
        pts = [(box.left + (i + 0.5) * box.width / nx, box.bottom + (j + 0.5) * box.height / ny)
               for i in range(nx) for j in range(ny)]
        return sum(any(_in_polygon(x, y, pg) for pg in polys) for x, y in pts) / len(pts)

    def ink(self, oid: str) -> list[list[float]]:
        return self._ink.get(oid, [])

    def curves(self, plot_id: str) -> list[dict[str, Any]]:
        return self._plots.get(plot_id, {}).get("curves", [])


def _in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        (xi, yi), (xj, yj) = poly[i], poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


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


def _not_real(scene: Scene) -> list[dict[str, Any]]:
    """A fractional power of a negative number is complex in Python (NaN through
    np.power): that part of a curve is not drawn, at any parameter value it reaches."""
    import numpy as np
    out = []
    shown = {g for st in timeline(scene) for g in st.values()} if scene.steps else set()
    for obj in scene.objects:
        if obj.kind != "plot" or obj.id not in shown:
            continue
        lo, hi = (float(v) for v in obj.payload.get("x_range", [-5, 5])[:2])
        xs = [lo + (hi - lo) * (k + 0.5) / 24 for k in range(24)]
        for i, f in enumerate(obj.payload.get("functions", []), 1):
            expr = str(f.get("expr", "x") if isinstance(f, dict) else f)
            if "**" not in expr and "power" not in expr:
                continue
            try:
                fn = compile_fn(expr, scene.params)
            except Exception:  # noqa: BLE001
                continue
            on = {i for i, st in enumerate(timeline(scene), 1) if obj.id in st.values()}
            for vals, steps in param_moments(scene):
                if not (set(steps) & on):
                    continue  # the plot is not on screen at these values
                bad = []
                for xv in xs:
                    try:
                        with np.errstate(all="ignore"):
                            y = fn(xv, **vals)
                    except Exception:  # noqa: BLE001 - poles are another matter
                        continue
                    if np.iscomplexobj(y) or ("power" in expr and np.all(np.isnan(y))):
                        bad.append(xv)
                if bad:
                    at = ", ".join(f"{k}={v:.3g}" for k, v in vals.items() if uses_params(expr, {k: 0.0}))
                    out.append(_obl(scene.id, "range", "warning",
                                    f"curve {i} of '{obj.id}' ('{expr}') is not real for x in about "
                                    f"[{min(bad):.3g}, {max(bad):.3g}]" + (f" when {at}" if at else "")
                                    + " (a fractional power of a negative number): that part is "
                                    "not drawn",
                                    "for the real cube root write cbrt(x); for |x|^p write "
                                    "abs(x)**p, or narrow x_range", target=obj.id))
                    break
    return out


def param_moments(scene: Scene) -> list[tuple[dict[str, float], set[int]]]:
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
        fracs = tween_fractions()
        for k, frac in enumerate(fracs, 1):
            v = dict(old)
            for st in tweens:  # each over its own duration, like the probe
                if st.param in v and st.value is not None:
                    a = min(1.0, frac * total / max(float(st.duration), 1e-9))
                    v[st.param] = old[st.param] + (float(st.value) - old[st.param]) * a
            out.append((v, set(during) | (set(after) if k == len(fracs) else set())))
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


def _still_steps(scene: Scene) -> list[dict[str, Any]]:
    """Tweens and camera moves that change nothing on screen."""
    out: list[dict[str, Any]] = []
    unused = set(unused_tweens(scene))
    states = timeline(scene)
    steps = scene.steps
    value = dict(scene.params)
    cams = camera_by_step(scene)
    for n, st in enumerate(steps):
        if st.kind == "camera_move":
            before, after = cams.get(st.index, (None, None))
            if before is not None and before == after:
                out.append(_obl(scene.id, "timing", "warning",
                                f"the camera_move at step {st.index} does not change the view "
                                "(zoom is absolute, and the focus is already there): nothing moves",
                                "change its zoom or focus, or drop the step", step=st.index))
        if st.kind != "tween" or st.param not in value or st.value is None:
            continue
        cur, value[st.param] = value[st.param], float(st.value)
        if st.index in unused:
            continue
        if abs(cur - float(st.value)) < 1e-12:
            out.append(_obl(scene.id, "timing", "warning",
                            f"step {st.index} tweens '{st.param}' to {cur:g}, the value it already "
                            "has: nothing moves", "tween to another value, or use a wait",
                            step=st.index))
            continue
        # on screen during the beat (a with_previous write joins the same play)
        end = n
        while end + 1 < len(steps) and steps[end + 1].with_previous:
            end += 1
        start = n
        while start > 0 and steps[start].with_previous:
            start -= 1
        on = set()
        for k in range(max(start - 1, 0), end + 1):
            on |= set(states[k].values())
        ever = {g for s_ in states for g in s_.values()}
        users = {o.id for o in scene.objects if st.param in obj_params(o, scene)}
        if not (users & ever):
            continue  # its users are never shown at all: that warning names this tween
        if not any(st.param in obj_params(scene.get_obj(g), scene) for g in on):
            out.append(_obl(scene.id, "timing", "warning",
                            f"step {st.index} tweens '{st.param}' while nothing that uses it is on "
                            "screen: the motion is not seen",
                            "write the object(s) using it before the tween, or drop the step",
                            step=st.index))
    return out


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


def _base_views(scene: Scene) -> dict[int, tuple[BBox, BBox]]:
    """Before/after each step, the last camera frame that was not zoomed in.
    Zooming in on a detail is meant to crop the rest; what it crops is judged in
    the frame the scene zoomed in from."""
    cams, base, out = camera_by_step(scene), DEFAULT_VIEW, {}
    for st in scene.steps:
        before = base
        after = cams[st.index][1]
        if after.width >= FRAME_WIDTH - 1e-9:
            base = after
        out[st.index] = (before, base)
    return out


def _zoom_in_start(scene: Scene) -> dict[int, int | None]:
    """After each step: the step at which the current zoomed-in stretch began."""
    cams, start, out = camera_by_step(scene), None, {}
    for st in scene.steps:
        zoomed = cams[st.index][1].width < FRAME_WIDTH - 1e-9
        start = (start if start is not None else st.index) if zoomed else None
        out[st.index] = start
    return out


def _judge(box: BBox, view: BBox, base: BBox) -> tuple[float, BBox]:
    """(overflow, frame it is judged in). A box partly inside a zoomed-in view is
    cropped by design: it is judged in the base frame instead. One entirely
    outside it is not seen at all at that moment."""
    zoomed_in = view.width < base.width - 1e-9
    seen = (box.left < view.right and box.right > view.left and
            box.bottom < view.top and box.top > view.bottom)
    here = (overflow(box, view), view)
    if zoomed_in and seen:
        return min(here, (overflow(box, base), base), key=lambda t: t[0])
    return here


def _lerp_view(a: BBox, b: BBox, f: float) -> BBox:
    return BBox(cx=a.cx + (b.cx - a.cx) * f, cy=a.cy + (b.cy - a.cy) * f,
                width=a.width + (b.width - a.width) * f, height=a.height + (b.height - a.height) * f)


def _framing_entries(scene: Scene, geo: "Geometry", oid: str,
                     seen: list[tuple[int, BBox]]) -> list[tuple[BBox, list[tuple[BBox, BBox]]]]:
    """Each measured box of ``oid`` with the camera frames it is on screen under:
    its first state under the views of the steps before any tween (all its steps if
    it never moves); a tween frame under the camera at that moment (a camera_move
    in the same beat is interpolated); where a tween lands, under the views of
    the steps until the next tween. Each view comes with its base frame."""
    steps = {i for i, _ in seen}
    bases, zin = _base_views(scene), _zoom_in_start(scene)
    first = min(steps)

    def allowed(base: BBox, view: BBox, zoom_from: int | None) -> BBox:
        # cropping is by design only for what was on screen before the zoom-in;
        # something written while zoomed in is meant to be seen in that view
        return base if zoom_from is not None and first < zoom_from else view

    end_view = {i: (v, allowed(bases[i][1], v, zin.get(i))) if i in bases else (v, v)
                for i, v in seen}  # 0: a scene without steps
    obj = scene.get_obj(oid)
    dynamic = obj is not None and is_dynamic(obj, scene)
    initial, states = geo.states_while(oid, steps)
    out: list[tuple[BBox, list[tuple[BBox, BBox]]]] = []
    box = geo.bbox(oid)
    if initial and box:
        init_steps = steps & initial_window(scene) if dynamic and scene.steps else steps
        out.append((box, [end_view[i] for i in sorted(init_steps)] or [(DEFAULT_VIEW, DEFAULT_VIEW)]))
    windows, cams = _beat_windows(scene), camera_by_step(scene)
    by_index = {s.index: s for s in scene.steps}
    for st in states:
        b = geo._box(st["objects"].get(oid))
        if not b:
            continue
        during, after = windows.get(int(st.get("step", 0)), ([], []))
        views: list[tuple[BBox, BBox]] = []
        if any(i in steps for i in during):
            cam = [by_index[i] for i in during if by_index[i].kind == "camera_move" and by_index[i].to]
            if cam:
                tween_t = max(by_index[i].duration for i in during if by_index[i].kind == "tween")
                c = cam[-1]
                # frac is the time fraction of the beat; the camera eases over its own run_time
                f = smooth(min(1.0, float(st.get("frac", 1.0)) * tween_t / max(c.duration, 1e-9)))
                v = _lerp_view(cams[during[0]][0], cams[c.index][1], f)
                views.append((v, allowed(bases[during[0]][0], v, c.index)))
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


PULSE = 1.2  # Manim's Indicate scales its target by this much at the peak


def _pulse_framing(scene: Scene, geo: Geometry) -> list[dict[str, Any]]:
    """A highlight grows its targets 1.2x about their centre for a moment: an
    object near the edge is cut off at the peak."""
    out, done = [], set()
    cams = camera_by_step(scene)
    for st in scene.steps:
        if st.kind != "highlight":
            continue
        view = cams[st.index][1]
        for oid in st.targets:
            box = geo.bbox(oid) if geo.measured else None
            if box is None or oid in done:
                continue
            big = BBox(cx=box.cx, cy=box.cy, width=box.width * PULSE, height=box.height * PULSE)
            if (o := overflow(big, view)) > OFFSCREEN_TOLERANCE and overflow(box, view) <= OFFSCREEN_TOLERANCE:
                done.add(oid)
                out.append(_obl(scene.id, "framing", "error",
                                f"the highlight at step {st.index} grows '{oid}' {PULSE}x and cuts it "
                                f"{o:.2f} units {_where(view)} at its peak (measured)",
                                "move it toward the center (about 10% of its size) or reduce its "
                                "scale", target=oid, step=st.index, overflow=round(o, 3),
                                view=[view.cx, view.cy, view.width, view.height]))
    return out


def _cropped_at_end(scene: Scene, geo: Geometry, oid: str) -> float | None:
    """Overflow beyond the final camera view, if ``oid`` is on screen at the end
    under a camera that moved back out to frame size or wider. Ending zoomed in on a
    detail is a choice (a plot cut by the view is fine), but a text or equation the
    final view slices through is not; ending at zoom 1 but off-centre is the trap
    (zoom=1 keeps the current centre, it does not restore the opening view)."""
    last = scene.steps[-1].index
    view = camera_by_step(scene)[last][1]
    if view == DEFAULT_VIEW or oid not in timeline(scene)[-1].values():
        return None
    if view.width < FRAME_WIDTH - 1e-6:
        box = geo.bbox(oid) if geo.measured else None
        if scene.get_obj(oid).kind not in TEXTLIKE or box is None or box.overlap_area(view) <= 0:
            return None  # zoomed in on a detail: what is left out is a choice
    if not geo.measured:  # estimated: only a centre outside the final view is certain
        cx, cy = scene.get_obj(oid).position
        o = overflow(BBox(cx=cx, cy=cy, width=0.0, height=0.0), view)
        return o if o > 0 else None
    box = geo.bbox(oid)
    ends = [st for st in geo.states if st.get("beat_end")]
    if ends and oid in ends[-1]["objects"]:
        box = geo._box(ends[-1]["objects"][oid]) or box
    if box is None:
        return None
    o = overflow(box, view)
    return o if o > OFFSCREEN_TOLERANCE else None


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
            elif scene.steps and (end := _cropped_at_end(scene, geo, oid)) is not None:
                out.append(_obl(scene.id, "framing", "warning",
                                f"the scene ends with '{oid}' outside the camera's final view "
                                "(the last frame, where a GIF loops, does not show it)",
                                "end with a camera_move that frames it (e.g. zoom=1, focus_x=0, "
                                "focus_y=0), or fade it before the end", target=oid))
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
            o, v = min((_judge(b, view, base) for view, base in vs), key=lambda t: t[0])
            if over is None or o > over:
                over, best = o, v
        if over is None:
            continue
        if over > OFFSCREEN_TOLERANCE:
            placed = obj.payload.get("anchor") or "x2" in obj.payload
            out.append(_obl(scene.id, "framing", "error",
                            f"object '{oid}' extends {over:.2f} units {_where(best)} (measured)",
                            ("it is placed by its plot coordinates/endpoints: change them, or the "
                             "range of the parameter it follows (fix_scene cannot move it)")
                            if placed else
                            "move it toward the center or reduce its scale (fix_scene can do "
                            "this), or move the camera", target=oid, overflow=round(over, 3),
                            view=[best.cx, best.cy, best.width, best.height]))
        elif scene.steps and (end := _cropped_at_end(scene, geo, oid)) is not None:
            out.append(_obl(scene.id, "framing", "warning",
                            f"the scene ends with '{oid}' {end:.2f} units outside the camera's "
                            "final view (the last frame, where a GIF loops, shows it cropped)",
                            "end with a camera_move that frames it (e.g. zoom=1, focus_x=0, "
                            "focus_y=0), or fade it before the end", target=oid))
        elif over > MARGIN_TOLERANCE - view_margin(best):
            # inside the frame but near its edge: worth knowing, not a defect of a GIF
            out.append(_obl(scene.id, "framing", "info",
                            f"object '{oid}' reaches {over + view_margin(best):.2f} units into the "
                            f"{view_margin(best):.2g}-unit safe margin (measured)",
                            "nudge it toward the center", target=oid))
    if scene.steps and geo.measured:
        view = camera_by_step(scene)[scene.steps[-1].index][1]
        on = [g for g in dict.fromkeys(timeline(scene)[-1].values()) if geo.bbox(g)]
        if view != DEFAULT_VIEW and on and not any(geo.bbox(g).overlap_area(view) > 0 for g in on):
            out.append(_obl(scene.id, "framing", "warning",
                            "the scene ends on a camera view with nothing in it (the last frame, "
                            "where a GIF loops, is blank)",
                            "aim the last camera_move at an object (focus_x/focus_y in scene "
                            "units), or end with zoom=1, focus_x=0, focus_y=0"))
    return out + _pulse_framing(scene, geo)


def _chain_root(scene: Scene, oid: str) -> str:
    """First object of the transform chain ``oid`` belongs to (a -> b -> c: a)."""
    src = {st.to: st.targets[0] for st in scene.steps if st.kind == "transform" and st.to and st.targets}
    seen = set()
    while oid in src and oid not in seen:
        seen.add(oid)
        oid = src[oid]
    return oid


def _layer(scene: Scene) -> dict[str, tuple[int, int]]:
    """(step, place among its targets) each object was last added to the scene at:
    Manim draws later additions on top. A write adds an object that is not on
    screen (first time, or after a fade); re-writing one already there redraws it
    in place, without bringing it forward. A transform's destination is drawn by
    its source."""
    out: dict[str, tuple[int, int]] = {}
    on: set[str] = set()
    for st in scene.steps:
        if st.kind == "write":
            for k, t in enumerate(st.targets):
                if t not in on:
                    out[t] = (st.index, k)
                    on.add(t)
        elif st.kind == "fade":
            on.difference_update(st.targets)
        elif st.kind == "transform" and st.to and st.targets and st.targets[0] in out:
            out[st.to] = out[st.targets[0]]
            if st.targets[0] in on:
                on.add(st.to)
    return out


def is_dot(obj: Obj | None) -> bool:
    return obj is not None and obj.kind == "shape" and obj.payload.get("shape") == "dot"


def _disc_cover(dot: BBox, box: BBox) -> float:
    """Share of ``box`` inside the filled disc of a dot (9 x 5 grid)."""
    r = min(dot.width, dot.height) / 2
    pts = [(box.left + (i + 0.5) * box.width / 9, box.bottom + (j + 0.5) * box.height / 5)
           for i in range(9) for j in range(5)]
    return sum((x - dot.cx) ** 2 + (y - dot.cy) ** 2 <= r * r for x, y in pts) / len(pts)


def _tip_cover(p: dict[str, Any], db: BBox) -> float:
    """Share of an explicit arrow's tip (Manim: 0.35 long, at most a quarter of the
    arrow) inside a dot's disc: an apex just touching the dot leaves the tip visible."""
    x1, y1, x2, y2 = (float(p[k]) for k in ("x1", "y1", "x2", "y2"))
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1e-9:
        return 0.0
    tip = min(0.35, 0.25 * length)
    r = min(db.width, db.height) / 2
    pts = [(x2 - (x2 - x1) * tip * k / 10 / length, y2 - (y2 - y1) * tip * k / 10 / length) for k in range(11)]
    return sum((x - db.cx) ** 2 + (y - db.cy) ** 2 <= r * r for x, y in pts) / len(pts)


def under_later_dots(scene: Scene, geo: Geometry, oid: str, pts: list[list[float]],
                     with_id: str | None = None) -> list[list[float]]:
    """The points of ``oid``'s ink covered by no filled dot drawn after it. With
    ``with_id``, only a dot that stays put and is on screen whenever both are counts."""
    order = _layer(scene)
    if with_id is not None:
        (a0, a1), (b0, b1) = _shown_span(scene, oid), _shown_span(scene, with_id)
        lo, hi = max(a0, b0), min(a1, b1)

    def holds(d: Obj) -> bool:
        if with_id is None:
            return True
        d0, d1 = _shown_span(scene, d.id)
        # a node recoloured in place by a transform: its dot stays there
        cur, seen = d.id, {d.id}
        for st in scene.steps:
            if st.kind == "transform" and st.targets and st.targets[0] == cur and st.to not in seen:
                a, b = geo.bbox(cur), geo.bbox(st.to)
                if not (a and b and is_dot(scene.get_obj(st.to))
                        and math.hypot(a.cx - b.cx, a.cy - b.cy) < min(a.width, b.width) / 4):
                    break
                cur = st.to
                seen.add(cur)
                d1 = _shown_span(scene, cur)[1]
        return not is_dynamic(d, scene) and d0 <= lo and d1 >= hi
    discs = [(b.cx, b.cy, min(b.width, b.height) / 2) for d in scene.objects
             if is_dot(d) and d.id != oid and order.get(d.id, (0, 0)) > order.get(oid, (0, 0))
             and holds(d) and (b := geo.bbox(d.id))]
    if not discs:
        return pts
    return [p for p in pts if not any((p[0] - x) ** 2 + (p[1] - y) ** 2 <= r * r for x, y, r in discs)]


def _under_dot(scene: Scene, geo: Geometry, dot: Obj, other: Obj) -> tuple[str, str] | None:
    """A filled dot drawn after a label, line or arrow hides what it covers."""
    order = _layer(scene)
    if order.get(dot.id, (0, 0)) <= order.get(other.id, (0, 0)):
        return None  # the other is drawn on top of the dot
    db = geo.bbox(dot.id)
    if db is None:
        return None
    r = min(db.width, db.height) / 2
    if other.kind in TEXTLIKE:
        tb = geo.bbox(other.id)
        cover = _disc_cover(db, tb) if tb is not None else 0.0
        if cover >= 0.6:  # a label meant to sit on its node, drawn in the wrong order
            return ("error", f"'{other.id}' is hidden under the dot '{dot.id}' (the dot is drawn "
                             "after it, measured)")
        glyphs = geo.glyph_boxes(other.id) or ([tb] if tb else [])
        if cover >= 0.1 or any(_disc_cover(db, g) > 0 for g in glyphs if g.area > 0):
            # a label beside its point, too close (or a small dot on a few letters): move it
            return ("warning", f"'{other.id}' is partly covered by the dot '{dot.id}' "
                               f"(~{max(cover, 0.01) * 100:.0f}% of it, measured)")
        return None
    pts = geo.ink(other.id)
    if not pts:
        return None
    # a bond between two atoms: covered by the dots together (reported once, by the first)
    later = [d for d in scene.objects if is_dot(d) and geo.bbox(d.id)
             and order.get(d.id, (0, 0)) > order.get(other.id, (0, 0))]
    discs = [(b.cx, b.cy, min(b.width, b.height) / 2 - 0.02) for b in (geo.bbox(d.id) for d in later)]
    inside = [any((p[0] - x) ** 2 + (p[1] - y) ** 2 <= rr * rr for x, y, rr in discs) for p in pts]
    if sum(inside) >= 0.6 * len(pts) and later and later[0].id == dot.id:
        names = ", ".join(f"'{d.id}'" for d in later if any(
            (p[0] - geo.bbox(d.id).cx) ** 2 + (p[1] - geo.bbox(d.id).cy) ** 2
            <= (min(geo.bbox(d.id).width, geo.bbox(d.id).height) / 2) ** 2 for p in pts))
        return ("warning", f"'{other.id}' is hidden under the dot(s) {names} (drawn after it, "
                           "measured)")
    if sum(inside) >= 0.6 * len(pts):
        return None
    p = other.payload
    if p.get("shape") == "arrow" and "x2" not in p and (ab := geo.bbox(other.id)):
        p = {"x1": ab.left, "y1": ab.cy, "x2": ab.right, "y2": ab.cy, "shape": "arrow"}
    if p.get("shape") == "arrow" and _tip_cover(p, db) >= 0.5:
        return ("warning", f"the tip of '{other.id}' is hidden under the dot '{dot.id}' (the dot "
                           "is drawn after it, measured)")
    return None


def _pair_finding(scene: Scene, geo: Geometry, a: Obj, b: Obj,
                  initial: bool = True) -> tuple[str, str] | None:
    """(severity, summary) if the two shown shapes genuinely collide, else None.
    ``initial``: whether they are on screen together at the declared parameter
    values (geometry here is measured there; tween states are checked separately)."""
    if a.kind in ("area", "riemann") or b.kind in ("area", "riemann"):
        # fills lie under everything (z_index -1): what is on one is read against
        # its colour (legibility lane), nothing is painted over
        return None
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
        # side by side on one line: a small horizontal overlap is already letters
        # running into each other (stacked lines overlap a little vertically, fine)
        dx = min(ba.right, bb.right) - max(ba.left, bb.left)
        dy = min(ba.top, bb.top) - max(ba.bottom, bb.bottom)
        if dx > 0.03 and dy > 0.6 * min(ba.height, bb.height) and ratio < MEASURED_TEXT_OVERLAP:
            return ("error", f"'{a.id}' and '{b.id}' run into each other on the same line "
                             f"({dx:.2f} units of overlap, measured)")
        if -0.06 < dx <= 0.03 and dy > 0.6 * min(ba.height, bb.height) and ratio < MEASURED_TEXT_OVERLAP:
            return ("warning", f"'{a.id}' and '{b.id}' touch on the same line with no space between "
                               f"them ({-dx:.2f} units apart, measured): they read as one word")
        if ratio >= MEASURED_TEXT_OVERLAP:
            return ("error", f"'{a.id}' and '{b.id}' overlap (~{ratio * 100:.0f}% of the smaller, measured)")
        return None
    if (is_dot(a) or is_dot(b)) and not static_only:
        dot, other = (a, b) if is_dot(a) else (b, a)
        if other.kind in TEXTLIKE + ("shape",) and not is_dot(other):
            if (hidden := _under_dot(scene, geo, dot, other)):
                return hidden
            db, tb = geo.bbox(dot.id), geo.bbox(other.id)
            if other.kind in TEXTLIKE and db and tb and _disc_cover(db, tb) >= 0.9:
                return None  # a label on its node: read against the dot (legibility lane)
    if ka in TEXTLIKE or kb in TEXTLIKE:
        txt, other = (a, b) if ka in TEXTLIKE else (b, a)
        tbox = geo.bbox(txt.id)
        if other.kind == "plot" and tbox:
            # the plot's own texts: axis labels and tick numbers
            marks = geo.named_label_boxes(other.id) + [("tick number", b) for b in geo.number_boxes(other.id)]
            glyphs = geo.glyph_boxes(txt.id) or [tbox]

            def crosses(g: BBox, lb: BBox) -> bool:
                # a thin stroke (minus, fraction bar, \vert) through a number counts:
                # it must reach into the box, and run across it in one direction
                w = min(g.right, lb.right) - max(g.left, lb.left)
                h = min(g.top, lb.top) - max(g.bottom, lb.bottom)
                return w > 0.004 and h > 0.004 and max(w, h) > 0.02
            for which, lb in marks:
                w = min(tbox.right, lb.right) - max(tbox.left, lb.left)
                h = min(tbox.top, lb.top) - max(tbox.bottom, lb.bottom)
                if crosses(tbox, lb) and any(crosses(g, lb) for g in glyphs):  # glyphs really intersect
                    return ("error", f"'{txt.id}' is printed over the {which} of '{other.id}' "
                                     f"({w:.2f} x {h:.2f} units, measured)")
            for which, lb in marks:
                # glyphs this close to a number or label read as part of it
                gaps = [max(lb.left - g.right, g.left - lb.right, lb.bottom - g.top, g.bottom - lb.top)
                        for g in glyphs]
                if gaps and min(gaps) < 0.08:
                    return ("warning", f"'{txt.id}' touches the {which} of '{other.id}' "
                                       f"({max(0.0, min(gaps)):.2f} units apart, measured): they "
                                       "read as one")
        if static_only and other.kind != "plot":
            return None
        pts = (geo.axes_points(other.id) if static_only else geo.curve_points(other.id)) \
            if other.kind == "plot" else under_later_dots(scene, geo, other.id, geo.ink(other.id), txt.id)
        # a shape against the glyphs themselves (a label's box has empty corners)
        boxes = (geo.glyph_boxes(txt.id) or [tbox]) if other.kind == "shape" else [tbox]
        hits = sum(1 for p in pts if any(_inside(p, g) for g in boxes))
        if hits:
            what = "the curves/axes of" if other.kind == "plot" else "the outline of"
            return ("warning", f"'{txt.id}' is crossed by {what} '{other.id}' (measured)")
        if other.kind == "shape" and any(_inside(p, g, shrink=-STROKE_CLEARANCE)
                                         for p in pts for g in boxes):
            # the stroke has a width: a glyph this close to its centre line touches it
            return ("warning", f"'{txt.id}' touches the outline of '{other.id}' (measured)")
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
        if xa and xb and xa.overlap_area(xb) > 0.01:
            return ("warning", f"the axes of '{a.id}' and '{b.id}' run into each other (measured)")
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
    # one play() per beat: what is on screen together is the state after the beat,
    # not after each of its steps (a swap or a crossfade has no in-between state)
    ends = {s.index for n, s in enumerate(scene.steps)
            if n + 1 == len(scene.steps) or not scene.steps[n + 1].with_previous}
    for n, st in enumerate(states, 1):
        if scene.steps and n not in ends:
            continue
        geoms = list(dict.fromkeys(st.values()))
        for i in range(len(geoms)):
            for j in range(i + 1, len(geoms)):
                key = tuple(sorted((geoms[i], geoms[j])))
                together[key] = together.get(key, False) or n in init
    printed: set[tuple[str, str, str]] = set()
    for key, initial in together.items():
        seen.add(key)
        a, b = scene.get_obj(key[0]), scene.get_obj(key[1])
        if a is None or b is None:
            continue
        found = _pair_finding(scene, geo, a, b, initial)
        if found and "printed over" in found[1]:
            txt_id = a.id if a.kind in TEXTLIKE else b.id
            plot_id = b.id if txt_id == a.id else a.id
            k = (txt_id, _chain_root(scene, plot_id), found[1].split(" of '")[0])
            if k in printed:
                continue
            printed.add(k)
        if found:
            sev, summary = found
            for g in key:
                if g in shown_as:
                    summary += f" ('{g}' is on screen as {', '.join(repr(s) for s in sorted(shown_as[g]))})"
            # the text-like side is the target (the one to move), as fix_scene does
            tgt, oth = key[1], key[0]
            if a.kind in TEXTLIKE and b.kind not in TEXTLIKE:
                tgt, oth = a.id, b.id
            elif b.kind in TEXTLIKE and a.kind not in TEXTLIKE:
                tgt, oth = b.id, a.id
            hidden = "hidden under the dot" in summary
            out.append(_obl(scene.id, "collision", sev, summary,
                            ("draw order: write the dot first and this after it (or list it "
                             "after the dot in the same step's targets); a line or arrow can "
                             "also end at the dot's edge") if hidden else
                            "move one of them, or stagger their steps so they are not "
                            "visible together", target=tgt, other=oth))
    # moving curves/shapes: check every measured moment of every tween against what
    # is on screen during that beat -- and, for where the beat leaves them, against
    # everything shown until the next tween moves them again
    tl = timeline(scene) if scene.steps else []
    windows = _beat_windows(scene)
    crossed: set[tuple[str, str]] = set()
    layer = _layer(scene)
    buried: set[tuple[str, str]] = set()  # a dot left resting on the label: one error
    def beat_last(i: int) -> int:  # the last step of the beat step i starts
        while i < len(scene.steps) and scene.steps[i].with_previous:
            i += 1
        return i
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
                if tobj is None or tobj.kind not in TEXTLIKE:
                    continue
                tbox = geo.bbox(tid)
                rest = geo._box(st.get("objects", {}).get(mover)) if st.get("beat_end") else None
                last = beat_last(start)
                # where the dot stops and stays (steps follow, or the scene ends there),
                # over a label still on screen once the beat is over
                cover = _disc_cover(rest, tbox) if tbox and rest else 0.0
                if (cover >= 0.1 and (tid, mover) not in buried and is_dot(scene.get_obj(mover))
                        and tid in tl[last - 1].values()
                        and layer.get(mover, (0, 0)) > layer.get(tid, (0, 0))
                        and (after or last == len(scene.steps))):
                    crossed.add((tid, mover))
                    buried.add((tid, mover))
                    if cover >= 0.6:  # same thresholds as a dot that does not move
                        out.append(_obl(scene.id, "collision", "error",
                                        f"'{tid}' is hidden under the dot '{mover}' where the tween "
                                        f"of step {start} leaves it (measured)",
                                        "move the label away from where the dot stops, change the "
                                        "tween's end value, or write the label after the dot",
                                        target=tid, other=mover, step=start))
                    else:
                        out.append(_obl(scene.id, "collision", "warning",
                                        f"'{tid}' is partly covered by the dot '{mover}' where the "
                                        f"tween of step {start} leaves it (~{cover * 100:.0f}% of "
                                        "it, measured)",
                                        "move the label away from where the dot stops, or change "
                                        "the tween's end value", target=tid, other=mover, step=start))
                    continue
                if (tid, mover) in crossed:
                    continue
                if tbox and any(_inside(p, tbox) for p in pts):
                    crossed.add((tid, mover))
                    out.append(_obl(scene.id, "collision", "warning",
                                    f"'{tid}' is crossed by '{mover}' during or after the tween "
                                    f"of step {start} (measured)",
                                    "move the label out of the path, or limit the tween's range",
                                    target=tid, other=mover, step=start))
    # the error says it all: drop the "crossed during the tween" warning for that pair
    out = [o for o in out if not (o["lane"] == "collision" and o["severity"] == "warning"
                                  and (o.get("target"), o.get("other")) in buried
                                  and "during or after the tween" in o["summary"])]
    # a plot's own axis labels: a wide label is pushed back on screen, over the
    # axis tip, where its own curve (or a dot on it) can strike through it. Strict
    # overlap: the curve ending 0.1 units before the label is Manim's normal layout.
    for obj in scene.objects:
        if obj.kind != "plot" or not geo.measured:
            continue
        lbs = geo.label_boxes(obj.id)
        if not lbs and not geo.number_boxes(obj.id):
            continue
        pl = obj.payload
        pts = [p for c in geo.curves(obj.id) for p in c.get("points", [])] + geo.moving_points(obj.id)
        still = [p for c in geo.curves(obj.id) for p in c.get("points", [])]
        hit = [nb for nb in geo.number_boxes(obj.id)
               if sum(_inside(p, nb, shrink=0.0) for p in pts) >= 2]
        if hit:
            moving = not any(sum(_inside(p, nb, shrink=0.0) for p in still) >= 2 for nb in hit)
            pl = obj.payload
            zero_in = all(float(r[0]) <= 0 <= float(r[1]) for r in (pl["x_range"], pl["y_range"]))
            out.append(_obl(scene.id, "collision", "warning",
                            f"a curve of '{obj.id}' runs through {len(hit)} of its tick numbers"
                            + (" while it moves (tween)" if moving else "") + " (measured)",
                            "drop show_numbers, or use a larger tick step so fewer numbers sit "
                            "on the curve" if zero_in else
                            "include 0 in x_range/y_range so the axes sit clear of the curve, "
                            "or drop show_numbers", target=obj.id))
        static = [p for c in geo.curves(obj.id) for p in c.get("points", [])]
        axes = geo.axes_box(obj.id)
        for which, lb in geo.named_label_boxes(obj.id):
            pushed = axes is not None and (lb.left < axes.right - 0.05 if which == "x_label"
                                           else lb.bottom < axes.top - 0.05)
            cure = ("the plot is too wide/tall for its label, which Manim pushed back over the "
                    "axis tip: reduce width/height (or the scale) or shorten the label"
                    if pushed else None)
            if any(_inside(p, lb, shrink=0.0) for p in static) and not is_dynamic(obj, scene):
                out.append(_obl(scene.id, "collision", "error",
                                f"a curve of '{obj.id}' runs through its own {which} (measured)",
                                cure or "change y_range/x_range so the curve clears the axis end, "
                                "or shorten the label", target=obj.id))
            elif any(_inside(p, lb, shrink=0.0) for p in pts):
                tw = [st.index for st in scene.steps if st.kind == "tween"
                      and st.param in obj_params(obj, scene)]
                out.append(_obl(scene.id, "collision", "error",
                                f"a curve of '{obj.id}' runs through its own {which} during or "
                                f"after the tween(s) at step(s) {tw} (measured)",
                                cure or "limit the tween's range, or widen y_range/x_range so "
                                "the moving curve clears the axis end", target=obj.id,
                                step=tw[0] if tw else None))
        for sh in scene.objects:
            if sh.kind != "shape" or (sh.payload.get("anchor") or {}).get("plot") != obj.id:
                continue
            boxes = [geo._box(st["objects"][sh.id]) for st in geo.states if sh.id in st["objects"]]
            boxes = [b for b in boxes + [geo.bbox(sh.id)] if b]
            if any(b.overlap_area(lb) > 0.01 * min(b.area, lb.area) for b in boxes for lb in lbs):
                out.append(_obl(scene.id, "collision", "warning",
                                f"'{sh.id}' covers an axis label of '{obj.id}' (measured)",
                                "move the anchor away from the axis end, or drop the label",
                                target=sh.id, other=obj.id))
    # between two samples a moving shape sweeps a segment: a small label can sit
    # between them (samples land ~1 unit apart on a fast tween)
    by_beat: dict[int, list[dict[str, Any]]] = {}
    for st in geo.states:
        by_beat.setdefault(int(st.get("step", 0)), []).append(st)
    prev_boxes: dict[str, BBox] = {}
    beat_params = dict(scene.params)  # the parameters as each beat starts
    for start in sorted(by_beat):
        start_params, beat_params = beat_params, dict(by_beat[start][-1].get("params") or beat_params)
        during, _after = windows.get(start, ([], []))
        shown_with: dict[str, set[str]] = {}
        for i in during:
            if 0 < i <= len(tl):
                for g in tl[i - 1].values():
                    shown_with.setdefault(g, set()).update(tl[i - 1].values())
        for mover in {m for st in by_beat[start] for m in st["objects"]}:
            mobj = scene.get_obj(mover)
            if mobj is None or mobj.kind != "shape" or mover not in shown_with:
                continue
            path = [(prev_boxes.get(mover) or geo.bbox(mover), start_params)]
            path += [(geo._box(st["objects"].get(mover)), st.get("params"))
                     for st in by_beat[start]]
            path = [(b, pv) for b, pv in path if b]
            if path:
                prev_boxes[mover] = path[-1][0]
            route = _anchor_route(scene, mobj)
            for tid in shown_with[mover]:
                tobj = scene.get_obj(tid)
                if tobj is None or tobj.kind not in TEXTLIKE or (tid, mover) in crossed:
                    continue
                tbox = geo.bbox(tid)
                if tbox is None:
                    continue
                for (p, pp), (q, qp) in zip(path, path[1:]):
                    pad = BBox(cx=tbox.cx, cy=tbox.cy, width=tbox.width + max(p.width, q.width),
                               height=tbox.height + max(p.height, q.height))
                    # a dot on a formula follows a curve between samples (an orbit
                    # sampled 16 times a beat): trace it rather than the chord
                    pts = route(pp, qp) if route and pp is not None and qp is not None else None
                    pts = pts or [(p.cx, p.cy), (q.cx, q.cy)]
                    if any(_segment_hits(*a, *b, pad) for a, b in zip(pts, pts[1:])):
                        crossed.add((tid, mover))
                        out.append(_obl(scene.id, "collision", "warning",
                                        f"'{tid}' is crossed by '{mover}' during the tween of step "
                                        f"{start} (measured path)",
                                        "move the label out of the path, or limit the tween's range",
                                        target=tid, other=mover, step=start))
                        break
    return out


ROUTE_SUBSTEPS = 48  # points traced between two measured samples of an anchored shape


def _anchor_route(scene: Scene, obj: Obj):
    """For a shape anchored on a static plot by formulas: f(params_a, params_b) ->
    the scene points it passes through as the parameters move linearly from a to
    b (tweens are linear), or None when it cannot be traced."""
    from .model import plot_to_scene
    anchor = obj.payload.get("anchor") or {}
    plot = scene.get_obj(anchor.get("plot", ""))
    if plot is None or plot.kind != "plot" or is_dynamic(plot, scene):
        return None
    try:
        m = plot_to_scene(plot)
        fx = compile_fn(str(anchor.get("x", "0")), scene.params)
        fy = compile_fn(str(anchor.get("y", "0")), scene.params)
    except Exception:  # noqa: BLE001 - an untraceable anchor keeps the chord test
        return None
    (ox, oy), sx, sy = m["origin"], m["x_unit"], m["y_unit"]

    def route(a: dict[str, float], b: dict[str, float]) -> list[tuple[float, float]] | None:
        out = []
        for k in range(ROUTE_SUBSTEPS + 1):
            f = k / ROUTE_SUBSTEPS
            pv = {n: float(a.get(n, v)) + f * (float(b.get(n, v)) - float(a.get(n, v)))
                  for n, v in scene.params.items()}
            try:
                u, v = float(fx(0, **pv)), float(fy(0, **pv))
            except Exception:  # noqa: BLE001
                return None
            if not (math.isfinite(u) and math.isfinite(v)):
                return None
            out.append((ox + sx * u, oy + sy * v))
        return out

    return route


def _segment_hits(x0: float, y0: float, x1: float, y1: float, box: BBox) -> bool:
    """Does the segment cross the box (Liang-Barsky)?"""
    t0, t1, dx, dy = 0.0, 1.0, x1 - x0, y1 - y0
    for p, q in ((-dx, x0 - box.left), (dx, box.right - x0), (-dy, y0 - box.bottom), (dy, box.top - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return True


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


def _blend(color: str, bg: str, opacity: float) -> str:
    """The color a translucent fill shows over the background."""
    def rgb(c: str) -> list[int]:
        h = _COLOR_HEX.get(c.upper(), c).lstrip("#")
        h = "".join(ch * 2 for ch in h) if len(h) == 3 else h
        return [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    try:
        a, b = rgb(color), rgb(bg)
    except ValueError:
        return color
    return "#" + "".join(f"{round(x * opacity + y * (1 - opacity)):02x}" for x, y in zip(a, b))


MIN_FILL_CONTRAST = 1.1  # a translucent fill that barely tints the board


def _explicit_colors(obj: Obj) -> list[tuple[str, str]]:
    """(what, color) for every color the author set on a stroked/text object."""
    if obj.kind in ("area", "riemann"):
        return []  # translucent fills: see _fill_colors
    out = [(f"'{obj.id}'", obj.color)] if obj.color else []
    if obj.kind == "plot" and obj.color and all(
            isinstance(f, dict) and f.get("color") for f in obj.payload.get("functions", [])):
        out = []  # every curve has its own colour: the plot colour draws nothing
    if obj.kind == "plot":
        for i, f in enumerate(obj.payload.get("functions", []), 1):
            if isinstance(f, dict) and f.get("color"):
                out.append((f"curve {i} of '{obj.id}'", str(f["color"])))
    return out


STROKE_CLEARANCE = 0.05  # units from a stroke's centre line where a glyph touches it
MIN_NUMBER_CONTRAST = 3.0  # tick numbers under a fill (WCAG's 3:1 for large text)
MIN_GLYPH_SIZE = 12.0  # pt: a subscript is ~0.7 and a sub-subscript ~0.5 of the size


def tick_decimals(step: float) -> int:
    """Decimals a tick step needs (1 -> 0, 0.25 -> 2)."""
    for d in range(7):
        if abs(step * 10 ** d - round(step * 10 ** d)) < 1e-9:
            return d
    return 6


def _crowded_ticks(obj: Obj) -> str | None:
    """'x-axis'/'y-axis' when adjacent tick numbers would overlap (24pt numbers are
    ~0.3 units tall and ~0.16 units per character wide, before scaling)."""
    p = obj.payload
    for name, rng, length, along in (("x-axis", p.get("x_range", [-5, 5, 1]), p.get("width", 8.0), "w"),
                                     ("y-axis", p.get("y_range", [-3, 3, 1]), p.get("height", 5.0), "h")):
        lo, hi = float(rng[0]), float(rng[1])
        step = float(rng[2]) if len(rng) > 2 else 1.0
        if step <= 0 or hi <= lo:
            continue
        gap = float(length) * obj.scale * step / (hi - lo)
        d = tick_decimals(step)
        chars = max(len(f"{v:.{d}f}") for v in (lo, hi))
        need = (0.3 if along == "h" else 0.16 * chars + 0.08) * obj.scale
        if gap < need:
            return name
    return None


def _shown_span(scene: Scene, oid: str) -> tuple[float, float]:
    """(step shown, step gone) of an object; inf when never shown or never gone.
    A transform's destination is shown from that step by its source's mobject,
    and is gone when that mobject is faded or morphed into something else."""
    inf = float("inf")
    carrier: dict[str, str] = {}  # object id -> the mobject showing it
    start: float = inf
    for st in scene.steps:
        if st.kind == "write":
            for t in st.targets:
                carrier[t] = t
            if oid in st.targets and start == inf:
                start = st.index
        elif st.kind == "transform" and st.to and st.targets and st.targets[0] in carrier:
            mob = carrier.pop(st.targets[0])
            for k in [k for k, v in carrier.items() if v == mob]:
                del carrier[k]  # the mobject now shows the destination only
            carrier[st.to] = mob
            if st.to == oid and start == inf:
                start = st.index
        elif st.kind == "fade":
            gone = set(st.targets) | {carrier[t] for t in st.targets if t in carrier}
            for k in [k for k, v in carrier.items() if k in gone or v in gone]:
                del carrier[k]
        if start != inf and oid not in carrier:
            return start, st.index
    return start, inf


def _text_on_fill(scene: Scene, geo: Geometry, bg: str,
                  ink: str) -> tuple[list[dict[str, Any]], set[str]]:
    """Text on a fill (which lies under it) is read against the fill's colour, not
    the background. Returns the findings and the ids of texts that sit wholly on a
    fill for as long as they are shown (only those skip the background check)."""
    out: list[dict[str, Any]] = []
    on_fill: set[str] = set()
    spans: dict[str, list[tuple[float, float]]] = {}  # text -> when a fill/dot lies under all of it
    if not geo.measured:
        return out, on_fill
    for f in scene.objects:
        if f.kind not in ("area", "riemann"):
            continue
        op = float(f.payload.get("opacity", 0.5)) if f.kind == "area" else 0.6
        under = _blend(f.color or "#5a8fd6", bg, op)
        f0, f1 = _shown_span(scene, f.id)
        for t in scene.objects:
            if t.kind not in TEXTLIKE:
                continue
            t0, t1 = _shown_span(scene, t.id)
            tb = geo.bbox(t.id)
            if max(f0, t0) >= min(f1, t1) or tb is None:
                continue  # never on screen together
            cover = geo.fill_cover(f.id, tb) or 0
            if cover < 0.25:
                continue
            if cover >= 0.9:
                spans.setdefault(t.id, []).append((f0, f1))
            col = t.color or ink
            c = contrast(col, under)
            if c is not None and c < MIN_TEXT_CONTRAST:
                out.append(_obl(scene.id, "legibility", "warning",
                                f"'{t.id}' ({col}) is written on the fill '{f.id}', where it has a "
                                f"contrast of {c:.2f}:1 (measured)",
                                "move the text off the fill, give it a color that contrasts with "
                                f"{f.color or 'the fill'}, or lower the fill's opacity",
                                target=t.id, other=f.id))
    # a label written on a filled dot (a node) is read against the dot
    order = _layer(scene)
    for d in scene.objects:
        # a dot that moves leaves the label behind: only a still one lies under it
        # for good; a moving one is judged where it comes to rest
        moving = is_dot(d) and is_dynamic(d, scene)
        if moving:
            ends = [st_ for st_ in geo.states if st_.get("beat_end") and d.id in st_.get("objects", {})]
            db = geo._box(ends[-1]["objects"][d.id]) if ends else None
        else:
            db = geo.bbox(d.id) if is_dot(d) else None
        if db is None:
            continue
        d0, d1 = _shown_span(scene, d.id)
        for t in scene.objects:
            tb = geo.bbox(t.id) if t.kind in TEXTLIKE else None
            if tb is None or order.get(t.id, (0, 0)) < order.get(d.id, (0, 0)):
                continue
            t0, t1 = _shown_span(scene, t.id)
            if max(d0, t0) >= min(d1, t1) or _disc_cover(db, tb) < (0.5 if moving else 0.9):
                continue
            if not moving:
                spans.setdefault(t.id, []).append((d0, d1))
            col, under = t.color or ink, d.color or ink
            c = contrast(col, under)
            if c is not None and c < MIN_TEXT_CONTRAST:
                out.append(_obl(scene.id, "legibility", "warning",
                                f"'{t.id}' ({col}) is written on the dot '{d.id}' ({under}), where "
                                f"it has a contrast of {c:.2f}:1 (measured)",
                                "give the label or the dot a contrasting color, or move the label "
                                "beside the dot", target=t.id, other=d.id))
    # wholly on a fill or dot for as long as it is shown (a node recoloured by a
    # transform hands over to the new dot): the background never shows behind it
    for tid, sp in spans.items():
        t0, t1 = _shown_span(scene, tid)
        at = t0
        for a, b in sorted(sp):
            if a <= at:
                at = max(at, b)
        if at >= t1:
            on_fill.add(tid)
    return out, on_fill


def _lane_legibility(scene: Scene, geo: Geometry, theme_name: str = DEFAULT_THEME) -> list[dict[str, Any]]:
    out = []
    zoom = zoom_while_visible(scene)  # a zoomed-out camera shrinks everything on screen
    bg = theme(theme_name)["background_color"]
    ink = theme(theme_name)["ink"]
    for f in scene.objects:
        # a fill under its own plot's tick numbers: the ink read against the fill
        if f.kind not in ("area", "riemann") or not geo.measured:
            continue
        plot = scene.get_obj(f.payload.get("plot", ""))
        if plot is None or not plot.payload.get("show_numbers"):
            continue
        op = float(f.payload.get("opacity", 0.5)) if f.kind == "area" else 0.6
        col = f.color or "#5a8fd6"
        c = contrast(ink, _blend(col, bg, op))
        covered = [nb for nb in geo.number_boxes(plot.id) if (geo.fill_cover(f.id, nb) or 0) >= 0.4]
        if covered and c is not None and c < MIN_NUMBER_CONTRAST:
            out.append(_obl(scene.id, "legibility", "warning",
                            f"the fill '{f.id}' covers {len(covered)} tick number(s) of '{plot.id}', "
                            f"which have a contrast of {c:.2f}:1 on it (measured)",
                            ("lower its opacity (e.g. 0.3) or pick a colour closer to the background, "
                             "or drop show_numbers") if f.kind == "area" else
                            ("pick a colour closer to the background (a Riemann sum's opacity is "
                             "fixed at 0.6), or drop show_numbers"), target=f.id, other=plot.id))
    found, on_fill = _text_on_fill(scene, geo, bg, ink)
    out.extend(found)
    for obj in scene.objects:
        if obj.kind in ("area", "riemann") and obj.color:
            op = float(obj.payload.get("opacity", 0.5)) if obj.kind == "area" else 0.6
            c = contrast(_blend(obj.color, bg, op), bg)
            if c is not None and c < MIN_FILL_CONTRAST and op >= 0.5:
                # it lies under the curves and axes: it erases nothing, it just does not show
                out.append(_obl(scene.id, "legibility", "warning",
                                f"fill of '{obj.id}' ({obj.color} at opacity {op}) is the background "
                                f"colour: it does not show (contrast {c:.2f}:1)",
                                "pick a color that contrasts with the theme, or reset it with update_object(color=\"default\")",
                                target=obj.id))
            elif c is not None and c < MIN_FILL_CONTRAST:
                out.append(_obl(scene.id, "legibility", "warning",
                                f"fill of '{obj.id}' ({obj.color} at opacity {op}) barely shows on "
                                f"the {theme_name} background (contrast {c:.2f}:1)",
                                "pick a color that contrasts with the theme, or reset it with update_object(color=\"default\")",
                                target=obj.id))
        for what, col in ([] if obj.id in on_fill else _explicit_colors(obj)):
            c = contrast(col, bg)
            need = MIN_TEXT_CONTRAST if obj.kind in TEXTLIKE else MIN_CONTRAST
            if c is not None and c < need:
                out.append(_obl(scene.id, "legibility", "warning",
                                f"{what} is {col} on the {theme_name} background {bg} (contrast "
                                f"{c:.2f}:1): " + ("nearly invisible" if c < MIN_CONTRAST else "hard to read as text"),
                                "pick a color that contrasts with "
                                "the theme, or update_object(color=\"default\") to use the theme's palette", target=obj.id))
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
            elif (g := geo.glyph_pt(obj.id)) is not None and g * z < MIN_GLYPH_SIZE:
                # nested fractions, sub-subscripts: the smallest symbols, measured
                out.append(_obl(scene.id, "legibility", "warning",
                                f"the smallest symbols of equation '{obj.id}' render at ~{g * z:.0f}pt "
                                f"(below {MIN_GLYPH_SIZE:.0f}pt; measured)",
                                "increase its scale, or flatten nested fractions/scripts",
                                target=obj.id))
        if obj.kind == "plot" and (obj.payload.get("x_label") or obj.payload.get("y_label")):
            fs = MATHTEX_FONT_SIZE * obj.scale * z
            if fs < MIN_FONT_SIZE:
                out.append(_obl(scene.id, "legibility", "warning",
                                f"axis labels of '{obj.id}' render at ~{fs:.0f}pt (below "
                                f"{MIN_FONT_SIZE:.0f}pt)", "increase the plot's scale", target=obj.id))
        if obj.kind == "plot" and obj.payload.get("show_numbers"):
            crowded = _crowded_ticks(obj)
            if crowded:
                out.append(_obl(scene.id, "legibility", "warning",
                                f"tick numbers of '{obj.id}' overlap on its {crowded} (too many "
                                "ticks for the axis length)", "use a larger tick step "
                                "(x_range/y_range [min, max, step]), a longer axis, or drop "
                                "show_numbers", target=obj.id))
            fs = TICK_FONT_SIZE * obj.scale * z
            if fs < MIN_FONT_SIZE:
                out.append(_obl(scene.id, "legibility", "warning",
                                f"tick numbers of '{obj.id}' render at ~{fs:.0f}pt (below "
                                f"{MIN_FONT_SIZE:.0f}pt)", "increase the plot's scale or drop "
                                "show_numbers", target=obj.id))
    for o in out:  # a size made small by a zoom-out: name the zoom, the object may be fine
        z = zoom.get(o.get("target") or "", 1.0)
        if z < 1.0 and ("pt" in o["summary"] and ("render" in o["summary"] or "small" in o["summary"])):
            o["summary"] += f" while the camera is zoomed out (zoom {z:g})"
            o["suggested_fix"] += ", or zoom out less (camera_move zoom closer to 1)"
    return out


def _lane_timing(scene: Scene) -> list[dict[str, Any]]:
    out = []
    defined = {o.id for o in scene.objects}
    if not scene.objects:
        out.append(_obl(scene.id, "timing", "error", "the scene is empty: the video is blank",
                        "add objects (add_text, add_plot, ...) and write steps for them"))
        return out
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
                elif tid in defined and tid not in on_screen and step.kind == "fade":
                    out.append(_obl(scene.id, "timing", "error",
                                    f"step {step.index} fades '{tid}', which is already faded out: "
                                    "Manim shows it again at full opacity to fade it",
                                    f"drop '{tid}' from this fade", step=step.index, target=tid))
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
    out += _still_steps(scene)
    # Areas, Riemann sums and anchored shapes are drawn from their plot's axes but
    # are separate objects: fading (or never writing) the plot leaves them floating.
    floating: set[str] = set()
    for st in timeline(scene):
        on = set(st) | set(st.values())
        for oid in st:
            o = scene.get_obj(st[oid])
            parent = (o.payload.get("plot") or (o.payload.get("anchor") or {}).get("plot")) if o else None
            if parent and parent in st and st[parent] != parent and parent not in st.values() \
                    and oid not in floating:
                floating.add(oid)
                out.append(_obl(scene.id, "timing", "warning",
                                f"'{oid}' is drawn on the axes and curve of '{parent}', but '{parent}' "
                                f"now shows '{st[parent]}' (transformed): it no longer matches",
                                f"fade '{oid}' before the transform, and add the overlay to "
                                f"'{st[parent]}' instead", target=oid))
            elif parent and parent not in on and oid not in floating:
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
            if st.to and st.kind == "transform":  # a camera_move's `to` is its zoom@focus
                used.add(st.to)
        if not used:
            out.append(_obl(scene.id, "timing", "error",
                            "no step shows anything (only waits, tweens or camera moves): the "
                            "video is blank", "add write steps for the objects to display"))
        shown_params = set().union(*(obj_params(o, scene) for o in scene.objects if o.id in used)) \
            if used else set()
        for o in scene.objects:
            if o.id not in used:
                idle = sorted({st.index for st in scene.steps if st.kind == "tween"
                               and st.param in obj_params(o, scene) - shown_params})
                out.append(_obl(scene.id, "timing", "warning",
                                f"object '{o.id}' is never shown (no step uses it)"
                                + (f", so the tween(s) at step(s) {idle} change nothing on screen"
                                   if idle else ""),
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
    obligations.extend(_not_real(scene))
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
                       "geometry was not measured (Manim is missing, or the geometry probe "
                       "failed: see any compile/LaTeX error), so the fidelity lane and curve "
                       "clipping were skipped",
                       "shapes anchored to a plot are not placed in estimated geometry",
                       "intent lane (vision) is not available yet"]

    # one cause, one finding: a dot that leaves the frame has left its axes too
    off = {o.get("target") for o in obligations if o["lane"] == "framing" and o["severity"] == "error"}
    cause = {o.get("target"): o for o in obligations
             if "leaves the axes" in o["summary"] and o.get("target") in off}
    for o in obligations:
        c = cause.get(o.get("target"))
        if c and o["lane"] == "framing" and o["severity"] == "error":
            o["summary"] += f" -- cause: {c['summary']}"
            if c.get("step") is not None and o.get("step") is None:
                o["step"] = c["step"]
    obligations = [o for o in obligations
                   if not ("leaves the axes" in o["summary"] and o.get("target") in off
                           and o["lane"] != "framing")
                   and not ("(measured path)" in o["summary"] and o.get("other") in off)]
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
