"""Automatic repairs for measured obligations (``fix_scene``).

Only findings with an unambiguous mechanical fix are repaired; everything that
needs the author's intent (a LaTeX typo, a forgotten step, a missing object) is
returned as ``unresolved``. Every change is reported, never applied silently.

* framing    -> translate the object back inside the safe area (scale down if it
                cannot fit);
* collision  -> move the label to the nearest free spot (no curve, outline or
                other label inside it, inside the safe area);
* fidelity   -> drop explicit sampling overrides so adaptive sampling applies;
                if there were none, sample more finely without smoothing;
* legibility -> enlarge the text / equation to the minimum readable size.
"""

from __future__ import annotations

import math
from typing import Any

from .geometry import BBox
from .model import Scene, plot_to_scene
from .verify import (DEFAULT_VIEW, MIN_FONT_SIZE, TEXTLIKE, Geometry, timeline, view_margin,
                     visible_views, zoom_while_visible)



def _safe_bounds(o: dict[str, Any]) -> tuple[float, float, float, float]:
    """Safe area (l, r, b, t) of the camera view a framing finding was measured in."""
    v = BBox(*o["view"]) if o.get("view") else DEFAULT_VIEW
    m = view_margin(v)
    return v.left + m, v.right - m, v.bottom + m, v.top - m


def _into(box: BBox, l: float, r: float, b: float, t: float) -> tuple[float, float]:
    dx = (l - box.left) if box.left < l else (r - box.right) if box.right > r else 0.0
    dy = (b - box.bottom) if box.bottom < b else (t - box.top) if box.top > t else 0.0
    return dx, dy


def _shift(box: BBox, dx: float, dy: float) -> BBox:
    return BBox(cx=box.cx + dx, cy=box.cy + dy, width=box.width, height=box.height)


def _co_visible(scene: Scene, oid: str) -> set[str]:
    """Shapes shown at the same time as ``oid`` (by geometry id)."""
    out: set[str] = set()
    for st in timeline(scene):
        geoms = set(st.values())
        if oid in geoms:
            out |= geoms
    out.discard(oid)
    return out


def _free_spot(scene: Scene, geo: Geometry, oid: str, box: BBox) -> tuple[float, float] | None:
    """Smallest translation putting ``box`` in a free, safe spot."""
    others = _co_visible(scene, oid)
    pts: list[list[float]] = []
    boxes: list[BBox] = []
    marks: list[BBox] = []  # a plot's own texts: keep a wider gap (not part of the axes)
    fills: list[str] = []
    for g in others:
        o = scene.get_obj(g)
        if o is None:
            continue
        if o.kind in ("area", "riemann"):
            fills.append(g)
        elif o.kind == "plot":
            pts += geo.curve_points(g) + geo.moving_points(g)
            marks += [lb for _, lb in geo.named_label_boxes(g)] + geo.number_boxes(g)
        elif o.kind == "shape":
            pts += geo.ink(g) + geo.moving_points(g)
        elif o.kind in TEXTLIKE:
            b = geo.bbox(g)
            if b:
                boxes.append(b)

    # inside the camera view the label is written in (not the default frame)
    view = next(iter(visible_views(scene).get(oid, [])), (0, DEFAULT_VIEW))[1]
    m = view_margin(view)
    sl, sr, sb, st_ = view.left + m, view.right - m, view.bottom + m, view.top - m

    def free(b: BBox) -> bool:
        pad = 0.08
        if not (b.left >= sl and b.right <= sr and b.bottom >= sb and b.top <= st_):
            return False
        if any(b.left - pad <= p[0] <= b.right + pad and b.bottom - pad <= p[1] <= b.top + pad for p in pts):
            return False
        if any((geo.fill_cover(f, b) or 0) >= 0.25 for f in fills):
            return False  # under (or on) a fill: painted over, or read against its colour
        gap = 0.1  # labels next to each other keep a visible gap
        if not all(b.overlap_area(BBox(cx=o.cx, cy=o.cy, width=o.width + 2 * gap,
                                       height=o.height + 2 * gap)) <= 0 for o in boxes):
            return False
        gap = 0.3  # so a label does not read as a tick number or an axis label
        return all(b.overlap_area(BBox(cx=o.cx, cy=o.cy, width=o.width + 2 * gap,
                                       height=o.height + 2 * gap)) <= 0 for o in marks)

    best = None
    step = 0.1
    for ring in range(1, 60):
        r = ring * step
        cands = []
        n = max(8, int(2 * math.pi * r / step))
        for k in range(n):
            ang = 2 * math.pi * k / n
            cands.append((r * math.cos(ang), r * math.sin(ang)))
        # prefer vertical moves for labels (keeps reading order), then the rest
        cands.sort(key=lambda d: (abs(d[0]) > 1e-9, abs(d[0]) + abs(d[1])))
        for dx, dy in cands:
            if free(_shift(box, dx, dy)):
                best = (dx, dy)
                break
        if best:
            return best
    return None


def apply_fixes(scene: Scene, obligations: list[dict[str, Any]],
                measurement: dict[str, Any] | None,
                attempts: dict[str, int] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply mechanical fixes in place. Returns (changes, unresolved obligations)."""
    geo = Geometry(scene, measurement)
    changes: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    touched: set[str] = set()

    for o in obligations:
        if o["severity"] not in ("error", "warning"):
            continue
        lane, oid = o["lane"], o.get("target")
        obj = scene.get_obj(oid) if oid else None

        if lane == "framing" and obj is not None and obj.kind in ("area", "riemann") \
                and geo.measured:
            # a fill is drawn from its plot's axes: move the plot by the overflow
            plot = scene.get_obj(obj.payload.get("plot", ""))
            box = geo.bbox(oid)
            if plot is None or box is None or plot.id in touched:
                unresolved.append(o)
                continue
            dx, dy = _into(box, *_safe_bounds(o))
            if abs(dx) < 1e-3 and abs(dy) < 1e-3:
                unresolved.append(o)  # inside at its first state: it is a tween that overflows
                continue
            x, y = plot.position
            plot.position = (round(x + dx, 3), round(y + dy, 3))
            for dep in scene.objects:
                if dep.payload.get("plot") == plot.id:
                    dep.position = plot.position
            touched.add(plot.id)
            changes.append({"object": plot.id, "lane": lane, "position": list(plot.position),
                            "reason": f"{o['summary']} (moved its plot)"})

        elif lane == "framing" and obj is not None and geo.measured and oid not in touched:
            if obj.payload.get("anchor") or "x2" in obj.payload:
                # anchored / explicit segment: placed by plot coords or endpoints
                unresolved.append({**o, "summary": f"'{oid}' is placed by "
                                   + ("its plot coordinates (and any parameter they follow)"
                                      if obj.payload.get("anchor") else "its endpoints")
                                   + ": change those; fix_scene does not move it"})
                continue
            box = geo.bbox(oid)
            if box is None:
                unresolved.append(o)
                continue
            new_scale = obj.scale
            sl, sr, sb, st_ = _safe_bounds(o)
            sw, sh = sr - sl, st_ - sb
            factor = min(1.0, 0.97 * sw / box.width if box.width else 1, 0.97 * sh / box.height if box.height else 1)
            if factor < 1.0:
                new_scale = round(obj.scale * factor, 3)
                if obj.kind in TEXTLIKE:
                    # never below the readable size: the legibility repair would undo it
                    z = zoom_while_visible(scene).get(oid, 1.0)
                    base = float(obj.payload.get("font_size", 36)) if obj.kind == "text" else 48.0
                    if base * new_scale * z < MIN_FONT_SIZE:
                        unresolved.append({**o, "summary": o["summary"] + f"; '{oid}' is too long to "
                                           "fit at a readable size: split it into shorter lines or "
                                           "objects"})
                        continue
                box = BBox(cx=box.cx, cy=box.cy, width=box.width * factor, height=box.height * factor)
            dx, dy = _into(box, sl, sr, sb, st_)
            if factor == 1.0 and abs(dx) < 1e-3 and abs(dy) < 1e-3:
                # already inside at its first state: the overflow comes from a tween,
                # which moving the object would not fix -- say so instead of a no-op
                unresolved.append(o)
                continue
            x, y = obj.position
            f = new_scale / obj.scale if obj.scale else 1.0
            if obj.kind == "plot" and (ab := geo.axes_box(oid)) is not None:
                # never push a plot into its neighbour: two axes then read as one
                moved = BBox(cx=x + dx + (ab.cx - x) * f, cy=y + dy + (ab.cy - y) * f,
                             width=ab.width * f, height=ab.height * f)
                hit = [g.id for g in scene.objects if g.kind == "plot" and g.id != oid
                       and (gb := geo.axes_box(g.id)) is not None
                       and moved.overlap_area(gb) > 0.01 and ab.overlap_area(gb) <= 0.01]
                if hit:
                    unresolved.append({**o, "summary": o["summary"] + f"; moving '{oid}' inside the "
                                       f"frame would put its axes over {hit}: reduce its width/scale "
                                       "or move the other plot"})
                    continue
            obj.position = (round(x + dx, 3), round(y + dy, 3))
            obj.scale = new_scale

            def follow(px: float, py: float) -> tuple[float, float]:
                # a point placed in data terms (to_scene) keeps its data coordinates:
                # the data centre goes to the new position, distances scale by f
                return (round(x + dx + (px - x) * f, 3), round(y + dy + (py - y) * f, 3))
            drawn = {}
            for st in scene.steps:
                for t in ([*st.targets, st.to] if st.kind == "transform" else st.targets):
                    drawn.setdefault(t, st.index)
            if obj.kind == "plot":  # overlays follow their plot (kept in the model too)
                for dep in scene.objects:
                    if dep.payload.get("plot") == obj.id:
                        dep.position = obj.position
                # explicit segments drawn on its axes (vectors) are part of the figure
                axes = geo.axes_box(oid)
                for seg in scene.objects:
                    pl = seg.payload
                    if axes is None or "x2" not in pl or seg.id in touched:
                        continue
                    ends = ((pl["x1"], pl["y1"]), (pl["x2"], pl["y2"]))
                    if all(axes.left <= ex <= axes.right and axes.bottom <= ey <= axes.top
                           for ex, ey in ends):
                        (x1, y1), (x2, y2) = follow(pl["x1"], pl["y1"]), follow(pl["x2"], pl["y2"])
                        pl.update(x1=x1, y1=y1, x2=x2, y2=y2)
                        touched.add(seg.id)
                        changes.append({"object": seg.id, "lane": lane, "endpoints":
                                        [pl["x1"], pl["y1"], pl["x2"], pl["y2"]],
                                        "reason": f"drawn on '{oid}', moved with it"})
                # labels and dots placed by hand over its data rectangle (via to_scene)
                for o2 in scene.objects:
                    if (axes is None or o2.id in touched or "x2" in o2.payload
                            or o2.payload.get("anchor") or o2.kind not in ("text", "equation", "shape")):
                        continue
                    px, py = o2.position
                    if (o2.kind in ("text", "equation") and abs(px) < 0.05
                            and drawn.get(o2.id, 0) < drawn.get(oid, 0)):
                        continue  # a frame-centred title written before the plot: not part of it
                    if axes.left <= px <= axes.right and axes.bottom <= py <= axes.top:
                        o2.position = follow(px, py)
                        touched.add(o2.id)
                        changes.append({"object": o2.id, "lane": lane, "position": list(o2.position),
                                        "reason": f"placed over '{oid}', moved with it"})
            touched.add(oid)
            changes.append({"object": oid, "lane": lane, "position": list(obj.position),
                            "scale": obj.scale, "reason": o["summary"],
                            **({"to_scene": plot_to_scene(obj)} if obj.kind == "plot" else {})})

        elif lane == "collision" and geo.measured:
            # move the label (the text-like side) to the nearest free spot
            cand = [c for c in (oid, o.get("other")) if c and scene.get_obj(c)
                    and scene.get_obj(c).kind in TEXTLIKE and c not in touched]
            # move the label written later: the earlier one (often the title) keeps its place
            first = {}
            for st in scene.steps:
                for t in ([*st.targets, st.to] if st.kind == "transform" else st.targets):
                    first.setdefault(t, st.index)
            cand.sort(key=lambda c: -first.get(c, 0))
            target = cand[0] if cand else None
            if target is None:
                unresolved.append({**o, "summary": o["summary"] + ": no movable label in it "
                                   "(fix_scene moves text and equations only); change the "
                                   "ranges, the tween or the positions"})
                continue
            tobj = scene.get_obj(target)
            box = geo.bbox(target)
            spot = _free_spot(scene, geo, target, box) if box else None
            if spot is None:
                unresolved.append(o)
                continue
            other = next((c for c in (oid, o.get("other")) if c and c != target), None)
            if (other and scene.get_obj(other) and scene.get_obj(other).kind not in TEXTLIKE
                    and first.get(target, 0) < first.get(other, 0) and math.hypot(*spot) > 0.5):
                # written first (a title): it keeps its place; what came later moves
                unresolved.append({**o, "summary": o["summary"] + f": '{target}' was written "
                                   f"before '{other}' and the nearest free spot is "
                                   f"{math.hypot(*spot):.2f} units away; move '{other}' instead "
                                   "(fix_scene moves text and equations only)"})
                continue
            # a point's label must stay by its point: far away it reads as another's.
            # Distances run from the dot to the label's nearest edge (a wide label's
            # centre can be far from a dot right beside its first letter)
            def gap(db, b):
                return math.hypot(max(b.left - db.cx, 0.0, db.cx - b.right),
                                  max(b.bottom - db.cy, 0.0, db.cy - b.top))
            moved = BBox(cx=box.cx + spot[0], cy=box.cy + spot[1], width=box.width, height=box.height)
            dots = [(d.id, db) for d in scene.objects if d.kind == "shape"
                    and d.payload.get("shape") == "dot" and (db := geo.bbox(d.id)) and gap(db, box) < 0.5]
            far = max((gap(db, moved) for _, db in dots), default=0.0)
            if dots and far > 0.5:
                unresolved.append({**o, "summary": o["summary"] + f": the nearest free spot leaves it "
                                   f"{far:.2f} units from its point {[d for d, _ in dots]}, too far "
                                   "to read as its label; place it yourself (e.g. on the other side "
                                   "of the point)"})
                continue
            x, y = tobj.position
            tobj.position = (round(x + spot[0], 3), round(y + spot[1], 3))
            touched.add(target)
            changes.append({"object": target, "lane": lane, "position": list(tobj.position),
                            "reason": o["summary"]})

        elif lane == "fidelity" and obj is not None and obj.kind == "plot":
            i = int(o.get("fn", 1)) - 1
            fns = obj.payload.get("functions", [])
            if not (0 <= i < len(fns)) or not isinstance(fns[i], dict):
                unresolved.append(o)
                continue
            fn = fns[i]
            xr = obj.payload.get("x_range", [-5, 5])
            finest = (float(xr[1]) - float(xr[0])) / 3000.0
            if fn.get("smooth") is False and abs(float(fn.get("dx", 0)) - finest) < 1e-12:
                unresolved.append({**o, "summary": f"'{oid}' curve {i + 1} already uses the finest "
                                   "sampling fix_scene sets: the remaining error is the curve itself "
                                   "(steep or near a pole at this range); narrow x_range or y_range"})
                continue
            key = f"{oid}:{i}"
            stage = (attempts or {}).get(key, 0)
            if attempts is not None:
                attempts[key] = stage + 1
            if stage >= 2 or (stage == 1 and fn.get("smooth") is False and "dx" in fn):
                unresolved.append(o)  # both repairs tried: needs the author
                continue
            if stage == 0 and ("smooth" in fn or "dx" in fn):
                removed = [k for k in ("smooth", "dx") if k in fn]
                for k in removed:
                    fn.pop(k)
                what = f"removed {', '.join(removed)} override (adaptive sampling)"
            else:
                xr = obj.payload.get("x_range", [-5, 5])
                fn["dx"] = (float(xr[1]) - float(xr[0])) / 3000.0
                fn["smooth"] = False
                what = f"finer sampling dx={fn['dx']:.4g}, smooth=false"
            changes.append({"object": oid, "lane": lane, "function": i + 1, "change": what,
                            "reason": o["summary"]})

        elif lane == "legibility" and obj is not None and obj.kind in ("text", "equation"):
            z = zoom_while_visible(scene).get(oid, 1.0)  # the check divides by the zoom too
            if obj.kind == "text":
                old = float(obj.payload.get("font_size", 36))
                new = max(round(1.05 * MIN_FONT_SIZE / (obj.scale * z), 1), old)
                if new == old:
                    unresolved.append(o)
                    continue
                obj.payload["font_size"] = new
                changes.append({"object": oid, "lane": lane, "font_size": new, "reason": o["summary"]})
            else:
                old = obj.scale
                new = max(old, round(1.25 * MIN_FONT_SIZE / (48 * z), 3))  # MathTex: 48pt at scale 1
                if new == old:
                    unresolved.append(o)
                    continue
                obj.scale = new
                changes.append({"object": oid, "lane": lane, "scale": new, "reason": o["summary"]})
        else:
            unresolved.append(o)
    return changes, unresolved
