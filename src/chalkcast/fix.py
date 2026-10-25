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

from .geometry import SAFE_MARGIN, FRAME_HEIGHT, FRAME_WIDTH, BBox
from .model import Scene
from .verify import MIN_FONT_SIZE, TEXTLIKE, Geometry, timeline

_SAFE_L, _SAFE_R = -FRAME_WIDTH / 2 + SAFE_MARGIN, FRAME_WIDTH / 2 - SAFE_MARGIN
_SAFE_B, _SAFE_T = -FRAME_HEIGHT / 2 + SAFE_MARGIN, FRAME_HEIGHT / 2 - SAFE_MARGIN


def _shift(box: BBox, dx: float, dy: float) -> BBox:
    return BBox(cx=box.cx + dx, cy=box.cy + dy, width=box.width, height=box.height)


def _in_safe(box: BBox) -> bool:
    return box.left >= _SAFE_L and box.right <= _SAFE_R and box.bottom >= _SAFE_B and box.top <= _SAFE_T


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
    for g in others:
        o = scene.get_obj(g)
        if o is None:
            continue
        if o.kind == "plot":
            pts += geo.curve_points(g)
        elif o.kind == "shape":
            pts += geo.ink(g)
        elif o.kind in TEXTLIKE:
            b = geo.bbox(g)
            if b:
                boxes.append(b)

    def free(b: BBox) -> bool:
        pad = 0.08
        if not _in_safe(b):
            return False
        if any(b.left - pad <= p[0] <= b.right + pad and b.bottom - pad <= p[1] <= b.top + pad for p in pts):
            return False
        return all(b.overlap_area(o) <= 0 for o in boxes)

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
            dx = (_SAFE_L - box.left) if box.left < _SAFE_L else (_SAFE_R - box.right) if box.right > _SAFE_R else 0.0
            dy = (_SAFE_B - box.bottom) if box.bottom < _SAFE_B else (_SAFE_T - box.top) if box.top > _SAFE_T else 0.0
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
                unresolved.append(o)  # moves with a parameter: needs a different range
                continue
            box = geo.bbox(oid)
            if box is None:
                unresolved.append(o)
                continue
            new_scale = obj.scale
            sw, sh = (_SAFE_R - _SAFE_L), (_SAFE_T - _SAFE_B)
            factor = min(1.0, 0.97 * sw / box.width if box.width else 1, 0.97 * sh / box.height if box.height else 1)
            if factor < 1.0:
                new_scale = round(obj.scale * factor, 3)
                box = BBox(cx=box.cx, cy=box.cy, width=box.width * factor, height=box.height * factor)
            dx = (_SAFE_L - box.left) if box.left < _SAFE_L else (_SAFE_R - box.right) if box.right > _SAFE_R else 0.0
            dy = (_SAFE_B - box.bottom) if box.bottom < _SAFE_B else (_SAFE_T - box.top) if box.top > _SAFE_T else 0.0
            x, y = obj.position
            obj.position = (round(x + dx, 3), round(y + dy, 3))
            obj.scale = new_scale
            touched.add(oid)
            changes.append({"object": oid, "lane": lane, "position": list(obj.position),
                            "scale": obj.scale, "reason": o["summary"]})

        elif lane == "collision" and geo.measured:
            # move the label (the text-like side) to the nearest free spot
            cand = [c for c in (oid, o.get("other")) if c and scene.get_obj(c)
                    and scene.get_obj(c).kind in TEXTLIKE and c not in touched]
            target = cand[0] if cand else None
            if target is None:
                unresolved.append(o)
                continue
            tobj = scene.get_obj(target)
            box = geo.bbox(target)
            spot = _free_spot(scene, geo, target, box) if box else None
            if spot is None:
                unresolved.append(o)
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

        elif lane == "legibility" and obj is not None:
            if obj.kind == "text":
                obj.payload["font_size"] = max(MIN_FONT_SIZE / obj.scale, float(obj.payload.get("font_size", 36)))
                changes.append({"object": oid, "lane": lane, "font_size": obj.payload["font_size"],
                                "reason": o["summary"]})
            elif obj.kind == "equation":
                # glyph size, like the check: MathTex is 48pt at scale 1
                obj.scale = max(obj.scale, round(1.25 * MIN_FONT_SIZE / 48, 3))
                changes.append({"object": oid, "lane": lane, "scale": obj.scale, "reason": o["summary"]})
            else:
                unresolved.append(o)
        else:
            unresolved.append(o)
    return changes, unresolved
