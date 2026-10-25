"""Geometry probe: build a generated scene's Mobjects and *measure* them.

Run in a subprocess (it imports Manim):

    python -m chalkcast.probe <generated_module.py> <spec.json>

It calls the module's ``_cc_build()`` -- the very function the rendered scene
uses -- without playing any animation, so it is fast (no video encoding) and
what it reports is exactly what will be drawn. Output (JSON on stdout):

* ``objects``: exact bounding boxes of every object (and of every scene-parameter
  state the timeline visits, for objects that move);
* ``plots``: per graph, a polyline sample in scene coordinates (for curve/label
  collisions), the fraction clipped to the axes box, and a *fidelity* score: the
  worst deviation between the drawn Bezier curve and the true function, as a
  fraction of the y-axis span (catches spline overshoot and aliasing);
* ``ink``: outline samples of shapes (for shape/label collisions);
* ``ok: false`` + ``error`` when construction fails (LaTeX errors, bad
  expressions, ...), which the compile lane reports without a full render.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import traceback
from typing import Any


def _bbox(m: Any) -> list[float] | None:
    import numpy as np

    if m is None or not len(m.get_family()) or m.get_all_points().shape[0] == 0:
        return None
    pts = m.get_all_points()
    if not np.isfinite(pts).all():
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] == 0:
            return None
    return [float(pts[:, 0].min()), float(pts[:, 0].max()),
            float(pts[:, 1].min()), float(pts[:, 1].max())]


def _pieces(graph: Any) -> list[Any]:
    """The ParametricFunction pieces of a (possibly clipped / redrawn) graph."""
    out = []
    for m in graph.get_family():
        if len(m.points) >= 4 and not m.submobjects:
            out.append(m)
    return out


def _fidelity(ax: Any, f: Any, graph: Any) -> dict[str, Any]:
    import numpy as np

    y0, y1 = float(ax.y_range[0]), float(ax.y_range[1])
    span = (y1 - y0) or 1.0
    worst, worst_x, n_seg = 0.0, None, 0
    samples: list[list[float]] = []
    for piece in _pieces(graph):
        pts = piece.points
        n = len(pts) // 4
        for k in range(n):
            p0, p1, p2, p3 = pts[4 * k: 4 * k + 4]
            n_seg += 1
            for t in (0.25, 0.5, 0.75):
                q = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3
                xq, yq = ax.point_to_coords(q)[:2]
                xa, xb = ax.point_to_coords(p0)[0], ax.point_to_coords(p3)[0]
                lo, hi = (xa, xb) if xa <= xb else (xb, xa)
                fine = np.linspace(lo, hi, 33)
                with np.errstate(all="ignore"):
                    try:
                        fv = np.asarray([float(f(v)) for v in fine])
                        ftrue = float(f(xq))
                    except Exception:
                        continue
                if not (np.isfinite(fv).all() and np.isfinite(ftrue)):
                    continue
                steps = np.abs(np.diff(fv))
                # a discontinuity is one big step with little variation elsewhere;
                # an oscillation the segment skips over (aliasing) varies everywhere.
                jump = steps.max() > 0.01 * span and steps.sum() - steps.max() < steps.max()
                if jump:
                    # a drawn segment that spans a discontinuity is a riser the function
                    # does not have (curves are split at jumps unless smooth=True is
                    # forced): measure the distance to the nearest true value nearby
                    err = float(np.abs(fv - yq).min())
                else:
                    err = abs(yq - ftrue)
                rel = err / span
                if rel > worst:
                    worst, worst_x = rel, float(xq)
            if k % max(1, n // 150) == 0:
                samples.append([float(p0[0]), float(p0[1])])
        if len(pts):
            samples.append([float(pts[-1][0]), float(pts[-1][1])])
    return {"max_error": worst, "at_x": worst_x, "segments": n_seg, "points": samples}


def _domain(ax: Any, f: Any) -> tuple[float, float]:
    """(fraction clipped by the y-range, fraction where the function is defined)."""
    import numpy as np

    x0, x1 = float(ax.x_range[0]), float(ax.x_range[1])
    y0, y1 = float(ax.y_range[0]), float(ax.y_range[1])
    xs = np.linspace(x0, x1, 801)
    with np.errstate(all="ignore"):
        ys = []
        for v in xs:
            try:
                ys.append(float(f(v)))
            except Exception:
                ys.append(np.nan)
    ys = np.asarray(ys)
    margin = 0.02 * (y1 - y0)
    # values outside the axes box; undefined points (NaN: outside the domain, e.g.
    # sqrt(x) for x<0) are not "clipped" -- the function simply does not exist there.
    fin = np.isfinite(ys)
    outside = fin & ((ys < y0 - margin) | (ys > y1 + margin))
    return float(outside.mean()), float(fin.mean())


def _path_samples(m: Any, per_leaf: int = 24) -> list[list[float]]:
    """Points along every stroked path of a mobject (axis lines, tips, ticks, curves).

    Every member with its own points counts (an Arrow is a line whose tip is a
    submobject). Each cubic segment is evaluated at enough parameters that a leaf
    gets at least ``per_leaf`` points, so a straight arrow or a rectangle side --
    one segment, two anchors -- is sampled along its length, not just at its ends.
    Vectorised: point_from_proportion would re-measure the path for every point.
    """
    import numpy as np

    out: list[list[float]] = []
    for leaf in m.get_family():
        pts = np.asarray(leaf.points)
        n = len(pts) // 4
        if n == 0:
            continue
        seg = pts[: 4 * n].reshape(n, 4, 3)
        k = max(2, -(-per_leaf // n))  # parameters per segment (ceil)
        t = np.linspace(0, 1, k)[None, :, None]
        p0, p1, p2, p3 = (seg[:, i][:, None, :] for i in range(4))
        q = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + t ** 3 * p3
        q = q.reshape(-1, 3)
        q = q[np.isfinite(q).all(axis=1)]
        out.extend([float(a), float(b)] for a, b, _ in q)
    return out


def _ink(m: Any, kind: str) -> list[list[float]]:
    """Outline sample of a shape; filled dots are represented by their bbox grid."""
    import numpy as np

    pts = m.get_all_points()
    if pts.shape[0] == 0:
        return []
    if m.get_fill_opacity() > 0.5:  # filled (dot): sample the whole disc
        box = _bbox(m)
        if box is None:  # no finite point (anchor undefined): nothing drawn
            return []
        l, r, b, t = box
        return [[float(x), float(y)] for x in np.linspace(l, r, 5) for y in np.linspace(b, t, 5)]
    # along the path, not just its control points: a straight arrow or a rectangle
    # side has only its endpoints and two handles, which miss a label in the middle
    return _path_samples(m, per_leaf=160)


def _latex_errors(spec: dict[str, Any]) -> list[str]:
    """LaTeX errors in this scene's equations, from a strict compile of each one.

    Manim compiles in batch mode, where LaTeX recovers from many errors (a missing
    brace, an unknown command) and still emits output -- so garbled math renders
    without any complaint and the log is discarded. We re-compile each equation
    with the same template under ``-halt-on-error`` (cached by content hash) and
    report the first ``!`` line of the log.
    """
    import hashlib
    import shutil
    import subprocess
    from pathlib import Path

    from manim import config

    latex = shutil.which("latex")
    if latex is None:
        return []
    cache = Path(".texcheck")
    cache.mkdir(exist_ok=True)
    out = []
    items = []
    for o in spec["objects"]:
        if o["kind"] == "equation" and o.get("tex"):
            items.append((o["id"], o["tex"]))
        for k, lab in (o.get("labels") or {}).items():  # axis labels are MathTex too
            items.append((f"{o['id']} {k}", lab))
    for oid, tex in items:
        o = {"id": oid}
        code = config.tex_template.get_texcode_for_expression_in_env(tex, "align*")
        h = hashlib.sha1(code.encode()).hexdigest()[:16]
        ok_marker, err_marker = cache / f"{h}.ok", cache / f"{h}.err"
        if ok_marker.exists():
            continue
        if err_marker.exists():
            out.append(f"{o['id']}: {err_marker.read_text()}")
            continue
        src = cache / f"{h}.tex"
        src.write_text(code, encoding="utf-8")
        proc = subprocess.run([latex, "-interaction=nonstopmode", "-halt-on-error", src.name],
                              cwd=str(cache), capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            ok_marker.write_text("")
            continue
        log = (cache / f"{h}.log")
        lines = log.read_text(errors="replace").splitlines() if log.exists() else proc.stdout.splitlines()
        msg = "unknown LaTeX error"
        for k, line in enumerate(lines):
            if line.startswith("! "):
                ctx = " ".join(ln.strip() for ln in lines[k + 1:k + 3] if ln.strip())
                msg = f"{line[2:].strip()} {ctx}".strip()
                break
        err_marker.write_text(msg)
        out.append(f"{o['id']}: {msg}")
    return out


def measure(module_path: str, spec: dict[str, Any]) -> dict[str, Any]:
    spec_mod = importlib.util.spec_from_file_location("_cc_scene", module_path)
    mod = importlib.util.module_from_spec(spec_mod)
    spec_mod.loader.exec_module(mod)  # type: ignore[union-attr]
    try:
        ns = mod._cc_build()
    except Exception as exc:
        # Manim's own messages for a LaTeX failure are misleading ("installation does
        # not support converting .dvi files", an IndexError, a log path): name the
        # equation and the first LaTeX error instead, when the strict check finds it
        errors = _latex_errors(spec)
        if errors:
            return {"ok": False, "error": "LaTeX error: " + "; ".join(errors),
                    "traceback": "\n".join(errors)}
        raise exc
    P = ns.get("P", {})

    objects: dict[str, Any] = {}
    plots: dict[str, Any] = {}
    ink: dict[str, Any] = {}
    for o in spec["objects"]:
        oid, kind = o["id"], o["kind"]
        m = ns.get(oid)
        if m is None:
            continue
        for mob in m.get_family():  # settle updaters (anchored shapes, redraws)
            mob.update(0)
        objects[oid] = {"kind": kind, "bbox": _bbox(m)}
        if kind == "plot":
            ax = ns[f"{oid}_ax"]
            curves = []
            for i in range(1, o.get("n_functions", 0) + 1):
                f, g = ns[f"{oid}_f{i}"], ns[f"{oid}_g{i}"]
                fid = _fidelity(ax, f, g)
                fid["fn"] = i
                fid["clipped_fraction"], fid["defined_fraction"] = _domain(ax, f)
                curves.append(fid)
            # axis labels are ink a note must not sit on, like the axes themselves
            labels = ns.get(f"{oid}_labels")
            plots[oid] = {"axes_bbox": _bbox(ax), "curves": curves,
                          # an axis line is one cubic segment: sample it every few hundredths
                          "axes_points": _path_samples(ax, per_leaf=400) +
                          (_path_samples(labels) if labels else [])}
        elif kind == "shape":
            ink[oid] = _ink(m, kind)

    latex_errors = _latex_errors(spec)
    if latex_errors:
        return {"ok": False, "error": "LaTeX error: " + "; ".join(latex_errors),
                "traceback": "\n".join(latex_errors)}

    # Re-measure moving objects at every parameter value the timeline visits.
    states = []
    for st in spec.get("param_states", []):
        for name, val in st["params"].items():
            if name in P:
                P[name].set_value(val)
        boxes, ink_at, curves_at = {}, {}, {}
        for o in spec["objects"]:
            m = ns.get(o["id"])
            if m is None or not o.get("dynamic"):
                continue
            for mob in m.get_family():
                mob.update(0)
            m.update(0)
            boxes[o["id"]] = _bbox(m)
            # what a label must not sit on at this moment
            if o["kind"] == "plot":
                pts, curves = [], []
                for i in range(1, o.get("n_functions", 0) + 1):
                    g = ns[f"{o['id']}_g{i}"]
                    pts += _path_samples(g, per_leaf=200)
                    if st.get("beat_end"):  # fidelity/clipping where each tween lands
                        ax, f = ns[f"{o['id']}_ax"], ns[f"{o['id']}_f{i}"]
                        fid = _fidelity(ax, f, g)
                        clipped, defined = _domain(ax, f)
                        curves.append({"fn": i, "max_error": fid["max_error"], "at_x": fid["at_x"],
                                       "clipped_fraction": clipped, "defined_fraction": defined})
                ink_at[o["id"]] = pts
                if curves:
                    curves_at[o["id"]] = curves
            elif o["kind"] == "shape":
                ink_at[o["id"]] = _ink(m, o["kind"])
        if boxes:
            states.append({"step": st["step"], "params": st["params"], "frac": st.get("frac", 1.0),
                           "objects": boxes,
                           "ink": ink_at, "curves": curves_at, "beat_end": st.get("beat_end", False)})
    return {"ok": True, "objects": objects, "plots": plots, "ink": ink, "states": states}


def main(argv: list[str]) -> int:
    module_path, spec_path = argv[1], argv[2]
    with open(spec_path, encoding="utf-8") as fh:
        spec = json.load(fh)
    try:
        out = measure(module_path, spec)
    except Exception as exc:  # noqa: BLE001 - every failure is a compile finding
        tb = traceback.format_exc()
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": tb[-2500:]}
    sys.stdout.write("\n@@CHALKCAST_PROBE@@" + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
