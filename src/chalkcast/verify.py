"""Geometric verification lanes (Layer C, v0.1).

These lanes reason over the scene graph in Manim units -- no pixels required --
so they run on any machine, with or without a render engine. The ``intent`` lane
(frame-vs-description scoring by a vision model) is v0.2 and is reported as
skipped here.

Each detected problem becomes an *obligation* (see SPEC §7): ``error`` obligations
block export; ``warning``/``info`` can be accepted.
"""

from __future__ import annotations

from typing import Any

from .geometry import FRAME_HEIGHT, FRAME_WIDTH, within_safe_area
from .model import Scene

COLLISION_THRESHOLD = 0.35  # overlap as a fraction of the smaller object's area
MIN_FONT_SIZE = 20.0

# All geometric lanes available in v0.1. 'compile' runs only when a render report
# is supplied; 'intent' is v0.2.
GEOMETRIC_LANES = ["framing", "collision", "legibility", "timing"]


def _obl(scene_id: str, lane: str, severity: str, summary: str, fix: str, step: int | None = None) -> dict[str, Any]:
    return {
        "scene": scene_id,
        "step": step,
        "lane": lane,
        "severity": severity,
        "summary": summary,
        "suggested_fix": fix,
        "status": "open",
    }


def _lane_framing(scene: Scene) -> list[dict[str, Any]]:
    out = []
    for obj in scene.objects:
        cx, cy = obj.position
        # A center outside the frame is definitely wrong -> error.
        if abs(cx) > FRAME_WIDTH / 2 or abs(cy) > FRAME_HEIGHT / 2:
            out.append(_obl(
                scene.id, "framing", "error",
                f"object '{obj.id}' is positioned off-screen (center at {cx:.1f},{cy:.1f})",
                "move its position back inside the frame",
            ))
        # A bbox that crosses the safe area is only an *estimate* (font/LaTeX
        # width is unknown pre-render) -> warning, so it never blocks export.
        elif not within_safe_area(obj.estimate_bbox()):
            out.append(_obl(
                scene.id, "framing", "warning",
                f"object '{obj.id}' may extend outside the safe area (estimated)",
                "nudge it toward the center or reduce its scale if the render confirms it",
            ))
    return out


def _lane_collision(scene: Scene) -> list[dict[str, Any]]:
    out = []
    # area/riemann overlays are meant to sit on top of their parent plot
    objs = [o for o in scene.objects if o.kind not in ("area", "riemann")]
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            a, b = objs[i].estimate_bbox(), objs[j].estimate_bbox()
            ov = a.overlap_area(b)
            if ov <= 0:
                continue
            ratio = ov / min(a.area, b.area)
            if ratio >= COLLISION_THRESHOLD:
                out.append(_obl(
                    scene.id, "collision", "warning",
                    f"'{objs[i].id}' and '{objs[j].id}' overlap "
                    f"(~{ratio * 100:.0f}% of the smaller)",
                    "separate them, or stagger their steps so they are not visible together",
                ))
    return out


def _lane_legibility(scene: Scene) -> list[dict[str, Any]]:
    out = []
    for obj in scene.objects:
        if obj.kind == "text":
            fs = float(obj.payload.get("font_size", 36)) * obj.scale
            if fs < MIN_FONT_SIZE:
                out.append(_obl(
                    scene.id, "legibility", "warning",
                    f"text '{obj.id}' renders at ~{fs:.0f}pt (below {MIN_FONT_SIZE:.0f}pt)",
                    "increase font_size or scale",
                ))
    return out


def _lane_timing(scene: Scene) -> list[dict[str, Any]]:
    out = []
    defined = {o.id for o in scene.objects}
    introduced: set[str] = set()
    for step in scene.steps:
        # Every referenced target must be a defined object, or codegen emits an
        # undefined variable and the render crashes with a NameError.
        for tid in step.targets:
            if tid not in defined:
                out.append(_obl(
                    scene.id, "timing", "error",
                    f"step {step.index} ({step.kind}) targets unknown object '{tid}'",
                    "use an object id returned by an add_* call",
                    step=step.index,
                ))
        if step.kind == "write":
            introduced.update(t for t in step.targets if t in defined)
            continue
        # transform/highlight/fade must target an object already on screen.
        if step.kind in ("highlight", "fade", "transform"):
            for tid in step.targets:
                if tid in defined and tid not in introduced:
                    out.append(_obl(
                        scene.id, "timing", "error",
                        f"step {step.index} ({step.kind}) targets '{tid}' "
                        f"before it is introduced",
                        f"add a 'write' step for '{tid}' first",
                        step=step.index,
                    ))
            if step.kind == "transform" and step.to and step.to not in defined:
                out.append(_obl(
                    scene.id, "timing", "error",
                    f"step {step.index} transforms into undefined object '{step.to}'",
                    f"create object '{step.to}' before the transform",
                    step=step.index,
                ))
    return out


def inspect_scene(scene: Scene, render_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the geometric lanes (+ compile if a render report is given).

    Returns ``(verdict, obligations, lanes_run, lanes_skipped, limitations)``
    packed in a dict.
    """
    obligations: list[dict[str, Any]] = []
    lanes_run: list[str] = []
    limitations: list[str] = [
        "bounding boxes are estimated from the scene graph, not measured from "
        "rendered pixels; treat framing/collision as approximate",
    ]

    # compile lane (only with render evidence)
    if render_report is not None:
        lanes_run.append("compile")
        if not render_report.get("ok", False):
            obligations.append(_obl(
                scene.id, "compile", "error",
                "render did not complete: " + str(render_report.get("error", "unknown")),
                "fix the reported Manim/LaTeX error",
            ))

    for lane, fn in (
        ("framing", _lane_framing),
        ("collision", _lane_collision),
        ("legibility", _lane_legibility),
        ("timing", _lane_timing),
    ):
        lanes_run.append(lane)
        obligations.extend(fn(scene))

    lanes_skipped = ["intent"]
    limitations.append("intent lane (vision) is v0.2 and was skipped")

    errors = [o for o in obligations if o["severity"] == "error"]
    warnings = [o for o in obligations if o["severity"] == "warning"]
    # Simple score: start at 1.0, subtract per issue, floored at 0.
    score = max(0.0, 1.0 - 0.25 * len(errors) - 0.1 * len(warnings))
    status = "needs_fix" if errors else ("pass_with_warnings" if warnings else "pass")

    return {
        "verdict": {"status": status, "score": round(score, 2),
                     "errors": len(errors), "warnings": len(warnings)},
        "obligations": obligations,
        "lanes_run": lanes_run,
        "lanes_skipped": lanes_skipped,
        "limitations": limitations,
    }
