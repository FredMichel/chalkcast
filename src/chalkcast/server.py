"""Chalkcast MCP server (v0.1).

Exposes the baseline roadmap tools, grouped by capability with ``verb_object``
names, each returning the common evidence-bearing envelope (SPEC §5).
"""

from __future__ import annotations

from typing import Any

try:  # mcp >= 2.0: FastMCP was renamed to MCPServer
    from mcp.server.mcpserver import MCPServer as _Server
except ModuleNotFoundError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from . import render as rnd
from . import session
from .envelope import err, ok
from .model import Project, Scene
from .verify import GEOMETRIC_LANES, inspect_scene

mcp = _Server("chalkcast")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _require_scene(project_id: str | None, scene_id: str | None) -> tuple[Project, Scene] | dict[str, Any]:
    project = session.get_project(project_id)
    if project is None:
        return err("no open project; call open_project first")
    scene = project.get_scene(scene_id)
    if scene is None:
        return err("no such scene; call add_scene first")
    return project, scene


def _register_obligations(project: Project, scene_id: str, raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace this scene's open obligations with a freshly measured set."""
    project.obligations = [
        o for o in project.obligations if not (o["scene"] == scene_id and o["status"] == "open")
    ]
    registered = []
    for o in raw:
        o = {**o, "id": project.next_obligation_id()}
        project.obligations.append(o)
        registered.append(o)
    return registered


# ---------------------------------------------------------------------------
# 6.1 Session & project
# ---------------------------------------------------------------------------

@mcp.tool()
def open_project(quality: str = "low", fps: int = 30, theme: str = "chalkboard") -> dict[str, Any]:
    """Start a project session. quality: draft|low|medium|high. theme: chalkboard|whiteboard|dark."""
    project = session.create_project(quality=quality, fps=fps, theme=theme)
    return ok({"project_id": project.id, "quality": quality, "theme": theme},
              coverage={"scope": project.id})


@mcp.tool()
def inspect_project(project_id: str | None = None) -> dict[str, Any]:
    """Current project state: scenes, steps, and open obligations."""
    project = session.get_project(project_id)
    if project is None:
        return err("no open project")
    return ok(project.to_dict(),
              obligations=[o for o in project.obligations if o["status"] == "open"],
              coverage={"scope": project.id})


@mcp.tool()
def close_project(project_id: str | None = None) -> dict[str, Any]:
    """Release the project session."""
    closed = session.close_project(project_id)
    return ok({"closed": closed})


# ---------------------------------------------------------------------------
# 6.2 Authoring
# ---------------------------------------------------------------------------

@mcp.tool()
def add_scene(title: str = "", project_id: str | None = None) -> dict[str, Any]:
    """Append a scene to the project."""
    project = session.get_project(project_id)
    if project is None:
        return err("no open project; call open_project first")
    scene = project.new_scene(title=title)
    return ok({"scene_id": scene.id, "title": title}, coverage={"scope": scene.id})


@mcp.tool()
def add_equation(tex: str, x: float = 0, y: float = 0, color: str | None = None,
                 scale: float = 1.0, scene_id: str | None = None,
                 project_id: str | None = None) -> dict[str, Any]:
    """Add a LaTeX equation (e.g. tex=r'e^{i\\pi}+1=0')."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    oid = scene.new_obj_id("equation")
    from .model import Obj
    scene.objects.append(Obj(id=oid, kind="equation", position=(x, y), color=color,
                             scale=scale, payload={"tex": tex}))
    return ok({"object_id": oid},
              limitations=["LaTeX validity is confirmed only at render time"],
              untrusted=["tex"])


@mcp.tool()
def add_plot(functions: list[dict[str, Any]] | None = None,
             x_range: list[float] | None = None, y_range: list[float] | None = None,
             x: float = 0, y: float = 0, width: float = 8.0, height: float = 5.0,
             scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Add axes with one or more functions. functions=[{'expr':'x**2','color':'YELLOW'}]."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    oid = scene.new_obj_id("plot")
    from .model import Obj
    scene.objects.append(Obj(id=oid, kind="plot", position=(x, y), payload={
        "functions": functions or [{"expr": "x**2"}],
        "x_range": x_range or [-5, 5, 1],
        "y_range": y_range or [-3, 3, 1],
        "width": width, "height": height,
    }))
    return ok({"object_id": oid}, untrusted=["expr"],
              limitations=["function expressions are evaluated by Manim at render time"])


@mcp.tool()
def add_text(text: str, x: float = 0, y: float = 0, font_size: float = 36,
             color: str | None = None, scale: float = 1.0,
             scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Add a title/label/callout."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    oid = scene.new_obj_id("text")
    from .model import Obj
    scene.objects.append(Obj(id=oid, kind="text", position=(x, y), color=color,
                             scale=scale, payload={"text": text, "font_size": font_size}))
    return ok({"object_id": oid})


@mcp.tool()
def add_shape(shape: str = "circle", size: float = 1.0, x: float = 0, y: float = 0,
              color: str | None = None, width: float | None = None, height: float | None = None,
              scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Add primitive geometry: circle|square|rectangle|triangle|arrow|dot|line."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    oid = scene.new_obj_id("shape")
    payload: dict[str, Any] = {"shape": shape, "size": size}
    if width is not None:
        payload["width"] = width
    if height is not None:
        payload["height"] = height
    from .model import Obj
    scene.objects.append(Obj(id=oid, kind="shape", position=(x, y), color=color, payload=payload))
    return ok({"object_id": oid})


@mcp.tool()
def add_step(kind: str, targets: list[str] | None = None, to: str | None = None,
             duration: float = 1.0, note: str = "", zoom: float | None = None,
             focus_x: float = 0, focus_y: float = 0,
             scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Append a timed beat. kind: write|transform|highlight|fade|camera_move|wait.
    For transform, set `to` (destination object id). For camera_move, set zoom/focus."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    _, scene = got
    if kind == "camera_move":
        to = f"{zoom or 1}@{focus_x},{focus_y}"
    step = scene.add_step(kind, targets=targets or [], to=to, duration=duration, note=note)
    return ok({"step_index": step.index, "kind": kind})


# ---------------------------------------------------------------------------
# 6.3 Rendering + 6.4 Verification
# ---------------------------------------------------------------------------

@mcp.tool()
def render_scene(scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Render one scene to video + sampled frames, then run verification lanes."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    project, scene = got
    wd = session.workdir(project.id, scene.id)
    report = rnd.render_scene(project, scene, wd)
    session.set_last_render(project.id, scene.id, report)

    insp = inspect_scene(scene, render_report=report)
    obls = _register_obligations(project, scene.id, insp["obligations"])

    if not report.get("ok"):
        return err("render failed", payload={"scene_id": scene.id},
                   timing=report.get("timing", {}),
                   coverage={"lanes_run": insp["lanes_run"], "lanes_skipped": insp["lanes_skipped"],
                             "scope": scene.id},
                   verdict=insp["verdict"], obligations=obls,
                   limitations=insp["limitations"] + [str(report.get("error", "")),
                                                       report.get("stderr_tail", "")],
                   untrusted=["manim_stderr", "latex_log"])
    return ok({"scene_id": scene.id, "artifact": report.get("artifact")},
              artifacts=[report["artifact"]] if report.get("artifact") else [],
              frames=report.get("frames", []), timing=report.get("timing", {}),
              coverage={"lanes_run": insp["lanes_run"], "lanes_skipped": insp["lanes_skipped"],
                        "scope": scene.id},
              verdict=insp["verdict"], obligations=obls,
              limitations=insp["limitations"], untrusted=["manim_stderr", "latex_log"])


@mcp.tool()
def render_frame(scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Fast single-still preview of a scene (last frame)."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    project, scene = got
    wd = session.workdir(project.id, scene.id)
    report = rnd.render_scene(project, scene, wd, single_frame=True)
    if not report.get("ok"):
        return err("preview failed", limitations=[str(report.get("error", "")),
                                                   report.get("stderr_tail", "")],
                   untrusted=["manim_stderr"])
    return ok({"scene_id": scene.id}, frames=report.get("frames", []),
              timing=report.get("timing", {}))


@mcp.tool()
def inspect_render(scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Run the geometric verification lanes on a scene and return a verdict + obligations.
    Uses the latest render (if any) for the compile lane; otherwise runs graph-only lanes."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    project, scene = got
    report = session.get_last_render(project.id, scene.id)
    insp = inspect_scene(scene, render_report=report)
    obls = _register_obligations(project, scene.id, insp["obligations"])
    return ok({"scene_id": scene.id},
              coverage={"lanes_run": insp["lanes_run"], "lanes_skipped": insp["lanes_skipped"],
                        "scope": scene.id, "available_lanes": GEOMETRIC_LANES},
              verdict=insp["verdict"], obligations=obls, limitations=insp["limitations"])


# ---------------------------------------------------------------------------
# 6.5 Export
# ---------------------------------------------------------------------------

@mcp.tool()
def export_gif(scene_id: str | None = None, project_id: str | None = None) -> dict[str, Any]:
    """Export the latest render of a scene as a shareable GIF (blocked by open error obligations)."""
    got = _require_scene(project_id, scene_id)
    if isinstance(got, dict):
        return got
    project, scene = got
    blocking = [o for o in project.obligations
                if o["scene"] == scene.id and o["status"] == "open" and o["severity"] == "error"]
    if blocking:
        return err("export blocked by open error obligations; fix them or re-render first",
                   obligations=blocking)
    report = session.get_last_render(project.id, scene.id)
    if not report or not report.get("ok") or not report.get("artifact"):
        return err("no rendered video for this scene; call render_scene first")
    from pathlib import Path
    res = rnd.export_gif(Path(report["artifact"]), session.workdir(project.id, scene.id))
    if not res.get("ok"):
        return err(str(res.get("error")), limitations=[res.get("stderr_tail", "")])
    return ok({"scene_id": scene.id, "gif": res["artifact"]}, artifacts=[res["artifact"]])


# ---------------------------------------------------------------------------
# 6.6 Discovery
# ---------------------------------------------------------------------------

@mcp.tool()
def inspect_environment() -> dict[str, Any]:
    """Which providers are present (Manim, LaTeX, ffmpeg, vision) and what degrades without each."""
    env = rnd.environment()
    limits = []
    if not env["manim"]:
        limits.append("manim missing: no video output (authoring + geometric verification still work)")
    if not env["latex"]:
        limits.append("latex missing: equations unavailable")
    if not env["ffmpeg"]:
        limits.append("ffmpeg missing: GIF/thumbnail export disabled")
    return ok(env, limitations=limits)


def run() -> None:
    mcp.run()


if __name__ == "__main__":
    run()
