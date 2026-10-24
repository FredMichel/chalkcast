#!/usr/bin/env python
"""Your own Chalkcast scene — edit the EDIT block, then run:

    source scripts/dev-env.sh        # once per terminal (venv + cairo + ffmpeg + LaTeX)
    python examples/my_scene.py      # renders + exports examples/my_scene.gif

Everything goes through the real MCP server (the same tools an AI agent calls).
Objects return an id; use that id in the steps.
"""
import asyncio
from pathlib import Path

from chalkcast.server import mcp


# --- tiny helper so calls read nicely --------------------------------------
def _unwrap(res):
    for a in ("structuredContent", "structured_content"):
        v = getattr(res, a, None)
        if v:
            return v
    return res[1] if isinstance(res, tuple) else res


async def call(name, **args):
    out = _unwrap(await mcp.call_tool(name, args))
    if not out.get("ok"):
        raise SystemExit(f"{name} failed: {out.get('error')}  {out.get('obligations', '')}")
    return out


def oid(result):  # pull the object_id out of an add_* result
    return result["result"]["object_id"]


async def build():
    # =======================================================================
    # EDIT FROM HERE
    # =======================================================================
    await call("open_project", theme="chalkboard", quality="medium")  # theme: chalkboard|whiteboard|dark
    await call("add_scene", title="My scene")

    # --- add objects (each returns an id you reuse in steps) ---
    title = oid(await call("add_text", text="Euler's identity", y=3.0, font_size=40))

    eq = oid(await call("add_equation", tex=r"e^{i\pi} + 1 = 0", y=0.5))

    # a plot with one function (expr is Python math on x):
    plot = oid(await call("add_plot",
                          functions=[{"expr": "x**2", "color": "YELLOW"}],
                          x_range=[-2, 2, 1], y_range=[0, 4, 1], y=-1.2, width=6, height=3))

    # optional: shade the area under the plotted function (the integral)
    # area = oid(await call("add_area", plot_id=plot, x_min=0, x_max=1, color="BLUE", opacity=0.6))

    # a shape: circle | square | rectangle | triangle | arrow | dot | line
    # dot = oid(await call("add_shape", shape="dot", x=1, y=-0.5, color="RED"))

    # --- timeline: order matters; write an object before you highlight it ---
    await call("add_step", kind="write", targets=[title], duration=1.0)
    await call("add_step", kind="write", targets=[plot], duration=1.4)
    await call("add_step", kind="write", targets=[eq], duration=1.0)
    await call("add_step", kind="highlight", targets=[eq], duration=1.0)
    # await call("add_step", kind="transform", targets=[a], to=b, duration=1.2)  # morph a into b
    # await call("add_step", kind="camera_move", zoom=1.5, focus_x=0, focus_y=0, duration=1.0)
    await call("add_step", kind="wait", duration=0.5)
    # =======================================================================
    # EDIT TO HERE
    # =======================================================================

    print("rendering (first run compiles LaTeX, give it a few seconds)...")
    r = await call("render_scene")
    print("verdict:", r["verdict"])
    for o in r.get("obligations", []):
        print(f"  [{o['severity']}] {o['lane']}: {o['summary']}")

    g = await call("export_gif")
    out = Path("examples/my_scene.gif")
    out.write_bytes(Path(g["result"]["gif"]).read_bytes())
    print(f"\n✓ done -> {out.resolve()}")


asyncio.run(build())
