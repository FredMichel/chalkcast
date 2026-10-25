#!/usr/bin/env python
"""Showcase: a Fourier series *growing* into a square wave, Gibbs overshoot included.

    python examples/showcase.py      # renders + exports assets/showcase.gif

One continuous animation: a scene parameter N is tweened from 0 to 15 and each
harmonic fades in as N passes it (clip(N-k, 0, 1)), so the partial sum morphs
smoothly instead of jumping between frames. Everything goes through the real MCP
server (the same tools an agent calls), including inspect -> fix_scene -> render.
"""
import asyncio
from pathlib import Path

from chalkcast.server import mcp


def _unwrap(res):
    for a in ("structuredContent", "structured_content"):
        v = getattr(res, a, None)
        if v:
            return v
    return res[1] if isinstance(res, tuple) else res


async def call(tool, **args):
    out = _unwrap(await mcp.call_tool(tool, args))
    if not out.get("ok"):
        raise SystemExit(f"{tool} failed: {out.get('error')}  {out.get('obligations', '')}")
    return out


def oid(result):
    return result["result"]["object_id"]


async def build():
    await call("open_project", theme="chalkboard", quality="medium")
    await call("add_scene", title="Fourier series to a square wave")
    await call("add_parameter", name="N", value=0.0)

    title = oid(await call("add_text", text="Fourier series of a square wave", y=3.35, font_size=36))
    series = oid(await call("add_equation", y=2.35, scale=0.75,
                            tex=r"S_N(x)=\frac{4}{\pi}\sum_{k<N}\frac{\sin\big((2k+1)x\big)}{2k+1}"))
    plot = oid(await call("add_plot", x_range=[-6.5, 6.5, 1], y_range=[-1.6, 1.6, 1],
                          width=11, height=4.2, y=-0.75, functions=[
                              {"expr": "sign(sin(x))", "color": "GRAY"},
                              {"expr": "4/pi*sum(clip(N-k, 0, 1)*sin((2*k+1)*x)/(2*k+1) "
                                       "for k in range(15))", "color": "YELLOW"}]))
    gibbs = oid(await call("add_text", text="Gibbs: a 9% overshoot that never goes away",
                           y=-3.4, font_size=28, color="ORANGE"))

    await call("add_step", kind="write", targets=[title], duration=1.0)
    await call("add_step", kind="write", targets=[plot], duration=1.5)
    await call("add_step", kind="write", targets=[series], duration=1.2)
    await call("add_step", kind="tween", param="N", value=1, duration=1.2)
    await call("add_step", kind="tween", param="N", value=15, duration=6.0)
    await call("add_step", kind="write", targets=[gibbs], duration=1.0)
    await call("add_step", kind="wait", duration=1.5)

    insp = await call("inspect_render")
    if insp["verdict"]["errors"]:
        await call("fix_scene")
    r = await call("render_scene")
    print("verdict:", r["verdict"])
    for o in r.get("obligations", []):
        print(f"  [{o['severity']}] {o['lane']}: {o['summary']}")

    g = await call("export_gif")
    out = Path(__file__).resolve().parent.parent / "assets" / "showcase.gif"
    out.write_bytes(Path(g["result"]["gif"]).read_bytes())
    print(f"\n✓ done -> {out}")


asyncio.run(build())
