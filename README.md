<div align="center">

# Chalkcast

### Describe a concept. Get a *verified* animated explainer.

Chalkcast is an [MCP](https://modelcontextprotocol.io) server that lets an agent
author, render, **and visually verify** [Manim](https://www.manim.community/)
animations — so the output is a video that's actually correct on screen, not a
plausible script that renders wrong.

![Chalkcast demo](assets/demo.gif)

<sub>Authored, rendered, and verified entirely through the MCP server — the
chalkboard theme, a typeset LaTeX integral, the parabola, and a highlighted point,
from a handful of tool calls.</sub>

</div>

---

## Why another Manim MCP?

Existing Manim MCP servers give the agent **one tool**: *run this code, return the
video.* The agent writes Manim blind and hopes — but Manim is unforgiving (a LaTeX
typo, an off-screen label, two terms overlapping, a transition firing before its
target exists). The video "renders" and is still wrong, and the agent has no way to
know.

**Chalkcast gives the agent eyes and a correction loop:**

```
author scene ──► render ──► inspect (verification lanes) ──► verdict
     ▲                                                          │
     └──────────────── fix obligations ◄──── needs_fix ─────────┘
```

Every render returns **evidence** (the artifact + sampled frames) and a structured
**verdict**. Problems become **obligations** that block export until resolved. See
[`docs/SPEC.md`](docs/SPEC.md) for the full design.

## Install

```bash
pip install "chalkcast[render]"   # server + Manim
# you also need a LaTeX distribution for math, and ffmpeg for GIF export
```

Register with an MCP client (e.g. Claude Code):

```bash
claude mcp add chalkcast -- chalkcast serve
```

Check what's available on your machine:

```bash
chalkcast doctor   # reports Manim / LaTeX / ffmpeg / vision providers
chalkcast demo     # builds + verifies (+ renders) the demo scene
```

## What the agent gets (v0.1 tools)

| Group | Tools |
|---|---|
| **Session** | `open_project` · `inspect_project` · `close_project` |
| **Authoring** | `add_scene` · `add_equation` · `add_plot` · `add_text` · `add_shape` · `add_step` |
| **Rendering** | `render_scene` · `render_frame` |
| **Verification** | `inspect_render` (framing · collision · legibility · timing · compile lanes) |
| **Export** | `export_gif` |
| **Discovery** | `inspect_environment` |

## Example (what the agent does)

```
open_project(theme="chalkboard")
add_scene(title="Area under a parabola")
add_text("Area under y = x^2", y=3)
add_plot(functions=[{"expr":"x**2","color":"YELLOW"}], x_range=[0,2,1], y=-0.8)
add_equation(r"\int_0^1 x^2 \, dx = \tfrac{1}{3}", y=2)
add_step("write", targets=["text_1"])
add_step("write", targets=["plot_1"])
add_step("write", targets=["equation_1"])
add_step("highlight", targets=["equation_1"])
render_scene()      # -> video + frames + verdict + obligations
export_gif()        # -> shareable GIF (blocked if error obligations remain)
```

## Status

v0.1 baseline. Vision-based `intent` verification and auto-`fix_scene` land in v0.2
(see the roadmap in [`docs/SPEC.md`](docs/SPEC.md)).

## License

MIT.
