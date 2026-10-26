<div align="center">

# Chalkcast

### Describe a concept. Get a *verified* animated explainer.

Chalkcast is an [MCP](https://modelcontextprotocol.io) server that lets an agent
author, render, **and visually verify** [Manim](https://www.manim.community/)
animations — so the output is a video that's actually correct on screen, not a
plausible script that renders wrong.

![license](https://img.shields.io/badge/license-MIT-blue)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![status](https://img.shields.io/badge/status-alpha-orange)
![MCP](https://img.shields.io/badge/MCP-server-8A2BE2)

<!-- mcp-name: io.github.FredMichel/chalkcast -->

![Chalkcast showcase](assets/showcase.gif)

<sub>One continuous animation, authored, verified and rendered entirely through the
MCP server (passed on the first render): a Fourier series grows harmonic by harmonic
into a square wave, and the Gibbs overshoot (the true one, ≈ 1.179) stays at the
edges. Reproduce it with <code>python examples/showcase.py</code>.</sub>

</div>

---

## Who it's for

- **Educators & students** — turn "explain the Fourier series" into a clean
  animated explainer, without learning Manim or writing LaTeX by hand.
- **STEM content creators** — produce 3Blue1Brown-style animations in minutes
  instead of hours of coding and debugging renders.
- **Agents & tooling** — a typed, verifiable MCP surface an AI can drive end to
  end, with a verdict it can act on instead of a video it can't inspect.

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

**One command** (Claude Code; needs [uv](https://docs.astral.sh/uv/)):

```bash
claude mcp add chalkcast -- uvx --python 3.12 --from "chalkcast[render]" chalkcast serve
```

`--python 3.12` matters: Chalkcast needs Python ≥ 3.11, and uv downloads one if the
system Python is older (macOS ships 3.9). Then just ask in natural language.

Or with pip, into any Python ≥ 3.11 environment:

```bash
pip install "chalkcast[render]"
claude mcp add chalkcast -- chalkcast serve
```

`[render]` brings Manim and a bundled ffmpeg. Equations also need a **LaTeX**
distribution. `chalkcast doctor` checks everything and prints the exact fix command
for your OS:

```bash
chalkcast doctor               # Manim / LaTeX / dvisvgm / physics.sty / ffmpeg
chalkcast demo                 # builds + verifies + renders the demo scene
```

> **macOS:** `pycairo` (pulled in by Manim) has no macOS wheel and builds from
> source, so it needs `brew install cairo pkg-config` first. On a machine where
> Manim's native deps were built by hand, see `scripts/dev-env.sh` and
> `scripts/serve.sh`.

## What the agent gets

| Group | Tools |
|---|---|
| **Session** | `open_project` · `inspect_project` · `close_project` |
| **Authoring** | `add_scene` · `add_equation` · `add_plot` · `add_area` · `add_riemann` · `add_text` · `add_shape` · `add_parameter` · `add_step` |
| **Editing** | `update_object` · `remove_object` (cascade) · `remove_step` |
| **Verification** | `inspect_render` (measured, no video) · `fix_scene` (auto-repair) |
| **Rendering** | `render_scene` · `render_frame` |
| **Export** | `export_gif` (blocked while error obligations remain) |
| **Discovery** | `inspect_environment` |

- **Continuous animation:** `add_parameter("t")`, use `t` in plot expressions or
  shape anchors (`at_x="t"`), then `add_step("tween", param="t", value=6.28)`.
  Curves, areas and anchored dots follow it frame by frame. `with_previous=True`
  plays a step at the same time as the one before.
- **Fail fast, nothing implicit:** expressions are validated when authored (an AST
  whitelist: x, math functions, `np.<fn>`, declared parameters). Colors, ranges,
  step targets, timeline order, and arguments that don't apply to a shape or step
  are rejected with a message saying what to do. Edits are all-or-nothing. Side
  effects (cascade removals, regrouped steps, redeclared parameters, default
  ranges) are reported in the result. `export_gif` refuses a render that predates
  the latest edit. Twenty-eight independent audits (22, then 17, 14, 8, 4, 9, 4, 7, 5, 5, 7, 8, 5, 8, 7, 7, 10, 4, 5, 5, 8, 7, 7, 5, 7, 9, 7 and 4 findings; in the last ten, 9, 10, 8, 10, 9, 9, 9, 9, 9 and 10 of 10 fresh explainers reached a correct GIF within two renders; the last nineteen built realistic explainers end to end and compared verdicts with frames, one of them over the real MCP stdio protocol) are
  pinned one test per finding in `tests/test_ergonomics.py`.
- **Faithful plots:** dense sampling, curves split at jumps and poles (no invented
  risers or overshoot), clipping to the axes. A *fidelity* lane measures the drawn
  curve against the true function. Areas and Riemann sums are refused when they
  could not show the integral (function undefined or off the axes, y = 0 not
  visible).

### How verification works

`inspect_render` builds the real Manim objects without encoding video (1–3 s) and
**measures** them: exact bounding boxes, curve and axis ink, the drawn-vs-true error
of every graph, and a strict LaTeX compile (Manim's batch mode silently "recovers"
from LaTeX errors). Lanes: framing · collision · legibility · timing · compile ·
fidelity · range. `fix_scene` repairs the mechanical problems (moves, rescales,
resampling) and leaves the rest as obligations for the agent.

Measured on labelled scenes (`chalkcast bench`, `chalkcast bench --holdout N`).
Each held-out set was written and labelled (and checked frame by frame) by an
author who never read the verifier. **The first run of each set is the honest
out-of-sample figure**:

| Set | Cases | First run: detection / false positives | Bugs that run exposed |
|---|---|---|---|
| Held-out 1 | 39 (24 defective) | 95.8 % / 7.7 % | over-strict safe margin; aliasing taken for a jump |
| Held-out 2 | 47 (31 defective) | 90.3 % / 12.8 % | estimated noise after a compile error; straight arrows and box sides sampled only at their ends; equation size judged by box height |
| Held-out 3 | 44 (26 defective) | **96.2 % / 0 %** | an arrow's shaft was never sampled (only its tip) |
| Held-out 4 | 46 (25 defective) | **96.0 % / 0 %** | a label written *during* a zoom-in was excused as an intentional crop |

After each set's bugs were fixed, every set (and the 40-case development set)
scores 100 % / 0 %. From then on they are regression tests (`tests/test_measured.py`),
not independent measures. Held-out 3 was written after the fixes for sets 1 and 2,
and held-out 4 (continuous tweens, simultaneous beats, transforms, camera moves)
after eleven ergonomics audits; each was run once, untouched. Held-out 4 is the
current out-of-sample estimate.

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
inspect_render()    # -> measured verdict in ~2 s, no video
fix_scene()         # -> repairs framing / collision / fidelity if needed
render_scene()      # -> video + frames + verdict + obligations
export_gif()        # -> shareable GIF (blocked if error obligations remain)
```

## Status

Alpha. Scientific correctness is tested numerically (`tests/test_science.py`): shaded
areas equal their integrals, Riemann rectangles their sum, drawn curves their
function (including sin 30x), and the Gibbs peak its theoretical value. On
[PyPI](https://pypi.org/project/chalkcast/) and the
[MCP Registry](https://registry.modelcontextprotocol.io/) (`io.github.FredMichel/chalkcast`).
Vision-based `intent` verification is next (see [`docs/SPEC.md`](docs/SPEC.md)).

## License

MIT.
