# Chalkcast — Design Specification

> **Chalkcast turns a natural-language concept into a *verified* animated explainer.**
> An agent describes what to teach; Chalkcast authors, renders, and visually
> verifies a [Manim](https://www.manim.community/) animation — then self-corrects
> until the frame actually matches the intent.

- **Status:** v0.1 implemented, plus measured verification, `fix_scene`, editing
  tools and continuous parameters (see roadmap).
- **Audience:** educators, students, STEM content creators, and the agents working for them.
- **License (intended):** MIT.
- **Positioning:** Chalkcast is a *bridge between agents and Manim*. It is not "a
  tool that runs Manim code" — a dozen of those already exist and plateau around a
  few hundred stars because the agent is **blind**: it emits code and hopes. Chalkcast
  gives the agent **eyes and a correction loop**, so the output is a finished,
  on-screen-correct video rather than a plausible script.

This document is the entry point. It is deliberately modeled on the conventions of
the REA MCP server: capability-grouped tools, descriptive `verb_object` names, a
persistent session model, and — above all — **evidence over assertion**: every tool
result carries what it produced, its coverage, and its limitations.

---

## 1. Design principles

These mirror REA's and are the contract every tool must honor.

1. **Local-first.** Rendering happens on the user's machine with their Manim/LaTeX
   install. No source, prompt, or frame is uploaded by the server itself.
2. **Evidence over assertion.** A render tool never claims success in prose. It
   returns the artifact path, a sampled frame (or frames), timing, and a structured
   verdict. "It rendered" is proven by the frame, not stated.
3. **Verification is first-class, not a nicety.** The render → inspect → correct loop
   is the product. Unresolved visual problems are carried forward as explicit
   **obligations** (see §7), the way REA carries *unknowns* forward.
4. **Explicit requirements.** Every capability states its prerequisites (Manim CE
   version, a working LaTeX for math, ffmpeg for stitching, a vision-capable model
   for verification) and degrades gracefully when one is missing.
5. **User control.** Chalkcast proposes; it does not surprise. Writing files,
   installing a missing TeX package, or opening a long render are surfaced, bounded,
   and (for anything slow or destructive) confirmable.
6. **Agent-first, CLI parity.** Everything an agent can do over MCP, a human can do
   from the `chalkcast` CLI, running the identical workflow.
7. **Untrusted content.** Rendered text, LaTeX logs, and Manim stderr originate from
   the scene and from external engines. They are returned as data, never followed as
   instructions, and are marked as such (`provenance.untrusted_content`).

---

## 2. Architecture

```
          ┌──────────────────────────────────────────────────────────┐
          │                        Agent (MCP)                        │
          │              or human via the chalkcast CLI               │
          └───────────────────────────┬──────────────────────────────┘
                                       │  same workflows, same validation
          ┌────────────────────────────▼─────────────────────────────┐
          │                     Chalkcast server                      │
          │                                                           │
          │  Layer A — Authoring        Pedagogical primitives:       │
          │                             scenes, steps, equations,     │
          │                             axes/plots, shapes, text,     │
          │                             transitions, camera, timing   │
          │                                     │                     │
          │  Layer B — Rendering        Compile the scene graph to    │
          │                             Manim code → video/frames     │
          │                                     │                     │
          │  Layer C — Verification     Sample frames, run visual      │
          │  (the differentiator)       checks, emit a verdict +      │
          │                             obligations → feed correction │
          └────────────────────────────┬─────────────────────────────┘
                                        │
          ┌────────────────────────────▼─────────────────────────────┐
          │                      Provider layer                       │
          │  Manim CE · LaTeX/TeX · ffmpeg · cairo · vision verifier  │
          └───────────────────────────────────────────────────────────┘
```

### 2.1 Why three layers (and why it wins)

The existing `manim-mcp` servers collapse everything into one tool: *"run this Manim
code, return the video."* The agent is left to write correct Manim blind, and Manim
is unforgiving — a LaTeX typo, an off-screen object, two labels overlapping, a
transition that fires before its target exists. The video "renders" but is wrong, and
the agent has no way to know.

Chalkcast splits the problem:

- **Layer A (Authoring)** lets the agent think in *teaching units* (an equation that
  morphs, an axis that a curve is drawn onto, a step that highlights a term) instead
  of raw `Mobject` boilerplate. This is a validated **scene graph**, not a string of
  code.
- **Layer B (Rendering)** compiles that graph to Manim and runs it, returning the
  artifact plus sampled frames — the *evidence*.
- **Layer C (Verification)** is the moat. It inspects the rendered frames against the
  declared intent and returns a structured verdict: LaTeX compiled? anything off the
  safe area? labels colliding? the highlighted term actually highlighted? Each failure
  becomes an **obligation** the agent (or an auto-fix pass) must discharge before the
  scene is considered done.

### 2.2 Schema profiles

Following REA, tools expose two input-schema profiles over the **same** tool set and
the same server-side validation:

- **`compact`** (default) — minimal fields, to keep agent context small.
- **`full`** — every option surfaced, for power users and the CLI.

### 2.3 Provider layer

| Provider | Role | Required for |
|---|---|---|
| **Manim CE** | render engine (scene graph → video/frames) | everything |
| **LaTeX / TeX** | typeset math (`MathTex`, `Tex`) | equations, labeled plots |
| **ffmpeg** | stitch scenes, transcode, GIF/thumbnail export | multi-scene videos, shareable exports |
| **cairo** | Manim's 2D backend | default rendering |
| **vision verifier** | a vision-capable model that scores a frame against intent | Layer C (can degrade to geometric-only checks) |

Each provider is probed at startup; `inspect_environment` (see §6) reports which are
present and what degrades without them.

---

## 3. Session & scene model

Like a REA target, a Chalkcast **project** is opened once and **persists across
turns**. The agent does not resend the whole scene graph on every call; it mutates a
server-held project and renders slices of it.

- **Project** — the top-level session: output resolution, quality tier, frame rate,
  background, theme (the "chalkboard" look is the default brand), and an ordered list
  of scenes. Opened with `open_project`, released with `close_project` (which can
  emit a snapshot for later resumption, mirroring REA's `close_binary` snapshot).
- **Scene** — one continuous animation (≈ one idea). Holds a **scene graph** of
  pedagogical objects and a **timeline** of steps.
- **Step** — a timed beat in the timeline (`write`, `transform`, `highlight`,
  `fade`, `camera_move`, `wait`). Steps are what Layer C verifies.
- **Obligation** — an unresolved visual problem attached to a scene (see §7).

A project persists until closed; renders, frames, verdicts, and obligations
accumulate against it so the agent can work incrementally.

---

## 4. The verification loop (the core workflow)

```
author/mutate scene ──► render_scene ──► inspect_render ──► verdict
        ▲                                                     │
        │                                   pass ─────────────┤──► done
        └──────────── discharge obligations ◄── fail ─────────┘
```

1. The agent builds or edits a scene (Layer A).
2. `render_scene` produces a video + sampled frames (Layer B) — **evidence**.
3. `inspect_render` runs verification lanes (Layer C) and returns a **verdict** plus
   any **obligations**.
4. If obligations remain, the agent fixes the graph and re-renders. `fix_scene` can
   attempt common fixes automatically (rescale to safe area, de-overlap labels,
   repair a LaTeX expression) and report what it changed.

This loop is what makes Chalkcast's output *correct on screen*, and it is the story
the README will lead with.

### 4.1 Verification lanes

Each lane is cheap-first, escalating to the vision model only when needed.

| Lane | Checks | Method |
|---|---|---|
| `compile` | LaTeX compiled; no Manim exception; non-empty frames | parse engine logs + frame histogram |
| `framing` | nothing crosses the safe area; scene fits | geometric bounds from the scene graph |
| `collision` | no two visible objects overlap beyond a threshold | bounding-box / mask overlap |
| `legibility` | min font size, min contrast against background | geometry + pixel sampling |
| `intent` | the frame shows what the step declared (e.g. the right term is highlighted, the curve is on the axes) | vision model scoring frame vs. step description |
| `timing` | no step fires before its target exists; total duration sane | timeline analysis |

---

## 5. Result contract

Every tool returns a common envelope, modeled on REA's evidence-bearing results.

```jsonc
{
  "ok": true,
  "result": { /* tool-specific payload */ },
  "evidence": {
    "artifacts": ["media/scene_0001.mp4"],      // produced files
    "frames": ["media/frames/0001_t0.png", "..."], // sampled frames (proof)
    "timing": { "render_ms": 8421, "frames": 3 }
  },
  "coverage": {
    "lanes_run": ["compile", "framing", "collision"],
    "lanes_skipped": ["intent"],                 // e.g. vision provider absent
    "scope": "scene:0001 steps 1-6"
  },
  "verdict": { "status": "needs_fix", "score": 0.72 },  // on verify tools
  "obligations": [ /* see §7 */ ],
  "limitations": [
    "intent lane skipped: no vision provider configured",
    "LaTeX warning: overfull hbox (non-fatal)"
  ],
  "provenance": { "untrusted_content": ["manim_stderr", "latex_log"] }
}
```

- **`evidence`** — never empty on a render; it is how success is *proven*.
- **`coverage`** — what was actually checked and what scope it covered, so the agent
  never mistakes "no finding" for "fully verified."
- **`limitations`** — honest statement of what could not be done this call.
- **`obligations`** — carried forward across turns until discharged.

---

## 6. Tool catalog

Tools are grouped by capability (not a flat list) and use `verb_object` names. Inputs
shown in the `compact` profile. ✎ = mutates the project, ⚙ = runs an engine.

### 6.1 Session & project
| Tool | Purpose |
|---|---|
| `open_project` | Start/resume a project session (resolution, quality, fps, theme). |
| `inspect_project` | Current project state: scenes, steps, pending obligations. |
| `snapshot_project` | Persist the project graph for later resumption. |
| `close_project` ✎ | Release the session, optionally emitting a snapshot. |

### 6.2 Authoring — pedagogical primitives (Layer A) ✎
| Tool | Purpose |
|---|---|
| `add_scene` | Append a scene to the project. |
| `add_equation` | Add a LaTeX equation (validated before it reaches a render). |
| `add_plot` | Add axes + one or more functions / data series. |
| `add_shape` | Primitive geometry (circle, polygon, arrow, brace…). |
| `add_text` | Titles, labels, callouts with theme-aware styling. |
| `add_step` | Append a timed beat to the scene timeline. |
| `transform_object` | Declare a morph from one object into another. |
| `highlight_term` | Mark a sub-expression / region to emphasize in a step. |
| `set_camera` | Pan / zoom / focus for a step. |
| `edit_object` | Modify or remove an existing object by id. |

### 6.3 Rendering (Layer B) ⚙
| Tool | Purpose |
|---|---|
| `render_scene` | Render one scene → video + sampled frames (evidence). |
| `render_frame` | Render a single still at time *t* (fast preview). |
| `render_project` | Render all scenes and stitch via ffmpeg. |
| `preview_timeline` | Low-res/draft render for quick iteration. |

### 6.4 Verification & coverage (Layer C) ⚙
| Tool | Purpose |
|---|---|
| `inspect_render` | Run verification lanes on the latest render → verdict + obligations. |
| `inspect_frame` | Verify a single frame against a step's declared intent. |
| `list_obligations` | Outstanding visual problems for a scene/project. |
| `fix_scene` ✎ | Attempt automatic fixes (safe-area rescale, de-overlap, LaTeX repair) and report the diff. |
| `evaluate_coverage` | What has/has not been verified, and residual risk. |

### 6.5 Export & publish ⚙
| Tool | Purpose |
|---|---|
| `export_video` | Final MP4/WebM at target quality. |
| `export_gif` | Shareable GIF (README / social). |
| `export_thumbnail` | Poster frame for the video. |
| `export_scene_code` | Emit the generated Manim code (for hand-editing / transparency). |

### 6.6 Discovery & inventory
| Tool | Purpose |
|---|---|
| `inspect_environment` | Which providers are present (Manim, LaTeX, ffmpeg, vision) and what degrades without each. |
| `list_primitives` | Catalog of available pedagogical primitives and theme presets. |
| `list_capabilities` | Generated catalog of tools + providers (REA-style reference). |

---

## 7. Obligations (carried-forward unknowns)

An **obligation** is a verification failure that is not yet resolved. It is the
Chalkcast analog of REA's *unknowns*: the agent must not call a scene finished while
obligations remain.

```jsonc
{
  "id": "obl-0003",
  "scene": "scene:0001",
  "step": 4,
  "lane": "collision",
  "severity": "error",            // error | warning | info
  "summary": "label 'x₀' overlaps the curve near t=2.0s",
  "evidence_frame": "media/frames/0001_t2.png",
  "suggested_fix": "shift label up 0.4u or reduce font to 28pt",
  "status": "open"                // open | discharged | accepted
}
```

- `error` obligations block `export_*`.
- `warning`/`info` can be explicitly **accepted** (user decides a trade-off is fine),
  which is recorded — mirroring REA's "separate what was observed from what remains."

---

## 8. Requirements

| Requirement | Needed for | Degradation if missing |
|---|---|---|
| Python 3.11+, Manim CE ≥ 0.18 | all rendering | server refuses to render; authoring still works |
| A LaTeX distribution (TeX Live / MiKTeX) | equations, math labels | math primitives disabled; text/plots still work |
| ffmpeg | multi-scene stitching, GIF/thumbnail | single-scene renders only |
| Vision-capable model (via the host agent or a configured endpoint) | `intent` lane | verification runs geometry-only lanes and says so in `coverage` |

Chalkcast installs nothing without consent and reports its environment via
`inspect_environment`.

---

## 9. Non-goals (scope discipline)

- **Not** a generic Manim code runner — code execution is an *export*, not the
  interface.
- **Not** a voiceover / TTS / avatar product (possible future provider, not core).
- **Not** a cloud render farm — local-first by design.
- **Not** a replacement for an educator's judgment; it drafts and verifies, the human
  directs.

---

## 10. Roadmap (post-baseline)

1. **v0.1 — Baseline:** project/scene model, Layer A primitives (equation, plot,
   text, shape, step), `render_scene`/`render_frame`, `inspect_render` with
   geometry lanes, `export_gif`. Ship the killer-demo GIF in the README.
   *Done.*
   - **v0.1.x — Measured verification (done):** a geometry probe builds the
     real Mobjects without rendering and measures bboxes, ink, drawn-vs-true
     fidelity, and strict LaTeX.
   - Timeline-aware collisions; `fix_scene` auto-repairs.
   - Editing tools (`update_object`, `remove_object`, `remove_step`).
   - Continuous parameters (`add_parameter` + `tween`, anchored shapes,
     `with_previous`).
   - Verdict benchmark plus a held-out set (`chalkcast bench [--holdout]`).
   - Numerically checked reference scenes (`tests/test_science.py`).
2. **v0.2 — Vision verification:** the `intent` lane (does the frame show what was
   meant?).
3. **v0.3 — Full project renders:** multi-scene stitching, transitions library,
   theme presets beyond the chalkboard look.
4. **v0.4 — CLI parity + generated catalog** (`chalkcast` CLI, `list_capabilities`).
5. **Later:** voiceover provider, 3D scenes (`ThreeDScene`), interactive slide export
   (Manim Slides), a community primitive/template gallery.

---

## 11. Naming & positioning

- **Name:** Chalkcast — the chalkboard of a great teacher (`chalk`) + broadcasting a
  lesson (`cast`). Distinct from the npm `chalk` package for discoverability.
- **One-liner:** *"Describe a concept. Get a verified animated explainer."*
- **Brand look:** a modern chalkboard theme as the default scene style — instantly
  recognizable in a shared GIF, which is the primary growth lever.
