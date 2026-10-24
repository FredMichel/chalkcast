# Contributing to Chalkcast

Thanks for your interest! Chalkcast is early (alpha) and contributions are welcome.

## Dev setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[render,dev]"
pytest -q
```

The authoring and verification layers run without a render engine, so most of the
test suite passes even without Manim/LaTeX/ffmpeg installed. The rendering tools
degrade gracefully and report missing providers via `inspect_environment`.

## Project shape

- `src/chalkcast/model.py` — the scene graph (Project / Scene / Step / Obj)
- `src/chalkcast/codegen.py` — scene graph → Manim code
- `src/chalkcast/verify.py` — geometric verification lanes
- `src/chalkcast/render.py` — Manim/ffmpeg rendering + frame sampling
- `src/chalkcast/server.py` — the MCP tools
- `docs/SPEC.md` — the design and conventions

## Guidelines

- Add a test for any new primitive or fix (see `tests/`).
- Keep tool results in the common evidence envelope (see `docs/SPEC.md` §5).
- New authoring primitives should also get a geometric check where it makes sense.

## Good first issues

- The `intent` (vision) verification lane (v0.2) — score a rendered frame against
  a step's declared intent.
- More primitives: number lines, vectors/matrices, 3D scenes, animated tables.
- A richer theme gallery beyond chalkboard / whiteboard / dark.
