"""Rendering (Layer B): scene graph -> Manim code -> video + sampled frames.

Local-first and engine-aware. If Manim is not installed the render tools fail
gracefully with a clear ``limitations`` entry instead of raising, and the
authoring/verification layers keep working.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .codegen import generate_module, plot_is_dynamic, uses_params
from .model import Project, Scene

TWEEN_SAMPLES = 8

QUALITY_FLAGS = {
    "low": "-ql",     # 480p
    "medium": "-qm",  # 720p
    "high": "-qh",    # 1080p (the frame rate is the project's fps)
}


def have_manim() -> bool:
    """Manim importable by this interpreter (no PATH activation needed, e.g. under uvx)."""
    import importlib.util
    return importlib.util.find_spec("manim") is not None


def ffmpeg_exe() -> str | None:
    """ffmpeg on PATH, else the binary bundled by the imageio-ffmpeg wheel (pip-only install)."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - not installed / no binary for this platform
        return None


def have_ffmpeg() -> bool:
    return ffmpeg_exe() is not None


def have_latex() -> bool:
    return shutil.which("latex") is not None or shutil.which("pdflatex") is not None


def environment() -> dict[str, object]:
    return {
        "manim": have_manim(),
        "ffmpeg": have_ffmpeg(),
        "latex": have_latex(),
        "vision": False,  # v0.2
        "degrades": {
            "no_manim": "authoring + geometric verification still work; no video",
            "no_latex": "equations/math labels unavailable; text/plots still render",
            "no_ffmpeg": "single-scene renders only; GIF/thumbnail export disabled",
            "no_vision": "intent lane skipped; geometric lanes still run",
        },
    }


def probe_spec(scene: Scene) -> dict[str, Any]:
    """What the probe should measure: objects, and the parameter states visited."""
    objs = []
    for o in scene.objects:
        dyn = False
        if o.kind == "plot":
            dyn = plot_is_dynamic(o, scene.params)
        elif o.kind in ("area", "riemann"):
            parent = scene.get_obj(o.payload.get("plot", ""))
            dyn = parent is not None and plot_is_dynamic(parent, scene.params)
        elif o.kind == "shape" and o.payload.get("anchor"):
            a = o.payload["anchor"]
            dyn = uses_params(str(a.get("x", "")), scene.params) or uses_params(str(a.get("y", "")), scene.params)
        objs.append({"id": o.id, "kind": o.kind, "dynamic": dyn,
                     "n_functions": len(o.payload.get("functions", [])) if o.kind == "plot" else 0,
                     "tex": o.payload.get("tex") if o.kind == "equation" else None})
    # Parameter values the timeline passes through: each tween beat (tweens joined by
    # with_previous move together) sampled at TWEEN_SAMPLES points, so something
    # that only collides or leaves the frame mid-tween is still measured.
    states: list[dict[str, Any]] = []
    cur = dict(scene.params)
    beat: dict[str, float] = {}
    beat_step = 0

    def flush() -> None:
        start = dict(cur)
        for k in range(1, TWEEN_SAMPLES + 1):
            a = k / TWEEN_SAMPLES
            vals = {**start, **{p: start[p] + a * (v - start[p]) for p, v in beat.items()}}
            states.append({"step": beat_step, "params": vals, "beat_end": k == TWEEN_SAMPLES})
        cur.update(beat)
        beat.clear()

    for st in scene.steps:
        if beat and not st.with_previous:
            flush()
        if st.kind == "tween" and st.param in cur and st.value is not None:
            if not beat:
                beat_step = st.index
            beat[st.param] = float(st.value)
    if beat:
        flush()
    return {"objects": objs, "param_states": states}


def probe_scene(project: Project, scene: Scene, workdir: Path, timeout: float = 120) -> dict[str, Any]:
    """Build the scene's Mobjects in a subprocess and measure them (no video).

    Returns the probe's JSON report, or ``{"ok": None, ...}`` when the probe could
    not run at all (no Manim) -- callers then fall back to estimated geometry.
    """
    if not have_manim():
        return {"ok": None, "error": "manim is not importable; geometry is estimated"}
    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    module_path = workdir / f"{scene.class_name()}.py"
    module_path.write_text(generate_module(scene, project.theme), encoding="utf-8")
    spec_path = workdir / "probe_spec.json"
    spec_path.write_text(json.dumps(probe_spec(scene)), encoding="utf-8")
    t0 = time.time()
    try:
        proc = subprocess.run([sys.executable, "-m", "chalkcast.probe", str(module_path), str(spec_path)],
                              capture_output=True, text=True, cwd=str(workdir), timeout=timeout)
    except subprocess.TimeoutExpired:
        # slow to measure is not broken: fall back to estimated geometry, say why
        return {"ok": None, "timeout": True,
                "error": f"geometry probe timed out after {timeout:.0f}s"}
    marker = "@@CHALKCAST_PROBE@@"
    if marker not in proc.stdout:
        return {"ok": False, "error": "probe crashed", "traceback": proc.stderr[-2500:]}
    report = json.loads(proc.stdout.rsplit(marker, 1)[1])
    report["timing"] = {"probe_ms": int((time.time() - t0) * 1000)}
    return report


def _find_output(media_dir: Path, ext: str) -> Path | None:
    candidates = sorted(media_dir.rglob(f"*.{ext}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def render_scene(project: Project, scene: Scene, workdir: Path, *, single_frame: bool = False) -> dict[str, object]:
    """Render one scene. Returns a report dict (ok / artifact / frames / timing / error)."""
    if not have_manim():
        return {
            "ok": False,
            "error": "manim is not installed",
            "hint": "pip install 'chalkcast[render]' and a LaTeX distribution for math",
        }

    workdir = workdir.resolve()  # absolute, so cwd=workdir never double-counts
    workdir.mkdir(parents=True, exist_ok=True)
    module_path = workdir / f"{scene.class_name()}.py"
    module_path.write_text(generate_module(scene, project.theme), encoding="utf-8")

    media_dir = workdir / "media"
    flag = QUALITY_FLAGS.get(project.quality, "-ql")
    cmd = [sys.executable, "-m", "manim", "render", flag, "--fps", str(project.fps),
           "--media_dir", str(media_dir)]
    if single_frame:
        cmd.append("-s")  # last-frame still (fast preview)
    # module arg is the bare filename because we run with cwd=workdir
    cmd += [module_path.name, scene.class_name()]

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(workdir))
    elapsed = int((time.time() - t0) * 1000)

    if proc.returncode != 0:
        return {
            "ok": False,
            "error": "manim render failed",
            "stderr_tail": proc.stderr[-1500:],
            "timing": {"render_ms": elapsed},
        }

    ext = "png" if single_frame else "mp4"
    artifact = _find_output(media_dir, ext)
    frames = _sample_frames(artifact, workdir, scene.duration()) if (artifact and not single_frame) else (
        [str(artifact)] if artifact else []
    )
    return {
        "ok": True,
        "artifact": str(artifact) if artifact else None,
        "frames": frames,
        "timing": {"render_ms": elapsed, "frames": len(frames)},
        "stderr_tail": proc.stderr[-500:],
    }


def _sample_frames(video: Path, workdir: Path, duration: float, n: int = 4) -> list[str]:
    """Evidence frames spread over the whole video, always including the last one."""
    if not have_ffmpeg() or video is None:
        return []
    out_dir = workdir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob(f"{video.stem}_*.png"):
        old.unlink()
    frames = []
    for i in range(n):
        # the last one just before the end (a seek past the final frame yields nothing)
        t = max(0.0, min(duration * (i + 1) / n, duration - 0.2))
        out = out_dir / f"{video.stem}_{i + 1:03d}.png"
        proc = subprocess.run([ffmpeg_exe(), "-y", "-ss", f"{t:.3f}", "-i", str(video),
                               "-frames:v", "1", "-vf", "scale=640:-1", str(out)],
                              capture_output=True, text=True)
        if proc.returncode == 0 and out.exists():
            frames.append(str(out))
    return frames


def export_gif(video: Path, workdir: Path, width: int = 640, fps: int = 15) -> dict[str, object]:
    """Convert a rendered MP4 to a shareable GIF (needs ffmpeg)."""
    if not have_ffmpeg():
        return {"ok": False, "error": "ffmpeg is not installed"}
    if not video or not Path(video).exists():
        return {"ok": False, "error": "no rendered video to export; render the scene first"}
    out = workdir / (Path(video).stem + ".gif")
    palette = workdir / "_palette.png"
    subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(video), "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,palettegen", str(palette)],
        capture_output=True, text=True,
    )
    proc = subprocess.run(
        [ffmpeg_exe(), "-y", "-i", str(video), "-i", str(palette),
         "-lavfi", f"fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse", str(out)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": "gif export failed", "stderr_tail": proc.stderr[-800:]}
    return {"ok": True, "artifact": str(out)}
