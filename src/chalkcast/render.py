"""Rendering (Layer B): scene graph -> Manim code -> video + sampled frames.

Local-first and engine-aware. If Manim is not installed the render tools fail
gracefully with a clear ``limitations`` entry instead of raising, and the
authoring/verification layers keep working.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .codegen import generate_module
from .model import Project, Scene

QUALITY_FLAGS = {
    "draft": "-ql",   # 480p15
    "low": "-ql",
    "medium": "-qm",  # 720p30
    "high": "-qh",    # 1080p60
}


def have_manim() -> bool:
    return shutil.which("manim") is not None


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


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
    cmd = ["manim", "render", flag, "--media_dir", str(media_dir)]
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
    frames = _sample_frames(artifact, workdir) if (artifact and not single_frame) else (
        [str(artifact)] if artifact else []
    )
    return {
        "ok": True,
        "artifact": str(artifact) if artifact else None,
        "frames": frames,
        "timing": {"render_ms": elapsed, "frames": len(frames)},
        "stderr_tail": proc.stderr[-500:],
    }


def _sample_frames(video: Path, workdir: Path, n: int = 3) -> list[str]:
    """Extract evidence frames from the rendered video (needs ffmpeg)."""
    if not have_ffmpeg() or video is None:
        return []
    out_dir = workdir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / f"{video.stem}_%03d.png")
    # Grab `n` evenly spaced frames via the thumbnail/fps filter.
    cmd = [
        "ffmpeg", "-y", "-i", str(video),
        "-vf", f"select='not(mod(n\\,30))',scale=640:-1",
        "-vsync", "vfr", "-frames:v", str(n), pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return []
    return [str(p) for p in sorted(out_dir.glob(f"{video.stem}_*.png"))][:n]


def export_gif(video: Path, workdir: Path) -> dict[str, object]:
    """Convert a rendered MP4 to a shareable GIF (needs ffmpeg)."""
    if not have_ffmpeg():
        return {"ok": False, "error": "ffmpeg is not installed"}
    if not video or not Path(video).exists():
        return {"ok": False, "error": "no rendered video to export; render the scene first"}
    out = workdir / (Path(video).stem + ".gif")
    palette = workdir / "_palette.png"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-vf", "fps=15,scale=640:-1:flags=lanczos,palettegen", str(palette)],
        capture_output=True, text=True,
    )
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-i", str(palette),
         "-lavfi", "fps=15,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse", str(out)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": "gif export failed", "stderr_tail": proc.stderr[-800:]}
    return {"ok": True, "artifact": str(out)}
