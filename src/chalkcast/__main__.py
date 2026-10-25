"""Chalkcast CLI — the same workflows the agent drives over MCP.

    chalkcast serve     # run the MCP server (stdio)
    chalkcast doctor    # report which render providers are present
    chalkcast demo      # build + verify (+ render if Manim is installed) the demo scene
    chalkcast bench     # verdict benchmark: detection / false-positive rates
"""

from __future__ import annotations

import argparse
import json
import sys


_FIXES = {
    "darwin": {
        "manim": "pip install 'chalkcast[render]'   (if pycairo fails: brew install cairo pkg-config)",
        "latex": "brew install --cask basictex   (then open a new terminal)",
        "dvisvgm": "sudo tlmgr install dvisvgm",
        "physics": "sudo tlmgr install physics   (optional: fallback macros are built in)",
        "ffmpeg": "pip install imageio-ffmpeg",
    },
    "linux": {
        "manim": "sudo apt install libcairo2-dev libpango1.0-dev pkg-config && pip install 'chalkcast[render]'",
        "latex": "sudo apt install texlive texlive-latex-extra",
        "dvisvgm": "sudo apt install dvisvgm",
        "physics": "sudo apt install texlive-science   (optional: fallback macros are built in)",
        "ffmpeg": "pip install imageio-ffmpeg",
    },
    "win32": {
        "manim": "pip install \"chalkcast[render]\"",
        "latex": "install MiKTeX: https://miktex.org/download",
        "dvisvgm": "included with MiKTeX",
        "physics": "MiKTeX installs packages on first use",
        "ffmpeg": "pip install imageio-ffmpeg",
    },
}


def _doctor() -> int:
    """Check every render prerequisite and print the exact command to fix each gap."""
    import importlib.util
    import shutil
    import subprocess

    from .render import environment, ffmpeg_exe

    env = environment()
    plat = "linux" if sys.platform.startswith("linux") else sys.platform
    fixes = _FIXES.get(plat, _FIXES["linux"])
    manim_ok = False
    if importlib.util.find_spec("manim") is not None:
        # a real import: catches broken native libraries (cairo/pango), not just presence
        probe = subprocess.run([sys.executable, "-c", "import manim"], capture_output=True, text=True)
        manim_ok = probe.returncode == 0
        if not manim_ok:
            print("manim is installed but fails to import:\n  "
                  + (probe.stderr.strip().splitlines() or ["?"])[-1] + "\n")
    physics = False
    if shutil.which("kpsewhich"):
        physics = bool(subprocess.run(["kpsewhich", "physics.sty"], capture_output=True,
                                      text=True).stdout.strip())
    checks = [
        ("manim", manim_ok, "rendering video and measured verification", True),
        ("latex", env["latex"], "equations (add_equation, axis labels)", False),
        ("dvisvgm", shutil.which("dvisvgm") is not None, "equations (LaTeX -> SVG)", False),
        ("physics", physics, "\\ket / \\dv etc. from the physics package", False),
        ("ffmpeg", ffmpeg_exe() is not None, "GIF export and evidence frames", False),
    ]
    print(f"chalkcast doctor  (python {sys.version.split()[0]}, {sys.platform})\n")
    for name, present, needed_for, _required in checks:
        mark = "ok " if present else "-- "
        print(f"  {mark}{name:<8} {needed_for}")
        if not present:
            print(f"           fix: {fixes[name]}")
    ready = manim_ok
    print("\n" + ("Ready: renders + measured verification available."
                  if ready else "Not ready to render; authoring + estimated verification still work."))
    if "--json" in sys.argv:
        print(json.dumps(env, indent=2))
    return 0 if ready else 1


def _demo() -> int:
    from pathlib import Path

    from .demo import build_demo_project
    from .render import have_manim, render_scene
    from .verify import inspect_scene

    project, scene = build_demo_project()
    print(f"Built {scene.id!r}: {len(scene.objects)} objects, {len(scene.steps)} steps")

    insp = inspect_scene(scene)
    print("Verification verdict:", json.dumps(insp["verdict"]))
    for o in insp["obligations"]:
        print(f"  [{o['severity']}] {o['lane']}: {o['summary']}")

    if not have_manim():
        print("\nManim not installed — skipping render. "
              "Install with: pip install 'chalkcast[render]'")
        return 0

    wd = Path.cwd() / ".chalkcast" / "demo"
    report = render_scene(project, scene, wd)
    print("\nRender:", json.dumps({k: report.get(k) for k in ("ok", "artifact", "timing")}, indent=2))
    return 0 if report.get("ok") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="chalkcast")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("serve", help="run the MCP server over stdio")
    sub.add_parser("doctor", help="report render provider availability")
    sub.add_parser("demo", help="build + verify (+ render) the demo scene")
    sub.add_parser("bench", help="verdict benchmark (detection / false-positive rates)",
                   add_help=False)

    args, rest = parser.parse_known_args(argv)
    if args.cmd == "bench":
        from .bench import main as bench_main
        return bench_main(rest)
    if args.cmd == "serve" or args.cmd is None:
        from .server import run
        run()
        return 0
    if args.cmd == "doctor":
        return _doctor()
    if args.cmd == "demo":
        return _demo()
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
