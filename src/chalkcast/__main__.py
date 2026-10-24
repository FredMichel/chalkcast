"""Chalkcast CLI — the same workflows the agent drives over MCP.

    chalkcast serve     # run the MCP server (stdio)
    chalkcast doctor    # report which render providers are present
    chalkcast demo      # build + verify (+ render if Manim is installed) the demo scene
"""

from __future__ import annotations

import argparse
import json
import sys


def _doctor() -> int:
    from .render import environment
    env = environment()
    print(json.dumps(env, indent=2))
    return 0 if env["manim"] else 1


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

    args = parser.parse_args(argv)
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
