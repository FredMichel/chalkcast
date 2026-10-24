#!/usr/bin/env bash
# MCP entry point with the render environment baked in. Register this with your
# MCP client so Manim can find libcairo + ffmpeg at runtime:
#
#   claude mcp add chalkcast -- /Users/frederic/projects/chalkcast/scripts/serve.sh
set -e
_here="$(cd "$(dirname "$0")/.." && pwd)"
source "$_here/scripts/dev-env.sh" >/dev/null 2>&1
exec chalkcast serve
