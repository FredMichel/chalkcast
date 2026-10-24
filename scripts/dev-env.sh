#!/usr/bin/env bash
# Source this to get a working Chalkcast render environment on this machine
# (macOS 14 / Apple Silicon, where Manim's native deps were set up by hand).
#
#   source scripts/dev-env.sh
#
# On a standard OS (Linux, or macOS with a working Homebrew) you do NOT need this:
# `pip install "chalkcast[render]"` plus ffmpeg is enough.

# Resolve the repo root whether sourced from bash (BASH_SOURCE) or zsh ($0).
_src="${BASH_SOURCE[0]:-$0}"
_here="$(cd "$(dirname "$_src")/.." && pwd)"

# the locally-built libcairo (pycairo loads it at runtime)
export DYLD_FALLBACK_LIBRARY_PATH="$HOME/.local/chalkcast-deps/cairo/lib:/opt/homebrew/lib${DYLD_FALLBACK_LIBRARY_PATH:+:$DYLD_FALLBACK_LIBRARY_PATH}"

# ffmpeg (installed via the imageio-ffmpeg pip package, symlinked here)
export PATH="$HOME/.local/bin:$PATH"

# LaTeX (BasicTeX) for MathTex/equation rendering, if installed
[ -d /Library/TeX/texbin ] && export PATH="/Library/TeX/texbin:$PATH"

# activate the project venv if present
[ -f "$_here/.venv/bin/activate" ] && source "$_here/.venv/bin/activate"

echo "chalkcast dev env ready:"
echo "  python : $(command -v python)"
echo "  ffmpeg : $(command -v ffmpeg)"
python - <<'PY' 2>/dev/null || echo "  (manim import failed)"
import cairo, manim
print(f"  manim  : {manim.__version__} (pycairo {cairo.version})")
PY
