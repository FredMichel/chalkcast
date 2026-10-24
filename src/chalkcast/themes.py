"""Scene themes. The ``chalkboard`` look is the Chalkcast brand default -- it is
what makes a shared GIF instantly recognizable, which is the primary growth lever.
"""

from __future__ import annotations

THEMES: dict[str, dict[str, str]] = {
    "chalkboard": {
        "background_color": "#17241d",  # dark slate-green board
        "ink": "#f4f1e8",              # chalk white
        "accent": "#f6c76b",           # chalk yellow
    },
    "whiteboard": {
        "background_color": "#f7f7f4",
        "ink": "#1b1b1b",
        "accent": "#2a6df5",
    },
    "dark": {
        "background_color": "#0d0d10",
        "ink": "#e8e8ea",
        "accent": "#7c5cff",
    },
}

DEFAULT_THEME = "chalkboard"


def theme(name: str) -> dict[str, str]:
    return THEMES.get(name, THEMES[DEFAULT_THEME])
