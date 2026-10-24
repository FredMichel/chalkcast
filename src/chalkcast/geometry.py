"""Manim coordinate frame constants and bounding-box helpers.

Manim Community's default camera frame is 8 units tall; width follows the 16:9
aspect ratio. The origin is the center of the screen. The geometric verification
lanes (framing, collision) reason in these units straight from the scene graph,
without touching rendered pixels -- this is intentionally approximate and is
reported as such in each result's ``limitations``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Default Manim CE frame (config.frame_height / frame_width).
FRAME_HEIGHT: float = 8.0
FRAME_WIDTH: float = 8.0 * 16.0 / 9.0  # ~14.222

# Fraction of the frame kept clear of content on each edge. Content crossing the
# safe area is flagged by the ``framing`` lane.
SAFE_MARGIN: float = 0.5  # units


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box centered at (cx, cy)."""

    cx: float
    cy: float
    width: float
    height: float

    @property
    def left(self) -> float:
        return self.cx - self.width / 2

    @property
    def right(self) -> float:
        return self.cx + self.width / 2

    @property
    def bottom(self) -> float:
        return self.cy - self.height / 2

    @property
    def top(self) -> float:
        return self.cy + self.height / 2

    def overlap_area(self, other: "BBox") -> float:
        dx = min(self.right, other.right) - max(self.left, other.left)
        dy = min(self.top, other.top) - max(self.bottom, other.bottom)
        if dx <= 0 or dy <= 0:
            return 0.0
        return dx * dy

    @property
    def area(self) -> float:
        return max(self.width, 1e-9) * max(self.height, 1e-9)


def safe_frame() -> BBox:
    """The region content should stay within."""
    return BBox(
        cx=0.0,
        cy=0.0,
        width=FRAME_WIDTH - 2 * SAFE_MARGIN,
        height=FRAME_HEIGHT - 2 * SAFE_MARGIN,
    )


def within_safe_area(box: BBox) -> bool:
    safe = safe_frame()
    return (
        box.left >= safe.left - 1e-6
        and box.right <= safe.right + 1e-6
        and box.bottom >= safe.bottom - 1e-6
        and box.top <= safe.top + 1e-6
    )
