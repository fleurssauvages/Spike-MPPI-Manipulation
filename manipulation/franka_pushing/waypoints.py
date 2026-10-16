"""Arrival-triggered two-point, square, and four-point pingpong routes."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_START_XY = (0.40, 0.0)
DEFAULT_GOAL_XY = (0.40, 0.18)
DEFAULT_DEPTH_X = 0.18


def square_corners(start_xy, second_xy) -> np.ndarray:
    """Generate A -> B -> C -> D, rotating segment AB clockwise by 90 deg.

    The existing start and goal arguments give the first two corners of the
    square, so the previous point-pushing defaults remain meaningful.
    """
    a = np.asarray(start_xy, dtype=np.float64).reshape(2)
    b = np.asarray(second_xy, dtype=np.float64).reshape(2)
    delta = b - a
    if not np.all(np.isfinite(np.stack((a, b)))) or np.linalg.norm(delta) < 1e-6:
        raise ValueError("square's first two corners must be distinct, finite XY points")
    right = np.array([delta[1], -delta[0]])
    return np.stack((a, b, b + right, a + right))


def resolve_waypoints(*, start_xy, goal_xy, route: str, waypoints=None, depth: float = DEFAULT_DEPTH_X) -> np.ndarray:
    """Resolve the CLI geometry and validate adjacent goal separation."""
    if route not in ("triangle", "backforth", "square", "pingpong", "point"):
        raise ValueError("route must be triangle, backforth, square, pingpong, or point")
    if waypoints is not None:
        if route == "point":
            raise ValueError("--waypoints is not valid for the one-way point route")
        xy = np.asarray(waypoints, dtype=np.float64)
        expected = 6 if route == "triangle" else 8
        if xy.size != expected:
            raise ValueError(f"--waypoints requires {expected // 2} XY pairs ({expected} numbers)")
        xy = xy.reshape(-1, 2)
        if route == "backforth":
            xy = xy[:2]
    elif route in ("point", "backforth"):
        xy = np.asarray([start_xy, goal_xy], dtype=np.float64).reshape(2, 2)
    elif route == "triangle":
        a = np.asarray(start_xy, dtype=np.float64).reshape(2)
        b = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        if not np.isfinite(depth) or depth <= 0:
            raise ValueError("triangle depth must be finite and positive")
        normal = np.array([b[1] - a[1], -(b[0] - a[0])], dtype=np.float64)
        span = np.linalg.norm(normal)
        if span < 1e-6:
            raise ValueError("triangle's first two points must differ")
        c = 0.5 * (a + b) + depth * normal / span
        xy = np.stack((a, b, c))
    else:
        xy = square_corners(start_xy, goal_xy)
    if not np.all(np.isfinite(xy)):
        raise ValueError("all keypoints must be finite")
    if np.any(np.linalg.norm(np.diff(xy, axis=0), axis=1) < 1e-6):
        raise ValueError("consecutive waypoints must be different")
    if route != "point" and np.linalg.norm(xy[-1] - xy[0]) < 1e-6:
        raise ValueError("the last and first waypoints must be different")
    return xy.copy()


@dataclass
class WaypointRoute:
    """Advance the active MPPI goal only after a box-position dwell at it.

    Routes start with the physical box at waypoint A (index 0) and target B
    (index 1). Backforth is A->B->A; a full cycle visits B and returns to A.
    A completed square is A->B->C->D->A. Four-point pingpong is
    A->B->C->D->C->B->A. No target changes happen within an MPPI rollout.
    """

    mode: str = "triangle"
    cycles: int = 2
    active_segment: int = 0
    segments_completed: int = 0
    cycles_completed: int = 0
    dwell_seconds: float = 0.0
    done: bool = False

    def __post_init__(self):
        if self.mode == "triangle":
            self._sequence = (1, 2, 0)
        elif self.mode == "backforth":
            self._sequence = (1, 0)
        elif self.mode == "square":
            self._sequence = (1, 2, 3, 0)
        elif self.mode == "pingpong":
            self._sequence = (1, 2, 3, 2, 1, 0)
        elif self.mode == "point":
            self._sequence = (1,)
        else:
            raise ValueError(f"invalid route mode {self.mode!r}")
        if self.cycles < 0:
            raise ValueError("cycles must be nonnegative (0 means repeat until --steps)")

    @property
    def goal_index(self) -> int:
        # After finite-cycle completion, the displayed goal stays at the last
        # reached waypoint rather than jumping to the next cycle's first one.
        return self._sequence[-1] if self.done else self._sequence[self.active_segment]

    def observe(self, *, distance: float, dt: float, tolerance: float, hold_seconds: float) -> int | None:
        """Return an arrived-at waypoint index, or None if not yet validated.

        When the last segment of a cycle is reached, increment cycles_completed.
        For finite cycles, stop after the requested number of complete routes.
        Point mode always stops after its first reached goal.
        """
        if self.done:
            return None
        if not (np.isfinite(distance) and np.isfinite(dt)) or dt < 0:
            raise ValueError("distance and timestep must be finite, with dt >= 0")
        if tolerance <= 0 or hold_seconds <= 0:
            raise ValueError("goal tolerance and hold duration must be positive")
        self.dwell_seconds = self.dwell_seconds + dt if distance <= tolerance else 0.0
        if self.dwell_seconds + 1e-12 < hold_seconds:
            return None
        arrived = self.goal_index
        self.dwell_seconds = 0.0
        self.segments_completed += 1
        self.active_segment += 1
        if self.active_segment == len(self._sequence):
            self.active_segment = 0
            self.cycles_completed += 1
            self.done = self.mode == "point" or (
                self.cycles > 0 and self.cycles_completed >= self.cycles
            )
        return arrived


__all__ = ["DEFAULT_START_XY", "DEFAULT_GOAL_XY", "DEFAULT_DEPTH_X", "square_corners", "resolve_waypoints", "WaypointRoute"]
