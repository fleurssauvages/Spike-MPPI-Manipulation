from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class StraightTrack:
    """Cartesian line/corridor adapter for manipulation progress.

    Progress is non-periodic. ``set_line`` can retarget the same object
    in-place; ``fused_cache_token`` changes whenever
    that happens so the native evaluator refreshes its cached geometry.
    """

    origin_xy: tuple[float, float] = (0.0, 0.0)
    direction: tuple[float, float] = (1.0, 0.0)
    road_width: float = 0.6
    length: float = 10.0
    goal_s: float = 0.0
    _version: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if float(self.road_width) <= 0.0:
            raise ValueError("StraightTrack road_width must be positive")
        self._set_geometry(self.origin_xy, self.direction, bump=False)

    def _set_geometry(self, origin_xy, direction, *, bump: bool) -> None:
        origin = np.asarray(origin_xy, dtype=np.float64)
        d = np.asarray(direction, dtype=np.float64)
        if origin.shape != (2,) or d.shape != (2,):
            raise ValueError("StraightTrack origin and direction must be 2D")
        norm = float(np.linalg.norm(d))
        if norm <= 1e-12:
            raise ValueError("StraightTrack direction must be nonzero")
        self._origin = origin.copy()
        self._tangent = d / norm
        self._normal = np.asarray([-self._tangent[1], self._tangent[0]], dtype=np.float64)
        self.origin_xy = (float(self._origin[0]), float(self._origin[1]))
        self.direction = (float(self._tangent[0]), float(self._tangent[1]))
        if bump:
            self._version += 1

    def set_line(
        self, origin_xy, direction, *, length: float | None = None, goal_s: float | None = None
    ) -> None:
        self._set_geometry(origin_xy, direction, bump=True)
        if length is not None:
            self.length = float(length)
        if goal_s is not None:
            value = float(goal_s)
            if value <= 0.0:
                raise ValueError("StraightTrack goal_s must be positive")
            self.goal_s = value

    @property
    def fused_cache_token(self) -> int:
        return int(self._version)

    @property
    def wrap_progress(self) -> bool:
        return False

    @property
    def start_s(self) -> float:
        return 0.0

    def project(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(xy, dtype=np.float64)
        delta = p - self._origin
        s = np.sum(delta * self._tangent, axis=-1)
        lateral = np.sum(delta * self._normal, axis=-1)
        return np.asarray(s), np.asarray(lateral * lateral)

    def sample(self, s) -> np.ndarray:
        value = np.asarray(s, dtype=np.float64)
        return self._origin + value[..., None] * self._tangent

    def tangent(self, s) -> np.ndarray:
        value = np.asarray(s)
        if value.ndim == 0:
            return self._tangent.copy()
        return np.broadcast_to(self._tangent, value.shape + (2,)).copy()

    def normal(self, s) -> np.ndarray:
        value = np.asarray(s)
        if value.ndim == 0:
            return self._normal.copy()
        return np.broadcast_to(self._normal, value.shape + (2,)).copy()

    def curvature(self, s) -> np.ndarray | float:
        value = np.asarray(s)
        if value.ndim == 0:
            return 0.0
        return np.zeros_like(value, dtype=np.float64)

    def signed_progress_delta(self, new_s: float, old_s: float) -> float:
        return float(new_s) - float(old_s)

    def polyline(self, count: int = 100) -> np.ndarray:
        return self.sample(np.linspace(0.0, float(self.length), max(2, int(count))))


__all__ = ["StraightTrack"]
