"""Franka single-finger, free box and retargetable tabletop waypoints."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .fetch_franka_menagerie import DEFAULT_DIR, fetch_franka_menagerie
from .franka import FrankaReachRobot


class FrankaPushRobot(FrankaReachRobot):
    """Panda arm from reaching, now controlling a *physical* tabletop box.

    Start and goal are XY keypoints on the table. Their Z coordinate is the
    resting center of the box. Neither box nor tool orientation is controlled.
    """

    name = "franka-push"
    fused_task_mode = 4

    def __init__(
        self,
        *,
        start_xy=(0.55, 0.0),
        goal_xy=(0.55, 0.18),
        auto_fetch: bool = True,
        motion_limit_scale: float = 1.0,
        acceleration_limit_scale: float = 1.0,
        xml_path: str | Path | None = None,
    ) -> None:
        start = np.asarray(start_xy, dtype=np.float64).reshape(2)
        goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(start)) or not np.all(np.isfinite(goal)):
            raise ValueError("start and goal XY must be finite")
        self._start_xy = start.copy()
        self._goal_xy = goal.copy()
        if xml_path is None:
            if auto_fetch:
                fetch_franka_menagerie(DEFAULT_DIR)
            xml_path = DEFAULT_DIR / "franka_pushing_scene.xml"
        # The pushing scene uses the existing reach_target free joint, now a
        # massive colliding box, so the rest of the Panda adapter is shared.
        super().__init__(
            xml_path=xml_path,
            auto_fetch=False,
            motion_limit_scale=motion_limit_scale,
            acceleration_limit_scale=acceleration_limit_scale,
            target_xyz=(float(start[0]), float(start[1]), 0.370),
        )
        self._table_geom_id = self._geom_id("table")
        self._table_body_id = int(self.model.geom_bodyid[self._table_geom_id])
        self._goal_body_id = self._body_id("push_goal")
        self._goal_site_id = int(self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SITE, "push_goal_site"
        ))
        self._start_site_id = int(self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SITE, "push_start_keypoint"
        ))
        self._goal_mocap_id = int(self.model.body_mocapid[self._goal_body_id])
        self._waypoint_site_ids = tuple(
            int(self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_SITE, f"push_waypoint_{i}"
            )) for i in range(1, 4)
        )
        if self._goal_site_id < 0 or self._goal_mocap_id < 0:
            raise ValueError("pushing scene requires a mocap push_goal with push_goal_site")
        self._box_rest_z = float(
            self.model.geom_pos[self._table_geom_id, 2]
            + self.model.geom_size[self._table_geom_id, 2]
            + self.model.geom_size[self._task_geom_id, 2]
        )
        table_center = np.asarray(self.model.geom_pos[self._table_geom_id, :2], dtype=float)
        table_half = np.asarray(self.model.geom_size[self._table_geom_id, :2], dtype=float)
        box_half = np.asarray(self.model.geom_size[self._task_geom_id, :2], dtype=float)
        self._min_xy = table_center - table_half + box_half
        self._max_xy = table_center + table_half - box_half
        self._validate_xy(self._start_xy, "start")
        self._validate_xy(self._goal_xy, "goal")
        self.reset()
        self.set_goal_xy(self._goal_xy)

    def _validate_xy(self, point: np.ndarray, label: str) -> None:
        if np.any(point < self._min_xy) or np.any(point > self._max_xy):
            raise ValueError(
                f"{label} point {point.tolist()} does not fit fully on the table; "
                f"valid XY center range: {self._min_xy.tolist()} to {self._max_xy.tolist()}"
            )

    def reset(self) -> None:
        super().reset()
        if hasattr(self, "_box_rest_z"):
            self.set_start_xy(self._start_xy)
            if hasattr(self, "_goal_site_id"):
                self.set_goal_xy(self._goal_xy)

    def set_start_xy(self, point) -> None:
        xy = np.asarray(point, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(xy)):
            raise ValueError("start point must be finite")
        if hasattr(self, "_min_xy"):
            self._validate_xy(xy, "start")
        self._start_xy = xy.copy()
        z = self._box_rest_z
        super().set_target_xyz([xy[0], xy[1], z])
        if getattr(self, "_start_site_id", -1) >= 0:
            self.model.site_pos[self._start_site_id] = np.array([xy[0], xy[1], z])
            self.mujoco.mj_forward(self.model, self.data)

    def set_waypoints_xy(self, points) -> None:
        """Display and validate three or four tabletop waypoints."""
        xy = np.asarray(points, dtype=np.float64)
        if xy.shape not in ((3, 2), (4, 2)) or not np.all(np.isfinite(xy)):
            raise ValueError("waypoints must contain three or four finite XY points")
        if not np.allclose(xy[0], self._start_xy, atol=1e-9, rtol=0.0):
            raise ValueError("first waypoint must equal the physical box's start XY")
        for i, point in enumerate(xy):
            self._validate_xy(point, f"waypoint {i}")
        if any(sid < 0 for sid in self._waypoint_site_ids):
            raise ValueError("pushing scene requires push_waypoint_1/2/3 sites")
        for i, sid in enumerate(self._waypoint_site_ids, start=1):
            if i < len(xy):
                self.model.site_pos[sid] = [xy[i, 0], xy[i, 1], self._box_rest_z]
                self.model.site_rgba[sid, 3] = 0.45
            else:
                self.model.site_rgba[sid, 3] = 0.0
        self._waypoints_xy = xy.copy()
        self.mujoco.mj_forward(self.model, self.data)

    def hide_waypoints(self) -> None:
        """Hide unused square markers for the original two-keypoint mode."""
        for sid in self._waypoint_site_ids:
            if sid >= 0:
                self.model.site_rgba[sid, 3] = 0.0

    def sync_goal_mocap(self, data) -> None:
        """Refresh an existing rollout MjData's mocap goal after retargeting.

        MuJoCo's mjSTATE_FULLPHYSICS omits mocap poses, so changing the plant's
        mocap pose alone is not enough for the long-lived mj_rollout buffers.
        """
        data.mocap_pos[self._goal_mocap_id] = (
            self._goal_xy[0], self._goal_xy[1], self._box_rest_z
        )
        self.mujoco.mj_forward(self.model, data)

    def set_goal_xy(self, point) -> None:
        xy = np.asarray(point, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(xy)):
            raise ValueError("goal point must be finite")
        self._validate_xy(xy, "goal")
        self._goal_xy = xy.copy()
        goal_xyz = np.asarray([xy[0], xy[1], self._box_rest_z], dtype=np.float64)
        # Model default is copied to new rollout data and the fused native MJB;
        # live mocap position is updated for the viewer/plant as well.
        self.model.body_pos[self._goal_body_id] = goal_xyz
        self.data.mocap_pos[self._goal_mocap_id] = goal_xyz
        self.mujoco.mj_forward(self.model, self.data)

    def goal_xyz(self, data=None) -> np.ndarray:
        d = self.data if data is None else data
        return np.asarray(d.site_xpos[self._goal_site_id], dtype=np.float64).copy()

    def start_xyz(self) -> np.ndarray:
        return np.array([self._start_xy[0], self._start_xy[1], self._box_rest_z], dtype=np.float64)

    @property
    def table_body_id(self) -> int:
        return self._table_body_id

    @property
    def table_geom_id(self) -> int:
        return self._table_geom_id

    @property
    def table_geom_ids(self) -> np.ndarray:
        return np.asarray([self._table_geom_id], dtype=np.int32)

    @property
    def task_rest_height(self) -> float:
        return float(self._box_rest_z)

    @property
    def task_half_height(self) -> float:
        return float(self.model.geom_size[self._task_geom_id, 2])

    def task_up(self, data=None) -> float:
        d = self.data if data is None else data
        return float(np.asarray(d.xmat[self.task_body_id]).reshape(3, 3)[2, 2])

    def tip_box_surface_distance(self, data=None) -> float:
        """Euclidean fingertip-to-closest-point distance on the oriented box."""
        d = self.data if data is None else data
        center = np.asarray(d.geom_xpos[self.task_geom_id], dtype=np.float64)
        rotation = np.asarray(d.geom_xmat[self.task_geom_id], dtype=np.float64).reshape(3, 3)
        local = rotation.T @ (self.tip_xyz(d) - center)
        half = np.asarray(self.model.geom_size[self.task_geom_id], dtype=np.float64)
        return float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0)))

    def tip_push_face_distance(self, direction_xy, data=None) -> float:
        """Tip distance to the side face opposite the desired XY push direction."""
        d = self.data if data is None else data
        direction = np.asarray(direction_xy, dtype=np.float64).reshape(2)
        magnitude = float(np.linalg.norm(direction))
        if magnitude <= 1e-12:
            return self.tip_box_surface_distance(d)
        direction /= magnitude
        center = np.asarray(d.geom_xpos[self.task_geom_id], dtype=np.float64)
        rotation = np.asarray(d.geom_xmat[self.task_geom_id], dtype=np.float64).reshape(3, 3)
        local = rotation.T @ (self.tip_xyz(d) - center)
        half = np.asarray(self.model.geom_size[self.task_geom_id], dtype=np.float64)
        best_alignment = -np.inf
        best_distance = np.inf
        for axis in (0, 1):
            for sign in (-1.0, 1.0):
                alignment = float(np.dot(-sign * rotation[:2, axis], direction))
                if alignment <= best_alignment:
                    continue
                closest = np.clip(local, -half, half)
                closest[axis] = sign * half[axis]
                best_alignment = alignment
                best_distance = float(np.linalg.norm(local - closest))
        return best_distance

    def tip_target_distance(self, data=None) -> float:
        return self.tip_box_surface_distance(data)

    def box_goal_distance(self, data=None) -> float:
        return float(np.linalg.norm(self.task_xyz(data) - self.goal_xyz(data)))


__all__ = ["FrankaPushRobot"]
