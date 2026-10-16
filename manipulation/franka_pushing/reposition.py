"""Deterministic collision-clearance transfer between opposite push faces.

A 0.2-second pushing MPPI horizon cannot discover the ~0.4-m excursion
needed to switch sides of a 32-cm box.  Between waypoint legs, a small
Cartesian IK controller takes the same three-stage path for *every* sampler:

    lift above box -> cross over box -> lower behind the new push face.

MPPI remains responsible for pushing.  Reposition steps are recorded separately
and excluded from the MPPI runtime average.  This is an explicit benchmark
choice, not a change to the sampled pushing objective or MuJoCo physics.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def behind_box_target(box_center, box_rotation, half_extents, goal_xy,
                      standoff: float) -> np.ndarray:
    """World XYZ near the middle of the supporting face opposite the goal.

    For a rotated box, support is the XY projection of its three box axes,
    rather than assuming the box remains axis-aligned.  The direction is
    recomputed against the live box center to tolerate small box drift.
    """
    center = np.asarray(box_center, dtype=float).reshape(3)
    rotation = np.asarray(box_rotation, dtype=float).reshape(3, 3)
    half = np.asarray(half_extents, dtype=float).reshape(3)
    goal = np.asarray(goal_xy, dtype=float).reshape(2)
    delta = goal - center[:2]
    distance = np.linalg.norm(delta)
    if distance < 1e-6:
        raise ValueError("Reposition goal must be distinct from the box position")
    direction = delta / distance
    # Project the box onto the direction of travel.  Include both XY axes,
    # ignoring small pitch/roll for the target-height convention.
    support = sum(abs(np.dot(rotation[:2, axis], direction)) * half[axis]
                  for axis in (0, 1))
    return np.array([*(center[:2] - (support + standoff) * direction), center[2]])


@dataclass
class PushReposition:
    """Stateful lift/cross/lower trajectory, executed at the control rate."""

    goal_xy: tuple[float, float]
    clearance: float = 0.12  # Above the box's top surface, measured at tip center.
    standoff: float = 0.045  # Beyond the opposite side face.
    max_seconds: float = 9.0
    speed_mps: float = 0.30
    tolerance_m: float = 0.023
    damping: float = 0.075
    stage: str = "lift"
    elapsed: float = 0.0
    finished: bool = False
    failed: bool = False

    def _box_pose(self, robot):
        gid = int(robot.task_geom_id)
        model, data = robot.model, robot.data
        center = np.asarray(data.geom_xpos[gid], dtype=float).copy()
        rotation = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3).copy()
        half = np.asarray(model.geom_size[gid], dtype=float).copy()
        return center, rotation, half

    def target(self, robot) -> np.ndarray:
        center, rotation, half = self._box_pose(robot)
        behind = behind_box_target(center, rotation, half, self.goal_xy, self.standoff)
        tip = np.asarray(robot.tip_xyz(), dtype=float)
        safe_z = max(center[2] + half[2] + self.clearance,
                     float(robot.task_rest_height) + half[2] + self.clearance)
        if self.stage == "lift":
            return np.array([tip[0], tip[1], safe_z])
        if self.stage == "cross":
            return np.array([behind[0], behind[1], safe_z])
        if self.stage == "lower":
            return behind
        raise RuntimeError(f"invalid reposition stage {self.stage!r}")

    def action(self, robot, control_dt: float) -> np.ndarray:
        """Damped least-squares Cartesian IK mapped onto Panda position servos."""
        tip = np.asarray(robot.tip_xyz(), dtype=float)
        err = self.target(robot) - tip
        # In lift and cross, never command a downward component that would
        # send the finger through the physical box.
        velocity = np.clip(3.0 * err, -self.speed_mps, self.speed_mps)
        norm = float(np.linalg.norm(velocity))
        if norm > self.speed_mps:
            velocity *= self.speed_mps / norm
        jac = np.asarray(robot.cartesian_jacobian(), dtype=float)
        if jac.shape != (3, 7):
            raise ValueError(f"expected 3x7 tip Jacobian, got {jac.shape}")
        dq = jac.T @ np.linalg.solve(
            jac @ jac.T + self.damping**2 * np.eye(3), velocity
        )
        max_joint_v = np.asarray(robot.joint_velocity_limits, dtype=float) * float(robot.motion_limit_scale)
        dq = np.clip(dq, -max_joint_v, max_joint_v)
        # The position servos follow the previously applied command under
        # Panda's velocity limiter.  A 0.10-s lookahead avoids slow creeping.
        q = np.asarray(robot.data.qpos[robot._joint_qadr], dtype=float)
        requested = q + dq * max(0.10, 4.0 * control_dt)
        return robot.clip_ctrl(requested)

    def observe(self, robot, dt: float) -> None:
        """Advance through waypoints only after actual fingertip arrival."""
        if self.finished or self.failed:
            return
        self.elapsed += dt
        if self.elapsed > self.max_seconds:
            self.failed = True
            return
        tip = np.asarray(robot.tip_xyz(), dtype=float)
        dest = self.target(robot)
        distance = float(np.linalg.norm(tip - dest))
        if self.stage == "lift":
            if abs(float(tip[2] - dest[2])) < self.tolerance_m:
                self.stage = "cross"
        elif self.stage == "cross":
            if distance < 1.5 * self.tolerance_m:
                self.stage = "lower"
        elif self.stage == "lower":
            if distance < self.tolerance_m:
                self.finished = True


__all__ = ["PushReposition", "behind_box_target"]
