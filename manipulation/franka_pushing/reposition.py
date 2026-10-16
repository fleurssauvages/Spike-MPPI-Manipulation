"""MPPI-controlled box-avoiding transfer between pushing faces.

Samples with the SAME controller proposal generator as pushing; separate Python
MuJoCo rollouts implement a collision-aware reaching objective. Pushing remains
native fused. No deterministic IK action is used during transfer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np
import mujoco

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
    goal_xy: tuple[float, float]
    clearance: float = 0.12
    standoff: float = 0.045
    max_seconds: float = 9.0
    tolerance_m: float = 0.023
    stage: str = "lift"
    elapsed: float = 0.0
    finished: bool = False
    failed: bool = False
    horizon: int = 20
    num_rollouts: int = 32
    target_weight: float = 30.0
    contact_weight: float = 150.0
    safety_margin: float = 0.025
    _nominal: np.ndarray | None = field(default=None, repr=False)
    _scratch: object = field(default=None, repr=False)

    def _box_pose(self, robot):
        gid = int(robot.task_geom_id)
        return (np.asarray(robot.data.geom_xpos[gid]).copy(),
                np.asarray(robot.data.geom_xmat[gid]).reshape(3, 3).copy(),
                np.asarray(robot.model.geom_size[gid]).copy())

    def target(self, robot):
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
        raise RuntimeError(self.stage)

    def _sample(self, controller, nominal, data):
        option = str(controller.sampling.value)
        if option == 'spike':
            samples = controller._sample_spike(nominal)
        elif option == 'poisson':
            samples = controller._sample_poisson(nominal)
        elif option in ('gaussian-poisson', 'gaussian-spike'):
            samples = controller._sample_spectrum_matched_gaussian(nominal)
        elif option == 'spline':
            samples = controller._sample_spline(nominal)
        elif option == 'guided':
            samples = controller._sample_guided(nominal, adaptive_diagonal=False)
        elif option == 'diag-lowrank':
            samples = controller._sample_guided(nominal, adaptive_diagonal=True)
        elif option == 'icem':
            samples = controller._sample_icem(nominal)
        else:
            samples = controller._sample_standard(nominal)
        samples = np.array(samples, dtype=np.float64, copy=True)
        samples = controller._convert_proposals_to_effector_space(samples, nominal, data)
        samples[0] = nominal
        return samples

    def _copy_state(self, model, source, dest):
        # Keep the same physical scene, including target-box pose and mocap state.
        mujoco.mj_resetData(model, dest)
        dest.qpos[:] = source.qpos
        dest.qvel[:] = source.qvel
        dest.act[:] = source.act
        dest.ctrl[:] = source.ctrl
        dest.time = source.time
        dest.mocap_pos[:] = source.mocap_pos
        dest.mocap_quat[:] = source.mocap_quat
        if model.nuserdata:
            dest.userdata[:] = source.userdata
        mujoco.mj_forward(model, dest)

    def _collision_cost(self, robot, data, robot_geom_ids):
        box_id = int(robot.task_geom_id)
        # Margin begins penalizing before interpenetration; all actual robot-box
        # contacts receive an additional prohibitive collision cost.
        cost = 0.0
        for k in range(data.ncon):
            c = data.contact[k]
            other = c.geom2 if c.geom1 == box_id else (c.geom1 if c.geom2 == box_id else -1)
            if other in robot_geom_ids:
                cost += self.contact_weight * (1.0 + max(0., self.safety_margin - float(c.dist)))
        return cost

    def action(self, robot, controller, control_dt):
        """Return sampled MPPI action and timing diagnostics."""
        tic = time.perf_counter()
        model = robot.model
        if self._scratch is None:
            self._scratch = mujoco.MjData(model)
        horizon = int(controller.cfg.horizon)
        nroll = int(controller.cfg.num_rollouts)
        if self._nominal is None or self._nominal.shape != (horizon, robot.nu):
            self._nominal = np.tile(robot.data.ctrl.copy(), (horizon, 1))
        target = self.target(robot)
        samples = self._sample(controller, self._nominal, robot.data)
        costs = np.empty(len(samples), dtype=float)
        substeps = int(controller.control_substeps)
        box_body = int(robot.task_body_id)
        robot_geom_ids = set(int(i) for i in range(model.ngeom)
                             if model.geom_bodyid[i] != box_body
                             and model.body_rootid[model.geom_bodyid[i]] ==
                                 model.body_rootid[model.geom_bodyid[int(robot._tip_geom_ids[0])]]
                             and model.geom_contype[i] != 0)
        for i, trajectory in enumerate(samples):
            data = self._scratch
            self._copy_state(model, robot.data, data)
            total = 0.0
            for command in trajectory:
                data.ctrl[:] = robot.rate_limited_ctrl(command, control_dt=control_dt, data=data)
                mujoco.mj_step(model, data, nstep=substeps)
                tip = robot.tip_xyz(data)
                error = tip - target
                total += self.target_weight * float(error @ error)
                total += self._collision_cost(robot, data, robot_geom_ids)
                # Penalize entering the box even when fast movement skips contact.
                center = data.geom_xpos[robot.task_geom_id]
                rot = data.geom_xmat[robot.task_geom_id].reshape(3, 3)
                local = rot.T @ (tip - center)
                half = model.geom_size[robot.task_geom_id]
                if np.all(np.abs(local) < half + self.safety_margin):
                    total += self.contact_weight
            costs[i] = total / max(1, horizon)
        # Standard stable MPPI exponential weighting, with adaptive cost scale.
        scaled = costs - np.min(costs)
        positive = scaled[scaled > 1e-9]
        temperature = max(1e-5, float(np.median(positive)) if positive.size else 1.0)
        weights = np.exp(-scaled / temperature)
        weights /= np.sum(weights)
        self._nominal = np.einsum('n,ntu->tu', weights, samples)
        command = self._nominal[0].copy()
        self._nominal[:-1] = self._nominal[1:].copy()
        self._nominal[-1] = self._nominal[-2]
        elapsed_ms = (time.perf_counter() - tic) * 1000.0
        return command, {'timing_ms': {'total': elapsed_ms, 'rollouts': elapsed_ms},
                         'best_cost': float(np.min(costs)),
                         'weighted_rollout_cost': float(weights @ costs)}

    def observe(self, robot, dt):
        if self.finished or self.failed:
            return
        self.elapsed += dt
        if self.elapsed > self.max_seconds:
            self.failed = True
            return
        tip = np.asarray(robot.tip_xyz(), dtype=float)
        goal = self.target(robot)
        d = float(np.linalg.norm(tip-goal))
        if self.stage == 'lift' and abs(float(tip[2]-goal[2])) < self.tolerance_m:
            self.stage = 'cross'
            self._nominal = None
        elif self.stage == 'cross' and d < 1.5*self.tolerance_m:
            self.stage = 'lower'
            self._nominal = None
        elif self.stage == 'lower' and d < self.tolerance_m:
            self.finished = True

__all__ = ['PushReposition', 'behind_box_target']
