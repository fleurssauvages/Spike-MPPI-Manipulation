"""Separate fused, reaching-only velocity rollout adapter.

Preserves MPPI proposal generation; replaces candidate reaching evaluation only.
This native evaluator is a kinematic approximation. It does not simulate contact,
real FR3 inertia, or the original fused backend's complete cost function.
"""
from __future__ import annotations
import numpy as np


class FusedVelocityBatch:
    def __init__(self, robot, *, kp=2.0, kd=0.2, vmax=0.4, amax=0.6,
                 lowlevel_amax=2.0, lowlevel_jerk=100.0, lowlevel_tau=0.03,
                 period=0.01):
        try:
            from manipulation._fused_velocity import FusedVelocityReaching
        except ImportError as exc:
            raise RuntimeError(
                "Build the separate velocity native extension first: "
                "python manipulation/setup_velocity_native.py build_ext --inplace"
            ) from exc
        if int(getattr(robot, 'fused_task_mode', -1)) != 3:
            raise ValueError('separate velocity fused backend supports free-space reaching only')
        self.robot = robot
        self.native = FusedVelocityReaching(
            str(robot.xml_path), 'slim_tip',
            np.asarray(robot._joint_qadr, dtype=np.int32),
            np.asarray(robot._joint_dadr, dtype=np.int32))
        self.gains = dict(kp=kp, kd=kd, vmax=vmax, outer_amax=amax,
                          low_amax=lowlevel_amax, jerk=lowlevel_jerk,
                          tau=lowlevel_tau, period=period)

    def __call__(self, robot, start_snapshot, control_batch, track, current_s,
                 *, control_substeps, nominal_controls=None, cost_cfg=None,
                 workers=0, native_batcher=None, **unused):
        if robot is not self.robot:
            raise ValueError('fused velocity evaluator robot mismatch')
        controls = np.ascontiguousarray(control_batch, dtype=np.float64)
        n, h, d = controls.shape
        if d != 7:
            raise ValueError('expected 7 arm controls')
        if nominal_controls is None:
            nominal = np.zeros((h, 7), dtype=np.float64)
        else:
            nominal = np.ascontiguousarray(nominal_controls, dtype=np.float64)
        cfg = cost_cfg
        if cfg is None:
            from manipulation.mppi.rollout import RolloutCostConfig
            cfg = RolloutCostConfig()
        params = dict(
            radius=float(cfg.reach_safe_radius),
            region_weight=float(cfg.reach_region_weight),
            terminal_weight=float(cfg.reach_terminal_weight),
            velocity_weight=float(cfg.reach_velocity_weight),
            velocity_sigma=float(cfg.reach_velocity_sigma),
            velocity_ref=float(cfg.reach_velocity_ref),
            settle_weight=float(cfg.reach_settle_weight),
            settle_steps=int(cfg.reach_settle_steps),
            accel_weight=float(cfg.reach_acceleration_weight),
            control_weight=float(cfg.control_deviation_weight),
        )
        return self.native.evaluate(
            controls,
            np.ascontiguousarray(start_snapshot.qpos, dtype=np.float64),
            np.ascontiguousarray(start_snapshot.qvel, dtype=np.float64),
            np.ascontiguousarray(robot.task_xyz(), dtype=np.float64),
            nominal,
            **self.gains, **params)
