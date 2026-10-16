"""Franka with a rigid offset tray/racket and a freely moving MuJoCo ball.

The same seven actuators and all existing MPPI samplers are retained. The task
cost is fused into a dedicated native C++ evaluator, with the original
MuJoCo batched state/sensor path retained for cost-parity verification.
"""
from __future__ import annotations

from importlib import import_module
from pathlib import Path
import numpy as np

from .fetch_franka_menagerie import DEFAULT_DIR, fetch_franka_menagerie
from .franka import FrankaReachRobot


class FrankaBallRobot(FrankaReachRobot):
    fused_task_mode = 5  # ball evaluator is separate from legacy modes 1..4

    def __init__(
        self, *, task: str,
        auto_fetch: bool = True,
        motion_limit_scale: float = 1.0,
        initial_ball_offset_xy: tuple[float, float] = (0.0, 0.0),
        initial_ball_velocity_xy: tuple[float, float] = (0.0, 0.0),
        initial_ball_spin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
        drop_height: float = 0.24,
        xml_path: str | Path | None = None,
    ) -> None:
        if task not in {"balancing", "bouncing"}:
            raise ValueError("task must be 'balancing' or 'bouncing'")
        self.task = task
        self.name = f"franka-ball-{task}"
        self._tool_site_id = -1
        self._initial_offset = np.asarray(initial_ball_offset_xy, dtype=float).reshape(2)
        self._initial_velocity = np.asarray(initial_ball_velocity_xy, dtype=float).reshape(2)
        self._initial_spin = np.asarray(initial_ball_spin_xyz, dtype=float).reshape(3)
        if (not np.all(np.isfinite(self._initial_offset)) or
            not np.all(np.isfinite(self._initial_velocity)) or
            not np.all(np.isfinite(self._initial_spin))):
            raise ValueError("initial ball state must be finite")
        if not (np.isfinite(drop_height) and drop_height >= 0.0):
            raise ValueError("drop_height must be finite and nonnegative")
        self.drop_height = float(drop_height)
        if xml_path is None:
            if auto_fetch:
                fetch_franka_menagerie(DEFAULT_DIR)
            xml_path = DEFAULT_DIR / f"franka_ball_{task}_scene.xml"
        # The free joint named reach_target is now a *dynamic ball*, not a goal.
        super().__init__(
            xml_path=xml_path, auto_fetch=False,
            motion_limit_scale=motion_limit_scale,
            target_xyz=(0.5545, 0.125, 0.50),
        )
        self._tool_site_id = int(self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_SITE, "ball_tool_surface"
        ))
        self._tool_geom_id = self._geom_id("ball_tool_geom")
        if self._tool_site_id < 0:
            raise ValueError("ball scene must contain 'ball_tool_surface' site")
        self.ball_radius = float(self.model.geom_size[self.task_geom_id, 0])
        self.tool_half_width = float(self.model.geom_size[self._tool_geom_id, 0])
        self._sensors = {}
        for name in (
            "ball_tool_pos", "ball_tool_x", "ball_tool_y", "ball_tool_z",
            "ball_tool_linvel", "ball_tool_angvel",
        ):
            sid = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_SENSOR, name))
            if sid < 0 or int(self.model.sensor_dim[sid]) != 3:
                raise ValueError(f"missing three-component sensor {name}")
            adr = int(self.model.sensor_adr[sid])
            self._sensors[name] = slice(adr, adr + 3)
        cost_module = import_module(f"manipulation.franka_{task}.cost")
        self._score_rollouts = cost_module.score_rollouts
        self.cost_weights = (cost_module.BalanceWeights() if task == "balancing"
                             else cost_module.BounceWeights())
        self._previous_ball_contact = False
        self._impact_episode_incoming = False
        self._impact_counted = False
        self.bounce_count = 0
        self.last_outgoing_vz = float("nan")
        self.reset()
        self.initial_root_height = float(self.tip_xyz()[2])

    def reset(self) -> None:
        super().reset()
        if self._tool_site_id >= 0:
            self.reset_ball()

    def tool_frame(self, data=None) -> tuple[np.ndarray, np.ndarray]:
        d = self.data if data is None else data
        pos = np.asarray(d.site_xpos[self._tool_site_id], dtype=np.float64).copy()
        rot = np.asarray(d.site_xmat[self._tool_site_id], dtype=np.float64).reshape(3, 3).copy()
        return pos, rot

    def reset_ball(self, *, offset_xy=None, velocity_xy=None) -> None:
        """Place the ball relative to the tool's *actual* current frame."""
        offset = self._initial_offset if offset_xy is None else np.asarray(offset_xy, dtype=float).reshape(2)
        velocity = self._initial_velocity if velocity_xy is None else np.asarray(velocity_xy, dtype=float).reshape(2)
        if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(velocity)):
            raise ValueError("ball reset state must be finite")
        self.mujoco.mj_forward(self.model, self.data)
        pos, axes = self.tool_frame()
        z_gap = 0.0005 if self.task == "balancing" else self.drop_height
        ball_pos = (
            pos + offset[0] * axes[:, 0] + offset[1] * axes[:, 1]
            + (self.ball_radius + z_gap) * axes[:, 2]
        )
        self.set_target_xyz(ball_pos)
        v = velocity[0] * axes[:, 0] + velocity[1] * axes[:, 1]
        dadr = int(self._task_dof_adr)
        self.data.qvel[dadr:dadr + 3] = v
        # Free-joint angular qvel is in the world frame. The ball starts with
        # physically consistent, deterministic spin; later contact generates
        # additional spin through friction without scripted trajectories.
        self.data.qvel[dadr + 3:dadr + 6] = self._initial_spin
        self.mujoco.mj_forward(self.model, self.data)
        self.bounce_count = 0
        self.last_outgoing_vz = float("nan")
        self._previous_ball_contact = False
        self._impact_episode_incoming = False
        self._impact_counted = False

    def tip_xyz(self, data=None) -> np.ndarray:
        # For IK proposals the tool frame, not the original small fingertip,
        # is the controlled Cartesian point. Guard the superclass constructor.
        if self._tool_site_id < 0:
            return super().tip_xyz(data)
        return self.tool_frame(data)[0]

    def cartesian_jacobian(self, data=None) -> np.ndarray:
        if self._tool_site_id < 0:
            return super().cartesian_jacobian(data)
        d = self.data if data is None else data
        jp = np.zeros((3, self.model.nv), dtype=np.float64)
        jr = np.zeros_like(jp)
        self.mujoco.mj_jacSite(self.model, d, jp, jr, self._tool_site_id)
        return jp[:, self._joint_dadr].copy()

    def ball_local_state(self, data=None) -> tuple[np.ndarray, np.ndarray, float]:
        d = self.data if data is None else data
        center, rot = self.tool_frame(d)
        relative = np.asarray(self.task_xyz(d) - center)
        coords = rot.T @ relative
        # Velocity of the point on the plate underneath the ball, including
        # tool angular velocity. For diagnostics use MuJoCo frame sensors.
        sensor = np.asarray(d.sensordata, dtype=np.float64)
        lin = sensor[self._sensors["ball_tool_linvel"]]
        ang = sensor[self._sensors["ball_tool_angvel"]]
        v_rel = self.task_velocity(d) - lin - np.cross(ang, relative)
        vel_xy = (rot.T @ v_rel)[:2]
        return coords[:2].copy(), np.asarray(vel_xy).copy(), float(coords[2] - self.ball_radius)

    def has_ball_tool_contact(self, data=None) -> bool:
        d = self.data if data is None else data
        ball, tool = int(self.task_geom_id), int(self._tool_geom_id)
        for i in range(int(d.ncon)):
            c = d.contact[i]
            if ((int(c.geom1) == ball and int(c.geom2) == tool) or
                (int(c.geom1) == tool and int(c.geom2) == ball)):
                return True
        return False

    def limit_rollout_controls(self, controls: np.ndarray, initial_ctrl: np.ndarray, dt: float) -> np.ndarray:
        """Rate-limit each candidate like FrankaReachRobot.step_control."""
        arr = np.asarray(controls, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[2] != self.nu:
            raise ValueError("rollout controls must have shape (N, horizon, nu)")
        low, high = self.control_bounds()
        bound = self.motion_limit_scale * self._joint_velocity_limits * float(dt)
        applied = np.empty_like(arr)
        prev = np.broadcast_to(np.asarray(initial_ctrl, dtype=float), (arr.shape[0], self.nu)).copy()
        for t in range(arr.shape[1]):
            desired = np.clip(arr[:, t], low, high)
            prev = np.clip(prev + np.clip(desired - prev, -bound, bound), low, high)
            applied[:, t] = prev
        return applied

    def evaluate_ball_rollouts(
        self, states: np.ndarray, sensors: np.ndarray,
        controls: np.ndarray, nominal: np.ndarray, initial_ctrl: np.ndarray,
        ctrl_scale: np.ndarray, *, substeps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score sampled *real physics* trajectories; no surrogate ball dynamics."""
        qadr = 1 + int(self.task_qpos_adr)
        dadr = 1 + int(self.model.nq) + int(self._task_dof_adr)
        pos = states[..., qadr:qadr + 3]
        vel = states[..., dadr:dadr + 3]
        def ss(name):
            return sensors[..., self._sensors[name]]
        tool = ss("ball_tool_pos")
        xx, yy, zz = ss("ball_tool_x"), ss("ball_tool_y"), ss("ball_tool_z")
        lin, ang = ss("ball_tool_linvel"), ss("ball_tool_angvel")
        delta = pos - tool
        relative_velocity = vel - lin - np.cross(ang, delta)
        features = {
            "ball_pos": pos,
            "ball_velocity": vel,
            "tool_pos": tool,
            "tool_velocity": lin,
            "normal": zz,
            "relative_xy": np.stack((np.sum(delta * xx, axis=-1), np.sum(delta * yy, axis=-1)), axis=-1),
            "relative_velocity_xy": np.stack((np.sum(relative_velocity * xx, axis=-1), np.sum(relative_velocity * yy, axis=-1)), axis=-1),
            "relative_velocity_normal": np.sum(relative_velocity * zz, axis=-1),
            "gap": np.sum(delta * zz, axis=-1) - self.ball_radius,
            "half_width": self.tool_half_width,
            "ball_radius": self.ball_radius,
            "dt": self.physics_dt,
        }
        cost = self._score_rollouts(
            features, controls, nominal, initial_ctrl, ctrl_scale,
            substeps=substeps, weights=self.cost_weights,
        )
        invalid = ~np.isfinite(cost)
        return np.where(invalid, np.inf, cost), invalid

    def step_control(self, ctrl: np.ndarray, *, substeps: int = 1, data=None) -> None:
        d = self.data if data is None else data
        nstep = max(1, int(substeps))
        d.ctrl[:] = self.rate_limited_ctrl(ctrl, control_dt=nstep * self.physics_dt, data=d)
        for _ in range(nstep):
            previous_vz = float(self.task_velocity(d)[2])
            self.mujoco.mj_step(self.model, d)
            if self.task == "bouncing" and d is self.data:
                contact = self.has_ball_tool_contact(d)
                outgoing_vz = float(self.task_velocity(d)[2])
                if contact and not self._previous_ball_contact:
                    self._impact_episode_incoming = previous_vz < -0.1
                    self._impact_counted = False
                if (contact and self._impact_episode_incoming and
                    not self._impact_counted and outgoing_vz > 0.1):
                    self.bounce_count += 1
                    self.last_outgoing_vz = outgoing_vz
                    self._impact_counted = True
                if not contact:
                    self._impact_episode_incoming = False
                    self._impact_counted = False
                self._previous_ball_contact = contact


__all__ = ["FrankaBallRobot"]
