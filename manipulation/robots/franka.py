from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .fetch_franka_menagerie import DEFAULT_DIR, fetch_franka_menagerie


@dataclass
class RobotSnapshot:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray


FRANKA_HOME_QPOS = np.asarray(
    [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853], dtype=np.float64
)
PANDA_JOINT_VELOCITY_LIMITS = np.asarray(
    [2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610], dtype=np.float64
)
# Metadata retained for the native evaluator constructor.  The online command
# path intentionally applies only the Panda velocity limit, not acceleration.
PANDA_JOINT_ACCELERATION_LIMITS = np.asarray(
    [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], dtype=np.float64
)


class FrankaReachRobot:
    """Official Panda arm with a short central tip for free-space reaching tests."""

    name = "franka-reach"
    fused_task_mode = 3  # fixed 3-D point target in the fused evaluator

    def __init__(
        self,
        xml_path: str | Path | None = None,
        *,
        auto_fetch: bool = True,
        motion_limit_scale: float = 1.0,
        acceleration_limit_scale: float | None = 1.0,
        target_xyz=(0.45, -0.25, 0.50),
        actuator_lag_kp: float | None = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("Install mujoco before constructing FrankaReachRobot") from exc

        self.mujoco = mujoco
        model_dir = DEFAULT_DIR
        if xml_path is None:
            if auto_fetch:
                try:
                    fetch_franka_menagerie(model_dir)
                except Exception as exc:
                    raise RuntimeError(
                        "Could not fetch/build the Panda slim-tool model. Run "
                        "`python -m manipulation.robots.fetch_franka_menagerie` once while online."
                    ) from exc
            self.xml_path = model_dir / "franka_reach_scene.xml"
        else:
            self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(self.xml_path)

        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self._joint_names = tuple(f"joint{i}" for i in range(1, 8))
        self._joint_ids = np.asarray(
            [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self._joint_names],
            dtype=np.int32,
        )
        if np.any(self._joint_ids < 0):
            raise ValueError("Panda arm is missing one or more joints")
        self._joint_qadr = np.asarray(
            [self.model.jnt_qposadr[int(j)] for j in self._joint_ids], dtype=np.int32
        )
        self._joint_dadr = np.asarray(
            [self.model.jnt_dofadr[int(j)] for j in self._joint_ids], dtype=np.int32
        )
        self._arm_actuator_ids = np.arange(7, dtype=np.int32)
        if int(self.model.nu) != 7:
            raise ValueError(f"slim-tool Panda should expose 7 arm actuators, got {self.model.nu}")
        self.actuator_lag_kp = None if actuator_lag_kp is None else float(actuator_lag_kp)
        if self.actuator_lag_kp is not None:
            self._apply_actuator_lag(self.actuator_lag_kp)

        self.motion_limit_scale = float(motion_limit_scale)
        if not np.isfinite(self.motion_limit_scale) or not (0.0 < self.motion_limit_scale <= 1.0):
            raise ValueError("motion_limit_scale must be finite and in (0, 1]")
        self.acceleration_limit_scale = float(1.0 if acceleration_limit_scale is None else acceleration_limit_scale)
        self._joint_velocity_limits = PANDA_JOINT_VELOCITY_LIMITS.copy()
        self._joint_acceleration_limits = PANDA_JOINT_ACCELERATION_LIMITS.copy()

        self.base_body_id = self._body_id("link0")
        self._tool_body_id = self._body_id("slim_gripper")
        self.root_body_id = self._tool_body_id
        self.rollout_root_body_id = self._tool_body_id
        self._tip_geom_ids = (self._geom_id("slim_tip"),)
        self._floor_geom_id = self._geom_id("floor")

        self._task_body_id = self._body_id("reach_target")
        self._task_geom_id = self._geom_id("reach_target_geom")
        self._task_qpos_adr = self._find_free_qpos_adr(self._task_body_id)
        self._task_dof_adr = self._find_free_dof_adr(self._task_body_id)
        if self._task_qpos_adr is None or self._task_dof_adr is None:
            raise ValueError("reach_target must have a free joint")

        self._robot_collision_geom_ids = self._find_robot_collision_geoms()
        self.default_ctrl = FRANKA_HOME_QPOS.copy()
        self._target_xyz = np.asarray(target_xyz, dtype=np.float64).reshape(3).copy()
        if not np.all(np.isfinite(self._target_xyz)):
            raise ValueError("target_xyz must be finite")
        self.reset()
        self.set_target_xyz(self._target_xyz)
        self.initial_root_height = float(self.tip_xyz()[2])

    def _apply_actuator_lag(self, lag_kp: float) -> None:
        """Make the arm follow ctrl like the real 1 kHz loop qdot = lag_kp * (ctrl - q).

        Gravity is compensated on every arm body (the real velocity loop has no
        steady-state sag), and each position actuator gets kv = kp / lag_kp. With
        the stiff Menagerie kp, inertia is negligible and the joint response is
        first order with time constant 1 / lag_kp. Must run before rollout
        backends are built, since they copy this in-memory model.
        """
        if not np.isfinite(lag_kp) or lag_kp <= 0.0:
            raise ValueError("actuator_lag_kp must be finite and positive")
        base = self._body_id("link0")
        for body in range(int(self.model.nbody)):
            if body == base or self._is_descendant_body(body, base):
                self.model.body_gravcomp[body] = 1.0
        ids = self._arm_actuator_ids
        self.model.actuator_biasprm[ids, 2] = -self.model.actuator_gainprm[ids, 0] / lag_kp

    def _body_id(self, name: str) -> int:
        out = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name))
        if out < 0:
            raise ValueError(f"missing body {name!r}")
        return out

    def _geom_id(self, name: str) -> int:
        out = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, name))
        if out < 0:
            raise ValueError(f"missing geom {name!r}")
        return out

    def _find_free_qpos_adr(self, body_id: int) -> int | None:
        free_type = int(self.mujoco.mjtJoint.mjJNT_FREE)
        for j in range(int(self.model.njnt)):
            if int(self.model.jnt_bodyid[j]) == int(body_id) and int(self.model.jnt_type[j]) == free_type:
                return int(self.model.jnt_qposadr[j])
        return None

    def _find_free_dof_adr(self, body_id: int) -> int | None:
        free_type = int(self.mujoco.mjtJoint.mjJNT_FREE)
        for j in range(int(self.model.njnt)):
            if int(self.model.jnt_bodyid[j]) == int(body_id) and int(self.model.jnt_type[j]) == free_type:
                return int(self.model.jnt_dofadr[j])
        return None

    def _is_descendant_body(self, body_id: int, ancestor_id: int) -> bool:
        body = int(body_id)
        ancestor = int(ancestor_id)
        while body > 0:
            if body == ancestor:
                return True
            body = int(self.model.body_parentid[body])
        return body == ancestor

    def _find_robot_collision_geoms(self) -> tuple[int, ...]:
        ids: list[int] = []
        for geom_id in range(int(self.model.ngeom)):
            body_id = int(self.model.geom_bodyid[geom_id])
            if not self._is_descendant_body(body_id, self.base_body_id):
                continue
            if int(self.model.geom_contype[geom_id]) == 0 and int(self.model.geom_conaffinity[geom_id]) == 0:
                continue
            ids.append(int(geom_id))
        return tuple(ids)

    @property
    def is_muscle_model(self) -> bool:
        return False

    @property
    def muscle_pairs(self) -> tuple[tuple[int, int], ...]:
        return ()

    @property
    def motor_pool_count(self) -> int:
        return self.nu

    @property
    def nu(self) -> int:
        return int(self.model.nu)

    @property
    def physics_dt(self) -> float:
        return float(self.model.opt.timestep)

    @property
    def task_body_id(self) -> int:
        return int(self._task_body_id)

    @property
    def task_qpos_adr(self) -> int:
        return int(self._task_qpos_adr)

    @property
    def task_geom_id(self) -> int:
        return int(self._task_geom_id)

    # Compatibility IDs required by the generic/fused rollout constructor.
    # The free-space benchmark assigns zero weight to all contact terms.
    @property
    def table_body_id(self) -> int:
        return 0  # MuJoCo world body; the floor geom belongs to it.

    @property
    def table_geom_id(self) -> int:
        return int(self._floor_geom_id)

    @property
    def table_geom_ids(self) -> np.ndarray:
        return np.asarray([self._floor_geom_id], dtype=np.int32)

    @property
    def task_rest_height(self) -> float:
        return float(self.task_xyz()[2])

    @property
    def task_half_height(self) -> float:
        return float(self.model.geom_size[int(self._task_geom_id), 0])

    @property
    def rollout_root_geom_ids(self) -> np.ndarray:
        return np.asarray(self._tip_geom_ids, dtype=np.int32)

    @property
    def rollout_arm_actuator_ids(self) -> np.ndarray:
        return self._arm_actuator_ids.copy()

    @property
    def rollout_arm_qpos_adr(self) -> np.ndarray:
        return self._joint_qadr.copy()

    @property
    def rollout_arm_dof_adr(self) -> np.ndarray:
        return self._joint_dadr.copy()

    @property
    def joint_velocity_limits(self) -> np.ndarray:
        return self._joint_velocity_limits.copy()

    @property
    def joint_acceleration_limits(self) -> np.ndarray:
        return self._joint_acceleration_limits.copy()

    @property
    def active_control_mask(self) -> np.ndarray:
        return np.ones(self.nu, dtype=bool)

    @property
    def robot_collision_geom_ids(self) -> np.ndarray:
        return np.asarray(self._robot_collision_geom_ids, dtype=np.int32)

    @property
    def forbidden_box_contact_geom_ids(self) -> np.ndarray:
        return np.empty(0, dtype=np.int32)

    def reset(self) -> None:
        home_key = int(self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_KEY, "home"))
        if home_key >= 0:
            self.mujoco.mj_resetDataKeyframe(self.model, self.data, home_key)
        else:
            self.mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[self._joint_qadr] = FRANKA_HOME_QPOS
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_ctrl
        adr = int(self._task_qpos_adr)
        self.data.qpos[adr:adr + 7] = self.model.qpos0[adr:adr + 7]
        self.mujoco.mj_forward(self.model, self.data)

    def set_target_xyz(self, xyz, data=None) -> None:
        target = np.asarray(xyz, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(target)):
            raise ValueError("target xyz must be finite")
        d = self.data if data is None else data
        qadr = int(self._task_qpos_adr)
        d.qpos[qadr:qadr + 3] = target
        d.qpos[qadr + 3:qadr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0])
        dadr = int(self._task_dof_adr)
        d.qvel[dadr:dadr + 6] = 0.0
        self._target_xyz = target.copy()
        self.mujoco.mj_forward(self.model, d)

    def has_table_contact(self, data=None) -> bool:
        # Contact is not part of the free-space benchmark objective.
        return False

    def tip_xyz(self, data=None) -> np.ndarray:
        d = self.data if data is None else data
        return np.mean(np.asarray(d.geom_xpos[list(self._tip_geom_ids)], dtype=np.float64), axis=0).copy()

    def cartesian_jacobian(self, data=None) -> np.ndarray:
        """Linear fingertip Jacobian, with columns ordered by arm actuators."""
        d = self.data if data is None else data
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self.mujoco.mj_jacGeom(self.model, d, jacp, jacr, int(self._tip_geom_ids[0]))
        return jacp[:, self._joint_dadr].copy()

    def xy(self, data=None) -> np.ndarray:
        return self.tip_xyz(data)[:2]

    def task_xyz(self, data=None) -> np.ndarray:
        d = self.data if data is None else data
        qadr = int(self._task_qpos_adr)
        return np.asarray(d.qpos[qadr:qadr + 3], dtype=np.float64).copy()

    def task_xy(self, data=None) -> np.ndarray:
        return self.task_xyz(data)[:2]

    def task_velocity(self, data=None) -> np.ndarray:
        d = self.data if data is None else data
        dadr = int(self._task_dof_adr)
        return np.asarray(d.qvel[dadr:dadr + 3], dtype=np.float64).copy()

    def task_height(self, data=None) -> float:
        return float(self.task_xyz(data)[2])

    def task_up(self, data=None) -> float:
        return 1.0

    def tip_target_distance(self, data=None) -> float:
        return float(np.linalg.norm(self.tip_xyz(data) - self.task_xyz(data)))

    # Compatibility hooks used by the generic rollout cost.  This task is a
    # point-reach objective, so both names mean direct 3-D tip-to-target distance.
    def tip_box_surface_distance(self, data=None) -> float:
        return self.tip_target_distance(data)

    def tip_push_face_distance(self, _direction_xy, data=None) -> float:
        return self.tip_target_distance(data)

    def downward_alignment(self, data=None) -> float:
        d = self.data if data is None else data
        mat = np.asarray(d.xmat[self._tool_body_id], dtype=np.float64).reshape(3, 3)
        return float(-mat[2, 2])

    def root_up(self, data=None) -> float:
        return self.downward_alignment(data)

    def has_fallen(self, data=None, *, flipped_threshold: float = 0.0) -> bool:
        return False

    def snapshot(self, data=None) -> RobotSnapshot:
        d = self.data if data is None else data
        return RobotSnapshot(
            time=float(d.time),
            qpos=np.asarray(d.qpos, dtype=np.float64).copy(),
            qvel=np.asarray(d.qvel, dtype=np.float64).copy(),
            act=np.asarray(d.act, dtype=np.float64).copy(),
            ctrl=np.asarray(d.ctrl, dtype=np.float64).copy(),
        )

    def restore(self, snapshot: RobotSnapshot, data=None) -> None:
        d = self.data if data is None else data
        d.time = float(snapshot.time)
        d.qpos[:] = snapshot.qpos
        d.qvel[:] = snapshot.qvel
        if self.model.na:
            d.act[:] = snapshot.act
        d.ctrl[:] = snapshot.ctrl
        self.mujoco.mj_forward(self.model, d)

    def new_data(self, snapshot: RobotSnapshot | None = None):
        d = self.mujoco.MjData(self.model)
        self.restore(self.snapshot() if snapshot is None else snapshot, d)
        return d

    def nominal_ctrl(self, data=None) -> np.ndarray:
        d = self.data if data is None else data
        out = np.asarray(d.ctrl, dtype=np.float64).copy()
        if out.shape != (self.nu,):
            out = np.asarray(d.qpos[self._joint_qadr], dtype=np.float64).copy()
        return self.clip_ctrl(out)

    def control_bounds(self, unlimited_span: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        low = np.full(self.nu, -abs(float(unlimited_span)), dtype=np.float64)
        high = np.full(self.nu, abs(float(unlimited_span)), dtype=np.float64)
        limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
        ranges = np.asarray(self.model.actuator_ctrlrange, dtype=np.float64)
        low[limited] = ranges[limited, 0]
        high[limited] = ranges[limited, 1]
        return low, high

    def control_scale(
        self, fraction: float = 0.08, unlimited_span: float = 1.0, minimum: float = 1e-3
    ) -> np.ndarray:
        low, high = self.control_bounds(unlimited_span)
        return np.maximum(float(fraction) * (high - low), float(minimum))

    def clip_ctrl(self, controls: np.ndarray, unlimited_span: float = 1.0) -> np.ndarray:
        low, high = self.control_bounds(unlimited_span)
        return np.clip(np.asarray(controls, dtype=np.float64), low, high)

    def rate_limited_ctrl(self, controls: np.ndarray, *, control_dt: float, data=None) -> np.ndarray:
        dt = float(control_dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("control_dt must be finite and positive")
        d = self.data if data is None else data
        requested = self.clip_ctrl(controls)
        previous = np.asarray(d.ctrl, dtype=np.float64).copy()
        if previous.shape != (self.nu,):
            previous = np.asarray(d.qpos[self._joint_qadr], dtype=np.float64).copy()
        max_step = self.motion_limit_scale * self._joint_velocity_limits * dt
        return self.clip_ctrl(previous + np.clip(requested - previous, -max_step, max_step))

    def step_control(self, ctrl: np.ndarray, *, substeps: int = 1, data=None) -> None:
        d = self.data if data is None else data
        nstep = max(1, int(substeps))
        d.ctrl[:] = self.rate_limited_ctrl(ctrl, control_dt=nstep * self.physics_dt, data=d)
        self.mujoco.mj_step(self.model, d, nstep=nstep)


__all__ = [
    "FrankaReachRobot", "FRANKA_HOME_QPOS", "RobotSnapshot",
    "PANDA_JOINT_VELOCITY_LIMITS", "PANDA_JOINT_ACCELERATION_LIMITS",
]
