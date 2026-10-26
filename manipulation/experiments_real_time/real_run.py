"""FR3 100 Hz original fused-position MPPI to joint-position P control.

The MPPI reaching point is the MuJoCo slim_gripper BODY ORIGIN (link8
proxy), rather than the model-only slim_tip fingertip. Real reaching success is
estimated by MuJoCo FK from UDP joint feedback only. Model and robot may
have kinematic discrepancies; this is not an FR3 link8 calibration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import time
import traceback
import numpy as np

from manipulation.franka_reaching.run import (REACH_TARGETS, _save_recording, build_controller,
                  make_parser as make_sim_parser, target_sequence_indices)
from manipulation.experiments_real_time.pose_transport import PoseUdpRobot
from manipulation.robots import FrankaReachRobot
from manipulation.tracks import StraightTrack

DIAGNOSTIC_NAMES = (
    'best_cost', 'weighted_rollout_cost', 'nominal_cost', 'ess',
    'temperature', 'lbps_score', 'finite_rollouts', 'collision_rollouts',
    'cost_spread', 'spike_events_mean', 'poisson_events_mean',
    'applied_residual_l2', 'plan_update_l2', 'applied_saturation_fraction',
    'timing_total_ms', 'timing_sampling_ms', 'timing_rollouts_ms',
    'requested_step_l2', 'applied_step_l2', 'target_distance_m',
    'dwell_seconds', 'noise_scale', 'validated_targets',
)


def make_parser() -> argparse.ArgumentParser:
    p = make_sim_parser()  # preserves all options for sampling comparisons
    p.description = 'FR3 fused-position MPPI at 100 Hz (UDP joint-position commands)'
    p.set_defaults(control_dt=0.01, viewer=False, analysis_trace=True,
                   save='results/reaching_real/run_seed_1.npz',
                   steps=30000, motion_limit_scale=0.25, rollouts=32,
                   actuator_lag_kp=4.0)  # PC1: 1 kHz qdot = 4 (q_des - q)
    p.add_argument('--pc1-ip', default='10.20.0.1')
    p.add_argument('--pc2-ip', default='10.20.0.17')
    p.add_argument('--execute', action='store_true',
                   help='ENABLE REAL ROBOT MOTION (without this, commands are not sent)')
    p.add_argument('--joint-limit-margin', type=float, default=0.07,
                   help='minimum distance from each joint limit in rad')
    p.add_argument('--feedback-timeout', type=float, default=0.04,
                   help='reject joint feedback older than this many seconds')
    p.add_argument('--max-compute-ms', type=float, default=25.0,
                   help='MPPI computation budget (ms); 100 Hz requires completion in about 10 ms')
    p.add_argument('--max-tick-lateness-ms', type=float, default=8.0,
                   help='abort when high-level schedule lateness exceeds this value')
    p.add_argument('--command-deadline-ms', type=float, default=30.0,
                   help='abort if a joint command cannot be sent within this many ms of the '
                        'feedback sample it was computed from; slower ticks only delay the schedule')
    p.add_argument('--warmup-steps', type=int, default=2,
                   help='preflight MPPI rollouts, to trigger JIT before realtime operation')
    p.add_argument('--home-hold', type=float, default=0.5,
                   help='time to remain within home tolerance, seconds')
    p.add_argument('--home-timeout', type=float, default=60.0,
                   help='abort homing after this many seconds')
    p.add_argument('--home-joint-tolerance', type=float, default=0.04,
                   help='maximum absolute joint-position error at home (rad)')
    p.add_argument('--home-settle-joint-speed', type=float, default=0.08,
                   help='maximum measured joint speed at home in rad/s')
    p.add_argument('--home-min-z', type=float, default=0.10,
                   help='minimum permitted commanded link8 Z in base frame (metres)')
    p.add_argument('--viewer-every', type=int, default=10,
                   help='refresh optional MuJoCo measured-state viewer every N control ticks; visualization can affect control timing')
    p.add_argument('--print-every', type=int, default=50)
    p.add_argument('--model-ctrl', choices=['last-command', 'measured'], default='last-command',
                   help='MPPI model ctrl at each tick: the last joint target sent to PC1 '
                        '(matches the real P-controller setpoint) or the measured joint angles')
    return p


def _validate(args) -> None:
    if not any(abs(args.control_dt - dt) < 1e-10 for dt in (0.01, 0.02)):
        raise ValueError('Supported hardware control periods: 0.01 s (100 Hz) or 0.02 s (50 Hz)')
    if not np.isfinite(args.home_joint_tolerance) or args.home_joint_tolerance <= 0:
        raise ValueError('--home-joint-tolerance must be positive')
    if args.max_compute_ms is None:
        args.max_compute_ms = 1000.0 * args.control_dt
    if args.no_save and args.execute:
        raise ValueError('Hardware execution requires saving an NPZ recording')
    if args.viewer_every < 1:
        raise ValueError('--viewer-every must be >=1')
    if args.steps < 1 or args.cycles < 1 or args.warmup_steps < 0:
        raise ValueError('steps and cycles must be >=1 and warmup >=0')
    if args.rollouts < 1 or args.horizon < 1:
        raise ValueError('rollouts/horizon must be positive')
    for name in ('joint_limit_margin', 'feedback_timeout',
                 'max_compute_ms', 'max_tick_lateness_ms', 'command_deadline_ms', 'goal_tolerance',
                 'hold_seconds', 'dslow',
                 'home_hold', 'home_timeout',
                 'home_settle_joint_speed'):
        val = float(getattr(args, name))
        if not np.isfinite(val) or val <= 0.0:
            raise ValueError(f'--{name.replace("_", "-")} must be finite and >0')
    if not 0 < args.motion_limit_scale <= 1 or not 0 < args.lbps_delta < 1:
        raise ValueError('Invalid motion-limit-scale or lbps-delta')




def update_measured_model(robot, sample, *, elapsed, joint_low, joint_high, last_command=None):
    q, dq = sample.q, sample.dq
    if np.any(q < joint_low) or np.any(q > joint_high):
        raise RuntimeError('measured joint is inside safety margin of a position limit')
    if np.any(~np.isfinite(dq)):
        raise RuntimeError('non-finite measured joint velocity')
    robot.data.qpos[robot._joint_qadr] = q
    robot.data.qvel[robot._joint_dadr] = dq
    if np.max(np.abs(robot.clip_ctrl(q) - q)) > 1e-6:
        raise RuntimeError('measured joints outside MPPI actuator position limits')
    # The MPPI model is joint-position controlled. Its ctrl is the setpoint the
    # real P controller is currently tracking (the last command sent), so the
    # model starts with the same kp*(ctrl - q) drive and the rate limiter
    # continues from that command. Without a sent command, use the measured q.
    setpoint = q if last_command is None else np.asarray(last_command, dtype=float)
    if not np.isfinite(setpoint).all():
        raise RuntimeError('non-finite last joint command')
    robot.data.ctrl[:] = robot.clip_ctrl(setpoint)
    robot.data.time = float(elapsed)
    robot.mujoco.mj_forward(robot.model, robot.data)




def joint_feedback_with_model_fk(udp, robot, timeout, *, elapsed, joint_low, joint_high,
                                 last_command=None):
    """Return fresh joint feedback and an FK-derived link8-proxy position.

    This is NOT an independent Cartesian measurement; any model/base frame
    calibration error appears in the estimated pose and reaching error.
    """
    joint = udp.fresh(timeout)
    update_measured_model(robot, joint, elapsed=elapsed,
                          joint_low=joint_low, joint_high=joint_high,
                          last_command=last_command)
    xyz = np.asarray(robot.tip_xyz(), dtype=float).copy()
    if not np.isfinite(xyz).all():
        raise RuntimeError('nonfinite link8 model FK from joint feedback')
    return joint, xyz

def _build_real_controller(robot, args):
    track = StraightTrack(origin_xy=tuple(robot.task_xy()), direction=(1., 0.),
                          road_width=100., length=10.)
    return build_controller(
        robot, track, sampling=args.sampling, seed=args.seed,
        horizon=args.horizon, num_rollouts=args.rollouts,
        control_dt=args.control_dt, twitch_rise_s=args.twitch_rise_s,
        twitch_decay_s=args.twitch_decay_s, twitch_duration_s=args.twitch_duration_s,
        joint_noise_fraction=args.joint_noise, noise_space=args.noise_space,
        effector_noise_std=args.effector_noise_std,
        rollout_workers=args.rollout_workers,
        target_distance_weight=args.target_distance_weight,
        control_deviation_weight=args.control_deviation_weight,
        downward_weight=args.downward_weight, lbps_delta=args.lbps_delta,
        target_tolerance=args.goal_tolerance,
        noise_min_fraction=args.noise_min_fraction, dslow=args.dslow,
        noise_decay_order=args.noise_decay_order,
        reach_terminal_weight=args.reach_terminal_weight,
        reach_velocity_weight=args.reach_velocity_weight,
        reach_region_weight=args.reach_region_weight,
        reach_settle_weight=args.reach_settle_weight,
        reach_acceleration_weight=args.reach_acceleration_weight,
        reach_safe_radius=args.reach_safe_radius,
        reach_velocity_ref=args.reach_velocity_ref,
        reach_velocity_sigma=args.reach_velocity_sigma,
        reach_settle_steps=args.reach_settle_steps)


def _one_step_model_cartesian_velocity(robot, q_target, body_id, dt):
    """Predict the actual XYZ motion from ONE actuator-held MuJoCo step.

    The MPC backend optimizes joint position SETPOINTS. We apply its first
    setpoint to a separate copied measured model state and advance by dt; the
    Cartesian displacement over that interval becomes desired linear twist.
    This is NOT a Jacobian setpoint difference / dt, which would overstate
    physical movement for position actuators. The live model is unchanged.
    """
    mj = robot.mujoco
    m = robot.model
    sim = mj.MjData(m)
    sim.qpos[:] = robot.data.qpos
    sim.qvel[:] = robot.data.qvel
    if m.na:
        sim.act[:] = robot.data.act
    sim.ctrl[:] = robot.data.ctrl
    sim.time = robot.data.time
    mj.mj_forward(m, sim)
    p0 = np.asarray(sim.xpos[body_id], dtype=float).copy()
    sim.ctrl[:] = robot.clip_ctrl(q_target)
    remaining = float(dt)
    model_dt = float(m.opt.timestep)
    if not np.isfinite(model_dt) or model_dt <= 0:
        raise RuntimeError('invalid MuJoCo model timestep')
    while remaining > 1.e-10:
        # Use the regular model integration step (this helper is not used by joint-position commands).
        mj.mj_step(m, sim)
        remaining -= model_dt
    p1 = np.asarray(sim.xpos[body_id], dtype=float).copy()
    velocity = (p1 - p0) / float(dt)
    if not np.isfinite(velocity).all():
        raise RuntimeError('nonfinite MPPI one-step Cartesian velocity')
    return velocity, p1 - p0


def _model_body_xyz(robot, q, body_id):
    """Evaluate the Panda wrist/body position for a joint vector, restoring state."""
    old_q = robot.data.qpos[robot._joint_qadr].copy()
    try:
        robot.data.qpos[robot._joint_qadr] = q
        robot.mujoco.mj_forward(robot.model, robot.data)
        return np.asarray(robot.data.xpos[body_id], dtype=float).copy()
    finally:
        robot.data.qpos[robot._joint_qadr] = old_q
        robot.mujoco.mj_forward(robot.model, robot.data)


def _model_tip_xyz_at_q(robot, q):
    """Evaluate the *same virtual tip* used in the MPPI reaching cost."""
    old_q = robot.data.qpos[robot._joint_qadr].copy()
    try:
        robot.data.qpos[robot._joint_qadr] = q
        robot.mujoco.mj_forward(robot.model, robot.data)
        return np.asarray(robot.tip_xyz(), dtype=float).copy()
    finally:
        robot.data.qpos[robot._joint_qadr] = old_q
        robot.mujoco.mj_forward(robot.model, robot.data)


# Measured FR3 home position (first 7 arm joints only).
# The last two supplied values (0.0, 0.0) belong to the gripper and are not
# part of this seven-joint velocity-control interface.
CARTESIAN_HOME_Z_RAISE_M = 0.0  # No Cartesian home offset: real homing uses the exact HOME_Q.

DEFAULT_HOME_Q = np.array([
    -0.156,
    -0.173,
    0.04,
    -2.43,
    0.01,
    2.28,
    0.74,
], dtype=float)


def run(args: argparse.Namespace) -> dict:
    _validate(args)
    ids = target_sequence_indices(args.cycles)
    robot = FrankaReachRobot(
        auto_fetch=not args.no_auto_fetch,
        motion_limit_scale=float(args.motion_limit_scale),
        target_xyz=REACH_TARGETS[0],
        actuator_lag_kp=args.actuator_lag_kp)
    # Reposition the geometry the original fused-position backend uses for its
    # end-effector objective. It now coincides with the slim_gripper BODY ORIGIN,
    # our Panda-model proxy for FR3 link8. No virtual fingertip is used for cost,
    # Jacobians, reach validation or action reconstruction. This modifies only
    # this in-memory, real-run model, not the original simulation asset.
    wrist_id = robot._body_id('slim_gripper')
    endpoint_geom_id = robot._geom_id('slim_tip')
    # The native position-rollout kernel scores this geom ID. Relocate its
    # CENTER to the slim_gripper body origin (model's link8 proxy); do not
    # change its ID or remove it from the model.
    robot.model.geom_pos[endpoint_geom_id, :] = 0.0
    # The old custom tool consists of THREE geoms, not just slim_tip. Hide
    # all of them in this real-robot-only model and exclude their contacts.
    # Keep the tiny reference geom so the native fused evaluator still works.
    for name in ('slim_bridge', 'slim_finger', 'slim_tip'):
        geom_id = robot._geom_id(name)
        robot.model.geom_rgba[geom_id, 3] = 0.0
        robot.model.geom_contype[geom_id] = 0
        robot.model.geom_conaffinity[geom_id] = 0
    # Remove the old visual site, otherwise a tip marker can remain visible.
    marker_id = robot.mujoco.mj_name2id(robot.model, robot.mujoco.mjtObj.mjOBJ_SITE, 'slim_tip_mid')
    if marker_id >= 0:
        robot.model.site_rgba[marker_id, 3] = 0.0
    robot.mujoco.mj_forward(robot.model, robot.data)
    tip_origin_error = float(np.linalg.norm(robot.tip_xyz() - robot.data.xpos[wrist_id]))
    if tip_origin_error > 1e-6:
        raise RuntimeError(f'Native MPPI endpoint does not coincide with model link8 proxy: {tip_origin_error:.8f} m')
    # The fused kernel and FrankaReachRobot both use the geom ID.
    # Capture the *simulation's fixed home pose*, before measured q overwrites it.
    original_home_q = robot.data.qpos[robot._joint_qadr].copy()
    qlimits = np.asarray(robot.model.jnt_range[robot._joint_ids], dtype=float)
    qlow = qlimits[:, 0] + args.joint_limit_margin
    qhigh = qlimits[:, 1] - args.joint_limit_margin
    if np.any(qlow >= qhigh):
        raise ValueError('joint safety margins invalid')
    home_q = DEFAULT_HOME_Q.copy()
    if not np.all(np.isfinite(home_q)) or np.any(home_q < qlow) or np.any(home_q > qhigh):
        raise ValueError('Configured fixed home violates model joint safety limits')
    robot.data.qpos[robot._joint_qadr] = original_home_q
    robot.mujoco.mj_forward(robot.model, robot.data)
    original_home_xyz = robot.tip_xyz().copy()
    robot.data.qpos[robot._joint_qadr] = home_q
    robot.mujoco.mj_forward(robot.model, robot.data)
    fixed_home_xyz = robot.tip_xyz().copy()
    # Shift the original absolute reaching targets by the model FK difference
    # between the original model home and the selected fixed joint-space home.
    # This preserves the entire target pattern relative to the new home.
    target_offset = fixed_home_xyz - original_home_xyz + np.array([0.0, 0.0, CARTESIAN_HOME_Z_RAISE_M])
    shifted_targets = REACH_TARGETS
    sequence = shifted_targets[np.asarray(ids, dtype=int)]
    robot.set_target_xyz(sequence[0])
    # Real targets are re-anchored to the measured Cartesian home after homing.
    # The simulated objective is then shifted to the live link8 feedback frame.
    pass
    pass
    pass
    pass
    pass
    pass
    pass

    controller = None
    started = time.monotonic()
    experiment_start = None  # only set after joint homing and preflight
    experiment_end = None
    completed = False
    stop_reason = 'steps_exhausted'
    stop_detail = ''
    target_ptr = 0
    validated = 0
    dwell = 0.0
    min_error = float('inf')
    previous_cmd = np.zeros(7, dtype=float)
    diagnostics = {name: [] for name in DIAGNOSTIC_NAMES}
    overrun_ticks = 0
    extra_diag = {name: [] for name in (
        'loop_ms', 'tick_lateness_ms', 'feedback_age_ms',
        'measured_joint_stamp_sec', 'measured_joint_stamp_nsec',
        'joint_feedback_sequence')}
    qpos_hist, qvel_hist, act_hist, state_time_hist = [], [], [], []
    tip_hist, tip_vel_hist, targets_hist, targets_idx = [], [], [], []
    controls_hist, velocity_hist, desired_hist = [], [], []
    link8_feedback_hist, link8_command_hist = [], []
    virtual_tip_estimate_hist, virtual_tip_model_hist = [], []
    tip_desired_delta_hist, tip_goal_progress_hist = [], []
    feedforward_hist, correction_hist, reference_hist, command_twist_hist = [], [], [], []  # legacy empty arrays
    validation_steps, validation_times = [], []
    backend = 'not_started'
    homing_completed = False
    homing_final_error = float('nan')
    steps = 0

    def append_state(joint, elapsed):
        # State arrays have length N+1; controls have N commands.
        snap = robot.snapshot()
        qpos_hist.append(snap.qpos.copy())
        qvel_hist.append(snap.qvel.copy())
        act_hist.append(snap.act.copy())
        state_time_hist.append(float(elapsed))
        tip_hist.append(robot.tip_xyz().copy())  # model FK from measured q
        tip_vel_hist.append(robot.cartesian_jacobian() @ joint.dq)
        targets_hist.append(robot.task_xyz())
        targets_idx.append(int(ids[target_ptr]))

    viewer = None
    try:
        if args.viewer:
            try:
                import mujoco.viewer
                viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
                pass
            except Exception as exc:
                raise RuntimeError(f'Could not open MuJoCo viewer: {exc}') from exc
        with PoseUdpRobot(pc1=args.pc1_ip, pc2=args.pc2_ip,
                          enabled=args.execute) as udp:
            try:
                initial_joint = udp.wait_ready(max_age=args.feedback_timeout)
                update_measured_model(robot, initial_joint, elapsed=0.,
                                      joint_low=qlow, joint_high=qhigh)
                initial_xyz = robot.tip_xyz().copy()
                # Use the ORIGINAL position-actuator MPPI fused backend.
                controller = _build_real_controller(robot, args)
                backend = 'cpu/fused-position'
                pass
                # PC1 accepts joint positions directly; IK and Cartesian PID are off.
                if args.verbose:
                    pass

                # Compile/JIT and allocate non-RT caches before sending commands.
                for _ in range(args.warmup_steps):
                    controller.step(robot.data, 0.)
                controller.reset_warm_start()
                initial_joint = udp.wait_ready(max_age=args.feedback_timeout)
                update_measured_model(robot, initial_joint, elapsed=0.,
                                      joint_low=qlow, joint_high=qhigh)
                initial_xyz = robot.tip_xyz().copy()
                # Joint-space homing uses the same seven-joint position interface
                # as the MPPI: no Cartesian conversions or velocity commands.
                distance_to_home = float(np.max(np.abs(home_q - initial_joint.q)))
                pass
                if args.execute:
                    pass
                    home_start = time.monotonic()
                    home_next = home_start
                    settled_since = None
                    while True:
                        now = time.monotonic()
                        if now < home_next:
                            time.sleep(home_next - now)
                        joint, measured_xyz = joint_feedback_with_model_fk(
                            udp, robot, args.feedback_timeout, elapsed=0.,
                            joint_low=qlow, joint_high=qhigh)
                        homing_final_error = float(np.max(np.abs(home_q - joint.q)))
                        if measured_xyz[2] < args.home_min_z:
                            raise RuntimeError('FK link8 below workspace minimum during homing')
                        if homing_final_error <= args.home_joint_tolerance and \
                                np.max(np.abs(joint.dq)) <= args.home_settle_joint_speed:
                            if settled_since is None:
                                settled_since = time.monotonic()
                            if time.monotonic() - settled_since >= args.home_hold:
                                homing_completed = True
                                break
                        else:
                            settled_since = None
                        if time.monotonic() - home_start > args.home_timeout:
                            raise RuntimeError(f'joint-position home timeout; max joint error={homing_final_error:.4f}rad')
                        udp.send_joint_positions(home_q)
                        home_next += args.control_dt
                        if home_next < time.monotonic() - args.control_dt:
                            home_next = time.monotonic()
                    udp.send_joint_positions(home_q)
                    pass
                else:
                    homing_final_error = distance_to_home
                    homing_completed = distance_to_home <= args.home_joint_tolerance
                    pass
                # Setpoint PC1 is tracking; only known when commands are actually sent.
                track_command = args.execute and args.model_ctrl == 'last-command'
                last_command = home_q.copy() if track_command else None
                initial_joint, initial_xyz = joint_feedback_with_model_fk(
                    udp, robot, args.feedback_timeout, elapsed=0.,
                    joint_low=qlow, joint_high=qhigh, last_command=last_command)
                # The real task is expressed entirely in model FK coordinates,
                # matching the native fused model's cost and Jacobian endpoint.
                real_targets = shifted_targets.copy()
                sequence = real_targets[np.asarray(ids, dtype=int)]
                pass
                robot.set_target_xyz(real_targets[ids[0]])
                controller.reset_warm_start()
                append_state(initial_joint, 0.)
                # MPPI actions are sent as seven joint-position targets directly.
                start = time.monotonic()
                experiment_start = start
                next_tick = start
                last_observed = start
                for step in range(args.steps):
                    # Pacing measured by a monotonic clock; NOT a hard real-time scheduler.
                    now = time.monotonic()
                    if now < next_tick:
                        time.sleep(next_tick - now)
                    tick_start = time.monotonic()
                    lateness_ms = max(0., (tick_start - next_tick) * 1e3)
                    if lateness_ms > args.max_tick_lateness_ms:
                        raise RuntimeError(f'{1.0/args.control_dt:.0f}Hz tick missed: lateness={lateness_ms:.2f} ms')
                    joint, link8_xyz = joint_feedback_with_model_fk(
                        udp, robot, args.feedback_timeout, elapsed=tick_start-start,
                        joint_low=qlow, joint_high=qhigh, last_command=last_command)
                    # The receiver runs independently. A sample can have the same
                    # sequence at adjacent ticks without indicating lost feedback;
                    # freshness is enforced by udp.fresh() using monotonic age.
                    elapsed = tick_start - start
                    if step > 0:
                        append_state(joint, elapsed)
                    # The cost and reaching metric both use model link8-proxy FK.
                    robot.set_target_xyz(real_targets[ids[target_ptr]])
                    previous_sim_ctrl = robot.data.ctrl.copy()
                    action, info = controller.step(robot.data, 0.)
                    mp_ms = (time.monotonic() - tick_start) * 1e3
                    if mp_ms > args.max_compute_ms:
                        raise RuntimeError(f'MPPI computation exceeded budget: {mp_ms:.2f} ms')
                    q_target = np.asarray(action, dtype=float).reshape(7)
                    # The first optimized action is ALREADY seven joint-position
                    # actuator setpoints. Transmit it unchanged to the PC1 P controller.
                    if not np.isfinite(q_target).all():
                        raise RuntimeError('nonfinite optimized joint-position command')
                    if np.any(q_target < qlow) or np.any(q_target > qhigh):
                        raise RuntimeError('optimized joint-position target exceeds safety margin')
                    predicted_xyz = _model_body_xyz(robot, q_target, wrist_id)
                    if not np.isfinite(predicted_xyz).all() or predicted_xyz[2] < args.home_min_z:
                        raise RuntimeError('optimized joint target below FK workspace Z minimum')
                    command_age_ms = (time.monotonic() - tick_start) * 1e3
                    if command_age_ms > args.command_deadline_ms:
                        raise RuntimeError(f'joint-position command deadline missed: '
                                           f'{command_age_ms:.2f} ms > {args.command_deadline_ms:.1f} ms')
                    udp.send_joint_positions(q_target)
                    if track_command:
                        last_command = q_target.copy()
                    desired_delta = predicted_xyz - link8_xyz
                    goal_vector = real_targets[ids[target_ptr]] - link8_xyz
                    delta_progress = float(np.dot(desired_delta, goal_vector) / max(np.linalg.norm(goal_vector), 1e-9))
                    xyz_cmd = predicted_xyz.copy()
                    v_cmd = np.zeros(7, dtype=float)  # PC1 owns the joint P law
                    tip_desired_delta_hist.append(desired_delta.copy())
                    tip_goal_progress_hist.append(delta_progress)
                    link8_feedback_hist.append(link8_xyz.copy())
                    link8_command_hist.append(xyz_cmd.copy())
                    # Compatibility with earlier NPZ fields; these are now
                    # link8-proxy positions, NOT virtual tool tip positions.
                    virtual_tip_model = robot.tip_xyz().copy()
                    wrist_current = _model_body_xyz(robot, joint.q, wrist_id)
                    virtual_tip_estimate = link8_xyz + (virtual_tip_model - wrist_current)
                    virtual_tip_estimate_hist.append(virtual_tip_estimate)
                    virtual_tip_model_hist.append(virtual_tip_model)
                    controls_hist.append(q_target.copy())
                    desired_hist.append(q_target.copy())
                    velocity_hist.append(v_cmd.copy())
                    steps += 1
                    cur_error = float(np.linalg.norm(link8_xyz - real_targets[ids[target_ptr]]))
                    min_error = min(min_error, cur_error)
                    if cur_error <= args.goal_tolerance:
                        dwell += max(0., tick_start - last_observed)
                    else:
                        dwell = 0.
                    last_observed = tick_start
                    diag = {
                        'timing_total_ms': info.get('timing_ms', {}).get('total'),
                        'timing_sampling_ms': info.get('timing_ms', {}).get('sampling'),
                        'timing_rollouts_ms': info.get('timing_ms', {}).get('rollouts'),
                        'requested_step_l2': np.linalg.norm(q_target - previous_sim_ctrl),
                        'applied_step_l2': np.linalg.norm(q_target - joint.q),
                        'target_distance_m': cur_error,
                        'dwell_seconds': dwell,
                        'noise_scale': args.noise_min_fraction + (1.-args.noise_min_fraction) *
                            min(1., (cur_error/args.dslow)**args.noise_decay_order),
                        'validated_targets': validated,
                    }
                    for name in DIAGNOSTIC_NAMES:
                        value = diag.get(name, info.get(name, np.nan))
                        try:
                            diagnostics[name].append(float(value))
                        except (TypeError, ValueError):
                            diagnostics[name].append(float('nan'))
                    for name, value in {
                        'loop_ms': (time.monotonic() - tick_start) * 1e3,
                        'tick_lateness_ms': lateness_ms,
                        'feedback_age_ms': (tick_start - joint.received_at) * 1e3,
                        'measured_joint_stamp_sec': (joint.stamp[0] if joint.stamp is not None else np.nan),
                        'measured_joint_stamp_nsec': (joint.stamp[1] if joint.stamp is not None else np.nan),
                        'joint_feedback_sequence': joint.sequence,
                    }.items():
                        extra_diag[name].append(float(value))

                    if (step + 1) % max(1,args.print_every) == 0 or step == 0:
                        pass

                    if dwell >= args.hold_seconds:
                        validated += 1
                        validation_steps.append(steps)
                        validation_times.append(float(elapsed))
                        diagnostics['validated_targets'][-1] = float(validated)
                        pass
                        if target_ptr + 1 >= len(ids):
                            completed = True
                            stop_reason = 'completed'
                        else:
                            target_ptr += 1
                            robot.set_target_xyz(real_targets[ids[target_ptr]])
                            controller.reset_warm_start()
                            dwell = 0.
                    next_tick += args.control_dt
                    if viewer is not None and (step % args.viewer_every == 0):
                        if not viewer.is_running():
                            raise RuntimeError('MuJoCo viewer closed')
                        viewer.sync()
                    # State data is measured at the NEXT loop iteration, so N+1
                    # state row is appended on exit separately.
                    if completed:
                        break
                    # A slow tick delays the schedule instead of aborting: PC1 keeps
                    # tracking the last setpoint, and the command deadline above
                    # still bounds how stale each sent command can be.
                    now = time.monotonic()
                    if now > next_tick + args.max_tick_lateness_ms/1000.:
                        next_tick = now
                        overrun_ticks += 1
            except KeyboardInterrupt:
                stop_reason = 'keyboard_interrupt'
                stop_detail = 'user requested stop'
            except Exception as exc:
                stop_reason = 'safety_abort'
                stop_detail = str(exc)
                pass
            finally:
                experiment_end = time.monotonic()
                # No zero joint-position packet exists. Reissue a fresh measured
                # position to request a hold, then stop transmitting so the PC1
                # controller's timeout behavior takes over.
                try:
                    hold_joint = udp.fresh(args.feedback_timeout)
                    udp.send_joint_positions(hold_joint.q)
                except Exception:
                    pass
                # Ensure state-aligned analysis arrays even on abort.
                # Command arrays length N; exactly N+1 measured/state arrays.
                if qpos_hist and steps and len(qpos_hist) == steps:
                    joint = udp.joint
                    if joint is not None:
                        try:
                            update_measured_model(robot, joint, elapsed=time.monotonic()-start,
                                                  joint_low=qlow, joint_high=qhigh)
                        except Exception:
                            pass
                    # Last available sample closes the N+1 state history.
                    append_state(joint if joint is not None else initial_joint,
                                 time.monotonic() - start)
    finally:
        if viewer is not None:
            viewer.close()
        if controller is not None:
            controller.close()

    if steps and len(qpos_hist) != steps + 1:
        raise RuntimeError(f'Internal recording length mismatch: states={len(qpos_hist)} controls={steps}')

    path = ''
    if not args.no_save and args.save and qpos_hist:
        # Key names match the original batch.py analyzer. A preflight failure
        # before the first joint sample cannot produce a meaningful NPZ.
        extra = {name: np.asarray(values, dtype=np.float64)
                 for name, values in {**diagnostics, **extra_diag}.items()}
        extra.update({
            'analysis_format_version': np.asarray(2, dtype=np.int64),
            'seed': np.asarray(int(args.seed), dtype=np.int64),
            'noise_space': np.asarray(args.noise_space),
            'tip_xyz': np.asarray(tip_hist, dtype=np.float64).reshape(-1,3),
            'tip_velocity_xyz': np.asarray(tip_vel_hist, dtype=np.float64).reshape(-1,3),
            'target_xyz_history': np.asarray(targets_hist, dtype=np.float64).reshape(-1,3),
            'joint_position': np.asarray(qpos_hist, dtype=float)[:, robot._joint_qadr],
            'joint_velocity': np.asarray(qvel_hist, dtype=float)[:, robot._joint_dadr],
            'target_validation_steps': np.asarray(validation_steps, dtype=np.int64),
            'target_validation_times': np.asarray(validation_times, dtype=np.float64),
            'success': np.asarray(bool(completed)),
            'goal_tolerance_m': np.asarray(float(args.goal_tolerance)),
            'hold_seconds': np.asarray(float(args.hold_seconds)),
            'dslow': np.asarray(float(args.dslow)),
            'noise_decay_order': np.asarray(float(args.noise_decay_order)),
            'noise_min_fraction': np.asarray(float(args.noise_min_fraction)),
            'joint_velocity_command': np.asarray(velocity_hist, dtype=float).reshape(-1,7),
            'desired_joint_position': np.asarray(desired_hist, dtype=float).reshape(-1,7),
            'hardware_mode': np.asarray('fr3_fused_position_joint_position_udp_to_p_controller'),
            'link8_position_feedback': np.asarray(link8_feedback_hist,dtype=float).reshape(-1,3),
            'link8_optimized_joint_target_fk': np.asarray(link8_command_hist,dtype=float).reshape(-1,3),
            'cartesian_velocity_command_xyz': np.asarray(command_twist_hist,dtype=float).reshape(-1,3),
            'cartesian_velocity_feedforward_xyz': np.asarray(feedforward_hist,dtype=float).reshape(-1,3),
            'cartesian_velocity_feedback_xyz': np.asarray(correction_hist,dtype=float).reshape(-1,3),
            'cartesian_trajectory_reference_xyz': np.asarray(reference_hist,dtype=float).reshape(-1,3),
            'joint_position_command_port': np.asarray(15007),
            'tip_position_source': np.asarray('mujoco_link8_proxy_fk_from_measured_joints'),
            'tip_action_delta_model': np.asarray(tip_desired_delta_hist, dtype=float).reshape(-1,3),
            'link8_action_toward_goal_m': np.asarray(tip_goal_progress_hist, dtype=float),
            'tip_action_toward_goal_m': np.asarray(tip_goal_progress_hist, dtype=float),  # legacy compatibility
            'cartesian_command_mapping': np.asarray('direct_optimized_joint_position_setpoint'),
            'model_link8_proxy_position': np.asarray(virtual_tip_model_hist,dtype=float).reshape(-1,3),
            'virtual_tip_position_model': np.asarray(virtual_tip_model_hist,dtype=float).reshape(-1,3),  # legacy key
            'virtual_tip_position_estimate_uncalibrated': np.asarray(virtual_tip_estimate_hist,dtype=float).reshape(-1,3),  # legacy key
            'virtual_tip_estimate_assumption': np.asarray('legacy key now contains link8 proxy; not a calibrated FR3 link8 FK'),
            'predicted_step_m': np.linalg.norm(
                np.asarray(link8_command_hist,dtype=float).reshape(-1,3) -
                np.asarray(link8_feedback_hist,dtype=float).reshape(-1,3),axis=1),
            'stop_reason': np.asarray(stop_reason),
            'stop_detail': np.asarray(stop_detail),
            'backend': np.asarray(backend),
            'pc1_ip': np.asarray(args.pc1_ip),
            'pc2_ip': np.asarray(args.pc2_ip),
            'commands_enabled': np.asarray(bool(args.execute)),
            'home_joint_target': home_q.copy(),
            'original_home_joint_target': original_home_q.copy(),
            'original_home_tip_xyz': original_home_xyz.copy(),
            'lowered_home_tip_xyz': fixed_home_xyz.copy(),
            'fixed_home_tip_xyz': fixed_home_xyz.copy(),
            'target_translation_xyz': target_offset.copy(),
            'real_link8_target_xyz': real_targets.copy() if 'real_targets' in locals() else np.empty((0,3)),
            'reaching_error_source': np.asarray('MuJoCo link8-proxy FK from UDP measured joints versus model targets'),
            'experiment_time_s': np.asarray(
                float(experiment_end - experiment_start)
                if experiment_start is not None and experiment_end is not None else np.nan),
            'experiment_time_definition': np.asarray('post_homing_to_stop_monotonic'),
            'homing_completed': np.asarray(homing_completed),
            'homing_final_max_joint_error_rad': np.asarray(homing_final_error),
        })
        # Save partial logs as well (including safety-aborted runs).
        path = str(_save_recording(
            args.save, robot=robot, sampling=args.sampling, control_dt=args.control_dt,
            target_sequence=sequence, target_index=targets_idx,
            qpos=qpos_hist, qvel=qvel_hist, act=act_hist,
            state_time=state_time_hist, controls=controls_hist, additional=extra))
    result = {
        'seed':int(args.seed), 'completed':bool(completed),
        'stop_reason':stop_reason, 'stop_detail':stop_detail,
        'steps':steps, 'targets_validated':validated,
        'targets_requested':len(ids), 'recording':path,
        'wall_time_s':time.monotonic()-started,  # includes homing; retained for compatibility
        'experiment_time_s':(float(experiment_end - experiment_start)
                             if experiment_start is not None and experiment_end is not None else float('nan')),
        'sampling':args.sampling, 'rollout_backend':backend,
        'max_loop_ms':float(max(extra_diag['loop_ms'], default=float('nan'))),
        'overrun_ticks':int(overrun_ticks),
        'max_compute_ms':float(max(diagnostics['timing_total_ms'], default=float('nan'))),
        'min_error_m':float(min_error),
        'homing_completed':homing_completed,
        'homing_final_max_joint_error_rad':homing_final_error,
    }
    if result['completed'] and np.isfinite(result['experiment_time_s']):
        print(f"Experiment time: {result['experiment_time_s']:.3f} s", flush=True)
    elif not result['completed']:
        print(f"STOP: {result['stop_reason']}: {result['stop_detail']}", flush=True)
    return result


def main():
    args = make_parser().parse_args()
    result = run(args)
    if result['stop_reason'] == 'safety_abort':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
