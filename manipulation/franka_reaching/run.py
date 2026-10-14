from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time
import numpy as np

from manipulation.mppi import ControllerConfig, JointMPPIController, SamplingOption
from manipulation.priors import GeometricPrior
from manipulation.robots import FrankaReachRobot
from manipulation.tracks import StraightTrack


# Five deterministic free-space targets.  One benchmark cycle traverses
# 1 -> 2 -> 3 -> 4 -> 5 -> 4 -> 3 -> 2 -> 1.
REACH_TARGETS = np.asarray(
    [
        [0.45, -0.25, 0.50],
        [0.50, -0.12, 0.62],
        [0.55,  0.00, 0.50],
        [0.50,  0.12, 0.62],
        [0.45,  0.25, 0.50],
    ],
    dtype=np.float64,
)


def target_sequence_indices(cycles: int) -> list[int]:
    cycles = int(cycles)
    if cycles < 1:
        raise ValueError("--cycles must be at least 1")
    sweep = [0, 1, 2, 3, 4, 3, 2, 1, 0]
    out = sweep.copy()
    for _ in range(1, cycles):
        out.extend(sweep[1:])
    return out


def build_controller(
    robot: FrankaReachRobot,
    track: StraightTrack,
    *,
    sampling: str = "spike",
    seed: int = 1,
    horizon: int = 20,
    num_rollouts: int = 32,
    control_dt: float = 0.01,
    joint_noise_fraction: float = 0.15,
    noise_space: str = 'effector',
    effector_noise_std: float = 0.01,
    rollout_workers: int = 8,
    target_distance_weight: float = 1.0,
    control_deviation_weight: float = 1e-4,
    downward_weight: float = 0.0,
    lbps_delta: float = 0.95,
    target_tolerance: float = 0.025,
    noise_min_fraction: float = 0.001,
    dslow: float = 0.5,
    noise_decay_order: float = 5.0,
    reach_terminal_weight: float = 2.0,
    reach_velocity_weight: float = 0.2,
    reach_region_weight: float = 0.0,
    reach_settle_weight: float = 0.0,
    reach_acceleration_weight: float = 0.01,
    reach_safe_radius: float = 0.015,
    reach_velocity_ref: float = 0.1,
    reach_velocity_sigma: float = 0.05,
    reach_settle_steps: int = 30,
) -> JointMPPIController:
    """MPPI objective for the free-space 3-D reaching benchmark."""
    cfg = ControllerConfig(
        control_dt=float(control_dt),
        horizon=int(horizon),
        num_rollouts=int(num_rollouts),
        joint_noise_fraction=float(joint_noise_fraction),
        noise_space=str(noise_space),
        effector_noise_std=float(effector_noise_std),
        noise_min_fraction=float(noise_min_fraction),
        dslow=float(dslow),
        noise_decay_order=float(noise_decay_order),
        reach_terminal_weight=float(reach_terminal_weight),
        reach_velocity_weight=float(reach_velocity_weight),
        reach_region_weight=float(reach_region_weight),
        reach_settle_weight=float(reach_settle_weight),
        reach_acceleration_weight=float(reach_acceleration_weight),
        reach_safe_radius=float(reach_safe_radius),
        reach_velocity_ref=float(reach_velocity_ref),
        reach_velocity_sigma=float(reach_velocity_sigma),
        reach_settle_steps=int(reach_settle_steps),
        rollout_workers=int(rollout_workers),
        warm_start=True,
        lbps_delta=float(lbps_delta),
        nominal_diagnostics=False,
        nominal_refine_iterations=0,
        screen_pool_size=0,
        screen_exploit=0,
        standard_sampler="auto",
        poisson_sampler="auto",
        spike_sampler="auto",
        upright_weight=float(downward_weight),
        box_progress_weight=0.0,
        robot_progress_weight=0.0,
        robot_box_approach_weight=float(target_distance_weight),
        box_max_lift=1e6,
        box_min_up=-1.0,
        box_lift_weight=0.0,
        control_deviation_weight=float(control_deviation_weight),
        command_derivative_weight=0.0,
        target_error_weight=0.0,
        terminal_target_weight=0.0,
        overshoot_weight=0.0,
        terminal_overshoot_weight=0.0,
        target_velocity_weight=0.0,
        target_tolerance=float(target_tolerance),
        non_tip_contact_weight=0.0,
        tip_contact_reward=0.0,
        tip_push_reward=0.0,
        table_contact_weight=0.0,
        box_speed_weight=0.0,
        preferred_box_speed=0.05,
        terminal_velocity_weight=0.0,
    )
    return JointMPPIController(
        robot, track, GeometricPrior(), cfg,
        sampling=SamplingOption(sampling), seed=int(seed),
    )


def _safe_viewer_sync(viewer) -> bool:
    if viewer is None:
        return False
    try:
        if not viewer.is_running():
            return False
        viewer.sync(state_only=True)
        return True
    except Exception:
        return False


def _close_viewer(viewer, *, settle_s: float = 0.20) -> None:
    if viewer is None:
        return
    try:
        viewer.close()
    except Exception:
        pass
    if settle_s > 0:
        time.sleep(float(settle_s))


def _save_recording(
    path: str | Path,
    *,
    robot: FrankaReachRobot,
    sampling: str,
    control_dt: float,
    target_sequence: np.ndarray,
    target_index: list[int],
    qpos: list[np.ndarray],
    qvel: list[np.ndarray],
    act: list[np.ndarray],
    state_time: list[float],
    controls: list[np.ndarray],
    additional: dict | None = None,
) -> Path:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **({} if additional is None else additional),
        replay_format_version=np.asarray(4, dtype=np.int64),
        robot_name=np.asarray(robot.name),
        sampling_option=np.asarray(str(sampling)),
        control_dt=np.asarray(float(control_dt), dtype=np.float64),
        physics_dt=np.asarray(float(robot.physics_dt), dtype=np.float64),
        target_xyz=np.asarray(target_sequence[0], dtype=np.float64),
        target_sequence_xyz=np.asarray(target_sequence, dtype=np.float64),
        active_target_index=np.asarray(target_index, dtype=np.int64),
        qpos=np.asarray(qpos, dtype=np.float64),
        qvel=np.asarray(qvel, dtype=np.float64),
        act=np.asarray(act, dtype=np.float64),
        state_time=np.asarray(state_time, dtype=np.float64),
        controls=np.asarray(controls, dtype=np.float64).reshape((-1, robot.nu)),
    )
    return path


def run(args: argparse.Namespace) -> dict[str, float | int | bool | str]:
    if not (0.0 < float(args.motion_limit_scale) <= 1.0):
        raise ValueError("--motion-limit-scale must be in (0, 1]")
    if not 0.0 < float(args.lbps_delta) < 1.0:
        raise ValueError("--lbps-delta must lie in (0, 1)")
    if float(args.goal_tolerance) <= 0.0:
        raise ValueError("--goal-tolerance must be positive")
    if float(args.hold_seconds) <= 0.0:
        raise ValueError("--hold-seconds must be positive")
    if int(args.cycles) < 1:
        raise ValueError("--cycles must be at least 1")
    for name in ("target_distance_weight", "control_deviation_weight", "downward_weight"):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")

    sequence_ids = target_sequence_indices(args.cycles)
    target_sequence = REACH_TARGETS[np.asarray(sequence_ids, dtype=np.int64)]
    active_sequence_index = 0
    robot = FrankaReachRobot(
        auto_fetch=not args.no_auto_fetch,
        motion_limit_scale=float(args.motion_limit_scale),
        acceleration_limit_scale=1.0,
        target_xyz=target_sequence[0],
    )

    # The generic controller still expects a broad 2-D progress adapter.  The
    # actual rollout objective is the direct 3-D fingertip-to-target distance.
    track = StraightTrack(
        origin_xy=tuple(robot.task_xy()), direction=(1.0, 0.0), road_width=100.0, length=10.0
    )
    current_s = 0.0
    controller = build_controller(
        robot,
        track,
        sampling=args.sampling,
        seed=args.seed,
        horizon=args.horizon,
        num_rollouts=args.rollouts,
        control_dt=args.control_dt,
        joint_noise_fraction=args.joint_noise,
        noise_space=args.noise_space,
        effector_noise_std=args.effector_noise_std,
        rollout_workers=args.rollout_workers,
        target_distance_weight=args.target_distance_weight,
        control_deviation_weight=args.control_deviation_weight,
        downward_weight=args.downward_weight,
        lbps_delta=args.lbps_delta,
        target_tolerance=args.goal_tolerance,
        noise_min_fraction=args.noise_min_fraction,
        dslow=args.dslow,
        noise_decay_order=args.noise_decay_order,
        reach_terminal_weight=args.reach_terminal_weight,
        reach_velocity_weight=args.reach_velocity_weight,
        reach_region_weight=args.reach_region_weight,
        reach_settle_weight=args.reach_settle_weight,
        reach_acceleration_weight=args.reach_acceleration_weight,
        reach_safe_radius=args.reach_safe_radius,
        reach_velocity_ref=args.reach_velocity_ref,
        reach_velocity_sigma=args.reach_velocity_sigma,
        reach_settle_steps=args.reach_settle_steps,
    )

    backend = controller.rollout_backend_name
    if not backend.startswith("cpu/fused") and not args.allow_python_rollout:
        controller.close()
        raise RuntimeError(
            f"reach task expected the fused backend, got {backend!r}. Rebuild with: "
            "python manipulation/setup_native.py build_ext --inplace. "
            "Use --allow-python-rollout only for debugging."
        )

    initial_distance = robot.tip_target_distance()
    if args.verbose:
        print(
            "tool: official Panda arm + short central single-finger end-effector  "
            f"nu={robot.nu}  finger_length~=0.086 m"
        )
        print("task: free-space 5-target sweep  sequence=1,2,3,4,5,4,3,2,1")
        for i, xyz in enumerate(REACH_TARGETS, start=1):
            print(f"target {i}: ({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) m")
        print(
            f"motion limits: velocity={robot.motion_limit_scale:.2f}x Panda max (command-space)  "
            f"control_dt={float(args.control_dt):.3f}s"
        )
        print(f"rollout backend: {backend}")
        print(
            "reward: normalized running 3-D tip-to-target distance + optional downward-facing + "
            "command regularization  "
            f"distance={float(args.target_distance_weight):.2f}  "
            f"downward={float(args.downward_weight):.2f}  "
            f"control={float(args.control_deviation_weight):.1e}"
        )
        print(f"LBPS: adaptive temperature on  delta={float(args.lbps_delta):.6f}")
        print(
            f"validation: distance <= {float(args.goal_tolerance):.3f} m continuously for "
            f"{float(args.hold_seconds):.2f}s; cycles={int(args.cycles)}"
        )
        print(f"initial target=1  initial_tip_target={initial_distance:.3f} m")

    initial = robot.snapshot()
    qpos_hist = [initial.qpos.copy()]
    qvel_hist = [initial.qvel.copy()]
    act_hist = [initial.act.copy()]
    state_time_hist = [float(initial.time)]
    controls_hist: list[np.ndarray] = []
    target_index_hist = [int(sequence_ids[active_sequence_index])]
    capture_analysis = bool(getattr(args, "analysis_trace", False))
    if capture_analysis:
        tip_xyz_hist = [robot.tip_xyz()]
        tip_vel_hist = [robot.cartesian_jacobian() @ np.asarray(robot.data.qvel)[robot._joint_dadr]]
        target_xyz_hist = [robot.task_xyz()]
        diagnostics: dict[str, list[float]] = {name: [] for name in (
            "best_cost", "weighted_rollout_cost", "nominal_cost", "ess",
            "temperature", "lbps_score", "finite_rollouts", "collision_rollouts",
            "cost_spread", "spike_events_mean", "poisson_events_mean",
            "applied_residual_l2", "plan_update_l2", "applied_saturation_fraction",
            "timing_total_ms", "timing_sampling_ms", "timing_rollouts_ms",
            "requested_step_l2", "applied_step_l2", "target_distance_m",
            "dwell_seconds", "noise_scale", "validated_targets",
        )}
        validation_steps: list[int] = []
        validation_times: list[float] = []

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(
            robot.model, robot.data,
            show_left_ui=bool(args.viewer_ui),
            show_right_ui=bool(args.viewer_ui),
        )
        viewer.cam.lookat[:] = np.asarray([0.48, 0.00, 0.53])
        viewer.cam.distance = 1.30
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -22.0
        _safe_viewer_sync(viewer)

    completed = False
    dwell_s = 0.0
    validated_targets = 0
    min_distance = float(initial_distance)
    min_distance_current = float(initial_distance)
    last_info: dict = {}
    step = -1
    start_wall = time.perf_counter()
    previous_sim_time = float(initial.time)
    try:
        for step in range(int(args.steps)):
            prev_ctrl = np.asarray(robot.data.ctrl, dtype=np.float64).copy()
            action, info = controller.step(robot.data, current_s)
            requested_step = float(np.linalg.norm(np.asarray(action) - prev_ctrl))
            robot.step_control(action, substeps=controller.control_substeps)
            after = robot.snapshot()
            applied_step = float(np.linalg.norm(after.ctrl - prev_ctrl))
            last_info = info

            sim_dt = max(0.0, float(after.time) - previous_sim_time)
            previous_sim_time = float(after.time)

            qpos_hist.append(after.qpos.copy())
            qvel_hist.append(after.qvel.copy())
            act_hist.append(after.act.copy())
            state_time_hist.append(float(after.time))
            controls_hist.append(after.ctrl.copy())
            target_index_hist.append(int(sequence_ids[active_sequence_index]))
            if capture_analysis:
                tip_xyz_hist.append(robot.tip_xyz())
                tip_vel_hist.append(robot.cartesian_jacobian() @ np.asarray(robot.data.qvel)[robot._joint_dadr])
                target_xyz_hist.append(robot.task_xyz())

            if viewer is not None and not _safe_viewer_sync(viewer):
                _close_viewer(viewer)
                viewer = None
                break

            tip_dist = robot.tip_target_distance()
            min_distance = min(min_distance, tip_dist)
            min_distance_current = min(min_distance_current, tip_dist)

            if tip_dist <= float(args.goal_tolerance):
                dwell_s += sim_dt
            else:
                dwell_s = 0.0

            if capture_analysis:
                diagnostic_values = {
                    "best_cost": info.get("best_cost"),
                    "weighted_rollout_cost": info.get("weighted_rollout_cost"),
                    "nominal_cost": info.get("nominal_cost"),
                    "ess": info.get("ess"),
                    "temperature": info.get("temperature"),
                    "lbps_score": info.get("lbps_score"),
                    "finite_rollouts": info.get("finite_rollouts"),
                    "collision_rollouts": info.get("collision_rollouts"),
                    "cost_spread": info.get("cost_spread"),
                    "spike_events_mean": info.get("spike_events_mean"),
                    "poisson_events_mean": info.get("poisson_events_mean"),
                    "applied_residual_l2": info.get("applied_residual_l2"),
                    "plan_update_l2": info.get("plan_update_l2"),
                    "applied_saturation_fraction": info.get("applied_saturation_fraction"),
                    "timing_total_ms": info.get("timing_ms", {}).get("total"),
                    "timing_sampling_ms": info.get("timing_ms", {}).get("sampling"),
                    "timing_rollouts_ms": info.get("timing_ms", {}).get("rollouts"),
                    "requested_step_l2": requested_step,
                    "applied_step_l2": applied_step,
                    "target_distance_m": tip_dist,
                    "dwell_seconds": dwell_s,
                    "noise_scale": float(args.noise_min_fraction) +
                        (1.0 - float(args.noise_min_fraction)) * min(
                            1.0, (max(0.0, tip_dist) / float(args.dslow)) ** float(args.noise_decay_order)),
                    "validated_targets": validated_targets,
                }
                for key in diagnostics:
                    value = diagnostic_values.get(key)
                    try:
                        diagnostics[key].append(float(value) if value is not None else float("nan"))
                    except (TypeError, ValueError):
                        diagnostics[key].append(float("nan"))

            if args.verbose and (step == 0 or (step + 1) % 10 == 0):
                event_text = ""
                if args.sampling == "spike":
                    event_text = f"  spikes={float(info.get('spike_events_mean', 0.0)):.1f}"
                target_number = int(sequence_ids[active_sequence_index]) + 1
                print(
                    f"step={step+1:04d}  target={target_number}  "
                    f"tip_target_dist={tip_dist:.3f} m  dwell={dwell_s:.2f}/{float(args.hold_seconds):.2f}s  "
                    f"down_align={robot.downward_alignment():+.2f}  "
                    f"best_cost={float(info['best_cost']):.4f}  ess={float(info['ess']):.1f}  "
                    f"temp={float(info.get('temperature', float('nan'))):.3g}  "
                    f"req_step={requested_step:.3f} rad  cmd_step={applied_step:.3f} rad"
                    f"{event_text}"
                )

            if dwell_s + 1e-12 >= float(args.hold_seconds):
                target_number = int(sequence_ids[active_sequence_index]) + 1
                validated_targets += 1
                if capture_analysis:
                    validation_steps.append(step + 1)
                    validation_times.append(float(after.time))
                    diagnostics["validated_targets"][-1] = float(validated_targets)
                if args.verbose:
                    print(
                        f"target {target_number} validated after {float(args.hold_seconds):.2f}s continuous hold  "
                        f"min_error={min_distance_current:.4f} m  "
                        f"progress={validated_targets}/{len(sequence_ids)}"
                    )
                if active_sequence_index + 1 >= len(sequence_ids):
                    completed = True
                    break

                active_sequence_index += 1
                next_target_id = int(sequence_ids[active_sequence_index])
                robot.set_target_xyz(REACH_TARGETS[next_target_id])
                controller.reset_warm_start()
                dwell_s = 0.0
                min_distance_current = robot.tip_target_distance()
                if args.verbose:
                    xyz = REACH_TARGETS[next_target_id]
                    print(
                        f"retarget -> {next_target_id + 1}  "
                        f"({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) m"
                    )
    finally:
        _close_viewer(viewer)
        viewer = None
        controller.close()
        gc.collect()

    elapsed = time.perf_counter() - start_wall
    final_distance = robot.tip_target_distance()
    saved_path = ""
    if not args.no_save and args.save:
        extra = None
        if capture_analysis:
            # State arrays contain N+1 samples; diagnostics and commands contain N.
            extra = {key: np.asarray(values, dtype=np.float64) for key, values in diagnostics.items()}
            extra.update({
                "analysis_format_version": np.asarray(1, dtype=np.int64),
                "seed": np.asarray(int(args.seed), dtype=np.int64),
                "noise_space": np.asarray(str(args.noise_space)),
                "tip_xyz": np.asarray(tip_xyz_hist, dtype=np.float64),
                "tip_velocity_xyz": np.asarray(tip_vel_hist, dtype=np.float64),
                "target_xyz_history": np.asarray(target_xyz_hist, dtype=np.float64),
                "joint_position": np.asarray(qpos_hist, dtype=np.float64)[:, robot._joint_qadr],
                "joint_velocity": np.asarray(qvel_hist, dtype=np.float64)[:, robot._joint_dadr],
                "target_validation_steps": np.asarray(validation_steps, dtype=np.int64),
                "target_validation_times": np.asarray(validation_times, dtype=np.float64),
                "success": np.asarray(bool(completed)),
                "goal_tolerance_m": np.asarray(float(args.goal_tolerance)),
                "hold_seconds": np.asarray(float(args.hold_seconds)),
                "dslow": np.asarray(float(args.dslow)),
                "noise_decay_order": np.asarray(float(args.noise_decay_order)),
                "noise_min_fraction": np.asarray(float(args.noise_min_fraction)),
            })
        saved = _save_recording(
            args.save,
            robot=robot,
            sampling=args.sampling,
            control_dt=args.control_dt,
            target_sequence=target_sequence,
            target_index=target_index_hist,
            qpos=qpos_hist,
            qvel=qvel_hist,
            act=act_hist,
            state_time=state_time_hist,
            controls=controls_hist,
            additional=extra,
        )
        saved_path = str(saved)
        if not bool(getattr(args, "batch_mode", False)):
            print(f"saved replay: {saved}")
            print(f"replay with: python -m manipulation.experiments.replay --file {saved}")
            print(
                "GIF with:    python -m manipulation.experiments.replay "
                f"--file {saved} --gif manipulation/results/last_run.gif"
            )

    return {
        "seed": int(args.seed),
        "mean_weighted_cost": float(np.nanmean(diagnostics["weighted_rollout_cost"])) if capture_analysis and diagnostics["weighted_rollout_cost"] and np.isfinite(diagnostics["weighted_rollout_cost"]).any() else float("nan"),
        "mean_compute_ms": float(np.nanmean(diagnostics["timing_total_ms"])) if capture_analysis and diagnostics["timing_total_ms"] else float("nan"),
        "mean_control_increment_l2": float(np.mean(np.linalg.norm(np.diff(np.asarray(controls_hist), axis=0), axis=1))) if len(controls_hist) > 1 else float("nan"),
        "mean_tip_speed_mps": float(np.mean(np.linalg.norm(np.asarray(tip_vel_hist), axis=1))) if capture_analysis else float("nan"),
        "completed": bool(completed),
        "steps": step + 1,
        "targets_validated": int(validated_targets),
        "targets_requested": int(len(sequence_ids)),
        "cycles_requested": int(args.cycles),
        "active_target": int(sequence_ids[active_sequence_index]) + 1,
        "final_error_m": float(final_distance),
        "min_error_m": float(min_distance),
        "wall_time_s": float(elapsed),
        "sampling": str(args.sampling),
        "noise_space": str(args.noise_space),
        "effector_noise_std_m": float(args.effector_noise_std),
        "rollout_backend": str(backend),
        "recording": saved_path,
        "best_cost": float(last_info.get("best_cost", np.nan)),
        "lbps_delta": float(args.lbps_delta),
        "goal_tolerance_m": float(args.goal_tolerance),
        "hold_seconds": float(args.hold_seconds),
    }


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Franka MPPI five-target free-space reaching benchmark")
    p.add_argument(
        "--sampling",
        choices=["standard", "poisson", "spike", "gaussian-poisson", "gaussian-spike"],
        default="spike",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--rollouts", type=int, default=32)
    p.add_argument("--control-dt", type=float, default=0.01)
    p.add_argument("--joint-noise", type=float, default=0.15)
    p.add_argument("--noise-space", choices=["joint", "effector"], default="effector",
                   help="Interpret MPPI proposals in joint or Cartesian XYZ tip space")
    p.add_argument("--effector-noise-std", type=float, default=0.01,
                   help="Cartesian proposal standard-deviation scale in metres")
    p.add_argument("--noise-min-fraction", type=float, default=0.1)
    p.add_argument("--dslow", type=float, default=0.50,
                   help="Distance in metres at which exploration reaches its full scale (default 0.50)")
    p.add_argument("--noise-decay-order", type=float, default=1.0,
                   help="Positive exponent n in min(1, (distance / dslow)**n); 1 is linear")
    p.add_argument("--reach-terminal-weight", type=float, default=2.0)
    p.add_argument("--reach-velocity-weight", type=float, default=0.2)
    p.add_argument("--reach-region-weight", type=float, default=0.0)
    p.add_argument("--reach-settle-weight", type=float, default=0.0)
    p.add_argument("--reach-acceleration-weight", type=float, default=0.01)
    p.add_argument("--reach-safe-radius", type=float, default=0.015)
    p.add_argument("--reach-velocity-ref", type=float, default=0.1)
    p.add_argument("--reach-velocity-sigma", type=float, default=0.05)
    p.add_argument("--reach-settle-steps", type=int, default=30)
    p.add_argument("--rollout-workers", type=int, default=8)
    p.add_argument("--motion-limit-scale", type=float, default=1.0)
    p.add_argument("--target-distance-weight", type=float, default=1.0)
    p.add_argument("--control-deviation-weight", type=float, default=1e-4)
    p.add_argument(
        "--downward-weight", type=float, default=0.0,
        help="optional tool-down orientation cost; default 0 for the sampling benchmark",
    )
    p.add_argument(
        "--lbps-delta", type=float, default=0.95,
        help="LBPS confidence/risk parameter delta; must lie in (0, 1)",
    )
    p.add_argument("--goal-tolerance", type=float, default=0.025)
    p.add_argument(
        "--hold-seconds", type=float, default=1.0,
        help="continuous time the tip must remain inside the tolerance sphere before a target is validated",
    )
    p.add_argument(
        "--cycles", type=int, default=1,
        help="number of complete 1-2-3-4-5-4-3-2-1 sweeps",
    )
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--viewer", dest="viewer", action="store_true", default=True)
    p.add_argument("--no-viewer", dest="viewer", action="store_false")
    p.add_argument("--viewer-ui", action="store_true", help="show MuJoCo side panels")
    p.add_argument("--allow-python-rollout", action="store_true")
    p.add_argument("--no-auto-fetch", action="store_true")
    p.add_argument("--analysis-trace", action="store_true", help="record joint, tip, and MPC diagnostics alongside replay states")
    p.add_argument("--save", default="manipulation/results/last_run.npz")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = make_parser().parse_args()
    result = run(args)
    print("franka_reach " + " ".join(f"{k}={v}" for k, v in result.items()))


if __name__ == "__main__":
    main()
