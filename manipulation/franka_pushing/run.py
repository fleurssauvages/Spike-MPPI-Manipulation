"""Franka MPPI: push a box between two or four tabletop waypoints."""
from __future__ import annotations

import argparse
import gc
from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np

from manipulation.mppi import ControllerConfig, JointMPPIController, SamplingOption
from manipulation.priors import GeometricPrior
from manipulation.robots import FrankaPushRobot
from manipulation.tracks import StraightTrack

from .waypoints import (
    DEFAULT_START_XY as START_XY, DEFAULT_GOAL_XY as GOAL_XY,
    WaypointRoute, resolve_waypoints, DEFAULT_DEPTH_X,
)


def build_controller(
    robot: FrankaPushRobot,
    track: StraightTrack,
    *,
    sampling: str = "spike",
    seed: int = 1,
    horizon: int = 20,
    num_rollouts: int = 32,
    control_dt: float = 0.01,
    twitch_rise_s: float = 0.016,
    twitch_decay_s: float = 0.064,
    twitch_duration_s: float = 0.200,
    joint_noise_fraction: float = 0.15,
    noise_space: str = "effector",
    effector_noise_std: float = 0.01,
    rollout_workers: int = 8,
    tip_box_weight: float = 0.1,
    box_goal_weight: float = 10.0,
    control_deviation_weight: float = 0.0,
    lbps_delta: float = 0.95,
) -> JointMPPIController:
    """Franka MPPI pushing with two distance terms active by default.

    Cost = mean_t [box_goal_weight * box_center_to_goal_distance(t)
                   + tip_box_weight * tip_to_nearest_surface_distance(t)].
    The optional control regularizer has default weight zero.
    Exploration covariance remains constant (adaptive_noise=False).
    """
    cfg = ControllerConfig(
        control_dt=float(control_dt),
        horizon=int(horizon),
        num_rollouts=int(num_rollouts),
        twitch_rise_s=float(twitch_rise_s),
        twitch_decay_s=float(twitch_decay_s),
        twitch_duration_s=float(twitch_duration_s),
        joint_noise_fraction=float(joint_noise_fraction),
        noise_space=str(noise_space),
        effector_noise_std=float(effector_noise_std),
        adaptive_noise=False,
        noise_min_fraction=1.0,
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
        # Only the box-to-goal and fingertip-to-box terms are active by default.
        robot_box_approach_weight=float(tip_box_weight),
        target_error_weight=float(box_goal_weight),
        control_deviation_weight=float(control_deviation_weight),
        # Every other objective weight is OFF.
        upright_weight=0.0,
        box_progress_weight=0.0,
        robot_progress_weight=0.0,
        box_lift_weight=0.0,
        terminal_target_weight=0.0,
        overshoot_weight=0.0,
        terminal_overshoot_weight=0.0,
        target_velocity_weight=0.0,
        non_tip_contact_weight=0.0,
        tip_contact_reward=0.0,
        tip_push_reward=0.0,
        table_contact_weight=0.0,
        box_speed_weight=0.0,
        terminal_velocity_weight=0.0,
        command_derivative_weight=0.0,
        reach_terminal_weight=0.0,
        reach_velocity_weight=0.0,
        reach_region_weight=0.0,
        reach_settle_weight=0.0,
        reach_acceleration_weight=0.0,
        # Not part of the cost; mode 4 does not reject tilted/lifted boxes.
        box_max_lift=1e6,
        box_min_up=-1.0,
    )
    return JointMPPIController(
        robot, track, GeometricPrior(), cfg,
        sampling=SamplingOption(sampling), seed=int(seed),
    )


def _retarget_goal(robot: FrankaPushRobot, controller: JointMPPIController, xy, viewer=None) -> None:
    """Switch the goal in the plant AND in all persistent MuJoCo planners.

    The fused C++ evaluator owns an independent model copied at creation, so
    changing the viewer's mocap marker does not change its rollout objective.
    Reload its model at each waypoint transition, and refresh the longer-lived
    Python mj_rollout MjData mocap buffers (FULLPHYSICS omits mocap positions).
    """
    lock = viewer.lock() if viewer is not None else nullcontext()
    with lock:
        robot.set_goal_xy(xy)
    seen: set[int] = set()
    for batcher in (controller.rollout_batcher, controller.nominal_batcher):
        if batcher is None or id(batcher) in seen:
            continue
        seen.add(id(batcher))
        buffers = batcher.data if isinstance(batcher.data, list) else [batcher.data]
        scratch = getattr(batcher, "_state_scratch", None)
        if scratch is not None:
            buffers = [*buffers, scratch]
        for data in buffers:
            robot.sync_goal_mocap(data)
    controller.sync_planning_model()
    controller.reset_warm_start()


def _viewer_sync(viewer) -> bool:
    if viewer is None:
        return False
    try:
        if not viewer.is_running():
            return False
        viewer.sync(state_only=True)
        return True
    except Exception:
        return False


def _close_viewer(viewer) -> None:
    if viewer is not None:
        try:
            viewer.close()
        except Exception:
            pass
        time.sleep(0.2)


def _mean_finite(values) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if len(finite) else float("nan")


def run(args: argparse.Namespace) -> dict[str, float | int | bool | str]:
    if not (0.0 < args.motion_limit_scale <= 1.0):
        raise ValueError("--motion-limit-scale must be in (0, 1]")
    if not 0.0 < args.lbps_delta < 1.0:
        raise ValueError("--lbps-delta must lie in (0, 1)")
    if args.steps < 1 or args.goal_tolerance <= 0 or args.hold_seconds <= 0:
        raise ValueError("steps, goal-tolerance and hold-seconds must be positive")
    if args.cycles < 0:
        raise ValueError("--cycles must be nonnegative (0 repeats until --steps)")
    for name in ("tip_box_weight", "box_goal_weight", "control_deviation_weight"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")

    points = resolve_waypoints(
        start_xy=args.start_xy, goal_xy=args.goal_xy,
        route=args.route, waypoints=args.waypoints, depth=args.triangle_depth,
    )
    route = WaypointRoute(mode=args.route, cycles=args.cycles)
    robot = FrankaPushRobot(
        start_xy=points[0],
        goal_xy=points[route.goal_index],
        auto_fetch=not args.no_auto_fetch,
        motion_limit_scale=args.motion_limit_scale,
    )
    if len(points) in (3, 4):
        robot.set_waypoints_xy(points)
    else:
        robot.hide_waypoints()
    initial_goal_xyz = robot.goal_xyz()
    # The straight track is only a compatibility adapter for MPPI's state
    # propagation and diagnostics. It contributes NO term to the pushing cost.
    delta = np.asarray(points[1], dtype=float) - np.asarray(points[0], dtype=float)
    direction = delta if float(np.linalg.norm(delta)) > 1e-9 else np.array([1.0, 0.0])
    track = StraightTrack(
        origin_xy=tuple(points[0]), direction=tuple(direction),
        road_width=100.0, length=10.0,
    )
    if args.physics_dt is not None:
        if not np.isfinite(args.physics_dt) or args.physics_dt <= 0:
            raise ValueError("physics-dt must be finite and positive")
        robot.model.opt.timestep = float(args.physics_dt)
    controller = build_controller(
        robot, track,
        sampling=args.sampling,
        seed=args.seed,
        horizon=args.horizon,
        num_rollouts=args.rollouts,
        control_dt=args.control_dt,
        twitch_rise_s=args.twitch_rise_s,
        twitch_decay_s=args.twitch_decay_s,
        twitch_duration_s=args.twitch_duration_s,
        joint_noise_fraction=args.joint_noise,
        noise_space=args.noise_space,
        effector_noise_std=args.effector_noise_std,
        rollout_workers=args.rollout_workers,
        tip_box_weight=args.tip_box_weight,
        box_goal_weight=args.box_goal_weight,
        control_deviation_weight=args.control_deviation_weight,
        lbps_delta=args.lbps_delta,
    )
    backend = controller.rollout_backend_name
    if not backend.startswith("cpu/fused") and not args.allow_python_rollout:
        controller.close()
        raise RuntimeError(
            f"pushing requires the new fused mode-4 evaluator (got {backend!r}). "
            "Rebuild it using `python manipulation/setup_native.py build_ext --inplace`, "
            "or use --allow-python-rollout for slower development/debugging."
        )

    initial = robot.snapshot()
    qpos_hist = [initial.qpos.copy()]
    qvel_hist = [initial.qvel.copy()]
    act_hist = [initial.act.copy()]
    time_hist = [float(initial.time)]
    controls_hist: list[np.ndarray] = []
    controller_times: list[float] = []
    # Per-state targets are essential for dynamic-goal replay, not just analysis.
    goal_hist = [robot.goal_xyz()]
    goal_index_hist = [route.goal_index]
    arrival_steps: list[int] = []
    arrival_times: list[float] = []
    arrival_waypoints: list[int] = []
    capture_analysis = bool(args.analysis_trace)
    if capture_analysis:
        tip_hist = [robot.tip_xyz()]
        box_hist = [robot.task_xyz()]
        box_vel_hist = [robot.task_velocity()]
        trace: dict[str, list[float]] = {k: [] for k in (
            "best_cost", "weighted_rollout_cost", "nominal_cost", "ess",
            "temperature", "lbps_score", "finite_rollouts", "collision_rollouts",
            "cost_spread", "spike_events_mean", "poisson_events_mean",
            "applied_residual_l2", "plan_update_l2", "applied_saturation_fraction",
            "timing_total_ms", "timing_sampling_ms", "timing_rollouts_ms",
            "tip_box_distance_m", "box_goal_distance_m", "dwell_seconds", "noise_scale",
        )}

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(
            robot.model, robot.data,
            show_left_ui=bool(args.viewer_ui),
            show_right_ui=bool(args.viewer_ui),
        )
        viewer.cam.lookat[:] = [0.55, 0.0, 0.44]
        viewer.cam.distance = 1.30
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -22.0
        _viewer_sync(viewer)

    initial_error = robot.box_goal_distance()
    closest_error = initial_error
    completed = False
    last_info: dict = {}
    step = -1
    t_wall = time.perf_counter()
    last_time = float(initial.time)
    try:
        if args.verbose:
            print(f"tabletop {args.route} push: {points.tolist()} "
                  f"(cycles={args.cycles if args.cycles else 'unlimited'})")
            objective = (f"{args.box_goal_weight:g} * box-goal distance + "
                         f"{args.tip_box_weight:g} * fingertip-to-box surface distance")
            if args.control_deviation_weight:
                objective += (f" + {args.control_deviation_weight:g} * "
                              "normalized control deviation")
            print("objective:", objective)
            print(f"sampling={args.sampling} noise-space={args.noise_space} "
                  f"covariance=fixed (no near-goal decay); backend={backend}")
        for step in range(int(args.steps)):
            action, info = controller.step(robot.data, 0.0)
            robot.step_control(action, substeps=controller.control_substeps)
            after = robot.snapshot()
            last_info = info
            controller_times.append(float(info.get("timing_ms", {}).get("total", np.nan)))
            dt = max(0.0, float(after.time) - last_time)
            last_time = float(after.time)
            qpos_hist.append(after.qpos.copy())
            qvel_hist.append(after.qvel.copy())
            act_hist.append(after.act.copy())
            time_hist.append(float(after.time))
            controls_hist.append(after.ctrl.copy())

            box_error = robot.box_goal_distance()  # Error to goal used for this step.
            tip_error = robot.tip_box_surface_distance()
            closest_error = min(closest_error, box_error)
            arrived = route.observe(
                distance=box_error, dt=dt,
                tolerance=args.goal_tolerance, hold_seconds=args.hold_seconds,
            )
            if arrived is not None:
                arrival_steps.append(step + 1)
                arrival_times.append(float(after.time))
                arrival_waypoints.append(arrived)
                if args.verbose:
                    print(f"waypoint {arrived} reached at {float(after.time):.2f}s; "
                          f"cycles completed={route.cycles_completed}")
                if not route.done:
                    _retarget_goal(robot, controller, points[route.goal_index], viewer)
            # Record the goal active at each stored state (the NEW one on an
            # arrival frame, since the next controller update sees this goal).
            goal_hist.append(robot.goal_xyz())
            goal_index_hist.append(route.goal_index)
            if capture_analysis:
                tip_hist.append(robot.tip_xyz())
                box_hist.append(robot.task_xyz())
                box_vel_hist.append(robot.task_velocity())
                for key in trace:
                    if key == "timing_total_ms":
                        value = info.get("timing_ms", {}).get("total", np.nan)
                    elif key == "timing_sampling_ms":
                        value = info.get("timing_ms", {}).get("sampling", np.nan)
                    elif key == "timing_rollouts_ms":
                        value = info.get("timing_ms", {}).get("rollouts", np.nan)
                    elif key == "tip_box_distance_m":
                        value = tip_error
                    elif key == "box_goal_distance_m":
                        value = robot.box_goal_distance()  # Active target after a switch.
                    elif key == "dwell_seconds":
                        value = route.dwell_seconds
                    elif key == "noise_scale":
                        value = 1.0  # Non-adaptive for every step and every waypoint.
                    else:
                        value = info.get(key, np.nan)
                    trace[key].append(float(value) if value is not None else float("nan"))

            if viewer is not None and not _viewer_sync(viewer):
                break
            if args.verbose and (step + 1) % 100 == 0:
                print(f"step={step+1} goal={route.goal_index} "
                      f"box-goal={robot.box_goal_distance():.4f}m "
                      f"tip-box={tip_error:.4f}m; cycles={route.cycles_completed}")
            if route.done:
                completed = True
                if args.verbose:
                    print(f"route complete: {route.cycles_completed} cycle(s), "
                          f"{route.segments_completed} waypoint arrivals")
                break
    finally:
        _close_viewer(viewer)
        controller.close()
        gc.collect()

    elapsed = time.perf_counter() - t_wall
    dest = ""
    if not args.no_save and args.save:
        path = Path(args.save).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        extra: dict = {}
        if capture_analysis:
            extra = {k: np.asarray(v, dtype=np.float64) for k, v in trace.items()}
            extra.update({
                "analysis_format_version": np.asarray(1, dtype=np.int64),
                "seed": np.asarray(int(args.seed), dtype=np.int64),
                "noise_space": np.asarray(args.noise_space),
                "tip_xyz": np.asarray(tip_hist, dtype=np.float64),
                "box_xyz": np.asarray(box_hist, dtype=np.float64),
                "box_velocity_xyz": np.asarray(box_vel_hist, dtype=np.float64),
                "joint_position": np.asarray(qpos_hist)[:, robot._joint_qadr],
                "joint_velocity": np.asarray(qvel_hist)[:, robot._joint_dadr],
                "success": np.asarray(bool(completed)),
                "goal_tolerance_m": np.asarray(float(args.goal_tolerance)),
                "hold_seconds": np.asarray(float(args.hold_seconds)),
                "tip_box_weight": np.asarray(float(args.tip_box_weight)),
                "box_goal_weight": np.asarray(float(args.box_goal_weight)),
                "control_deviation_weight": np.asarray(float(args.control_deviation_weight)),
                "adaptive_noise": np.asarray(False),
            })
        np.savez_compressed(
            path,
            **extra,
            replay_format_version=np.asarray(5, dtype=np.int64),
            route_mode=np.asarray(args.route),
            cycles_requested=np.asarray(int(args.cycles), dtype=np.int64),
            cycles_completed=np.asarray(int(route.cycles_completed), dtype=np.int64),
            waypoints_reached=np.asarray(int(route.segments_completed), dtype=np.int64),
            waypoints_xy=np.asarray(points, dtype=np.float64),
            goal_waypoint_index_history=np.asarray(goal_index_hist, dtype=np.int64),
            goal_xyz_history=np.asarray(goal_hist, dtype=np.float64),
            arrival_step_indices=np.asarray(arrival_steps, dtype=np.int64),
            arrival_times_s=np.asarray(arrival_times, dtype=np.float64),
            arrival_waypoint_indices=np.asarray(arrival_waypoints, dtype=np.int64),
            robot_name=np.asarray(robot.name),
            sampling_option=np.asarray(args.sampling),
            control_dt=np.asarray(float(args.control_dt), dtype=np.float64),
            physics_dt=np.asarray(float(robot.physics_dt), dtype=np.float64),
            target_xyz=initial_goal_xyz,
            start_xyz=robot.start_xyz(),
            goal_xyz=initial_goal_xyz,
            qpos=np.asarray(qpos_hist, dtype=np.float64),
            qvel=np.asarray(qvel_hist, dtype=np.float64),
            act=np.asarray(act_hist, dtype=np.float64),
            state_time=np.asarray(time_hist, dtype=np.float64),
            controls=np.asarray(controls_hist, dtype=np.float64).reshape((-1, robot.nu)),
        )
        dest = str(path)

    ctrl = np.asarray(controls_hist, dtype=np.float64)
    return {
        "seed": int(args.seed),
        "completed": bool(completed),
        "steps": step + 1,
        "route_mode": args.route,
        "cycles_completed": int(route.cycles_completed),
        "waypoints_reached": int(route.segments_completed),
        "cycles_requested": int(args.cycles),
        "simulation_time_s": float(time_hist[-1] - time_hist[0]),
        "initial_error_m": float(initial_error),
        "final_error_m": float(robot.box_goal_distance()),
        "min_error_m": float(closest_error),
        "final_tip_box_distance_m": float(robot.tip_box_surface_distance()),
        "mean_weighted_cost": _mean_finite(trace["weighted_rollout_cost"]) if capture_analysis else float("nan"),
        "mean_compute_ms": _mean_finite(controller_times),
        "mean_control_increment_l2": float(np.linalg.norm(np.diff(ctrl, axis=0), axis=1).mean())
            if len(ctrl) > 1 else float("nan"),
        "wall_time_s": float(elapsed),
        "sampling": args.sampling,
        "noise_space": args.noise_space,
        "effector_noise_std_m": float(args.effector_noise_std),
        "rollout_backend": backend,
        "recording": dest,
        "best_cost": float(last_info.get("best_cost", np.nan)),
        "lbps_delta": float(args.lbps_delta),
        "goal_tolerance_m": float(args.goal_tolerance),
        "hold_seconds": float(args.hold_seconds),
    }


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Franka MPPI: push a box along a two- or four-point route")
    p.add_argument("--sampling", choices=[a.value for a in SamplingOption], default="spike")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--rollouts", type=int, default=32)
    p.add_argument("--control-dt", type=float, default=0.01)
    p.add_argument("--physics-dt", type=float, default=None,
                   help="Override XML physics timestep (default 0.005 s); "
                        "use 0.005 for two physics substeps at 100 Hz")
    p.add_argument("--twitch-rise-s", type=float, default=0.016)
    p.add_argument("--twitch-decay-s", type=float, default=0.064)
    p.add_argument("--twitch-duration-s", type=float, default=0.200)
    p.add_argument("--joint-noise", type=float, default=0.15)
    p.add_argument("--noise-space", choices=["joint", "effector"], default="effector")
    p.add_argument("--effector-noise-std", type=float, default=0.005)
    p.add_argument("--rollout-workers", type=int, default=8)
    p.add_argument("--motion-limit-scale", type=float, default=1.0)
    p.add_argument("--start-xy", type=float, nargs=2, metavar=("X", "Y"), default=START_XY)
    p.add_argument("--goal-xy", type=float, nargs=2, metavar=("X", "Y"), default=GOAL_XY,
                   help="second corner; start->goal specifies the first square side")
    p.add_argument("--route", choices=["triangle", "backforth", "square", "pingpong", "point"], default="triangle",
                   help="triangle: A-B-C-A (default); backforth: A-B-A; square: A-B-C-D-A; "
                        "pingpong: A-B-C-D-C-B-A; point: one-way A-B")
    p.add_argument("--waypoints", type=float, nargs="+",
                   metavar="XY",
                   help="triangle needs 6 numbers (A,B,C); square/pingpong/backforth need 8")
    p.add_argument("--triangle-depth", type=float, default=DEFAULT_DEPTH_X,
                   help="third triangle waypoint depth, perpendicular to A->B (m)")
    p.add_argument("--cycles", type=int, default=1,
                   help="number of complete routes (triangle: A-B-C-A); 0 repeats until --steps")
    p.add_argument("--tip-box-weight", type=float, default=0.1)
    p.add_argument("--box-goal-weight", type=float, default=10.0)
    p.add_argument("--control-deviation-weight", type=float, default=0.0)
    p.add_argument("--lbps-delta", type=float, default=0.95)
    p.add_argument("--goal-tolerance", type=float, default=0.025)
    p.add_argument("--hold-seconds", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--viewer", dest="viewer", action="store_true", default=True)
    p.add_argument("--no-viewer", dest="viewer", action="store_false")
    p.add_argument("--viewer-ui", action="store_true")
    p.add_argument("--allow-python-rollout", action="store_true")
    p.add_argument("--no-auto-fetch", action="store_true")
    p.add_argument("--analysis-trace", action="store_true")
    p.add_argument("--save", default="manipulation/results/pushing_last_run.npz")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = make_parser().parse_args()
    result = run(args)
    print(f"box-goal: {result['initial_error_m']:.4f}m -> {result['final_error_m']:.4f}m "
          f"(best={result['min_error_m']:.4f}m)")
    print(f"route={result['route_mode']} cycles={result['cycles_completed']}/"
          f"{result['cycles_requested'] or 'unlimited'} "
          f"waypoints_reached={result['waypoints_reached']}")
    print(f"status: {'completed' if result['completed'] else 'incomplete'} "
          f"after {result['steps']} steps ({result['simulation_time_s']:.2f}s simulation)")
    print(f"backend: {result['rollout_backend']} "
          f"compute={result['mean_compute_ms']:.2f}ms/step "
          f"smoothness={result['mean_control_increment_l2']:.5f}")
    if result["recording"]:
        rec = result["recording"]
        print(f"Replay: python -m manipulation.experiments.replay --file {rec}")
        print(f"MP4:    python -m manipulation.experiments.replay --file {rec} "
              f"--mp4 {Path(rec).with_suffix('.mp4')}")


if __name__ == "__main__":
    main()
