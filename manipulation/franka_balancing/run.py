"""Run reproducible Franka ball-manipulation MPPI experiments.

Separates task-specific loss functions and scene variants while using the exact
same MPPI controller/sampling APIs as reaching/pushing. A task-specific fused C++ backend evaluates these losses during physics rollouts.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import time

import numpy as np
import mujoco

from manipulation.mppi import ControllerConfig, JointMPPIController, SamplingOption
from manipulation.priors import GeometricPrior
from manipulation.robots import FrankaBallRobot
from manipulation.tracks import StraightTrack


TASK = "balancing"


def make_parser() -> argparse.ArgumentParser:
    task = TASK
    p = argparse.ArgumentParser(
        description=f"Franka ball {task} with MPPI sampling comparisons"
    )
    p.add_argument("--sampling", choices=[a.value for a in SamplingOption], default="spike")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=2000 if task == "balancing" else 800)
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--rollouts", type=int, default=32)
    p.add_argument("--control-dt", type=float, default=0.01)
    p.add_argument("--physics-dt", type=float, default=None,
                   help="Override XML physics timestep (default 0.005 s); "
                        "use 0.005 for two physics substeps at 100 Hz")
    p.add_argument("--joint-noise", type=float, default=0.08)
    p.add_argument("--noise-space", choices=("joint", "effector"), default="effector")
    p.add_argument("--effector-noise-std", type=float, default=0.01)
    p.add_argument("--rollout-workers", type=int, default=4)
    p.add_argument("--motion-limit-scale", type=float, default=1.0)
    p.add_argument("--lbps-delta", type=float, default=0.95)
    p.add_argument("--twitch-rise-s", type=float, default=0.016)
    p.add_argument("--twitch-decay-s", type=float, default=0.064)
    p.add_argument("--twitch-duration-s", type=float, default=0.200)
    p.add_argument("--offset-x", type=float, default=0.035 if task == "balancing" else 0.0)
    p.add_argument("--offset-y", type=float, default=-0.025 if task == "balancing" else 0.0)
    p.add_argument("--velocity-x", type=float, default=0.09 if task == "balancing" else 0.015)
    p.add_argument("--velocity-y", type=float, default=0.00 if task == "balancing" else 0.0)
    p.add_argument("--wind-x", type=float, default=0.0,
                   help="Ambient world-frame wind X (m/s)")
    p.add_argument("--wind-y", type=float, default=0.0,
                   help="Ambient world-frame wind Y (m/s)")
    p.add_argument("--wind-z", type=float, default=0.0,
                   help="Ambient world-frame wind Z (m/s)")
    p.add_argument("--spin-x", type=float, default=0.0,
                   help="Initial world-frame ball spin X (rad/s)")
    p.add_argument("--spin-y", type=float, default=28.0 if task == "bouncing" else 0.0,
                   help="Initial world-frame ball spin Y (rad/s)")
    p.add_argument("--spin-z", type=float, default=0.0,
                   help="Initial world-frame ball spin Z (rad/s)")
    p.add_argument("--drop-height", type=float, default=0.23,
                   help="For bouncing, initial height ABOVE the racket (ball radius excluded)")
    if task == "bouncing":
        p.add_argument("--target-apex", type=float, default=0.2,
                       help="Target apex above racket surface (m), ball radius excluded")
        p.add_argument("--target-bounces", type=int, default=50)
    p.add_argument("--save", type=str, default=None,
                   help="Output NPZ path; default: results/franka_ball_<task>/<sampling>_<seed>.npz")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--continue-after-failure", action="store_true")
    p.add_argument("--no-auto-fetch", action="store_true")
    p.add_argument("--allow-stock-rollout", action="store_true",
                   help="Allow slower non-fused MuJoCo rollout for cost debugging")
    p.add_argument("--viewer", dest="viewer", action="store_true", help="Show MuJoCo viewer (default for interactive runs)")
    p.add_argument("--no-viewer", dest="viewer", action="store_false", help="Run headless")
    p.set_defaults(viewer=True)
    p.add_argument("--viewer-ui", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def build_controller(robot: FrankaBallRobot, args: argparse.Namespace) -> JointMPPIController:
    if int(args.rollouts) < 2 or int(args.horizon) < 2:
        raise ValueError("horizon and rollouts must be >= 2")
    # All the 'original reaching/pushing' cost weights are irrelevant when the
    # ball adapter supplies evaluate_ball_rollouts: both objectives are scored
    # ONLY by their own explicit cost.py.
    cfg = ControllerConfig(
        control_dt=float(args.control_dt),
        horizon=int(args.horizon),
        num_rollouts=int(args.rollouts),
        adaptive_noise=False,
        noise_min_fraction=1.0,
        joint_noise_fraction=float(args.joint_noise),
        noise_space=str(args.noise_space),
        effector_noise_std=float(args.effector_noise_std),
        rollout_workers=int(args.rollout_workers),
        lbps_delta=float(args.lbps_delta),
        twitch_rise_s=float(args.twitch_rise_s),
        twitch_decay_s=float(args.twitch_decay_s),
        twitch_duration_s=float(args.twitch_duration_s),
        nominal_diagnostics=False,
        nominal_refine_iterations=0,
        screen_pool_size=0,
        screen_exploit=0,
        warm_start=True,
    )
    point = robot.task_xy()
    # Only an API compatibility adapter: its geometry/progress contributes
    # NOTHING to the new objective.
    track = StraightTrack(origin_xy=(float(point[0]), float(point[1])),
                          direction=(1.0, 0.0), road_width=100.0, length=10.0)
    return JointMPPIController(robot, track, GeometricPrior(), cfg,
                               sampling=SamplingOption(args.sampling), seed=int(args.seed))


def run(args: argparse.Namespace) -> dict:
    task = TASK
    if args.steps < 1 or args.rollouts < 2 or args.horizon < 2:
        raise ValueError("steps >= 1 and rollouts/horizon >= 2 are required")
    if not (0.0 < args.motion_limit_scale <= 1.0):
        raise ValueError("motion-limit-scale must be in (0, 1]")
    if args.joint_noise <= 0 or args.control_dt <= 0:
        raise ValueError("joint-noise and control-dt must be positive")
    if task == "bouncing" and (args.target_apex <= 0 or args.target_bounces <= 0):
        raise ValueError("target-apex and target-bounces must be positive")
    if args.drop_height < 0:
        raise ValueError("drop-height must be nonnegative")
    if args.physics_dt is not None and (not np.isfinite(args.physics_dt) or args.physics_dt <= 0):
        raise ValueError("physics-dt must be finite and positive")
    if not np.all(np.isfinite([args.spin_x, args.spin_y, args.spin_z])):
        raise ValueError("ball spin must be finite")
    if not np.all(np.isfinite([args.wind_x, args.wind_y, args.wind_z])):
        raise ValueError("ambient wind must be finite")

    robot = FrankaBallRobot(
        task=task, auto_fetch=not args.no_auto_fetch,
        motion_limit_scale=args.motion_limit_scale,
        initial_ball_offset_xy=(args.offset_x, args.offset_y),
        initial_ball_velocity_xy=(args.velocity_x, args.velocity_y),
        initial_ball_spin_xyz=(args.spin_x, args.spin_y, args.spin_z),
        drop_height=args.drop_height,
    )
    # Look up the free ball's body once, then read its height after each step.
    ball_body_id = mujoco.mj_name2id(
        robot.model, mujoco.mjtObj.mjOBJ_BODY, "reach_target"
    )
    if ball_body_id < 0:
        raise ValueError("Ball body 'reach_target' not found in MuJoCo model")
    if args.physics_dt is not None:
        robot.model.opt.timestep = float(args.physics_dt)
    robot.model.opt.wind[:] = [args.wind_x, args.wind_y, args.wind_z]
    if task == "bouncing":
        robot.cost_weights = replace(robot.cost_weights, target_apex_m=float(args.target_apex))
    controller = build_controller(robot, args)
    backend = controller.rollout_backend_name
    if controller.rollout_batcher is None or not controller.rollout_batcher.supports_vectorized_cost:
        controller.close()
        raise RuntimeError(
            "The ball tasks require mujoco.rollout with per-step frame sensors. "
            "Install the MuJoCo Python package and use a compatible version."
        )
    if controller.rollout_batcher.ball_evaluator is None and not args.allow_stock_rollout:
        controller.close()
        raise RuntimeError(
            f"Expected native fused ball evaluator (got {backend!r}). "
            "Build with `python manipulation/setup_native.py build_ext --inplace` "
            "or pass --allow-stock-rollout for reference/debugging."
        )

    physics_per_control = int(controller.control_substeps)
    sample = robot.snapshot()
    history = {"qpos": [sample.qpos.copy()], "qvel": [sample.qvel.copy()],
               "act": [sample.act.copy()], "state_time": [float(sample.time)],
               "controls": [], "ball_xyz": [robot.task_xyz()],
               "tool_xyz": [robot.tip_xyz()], "ball_xy_local": [],
               "ball_v_xy_local": [], "ball_gap": [], "bounce_count": [0],
               "ball_spin_xyz": [robot.data.qvel[robot._task_dof_adr+3:robot._task_dof_adr+6].copy()],
               "best_cost": [], "weighted_rollout_cost": [], "ess": [],
               "temperature": [], "compute_ms": [], "sampling_ms": [],
               "rollout_ms": []}
    relative, velocity, gap = robot.ball_local_state()
    history["ball_xy_local"].append(relative)
    history["ball_v_xy_local"].append(velocity)
    history["ball_gap"].append(gap)
    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(
            robot.model,
            robot.data,
            show_left_ui=args.viewer_ui,
            show_right_ui=args.viewer_ui,
        )
        viewer.cam.lookat[:] = [0.55, 0.12, 0.51]
        viewer.cam.distance = 1.25
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25
        viewer.sync(state_only=True)
    failure = False
    finished_target = False
    n_steps = 0
    wall_start = time.perf_counter()
    try:
        if args.verbose:
            print(f"task={task} sampler={args.sampling} seed={args.seed} "
                  f"H={args.horizon} N={args.rollouts} dt={args.control_dt:.3f} "
                  f"backend={backend}")
            print("cost weights:", json.dumps(asdict(robot.cost_weights), sort_keys=True))
        for iteration in range(args.steps):
            action, info = controller.step(robot.data, 0.0)
            robot.step_control(action, substeps=physics_per_control)
            current = robot.snapshot()
            n_steps = iteration + 1
            history["qpos"].append(current.qpos.copy())
            history["qvel"].append(current.qvel.copy())
            history["act"].append(current.act.copy())
            history["state_time"].append(float(current.time))
            history["controls"].append(current.ctrl.copy())
            history["ball_xyz"].append(robot.task_xyz())
            history["tool_xyz"].append(robot.tip_xyz())
            xy, vxy, gap = robot.ball_local_state()
            history["ball_xy_local"].append(xy)
            history["ball_v_xy_local"].append(vxy)
            history["ball_gap"].append(gap)
            history["bounce_count"].append(int(robot.bounce_count))
            history["ball_spin_xyz"].append(
                robot.data.qvel[robot._task_dof_adr+3:robot._task_dof_adr+6].copy()
            )
            for key in ("best_cost", "weighted_rollout_cost", "ess", "temperature"):
                history[key].append(float(info.get(key, float("nan"))))
            for key, src in (("compute_ms", "total"), ("sampling_ms", "sampling"),
                             ("rollout_ms", "rollouts")):
                history[key].append(float(info.get("timing_ms", {}).get(src, float("nan"))))

            radius = max(robot.tool_half_width - robot.ball_radius, 1e-6)
            if task == "balancing":
                ball_z = robot.data.xpos[ball_body_id, 2]
                ground_z = 0.0  # Ground surface height in world coordinates.
                failure = ball_z < ground_z + 2.0 * robot.ball_radius
            else:
                failure = (gap < -0.055 or
                           (np.max(np.abs(xy)) > radius + 0.055 and gap < 0.05))
                finished_target = robot.bounce_count >= args.target_bounces
            if args.verbose and (n_steps % 25 == 0 or failure or finished_target):
                print(f"step={n_steps} t={current.time:.3f}s ball_local={xy.round(3)} "
                      f"gap={gap:.3f} bounces={robot.bounce_count} "
                      f"cost={info['best_cost']:.4f}")
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync(state_only=True)
            if finished_target or (failure and not args.continue_after_failure):
                break
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        controller.close()
        gc.collect()

    elapsed_wall = time.perf_counter() - wall_start
    def array(key):
        return np.asarray(history[key], dtype=np.float64)
    recorded = ""
    if not args.no_save:
        dest = (Path(args.save).expanduser() if args.save else
                Path("results") / f"franka_ball_{task}" / f"{args.sampling}_seed{args.seed}.npz")
        dest = dest.resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest, replay_format_version=np.asarray(5, dtype=np.int64),
            robot_name=np.asarray(robot.name), task=np.asarray(task),
            sampling_option=np.asarray(args.sampling), seed=np.asarray(args.seed),
            noise_space=np.asarray(args.noise_space),
            control_dt=np.asarray(args.control_dt), physics_dt=np.asarray(robot.physics_dt),
            horizon=np.asarray(args.horizon), rollouts=np.asarray(args.rollouts),
            cost_weights_json=np.asarray(json.dumps(asdict(robot.cost_weights))),
            initial_ball_offset_xy=np.asarray([args.offset_x, args.offset_y]),
            initial_ball_velocity_xy=np.asarray([args.velocity_x, args.velocity_y]),
            initial_ball_spin_xyz=np.asarray([args.spin_x, args.spin_y, args.spin_z]),
            ambient_wind_xyz=np.asarray([args.wind_x, args.wind_y, args.wind_z]),
            drop_height=np.asarray(args.drop_height),
            qpos=array("qpos"), qvel=array("qvel"), act=array("act"),
            state_time=array("state_time"), controls=array("controls"),
            ball_xyz=array("ball_xyz"), tool_xyz=array("tool_xyz"),
            ball_xy_local=array("ball_xy_local"),
            ball_v_xy_local=array("ball_v_xy_local"), ball_gap=array("ball_gap"),
            bounce_count=np.asarray(history["bounce_count"], dtype=np.int64),
            ball_spin_xyz=array("ball_spin_xyz"),
            best_cost=array("best_cost"), weighted_rollout_cost=array("weighted_rollout_cost"),
            ess=array("ess"), temperature=array("temperature"),
            compute_ms=array("compute_ms"), sampling_ms=array("sampling_ms"),
            rollout_ms=array("rollout_ms"),
        )
        recorded = str(dest)
    compute = array("compute_ms")
    return {
        "task": task, "seed": int(args.seed), "sampling": args.sampling,
        "success": (not failure and n_steps == args.steps) if task == "balancing" else finished_target,
        "failed": bool(failure), "steps": int(n_steps),
        "duration_s": float(history["state_time"][-1] - history["state_time"][0]),
        "bounces": int(robot.bounce_count),
        "mean_ball_spin_rad_s": float(np.linalg.norm(array("ball_spin_xyz"), axis=-1).mean()),
        "final_xy_error_m": float(np.linalg.norm(array("ball_xy_local")[-1])),
        "mean_xy_error_m": float(np.linalg.norm(array("ball_xy_local"), axis=-1).mean()),
        "mean_compute_ms": float(np.mean(compute[np.isfinite(compute)])) if np.isfinite(compute).any() else float("nan"),
        "elapsed_wall_s": elapsed_wall, "backend": backend, "recording": recorded,
    }


def main() -> None:
    args = make_parser().parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))
    if result["recording"]:
        print(f"Replay: python -m manipulation.experiments.replay --file '{result['recording']}'")


def batch_main() -> None:
    task = TASK
    p = make_parser()
    p.description = f"Multi-seed, multi-sampler Franka ball-{task} comparison"
    p.add_argument("--samplers", nargs="+", choices=[a.value for a in SamplingOption],
                   default=["standard", "pink", "gaussian-spike", "spike", "poisson"])
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--csv", type=str, default=f"results/franka_ball_{task}/summary.csv")
    p.set_defaults(viewer=False)
    args = p.parse_args()
    # Batch comparisons are headless unless --viewer is explicitly supplied.
    # The default viewer applies only to interactive `run.py`.
    if args.viewer:
        raise ValueError("Batch mode is headless; pass --no-viewer")
    rows = []
    for sampler in args.samplers:
        for seed in args.seeds:
            single = argparse.Namespace(**vars(args))
            single.sampling = sampler
            single.seed = int(seed)
            if args.save is None:
                single.save = f"results/franka_ball_{task}/{sampler}_seed{seed}.npz"
            result = run(single)
            print(json.dumps(result))
            rows.append(result)
    dest = Path(args.csv).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV: {dest}")


if __name__ == "__main__":
    main()