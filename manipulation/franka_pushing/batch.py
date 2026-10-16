"""Independent seeded square-pushing trials, matching the reaching batch workflow."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import traceback

import numpy as np

from .run import make_parser, run


ANALYSIS_KEYS = frozenset({
    "tip_xyz", "box_xyz", "box_velocity_xyz", "goal_xyz_history",
    "tip_box_distance_m", "box_goal_distance_m", "joint_position", "joint_velocity",
    "seed", "noise_space", "success", "goal_tolerance_m",
    "weighted_rollout_cost", "best_cost", "ess", "temperature",
    "waypoints_xy", "goal_waypoint_index_history", "route_mode",
    "cycles_completed", "waypoints_reached", "arrival_step_indices",
    "timing_total_ms", "timing_sampling_ms", "timing_rollouts_ms", "noise_scale",
})


def make_batch_parser() -> argparse.ArgumentParser:
    parser = make_parser()
    parser.description = "Independent Franka square/pingpong pushing trials"
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--output-dir", default="manipulation/results/pushing_batch")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def missing_analysis_keys(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as record:
        return sorted(ANALYSIS_KEYS.difference(record.files))


def summarize_recording(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as record:
        controls = np.asarray(record["controls"])
        box_xyz = np.asarray(record["box_xyz"])
        tip_xyz = np.asarray(record["tip_xyz"])
        goal_xyz = np.asarray(record["goal_xyz_history"])
        times = np.asarray(record["state_time"])
        box_error = np.linalg.norm(box_xyz - goal_xyz, axis=1)
        box_v = np.asarray(record["box_velocity_xyz"])
        tip_box_d = np.asarray(record["tip_box_distance_m"])
        q = np.asarray(record["joint_position"])
        dq = np.asarray(record["joint_velocity"])
        delta_u = np.diff(controls, axis=0)
        in_goal = box_error <= float(record["goal_tolerance_m"])

        def mean_finite(key):
            v = np.asarray(record[key], dtype=float)
            v = v[np.isfinite(v)]
            return float(v.mean()) if len(v) else float("nan")

        return {
            "seed": int(record["seed"]),
            "route_mode": str(np.asarray(record["route_mode"]).item()),
            "cycles_completed": int(record["cycles_completed"]),
            "waypoints_reached": int(record["waypoints_reached"]),
            "cycles_requested": int(record["cycles_requested"]),
            "sampling": str(np.asarray(record["sampling_option"]).item()),
            "noise_space": str(np.asarray(record["noise_space"]).item()),
            "success": bool(record["success"]),
            "duration_s": float(times[-1] - times[0]),
            "steps": int(len(controls)),
            "initial_box_error_m": float(box_error[0]),
            "final_box_error_m": float(box_error[-1]),
            "min_box_error_m": float(box_error.min()),
            "mean_box_error_m": float(box_error.mean()),
            "time_inside_tolerance_s": float(np.sum(np.diff(times) * in_goal[1:])),
            "mean_tip_box_distance_m": float(tip_box_d.mean()) if len(tip_box_d) else float("nan"),
            "final_tip_xyz": tip_xyz[-1].tolist(),
            "mean_box_speed_mps": float(np.linalg.norm(box_v, axis=1).mean()),
            "mean_joint_speed_radps": float(np.linalg.norm(dq, axis=1).mean()),
            "joint_path_length_rad": float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()),
            "command_smoothness_l2": float(np.linalg.norm(delta_u, axis=1).mean()) if len(delta_u) else float("nan"),
            "command_variation_l2": float(np.linalg.norm(delta_u, axis=1).sum()) if len(delta_u) else 0.0,
            "mean_weighted_cost": mean_finite("weighted_rollout_cost"),
            "mean_best_cost": mean_finite("best_cost"),
            "mean_ess": mean_finite("ess"),
            "mean_temperature": mean_finite("temperature"),
            "mean_compute_ms": mean_finite("timing_total_ms"),
            "mean_sampling_ms": mean_finite("timing_sampling_ms"),
            "mean_rollout_ms": mean_finite("timing_rollouts_ms"),
            "recording": str(path),
        }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main() -> None:
    args = make_batch_parser().parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    args.viewer = False
    args.analysis_trace = True
    args.no_save = False
    start_seed = args.seed if args.seed_start is None else args.seed_start
    config = {k: v for k, v in vars(args).items()
              if isinstance(v, (str, int, float, bool, tuple, list))}
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    records: list[dict] = []
    errors: list[dict] = []
    for i in range(args.repeats):
        seed = int(start_seed) + i
        dest = out / f"run_{i:03d}_seed_{seed}.npz"
        args.seed = seed
        args.save = str(dest)
        try:
            reuse = bool(args.resume and dest.exists())
            if reuse and missing_analysis_keys(dest):
                print(f"[{i+1}/{args.repeats}] seed={seed}: incomplete trace, rerunning", flush=True)
                reuse = False
            if not reuse:
                run(args)
            if not dest.is_file():
                raise RuntimeError(f"pushing run did not save its recording: {dest}")
            missing = missing_analysis_keys(dest)
            if missing:
                raise RuntimeError(f"pushing trace is incomplete ({', '.join(missing)}): {dest}")
            row = summarize_recording(dest)
            row["trial"] = i
            records.append(row)
            print(f"[{i+1}/{args.repeats}] seed={seed} success={row['success']} "
                  f"cycles={row['cycles_completed']} arrivals={row['waypoints_reached']} "
                  f"error={row['final_box_error_m']:.3f}m "
                  f"compute_ms={row['mean_compute_ms']:.2f}", flush=True)
        except Exception as exc:
            errors.append({"trial": i, "seed": seed,
                           "error": repr(exc), "traceback": traceback.format_exc()})
            print(f"[{i+1}/{args.repeats}] seed={seed} FAILED: {exc}", flush=True)
            if args.stop_on_error:
                raise
        finally:
            _write_csv(out / "summary.csv", records)
            (out / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    print(f"Completed {len(records)}/{args.repeats} trials; data in {out}")


if __name__ == "__main__":
    main()
