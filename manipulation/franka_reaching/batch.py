"""Independent seeded, replay-compatible reaching trials for sampler comparisons."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import traceback

import numpy as np

from .run import make_parser, run


def make_batch_parser() -> argparse.ArgumentParser:
    parser = make_parser()
    parser.description = "Repeated Franka reaching trials (independent seeded resets)"
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=None,
                        help="first seed (defaults to --seed)")
    parser.add_argument("--output-dir", default="manipulation/results/reaching_batch")
    parser.add_argument("--resume", action="store_true", help="skip completed run files")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


# Required fields distinguish a full analysis recording from a replay-only NPZ.
ANALYSIS_KEYS = frozenset({
    "tip_xyz", "tip_velocity_xyz", "target_xyz_history", "joint_position",
    "joint_velocity", "seed", "noise_space", "success",
    "target_validation_times", "goal_tolerance_m", "weighted_rollout_cost",
    "best_cost", "ess", "temperature", "timing_total_ms",
    "timing_sampling_ms", "timing_rollouts_ms",
})


def missing_analysis_keys(path: Path) -> list[str]:
    """Return missing metrics without loading large trajectory arrays."""
    with np.load(path, allow_pickle=False) as record:
        return sorted(ANALYSIS_KEYS.difference(record.files))


def summarize_recording(path: Path) -> dict:
    """Summaries calculated per independent seed (not per time step)."""
    with np.load(path, allow_pickle=False) as record:
        controls = np.asarray(record["controls"])
        tip = np.asarray(record["tip_xyz"])
        tip_v = np.asarray(record["tip_velocity_xyz"])
        targets = np.asarray(record["target_xyz_history"])
        errors = np.linalg.norm(tip - targets, axis=1)
        times = np.asarray(record["state_time"])
        q = np.asarray(record["joint_position"])
        dq = np.asarray(record["joint_velocity"])
        delta = np.diff(controls, axis=0)
        speed = np.linalg.norm(tip_v, axis=1)
        validated = np.asarray(record["target_validation_times"])
        goal_tol = float(record["goal_tolerance_m"])
        in_goal = errors <= goal_tol

        def mean_finite(key):
            values = np.asarray(record[key], dtype=float)
            values = values[np.isfinite(values)]
            return float(values.mean()) if len(values) else float("nan")

        return {
            "seed": int(record["seed"]),
            "sampling": str(np.asarray(record["sampling_option"]).item()),
            "noise_space": str(np.asarray(record["noise_space"]).item()),
            "success": bool(record["success"]),
            "validated_targets": int(len(validated)),
            "duration_s": float(times[-1] - times[0]),
            "steps": int(len(controls)),
            "final_error_m": float(errors[-1]),
            "min_error_m": float(errors.min()),
            "mean_error_m": float(errors.mean()),
            "time_inside_tolerance_s": float(np.sum(np.diff(times) * in_goal[1:])),
            "mean_tip_speed_mps": float(speed.mean()),
            "final_tip_speed_mps": float(speed[-1]),
            "mean_joint_speed_radps": float(np.linalg.norm(dq, axis=1).mean()),
            "joint_path_length_rad": float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()),
            "command_smoothness_l2": float(np.linalg.norm(delta, axis=1).mean()) if len(delta) else float("nan"),
            "command_variation_l2": float(np.linalg.norm(delta, axis=1).sum()) if len(delta) else 0.0,
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
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main() -> None:
    args = make_batch_parser().parse_args()
    if args.repeats < 1:
        raise ValueError('--repeats must be >= 1')
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.viewer:
        print('Batch mode disables the viewer; no changes to single-run viewer behavior.')
    args.viewer = False
    args.analysis_trace = True
    args.batch_mode = True
    args.no_save = False
    start = args.seed if args.seed_start is None else args.seed_start
    config = {key: value for key, value in vars(args).items()
              if isinstance(value, (str, int, float, bool))}
    (out / 'config.json').write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    records: list[dict] = []
    errors: list[dict] = []
    for i in range(args.repeats):
        seed = start + i
        dest = out / f'run_{i:03d}_seed_{seed}.npz'
        args.seed = seed
        args.save = str(dest)
        try:
            reuse = bool(args.resume and dest.exists())
            if reuse:
                missing = missing_analysis_keys(dest)
                if missing:
                    print(f'[{i+1}/{args.repeats}] seed={seed}: existing recording lacks '
                          f'analysis fields ({", ".join(missing)}); rerunning', flush=True)
                    reuse = False
            if not reuse:
                run(args)
            if not dest.is_file():
                raise RuntimeError(f'reaching run did not save its recording: {dest}')
            missing = missing_analysis_keys(dest)
            if missing:
                raise RuntimeError(
                    f'Simulation saved {dest}, but its NPZ is missing analysis fields: '
                    f'{", ".join(missing)}. Your installed franka_reaching/run.py is '
                    'out of sync with batch.py. Replace BOTH files from the updated '
                    'package, and ensure run.py saves the `additional` dictionary '
                    'with `analysis_trace=True`. The trajectory is still saved; '
                    'only summarization failed.'
                )
            row = summarize_recording(dest)
            row['trial'] = i
            records.append(row)
            print(f'[{i+1}/{args.repeats}] seed={seed} success={row["success"]} '
                  f'targets={row["validated_targets"]} steps={row["steps"]} '
                  f'mean_compute_ms={row["mean_compute_ms"]:.2f}', flush=True)
        except Exception as exc:
            errors.append({'trial': i, 'seed': seed, 'error': repr(exc), 'traceback': traceback.format_exc()})
            print(f'[{i+1}/{args.repeats}] seed={seed} FAILED: {exc}', flush=True)
            if args.stop_on_error:
                raise
        finally:
            _write_csv(out / 'summary.csv', records)
            (out / 'errors.json').write_text(json.dumps(errors, indent=2) + '\n')
    print(f'Completed {len(records)}/{args.repeats} trials; data in {out}')


if __name__ == '__main__':
    main()
