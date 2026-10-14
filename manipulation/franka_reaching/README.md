# Franka reaching

Run from the directory containing `manipulation`:

```bash
python -m manipulation.franka_reaching.run --sampling spike --noise-space effector --viewer
```

Five 3D fingertip targets are followed in a sweep. The objective includes running and terminal end-effector position error, near-goal velocity regulation, joint command curvature, and control deviation. Adaptive noise uses `--dslow`, `--noise-decay-order`, and `--noise-min-fraction`.

The shared `manipulation.mppi`, `manipulation.robots`, `manipulation.priors`, and `manipulation.tracks` modules remain outside the task folders. Replay remains in `manipulation.experiments.replay`.

## Repeated reaching experiment (100 seeds)

```bash
python -m manipulation.franka_reaching.batch \
  --sampling spike --noise-space effector \
  --repeats 100 --seed-start 1 \
  --output-dir manipulation/results/reaching_spike_effector
```

The batch runner executes **independently reset** trials with different random
seeds and the same ordered sequence of reaching targets. There is no graphical
viewer during batch execution. Each run is saved as a replay-compatible NPZ;
`summary.csv` has one row per seed, `config.json` records the command-line
settings, and `errors.json` contains any failed runs. Use `--resume` to skip
existing run files when restarting an interrupted study.

Trace arrays in each NPZ:
- `qpos`, `qvel`, `act`, `state_time`, `controls`, `active_target_index`;
- `joint_position`, `joint_velocity` (N+1 state samples), `tip_xyz`,
  `tip_velocity_xyz` (Jacobian times joint velocity), `target_xyz_history`;
- `best_cost`, `weighted_rollout_cost`, `nominal_cost`, `ess`, `temperature`,
  `lbps_score`, `timing_total_ms`, `timing_sampling_ms`, `timing_rollouts_ms`,
  `dwell_seconds`, `target_distance_m`, `noise_scale` (N control steps);
- validation step/time arrays and experimental metadata.

Positions/velocities are measured from the simulated state **after** each
command, with an additional initial state. The recorded `controls[t]` takes the
system from state `t` to state `t+1`. For statistics, treat each seed as one
independent replicate; the 9 within-run target visits are repeated measures.
The current `--steps` limit applies to the entire target sequence; timeouts
are reported as failures rather than silently discarded.
