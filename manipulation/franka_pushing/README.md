# Franka box pushing: four tabletop waypoints

A standalone `franka_pushing` task alongside `franka_reaching`, sharing the **same** Panda arm, single-finger tool, MPPI controller, sampling methods, warm start, LBPS temperature adaptation, motion limits, physics, replay, and batch infrastructure.

The physical box is initially at corner **A**. Its active goal advances to the next corner only once its **center** remains within the position tolerance for the required dwell time. The box is **not teleported or reset** when the target changes. There is no box/tool orientation objective. The current goal is shown by the green, non-colliding ghost box; the four corners are also shown by colored sites.

## Default square

The first two corners come from `--start-xy` and `--goal-xy`. The remaining corners are obtained by a clockwise 90-degree rotation of the first edge, keeping equal-length sides. Defaults (metres):

```
    B = (0.55, 0.18) ------ C = (0.73, 0.18)
         |                          |
         |                          |
    A = (0.55, 0.00) ------ D = (0.73, 0.00)
      initial box
```

Available traversals:

- `--route square` (default): **A -> B -> C -> D -> A**, then repeat.
- `--route pingpong`: **A -> B -> C -> D -> C -> B -> A**, then repeat in the same order (reverses direction at D).
- `--route point`: legacy one-way **A -> B**, stop when B is reached.

`--cycles 2` (default) completes two traversals. `--cycles 0` keeps repeating, up to the configured `--steps` bound; in that case `completed=false` means the continuous run stopped at its step limit, and `cycles_completed` gives the number of full traversals reached. `--goal-tolerance 0.025` m and `--hold-seconds 1.0` are the unchanged arrival thresholds.

You can explicitly specify any four tabletop points in visitation order using `--waypoints AX AY BX BY CX CY DX DY`. When provided, this overrides `--start-xy` and `--goal-xy`; the points need not be an exact square. Each point must keep the full box footprint on the table. The final edge D -> A must be nonzero.

## Cost (unchanged: exactly three terms)

For rollout horizon `H`, action `u_t`, nominal command `u_nom,t`, per-actuator control scales `s_j`, and **the current active keypoint** `g`:

```
J = (1/H) * sum_t [
       w_tip  * distance(tip(t), surface(box(t)))
     + w_goal * || center(box(t)) - g ||_2
     + w_u    * mean_j( ((u_t,j - u_nom,t,j) / s_j)^2 )
]
```

Defaults: `w_tip = 1`, `w_goal = 1`, `w_u = 1e-4`. No intermediate-waypoint lookahead, cumulative path cost, orientation, progress, terminal, velocity, contact, lift, tip-speed, acceleration, or other shaping terms. Simulated physical contacts, friction, and actuator limits are unchanged. MPPI plans toward only **one waypoint at a time**; once reached, the goal is retargeted outside the rollout.

**No covariance reduction near any corner:** `adaptive_noise=False` and `noise_min_fraction=1.0` keep the sampling multiplier fixed at `1.0` for every sampler and waypoint, as in the original point-pushing task.

The two distance terms are motivated by section II-H of the attached *Motoneuron-Inspired Sampling for Model Predictive Path Integral Control* paper (page 7); the paper does not prescribe this exact Cartesian objective.

## Running

From the directory containing `manipulation/`:

```bash
python manipulation/setup_native.py build_ext --inplace
python -m manipulation.franka_pushing.run --sampling spike --noise-space effector --viewer --verbose
```

The rebuilt native extension supports pushing mode 4 (`point_push_abi=1`). If it is unavailable, `--allow-python-rollout` allows slower scalar development/debugging. The original reaching code remains unchanged.

Run continuously around the square until the step limit:

```bash
python -m manipulation.franka_pushing.run --route square --cycles 0 --steps 10000 --viewer
```

Reverse direction at each end of the four-point path:

```bash
python -m manipulation.franka_pushing.run --route pingpong --cycles 2 --viewer
```

Change the size/location or supply four explicit waypoints:

```bash
python -m manipulation.franka_pushing.run \
    --start-xy 0.53 -0.08 --goal-xy 0.53 0.07 --route square --cycles 2
python -m manipulation.franka_pushing.run \
    --waypoints 0.48 -0.10 0.48 0.10 0.64 0.10 0.64 -0.10 --route pingpong
```

Use `--route point` to restore the original A-to-B scenario:

```bash
python -m manipulation.franka_pushing.run --route point --start-xy 0.55 0 --goal-xy 0.55 0.18
```

## Retargeting and replay

On each validated arrival, `robot.set_goal_xy()` moves the non-colliding goal marker, the controller reloads the **fused native MuJoCo model** with the updated goal, and any persistent `mj_rollout` mocap data are synchronized. The MPPI warm-start plan is reset at this discontinuous task change, avoiding reuse of commands intended for the previous direction. The physical box and simulation clock continue uninterrupted.

Every replay-compatible NPZ includes `qpos`, `qvel`, `act`, `state_time`, `controls`, `waypoints_xy`, `goal_xyz_history`, `goal_waypoint_index_history`, and the per-corner arrival times/indices. Replay and video export reconstruct the **changing goal marker**, not just the box motion. `--analysis-trace` additionally saves tip/box/joint trajectories and MPPI diagnostics.

```bash
python -m manipulation.experiments.replay --file manipulation/results/pushing_last_run.npz
python -m manipulation.experiments.replay --file manipulation/results/pushing_last_run.npz \
    --mp4 manipulation/results/pushing_last_run.mp4
```

Repeated trials:

```bash
python -m manipulation.franka_pushing.batch --sampling spike --repeats 100 \
    --output-dir manipulation/results/pushing_batch
```

`summary.csv` includes completed traversals, reached waypoints, success, active-goal box error, control smoothness, and computation time. The underlying objective never changes between legs; only the active goal position is updated.
