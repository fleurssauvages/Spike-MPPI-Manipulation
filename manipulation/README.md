## Task entry points

- Reaching: `python -m manipulation.franka_reaching.run --sampling spike --noise-space effector`
- Box pushing: `python -m manipulation.franka_pushing.run --sampling spike --noise-space effector`

Both experiments share the core MPPI and robot infrastructure but keep their task cost and experiment logic separate.

# Franka MPPI five-target reaching benchmark

This package is intentionally focused on one clean free-space reaching task for
comparing MPPI sampling structures. The wall, box-pushing, and button tasks are
removed.

## Robot

- Official MuJoCo Menagerie Panda arm through `link7`.
- Stock Panda hand/finger/tendon subtree replaced by one short central tip.
- Seven Panda arm position servos only.
- Position commands are capped by the Panda maximum joint velocities in command
  space; no extra acceleration limiter is applied.

## Five-target task

The fingertip moves through five fixed 3-D targets:

```text
1 = (0.45, -0.25, 0.50) m
2 = (0.50, -0.12, 0.62) m
3 = (0.55,  0.00, 0.50) m
4 = (0.50,  0.12, 0.62) m
5 = (0.45,  0.25, 0.50) m
```

One cycle is the deterministic back-and-forth sweep:

```text
1 -> 2 -> 3 -> 4 -> 5 -> 4 -> 3 -> 2 -> 1
```

The active target is shown as the larger green sphere. All five target positions
also have small translucent markers.

A target is validated only if the fingertip remains continuously inside the
Cartesian tolerance sphere for 1.0 s. Leaving the sphere resets the dwell timer.
With the default `control_dt=0.01 s`, this corresponds to 100 consecutive control
updates inside the target region.

## Cost

The default rollout cost is deliberately minimal:

1. normalized running 3-D fingertip-to-active-target distance;
2. very small nominal-command regularization.

Tool-down orientation is available through `--downward-weight`, but defaults to
zero so it does not confound sampling comparisons. There are no obstacle or
contact penalties.

Default weights:

```text
target distance           = 1.0
command regularization    = 1e-4
downward orientation      = 0.0
```

## Build fused rollout

The native evaluator uses the dedicated 3-D point-target mode. After replacing
an older package, rebuild once:

```bash
rm -f manipulation/_fused_mujoco*.so
python manipulation/setup_native.py build_ext --inplace
```

## Run

```bash
python -m manipulation.franka_reaching.run \
  --sampling spike \
  --verbose
```

## Joint-space vs end-effector-space exploration

All five sampling methods accept `--noise-space joint` (default, unchanged)
or `--noise-space effector`:

```bash
python -m manipulation.franka_reaching.run \
  --sampling spike --noise-space effector --effector-noise-std 0.02 --verbose
```

Effector mode uses the **first three normalized proposal channels** as XYZ
fingertip displacements (metres), and applies differential IK
`delta_q = pinv(J_xyz(q_nominal)) @ delta_xyz` at every nominal horizon step.
The remaining proposal channels are ignored in effector mode. The existing
sampler implementations, including their temporal statistics, are unchanged;
this is an XYZ interpretation of three source channels, not a new native
three-dimensional spike motor-pool implementation. MuJoCo rollouts and the
weighted MPPI update still use seven joint-position commands. Control limits
are respected through clipping. Linearized IK may deviate from the requested
Cartesian offset for large perturbations or near singularities; the redundant
joint null space is not explicitly sampled.

`--joint-noise` is used only for native joint-space proposal scaling (and is
removed when normalizing the first three channels in effector mode).
`--effector-noise-std` sets the Cartesian displacement scale in metres.

Sampling methods:

```text
standard
poisson
spike
gaussian-poisson
gaussian-spike
```

Main defaults:

```text
control_dt       = 0.01 s
horizon          = 100
rollouts         = 32
joint_noise      = 0.15
velocity cap     = 1.0 x Panda maximum
LBPS delta       = 0.95
goal tolerance   = 0.025 m
hold time        = 1.0 s
cycles           = 1
step budget      = 4000
```

LBPS delta remains selectable:

```bash
python -m manipulation.franka_reaching.run \
  --sampling spike \
  --lbps-delta 0.8 \
  --verbose
```

The hold criterion and number of sweeps can also be changed:

```bash
python -m manipulation.franka_reaching.run \
  --hold-seconds 1.0 \
  --cycles 2
```

## Metrics

Each run reports completion, validated targets, requested targets, active target,
final/minimum Cartesian error, steps, wall-clock runtime, rollout backend, best
cost, and LBPS delta. Verbose mode additionally prints the continuous dwell time
for the active target.

For sampler comparisons, vary `--seed` while keeping every task/controller
parameter identical.

## Replay / GIF

```bash
python -m manipulation.experiments.replay \
  --file manipulation/results/last_run.npz
```

```bash
python -m manipulation.experiments.replay \
  --file manipulation/results/last_run.npz \
  --gif manipulation/results/last_run.gif
```

### Reaching stabilization (Franka)

The reaching experiment enables the following shared stabilization configuration
for all sampling methods and for `--noise-space joint|effector`:

```bash
# Rebuild native fused rollout (required because the cost ABI has changed)
python -m pip install 'mujoco>=3.3' 'pybind11>=2.12'
python manipulation/setup_native.py build_ext --inplace

python -m manipulation.franka_reaching.run --sampling spike \
  --noise-space effector --reach-terminal-weight 2 \
  --reach-velocity-weight 0.2 --reach-region-weight 2 \
  --reach-settle-weight 0.2 --reach-acceleration-weight 0.01 \
  --dslow 0.10
```

Reaching costs added to the original running distance: normalized terminal
position error; a Gaussian-gated squared Cartesian tip velocity cost (finite
differences between actual simulated tip positions); a quadratic penalty outside
a 15 mm safety radius; and a position-plus-velocity settling cost over the last
30 control steps. The acceleration regularizer penalizes the **second difference
of joint-position commands** (scaled by joint command span), not measured joint
acceleration. It therefore measures control curvature, rather than dynamic
acceleration. Always-on noise scaling shrinks the proposal perturbation from 100%
when >=10 cm away to 10% at the exact target; for end-effector mode this scales
Cartesian proposal noise before inverse kinematics. Candidate zero remains the
unaltered nominal plan. To reproduce the original cost, set all `--reach-*-weight`
options to zero and set `--noise-min-fraction 1` to use constant exploration.

**Important:** This release changes the fused evaluator's cost parameter ABI
from 48 to 57. Any bundled binary from an earlier version MUST be rebuilt.
No rebuilt binary is bundled here. Without MuJoCo and the rebuilt extension,
use `--allow-python-rollout` only for functional debugging; it is much slower.


### Basic reaching objective and distance-dependent exploration

The default Franka reaching objective is running Cartesian tip-to-goal distance,
terminal tip-to-goal distance, near-target Cartesian tip-velocity penalty, and
second-difference joint-command smoothing (plus the existing small control
deviation regularizer). The optional `reach-region-weight` and
`reach-settle-weight` are **zero by default**; they can be enabled explicitly.
The same objective is used in `joint` and `effector` sampling modes.

Exploration uses

```text
alpha(d) = alpha_min + (1 - alpha_min) * min(1, (d / dslow)**n)
```

where `n = --noise-decay-order` (default `1.0`), `dslow = --dslow`
(default `0.10` metres), and `alpha_min = --noise-min-fraction` (default `0.1`).
Larger `n` causes sharper decay as the tip approaches the target; smaller
positive `n` preserves more exploration near the goal. The noise is scaled
before effector-space inverse kinematics. Example:

```bash
python -m manipulation.franka_reaching.run --sampling spike \
  --noise-space effector --dslow 0.10 --noise-decay-order 2 \
  --noise-min-fraction 0.1
```

The optional legacy region/settling costs remain implemented in both the C++
and Python evaluators, but are disabled by default. Rebuild the native
extension after modifying cost evaluator sources.

### Replay video export

Export a replay as high-quality MP4 (H.264, RGB input, yuv420p output):

```bash
python -m manipulation.experiments.replay --file manipulation/results/last_run.npz --mp4 manipulation/results/standard_reaching.mp4 --speed 3
```

Optional `--fps 30 --width 960 --height 540` controls output settings. Existing
`--gif`, `--gif-fps`, `--gif-width`, and `--gif-height` options still work.
Install `imageio-ffmpeg` to enable MP4 output (`pip install imageio-ffmpeg`).
MP4 avoids GIF's 256-color limit, but differences between the live MuJoCo viewer
and offscreen renderer can still remain; H.264 is also lossy.
