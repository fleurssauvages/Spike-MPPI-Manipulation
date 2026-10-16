# Franka box pushing — dedicated long-finger tool

Launch (from the parent of `manipulation/`):

```bash
python -m manipulation.franka_pushing.run --sampling spike
```

- The pushing scene is `robots/assets/franka_emika_panda/franka_pushing_scene.xml`.
- It includes **`panda_pushing_custom.xml`**, rather than auto-generated
  `panda_slim.xml`. Asset fetching rewrites the latter but not the custom model.
- Straight capsule finger: from local Z=0.020 to 0.140 m (**120 mm**),
  tip sphere and `slim_tip_mid` at 0.146 m. Original segment was 60 mm.
- Box dimensions are **0.32 x 0.32 x 0.05 m** (XY footprint x height).
  The XML `geom size` stores half-extents: `0.160 0.160 0.025`.
  This is explicitly a flat box, *not* a 32 cm cube; for a cube set the third
  half-extent to `0.160`, relocate the initial box/goal centers to z=0.505,
  and verify robot clearance before simulating.
- Box mass and table position are unchanged.

## Default route: first two waypoints, back and forth

The default pushing route uses only the first two tabletop points:

- **A = (0.40, 0.00) m** (`--start-xy`)
- **B = (0.40, 0.18) m** (`--goal-xy`)

It performs **4 full A -> B -> A cycles** (8 arrivals), instead of the
former 2 square cycles. A waypoint counts as reached after the box remains
within `--goal-tolerance` for `--hold-seconds`, as before. The box begins at A,
so B is the first target. If the run reaches `--steps` before all four cycles,
it reports an incomplete route.

```bash
python -m manipulation.franka_pushing.run --sampling spike
```

To specify other endpoints use `--start-xy X Y --goal-xy X Y`.
The old routes remain available via `--route square`, `--route pingpong`,
and `--route point` (one-way A to B). You can override the repetition
count with `--cycles N`.

## Cost function (native and Python reference)

```text
J = mean_t(10.0 * ||box_center(t) - goal_center(t)||
           + 0.1 * tip_to_nearest_box_surface_distance(t))
```

Default `--control-deviation-weight 0` (optional regularizer disabled).
Other MPPI cost terms in mode 4 are disabled. The target remains the selected
waypoint, and the preferred approach side is **not** constrained.
Change the weights with `--box-goal-weight` and `--tip-box-weight`.

Changing the scene XML requires restarting the simulation only. Changing
`native/fused_mujoco.cpp` requires rebuilding the C++ extension:

```bash
python manipulation/setup_native.py build_ext --inplace
```

Default timing remains 100 Hz control, 5 ms MuJoCo physics, 2 substeps, and
`implicit` integration. A large 32 cm object may require replanning the
route or robot starting pose to avoid undesired initial contacts.

## Ball task compatibility

This archive also contains `franka_ball_bouncing/` and
`franka_ball_balancing/` as thin entry-point aliases for the original
`franka_bouncing/` and `franka_balancing/` implementations. Both command
forms run the same controller and cost functions.

## Reversing direction around the 32 cm box

A 20-step, 100 Hz MPPI horizon lasts only 0.2 s. Reversing the box direction
requires moving the fingertip across roughly 40 cm to the *opposite* side.
The requested pushing cost (10 x box-goal + 0.1 x tip-box distance) does
not reward disengaging from the previous contact and taking this long detour.

On each `backforth` target switch, a **separate, deterministic** three-stage
Cartesian Jacobian-IK transition is enabled by default:

1. Lift the finger above the box top by 12 cm.
2. Cross over it toward a point 4.5 cm behind the new side face.
3. Lower to the side face's center height; resume MPPI pushing.

This is deliberately not an MPPI rollout: it is the *same* transition behavior
for every noise sampler. Reposition steps are labelled `control_phase` in the
recording and excluded from average MPPI computation time. The box-goal and
tip-box cost weights during pushing are unchanged. The episode limit is now
80 simulated seconds (`--steps 8000`) to accommodate four round trips.

```bash
python -m manipulation.franka_pushing.run --sampling spike --verbose
python -m manipulation.franka_pushing.run --sampling spike --no-reposition  # pure MPPI baseline
```

Configurable: `--reposition-clearance 0.12`, `--reposition-standoff 0.045`,
`--reposition-timeout 9`. A failed IK transfer **stops** rather than allowing
a push from the wrong side. It may still fail if the side is outside the arm's
workspace or a link collides: check the MuJoCo viewer. No claim of a
collision-free, model-validated IK transfer is made.

For sampler-comparison experiments, disclose the deterministic supervisor and
report MPPI-only and full-episode performance separately.
