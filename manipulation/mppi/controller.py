from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any
import math
import time
import numpy as np

from .lbps import optimize_lbps_temperature
from .rollout import (
    RolloutCostConfig,
    NominalRollout,
    NativeRolloutBatcher,
    evaluate_control_batch,
    rollout_control_nominal,
    refine_control_nominal,
)
from .fast_kernels import NUMBA_AVAILABLE, lbps_optimize_fast
from .standard_kernels import standard_gaussian_numba, resolve_standard_sampler
from .pink_noise import sample_pink_noise
from .poisson_kernels import StaticPoissonSampler, resolve_poisson_sampler
from .spike_motor_pool import SpikeMotorPoolSampler, resolve_spike_sampler
from .spectrum_matched import (
    SpectrumMatchedGaussianSampler,
    fit_cross_spectral_factor,
    fit_independent_identical_spectral_factor,
)


_SPECTRUM_FACTOR_CACHE: dict[tuple[Any, ...], np.ndarray] = {}


class SamplingOption(str, Enum):
    """Candidate-generation strategy used by the MPPI controller."""

    STANDARD = "standard"
    PINK = "pink"
    GUIDED = "guided"
    DIAG_LOWRANK = "diag-lowrank"
    SPLINE = "spline"
    ICEM = "icem"
    GAUSSIAN_POISSON = "gaussian-poisson"
    POISSON = "poisson"
    GAUSSIAN_SPIKE = "gaussian-spike"
    SPIKE = "spike"


@dataclass
class ControllerConfig:
    control_dt: float = 0.01
    horizon: int = 100
    num_rollouts: int = 32
    lambda_temperature: float = 1.0
    adaptive_temperature_lbps: bool = True
    lbps_delta: float = 0.95
    lbps_optimizer_iterations: int = 32
    temporal_noise_smoothing: float = 0.25

    # Direct joint-space proposal.
    joint_noise_fraction: float = 0.15
    # Proposal coordinates: native actuator commands or Cartesian XYZ tip offsets.
    noise_space: str = "joint"
    effector_noise_std: float = 0.02  # metres per Cartesian axis
    ik_rcond: float = 1e-4
    noise_min_fraction: float = 0.1
    dslow: float = 0.10
    noise_decay_order: float = 1.0
    adaptive_noise: bool = True  # False preserves proposal covariance at every distance.
    reach_terminal_weight: float = 0.0
    reach_velocity_weight: float = 0.0
    reach_region_weight: float = 0.0
    reach_settle_weight: float = 0.0
    reach_velocity_ref: float = 0.1
    reach_velocity_sigma: float = 0.05
    reach_safe_radius: float = 0.015
    reach_settle_steps: int = 30
    reach_acceleration_weight: float = 0.0
    unlimited_control_span: float = 1.0

    # Alternative sampling proposals.  These only change how candidate
    # controls are generated; rollout physics, objective, LBPS and the MPPI
    # weighted update remain identical to standard MPPI.
    guided_rank: int = 6
    guided_fraction: float = 0.50
    diag_lowrank_rate: float = 0.08
    diag_lowrank_min: float = 0.25
    diag_lowrank_max: float = 4.0
    spline_modes: int = 6
    icem_elites: int = 4

    # Shared marked-Poisson/twitch physiology. ``poisson`` uses one independent
    # point-process drive per original joint/motor pool and an ordered motor-unit
    # recruitment mark. On Ant-2, event sign selects the agonist/antagonist
    # muscle before twitch filtering.
    # Event timing remains fixed Poisson; only the event mark selects recruitment
    # depth through the size-principle-inspired pool.
    base_rate_hz: float = 8.0
    recruitment_levels: int = 6
    standard_sampler: str = "numba"
    poisson_sampler: str = "numba"
    spike_sampler: str = "numba"
    # Number of source trajectories used once (then cached) to estimate the
    # finite-horizon cross-spectrum of the Poisson/Spike Gaussian controls.
    spectrum_match_calibration_samples: int = 4096
    # ``variance`` keeps Poisson-MPPI exploration power matched to standard
    # Gaussian MPPI. ``peak`` makes one isolated full-recruitment twitch peak
    # at exactly the configured per-joint noise scale. Overlapping events can
    # still sum above that level, so ``peak`` is not a hard noise bound.
    proposal_scale_mode: str = "variance"


    twitch_rise_s: float = 0.016
    twitch_decay_s: float = 0.064
    twitch_duration_s: float = 0.200

    # Spike proposal-shape parameters.
    # Physiological constants (recruitment thresholds, firing-rate range,
    # refractory/renewal statistics, hysteresis and twitch heterogeneity) stay
    # fixed; these two parameters are the intended HPO surface.
    spike_drive_sigma: float = 0.20
    spike_drive_tau_s: float = 0.05

    # Optional full-horizon multi-fidelity screening for the five paper-facing
    # Standard / spectrum-matched Gaussian / Poisson / Spike comparisons.
    # Both values at zero recover the unscreened path.
    # When enabled, ``screen_pool_size`` candidates are ranked with the existing
    # full-horizon RK4 preview and only ``num_rollouts`` are evaluated
    # at full fidelity.
    screen_pool_size: int = 0
    screen_exploit: int = 0
    # Optional diagnostic audit cadence for screened candidate populations.
    # 0 disables the expensive full-population audit; N>0 audits every N MPC steps.
    screen_audit_every: int = 0

    # Optional local iLQR-style nominal refinement.
    nominal_refine_iterations: int = 0
    nominal_refine_damping: float = 1e-4
    nominal_refine_step_size: float = 0.35
    nominal_max_step_fraction: float = 0.15
    sensitivity_epsilon_fraction: float = 1e-3


    # Rollout constraints/cost.  A real fall requires BOTH an inverted
    # torso (root_up < min_root_up) and torso-ground contact. fall_height_fraction
    # is retained only for the state-only rollout fallback, where contact pairs
    # are unavailable and low root height is used as a conservative proxy.
    hard_collision_clearance: float = 0.02
    # Exact fused fall rule: root_up < min_root_up AND torso touches ground.
    # fall_height_fraction is used only by the state-only fallback as a contact proxy.
    fall_height_fraction: float = 0.45
    min_root_up: float = 0.0
    upright_weight: float = 0.05
    control_deviation_weight: float = 1e-4
    command_derivative_weight: float = 0.0

    # Original Ant push-box shaping. On the fixed-base Franka the fingertip
    # midpoint replaces the Ant root for coupled progress and approach.
    box_progress_weight: float = 1.0
    robot_progress_weight: float = 0.35
    robot_box_approach_weight: float = 1.00
    # Keep pushing in the quasi-planar regime. Ballistic/tipped box candidates
    # are not useful solutions to a ground-pushing task.
    box_max_lift: float = 0.035
    box_min_up: float = 0.85
    box_lift_weight: float = 0.0

    # Franka additions to the original push objective.  In this task only
    # overshoot and undesirable-contact penalties are enabled by default.
    target_error_weight: float = 0.0
    terminal_target_weight: float = 0.0
    overshoot_weight: float = 20.0
    terminal_overshoot_weight: float = 80.0
    target_velocity_weight: float = 0.0
    target_tolerance: float = 0.005
    non_tip_contact_weight: float = 5.0
    tip_contact_reward: float = 0.0
    tip_push_reward: float = 0.0
    table_contact_weight: float = 50.0
    box_speed_weight: float = 0.0
    preferred_box_speed: float = 0.05
    terminal_velocity_weight: float = 0.0
    rollout_workers: int = 16
    rollout_chunk_size: int = 0

    # Standard receding-horizon warm start. The first update starts from the
    # robot's neutral/default control (zero for the classic Ant); later updates
    # shift and reuse the previous MPPI solution.
    warm_start: bool = True
    # Nominal geometry is diagnostic-only unless nominal refinement is enabled.
    # Keep API compatibility; retained for API compatibility.
    nominal_diagnostics: bool = True

    def __post_init__(self) -> None:
        self.horizon = max(1, int(self.horizon))
        self.num_rollouts = max(1, int(self.num_rollouts))
        if not 0 <= self.noise_min_fraction <= 1 or self.dslow <= 0 or self.noise_decay_order <= 0:
            raise ValueError("noise_min_fraction must be in [0, 1]; dslow and noise_decay_order must be positive")
        if self.reach_velocity_ref <= 0 or self.reach_velocity_sigma <= 0 or self.reach_safe_radius <= 0:
            raise ValueError("stabilization distance/velocity scales must be positive")
        if self.reach_settle_steps < 1:
            raise ValueError("reach_settle_steps must be positive")
        if any(x < 0 for x in (self.reach_terminal_weight, self.reach_velocity_weight,
                              self.reach_region_weight, self.reach_settle_weight,
                              self.reach_acceleration_weight)):
            raise ValueError("stabilization weights must be nonnegative")
        self.lbps_optimizer_iterations = max(8, int(self.lbps_optimizer_iterations))
        if self.control_dt <= 0.0:
            raise ValueError("control_dt must be positive")
        if self.lambda_temperature <= 0.0:
            raise ValueError("lambda_temperature must be positive")
        if not 0.0 < self.lbps_delta < 1.0:
            raise ValueError("lbps_delta must lie in (0, 1)")
        if not 0.0 <= self.temporal_noise_smoothing < 1.0:
            raise ValueError("temporal_noise_smoothing must lie in [0, 1)")
        if self.noise_space not in {"joint", "effector"}:
            raise ValueError("noise_space must be 'joint' or 'effector'")
        if not np.isfinite(self.effector_noise_std) or self.effector_noise_std <= 0.0:
            raise ValueError("effector_noise_std must be finite and positive (metres)")
        if not np.isfinite(self.ik_rcond) or self.ik_rcond <= 0.0:
            raise ValueError("ik_rcond must be finite and positive")
        if self.sensitivity_epsilon_fraction <= 0.0:
            raise ValueError("sensitivity_epsilon_fraction must be positive")
        self.guided_rank = max(1, int(self.guided_rank))
        if not 0.0 <= self.guided_fraction < 1.0:
            raise ValueError("guided_fraction must lie in [0, 1)")
        if not 0.0 < self.diag_lowrank_rate <= 1.0:
            raise ValueError("diag_lowrank_rate must lie in (0, 1]")
        if not 0.0 < self.diag_lowrank_min <= 1.0:
            raise ValueError("diag_lowrank_min must lie in (0, 1]")
        if self.diag_lowrank_max < 1.0:
            raise ValueError("diag_lowrank_max must be >= 1")
        if self.diag_lowrank_max < self.diag_lowrank_min:
            raise ValueError("diag_lowrank_max must be >= diag_lowrank_min")
        self.spline_modes = max(4, int(self.spline_modes))
        self.icem_elites = max(0, int(self.icem_elites))
        if self.standard_sampler not in {"auto", "numpy", "numba"}:
            raise ValueError("standard_sampler must be auto, numpy, or numba")
        if self.poisson_sampler not in {"auto", "numpy", "numba"}:
            raise ValueError("poisson_sampler must be auto, numpy, or numba")
        if self.spike_sampler not in {"auto", "numpy", "numba"}:
            raise ValueError("spike_sampler must be auto, numpy, or numba")
        self.spectrum_match_calibration_samples = int(self.spectrum_match_calibration_samples)
        if self.spectrum_match_calibration_samples < 64:
            raise ValueError("spectrum_match_calibration_samples must be >= 64")
        if self.proposal_scale_mode not in {"variance", "peak"}:
            raise ValueError("proposal_scale_mode must be variance or peak")
        if not np.isfinite(self.base_rate_hz) or self.base_rate_hz <= 0:
            raise ValueError("base_rate_hz must be finite and positive")
        if not np.isfinite(self.spike_drive_sigma) or self.spike_drive_sigma <= 0.0:
            raise ValueError("spike_drive_sigma must be finite and positive")
        if not np.isfinite(self.spike_drive_tau_s) or self.spike_drive_tau_s <= 0.0:
            raise ValueError("spike_drive_tau_s must be finite and positive")
        self.recruitment_levels = max(1, int(self.recruitment_levels))
        if self.twitch_rise_s <= 0.0:
            raise ValueError("twitch_rise_s must be positive")
        if self.twitch_decay_s <= self.twitch_rise_s:
            raise ValueError("twitch_decay_s must be greater than twitch_rise_s")
        if self.twitch_duration_s <= 0.0:
            raise ValueError("twitch_duration_s must be positive")
        self.screen_pool_size = max(0, int(self.screen_pool_size))
        self.screen_exploit = max(0, int(self.screen_exploit))
        self.screen_audit_every = max(0, int(self.screen_audit_every))
        if (self.screen_pool_size == 0) != (self.screen_exploit == 0):
            raise ValueError("screen_pool_size and screen_exploit must both be zero or both be positive")
        if self.screen_pool_size > 0 and self.screen_pool_size < self.num_rollouts:
            raise ValueError("screen_pool_size must be >= num_rollouts when screening is enabled")
        if self.screen_exploit > self.num_rollouts:
            raise ValueError("screen_exploit must be <= num_rollouts")
        if self.screen_audit_every > 0 and self.screen_pool_size == 0:
            raise ValueError("screen_audit_every requires screening (--screen-pool/--screen-exploit)")
        if self.box_progress_weight < 0.0 or self.robot_progress_weight < 0.0 or self.robot_box_approach_weight < 0.0:
            raise ValueError("push-task shaping weights must be nonnegative")
        for name in ("control_deviation_weight", "command_derivative_weight"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "target_error_weight", "terminal_target_weight", "overshoot_weight",
            "terminal_overshoot_weight", "target_velocity_weight", "non_tip_contact_weight",
            "tip_contact_reward", "tip_push_reward", "table_contact_weight",
            "box_speed_weight", "terminal_velocity_weight", "box_lift_weight",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.target_tolerance) or self.target_tolerance <= 0.0:
            raise ValueError("target_tolerance must be finite and positive")
        if not np.isfinite(self.preferred_box_speed) or self.preferred_box_speed <= 0.0:
            raise ValueError("preferred_box_speed must be finite and positive")
        if self.box_max_lift < 0.0:
            raise ValueError("box_max_lift must be nonnegative")
        if not -1.0 <= self.box_min_up <= 1.0:
            raise ValueError("box_min_up must lie in [-1, 1]")
        self.rollout_chunk_size = max(0, int(self.rollout_chunk_size))

class JointMPPIController:
    """Native-MuJoCo direct-joint MPPI controller.

    The control vector is the full MuJoCo actuator vector::

        u_t == MjData.ctrl in R^(model.nu)

    The first update is centered on ``robot.default_ctrl`` when provided (zero
    for the classic Ant adapters). With warm start enabled, each later update
    shifts the previous optimized MPPI sequence forward by one step.
    """

    def __init__(
        self,
        robot,
        track,
        prior,
        cfg: ControllerConfig,
        *,
        sampling: SamplingOption | str = SamplingOption.STANDARD,
        seed: int = 1,
    ) -> None:
        self.robot = robot
        self.track = track
        self.prior = prior
        self.cfg = cfg
        self.sampling = SamplingOption(sampling)
        if cfg.noise_space == "effector":
            required = ("cartesian_jacobian", "rollout_arm_qpos_adr", "rollout_arm_actuator_ids", "new_data")
            missing = [name for name in required if not hasattr(robot, name)]
            if missing:
                raise ValueError(f"effector sampling requires robot IK support: {missing}")
            if len(robot.rollout_arm_actuator_ids) != robot.nu:
                raise ValueError("effector sampling currently requires all actuators to be arm position servos")
            if robot.nu < 3:
                raise ValueError("effector sampling requires at least three proposal channels")
        screen_requested = cfg.screen_pool_size > 0 and cfg.screen_exploit > 0
        screenable = {
            SamplingOption.STANDARD,
            SamplingOption.PINK,
            SamplingOption.GAUSSIAN_POISSON,
            SamplingOption.POISSON,
            SamplingOption.GAUSSIAN_SPIKE,
            SamplingOption.SPIKE,
        }
        self._screen_enabled = (
            self.sampling in screenable
            and screen_requested
        )
        if screen_requested and self.sampling not in screenable:
            raise ValueError(
                "--screen-pool/--screen-exploit are only valid with --sampling standard, pink, "
                "gaussian-poisson, poisson, gaussian-spike, or spike"
            )
        self.rng = np.random.default_rng(int(seed))
        ratio = float(cfg.control_dt) / max(float(robot.physics_dt), 1e-12)
        self.control_substeps = max(1, int(round(ratio)))
        actual = self.control_substeps * float(robot.physics_dt)
        if abs(actual - float(cfg.control_dt)) > 0.25 * float(robot.physics_dt):
            raise ValueError(
                f"control_dt={cfg.control_dt:g} is not close to an integer multiple of "
                f"MuJoCo timestep={robot.physics_dt:g}; nearest is {actual:g}."
            )
        self.cost_cfg = RolloutCostConfig(
            hard_collision_clearance=cfg.hard_collision_clearance,
            fall_height_fraction=cfg.fall_height_fraction,
            min_root_up=cfg.min_root_up,
            upright_weight=cfg.upright_weight,
            control_deviation_weight=cfg.control_deviation_weight,
            command_derivative_weight=cfg.command_derivative_weight,
            box_progress_weight=cfg.box_progress_weight,
            robot_progress_weight=cfg.robot_progress_weight,
            robot_box_approach_weight=cfg.robot_box_approach_weight,
            box_max_lift=cfg.box_max_lift,
            box_min_up=cfg.box_min_up,
            box_lift_weight=cfg.box_lift_weight,
            target_error_weight=cfg.target_error_weight,
            terminal_target_weight=cfg.terminal_target_weight,
            overshoot_weight=cfg.overshoot_weight,
            terminal_overshoot_weight=cfg.terminal_overshoot_weight,
            target_velocity_weight=cfg.target_velocity_weight,
            target_tolerance=cfg.target_tolerance,
            non_tip_contact_weight=cfg.non_tip_contact_weight,
            tip_contact_reward=cfg.tip_contact_reward,
            tip_push_reward=cfg.tip_push_reward,
            table_contact_weight=cfg.table_contact_weight,
            box_speed_weight=cfg.box_speed_weight,
            preferred_box_speed=cfg.preferred_box_speed,
            terminal_velocity_weight=cfg.terminal_velocity_weight,
            reach_terminal_weight=cfg.reach_terminal_weight,
            reach_velocity_weight=cfg.reach_velocity_weight,
            reach_region_weight=cfg.reach_region_weight,
            reach_settle_weight=cfg.reach_settle_weight,
            reach_velocity_ref=cfg.reach_velocity_ref,
            reach_velocity_sigma=cfg.reach_velocity_sigma,
            reach_safe_radius=cfg.reach_safe_radius,
            reach_settle_steps=cfg.reach_settle_steps,
            reach_acceleration_weight=cfg.reach_acceleration_weight,
        )
        self.rollout_batcher = None
        # The same persistent native batcher is reused for candidate evaluation
        # and warm-start nominal propagation.
        self.nominal_batcher = None
        # Backwards-compatible alias used by candidate rollout helper arguments.
        self.native_batcher = None
        self._previous_plan: np.ndarray | None = None
        self._weight_workspace = np.empty(cfg.num_rollouts, dtype=np.float64)
        self._ctrl_low, self._ctrl_high = robot.control_bounds(cfg.unlimited_control_span)
        # Actuator-scale normalization used only by benchmark/control-quality
        # diagnostics. Keep this independent of --joint-noise so normalized
        # residual/smoothness metrics remain comparable across samplers and
        # exploration-scale sweeps. This matches NativeRolloutBatcher.
        self._ctrl_scale = np.maximum(
            np.asarray(robot.control_scale(), dtype=np.float64), 1e-6
        )
        self._joint_std = robot.control_scale(
            fraction=cfg.joint_noise_fraction,
            unlimited_span=cfg.unlimited_control_span,
        )
        self._standard_workspace: tuple[np.ndarray, np.ndarray] | None = None
        if bool(getattr(robot, "is_muscle_model", False)):
            self._muscle_pos_idx = np.asarray([p for p, _ in robot.muscle_pairs], dtype=np.int32)
            self._muscle_neg_idx = np.asarray([n for _, n in robot.muscle_pairs], dtype=np.int32)
        else:
            self._muscle_pos_idx = np.empty(0, dtype=np.int32)
            self._muscle_neg_idx = np.empty(0, dtype=np.int32)
        self._direction_history: list[np.ndarray] = []
        self._last_proposal_rank = 0
        self._diag_variance = np.ones((cfg.horizon, robot.nu), dtype=np.float64)
        self._icem_elites: np.ndarray | None = None
        self._spline_basis = self._make_bspline_basis(cfg.horizon, cfg.spline_modes)
        self._standard_sampler_backend = (
            resolve_standard_sampler(cfg.standard_sampler)
            if self.sampling == SamplingOption.STANDARD
            else "unused"
        )
        self._poisson_sampler_backend = (
            resolve_poisson_sampler(cfg.poisson_sampler)
            if self.sampling in {SamplingOption.POISSON, SamplingOption.GAUSSIAN_POISSON}
            else "unused"
        )
        self._spike_sampler_backend = (
            resolve_spike_sampler(cfg.spike_sampler)
            if self.sampling in {SamplingOption.SPIKE, SamplingOption.GAUSSIAN_SPIKE}
            else "unused"
        )
        (
            self._poisson_levels,
            self._poisson_level_prior,
            self._poisson_motor_unit_strengths,
        ) = self._make_recruitment_marks(cfg.recruitment_levels)
        self._poisson_twitch = self._make_poisson_twitch_kernel(
            cfg.control_dt,
            cfg.twitch_rise_s,
            cfg.twitch_decay_s,
            cfg.twitch_duration_s,
        )
        h = int(cfg.horizon)
        # ``poisson`` keeps one point-process channel per original Ant joint. On
        # Ant-2, positive and negative events are decoded *before filtering*
        # into separate agonist/antagonist muscle channels, giving a native 16-D
        # physical control trajectory while preserving the original event budget.
        m = int(getattr(robot, "motor_pool_count", robot.nu))
        self._poisson_channels = np.eye(m, dtype=np.float64)
        poisson_shape = (h, m)
        self._poisson_total_rate_hz = float(m) * float(cfg.base_rate_hz)
        per_channel_rate = self._poisson_total_rate_hz / float(m)
        half_base = 0.5 * per_channel_rate
        self._poisson_pos_rate_map = np.full(poisson_shape, half_base, dtype=np.float64)
        self._poisson_neg_rate_map = np.full(poisson_shape, half_base, dtype=np.float64)
        self._poisson_mark_prob_map = np.broadcast_to(
            self._poisson_level_prior, poisson_shape + (len(self._poisson_levels),)
        ).copy()
        for array in (self._poisson_pos_rate_map, self._poisson_neg_rate_map,
                      self._poisson_mark_prob_map):
            array.flags.writeable = False
        self._last_event_count_mean = 0.0
        k2_prefix = np.cumsum(self._poisson_twitch ** 2)
        remaining = np.minimum(np.arange(h, 0, -1), len(k2_prefix))
        self._poisson_variance_weights = k2_prefix[remaining - 1] / float(h)
        self._poisson_variance_match_scale = self._compute_poisson_global_noise_scale()
        self._poisson_global_noise_scale = (
            1.0
            if cfg.proposal_scale_mode == "peak"
            else self._poisson_variance_match_scale
        )
        self._poisson_generator = None
        self._spike_generator = None
        self._matched_gaussian_generator: SpectrumMatchedGaussianSampler | None = None
        self._spike_last_stats: dict[str, float] = {}
        self._spike_variance_match_scale = math.nan
        self._spike_global_noise_scale = math.nan
        proposal_n = cfg.screen_pool_size if self._screen_enabled else cfg.num_rollouts
        if self.sampling == SamplingOption.POISSON:
            self._poisson_generator = StaticPoissonSampler(
                self._poisson_pos_rate_map, self._poisson_neg_rate_map,
                self._poisson_mark_prob_map, self._poisson_levels,
                cfg.control_dt, proposal_n, self._poisson_twitch,
                backend=self._poisson_sampler_backend,
            )
        if self.sampling in {SamplingOption.SPIKE, SamplingOption.GAUSSIAN_SPIKE}:
            spike_scale_source = SpikeMotorPoolSampler(
                n=proposal_n if self.sampling == SamplingOption.SPIKE else 1,
                h=h, nu=int(getattr(robot, "motor_pool_count", robot.nu)),
                dt=cfg.control_dt,
                levels=cfg.recruitment_levels,
                base_rate_hz=cfg.base_rate_hz,
                twitch_rise_s=cfg.twitch_rise_s,
                twitch_decay_s=cfg.twitch_decay_s,
                twitch_duration_s=cfg.twitch_duration_s,
                drive_sigma=cfg.spike_drive_sigma,
                drive_tau_s=cfg.spike_drive_tau_s,
                separate_antagonists=bool(getattr(robot, "is_muscle_model", False)),
                backend=self._spike_sampler_backend,
            )
            if self.sampling == SamplingOption.SPIKE:
                self._spike_generator = spike_scale_source
            spike_rms = spike_scale_source.estimate_unit_rms()
            self._spike_variance_match_scale = 1.0 / max(spike_rms, 1e-12)
            self._spike_global_noise_scale = (
                1.0 if cfg.proposal_scale_mode == "peak" else self._spike_variance_match_scale
            )
        if self.sampling == SamplingOption.GAUSSIAN_POISSON:
            self._matched_gaussian_generator = self._make_spectrum_matched_gaussian("poisson")
        elif self.sampling == SamplingOption.GAUSSIAN_SPIKE:
            self._matched_gaussian_generator = self._make_spectrum_matched_gaussian("spike")
        self._screen_step = 0
        self._last_screen_diag: dict[str, float] = {}

        self._fixed_poisson_diagnostics = self._make_fixed_poisson_diagnostics()

        # Online MPPI is CPU-only. Prefer the allocation-light fused C++
        # evaluator, then stock mujoco.rollout, then the Python fallback.
        try:
            self.rollout_batcher = NativeRolloutBatcher(
                robot, workers=cfg.rollout_workers,
                batch_hint=max(cfg.num_rollouts, cfg.screen_pool_size if self._screen_enabled else cfg.num_rollouts),
                chunk_size=cfg.rollout_chunk_size,
                # All tasks have native cost evaluators. Ball tasks use the
                # separate fused ball evaluator, preserving legacy task modes.
                fused=True,
                screening=self._screen_enabled,
            )
        except Exception:
            if self._screen_enabled:
                raise
            try:
                self.rollout_batcher = NativeRolloutBatcher(
                    robot, workers=cfg.rollout_workers, batch_hint=cfg.num_rollouts,
                    chunk_size=cfg.rollout_chunk_size, fused=False,
                )
            except Exception:
                self.rollout_batcher = None
        self.nominal_batcher = self.rollout_batcher
        self.native_batcher = self.rollout_batcher

    def _make_spectrum_matched_gaussian(
        self, source: str
    ) -> SpectrumMatchedGaussianSampler:
        """Build a Gaussian sampler matched to the source proposal spectrum.

        The fit is deterministic and independent of the controller RNG stream.
        Only one original joint/motor-pool group is calibrated because groups
        are identically distributed and independent in both source proposals.
        For Ant-2 the calibrated group contains the agonist/antagonist pair.
        Spike retains its fitted 2x2 cross-spectrum. Poisson instead enforces
        the model-implied independence of its positive/negative streams: the two
        channels share one pooled marginal spectrum and have exactly zero target
        cross-spectrum.
        """
        source = str(source).strip().lower()
        if source not in {"poisson", "spike"}:
            raise ValueError("source must be poisson or spike")

        h = int(self.cfg.horizon)
        is_muscle = bool(getattr(self.robot, "is_muscle_model", False))
        group_size = 2 if is_muscle else 1
        groups = int(getattr(self.robot, "motor_pool_count", self.robot.nu))
        calibration_n = int(self.cfg.spectrum_match_calibration_samples)
        backend = (
            self._poisson_sampler_backend if source == "poisson"
            else self._spike_sampler_backend
        )
        source_scale = (
            float(self._poisson_global_noise_scale) if source == "poisson"
            else float(self._spike_global_noise_scale)
        )
        cache_key = (
            source,
            h,
            float(self.cfg.control_dt),
            float(self.cfg.base_rate_hz),
            int(self.cfg.recruitment_levels),
            float(self.cfg.twitch_rise_s),
            float(self.cfg.twitch_decay_s),
            float(self.cfg.twitch_duration_s),
            float(self.cfg.spike_drive_sigma) if source == "spike" else None,
            float(self.cfg.spike_drive_tau_s) if source == "spike" else None,
            bool(is_muscle),
            str(backend),
            str(self.cfg.proposal_scale_mode),
            float(source_scale),
            calibration_n,
        )
        factor = _SPECTRUM_FACTOR_CACHE.get(cache_key)
        if factor is None:
            # Fixed seeds ensure calibration does not consume or couple to the
            # online MPPI random stream and is identical across matched trials.
            rng = np.random.default_rng(731921 if source == "poisson" else 731923)
            if source == "poisson":
                one_shape = (h, 1)
                pos = np.full(one_shape, 0.5 * float(self.cfg.base_rate_hz), dtype=np.float64)
                neg = np.full(one_shape, 0.5 * float(self.cfg.base_rate_hz), dtype=np.float64)
                mark_prob = np.broadcast_to(
                    self._poisson_level_prior,
                    one_shape + (len(self._poisson_levels),),
                ).copy()
                generator = StaticPoissonSampler(
                    pos,
                    neg,
                    mark_prob,
                    self._poisson_levels,
                    float(self.cfg.control_dt),
                    calibration_n,
                    self._poisson_twitch,
                    backend=backend,
                )
                if is_muscle:
                    source_samples, _ = generator.sample_projected_antagonistic_pairs(rng)
                else:
                    source_samples, _ = generator.sample_projected_identity(rng)
            else:
                generator = SpikeMotorPoolSampler(
                    n=calibration_n,
                    h=h,
                    nu=1,
                    dt=float(self.cfg.control_dt),
                    levels=int(self.cfg.recruitment_levels),
                    base_rate_hz=float(self.cfg.base_rate_hz),
                    twitch_rise_s=float(self.cfg.twitch_rise_s),
                    twitch_decay_s=float(self.cfg.twitch_decay_s),
                    twitch_duration_s=float(self.cfg.twitch_duration_s),
                    drive_sigma=float(self.cfg.spike_drive_sigma),
                    drive_tau_s=float(self.cfg.spike_drive_tau_s),
                    separate_antagonists=is_muscle,
                    backend=backend,
                )
                source_samples, _ = generator.sample(rng)

            source_samples = np.asarray(source_samples, dtype=np.float64).copy()
            source_samples *= source_scale
            if source == "poisson" and is_muscle:
                # The two physical channels are independent, identically
                # distributed thinnings of the symmetric signed Poisson law.
                # Pool them to estimate one marginal PSD and impose an exactly
                # diagonal target spectrum. This prevents finite calibration
                # noise from creating a spurious agonist/antagonist correlation.
                factor = fit_independent_identical_spectral_factor(source_samples)
            else:
                factor = fit_cross_spectral_factor(source_samples)
            factor.flags.writeable = False
            _SPECTRUM_FACTOR_CACHE[cache_key] = factor

        sampler = SpectrumMatchedGaussianSampler(
            horizon=h,
            groups=groups,
            spectral_factor=factor,
            center_across_rollouts=is_muscle,
        )
        if sampler.group_size != group_size or sampler.nu != int(self.robot.nu):
            raise RuntimeError(
                "spectrum-matched Gaussian dimensions do not match the robot actuator space"
            )
        return sampler

    @property
    def rollout_backend_name(self) -> str:
        batcher = self.rollout_batcher
        if batcher is None:
            return "cpu/python"
        if batcher.supports_vectorized_cost:
            suffix = f"/chunk{batcher.chunk_size}" if batcher.chunk_size > 0 else ""
            prefix = "cpu/fused" if batcher.uses_fused else "cpu/native"
            return f"{prefix}/{batcher.nthread}t{suffix}"
        return "cpu/python"

    @property
    def sampling_description(self) -> str:
        if self.sampling == SamplingOption.GUIDED:
            return (
                f"guided low-rank: history_rank={self.cfg.guided_rank}, "
                f"guided_fraction={self.cfg.guided_fraction:g}, baseline=standard MPPI"
            )
        if self.sampling == SamplingOption.DIAG_LOWRANK:
            return (
                f"diagonal + low-rank: history_rank={self.cfg.guided_rank}, "
                f"guided_fraction={self.cfg.guided_fraction:g}, "
                f"diag_rate={self.cfg.diag_lowrank_rate:g}"
            )
        if self.sampling == SamplingOption.SPLINE:
            return f"spline latent sampling: cubic B-spline modes={self._spline_basis.shape[1]} per joint"
        if self.sampling == SamplingOption.ICEM:
            return f"iCEM-style elite reuse: shifted_elites={self.cfg.icem_elites}"
        if self.sampling == SamplingOption.GAUSSIAN_POISSON:
            screening = (
                f", screening pool={self.cfg.screen_pool_size}"
                if self._screen_enabled else ""
            )
            return (
                "Gaussian spectrum control matched to Poisson: Gaussian perturbations with the "
                "finite-horizon Poisson power spectrum"
                + ("; agonist/antagonist marginals pooled and cross-spectrum fixed to zero" if bool(getattr(self.robot, "is_muscle_model", False)) else "")
                + f"; calibration={self.cfg.spectrum_match_calibration_samples}, runtime=numpy-fft{screening}"
            )
        if self.sampling == SamplingOption.POISSON:
            if self._screen_enabled:
                return (
                    "Poisson-MPPI with optional full-horizon screening: fixed direct-joint "
                    "Poisson events + ordered motor-unit recruitment + causal twitch; "
                    f"Npool={self.cfg.screen_pool_size}, preview_H={self.cfg.horizon}, "
                    f"select={self.cfg.screen_exploit}+{self.cfg.num_rollouts-self.cfg.screen_exploit}, "
                    f"scale={self.cfg.proposal_scale_mode}, preview=RK4, full update unchanged, "
                    f"implementation={self._poisson_sampler_backend}"
                )
            poisson_channels = int(self._poisson_channels.shape[0])
            decode = (
                f"sign-split to {self.robot.nu} antagonistic muscle controls"
                if bool(getattr(self.robot, "is_muscle_model", False))
                else f"identity decode to {self.robot.nu} controls"
            )
            return (
                "Poisson-MPPI: fixed Poisson events + ordered motor-unit recruitment + causal twitch; "
                f"event_channels={poisson_channels}, {decode}, motor_units={len(self._poisson_levels)}, "
                f"base_rate={self.cfg.base_rate_hz:g}Hz, scale={self.cfg.proposal_scale_mode}, "
                f"no learning, implementation={self._poisson_sampler_backend}"
            )
        if self.sampling == SamplingOption.GAUSSIAN_SPIKE:
            screening = (
                f", screening pool={self.cfg.screen_pool_size}"
                if self._screen_enabled else ""
            )
            return (
                "Gaussian spectrum control matched to Spike: Gaussian perturbations with the "
                "finite-horizon Spike power spectrum"
                + (" and agonist/antagonist cross-spectrum" if bool(getattr(self.robot, "is_muscle_model", False)) else "")
                + f"; calibration={self.cfg.spectrum_match_calibration_samples}, runtime=numpy-fft{screening}"
            )
        if self.sampling == SamplingOption.SPIKE:
            pool = self._spike_generator
            return (
                "Spike-MPPI: antagonistic motor pools + common drive + ordered threshold recruitment + "
                "rate coding + refractory gamma-renewal firing + heterogeneous twitches + hysteresis; "
                f"motor_units={self.cfg.recruitment_levels}/pool, min_rate={self.cfg.base_rate_hz:g}Hz, "
                f"max_rate={pool.max_rate_hz:g}Hz, scale={self.cfg.proposal_scale_mode}, implementation={self._spike_sampler_backend}"
            )
        if self.sampling == SamplingOption.PINK:
            screening = (
                f"; screening pool={self.cfg.screen_pool_size}"
                if self._screen_enabled else ""
            )
            return "pink Gaussian MPPI sampling: 1/f temporal power, independent channels, unit marginal variance; runtime=numpy-fft" + screening
        if self.sampling == SamplingOption.STANDARD and self._screen_enabled:
            return (
                "standard MPPI Gaussian sampling with full-horizon screening: "
                f"Npool={self.cfg.screen_pool_size}, preview_H={self.cfg.horizon}, "
                f"select={self.cfg.screen_exploit}+{self.cfg.num_rollouts-self.cfg.screen_exploit}, "
                f"preview=RK4, full update unchanged, implementation={self._standard_sampler_backend}"
            )
        return f"standard MPPI Gaussian sampling; implementation={self._standard_sampler_backend}"

    @property
    def sampling_backend_name(self) -> str:
        if self.sampling == SamplingOption.PINK:
            return "numpy-fft"
        if self.sampling == SamplingOption.STANDARD:
            return self._standard_sampler_backend
        if self.sampling in {SamplingOption.GAUSSIAN_POISSON, SamplingOption.GAUSSIAN_SPIKE}:
            return "numpy-fft"
        if self.sampling == SamplingOption.POISSON:
            return self._poisson_sampler_backend
        if self.sampling == SamplingOption.SPIKE:
            return self._spike_sampler_backend
        return "numpy"

    @property
    def proposal_description(self) -> str:
        """Backward-compatible alias for the former proposal terminology."""
        return self.sampling_description

    def sync_planning_model(self) -> None:
        """Synchronize explicit planner-model changes into active backends."""
        seen = set()
        for batcher in (self.rollout_batcher, self.nominal_batcher):
            if batcher is None or id(batcher) in seen:
                continue
            seen.add(id(batcher))
            if hasattr(batcher, "sync_model"):
                batcher.sync_model()
            elif hasattr(batcher, "sync_fused_model"):
                batcher.sync_fused_model()

    def reset_warm_start(self) -> None:
        """Discard the shifted MPPI plan before a discontinuous task retarget."""
        self._previous_plan = None

    def close(self) -> None:
        """Release rollout resources explicitly."""
        seen = set()
        for batcher in (self.rollout_batcher, self.nominal_batcher):
            if batcher is None or id(batcher) in seen:
                continue
            seen.add(id(batcher))
            batcher.close()
        self.rollout_batcher = None
        self.nominal_batcher = None
        self.native_batcher = None

    def _build_nominal(self, data, current_s: float):
        """Build the MPPI proposal center from zero or the shifted prior plan.

        The first update starts from ``robot.default_ctrl`` when the robot adapter
        provides one, otherwise from zero.  This keeps torque-controlled Ant
        behavior unchanged while allowing position-controlled manipulators to
        start from a meaningful hold command.  With warm start enabled,
        subsequent updates shift the previous optimized MPPI sequence and use
        that sequence as the proposal center.
        """
        t0 = time.perf_counter()
        start = self.robot.snapshot(data)
        used_warm_start = bool(
            self.cfg.warm_start
            and self._previous_plan is not None
            and self._previous_plan.shape == (self.cfg.horizon, self.robot.nu)
        )

        if used_warm_start:
            base_controls = np.empty_like(self._previous_plan)
            base_controls[:-1] = self._previous_plan[1:]
            base_controls[-1] = self._previous_plan[-1]
        else:
            if hasattr(self.robot, "nominal_ctrl"):
                neutral = np.asarray(self.robot.nominal_ctrl(data), dtype=np.float64).reshape(-1)
            else:
                neutral = np.asarray(
                    getattr(self.robot, "default_ctrl", np.zeros(self.robot.nu, dtype=np.float64)),
                    dtype=np.float64,
                ).reshape(-1)
            if neutral.shape != (self.robot.nu,):
                raise ValueError(
                    f"robot nominal control must have shape ({self.robot.nu},), got {neutral.shape}"
                )
            base_controls = np.broadcast_to(
                neutral, (self.cfg.horizon, self.robot.nu)
            ).copy()
        base_controls = self.robot.clip_ctrl(base_controls)

        need_geometry = bool(self.cfg.nominal_diagnostics or (used_warm_start and int(self.cfg.nominal_refine_iterations) > 0))
        if need_geometry:
            base_rollout = rollout_control_nominal(
                self.robot, start, base_controls, self.track, current_s,
                control_substeps=self.control_substeps, native_batcher=self.nominal_batcher,
            )
        else:
            base_rollout = NominalRollout(
                base_controls, [], np.empty((0, 2)), np.empty(0), np.empty(0)
            )
        t_base = time.perf_counter()

        refined = base_rollout
        if used_warm_start and int(self.cfg.nominal_refine_iterations) > 0:
            refined, _jac, _endpoints = refine_control_nominal(
                self.robot,
                start,
                base_rollout,
                self.track,
                self.prior,
                control_substeps=self.control_substeps,
                lookahead_steps=3,
                iterations=int(self.cfg.nominal_refine_iterations),
                damping=self.cfg.nominal_refine_damping,
                step_size=self.cfg.nominal_refine_step_size,
                max_control_step_fraction=self.cfg.nominal_max_step_fraction,
                epsilon_fraction=self.cfg.sensitivity_epsilon_fraction,
                native_batcher=self.nominal_batcher,
            )
        t_refine = time.perf_counter()

        endpoints = np.asarray(refined.positions, dtype=np.float64)
        if len(endpoints):
            endpoint_s, _ = self.track.project(endpoints)
            prior_mean, prior_cov = self.prior.sample(self.track, endpoint_s)
        else:
            prior_mean = np.empty((0, 2), dtype=np.float64)
            prior_cov = np.empty((0, 2, 2), dtype=np.float64)
        t_prior = time.perf_counter()

        nominal_timing_ms = {
            "warm_start": 1e3 * (t_base - t0) if used_warm_start else 0.0,
            "zero_init": 1e3 * (t_base - t0) if not used_warm_start else 0.0,
            "sensitivity": 1e3 * (t_refine - t_base),
            "prior": 1e3 * (t_prior - t_refine),
        }
        return (
            start,
            base_rollout,
            refined,
            None,
            endpoints,
            np.asarray(prior_mean),
            np.asarray(prior_cov),
            nominal_timing_ms,
        )

    def _convert_proposals_to_effector_space(
        self, proposals: np.ndarray, nominal: np.ndarray, data,
    ) -> np.ndarray:
        """Decode three proposal channels into XYZ offsets using differential IK.

        Preserve the sampler's temporal statistics in the first three channels,
        but interpret them as Cartesian perturbations rather than joint commands.
        The Jacobian is recomputed along the nominal *joint-position* sequence.
        MuJoCo rollout controls, costs and the MPPI update remain joint-space.
        """
        if self.cfg.noise_space == "joint":
            return proposals
        n, h, _ = proposals.shape
        arm_ids = np.asarray(self.robot.rollout_arm_actuator_ids, dtype=int)
        qadr = np.asarray(self.robot.rollout_arm_qpos_adr, dtype=int)
        reference = self.robot.new_data(self.robot.snapshot(data))
        std = np.maximum(self._joint_std[:3], 1e-12)
        # Native samplers have already applied their joint scale. Strip that
        # scale so --effector-noise controls the offsets in physical metres.
        xyz_noise = (proposals[:, :, :3] - nominal[None, :, :3]) / std
        xyz_noise *= float(self.cfg.effector_noise_std)
        result = np.broadcast_to(nominal[None, :, :], (n, h, self.robot.nu)).copy()
        for t in range(h):
            reference.qpos[qadr] = nominal[t, arm_ids]
            self.robot.mujoco.mj_forward(self.robot.model, reference)
            J = np.asarray(self.robot.cartesian_jacobian(reference), dtype=np.float64)
            if J.shape != (3, len(arm_ids)):
                raise ValueError(f"Cartesian Jacobian must have shape (3, {len(arm_ids)}), got {J.shape}")
            dq = xyz_noise[:, t, :] @ np.linalg.pinv(J, rcond=self.cfg.ik_rcond).T
            result[:, t, arm_ids] += dq
        np.clip(result, self._ctrl_low, self._ctrl_high, out=result)
        return result

    def _sample_standard(self, nominal: np.ndarray, *, count: int | None = None) -> np.ndarray:
        """Standard Gaussian MPPI in the robot's native actuator space.

        Ant-2 therefore uses 16 independent nonnegative muscle channels; no
        half-wave decoder or twitch filter is applied to the standard baseline.
        """
        n = self.cfg.num_rollouts if count is None else max(1, int(count))
        h = self.cfg.horizon
        nu = self.robot.nu
        rho = float(self.cfg.temporal_noise_smoothing)
        beta = math.sqrt(max(0.0, 1.0 - rho * rho))
        std = self._joint_std
        if (
            self._standard_workspace is None
            or self._standard_workspace[0].shape != (n, h, nu)
        ):
            self._standard_workspace = (
                np.empty((n, h, nu), dtype=np.float64),
                np.empty((n, h, nu), dtype=np.float64),
            )
        noise, controls = self._standard_workspace
        if self._standard_sampler_backend == "numba":
            standard_gaussian_numba(
                self.rng, nominal, std, self._ctrl_low, self._ctrl_high,
                rho, noise, controls,
            )
        else:
            self.rng.standard_normal(noise.shape, out=noise)
            np.multiply(noise, std[None, None, :], out=noise)
            for t in range(1, h):
                noise[:, t] = rho * noise[:, t - 1] + beta * noise[:, t]
            np.add(nominal[None, :, :], noise, out=controls)
            np.clip(controls, self._ctrl_low, self._ctrl_high, out=controls)
        return controls

    def _sample_pink(self, nominal: np.ndarray, count: int | None = None) -> np.ndarray:
        """Draw 1/f Gaussian perturbations with standard joint-noise scaling."""
        n = self.cfg.num_rollouts if count is None else max(1, int(count))
        noise = sample_pink_noise(self.rng, (n, self.cfg.horizon, self.robot.nu))
        noise *= self._joint_std[None, None, :]
        noise += nominal[None, :, :]
        np.clip(noise, self._ctrl_low, self._ctrl_high, out=noise)
        return noise

    @staticmethod
    def _make_bspline_basis(horizon: int, modes: int, degree: int = 3) -> np.ndarray:
        """Return a row-normalized clamped uniform B-spline basis.

        Row normalization keeps unit latent coefficient variance at roughly
        unit marginal variance in every control timestep, so ``joint_noise``
        retains the same scale interpretation as standard MPPI.
        """
        h = max(1, int(horizon))
        m = max(degree + 1, min(int(modes), h))
        n_internal = m - degree - 1
        if n_internal > 0:
            interior = np.linspace(0.0, 1.0, n_internal + 2, dtype=np.float64)[1:-1]
        else:
            interior = np.empty(0, dtype=np.float64)
        knots = np.concatenate((
            np.zeros(degree + 1, dtype=np.float64),
            interior,
            np.ones(degree + 1, dtype=np.float64),
        ))
        x = np.linspace(0.0, 1.0, h, dtype=np.float64)
        # Cox-de Boor recursion. There are m basis functions.
        basis = np.zeros((h, m), dtype=np.float64)
        for i in range(m):
            left, right = knots[i], knots[i + 1]
            basis[:, i] = ((x >= left) & (x < right)).astype(np.float64)
        basis[-1, -1] = 1.0
        for p in range(1, degree + 1):
            nxt = np.zeros_like(basis)
            for i in range(m):
                left_den = knots[i + p] - knots[i]
                if left_den > 0.0:
                    nxt[:, i] += ((x - knots[i]) / left_den) * basis[:, i]
                if i + 1 < m:
                    right_den = knots[i + p + 1] - knots[i + 1]
                    if right_den > 0.0:
                        nxt[:, i] += ((knots[i + p + 1] - x) / right_den) * basis[:, i + 1]
            basis = nxt
        basis[-1, :] = 0.0
        basis[-1, -1] = 1.0
        row_norm = np.sqrt(np.sum(basis * basis, axis=1, keepdims=True))
        basis /= np.maximum(row_norm, 1e-12)
        return basis

    def _temporal_filter_inplace(self, noise: np.ndarray) -> None:
        rho = float(self.cfg.temporal_noise_smoothing)
        beta = math.sqrt(max(0.0, 1.0 - rho * rho))
        for t in range(1, noise.shape[1]):
            noise[:, t] = rho * noise[:, t - 1] + beta * noise[:, t]

    def _history_basis(self) -> np.ndarray | None:
        """Orthonormal low-rank basis in normalized full-sequence space."""
        if not self._direction_history:
            return None
        d = self.cfg.horizon * self.robot.nu
        cols = []
        for direction in self._direction_history[-self.cfg.guided_rank:]:
            v = np.asarray(direction, dtype=np.float64).reshape(d)
            nrm = float(np.linalg.norm(v))
            if np.isfinite(nrm) and nrm > 1e-8:
                cols.append(v / nrm)
        if not cols:
            return None
        matrix = np.column_stack(cols)
        try:
            q, _ = np.linalg.qr(matrix, mode="reduced")
        except np.linalg.LinAlgError:
            return None
        if q.size == 0:
            return None
        return np.asarray(q[:, : min(q.shape[1], self.cfg.guided_rank)], dtype=np.float64)

    def _sample_guided(self, nominal: np.ndarray, *, adaptive_diagonal: bool) -> np.ndarray:
        n, h, nu = self.cfg.num_rollouts, self.cfg.horizon, self.robot.nu
        z = self.rng.standard_normal((n, h, nu))
        if adaptive_diagonal:
            z *= np.sqrt(np.maximum(self._diag_variance, 1e-12))[None, :, :]
        self._temporal_filter_inplace(z)

        q = self._history_basis()
        self._last_proposal_rank = 0 if q is None else int(q.shape[1])
        alpha = float(self.cfg.guided_fraction) if q is not None else 0.0
        if alpha > 0.0 and q is not None:
            # Blend an ordinary full-space draw with a low-rank guided draw,
            # then renormalize the expected trace back to D.  This is much less
            # aggressive than assigning alpha*D/r variance to each guided
            # direction (which is pathological for D=H*nu=400 and small r).
            d, r = h * nu, q.shape[1]
            z *= math.sqrt(max(0.0, 1.0 - alpha))
            coeff = self.rng.standard_normal((n, r))
            z.reshape(n, d)[:] += math.sqrt(alpha) * (coeff @ q.T)
            expected_trace = (1.0 - alpha) * d + alpha * r
            z *= math.sqrt(d / max(expected_trace, 1e-12))

        # ``z`` is no longer needed after proposal construction. Reuse it for
        # controls to avoid allocating separate noise and control tensors.
        np.multiply(z, self._joint_std[None, None, :], out=z)
        np.add(nominal[None, :, :], z, out=z)
        np.clip(z, self._ctrl_low, self._ctrl_high, out=z)
        return z

    def _sample_spline(self, nominal: np.ndarray) -> np.ndarray:
        n, h, nu = self.cfg.num_rollouts, self.cfg.horizon, self.robot.nu
        m = self._spline_basis.shape[1]
        coeff = self.rng.standard_normal((n, m, nu))
        z = np.einsum("tm,nmu->ntu", self._spline_basis, coeff, optimize=True)
        np.multiply(z, self._joint_std[None, None, :], out=z)
        np.add(nominal[None, :, :], z, out=z)
        np.clip(z, self._ctrl_low, self._ctrl_high, out=z)
        return z

    @staticmethod
    def _make_poisson_twitch_kernel(
        control_dt: float,
        rise_s: float,
        decay_s: float,
        duration_s: float,
    ) -> np.ndarray:
        """Return a causal, positive difference-of-exponentials twitch kernel."""
        steps = max(1, int(math.ceil(float(duration_s) / float(control_dt))))
        # Evaluate at the center/end of each discrete control bin rather than
        # exactly t=0; otherwise a difference of exponentials would have zero
        # effect on the control sample in which the event occurs.
        t = (np.arange(steps, dtype=np.float64) + 1.0) * float(control_dt)
        kernel = np.exp(-t / float(decay_s)) - np.exp(-t / float(rise_s))
        kernel = np.maximum(kernel, 0.0)
        peak = float(np.max(kernel)) if kernel.size else 0.0
        if peak <= 1e-12:
            return np.ones(1, dtype=np.float64)
        kernel /= peak
        return kernel

    @staticmethod
    def _make_recruitment_marks(
        levels: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build an ordered size-principle-inspired motor-unit pool.

        Recruitment marks no longer mean an arbitrary scalar amplitude.  Each
        actuator owns ``levels`` ordered motor units. Unit twitch strength grows
        monotonically with recruitment order (1, 2, ..., levels), while a mark
        at depth ``r`` recruits units 1..r cumulatively.  Because all recruited
        units share the same event time and twitch kernel, summing their twitches
        is exactly equivalent to using the cumulative amplitude returned here.

        The full-pool twitch is normalized to one, so ``peak`` scaling keeps its
        previous interpretation. ``variance`` scaling is recomputed from the new
        mark amplitudes, keeping expected exploration power comparable.
        """
        levels = max(1, int(levels))

        # Simplified Henneman-like size principle: later-recruited motor units
        # contribute more twitch force. Using rank-proportional strengths avoids
        # another tuning parameter while producing a genuinely hierarchical pool.
        unit_strengths = np.arange(1, levels + 1, dtype=np.float64)
        unit_strengths /= float(np.sum(unit_strengths))

        # Recruitment depth r activates every lower-threshold unit 1..r.
        amplitudes = np.cumsum(unit_strengths)
        amplitudes[-1] = 1.0

        # Keep the old fixed mark law: every recruitment depth is equally likely.
        # This isolates the contribution of ordered cumulative recruitment from
        # changes in firing rate or mark probabilities.
        prior = np.full(levels, 1.0 / float(levels), dtype=np.float64)

        return amplitudes, prior, unit_strengths

    def _compute_poisson_global_noise_scale(self) -> float:
        """Match only the *global* uniform-law motor power to the 8-neuron identity baseline.

        There is intentionally no per-joint variance matching.  Every sparse
        primitive has unit row energy, so the fixed population event budget
        determines total motor power.  This single scalar converts the common
        twitch/event law to the configured joint-noise scale while preserving
        natural anisotropy from primitive wiring and learned spike statistics.
        """
        total_rate = np.asarray(
            self._poisson_pos_rate_map + self._poisson_neg_rate_map, dtype=np.float64
        )
        mark_prob = np.asarray(self._poisson_mark_prob_map, dtype=np.float64)
        mark_sq = self._poisson_levels * self._poisson_levels
        expected_mark_sq = np.sum(mark_prob * mark_sq[None, None, :], axis=-1)
        impulse_variance = float(self.cfg.control_dt) * total_rate * expected_mark_sq
        channel_variance = self._poisson_variance_weights @ impulse_variance
        total_motor_variance = float(
            np.sum(channel_variance * np.sum(self._poisson_channels ** 2, axis=1))
        )
        # For Ant-2 the signed point process is split into two physical muscle
        # channels. Total variance is conserved by Poisson thinning, so divide by
        # all 16 actuator coordinates to match standard MPPI per muscle channel.
        variance_dim = int(self.robot.nu)
        mean_motor_variance = total_motor_variance / float(max(variance_dim, 1))
        return 1.0 / math.sqrt(max(mean_motor_variance, 1e-12))

    def _make_fixed_poisson_diagnostics(self) -> dict[str, float]:
        """Diagnostics for the fixed direct-joint Poisson baseline."""
        m = int(self._poisson_channels.shape[0])
        levels = len(self._poisson_levels)
        total_rate = float(self._poisson_total_rate_hz)
        per_rate = total_rate / max(m, 1)
        out = {
            "poisson_rate_hz_mean": per_rate, "poisson_rate_hz_std": 0.0,
            "poisson_rate_hz_min": per_rate, "poisson_rate_hz_max": per_rate,
            "poisson_pos_rate_hz_min": per_rate / 2, "poisson_pos_rate_hz_max": per_rate / 2,
            "poisson_neg_rate_hz_min": per_rate / 2, "poisson_neg_rate_hz_max": per_rate / 2,
            "poisson_sign_prob_min": 0.5, "poisson_sign_prob_max": 0.5,
            "poisson_sign_entropy": 1.0,
            "poisson_recruitment_entropy": float(levels > 1),
            "poisson_mark_entropy_mean": float(levels > 1),
            "poisson_mark_prob_min": 1.0 / levels, "poisson_mark_prob_max": 1.0 / levels,
            "poisson_mean_recruitment": float(np.mean(self._poisson_levels)),
            "poisson_motor_unit_pool_size": float(levels),
            "poisson_mean_recruited_units": float((levels + 1) / 2.0),
            "poisson_mean_recruited_fraction": float((levels + 1) / (2.0 * levels)),
            "poisson_low_threshold_unit_fraction": float(self._poisson_motor_unit_strengths[0]),
            "poisson_high_threshold_unit_fraction": float(self._poisson_motor_unit_strengths[-1]),
            "poisson_event_channels": float(m) if self.sampling == SamplingOption.POISSON else math.nan,
            "poisson_expected_events": self.cfg.control_dt * self.cfg.horizon * total_rate,
            "poisson_total_rate_hz": total_rate,
            "poisson_event_budget_fixed": 1.0,
            "poisson_firing_fixed": float(self.sampling == SamplingOption.POISSON),
            "poisson_noise_scale": float(self._poisson_global_noise_scale),
            "poisson_variance_match_scale": float(self._poisson_variance_match_scale),
            "poisson_expected_rms_ratio": float(
                self._poisson_global_noise_scale / max(self._poisson_variance_match_scale, 1e-12)
            ),
            "poisson_peak_scale_mode": float(self.cfg.proposal_scale_mode == "peak"),
        }
        if self.sampling != SamplingOption.POISSON:
            return {key: math.nan for key in out}
        return out

    def _event_diagnostics(self) -> dict[str, float]:
        if self.sampling == SamplingOption.POISSON:
            return self._fixed_poisson_diagnostics
        if self.sampling == SamplingOption.SPIKE:
            out = {key: math.nan for key in self._fixed_poisson_diagnostics}
            out.update({
                "spike_rate_hz_mean": float(self.cfg.base_rate_hz),
                "spike_rate_hz_min": float(self.cfg.base_rate_hz),
                "spike_rate_hz_max": float(3.0 * self.cfg.base_rate_hz),
                "spike_motor_unit_pool_size": float(self.cfg.recruitment_levels),
                "spike_noise_scale": float(self._spike_global_noise_scale),
                "spike_variance_match_scale": float(self._spike_variance_match_scale),
                "spike_expected_rms_ratio": float(
                    self._spike_global_noise_scale / max(self._spike_variance_match_scale, 1e-12)
                ),
                "spike_peak_scale_mode": float(self.cfg.proposal_scale_mode == "peak"),
                "spike_firing_fixed": 0.0,
            })
            out.update(self._spike_last_stats)
            return out
        return {key: math.nan for key in self._fixed_poisson_diagnostics}

    def _sample_spectrum_matched_gaussian(self, nominal: np.ndarray) -> np.ndarray:
        """Gaussian control matched to the Poisson or Spike source spectrum."""
        if self._matched_gaussian_generator is None:
            raise RuntimeError("spectrum-matched Gaussian generator is unavailable")
        n = self.cfg.screen_pool_size if self._screen_enabled else self.cfg.num_rollouts
        z = self._matched_gaussian_generator.sample(self.rng, n)
        z *= self._joint_std[None, None, :]
        z += nominal[None, :, :]
        np.clip(z, self._ctrl_low, self._ctrl_high, out=z)
        return z

    def _sample_poisson(self, nominal: np.ndarray) -> np.ndarray:
        """Fixed Poisson proposal in native actuator space.

        On Ant-2 each joint owns two physical muscle channels. Positive events
        are routed to the agonist channel and negative events to the antagonist
        channel before the common twitch kernel is applied. Both muscle channels
        may therefore overlap in time and express genuine co-contraction.
        """
        if self._poisson_generator is None:
            raise RuntimeError("Poisson generator is unavailable")
        if bool(getattr(self.robot, "is_muscle_model", False)):
            z, event_total = self._poisson_generator.sample_projected_antagonistic_pairs(self.rng)
            n = int(z.shape[0])
            # Each physical muscle receives only nonnegative twitch events. Center
            # across rollouts before the additive MPPI update, exactly as in
            # Spike, so warm-started muscle activations can move both up and
            # down without introducing a common positive drift.
            z -= np.mean(z, axis=0, keepdims=True)
            z *= self._poisson_global_noise_scale
            # Scale every physical muscle channel exactly like standard MPPI's
            # 16-D control coordinates.
            z *= self._joint_std[None, None, :]
            z += nominal[None, :, :]
            np.clip(z, self._ctrl_low, self._ctrl_high, out=z)
        else:
            z, event_total = self._poisson_generator.sample_projected_identity(self.rng)
            n = int(z.shape[0])
            z *= self._poisson_global_noise_scale
            z *= self._joint_std[None, None, :]
            z += nominal[None, :, :]
            np.clip(z, self._ctrl_low, self._ctrl_high, out=z)
        self._last_event_count_mean = float(event_total) / n
        return z

    def _sample_spike(self, nominal: np.ndarray) -> np.ndarray:
        """Biological motor-pool proposal with zero-mean MPPI perturbations."""
        if self._spike_generator is None:
            raise RuntimeError("Spike motor-pool generator is unavailable")
        z, stats = self._spike_generator.sample(self.rng)
        if bool(getattr(self.robot, "is_muscle_model", False)):
            # Separate antagonist twitches are nonnegative physical excitations,
            # but MPPI requires an additive perturbation around its warm-start
            # nominal. Remove the proposal mean while preserving joint/pool
            # covariance, including genuine common-mode co-contraction structure.
            z -= np.mean(z, axis=0, keepdims=True)
            z *= self._spike_global_noise_scale
            z *= self._joint_std[None, None, :]
        else:
            z *= self._spike_global_noise_scale
            z *= self._joint_std[None, None, :]
        z += nominal[None, :, :]
        np.clip(z, self._ctrl_low, self._ctrl_high, out=z)
        self._spike_last_stats = dict(stats)
        self._last_event_count_mean = float(stats.get("spike_events_mean", 0.0))
        return z

    def _screen_population(
        self,
        start,
        nominal: np.ndarray,
        population: np.ndarray,
        current_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        """Rank a large candidate population with a cheap MuJoCo preview.

        The first ``screen_exploit`` full-fidelity slots take the lowest
        preview costs. Remaining slots are sampled uniformly from the rest to
        retain coverage when the coarse model misranks candidates.
        """
        if self.native_batcher is None or not hasattr(self.native_batcher, 'evaluate_screen'):
            raise RuntimeError('screening requires the rebuilt fused screening evaluator')
        hs = int(self.cfg.horizon)
        preview = np.ascontiguousarray(population[:, :hs, :], dtype=np.float64)
        nominal_preview = np.ascontiguousarray(nominal[:hs], dtype=np.float64)
        _, cheap_costs, _, cheap_failed = self.native_batcher.evaluate_screen(
            start, preview, self.track, current_s,
            nominal_controls=nominal_preview, cost_cfg=self.cost_cfg,
            screen_timestep=float(self.cfg.control_dt),
            iterations=5,
            ls_iterations=1,
            tolerance=1e-4,
        )
        cheap_costs = np.asarray(cheap_costs, dtype=np.float64).reshape(-1)
        cheap_failed = np.asarray(cheap_failed, dtype=bool).reshape(-1)
        n_pool = int(len(cheap_costs))
        n_full = int(self.cfg.num_rollouts)
        n_exploit = min(int(self.cfg.screen_exploit), n_full, n_pool)
        order = np.argsort(cheap_costs, kind='stable')
        exploit_idx = np.asarray(order[:n_exploit], dtype=np.int64)

        selected_mask = np.zeros(n_pool, dtype=bool)
        selected_mask[exploit_idx] = True
        remaining_idx = np.flatnonzero(~selected_mask)
        n_explore = n_full - n_exploit
        if n_explore > 0:
            explore_idx = np.asarray(
                self.rng.choice(remaining_idx, size=n_explore, replace=False),
                dtype=np.int64,
            )
            selected_idx = np.concatenate((exploit_idx, explore_idx))
        else:
            selected_idx = exploit_idx
        # Reserve one full-fidelity slot for the nominal (zero-noise) candidate.
        # The nominal occupies population row 0, even if its preview cost is poor.
        if n_full > 0 and not np.any(selected_idx == 0):
            selected_idx[-1] = 0
        controls = np.ascontiguousarray(population[selected_idx], dtype=np.float64)

        finite = cheap_costs[np.isfinite(cheap_costs)]
        if finite.size:
            cheap_min = float(np.min(finite))
            cheap_median = float(np.median(finite))
        else:
            cheap_min = math.inf
            cheap_median = math.inf
        diag = {
            'screen_pool_size': float(n_pool),
            'screen_horizon': float(hs),
            'screen_exploit_count': float(n_exploit),
            'screen_explore_count': float(n_explore),
            'screen_failed_fraction': float(np.mean(cheap_failed)) if n_pool else 0.0,
            'screen_cost_min': cheap_min,
            'screen_cost_median': cheap_median,
            'screen_ms': float(getattr(self.native_batcher, 'last_screen_ms', 0.0)),
        }
        return controls, selected_idx, cheap_costs, diag

    @staticmethod
    def _weights_copy(costs: np.ndarray, temperature: float) -> np.ndarray:
        c = np.asarray(costs, dtype=np.float64).reshape(-1)
        w = np.zeros_like(c)
        finite = np.isfinite(c)
        count = int(np.count_nonzero(finite))
        if count == 0:
            if len(w):
                w.fill(1.0 / len(w))
            return w
        rho = float(np.min(c[finite]))
        w[finite] = np.exp(np.clip(-(c[finite] - rho) / max(float(temperature), 1e-300), -745.0, 0.0))
        total = float(np.sum(w))
        if total <= 1e-12:
            w[finite] = 1.0 / count
        else:
            w /= total
        return w

    def _sample_icem(self, nominal: np.ndarray) -> np.ndarray:
        controls = self._sample_standard(nominal)
        if self._icem_elites is None or self.cfg.icem_elites <= 0:
            return controls
        k = min(int(self.cfg.icem_elites), controls.shape[0], self._icem_elites.shape[0])
        if k <= 0:
            return controls
        # The sampled first k rows are discarded for elite reuse anyway, so
        # write the shifted elites directly into them instead of allocating an
        # intermediate k x H x nu tensor and copying it back.
        controls[:k, :-1] = self._icem_elites[:k, 1:]
        controls[:k, -1] = self._icem_elites[:k, -1]
        np.clip(controls[:k], self._ctrl_low, self._ctrl_high, out=controls[:k])
        return controls

    def _normalized_weights(self, costs: np.ndarray, temperature: float) -> np.ndarray:
        """Return normalized MPPI weights in a persistent N-element buffer.

        The same weights drive the control update, diagnostics, and (for the
        diagonal-low-rank sampler) covariance adaptation. Computing the
        exponential weights once avoids two or three duplicate passes per tick.
        The returned array is borrowed until the next controller step.
        """
        c = np.asarray(costs, dtype=np.float64).reshape(-1)
        if self._weight_workspace.shape != c.shape:
            self._weight_workspace = np.empty_like(c)
        w = self._weight_workspace
        finite = np.isfinite(c)
        w.fill(0.0)
        finite_count = int(np.count_nonzero(finite))
        if finite_count == 0:
            if len(w):
                w.fill(1.0 / len(w))
            return w
        rho = float(np.min(c[finite]))
        w[finite] = np.exp(
            np.clip(
                -(c[finite] - rho) / max(float(temperature), 1e-300),
                -745.0, 0.0,
            )
        )
        total = float(np.sum(w))
        if total <= 1e-12:
            w[finite] = 1.0 / finite_count
        else:
            w /= total
        return w

    def _shift_horizon_array(self, x: np.ndarray, *, tail) -> np.ndarray:
        y = np.empty_like(x)
        y[:-1] = x[1:]
        y[-1] = tail
        return y

    def _update_direction_memory(self, candidate: np.ndarray, nominal: np.ndarray) -> None:
        if self.sampling not in {SamplingOption.GUIDED, SamplingOption.DIAG_LOWRANK}:
            return
        denom = np.maximum(self._joint_std, 1e-12)
        direction = (np.asarray(candidate) - np.asarray(nominal)) / denom[None, :]
        if not np.all(np.isfinite(direction)):
            return
        if self.cfg.warm_start:
            shifted_history = []
            for old in self._direction_history:
                old_h = np.asarray(old, dtype=np.float64).reshape(self.cfg.horizon, self.robot.nu)
                old_shift = self._shift_horizon_array(old_h, tail=np.zeros(self.robot.nu))
                shifted_history.append(old_shift.reshape(-1))
            direction = self._shift_horizon_array(direction, tail=np.zeros(self.robot.nu))
            self._direction_history = shifted_history
        else:
            self._direction_history = []
        self._direction_history.append(direction.reshape(-1).copy())
        if len(self._direction_history) > self.cfg.guided_rank:
            self._direction_history = self._direction_history[-self.cfg.guided_rank :]

    def _update_adaptive_diagonal(
        self,
        controls: np.ndarray,
        nominal: np.ndarray,
        weights: np.ndarray,
        ess: float,
    ) -> None:
        if self.sampling != SamplingOption.DIAG_LOWRANK:
            return
        w = weights
        denom = np.maximum(self._joint_std, 1e-12)
        delta = (np.asarray(controls) - np.asarray(nominal)[None, :, :]) / denom[None, None, :]
        mean = np.einsum("n,nhu->hu", w, delta)
        centered = delta - mean[None, :, :]
        empirical = np.einsum("n,nhu->hu", w, centered * centered)

        # With small populations, only trust the empirical variance when the
        # MPPI weights retain a reasonable ESS. Otherwise shrink to standard
        # MPPI variance instead of collapsing around a few samples.
        target_ess = max(4.0, min(16.0, 0.5 * self.cfg.num_rollouts))
        confidence = float(np.clip((float(ess) - 1.0) / max(target_ess - 1.0, 1.0), 0.0, 1.0))
        target = (1.0 - confidence) + confidence * empirical
        target = np.clip(target, self.cfg.diag_lowrank_min, self.cfg.diag_lowrank_max)
        mean_target = float(np.mean(target))
        if mean_target > 1e-12:
            target /= mean_target
        rate = float(self.cfg.diag_lowrank_rate)
        updated = (1.0 - rate) * self._diag_variance + rate * target
        updated = np.clip(updated, self.cfg.diag_lowrank_min, self.cfg.diag_lowrank_max)
        mean_updated = float(np.mean(updated))
        if mean_updated > 1e-12:
            updated /= mean_updated
        if self.cfg.warm_start:
            updated = self._shift_horizon_array(updated, tail=np.ones(self.robot.nu))
        self._diag_variance = updated

    def _update_icem_elites(self, controls: np.ndarray, costs: np.ndarray) -> None:
        if self.sampling != SamplingOption.ICEM or self.cfg.icem_elites <= 0:
            return
        finite = np.flatnonzero(np.isfinite(costs))
        if finite.size == 0:
            return
        order = finite[np.argsort(np.asarray(costs)[finite])]
        k = min(int(self.cfg.icem_elites), len(order))
        self._icem_elites = np.asarray(controls[order[:k]], dtype=np.float64).copy()

    def _evaluate_batch(self, *args, **kwargs):
        return evaluate_control_batch(*args, **kwargs)

    def step(self, data, current_s: float) -> tuple[np.ndarray, dict[str, Any]]:
        t_total = time.perf_counter()

        start, base_nominal, refined, sensitivity, endpoints, prior_mean, prior_cov, nominal_parts = self._build_nominal(data, current_s)
        t_nominal = time.perf_counter()
        nominal = refined.controls
        screen_population = None
        screen_selected_idx = None
        screen_cheap_costs = None
        screen_diag: dict[str, float] = {}
        if self.sampling == SamplingOption.GUIDED:
            controls = self._sample_guided(nominal, adaptive_diagonal=False)
        elif self.sampling == SamplingOption.DIAG_LOWRANK:
            controls = self._sample_guided(nominal, adaptive_diagonal=True)
        elif self.sampling == SamplingOption.SPLINE:
            controls = self._sample_spline(nominal)
        elif self.sampling == SamplingOption.ICEM:
            controls = self._sample_icem(nominal)
        elif self.sampling in {SamplingOption.GAUSSIAN_POISSON, SamplingOption.GAUSSIAN_SPIKE}:
            controls = self._sample_spectrum_matched_gaussian(nominal)
        elif self.sampling == SamplingOption.POISSON:
            controls = self._sample_poisson(nominal)
        elif self.sampling == SamplingOption.SPIKE:
            controls = self._sample_spike(nominal)
        elif self.sampling == SamplingOption.PINK:
            controls = self._sample_pink(
                nominal,
                count=self.cfg.screen_pool_size if self._screen_enabled else None,
            )
        else:
            controls = self._sample_standard(
                nominal,
                count=self.cfg.screen_pool_size if self._screen_enabled else None,
            )
        # Scale deviations about the nominal *before* IK; all samplers retain
        # their temporal structure, and the exact nominal candidate is preserved.
        if self.cfg.adaptive_noise:
            distance = max(0.0, float(self.robot.tip_target_distance(data)))
            noise_scale = self.cfg.noise_min_fraction + (1.0 - self.cfg.noise_min_fraction) * min(
                1.0, (distance / self.cfg.dslow) ** self.cfg.noise_decay_order
            )
        else:
            noise_scale = 1.0
        controls = nominal[None, :, :] + noise_scale * (controls - nominal[None, :, :])
        controls = self._convert_proposals_to_effector_space(controls, nominal, data)
        # Always evaluate the current nominal control sequence as candidate 0.
        # Apply this after IK so numerical conversion cannot perturb the baseline.
        controls[0] = nominal
        if self._screen_enabled:
            screen_population = controls
            t_sample = time.perf_counter()
            controls, screen_selected_idx, screen_cheap_costs, screen_diag = self._screen_population(
                start, nominal, screen_population, current_s
            )
        else:
            t_sample = time.perf_counter()
        t_screen = time.perf_counter()

        self._screen_step += 1
        # Optional full-population diagnostic audit. On audit iterations the whole
        # screen pool is evaluated at full fidelity, allowing the selected subset to
        # be compared against the true full-pool MPPI population. Disabled by default.
        audit_now = (
            self._screen_enabled
            and self.cfg.screen_audit_every > 0
            and (self._screen_step % self.cfg.screen_audit_every == 0)
        )
        audit_reference_first = None
        if audit_now and screen_population is not None and screen_selected_idx is not None:
            all_positions, all_costs, all_progress, all_failed = self._evaluate_batch(
                self.robot, start, screen_population, self.track, current_s,
                control_substeps=self.control_substeps, nominal_controls=nominal,
                cost_cfg=self.cost_cfg, workers=self.cfg.rollout_workers,
                native_batcher=self.native_batcher,
            )
            positions = np.asarray(all_positions)[screen_selected_idx].copy()
            costs = np.asarray(all_costs)[screen_selected_idx].copy()
            terminal_progress = np.asarray(all_progress)[screen_selected_idx].copy()
            failed = np.asarray(all_failed, dtype=bool)[screen_selected_idx].copy()

            full_costs = np.asarray(all_costs, dtype=np.float64)
            if self.cfg.adaptive_temperature_lbps and NUMBA_AVAILABLE:
                full_temp, _, full_ess, _, _, _, _ = lbps_optimize_fast(
                    full_costs, float(self.cfg.lbps_delta),
                    float(self.cfg.lambda_temperature),
                    int(self.cfg.lbps_optimizer_iterations),
                )
            elif self.cfg.adaptive_temperature_lbps:
                full_lbps = optimize_lbps_temperature(
                    full_costs, delta=self.cfg.lbps_delta,
                    fallback_temperature=self.cfg.lambda_temperature,
                    iterations=self.cfg.lbps_optimizer_iterations,
                )
                full_temp, full_ess = full_lbps.temperature, full_lbps.ess
            else:
                full_temp = float(self.cfg.lambda_temperature)
                wf = self._weights_copy(full_costs, full_temp)
                full_ess = float(1.0 / max(np.sum(wf * wf), 1e-300))
            full_w = self._weights_copy(full_costs, full_temp)
            full_order = np.argsort(full_costs, kind='stable')
            top = full_order[: self.cfg.num_rollouts]
            selected_set = set(map(int, np.asarray(screen_selected_idx).tolist()))
            overlap = sum(int(i) in selected_set for i in top) / max(len(top), 1)
            captured_mass = float(np.sum(full_w[screen_selected_idx]))
            top_mass = float(np.sum(full_w[top]))
            full_candidate = np.einsum('n,nhu->hu', full_w, screen_population, optimize=False)
            audit_reference_first = np.asarray(full_candidate[0] - nominal[0], dtype=np.float64).copy()
            screen_diag.update({
                'screen_audit': 1.0,
                'screen_audit_full_ess': float(full_ess),
                'screen_audit_top_recall': float(overlap),
                'screen_audit_captured_weight_mass': captured_mass,
                'screen_audit_oracle_top_mass': top_mass,
            })
        else:
            positions, costs, terminal_progress, failed = self._evaluate_batch(
                self.robot,
                start,
                controls,
                self.track,
                current_s,
                control_substeps=self.control_substeps,
                nominal_controls=nominal,
                cost_cfg=self.cost_cfg,
                workers=self.cfg.rollout_workers,
                native_batcher=self.native_batcher,
            )
            if self._screen_enabled:
                screen_diag['screen_audit'] = 0.0
        t_rollout = time.perf_counter()
        rollout_physics_ms = (
            float(self.native_batcher.last_rollout_physics_ms)
            if self.native_batcher is not None else 0.0
        )
        rollout_cost_ms = (
            float(self.native_batcher.last_rollout_cost_ms)
            if self.native_batcher is not None else 0.0
        )
        rollout_fused_ms = (
            float(self.native_batcher.last_rollout_fused_ms)
            if self.native_batcher is not None else 0.0
        )

        if self.cfg.adaptive_temperature_lbps:
            if NUMBA_AVAILABLE:
                (
                    temperature, _alpha, ess, lbps_score, finite_count,
                    _reward_norm, _expected_return,
                ) = lbps_optimize_fast(
                    np.asarray(costs, dtype=np.float64),
                    float(self.cfg.lbps_delta),
                    float(self.cfg.lambda_temperature),
                    int(self.cfg.lbps_optimizer_iterations),
                )
            else:
                lbps = optimize_lbps_temperature(
                    costs,
                    delta=self.cfg.lbps_delta,
                    fallback_temperature=self.cfg.lambda_temperature,
                    iterations=self.cfg.lbps_optimizer_iterations,
                )
                temperature = lbps.temperature
                ess = lbps.ess
                finite_count = lbps.finite_count
                lbps_score = lbps.score
        else:
            temperature = self.cfg.lambda_temperature
            finite_count = int(np.count_nonzero(np.isfinite(costs)))
            finite = np.isfinite(costs)
            if finite_count:
                rho = np.min(costs[finite])
                w = np.exp(-(costs[finite] - rho) / temperature)
                ess = float(np.sum(w) ** 2 / max(np.sum(w * w), 1e-300))
            else:
                ess = 0.0
            lbps_score = math.nan

        # Compute MPPI weights once and reuse them for the weighted sequence,
        # diagnostics, and any sampler-specific adaptation.
        diag_weights = self._normalized_weights(costs, temperature)
        candidate = np.einsum("n,nhu->hu", diag_weights, controls, optimize=False)
        np.clip(candidate, self._ctrl_low, self._ctrl_high, out=candidate)
        if self.sampling == SamplingOption.DIAG_LOWRANK:
            self._update_adaptive_diagonal(controls, nominal, diag_weights, ess)
        if self.sampling in {SamplingOption.GUIDED, SamplingOption.DIAG_LOWRANK}:
            self._update_direction_memory(candidate, nominal)
        if self.sampling == SamplingOption.ICEM:
            self._update_icem_elites(controls, costs)

        if self._screen_enabled and screen_selected_idx is not None and screen_cheap_costs is not None:
            best_selected = int(np.argmin(costs)) if len(costs) else 0
            n_exploit = int(self.cfg.screen_exploit)
            selected_cheap = np.asarray(screen_cheap_costs)[screen_selected_idx]
            screen_diag['screen_selected_cost_median'] = float(
                np.median(selected_cheap[np.isfinite(selected_cheap)])
            ) if np.any(np.isfinite(selected_cheap)) else math.inf
            screen_diag['screen_full_best_from_explore'] = float(best_selected >= n_exploit)
            cheap_order = np.argsort(np.asarray(screen_cheap_costs), kind='stable')
            inverse_rank = np.empty_like(cheap_order)
            inverse_rank[cheap_order] = np.arange(len(cheap_order))
            screen_diag['screen_full_best_cheap_rank'] = float(
                inverse_rank[int(screen_selected_idx[best_selected])] + 1
            )
            if audit_reference_first is not None:
                a = np.asarray(candidate[0] - nominal[0], dtype=np.float64)
                b = np.asarray(audit_reference_first, dtype=np.float64)
                denom = float(np.linalg.norm(a) * np.linalg.norm(b))
                screen_diag['screen_audit_first_action_cosine'] = (
                    float(np.dot(a, b) / denom) if denom > 1e-12 else math.nan
                )
            self._last_screen_diag = dict(screen_diag)

        if self.cfg.warm_start:
            self._previous_plan = np.asarray(candidate, dtype=np.float64).copy()
        best = int(np.argmin(costs)) if len(costs) else 0
        if getattr(self.native_batcher, "returns_best_only", False):
            best_rollout = np.asarray(positions, dtype=np.float64).copy()
        else:
            best_rollout = positions[best].copy()
        t_update = time.perf_counter()

        # Optimization-quality diagnostics use only the sampled rollout
        # population.  The nominal sequence is the proposal center but is not
        # evaluated as an extra rollout, so nominal-cost deltas are undefined.
        finite_cost = np.isfinite(costs)
        nominal_cost = math.nan
        costs_arr = np.asarray(costs, dtype=np.float64)
        if np.any(finite_cost):
            weighted_rollout_cost = float(
                np.sum(diag_weights[finite_cost] * costs_arr[finite_cost])
            )
            best_finite_cost = float(np.min(costs_arr[finite_cost]))
        else:
            weighted_rollout_cost = math.nan
            best_finite_cost = math.nan

        if np.isfinite(nominal_cost) and np.isfinite(weighted_rollout_cost):
            nominal_weighted_improvement = nominal_cost - weighted_rollout_cost
            nominal_weighted_improvement_rel = nominal_weighted_improvement / max(
                abs(nominal_cost), 1e-12
            )
        else:
            nominal_weighted_improvement = math.nan
            nominal_weighted_improvement_rel = math.nan

        if np.isfinite(nominal_cost) and np.isfinite(best_finite_cost):
            nominal_best_improvement = nominal_cost - best_finite_cost
            nominal_best_improvement_rel = nominal_best_improvement / max(
                abs(nominal_cost), 1e-12
            )
        else:
            nominal_best_improvement = math.nan
            nominal_best_improvement_rel = math.nan

        applied_delta = np.asarray(candidate[0] - nominal[0], dtype=np.float64)
        applied_residual_l2 = float(np.linalg.norm(applied_delta))
        applied_residual_rms_norm = float(
            np.sqrt(np.mean((applied_delta / np.maximum(self._ctrl_scale, 1e-12)) ** 2))
        )
        # Full-horizon update magnitude complements the first-action residual.
        # This is useful for temporally structured proposals whose main correction
        # can occur later in the receding horizon.
        plan_delta = np.asarray(candidate - nominal, dtype=np.float64)
        plan_update_l2 = float(np.linalg.norm(plan_delta))
        plan_update_rms_norm = float(
            np.sqrt(np.mean((plan_delta / np.maximum(self._ctrl_scale[None, :], 1e-12)) ** 2))
        )
        ctrl0 = np.asarray(candidate[0], dtype=np.float64)
        sat_tol = 1e-6 * np.maximum(self._ctrl_high - self._ctrl_low, 1.0)
        applied_lower_bound_fraction = float(np.mean(ctrl0 <= self._ctrl_low + sat_tol))
        applied_upper_bound_fraction = float(np.mean(ctrl0 >= self._ctrl_high - sat_tol))
        # Backward-compatible any-bound metric. For Ant-2, a zero muscle
        # excitation is ordinary inactivity rather than pathological saturation;
        # paper-facing analysis should therefore prefer the separate upper-bound
        # metric when discussing actuator saturation.
        applied_saturation_fraction = float(
            np.mean((ctrl0 <= self._ctrl_low + sat_tol) | (ctrl0 >= self._ctrl_high - sat_tol))
        )
        spike_diag = self._event_diagnostics()
        finite_cost_values = costs[np.isfinite(costs)]
        cost_spread = (
            float(np.max(finite_cost_values) - np.min(finite_cost_values))
            if finite_cost_values.size else math.inf
        )

        info = {
            "planned_control_sequence": candidate,
            "base_nominal": base_nominal.controls,
            "nominal": nominal,
            "nominal_positions": refined.positions,
            "prior_mean": prior_mean,
            "spatial_covariance": prior_cov,
            "temperature": float(temperature),
            "ess": float(ess),
            "lbps_score": float(lbps_score),
            "finite_rollouts": int(finite_count),
            "collision_rollouts": int(np.count_nonzero(failed)),
            "best_rollout": best_rollout,
            "best_cost": float(costs[best]),
            "cost_spread": cost_spread,
            "best_terminal_progress": float(terminal_progress[best]),
            "rollout_backend": self.rollout_backend_name,
            "sampling_option": self.sampling.value,
            "proposal": self.sampling.value,  # backward-compatible info key
            "proposal_rank": int(self._last_proposal_rank),
            "diag_variance_min": (
                float(np.min(self._diag_variance))
                if self.sampling == SamplingOption.DIAG_LOWRANK else 1.0
            ),
            "diag_variance_max": (
                float(np.max(self._diag_variance))
                if self.sampling == SamplingOption.DIAG_LOWRANK else 1.0
            ),
            "icem_elites": 0 if self._icem_elites is None else int(len(self._icem_elites)),
            "standard_sampler_backend": self._standard_sampler_backend,
            "poisson_sampler_backend": self._poisson_sampler_backend,
            "spike_sampler_backend": self._spike_sampler_backend,
            "nominal_geometry_evaluated": bool(len(refined.positions)),
            "poisson_event_channels": int(self._poisson_channels.shape[0]) if self.sampling == SamplingOption.POISSON else math.nan,
            "poisson_events_mean": float(self._last_event_count_mean) if self.sampling == SamplingOption.POISSON else math.nan,
            "spike_events_mean": float(self._last_event_count_mean) if self.sampling == SamplingOption.SPIKE else math.nan,
            **spike_diag,
            **screen_diag,
            "nominal_cost": nominal_cost,
            "weighted_rollout_cost": weighted_rollout_cost,
            "best_finite_cost": best_finite_cost,
            "nominal_weighted_improvement": float(nominal_weighted_improvement),
            "nominal_weighted_improvement_rel": float(nominal_weighted_improvement_rel),
            "nominal_best_improvement": float(nominal_best_improvement),
            "nominal_best_improvement_rel": float(nominal_best_improvement_rel),
            "applied_residual_l2": applied_residual_l2,
            "applied_residual_rms_norm": applied_residual_rms_norm,
            "plan_update_l2": plan_update_l2,
            "plan_update_rms_norm": plan_update_rms_norm,
            "applied_saturation_fraction": applied_saturation_fraction,
            "applied_lower_bound_fraction": applied_lower_bound_fraction,
            "applied_upper_bound_fraction": applied_upper_bound_fraction,
            "timing_ms": {
                "nominal": 1e3 * (t_nominal - t_total),
                **nominal_parts,
                "sampling": 1e3 * (t_sample - t_nominal),
                "screening": 1e3 * (t_screen - t_sample),
                "rollouts": 1e3 * (t_rollout - t_screen),
                "rollout_physics": rollout_physics_ms,
                "rollout_cost": rollout_cost_ms,
                "rollout_fused": rollout_fused_ms,
                "update": 1e3 * (t_update - t_rollout),
                "total": 1e3 * (t_update - t_total),
            },
        }
        # Previous "total" ended before diagnostics, underreporting latency.
        action = candidate[0].copy()
        t_done = time.perf_counter()
        info["timing_ms"]["diagnostics"] = 1e3 * (t_done - t_update)
        info["timing_ms"]["total"] = 1e3 * (t_done - t_total)
        return action, info
