"""Ball-on-plate MPPI objective.

Literature basis: Zarzycki & Lawrynczuk (2021), Sensors 21(12), 3959,
Equation (23): squared reference tracking + squared control increments.
https://doi.org/10.3390/s21123959

Velocity, tilt, loss-of-contact, terminal and proposal-deviation terms are
explicit implementation extensions. Weights below are starting points to tune,
NOT numerical coefficients reported by that paper.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BalanceWeights:
    position: float = 3.0        # relative XY error, normalized by usable half-width
    velocity: float = 0.8        # relative XY ball speed normalized by 0.3 m/s
    tilt: float = 0.06          # squared normal-angle error (normalized)
    control_delta: float = 0.002  # squared normalized command increments
    proposal: float = 0.0001    # proposal vs nominal MPPI regularization
    air_gap: float = 2.0        # prevent exploiting ballistic ball flight
    terminal: float = 2.0       # terminal pos and speed penalty
    fall: float = 30.0          # nonfatal soft penalty: differentiates failed rollouts


def score_rollouts(
    f: dict, controls: np.ndarray, nominal: np.ndarray,
    initial_ctrl: np.ndarray, ctrl_scale: np.ndarray, *, substeps: int,
    weights: BalanceWeights = BalanceWeights(),
) -> np.ndarray:
    """Vectorized costs for (N, physics_steps, ...) MuJoCo trajectories."""
    r = np.asarray(f["relative_xy"], dtype=float)[:, substeps - 1::substeps]
    v = np.asarray(f["relative_velocity_xy"], dtype=float)[:, substeps - 1::substeps]
    nz = np.asarray(f["normal"], dtype=float)[:, substeps - 1::substeps, 2]
    gap = np.asarray(f["gap"], dtype=float)[:, substeps - 1::substeps]
    usable = max(float(f["half_width"]) - float(f["ball_radius"]), 1e-3)
    pos = np.sum((r / usable) ** 2, axis=-1)
    speed = np.sum((v / 0.3) ** 2, axis=-1)
    # Magnitude of normal deviation is well behaved near a horizontal plate.
    tilt = ((1.0 - nz) / 0.02) ** 2
    # The ball should stay close to the plate surface, without flying.
    air_gap = (np.maximum(gap - 0.015, 0.0) / 0.03) ** 2
    lost = ((np.max(np.abs(r), axis=-1) > usable) |
            (gap < -0.035) | (gap > 0.085))
    scale = np.maximum(np.asarray(ctrl_scale, dtype=float), 1e-6)
    prev = np.concatenate((np.broadcast_to(initial_ctrl, controls[:, :1].shape), controls[:, :-1]), axis=1)
    delta = np.mean(((controls - prev) / scale) ** 2, axis=-1)
    proposal = np.mean(((controls - nominal[None, :, :]) / scale) ** 2, axis=-1)
    running = (
        weights.position * pos + weights.velocity * speed +
        weights.tilt * tilt + weights.air_gap * air_gap +
        weights.control_delta * delta + weights.proposal * proposal
    )
    term = weights.terminal * (pos[:, -1] + speed[:, -1])
    # Track both time spent outside and whether a loss occurred at any time.
    loss = weights.fall * (np.mean(lost, axis=1) + 0.5 * np.any(lost, axis=1))
    return np.mean(running, axis=1) + term + loss
