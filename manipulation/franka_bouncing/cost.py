"""Ball-racket MPPI cost: anticipate the next contact and regulate outgoing apex.

Finite-horizon engineering adaptation of the impact objectives in:
Kulchenko & Todorov (ICRA 2011); Ronsse, Wei & Sternad (J Neurophysiol 2010).
Weights are not copied from either paper.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BounceWeights:
    landing_xy: float = 8.0
    impact_velocity: float = 18.0
    preimpact_velocity: float = 9.0
    flight_apex: float = 16.0
    tool_tilt: float = 0.03
    tool_workspace: float = 0.02
    control_delta: float = 0.003
    proposal: float = 0.0001
    miss: float = 100.0
    hit_bonus: float = 1.0
    target_apex_m: float = 0.20
    restitution_guess: float = 0.60


def score_rollouts(
    f: dict, controls: np.ndarray, nominal: np.ndarray,
    initial_ctrl: np.ndarray, ctrl_scale: np.ndarray, *, substeps: int,
    weights: BounceWeights = BounceWeights(),
) -> np.ndarray:
    pos = np.asarray(f['ball_pos'], dtype=float)
    vel = np.asarray(f['ball_velocity'], dtype=float)
    tool = np.asarray(f['tool_pos'], dtype=float)
    tool_v = np.asarray(f['tool_velocity'], dtype=float)
    normal_z = np.asarray(f['normal'], dtype=float)[..., 2]
    loc = np.asarray(f['relative_xy'], dtype=float)
    gap = np.asarray(f['gap'], dtype=float)
    usable = max(float(f['half_width']) - float(f['ball_radius']), 1e-3)
    g = 9.81
    desired_vz = np.sqrt(2 * g * weights.target_apex_m)
    e = float(weights.restitution_guess)
    # Approximate time to meet a constant-height surface. Dense guidance is
    # essential: an all-no-hit rollout must not get zero impact/apex cost.
    relative_vz = vel[..., 2] - tool_v[..., 2]
    discriminant = np.maximum(relative_vz**2 + 2*g*np.maximum(gap, 0.0), 0.0)
    tau = np.clip((relative_vz + np.sqrt(discriminant))/g, 0.0, 0.8)
    ball_at = pos[..., :2] + tau[..., None]*vel[..., :2]
    tool_at = tool[..., :2] + tau[..., None]*tool_v[..., :2]
    landing_err = np.sum(((ball_at - tool_at)/usable)**2, axis=-1)
    # Impact equation: v_out=(1+e)*v_racket-e*v_in.
    ball_v_in = vel[..., 2] - g*tau
    desired_tool_v = (desired_vz + e*ball_v_in)/(1+e)
    predicted_outgoing = (1+e)*tool_v[..., 2] - e*ball_v_in
    preimpact_err = ((tool_v[..., 2] - desired_tool_v)/max(desired_vz, 0.5))**2
    predicted_impact_err = ((predicted_outgoing-desired_vz)/desired_vz)**2
    # Prioritize imminent contacts; still penalize bad prospective hits early.
    imminence = 0.25 + 0.75*np.exp(-tau/0.12)
    approaching = (relative_vz < 0.2) & (gap > -0.02)
    preimpact_cost = np.mean(np.where(approaching, imminence*preimpact_err, 0.), axis=1)
    preview_cost = np.mean(np.where(approaching, imminence*predicted_impact_err, 0.), axis=1)

    dvz = vel[:, 1:, 2] - vel[:, :-1, 2]
    hit = ((dvz > 0.12) & (vel[:, 1:, 2] > 0.1) &
           (gap[:, 1:] > -0.025) & (gap[:, 1:] < 0.04) &
           (np.max(np.abs(loc[:, 1:, :]), axis=-1) < usable + 0.012))
    events = hit.copy()
    if hit.shape[1] > 1:
        events[:, 1:] &= ~hit[:, :-1]
    n_hit = np.sum(events, axis=1)
    impact_err = ((vel[:, 1:, 2]-desired_vz)/desired_vz)**2
    actual_impact_cost = np.sum(np.where(events, impact_err, 0.), axis=1)/np.maximum(n_hit, 1)
    # A detected contact is scored by the actual outgoing speed. For no-hit
    # rollouts use the anticipated impact error rather than silently zeroing it.
    velocity_cost = np.where(n_hit > 0, actual_impact_cost, preview_cost)

    ascending = (vel[..., 2] > 0.12) & (gap > 0.01)
    apex_height = pos[..., 2]-tool[..., 2]-float(f['ball_radius']) + np.maximum(vel[..., 2], 0.)**2/(2*g)
    apex_error = ((apex_height-weights.target_apex_m)/weights.target_apex_m)**2
    # Only score apex when ascending; otherwise the pre-impact target handles
    # it. Normalize by active frames, so a single meaningful bounce matters.
    n_ascent = np.sum(ascending, axis=1)
    apex_cost = np.sum(np.where(ascending, apex_error, 0.), axis=1)/np.maximum(n_ascent, 1)

    missed = (gap < -0.055) | ((np.max(np.abs(loc), axis=-1) > usable+0.055) & (gap < 0.05))
    miss_cost = weights.miss*(np.mean(missed, axis=1) + 0.5*np.any(missed, axis=1))
    # Terminal matters, especially if a candidate reaches the impact late.
    terminal_landing = landing_err[:, -1]
    tilt_cost = ((1.-normal_z)/0.03)**2
    workspace = np.sum(((tool[..., :2]-tool[:, :1, :2])/0.18)**2, axis=-1)
    scale = np.maximum(np.asarray(ctrl_scale, dtype=float), 1e-6)
    prev = np.concatenate((np.broadcast_to(initial_ctrl, controls[:, :1].shape), controls[:, :-1]), axis=1)
    du = np.mean(((controls-prev)/scale)**2, axis=-1)
    proposal = np.mean(((controls-nominal[None, :, :])/scale)**2, axis=-1)
    return (
        weights.landing_xy*(0.6*np.mean(landing_err, axis=1)+0.4*terminal_landing)
        + weights.preimpact_velocity*preimpact_cost
        + weights.impact_velocity*velocity_cost
        + weights.flight_apex*apex_cost
        + weights.tool_tilt*np.mean(tilt_cost, axis=1)
        + weights.tool_workspace*np.mean(workspace, axis=1)
        + weights.control_delta*np.mean(du, axis=1)
        + weights.proposal*np.mean(proposal, axis=1)
        + miss_cost - weights.hit_bonus*np.minimum(n_hit, 3)
    )
