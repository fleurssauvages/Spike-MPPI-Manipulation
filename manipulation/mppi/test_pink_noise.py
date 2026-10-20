"""Lightweight regression tests for the standalone pink proposal generator."""
import numpy as np

from .pink_noise import sample_pink_noise


def test_pink_noise_power_and_scale():
    noise = sample_pink_noise(np.random.default_rng(14), (2048, 64, 2))
    assert noise.shape == (2048, 64, 2)
    assert 0.92 < float(np.var(noise)) < 1.08
    power = np.mean(np.abs(np.fft.rfft(noise, axis=1)) ** 2, axis=(0, 2))
    assert 6.5 < power[1] / power[8] < 9.5
    # The DC component must be present: per-rollout mean shifts are essential
    # for exploring persistent control offsets over the horizon.
    horizon_means = noise.mean(axis=1)
    assert float(np.std(horizon_means)) > 0.1
    assert abs(float(horizon_means.mean())) < 0.05


def test_pink_noise_short_horizons():
    for horizon in (1, 2, 3, 20, 21):
        noise = sample_pink_noise(np.random.default_rng(1), (4, horizon, 7))
        assert noise.shape == (4, horizon, 7)
        assert np.isfinite(noise).all()


def test_pink_weighted_update_can_shift_horizon_mean():
    """Regression: cost-weighted MPPI should learn a sustained offset."""
    noise = sample_pink_noise(np.random.default_rng(7), (16000, 32, 1))[:, :, 0]
    # Toy cost prefers a constant positive shift in the action sequence.
    costs = np.mean((noise - 0.7) ** 2, axis=1)
    weights = np.exp(-(costs - np.min(costs)) / 0.08)
    weights /= weights.sum()
    updated = np.einsum("n,nh->h", weights, noise)
    assert float(updated.mean()) > 0.15
