"""Independent-channel, finite-horizon Gaussian pink (1/f) proposal noise."""
from __future__ import annotations

import numpy as np


def sample_pink_noise(rng: np.random.Generator, shape: tuple[int, int, int]) -> np.ndarray:
    """Return temporally correlated Gaussian proposals with unit marginal variance.

    The nonzero Fourier bins have power proportional to 1/f. The DC bin has
    finite power equal to the lowest nonzero bin, rather than being removed.
    Removing it would force every rollout's horizon-mean perturbation to zero,
    preventing MPPI from exploring sustained upward/downward control shifts.
    Each rollout and actuator channel is independently sampled.
    """
    n, h, channels = shape
    if min(n, h, channels) < 1:
        raise ValueError("pink noise shape dimensions must be positive")
    white = rng.standard_normal(shape)
    if h == 1:
        return white
    frequencies = np.fft.rfftfreq(h)
    weights = np.empty_like(frequencies)
    weights[1:] = 1.0 / np.sqrt(frequencies[1:])
    # Finite low-frequency continuation: preserve constant control offsets.
    weights[0] = weights[1]
    multiplicity = np.full(len(weights), 2.0)
    multiplicity[0] = 1.0
    if h % 2 == 0:
        multiplicity[-1] = 1.0
    expected_variance = np.dot(multiplicity, weights**2) / h
    weights /= np.sqrt(expected_variance)
    spectrum = np.fft.rfft(white, axis=1)
    spectrum *= weights[None, :, None]
    return np.fft.irfft(spectrum, n=h, axis=1)
