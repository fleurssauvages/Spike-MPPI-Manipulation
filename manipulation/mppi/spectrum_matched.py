"""Gaussian proposal sampling matched to a finite-horizon source spectrum.

The matched sampler keeps only second-order frequency-domain statistics from a
reference proposal. It estimates the reference cross-spectrum from final,
pre-clipping perturbations and then applies a frequency-dependent linear filter
to white Gaussian noise. The result is Gaussian by construction while matching
per-channel power spectra and, when desired, cross-channel spectra.

For source channels that are known analytically to be independent and
identically distributed (the agonist/antagonist channels of the Poisson Ant-2
proposal), :func:`fit_independent_identical_spectral_factor` pools both channels
to estimate one marginal spectrum and sets the off-diagonal spectrum exactly to
zero. This avoids turning finite calibration noise into a spurious correlation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _as_sample_tensor(samples: np.ndarray) -> np.ndarray:
    """Return validated ``(N, H, C)`` float64 samples."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 2:
        x = x[:, :, None]
    if x.ndim != 3 or min(x.shape) < 1:
        raise ValueError("samples must have nonempty shape (N, H, C)")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples must be finite")
    return x


def estimate_cross_spectrum(samples: np.ndarray) -> np.ndarray:
    """Estimate the centered finite-horizon cross-spectrum.

    Parameters
    ----------
    samples:
        Array with shape ``(N, H, C)``. The ensemble mean at every horizon
        coordinate is removed before the FFT, matching the proposal covariance
        rather than any deterministic/nonzero mean.

    Returns
    -------
    np.ndarray
        Complex Hermitian array with shape ``(H//2 + 1, C, C)`` containing
        ``E[X_f X_f^H]`` under NumPy's unnormalised forward FFT convention.
    """
    x = _as_sample_tensor(samples)
    n = int(x.shape[0])
    x = x - np.mean(x, axis=0, keepdims=True)
    xf = np.fft.rfft(x, axis=1)
    # S[f,a,b] = E[X_f,a conj(X_f,b)].
    spectral = np.einsum("nfa,nfb->fab", xf, xf.conj(), optimize=True) / float(n)
    return 0.5 * (spectral + np.swapaxes(spectral.conj(), 1, 2))


def factor_from_cross_spectrum(spectral: np.ndarray, horizon: int) -> np.ndarray:
    """Return a frequency-domain square root suitable for white-noise shaping.

    For unit-variance real white noise ``w`` and ``W = rfft(w)``, NumPy's FFT
    normalization gives ``E[W_f W_f^H] = H I``. This routine therefore returns
    ``A_f`` satisfying ``H A_f A_f^H = S_f``.
    """
    h = max(1, int(horizon))
    s = np.asarray(spectral, dtype=np.complex128)
    expected_freq = h // 2 + 1
    if s.ndim != 3 or s.shape[0] != expected_freq:
        raise ValueError("spectral must have shape (H//2+1, C, C)")
    if s.shape[1] != s.shape[2] or s.shape[1] < 1:
        raise ValueError("spectral channel dimensions must be square and nonempty")
    if not np.all(np.isfinite(s)):
        raise ValueError("spectral must be finite")

    channels = int(s.shape[1])
    factors = np.empty_like(s)
    n_freq = int(s.shape[0])
    for f in range(n_freq):
        sf = 0.5 * (s[f] + s[f].conj().T)
        boundary = f == 0 or (h % 2 == 0 and f == n_freq - 1)
        if boundary:
            # DC (and Nyquist for even H) coefficients of a real sequence are
            # real. Force a real symmetric square root at those bins.
            real_spectral = np.asarray(sf.real, dtype=np.float64)
            real_spectral = 0.5 * (real_spectral + real_spectral.T)
            eigvals, eigvecs = np.linalg.eigh(real_spectral)
            eigvals = np.maximum(eigvals, 0.0)
            root = (eigvecs * np.sqrt(eigvals / float(h))[None, :]) @ eigvecs.T
            factors[f] = root.astype(np.complex128)
        else:
            eigvals, eigvecs = np.linalg.eigh(sf)
            eigvals = np.maximum(eigvals.real, 0.0)
            root = (
                (eigvecs * np.sqrt(eigvals / float(h))[None, :])
                @ eigvecs.conj().T
            )
            factors[f] = root

    if factors.shape != (n_freq, channels, channels):
        raise RuntimeError("unexpected spectral factor shape")
    return factors


def fit_cross_spectral_factor(samples: np.ndarray) -> np.ndarray:
    """Fit a square-root cross-spectrum for FFT filtering of white noise.

    If ``W`` is the rFFT of unit-variance white Gaussian noise and ``A`` is the
    returned factor, ``Y_f = A_f W_f`` has the empirical source cross-spectrum
    at frequency bin ``f`` in expectation.
    """
    x = _as_sample_tensor(samples)
    return factor_from_cross_spectrum(estimate_cross_spectrum(x), x.shape[1])


def fit_independent_identical_spectral_factor(samples: np.ndarray) -> np.ndarray:
    """Fit identical marginal spectra with exactly zero cross-spectrum.

    This is intended for channel groups whose construction guarantees that the
    channels are independent and identically distributed. All channels are
    pooled to estimate one marginal spectrum, improving calibration efficiency;
    the returned factor is diagonal with that same scalar filter on every
    channel. Hence the target off-diagonal cross-spectrum is exactly zero.

    The Poisson Ant-2 agonist and antagonist channels satisfy these assumptions:
    positive and negative events are independent Poisson processes with the same
    rate, mark law and twitch kernel.
    """
    x = _as_sample_tensor(samples)
    n, h, channels = x.shape
    if channels == 1:
        return fit_cross_spectral_factor(x)

    # Center each physical channel as the controller does before pooling.  The
    # pooled array contains N*C independent draws from the same marginal law.
    centered = x - np.mean(x, axis=0, keepdims=True)
    pooled = np.transpose(centered, (2, 0, 1)).reshape(channels * n, h, 1)
    scalar = fit_cross_spectral_factor(pooled)

    factor = np.zeros((scalar.shape[0], channels, channels), dtype=np.complex128)
    for c in range(channels):
        factor[:, c, c] = scalar[:, 0, 0]
    return factor


@dataclass
class SpectrumMatchedGaussianSampler:
    """Generate Gaussian sequences with a fitted temporal/cross-channel spectrum.

    ``groups`` independent copies of the fitted ``group_size``-channel process
    are generated. For torque Ant the group size is one. For Ant-2 the group
    size is two: Spike can retain the fitted agonist/antagonist cross-spectrum,
    whereas Poisson uses a diagonal factor because its two sign streams are
    analytically independent.
    """

    horizon: int
    groups: int
    spectral_factor: np.ndarray
    center_across_rollouts: bool = False

    def __post_init__(self) -> None:
        self.horizon = max(1, int(self.horizon))
        self.groups = max(1, int(self.groups))
        factor = np.asarray(self.spectral_factor, dtype=np.complex128)
        expected_freq = self.horizon // 2 + 1
        if factor.ndim != 3 or factor.shape[0] != expected_freq:
            raise ValueError("spectral_factor must have shape (H//2+1, C, C)")
        if factor.shape[1] != factor.shape[2] or factor.shape[1] < 1:
            raise ValueError("spectral_factor channel dimensions must be square and nonempty")
        if not np.all(np.isfinite(factor)):
            raise ValueError("spectral_factor must be finite")
        self.spectral_factor = factor
        self.group_size = int(factor.shape[1])

    @property
    def nu(self) -> int:
        return self.groups * self.group_size

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        n = max(1, int(n))
        white = rng.standard_normal((n, self.horizon, self.groups, self.group_size))
        white_f = np.fft.rfft(white, axis=1)
        shaped_f = np.einsum(
            "nfgc,fac->nfga",
            white_f,
            self.spectral_factor,
            optimize=True,
        )
        shaped = np.fft.irfft(shaped_f, n=self.horizon, axis=1)
        out = np.asarray(shaped.reshape(n, self.horizon, self.nu), dtype=np.float64)
        if self.center_across_rollouts:
            out -= np.mean(out, axis=0, keepdims=True)
        return out
