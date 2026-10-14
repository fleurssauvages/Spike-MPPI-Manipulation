"""Backend-selectable standard Gaussian MPPI proposal sampling.

The Numba path preserves the same proposal law as the NumPy reference:
independent standard-normal innovations are scaled per actuator, passed through
the same stationary AR(1) temporal filter, added to the nominal horizon, and
clipped to actuator bounds.  Draw order is rollout -> time -> actuator so a
fixed NumPy Generator produces the same scalar innovation sequence as the
reference array draw on supported NumPy/Numba versions.
"""
from __future__ import annotations

import math
import warnings
import numpy as np

from .fast_kernels import NUMBA_AVAILABLE, njit


@njit(cache=True, nogil=True, fastmath=False)
def standard_gaussian_numba(
    rng,
    nominal: np.ndarray,
    std: np.ndarray,
    ctrl_low: np.ndarray,
    ctrl_high: np.ndarray,
    rho: float,
    noise: np.ndarray,
    controls: np.ndarray,
) -> None:
    """Fill reusable ``noise`` and ``controls`` workspaces in place."""
    n, h, nu = controls.shape
    beta = math.sqrt(max(0.0, 1.0 - rho * rho))
    for i in range(n):
        for t in range(h):
            for u in range(nu):
                value = std[u] * rng.standard_normal()
                if t > 0:
                    value = rho * noise[i, t - 1, u] + beta * value
                noise[i, t, u] = value
                control = nominal[t, u] + value
                if control < ctrl_low[u]:
                    control = ctrl_low[u]
                elif control > ctrl_high[u]:
                    control = ctrl_high[u]
                controls[i, t, u] = control


_prepared = False
_prepare_error: Exception | None = None


def resolve_standard_sampler(requested: str) -> str:
    """Resolve and precompile the standard sampler without touching controller RNG."""
    global _prepared, _prepare_error
    requested = str(requested).strip().lower()
    if requested not in {"auto", "numpy", "numba"}:
        raise ValueError("standard sampler must be auto, numpy, or numba")
    if requested == "numpy":
        return "numpy"
    if not NUMBA_AVAILABLE or not hasattr(standard_gaussian_numba, "signatures"):
        if requested == "numba":
            raise RuntimeError("--standard-sampler numba requires a working Numba installation with JIT enabled")
        return "numpy"
    if not _prepared and _prepare_error is None:
        try:
            rng = np.random.default_rng(1907)
            nominal = np.zeros((3, 2), dtype=np.float64)
            std = np.array([0.3, 0.4], dtype=np.float64)
            low = np.array([-1.0, -1.0], dtype=np.float64)
            high = np.array([1.0, 1.0], dtype=np.float64)
            noise = np.empty((2, 3, 2), dtype=np.float64)
            controls = np.empty_like(noise)
            standard_gaussian_numba(rng, nominal, std, low, high, 0.25, noise, controls)
            if not np.all(np.isfinite(controls)):
                raise RuntimeError("compiled standard sampler returned non-finite controls")
            _prepared = True
        except Exception as exc:  # pragma: no cover - depends on local JIT runtime
            _prepare_error = exc
            if requested == "auto":
                warnings.warn(
                    f"Standard Gaussian acceleration unavailable; using NumPy: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
    if not _prepared:
        if requested == "numba":
            raise RuntimeError("Numba could not compile the standard Gaussian sampler") from _prepare_error
        return "numpy"
    return "numba"
