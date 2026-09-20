"""Delay estimation Cramer-Rao bound for the point-target OFDM model.

Model: H(f_k) = exp(-j 2 pi f_k tau) + w, w complex CN(0, sigma^2).
For a deterministic signal in complex AWGN the Fisher information for the
delay is I(tau) = (2/sigma^2) * sum |dH/dtau|^2, hence

    var(tau_hat) >= sigma^2 / (8 pi^2 sum f_k^2).
"""

from __future__ import annotations

import numpy as np

from kimi_isac.core.constants import C_LIGHT


def delay_crb_s(freqs_hz: np.ndarray, noise_variance: float) -> float:
    """CRB standard deviation of the delay estimate (seconds)."""
    f2_sum = float(np.sum(np.asarray(freqs_hz, dtype=float) ** 2))
    if noise_variance <= 0.0 or f2_sum <= 0.0:
        raise ValueError("noise_variance must be positive and freqs must be nonzero")
    return float(np.sqrt(noise_variance / (8.0 * np.pi**2 * f2_sum)))


def range_crb_m(freqs_hz: np.ndarray, noise_variance: float) -> float:
    """CRB standard deviation of the round-trip range estimate (meters)."""
    return delay_crb_s(freqs_hz, noise_variance) * C_LIGHT / 2.0


def estimate_range_mc(
    freqs_hz: np.ndarray,
    range_m: float,
    noise_variance: float,
    n_trials: int = 500,
    grid_range_m: float = 200.0,
    grid_points: int = 20001,
    seed: int = 0,
) -> float:
    """Monte-Carlo RMSE of maximum-likelihood range estimation.

    ML estimate: the tau maximizing Re{sum_k H_k exp(j 2 pi f_k tau)} —
    the real part (matched-filter output), not the envelope.
    """
    rng = np.random.default_rng(seed)
    # dense grid around truth; spacing << CRB
    tau_grid = np.linspace(
        2.0 * (range_m - grid_range_m) / C_LIGHT,
        2.0 * (range_m + grid_range_m) / C_LIGHT,
        grid_points,
    )
    ref = np.exp(1j * 2.0 * np.pi * np.outer(freqs_hz, tau_grid))
    truth_tau = 2.0 * range_m / C_LIGHT
    estimates = np.empty(n_trials)
    for t in range(n_trials):
        h = np.exp(-1j * 2.0 * np.pi * freqs_hz * truth_tau)
        h = h + np.sqrt(noise_variance / 2.0) * (
            rng.normal(size=len(freqs_hz)) + 1j * rng.normal(size=len(freqs_hz))
        )
        corr = np.real(ref.T @ h)
        estimates[t] = tau_grid[int(np.argmax(corr))]
    return float(np.std(estimates) * C_LIGHT / 2.0)


def delay_fim_hz2(freqs_hz: np.ndarray, noise_variance: float) -> float:
    """Fisher information I(tau) = 8 pi^2 sum f^2 / sigma^2 (diagnostic)."""
    f2_sum = float(np.sum(np.asarray(freqs_hz, dtype=float) ** 2))
    return 8.0 * np.pi**2 * f2_sum / noise_variance
