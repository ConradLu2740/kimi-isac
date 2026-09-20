"""1D cell-averaging CFAR with analytic false-alarm calibration."""

from __future__ import annotations

import numpy as np


def alpha_ca_cfar(n_train: int, pfa: float) -> float:
    """Threshold multiplier for exponential noise power.

    The sum of N exponential training cells is Gamma(N); with the
    threshold scaled by the training MEAN, requiring the CUT to exceed
    it with probability ``pfa`` gives ``alpha = N (pfa^(-1/N) - 1)``.
    ``n_train`` is the TOTAL number of training cells (both sides).
    """
    if n_train < 1:
        raise ValueError("n_train must be >= 1")
    if not 0.0 < pfa < 1.0:
        raise ValueError("pfa must be in (0, 1)")
    return n_train * (pfa ** (-1.0 / n_train) - 1.0)


def _segment_sums(values: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """sum(values[starts[i]:stops[i]]) per i; NaN where the window hangs off the edge."""
    cs = np.concatenate([[0.0], np.cumsum(values)])
    s = np.clip(starts, 0, len(values))
    e = np.clip(stops, 0, len(values))
    out = cs[e] - cs[s]
    return np.where((starts >= 0) & (stops <= len(values)), out, np.nan)


def ca_cfar_1d(
    power: np.ndarray,
    n_train: int = 16,
    n_guard: int = 2,
    pfa: float = 1e-3,
) -> dict[str, np.ndarray]:
    """Cell-averaging CFAR over a 1D power profile.

    Training cells are the ``n_train`` cells on each side of the CUT,
    excluding the ``n_guard`` cells adjacent to the CUT. Detections are
    crossings of ``alpha * train_mean``, reduced to the maximum of each
    contiguous crossing run (one report per target).
    """
    power = np.asarray(power, dtype=float)
    if n_train < 1:
        raise ValueError("n_train must be >= 1")
    n = len(power)
    idx = np.arange(n)
    left = _segment_sums(power, idx - n_guard - n_train, idx - n_guard)
    right = _segment_sums(power, idx + n_guard + 1, idx + n_guard + n_train + 1)
    n_total = 2 * n_train
    train_mean = (left + right) / n_total
    threshold = alpha_ca_cfar(n_total, pfa) * train_mean
    crossings = power > threshold

    detections = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if crossings[i] and not np.isnan(threshold[i]):
            j = i
            while j + 1 < n and crossings[j + 1] and not np.isnan(threshold[j + 1]):
                j += 1
            detections[i + int(np.argmax(power[i : j + 1]))] = True
            i = j + 1
        else:
            i += 1
    return {"threshold": threshold, "detections": detections, "crossings": crossings}


def estimate_pfa_mc(
    n_trials: int = 100_000,
    n_train: int = 16,
    n_guard: int = 2,
    pfa: float = 1e-3,
    seed: int = 0,
) -> float:
    """Monte-Carlo false-alarm rate under noise-only cells (Swerling-0 model)."""
    rng = np.random.default_rng(seed)
    n_total = 2 * n_train
    alpha = alpha_ca_cfar(n_total, pfa)
    left = rng.exponential(1.0, size=(n_trials, n_train))
    right = rng.exponential(1.0, size=(n_trials, n_train))
    cut = rng.exponential(1.0, size=n_trials)
    threshold = alpha * (left.sum(axis=1) + right.sum(axis=1)) / n_total
    return float(np.mean(cut > threshold))
