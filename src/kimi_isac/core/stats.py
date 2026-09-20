"""Statistical helpers: bootstrap confidence intervals and multi-seed summaries.

Every headline metric in this project must be reported through
``summarize_runs`` (mean with a bootstrap 95% CI), never as a bare
single-run number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Percentile bootstrap: (mean, lo, hi) of the mean of ``values``."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("need at least 2 scalar samples")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(values.mean()), float(lo), float(hi)


@dataclass(frozen=True)
class Summary:
    n: int
    mean: float
    std: float
    ci95_lo: float
    ci95_hi: float

    def __str__(self) -> str:
        return f"{self.mean:.4g} +/- {(self.ci95_hi - self.ci95_lo) / 2:.3g} (95% CI, n={self.n})"


def summarize_runs(runs: list[float] | np.ndarray, seed: int = 0) -> Summary:
    """Aggregate repeated runs: mean, std, and bootstrap 95% CI of the mean."""
    values = np.asarray(runs, dtype=float)
    mean, lo, hi = bootstrap_ci(values, seed=seed)
    return Summary(
        n=int(values.size),
        mean=mean,
        std=float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        ci95_lo=lo,
        ci95_hi=hi,
    )


def paired_bootstrap_pvalue(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> float:
    """Two-sided paired bootstrap p-value for H0: mean(a - b) == 0."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size < 2:
        raise ValueError("paired samples must be equal-length (n >= 2)")
    diff = a - b
    centered = diff - diff.mean()  # resample under H0 (mean difference = 0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    boot_means = centered[idx].mean(axis=1)
    observed = float(diff.mean())
    frac = float(np.mean(np.abs(boot_means) >= abs(observed)))
    return max(frac, 1.0 / n_boot)
