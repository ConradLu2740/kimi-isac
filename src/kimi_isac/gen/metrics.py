"""Chamfer distance (symmetric, mean-squared form) and CD aggregation."""

from __future__ import annotations

import numpy as np

from kimi_isac.core import stats as stats_mod


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0.5 * (mean_{x in a} min_y |x-y|^2 + mean_{y in b} min_x |y-x|^2)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 3 or b.shape[1] != 3:
        raise ValueError("expected (N, 3) clouds")
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
    return float(0.5 * (d2.min(axis=1).mean() + d2.min(axis=0).mean()))


def summarize_cd(runs: list[float]) -> stats_mod.Summary:
    return stats_mod.summarize_runs(runs, seed=0)
