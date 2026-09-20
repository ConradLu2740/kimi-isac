"""Evaluation: ML heads vs the classical peak-picking baseline on the SAME samples.

The classical baseline sees the full-resolution log range-Doppler map; the
ML head sees its pooled version. Both operate on identical channel
realizations (same sample index). Metrics are reported per SNR level,
because the classical detector is calibrated per level while the ML head
is not.
"""

from __future__ import annotations

import numpy as np
import torch

from kimi_isac.core import stats as stats_mod
from kimi_isac.ml.model import SensingNet
from kimi_isac.ml.scenario import ScenarioConfig


@torch.no_grad()
def predict(model: SensingNet, loader, device: str) -> dict[str, np.ndarray]:
    model.eval()
    out: dict[str, list[np.ndarray]] = {
        "detect_p": [],
        "range": [],
        "vel": [],
        "class": [],
        "has_target": [],
        "range_true": [],
        "vel_true": [],
        "class_true": [],
        "map_full": [],
        "snr": [],
    }
    for batch in loader:
        pred = model(batch["map"].to(device))
        out["detect_p"].append(torch.sigmoid(pred["detect"]).cpu().numpy())
        out["range"].append(pred["range"].cpu().numpy())
        out["vel"].append(pred["vel"].cpu().numpy())
        out["class"].append(pred["class"].argmax(dim=-1).cpu().numpy())
        out["has_target"].append(batch["has_target"].numpy())
        out["range_true"].append(batch["range_m"].numpy())
        out["vel_true"].append(batch["vel_mps"].numpy())
        out["class_true"].append(batch["class"].numpy())
        out["map_full"].append(batch["map_full"].numpy())
        out["snr"].append(batch["snr"].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


@torch.no_grad()
def calibrate_thresholds(val_loader, cfg: ScenarioConfig) -> dict[float, float]:
    """Per-SNR 99th-percentile noise-only peak of the full-resolution maps."""
    peaks: dict[float, list[np.ndarray]] = {}
    for batch in val_loader:
        noise_only = batch["has_target"].numpy() < 0.5
        if not noise_only.any():
            continue
        maps = batch["map_full"].numpy()[noise_only]
        snrs = batch["snr"].numpy()[noise_only]
        for s in np.unique(snrs):
            peaks.setdefault(float(s), []).append(maps[snrs == s].max(axis=(1, 2)))
    return {s: float(np.quantile(np.concatenate(v), 0.99)) for s, v in peaks.items()}


def classical_baseline(
    maps_full: np.ndarray, cfg: ScenarioConfig, thresholds: dict[float, float], snr: np.ndarray
) -> dict[str, np.ndarray]:
    """Per-SNR energy-threshold detection + peak-picking localization."""
    thr = np.array([thresholds.get(float(s), max(thresholds.values())) for s in snr])
    detect = maps_full.max(axis=(1, 2)) > thr
    flat = maps_full.reshape(maps_full.shape[0], -1).argmax(axis=1)
    n_doppler = maps_full.shape[2]
    r_bin = flat // n_doppler
    d_bin = flat % n_doppler
    return {
        "detect": detect.astype(np.float32),
        "range": ((r_bin + 0.5) * cfg.range_resolution_m).astype(np.float32),
        "vel": cfg.doppler_bins_mps[d_bin].astype(np.float32),
    }


def compute_metrics(
    pred: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    cfg: ScenarioConfig,
    snr_filter: float | None = None,
) -> dict[str, float]:
    has = pred["has_target"] > 0.5
    if snr_filter is None:
        subset = np.ones(has.shape, dtype=bool)
    else:
        subset = np.abs(pred["snr"] - snr_filter) < 1e-6
    det_true = has[subset]
    detect_acc = float(np.mean((pred["detect_p"][subset] > 0.5) == det_true))
    sel = subset & has  # target-present samples in this SNR group

    ml_range_rmse = float(
        np.sqrt(np.mean((pred["range"][sel] * cfg.range_max_m - pred["range_true"][sel]) ** 2))
    )
    cl_range_rmse = float(np.sqrt(np.mean((baseline["range"][sel] - pred["range_true"][sel]) ** 2)))
    ml_vel_rmse = float(
        np.sqrt(np.mean((pred["vel"][sel] * cfg.vel_max_mps - pred["vel_true"][sel]) ** 2))
    )
    cl_vel_rmse = float(np.sqrt(np.mean((baseline["vel"][sel] - pred["vel_true"][sel]) ** 2)))
    cls_acc = float(np.mean(pred["class"][sel] == pred["class_true"][sel]))
    return {
        "detect_acc": detect_acc,
        "classical_detect_acc": float(np.mean(baseline["detect"][subset] == det_true)),
        "ml_range_rmse_m": ml_range_rmse,
        "classical_range_rmse_m": cl_range_rmse,
        "ml_vel_rmse_mps": ml_vel_rmse,
        "classical_vel_rmse_mps": cl_vel_rmse,
        "class_acc": cls_acc,
    }


def summarize_metrics(runs: list[dict[str, float]]) -> dict[str, stats_mod.Summary]:
    keys = runs[0].keys()
    return {k: stats_mod.summarize_runs([r[k] for r in runs], seed=0) for k in keys}
