"""Checkpoint helper: every checkpoint stores its metadata, not just weights.

Saves are atomic (temp file + os.replace) with retries: the workspace may
live inside a synced folder (OneDrive) that transiently locks files.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "config": config,
        "metrics": metrics,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)
            return path
        except (RuntimeError, OSError) as err:  # transient sync locks
            last_err = err
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"could not save checkpoint {path}: {last_err}")


def load_checkpoint(path: str | Path, model: torch.nn.Module) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return payload


def config_mismatches(payload: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Keys where the checkpoint config disagrees with the current settings."""
    saved = payload.get("config", {})
    return [k for k, v in expected.items() if saved.get(k) != v]
