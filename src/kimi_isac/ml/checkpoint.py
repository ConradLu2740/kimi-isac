"""Checkpoint helper: every checkpoint stores its metadata, not just weights."""

from __future__ import annotations

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
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, model: torch.nn.Module) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return payload


def config_mismatches(payload: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Keys where the checkpoint config disagrees with the current settings."""
    saved = payload.get("config", {})
    return [k for k, v in expected.items() if saved.get(k) != v]
