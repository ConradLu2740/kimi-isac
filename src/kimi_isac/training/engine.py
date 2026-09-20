"""Shared training-loop engine: val-based selection, early stopping, LR
scheduling, gradient clipping, checkpoint metadata. One implementation,
used by every trainer in this project."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from kimi_isac.ml.checkpoint import load_checkpoint, save_checkpoint


@dataclass
class EngineHooks:
    """Model-agnostic callbacks the engine drives each epoch."""

    build_model: Callable[[], nn.Module]
    make_optimizer: Callable[[nn.Module], tuple]
    train_epoch: Callable[[nn.Module, torch.optim.Optimizer, object], float]
    val_loss: Callable[[nn.Module, object], float]
    config: dict
    out_dir: Path
    clip_norm: float = 1.0
    device: str = "cpu"


def run_training(hooks: EngineHooks, *, epochs: int, patience: int, log: logging.Logger) -> dict:
    """Run the standard loop. Best checkpoint is loaded back into the model.

    Loaders travel through the hook closures; the engine stays model-agnostic.
    """
    model = hooks.build_model().to(hooks.device)
    opt, sched = hooks.make_optimizer(model)
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    epoch = -1
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_loss = hooks.train_epoch(model, opt, None)
        val = hooks.val_loss(model, None)
        if sched is not None:
            sched.step(val)
        if val < best_val - 1e-5:
            best_val, best_epoch, bad = val, epoch, 0
            save_checkpoint(
                hooks.out_dir / "best.pth",
                model,
                epoch=epoch,
                config=hooks.config,
                metrics={"val_loss": val},
                optimizer=opt,
            )
        else:
            bad += 1
        log.info(
            "epoch %3d/%d train=%.4f val=%.4f lr=%.2e (%.1fs)",
            epoch + 1,
            epochs,
            train_loss,
            val,
            opt.param_groups[0]["lr"],
            time.time() - t0,
        )
        if bad >= patience:
            log.info("early stop at epoch %d (best %d)", epoch + 1, best_epoch + 1)
            break
    load_checkpoint(hooks.out_dir / "best.pth", model)
    model.to(hooks.device)
    return {"best_val": best_val, "best_epoch": best_epoch, "epochs_run": epoch + 1}
