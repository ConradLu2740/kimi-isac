"""Unified training loop: val-based selection, early stopping, LR scheduling,
gradient clipping, logging, checkpoint metadata."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from kimi_isac.core import logging_setup, splits
from kimi_isac.core.rng import seed_all
from kimi_isac.ml.checkpoint import save_checkpoint
from kimi_isac.ml.dataset import SensingDataset
from kimi_isac.ml.evaluate import (
    calibrate_thresholds,
    classical_baseline,
    compute_metrics,
    predict,
)
from kimi_isac.ml.model import SensingNet
from kimi_isac.ml.scenario import ScenarioConfig


def build_loaders(split: dict, seed: int, cfg: ScenarioConfig, batch_size: int):
    train_ds = SensingDataset(split["train"], seed, cfg)
    val_ds = SensingDataset(split["val"], seed, cfg)
    test_ds = SensingDataset(split["test"], seed, cfg)
    ood_ds = SensingDataset(list(range(200, 260)), seed, cfg, ood=True)

    def mk(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return mk(train_ds, True), mk(val_ds, False), mk(test_ds, False), mk(ood_ds, False)


def loss_fn(
    pred: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], cfg: ScenarioConfig
) -> torch.Tensor:
    det = torch.nn.functional.binary_cross_entropy_with_logits(pred["detect"], batch["has_target"])
    has = batch["has_target"] > 0.5
    range_t = batch["range_m"][has] / cfg.range_max_m
    vel_t = batch["vel_mps"][has] / cfg.vel_max_mps
    reg = torch.nn.functional.mse_loss(pred["range"][has], range_t) + torch.nn.functional.mse_loss(
        pred["vel"][has], vel_t
    )
    cls = torch.nn.functional.cross_entropy(pred["class"], batch["class"])
    return det + reg + cls


def train_one(
    seed: int,
    cfg: ScenarioConfig,
    split: dict,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: str,
    out_dir: Path,
    log: logging.Logger,
) -> tuple[dict[str, float], dict[str, float]]:
    train_loader, val_loader, test_loader, ood_loader = build_loaders(split, seed, cfg, batch_size)
    model = SensingNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=max(1, patience // 2))
    config = {
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "split_hash": split["config_hash"],
        "n_train": len(split["train"]),
    }
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        tot = 0.0
        for batch in train_loader:
            maps = batch["map"].to(device)
            targets = {k: v.to(device) for k, v in batch.items() if k not in ("map", "map_full")}
            opt.zero_grad()
            loss = loss_fn(model(maps), targets, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tot += float(loss.item())
        val_loss = _val_loss(model, val_loader, cfg, device)
        sched.step(val_loss)
        if val_loss < best_val - 1e-5:
            best_val, best_epoch, bad_epochs = val_loss, epoch, 0
            save_checkpoint(
                out_dir / "best.pth",
                model,
                epoch=epoch,
                config=config,
                metrics={"val_loss": val_loss},
                optimizer=opt,
            )
        else:
            bad_epochs += 1
        log.info(
            "epoch %3d/%d train_loss=%.4f val_loss=%.4f lr=%.2e (%.1fs)",
            epoch + 1,
            epochs,
            tot / max(1, len(train_loader)),
            val_loss,
            opt.param_groups[0]["lr"],
            time.time() - t0,
        )
        if bad_epochs >= patience:
            log.info("early stopping at epoch %d (best epoch %d)", epoch + 1, best_epoch + 1)
            break
    from kimi_isac.ml.checkpoint import load_checkpoint

    load_checkpoint(out_dir / "best.pth", model)
    model.to(device)

    test_pred = predict(model, test_loader, device)
    ood_pred = predict(model, ood_loader, device)
    thresholds = calibrate_thresholds(val_loader, cfg)

    metrics: dict[str, float] = {}
    for name, pred in (("test", test_pred), ("ood", ood_pred)):
        baseline = classical_baseline(pred["map_full"], cfg, thresholds, pred["snr"])
        for snr in cfg.snr_levels:
            per = compute_metrics(pred, baseline, cfg, snr_filter=snr)
            metrics.update({f"{name}_snr{snr:g}_{k}": v for k, v in per.items()})
    return metrics, config


@torch.no_grad()
def _val_loss(model, loader, cfg, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for batch in loader:
        maps = batch["map"].to(device)
        targets = {k: v.to(device) for k, v in batch.items() if k not in ("map", "map_full")}
        tot += float(loss_fn(model(maps), targets, cfg).item())
        n += 1
    return tot / max(1, n)


def get_or_make_split(cfg: ScenarioConfig, seed: int, n_total: int, out_dir: Path) -> dict:
    path = out_dir / "splits" / f"sensing_{n_total}_{seed}.json"
    if path.exists():
        return splits.load_split(path)
    spec = splits.SplitSpec(n_total=n_total, seed=seed, name="sensing")
    split = splits.make_split(spec)
    splits.assert_disjoint(split)
    splits.save_split(split, path)
    return split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-total", type=int, default=600)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="tiny fast run for CI")
    args = parser.parse_args(argv)

    if args.smoke:
        args.epochs, args.n_total, args.batch_size = 2, 48, 16
    logging_setup.setup_logging()
    log = logging.getLogger("kimi_isac.ml.train")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    if args.device == "cuda":
        device = "cuda"
    seed_all(args.seed)

    cfg = ScenarioConfig()
    out_dir = Path(args.out) / "ml"
    split = get_or_make_split(cfg, args.seed, args.n_total, Path(args.out))
    metrics, _ = train_one(
        args.seed,
        cfg,
        split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=device,
        out_dir=out_dir,
        log=log,
    )
    for k, v in metrics.items():
        log.info("%s = %.4f", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
