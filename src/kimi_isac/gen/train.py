"""Two-stage generative training: VAE then latent DiT, on the shared engine.

Stage 1 trains the PointVAE (reconstruction + KL). Stage 2 freezes the
VAE, encodes all conditioning samples to latents, and trains the latent
DiT to predict DDPM noise. One model serves both tasks: with probability
``cond_drop`` the feature sequence is zeroed (unconditional path), so
sampling with zeroed features is class-conditional generation and
sampling with real features is echo-conditioned reconstruction.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kimi_isac.core import logging_setup
from kimi_isac.core.rng import seed_all
from kimi_isac.gen import dataset as ds
from kimi_isac.gen import dit, metrics, vae
from kimi_isac.ml.checkpoint import load_checkpoint
from kimi_isac.training.engine import EngineHooks, run_training

COND_DROP = 0.5
N_EVAL_UNCOND = 16


def _loader(dataset, batch_size: int, shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _recon_cd(model, loader, device: str) -> float:
    model.eval()
    cds = []
    for batch in loader:
        out = model(batch["cloud"].to(device))
        for a, b in zip(out["x_hat"].cpu().numpy(), batch["cloud"].numpy()):
            cds.append(metrics.chamfer_distance(a, b))
    return float(np.mean(cds))


def _vae_val(model, loader, device: str) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["cloud"].to(device)
            tot += float(vae.vae_loss(model(x), x).item())
            n += 1
    return tot / max(1, n)


def _make_opt(model: torch.nn.Module, lr: float, patience: int):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=max(1, patience // 2))
    return opt, sched


def _train_vae(split, cfg, device, out_dir, log, epochs, batch_size, lr, patience):
    train_loader = _loader(ds.UncondDataset(split["train"], cfg.seed, cfg), batch_size, True)
    val_loader = _loader(ds.UncondDataset(split["val"], cfg.seed, cfg), batch_size, False)

    def _train_epoch(model, opt, _):
        model.train()
        tot = 0.0
        for batch in train_loader:
            x = batch["cloud"].to(device)
            opt.zero_grad()
            loss = vae.vae_loss(model(x), x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
        return tot / max(1, len(train_loader))

    hooks = EngineHooks(
        build_model=vae.PointVAE,
        make_optimizer=lambda m: _make_opt(m, lr, patience),
        train_epoch=_train_epoch,
        val_loss=lambda m, _: _vae_val(m, val_loader, device),
        config={"stage": "vae", "seed": cfg.seed, "hash": cfg.config_hash()},
        out_dir=Path(out_dir) / "vae",
        device=device,
    )
    run_training(hooks, epochs=epochs, patience=patience, log=log)
    model = vae.PointVAE().to(device)
    load_checkpoint(Path(out_dir) / "vae" / "best.pth", model)
    return model


class _LatentCondData:
    """Precomputed (z, class, features, snr, ris_mode) tuples for DiT training."""

    def __init__(self, cond_dataset: ds.CondDataset, vae_model, device: str):
        zs, classes, feats, snrs, modes = [], [], [], [], []
        vae_model.eval()
        with torch.no_grad():
            for i in range(len(cond_dataset)):
                item = cond_dataset[i]
                cloud = item["cloud"].unsqueeze(0).to(device)
                mu, _ = vae_model.encode(cloud)
                zs.append(mu.squeeze(0).cpu())
                classes.append(int(item["class"]))
                feats.append(item["features"])
                snrs.append(float(item["snr"]))
                modes.append(item["ris_mode"])
        self.z = torch.stack(zs)
        self.cls = torch.tensor(classes, dtype=torch.long)
        self.feats = torch.stack(feats)
        self.snr = torch.tensor(snrs)
        self.modes = modes

    def __len__(self) -> int:
        return len(self.modes)


def _train_dit(
    data: _LatentCondData,
    device,
    out_dir,
    log,
    epochs,
    batch_size,
    lr,
    patience,
    T: int,
    n_classes: int,
):
    enc = dit.CondEncoder(
        n_classes=n_classes, feat_dim=data.feats.shape[-1], tau=data.feats.shape[1]
    ).to(device)
    model = dit.LatentDiT().to(device)
    sched_obj = dit.DDPMScheduler(T=T)
    params = list(enc.parameters()) + list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=max(1, patience // 2))

    def _batch(indices):
        return (
            data.z[indices].to(device),
            data.cls[indices].to(device),
            data.feats[indices].to(device),
        )

    best_val = float("inf")
    bad = 0
    n = len(data)
    for epoch in range(epochs):
        model.train()
        enc.train()
        perm = torch.randperm(n)
        tot = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            z, cls, feats = _batch(idx)
            # conditioning dropout: zero features for the unconditional path
            keep = (torch.rand(len(idx), device=device) > COND_DROP).float()[:, None, None]
            feats_in = feats * keep
            pooled, tokens = enc(cls, feats_in)
            t = torch.randint(0, T, (len(idx),), device=device)
            noise = torch.randn_like(z)
            z_noisy = sched_obj.add_noise(z, t, noise)
            loss = torch.nn.functional.mse_loss(model(z_noisy, t, pooled, tokens), noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += float(loss.item())
        val = _dit_val(data, enc, model, sched_obj, device, batch_size, T)
        sched.step(val)
        if val < best_val - 1e-6:
            best_val, bad = val, 0
            from kimi_isac.ml.checkpoint import save_checkpoint

            save_checkpoint(
                Path(out_dir) / "dit" / "best.pth",
                model,
                epoch=epoch,
                config={"stage": "dit", "T": T, "cond_drop": COND_DROP},
                metrics={"val_loss": val},
                optimizer=opt,
            )
            save_checkpoint(
                Path(out_dir) / "dit" / "enc_best.pth",
                enc,
                epoch=epoch,
                config={"stage": "cond_encoder"},
                metrics={"val_loss": val},
            )
        else:
            bad += 1
        log.info(
            "dit epoch %3d/%d train=%.4f val=%.4f lr=%.2e",
            epoch + 1,
            epochs,
            tot / max(1, (n + batch_size - 1) // batch_size),
            val,
            opt.param_groups[0]["lr"],
        )
        if bad >= patience:
            log.info("dit early stop at epoch %d", epoch + 1)
            break
    model = dit.LatentDiT().to(device)
    load_checkpoint(Path(out_dir) / "dit" / "best.pth", model)
    enc2 = dit.CondEncoder(
        n_classes=n_classes, feat_dim=data.feats.shape[-1], tau=data.feats.shape[1]
    ).to(device)
    load_checkpoint(Path(out_dir) / "dit" / "enc_best.pth", enc2)
    return enc2, model


def _dit_val(data, enc, model, sched_obj, device, batch_size, T) -> float:
    enc.eval()
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = torch.arange(start, min(start + batch_size, len(data)))
            z = data.z[idx].to(device)
            cls = data.cls[idx].to(device)
            feats = data.feats[idx].to(device)
            pooled, tokens = enc(cls, feats)
            t = torch.randint(0, T, (len(idx),), device=device)
            noise = torch.randn_like(z)
            z_noisy = sched_obj.add_noise(z, t, noise)
            tot += float(
                torch.nn.functional.mse_loss(model(z_noisy, t, pooled, tokens), noise).item()
            )
            n += 1
    return tot / max(1, n)


@torch.no_grad()
def _sample_cds(
    vae_model,
    enc,
    model,
    sched_obj,
    items: list[dict],
    device: str,
    conditional: bool,
    tau: int,
    n_eval: int,
) -> list[float]:
    """Sample clouds (optionally conditioned on features) and compute CD."""
    if not items:
        return []
    model.eval()
    enc.eval()
    feat_dim = enc.feat_proj.in_features
    cds = []
    for start in range(0, min(len(items), n_eval), 8):
        batch = items[start : start + 8]
        cls = torch.tensor([it["class"] for it in batch], dtype=torch.long, device=device)
        if conditional:
            feats = torch.stack([it["features"] for it in batch]).to(device)
        else:
            feats = torch.zeros(len(batch), tau, feat_dim, device=device)
        pooled, tokens = enc(cls, feats)
        z = sched_obj.sample(
            model, pooled, tokens, shape=(len(batch), model.n_tokens * 32), device=device
        )
        clouds = vae_model.decode(z).cpu().numpy()
        for cloud, it in zip(clouds, batch):
            cds.append(metrics.chamfer_distance(cloud, it["cloud"].numpy()))
    return cds


def run_gen(
    seed,
    cfg,
    epochs_vae,
    epochs_dit,
    device,
    out_dir,
    log,
    T: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 6,
) -> dict:
    seed_all(seed)
    out_dir = Path(out_dir)
    split = ds.get_or_make_split(cfg, out_dir.parent if out_dir.parent.name else out_dir)
    cache_dir = out_dir.parent / "gen_cache"

    vae_model = _train_vae(split, cfg, device, out_dir, log, epochs_vae, batch_size, lr, patience)

    heldout_loader = _loader(ds.UncondDataset(split["test"], cfg.seed, cfg), batch_size, False)
    vae_recon_cd = _recon_cd(vae_model, heldout_loader, device)

    # DiT stage on conditional samples (features + clouds) from the train split
    train_cond = ds.CondDataset(split["train"], cfg.seed, cfg, cache_dir=cache_dir)
    data = _LatentCondData(train_cond, vae_model, device)
    n_classes = len(ds.templates.all_class_names())
    enc, model = _train_dit(
        data, device, out_dir, log, epochs_dit, batch_size, lr, patience, T, n_classes
    )
    sched_obj = dit.DDPMScheduler(T=T)

    # unconditional CDs (class-conditional generation vs stratified references)
    train_items = [
        ds.UncondDataset(split["train"], cfg.seed, cfg)[i]
        for i in range(min(N_EVAL_UNCOND, len(split["train"])))
    ]
    heldout_items = ds.make_uncond_eval_items(cfg, cfg.seed, ood=False, per_class=2)
    ood_items = ds.make_uncond_eval_items(cfg, cfg.seed, ood=True, per_class=8)
    uncond_train = _sample_cds(
        vae_model,
        enc,
        model,
        sched_obj,
        train_items,
        device,
        conditional=False,
        tau=cfg.tau,
        n_eval=N_EVAL_UNCOND,
    )
    uncond_heldout = _sample_cds(
        vae_model,
        enc,
        model,
        sched_obj,
        heldout_items,
        device,
        conditional=False,
        tau=cfg.tau,
        n_eval=len(heldout_items),
    )
    uncond_ood = _sample_cds(
        vae_model,
        enc,
        model,
        sched_obj,
        ood_items,
        device,
        conditional=False,
        tau=cfg.tau,
        n_eval=len(ood_items),
    )

    # conditional CDs on the stratified (snr, ris_mode, class) grid
    cell_items = ds.make_cell_eval_items(cfg, cfg.seed, per_class=2, cache_dir=cache_dir)
    cond_cells: list[tuple[tuple[float, str], float]] = []
    aligned_cds: list[float] = []
    with torch.no_grad():
        for start in range(0, len(cell_items), 8):
            batch = cell_items[start : start + 8]
            cls = torch.tensor([it["class"] for it in batch], dtype=torch.long, device=device)
            feats = torch.stack([it["features"] for it in batch]).to(device)
            pooled, tokens = enc(cls, feats)
            z = sched_obj.sample(
                model, pooled, tokens, shape=(len(batch), model.n_tokens * 32), device=device
            )
            clouds = vae_model.decode(z).cpu().numpy()
            for cloud, it in zip(clouds, batch):
                cd = metrics.chamfer_distance(cloud, it["cloud"].numpy())
                cond_cells.append(((float(it["snr"]), it["ris_mode"]), cd))
                if it["ris_mode"] == "aligned":
                    aligned_cds.append(cd)

    oracle_ds = ds.OracleCondDataset(cfg.seed, cfg, cache_dir=cache_dir)
    oracle_items = [oracle_ds[i] for i in range(len(oracle_ds))]
    oracle_cds = _sample_cds(
        vae_model,
        enc,
        model,
        sched_obj,
        oracle_items,
        device,
        conditional=True,
        tau=cfg.tau,
        n_eval=len(oracle_items),
    )

    return {
        "vae_recon_cd": float(vae_recon_cd),
        "uncond_cd_train": float(np.mean(uncond_train)) if uncond_train else float("nan"),
        "uncond_cd_heldout": float(np.mean(uncond_heldout)) if uncond_heldout else float("nan"),
        "uncond_cd_ood": float(np.mean(uncond_ood)) if uncond_ood else float("nan"),
        "cond_cd": float(np.mean(aligned_cds)) if aligned_cds else float("nan"),
        "cond_cd_oracle": float(np.mean(oracle_cds)) if oracle_cds else float("nan"),
        "cond_cd_cells": cond_cells,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--epochs-vae", type=int, default=100)
    parser.add_argument("--epochs-dit", type=int, default=60)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--n-total", type=int, default=600)
    parser.add_argument("--n-elem", type=int, default=256)
    parser.add_argument("--tau", type=int, default=8)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true", help="tiny fast run for CI")
    args = parser.parse_args(argv)

    if args.smoke:
        args.epochs_vae, args.epochs_dit, args.T = 1, 1, 20
        args.n_total, args.n_elem, args.tau, args.batch_size = 24, 16, 4, 8
    logging_setup.setup_logging()
    log = logging.getLogger("kimi_isac.gen.train")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    if args.device == "cuda":
        device = "cuda"

    cfg = ds.GenConfig(n_total=args.n_total, seed=args.seed, tau=args.tau, n_elem=args.n_elem)
    metrics_out = run_gen(
        seed=args.seed,
        cfg=cfg,
        epochs_vae=args.epochs_vae,
        epochs_dit=args.epochs_dit,
        device=device,
        out_dir=Path(args.out) / "gen",
        log=log,
        T=args.T,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    for k, v in metrics_out.items():
        if k != "cond_cd_cells":
            log.info("%s = %.4f", k, v)
    log.info("cond_cd_cells: %d samples", len(metrics_out["cond_cd_cells"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
