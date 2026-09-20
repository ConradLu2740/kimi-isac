"""Multi-seed generative report: unconditional CD, conditional CD curves.

Usage:
    python -m kimi_isac.gen.report [--seeds 10] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from kimi_isac.core import logging_setup, stats as stats_mod
from kimi_isac.core.rng import seed_all
from kimi_isac.gen import dataset as ds
from kimi_isac.gen import train as gen_train


def run_report(
    seeds: int,
    epochs_vae: int,
    epochs_dit: int,
    out: Path,
    device: str,
    log: logging.Logger,
    cfg: ds.GenConfig | None = None,
    T: int = 100,
    batch_size: int = 64,
    patience: int = 8,
) -> dict:
    cfg = cfg or ds.GenConfig()
    runs = []
    for i in range(seeds):
        seed = 1000 + i
        seed_all(seed)
        log.info("=== gen seed %d (%d/%d) ===", seed, i + 1, seeds)
        runs.append(
            gen_train.run_gen(
                seed=seed,
                cfg=cfg,
                epochs_vae=epochs_vae,
                epochs_dit=epochs_dit,
                device=device,
                out_dir=out / "gen" / f"seed_{seed}",
                log=log,
                T=T,
                batch_size=batch_size,
                patience=patience,
            )
        )

    def agg(key: str) -> dict:
        s = stats_mod.summarize_runs([r[key] for r in runs], seed=0)
        return {
            "mean": s.mean,
            "std": s.std,
            "ci95": [s.ci95_lo, s.ci95_hi],
            "n": s.n,
        }

    gaps = [r["uncond_cd_heldout"] - r["uncond_cd_train"] for r in runs]
    gap = stats_mod.summarize_runs(gaps, seed=0)

    cells: dict[str, list[float]] = {}
    for run in runs:
        per_key: dict[str, list[float]] = {}
        for (snr, ris), cd in run["cond_cd_cells"]:
            per_key.setdefault(f"snr{snr:g}_{ris}", []).append(cd)
        for k, v in per_key.items():
            cells.setdefault(k, []).append(float(np.mean(v)))
    for key, values in cells.items():
        assert len(values) == seeds, f"cell {key} has n={len(values)} != {seeds}"
    cond = {}
    for k, v in sorted(cells.items()):
        s = stats_mod.summarize_runs(v, seed=0)
        cond[k] = {"mean": s.mean, "std": s.std, "ci95": [s.ci95_lo, s.ci95_hi], "n": s.n}

    report = {
        "n_seeds": seeds,
        "config_hash": cfg.config_hash(),
        "T": T,
        "vae_recon_cd": agg("vae_recon_cd"),
        "uncond_cd_heldout_inst": agg("uncond_cd_heldout"),
        "uncond_cd_heldout_class": agg("uncond_cd_ood"),
        "memorization_gap": {
            "mean": gap.mean,
            "std": gap.std,
            "ci95": [gap.ci95_lo, gap.ci95_hi],
            "n": gap.n,
        },
        "cond_cd": cond,
        "cond_cd_oracle": agg("cond_cd_oracle"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "gen_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
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
        args.seeds = 2
        args.epochs_vae, args.epochs_dit, args.T = 1, 1, 20
        args.n_total, args.n_elem, args.tau, args.batch_size = 24, 16, 4, 8

    logging_setup.setup_logging()
    log = logging.getLogger("kimi_isac.gen.report")
    device = "cuda" if args.device == "auto" and __import__("torch").cuda.is_available() else "cpu"

    cfg = ds.GenConfig(n_total=args.n_total, seed=42, tau=args.tau, n_elem=args.n_elem)
    report = run_report(
        seeds=args.seeds,
        epochs_vae=args.epochs_vae,
        epochs_dit=args.epochs_dit,
        out=Path(args.out),
        device=device,
        log=log,
        cfg=cfg,
        T=args.T,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    for key in (
        "vae_recon_cd",
        "uncond_cd_heldout_inst",
        "uncond_cd_heldout_class",
        "memorization_gap",
        "cond_cd_oracle",
    ):
        m = report[key]
        log.info(
            "%-24s %.4f +/- %.4f (95%% CI, n=%d)",
            key,
            m["mean"],
            (m["ci95"][1] - m["ci95"][0]) / 2,
            m["n"],
        )
    for key, m in report["cond_cd"].items():
        log.info(
            "cond_cd[%-16s] %.4f +/- %.4f (n=%d)",
            key,
            m["mean"],
            (m["ci95"][1] - m["ci95"][0]) / 2,
            m["n"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
