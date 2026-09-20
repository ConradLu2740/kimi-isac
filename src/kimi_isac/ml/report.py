"""Multi-seed ML report: every metric as mean +/- bootstrap 95% CI.

Usage:
    python -m kimi_isac.ml.report [--seeds 5] [--epochs 30] [--smoke]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from kimi_isac.core import logging_setup
from kimi_isac.core.rng import seed_all
from kimi_isac.ml.evaluate import summarize_metrics
from kimi_isac.ml.scenario import ScenarioConfig
from kimi_isac.ml.train import get_or_make_split, train_one


def run_report(
    seeds: int,
    epochs: int,
    batch_size: int,
    patience: int,
    n_total: int,
    out: Path,
    device: str,
    log: logging.Logger,
    lr: float = 1e-3,
) -> dict:
    cfg = ScenarioConfig()
    split = get_or_make_split(cfg, seeds_for_split(seeds), n_total, out)
    runs = []
    for i in range(seeds):
        seed = 1000 + i
        seed_all(seed)
        log.info("=== seed %d (%d/%d) ===", seed, i + 1, seeds)
        metrics, _ = train_one(
            seed,
            cfg,
            split,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            device=device,
            lr=lr,
            out_dir=out / "ml" / f"seed_{seed}",
            log=log,
        )
        metrics["seed"] = float(seed)
        runs.append(metrics)
    summaries = summarize_metrics(runs)
    report = {
        "n_seeds": seeds,
        "metrics": {
            k: {"mean": s.mean, "std": s.std, "ci95": [s.ci95_lo, s.ci95_hi], "n": s.n}
            for k, s in summaries.items()
        },
        "runs": runs,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "ml_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for k, s in summaries.items():
        log.info("%-28s %s", k, s)
    return report


def seeds_for_split(seeds: int) -> int:
    """Split seed is independent of the run seeds and fixed for a given arity."""
    return 42


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--n-total", type=int, default=600)
    parser.add_argument("--out", type=str, default="results")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.seeds, args.epochs, args.n_total, args.batch_size = 2, 2, 48, 16

    logging_setup.setup_logging()
    log = logging.getLogger("kimi_isac.ml.report")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    run_report(
        args.seeds,
        args.epochs,
        args.batch_size,
        args.patience,
        args.n_total,
        Path(args.out),
        device,
        log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
