"""Reproducible figures for TECH_REPORT.md.

Reads results artifacts (or runs the deterministic scenario when cheap)
and writes docs/figures/*.png. Every number in a figure comes from a run
artifact; nothing is hand-entered.

Usage: python scripts/make_figures.py [--skip-ml] [--skip-gen]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIG_DIR = ROOT / "docs" / "figures"


def fig_closedloop() -> None:
    from kimi_isac.closedloop import run_closedloop

    result = run_closedloop(
        n_frames=24, dt_s=0.5, n_elem=16384, blockage_db=60.0, smoke=False, seed=42
    )
    per_frame = result["per_frame_snr_db"]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    styles = {
        "direct_only": ("gray", "--"),
        "random": ("tab:orange", ":"),
        "tracked": ("tab:blue", "-"),
        "segmented_16": ("tab:green", "-."),
        "segmented_8": ("tab:red", "-."),
    }
    for name, (color, ls) in styles.items():
        ax.plot(per_frame[name], ls, color=color, label=name, lw=1.6)
    ax.set_xlabel("frame (0.5 s spacing, ISS overpass)")
    ax.set_ylabel("received SNR (dB)")
    ax.set_title("RIS phase control over an ISS overpass (60 dB blockage, 16384 elements)")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_closedloop.png")
    plt.close(fig)


def fig_ml_vs_classical() -> None:
    report = json.loads((ROOT / "results" / "ml_report.json").read_text())
    snrs = [6.0, 1.0, 0.25]

    def cell(key: str) -> tuple[float, float]:
        m = report["metrics"][key]
        return m["mean"], (m["ci95"][1] - m["ci95"][0]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), dpi=150)
    panels = [
        (
            "detection acc",
            [f"test_snr{s:g}_detect_acc" for s in snrs],
            [f"test_snr{s:g}_classical_detect_acc" for s in snrs],
            (0.8, 1.05),
        ),
        (
            "range RMSE (m)",
            [f"test_snr{s:g}_ml_range_rmse_m" for s in snrs],
            [f"test_snr{s:g}_classical_range_rmse_m" for s in snrs],
            None,
        ),
        (
            "velocity RMSE (m/s)",
            [f"test_snr{s:g}_ml_vel_rmse_mps" for s in snrs],
            [f"test_snr{s:g}_classical_vel_rmse_mps" for s in snrs],
            None,
        ),
    ]
    x = np.arange(3)
    for ax, (title, ml_keys, cl_keys, ylim) in zip(axes, panels):
        ml = [cell(k) for k in ml_keys]
        cl = [cell(k) for k in cl_keys]
        ax.bar(
            x - 0.2,
            [m[0] for m in ml],
            0.38,
            yerr=[m[1] for m in ml],
            label="ML head",
            color="tab:blue",
            capsize=3,
        )
        ax.bar(
            x + 0.2,
            [m[0] for m in cl],
            0.38,
            yerr=[m[1] for m in cl],
            label="classical",
            color="tab:orange",
            capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"SNR {s:g}" for s in snrs])
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.suptitle("ML head vs classical baseline (10 seeds, bootstrap 95% CI)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ml_vs_classical.png", bbox_inches="tight")
    plt.close(fig)


def fig_gen_cd() -> None:
    report = json.loads((ROOT / "results" / "gen_report.json").read_text())
    cells = report["cond_cd"]
    labels = list(cells.keys())
    means = [cells[k]["mean"] for k in labels]
    cis = [(cells[k]["ci95"][1] - cells[k]["ci95"][0]) / 2 for k in labels]
    prior = report["uncond_cd_heldout_inst"]["mean"]
    prior_ci = (
        report["uncond_cd_heldout_inst"]["ci95"][1] - report["uncond_cd_heldout_inst"]["ci95"][0]
    ) / 2
    oracle = report["cond_cd_oracle"]["mean"]
    oracle_ci = (report["cond_cd_oracle"]["ci95"][1] - report["cond_cd_oracle"]["ci95"][0]) / 2

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=150)
    x = np.arange(len(labels))
    ax.bar(x, means, 0.6, yerr=cis, capsize=3, color="tab:purple", alpha=0.85)
    ax.axhline(
        prior,
        color="tab:blue",
        ls="--",
        lw=1.4,
        label=f"class prior (unconditional) {prior:.3f}±{prior_ci:.3f}",
    )
    ax.axhline(
        oracle,
        color="tab:red",
        ls=":",
        lw=1.4,
        label=f"oracle per-scatterer RIS {oracle:.3f}±{oracle_ci:.3f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Chamfer distance")
    ax.set_title("Conditional reconstruction does not beat the class prior (10 seeds, 95% CI)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_gen_cd.png")
    plt.close(fig)


def fig_templates() -> None:
    from kimi_isac.gen import templates as tpl

    names = tpl.all_class_names()
    fig, axes = plt.subplots(2, len(names), figsize=(14, 3.2), dpi=150)
    for col, name in enumerate(names):
        for row in range(2):
            cloud = tpl.make_template(name, np.random.default_rng(100 + row))
            ax = axes[row, col]
            ax.scatter(cloud[:, 0], cloud[:, 1], s=0.4, c=cloud[:, 2], cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(name, fontsize=9)
    fig.suptitle("Procedural templates: two random instances per class (top-down view)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_templates.png")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ml", action="store_true")
    parser.add_argument("--skip-gen", action="store_true")
    args = parser.parse_args()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig_closedloop()
    fig_templates()
    if not args.skip_ml:
        fig_ml_vs_classical()
    if not args.skip_gen:
        fig_gen_cd()
    print("figures written to", FIG_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
