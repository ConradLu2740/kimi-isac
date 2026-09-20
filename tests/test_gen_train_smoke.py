"""In-process smoke test for the two-stage generative training."""

import logging

from kimi_isac.gen import dataset as ds
from kimi_isac.gen import train as gen_train


def test_smoke_training_runs(tmp_path):
    cfg = ds.GenConfig(n_total=24, seed=0, snr_levels=(6.0,), tau=4, n_elem=16)
    metrics = gen_train.run_gen(
        seed=0,
        cfg=cfg,
        epochs_vae=1,
        epochs_dit=1,
        device="cpu",
        out_dir=tmp_path,
        log=logging.getLogger("test"),
        T=20,
    )
    assert set(metrics) >= {
        "vae_recon_cd",
        "uncond_cd_train",
        "uncond_cd_heldout",
        "uncond_cd_ood",
        "cond_cd",
        "cond_cd_oracle",
        "cond_cd_cells",
    }
    for k, v in metrics.items():
        if k != "cond_cd_cells":
            assert v == v, f"{k} is NaN"
