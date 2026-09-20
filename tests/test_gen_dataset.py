"""Tests for gen datasets over fixed splits."""

import numpy as np
import torch

from kimi_isac.core import splits
from kimi_isac.gen import dataset as ds

SMOKE = ds.GenConfig(n_total=24, seed=42, snr_levels=(6.0,), n_elem=16, tau=4)


def test_uncond_dataset_shapes(tmp_path):
    split = ds.get_or_make_split(SMOKE, tmp_path)
    d = ds.UncondDataset(split["train"], SMOKE.seed, SMOKE)
    item = d[0]
    assert item["cloud"].shape == (512, 3)
    assert 0 <= int(item["class"]) < 8


def test_split_reused_and_disjoint(tmp_path):
    s1 = ds.get_or_make_split(SMOKE, tmp_path)
    s2 = ds.get_or_make_split(SMOKE, tmp_path)
    assert s1 == s2
    splits.assert_disjoint(s1)


def test_cond_dataset_features(tmp_path):
    split = ds.get_or_make_split(SMOKE, tmp_path)
    d = ds.CondDataset(split["test"], SMOKE.seed, SMOKE)
    item = d[0]
    assert item["features"].shape == (SMOKE.tau, 10)
    assert np.all(np.isfinite(item["features"].numpy()))
    assert item["ris_mode"] in ("aligned", "random", "none")


def test_cond_dataset_cache_roundtrip(tmp_path):
    split = ds.get_or_make_split(SMOKE, tmp_path)
    cache = tmp_path / "cache"
    d1 = ds.CondDataset(split["test"], SMOKE.seed, SMOKE, cache_dir=cache)
    plain = ds.CondDataset(split["test"], SMOKE.seed, SMOKE)
    a = d1[0]
    b = plain[0]
    assert torch.allclose(a["features"], b["features"])
    assert torch.allclose(a["cloud"], b["cloud"])
    assert list(cache.rglob("*.npz")), "cache files not written"


def test_oracle_dataset_mode(tmp_path):
    d = ds.OracleCondDataset(SMOKE.seed, SMOKE)
    assert d[0]["ris_mode"] == "oracle"


def test_class_snr_not_aliased(tmp_path):
    # Regression: the class draw and the SNR draw must not share an RNG stream
    # (they once did, aliasing class<->SNR through the same random word).
    cfg = ds.GenConfig(n_total=120, seed=42, snr_levels=(6.0, 1.0, 0.25), n_elem=8, tau=2)
    d = ds.CondDataset(list(range(120)), 42, cfg)
    classes: dict[int, set[float]] = {}
    snrs: dict[float, set[int]] = {}
    for i in range(len(d)):
        it = d[i]
        c = int(it["class"])
        s = round(float(it["snr"]), 2)
        classes.setdefault(c, set()).add(s)
        snrs.setdefault(s, set()).add(c)
    assert all(len(v) >= 2 for v in classes.values()), f"class-SNR aliasing: {classes}"
    assert all(len(v) >= 4 for v in snrs.values()), f"SNR-class aliasing: {snrs}"
