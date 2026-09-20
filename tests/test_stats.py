"""Tests for the statistics and split discipline."""

import numpy as np

from kimi_isac.core import splits, stats


def test_bootstrap_ci_covers_normal_mean() -> None:
    # Coverage MC: 95% percentile bootstrap must cover the true mean ~95%
    # of the time (sampling tolerance ~ +/- 1.5% at 300 repetitions).
    rng = np.random.default_rng(0)
    hits = 0
    trials = 600
    for t in range(trials):
        sample = rng.normal(loc=5.0, scale=2.0, size=40)
        _, lo, hi = stats.bootstrap_ci(sample, n_boot=2_000, seed=t)
        hits += int(lo <= 5.0 <= hi)
    coverage = hits / trials
    assert 0.92 < coverage < 0.98, f"coverage {coverage:.3f}"


def test_summarize_runs_known_values() -> None:
    summary = stats.summarize_runs([1.0, 2.0, 3.0, 4.0], seed=0)
    assert summary.n == 4
    assert abs(summary.mean - 2.5) < 1e-12
    assert abs(summary.std - np.std([1.0, 2.0, 3.0, 4.0], ddof=1)) < 1e-12
    assert summary.ci95_lo < summary.mean < summary.ci95_hi


def test_paired_bootstrap_detects_difference() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 0.1, size=30)
    b = rng.normal(1.2, 0.1, size=30)
    p = stats.paired_bootstrap_pvalue(a, b, seed=0)
    assert p < 0.01
    p_same = stats.paired_bootstrap_pvalue(a, a + 0.0, seed=0)
    assert p_same >= 0.05


def test_split_is_disjoint_deterministic_complete() -> None:
    spec = splits.SplitSpec(n_total=500, seed=42, name="unit")
    s1 = splits.make_split(spec)
    s2 = splits.make_split(spec)
    s3 = splits.make_split(splits.SplitSpec(n_total=500, seed=43, name="unit"))
    splits.assert_disjoint(s1)
    assert s1 == s2, "same spec must give the same split"
    assert s1["train"] != s3["train"], "different seed must give a different split"
    assert 0.65 < len(s1["train"]) / 500 < 0.75


def test_split_roundtrip(tmp_path) -> None:
    spec = splits.SplitSpec(n_total=100, seed=7)
    split = splits.make_split(spec)
    path = splits.save_split(split, tmp_path / "splits" / "unit.json")
    loaded = splits.load_split(path)
    assert loaded == split
