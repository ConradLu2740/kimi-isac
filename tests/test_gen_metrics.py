"""Tests for Chamfer distance metrics."""

import numpy as np

from kimi_isac.core import stats
from kimi_isac.gen import metrics


def test_identical_clouds_zero():
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, size=(32, 3))
    assert metrics.chamfer_distance(a, a) == 0.0


def test_known_two_point_cd():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    assert abs(metrics.chamfer_distance(a, b) - 1.0) < 1e-9


def test_symmetry():
    rng = np.random.default_rng(1)
    a = rng.uniform(-1, 1, size=(16, 3))
    b = rng.uniform(-1, 1, size=(24, 3))
    assert abs(metrics.chamfer_distance(a, b) - metrics.chamfer_distance(b, a)) < 1e-9


def test_summary_has_ci():
    s = metrics.summarize_cd([0.1, 0.12, 0.09, 0.11])
    assert isinstance(s, stats.Summary)
    assert s.ci95_lo <= s.mean <= s.ci95_hi
