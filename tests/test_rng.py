"""Verify BS-ready reproducibility of the seeding helper."""

import numpy as np

from kimi_isac.core.rng import seed_all


def test_seed_all_reproducible() -> None:
    seed_all(123)
    a = np.random.rand(3)
    seed_all(123)
    b = np.random.rand(3)
    assert np.array_equal(a, b)
    seed_all(124)
    c = np.random.rand(3)
    assert not np.allclose(a, c)
