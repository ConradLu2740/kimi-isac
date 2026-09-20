"""Tests for procedural point-cloud templates."""

import numpy as np

from kimi_isac.gen import templates as t


def test_shapes_and_normalization():
    rng = np.random.default_rng(0)
    for name in t.TRAIN_CLASSES + t.OOD_CLASSES:
        cloud = t.make_template(name, rng)
        assert cloud.shape == (t.N_POINTS, 3)
        assert cloud.min() >= -1.0 and cloud.max() <= 1.0


def test_deterministic_per_seed():
    a = t.make_template("building", np.random.default_rng(7))
    b = t.make_template("building", np.random.default_rng(7))
    assert np.array_equal(a, b)
    c = t.make_template("building", np.random.default_rng(8))
    assert not np.array_equal(a, c)


def test_class_index_unique():
    names = t.TRAIN_CLASSES + t.OOD_CLASSES
    idx = [t.class_index(n) for n in names]
    assert len(set(idx)) == len(names)


def test_distinct_classes_differ():
    rng = np.random.default_rng(1)
    a = t.make_template("building", rng)
    b = t.make_template("vehicle", rng)
    assert not np.allclose(a, b)


def test_varied_instances_within_class():
    a = t.make_template("building", np.random.default_rng(10))
    b = t.make_template("building", np.random.default_rng(11))
    assert not np.allclose(a, b)
