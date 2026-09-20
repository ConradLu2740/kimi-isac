"""Tests: RIS aperture gain and phase-optimization behavior."""

import numpy as np

from kimi_isac.core.ris import RISPanel, array_gain_db
from kimi_isac.opt.phase_opt import align_phases, greedy_bound, received_field, segment_track


def test_array_gain_matches_aperture_formula() -> None:
    # G = 10 log10(4 pi A / lambda^2), A = N (0.5 lambda)^2 * eta
    for n in (1, 4, 16, 100, 1024):
        expected = 10.0 * np.log10(4.0 * np.pi * n * 0.25 * 0.8)
        assert abs(array_gain_db(n) - expected) < 1e-9


def test_panel_phases_are_unit_modulus() -> None:
    rng = np.random.default_rng(0)
    panel = RISPanel(n_elements=16)
    phases = panel.random_phases(rng)
    assert np.allclose(np.abs(panel.unit_modulus(phases)), 1.0)


def test_align_phases_reaches_coherent_sum_without_direct_path() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=16) + 1j * rng.normal(size=16)
    phi = align_phases(a, h_d=0.0 + 0.0j, seed=0)
    aligned = abs(received_field(a, 0.0 + 0.0j, phi))
    assert abs(aligned - np.sum(np.abs(a))) < 1e-6


def test_align_phases_beats_random_on_average() -> None:
    rng = np.random.default_rng(2)
    gains = []
    for t in range(20):
        a = rng.normal(size=16) + 1j * rng.normal(size=16)
        h_d = rng.normal() + 1j * rng.normal()
        phi = align_phases(a, h_d, seed=t)
        aligned = abs(received_field(a, h_d, phi)) ** 2
        random_power = np.mean(
            [abs(received_field(a, h_d, rng.uniform(0, 2 * np.pi, 16))) ** 2 for _ in range(50)]
        )
        gains.append(aligned / random_power)
    assert np.median(gains) > 1.5


def test_align_stays_below_greedy_bound() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(size=32) + 1j * rng.normal(size=32)
    h_d = rng.normal() + 1j * rng.normal()
    phi = align_phases(a, h_d, seed=0)
    assert abs(received_field(a, h_d, phi)) <= greedy_bound(a, h_d) + 1e-9


def test_segment_track_freezes_phases() -> None:
    tracked = np.arange(40, dtype=float).reshape(10, 4)
    seg = segment_track(tracked, n_segments=2)
    assert np.allclose(seg[0:5], tracked[0])
    assert np.allclose(seg[5:10], tracked[5])
    seg_full = segment_track(tracked, n_segments=10)
    assert np.allclose(seg_full, np.repeat(tracked, 1, axis=0))
