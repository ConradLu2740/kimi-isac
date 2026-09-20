"""Phase 2 sensing tests: CFAR calibration, range-Doppler, MUSIC, CRB."""

import numpy as np

from kimi_isac.core.waveform import OFDMConfig
from kimi_isac.sensing import cfar, crb, doa, rangedoppler

CFG = OFDMConfig(n_subcarriers=1024, bandwidth_hz=30.72e6)


def test_ca_cfar_monte_carlo_pfa_matches_design() -> None:
    # Ground-truth check: simulated false-alarm rate must match the analytic design.
    pfa = 1e-3
    measured = cfar.estimate_pfa_mc(n_trials=100_000, n_train=16, n_guard=2, pfa=pfa, seed=1)
    assert 0.6 * pfa < measured < 1.6 * pfa, f"measured Pfa {measured:.2e} vs design {pfa}"


def test_ca_cfar_detects_strong_target_no_false_alarms() -> None:
    rng = np.random.default_rng(7)
    profile = rng.exponential(1.0, size=1500)
    target_idx = 700
    profile[target_idx] = 200.0
    result = cfar.ca_cfar_1d(profile, n_train=16, n_guard=2, pfa=1e-6)
    detected = np.flatnonzero(result["detections"])
    assert detected.size == 1 and detected[0] == target_idx


def test_ca_cfar_noise_only_no_detections() -> None:
    # At pfa = 1e-6 over 1000 cells, a handful of seeds should see no detections.
    for seed in range(10):
        rng = np.random.default_rng(seed)
        profile = rng.exponential(1.0, size=1000)
        result = cfar.ca_cfar_1d(profile, n_train=16, n_guard=2, pfa=1e-6)
        assert not result["detections"].any(), f"false alarm at seed {seed}"


def test_range_doppler_map_localizes_target() -> None:
    n_slow = 32
    target_range_m, target_doppler_hz = 200.0, 2_000.0
    symbols = rangedoppler.inject_target(CFG, n_slow, target_range_m, target_doppler_hz)
    rd = rangedoppler.range_doppler_map(symbols, CFG)
    peak = np.unravel_index(int(np.argmax(rd["map"])), rd["map"].shape)
    range_err = abs(rd["range_bins_m"][peak[0]] - target_range_m)
    doppler_err = abs(rd["doppler_bins_hz"][peak[1]] - target_doppler_hz)
    assert range_err < CFG.range_resolution_m
    assert doppler_err < 1.0 / (n_slow * CFG.useful_symbol_s)


def test_range_doppler_map_noisy_still_localizes() -> None:
    rng = np.random.default_rng(3)
    n_slow = 32
    symbols = rangedoppler.inject_target(CFG, n_slow, 120.0, -1_500.0, amplitude=3.0)
    noise = (rng.normal(size=symbols.shape) + 1j * rng.normal(size=symbols.shape)) * 0.05
    rd = rangedoppler.range_doppler_map(symbols + noise, CFG)
    peak = np.unravel_index(int(np.argmax(rd["map"])), rd["map"].shape)
    assert abs(rd["range_bins_m"][peak[0]] - 120.0) < CFG.range_resolution_m


def test_music_resolves_two_sources() -> None:
    rng = np.random.default_rng(11)
    x = doa.snapshot_matrix(
        angles_deg=[-10.0, 10.0], n_snapshots=200, n_elements=16, noise_std=0.01, rng=rng
    )
    peaks = doa.estimate_doa(x, n_sources=2)
    assert peaks.shape == (2,)
    assert abs(peaks[0] + 10.0) < 1.0 and abs(peaks[1] - 10.0) < 1.0


def test_delay_crb_vs_monte_carlo() -> None:
    freqs = np.arange(CFG.n_subcarriers) * CFG.subcarrier_spacing_hz
    noise_variance = 1.1  # chosen so CRB ~ 1 m
    crb_m = crb.range_crb_m(freqs, noise_variance)
    rmse_m = crb.estimate_range_mc(
        freqs, range_m=150.0, noise_variance=noise_variance, n_trials=400, seed=5
    )
    assert 0.7 * crb_m < rmse_m < 1.8 * crb_m, f"MC RMSE {rmse_m:.3f} m vs CRB {crb_m:.3f} m"
