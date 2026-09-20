"""Tests for the echo model (cloud -> per-frame features)."""

import numpy as np

from kimi_isac.core.channel import C_LIGHT, doppler_shift_hz, range_rate_mps
from kimi_isac.gen import echo, scene, templates

SMOKE_CFG = echo.EchoConfig(n_elem=32, tau=4)


def _cloud():
    return templates.make_template("building", np.random.default_rng(3))


def test_feature_shape_and_finiteness():
    out = echo.echo_and_features(_cloud(), "building", "aligned", seed=0, snr=6.0, cfg=SMOKE_CFG)
    f = out["features"]
    assert f.shape == (SMOKE_CFG.tau, 10)
    assert np.all(np.isfinite(f))


def test_delay_feature_matches_bistatic_geometry():
    cfg = SMOKE_CFG
    frames = echo.frame_geometry(cfg)
    cloud = _cloud()
    scat = scene.scatterers_ecef(cloud, frames[0]["ue"], frames[0]["up"], frames[0]["east"])
    centroid = scat.mean(axis=0)
    delay = (
        (
            np.linalg.norm(frames[0]["sat_pos"] - centroid)
            + np.linalg.norm(centroid - frames[0]["ue"])
        )
        / C_LIGHT
        * 1e3
    )
    f = echo.echo_and_features(cloud, "building", "none", seed=0, snr=1e6, cfg=cfg)["features"]
    assert abs(f[0, 4] - delay) < 1e-6


def test_doppler_matches_core_convention():
    cfg = SMOKE_CFG
    frames = echo.frame_geometry(cfg)
    cloud = _cloud()
    scat = scene.scatterers_ecef(cloud, frames[0]["ue"], frames[0]["up"], frames[0]["east"])
    centroid = scat.mean(axis=0)
    fr = frames[0]
    rr_scalar = range_rate_mps(fr["sat_pos"], fr["sat_vel"], centroid, np.zeros(3))
    f_d_core = doppler_shift_hz(rr_scalar, echo.FC_HZ)
    f = echo.echo_and_features(cloud, "building", "none", seed=0, snr=1e6, cfg=cfg)["features"]
    assert abs(f[0, 3] - f_d_core / 1e3) < 1e-9


def test_ris_aligned_beats_random_on_average():
    # Measure the RIS-path component itself: with N=32 the coherent sum is
    # ~sqrt(N)=5.7x the random-phase sum in amplitude. (In the TOTAL echo the
    # RIS path sits tens of dB below the direct path at this panel size, so
    # only the component test is meaningful at small N.)
    cloud = _cloud()
    gains = []
    for t in range(8):
        a = np.abs(
            echo.echo_and_features(cloud, "building", "aligned", seed=t, snr=6.0, cfg=SMOKE_CFG)[
                "Y_ris"
            ]
        ).mean()
        r = np.abs(
            echo.echo_and_features(cloud, "building", "random", seed=t, snr=6.0, cfg=SMOKE_CFG)[
                "Y_ris"
            ]
        ).mean()
        gains.append(a / r)
    assert np.median(gains) > 1.5


def test_oracle_upper_bound_and_none_mode():
    cloud = _cloud()
    out = echo.echo_and_features(cloud, "building", "oracle", seed=0, snr=6.0, cfg=SMOKE_CFG)
    assert out["Y_oracle"] is not None
    assert np.all(np.isfinite(out["Y_oracle"]))
    out_none = echo.echo_and_features(cloud, "building", "none", seed=0, snr=6.0, cfg=SMOKE_CFG)
    assert out_none["Y_oracle"] is None


def test_single_scatterer_echo_matches_hand_computation():
    # Pins the complex arithmetic: one scatterer at the ROI anchor, RIS off,
    # huge SNR (noise negligible) -> Y must equal the analytic complex sum.
    cfg = SMOKE_CFG
    frames = echo.frame_geometry(cfg)
    fr = frames[0]
    ue, up = fr["ue"], fr["up"]
    cloud = np.zeros((1, 3), dtype=float)  # single scatterer at the ROI anchor
    p = scene.roi_anchor(ue, up)
    d1 = float(np.linalg.norm(fr["sat_pos"] - p))
    d2 = float(np.linalg.norm(p - ue))
    rho = echo.REFLECTIVITY["building"]
    lam = C_LIGHT / echo.FC_HZ
    rate = float(np.dot(fr["sat_vel"], (fr["sat_pos"] - p) / d1))
    f_d = -rate / lam
    expected = (
        rho
        * (lam / (4.0 * np.pi * d1))
        * (lam / (4.0 * np.pi * d2))
        * np.exp(-1j * 2.0 * np.pi * (d1 + d2) / lam)
        * np.exp(1j * 2.0 * np.pi * f_d * fr["t"])
    )
    out = echo.echo_and_features(cloud, "building", "none", seed=0, snr=1e12, cfg=cfg)
    assert abs(out["Y"][0] - expected) < 1e-6 * abs(expected)
    assert abs(out["Y"][0].imag) > 1e-3 * abs(out["Y"][0]), "echo lost its imaginary part"
