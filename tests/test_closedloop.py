"""Tests: closed-loop smoke scenario and frame conversions."""

import numpy as np

from kimi_isac.closedloop import run_closedloop
from kimi_isac.core import frames


def test_geodetic_ecef_roundtrip() -> None:
    r = frames.geodetic_to_ecef(39.9042, 116.4074, 100.0)
    lat, lon, alt = frames.ecef_to_geodetic(r)
    assert abs(lat - 39.9042) < 1e-6
    assert abs(lon - 116.4074) < 1e-6
    assert abs(alt - 100.0) < 1e-6


def test_elevation_zenith_and_horizon() -> None:
    observer = frames.geodetic_to_ecef(0.0, 0.0, 0.0)
    lat, lon, _ = frames.ecef_to_geodetic(observer)
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    up = np.array([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
    zenith = observer + up * 1000.0
    assert abs(frames.elevation_deg(observer, zenith) - 90.0) < 1e-3
    east = np.cross([0.0, 0.0, 1.0], up)
    assert abs(frames.elevation_deg(observer, observer + east * 1000.0)) < 1e-3


def test_gmst_rotation_is_orthogonal() -> None:
    from kimi_isac.closedloop import _rotation_matrix

    r = _rotation_matrix(2460000.5)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12)


def test_closedloop_smoke_tracking_beats_random() -> None:
    result = run_closedloop(
        n_frames=16, dt_s=0.5, n_elem=1024, blockage_db=60.0, smoke=True, seed=42
    )
    s = result["summary"]
    assert s["tracked"]["gain_vs_random_db"] > 25.0
    assert s["tracked"]["mean_snr_db"] > s["direct_only"]["mean_snr_db"] + 30.0
    # segmented reconfiguration must degrade as segments shrink
    fracs = [s[f"segmented_{k}"]["frac_of_tracked"] for k in (1, 2, 4, 8, 16)]
    assert fracs == sorted(fracs), "coarser reconfiguration should lose gain"
    assert result["mean_doppler_hz"] != 0.0
