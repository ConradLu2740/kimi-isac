"""Sensing-communication closed-loop demo: RIS phase control over a LEO link.

Honest physics notes
--------------------
* The two-hop RIS path suffers double path loss, so it only dominates the
  direct path when the direct path is blocked. The scenario therefore
  models a UE in blockage shadow (configurable ``blockage_db``), which is
  the canonical RIS use case; with ``blockage_db=0`` the direct path wins
  and the RIS contribution is small by construction, not by tuning.
* Channels are recomputed per frame from real geometry (SGP4 overpass or
  straight-line smoke). "Tracked" phases use per-frame CSI (an oracle);
  segmented-K variants freeze phases across frames, which quantifies the
  reconfiguration-rate vs channel-drift trade-off at 30 GHz.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from kimi_isac.core import frames, logging_setup
from kimi_isac.core.channel import (
    doppler_shift_hz,
    free_space_channel,
    range_rate_mps,
)
from kimi_isac.core.constants import C_LIGHT, EARTH_ROT_RATE
from kimi_isac.core.orbit import ISS_TLE, satrec_from_tle
from kimi_isac.core.rng import seed_all
from kimi_isac.core.ris import array_gain_db
from kimi_isac.opt.phase_opt import align_phases, greedy_bound, segment_track

FC_HZ = 30.0e9
PT_W = 20.0
GT_DBI = 30.0
GR_DBI = 30.0
BANDWIDTH_HZ = 30.0e6
NOISE_FIGURE_DB = 5.0
N_ELEMENTS = 16384
BLOCKAGE_DB = 60.0
RIS_ELEM_OFFSET_M = 30.0  # panel center distance from the UE
SMOKE_ALT_M = 1.0e6
SMOKE_SPEED_MPS = 7.5e3


def noise_power_w(bandwidth_hz: float = BANDWIDTH_HZ, nf_db: float = NOISE_FIGURE_DB) -> float:
    from kimi_isac.core.link_budget import thermal_noise_dbw

    return 10 ** (thermal_noise_dbw(bandwidth_hz, nf_db) / 10.0)


def smoke_geometry(n_frames: int, dt_s: float, n_elem: int) -> list[dict]:
    """Straight-line fly-by: satellite passing at 1000 km altitude."""
    wavelength = C_LIGHT / FC_HZ
    ue = np.zeros(3)
    ris_center = np.array([RIS_ELEM_OFFSET_M, 0.0, 0.0])
    elem_spacing = wavelength / 2.0
    ris_elements = np.array(
        [
            ris_center + np.array([0.0, (i - (n_elem - 1) / 2.0) * elem_spacing, 0.0])
            for i in range(n_elem)
        ]
    )
    frames_geo = []
    for k in range(n_frames):
        t = k * dt_s
        sat_pos = np.array([0.0, -2000.0e3 + SMOKE_SPEED_MPS * t, SMOKE_ALT_M])
        sat_vel = np.array([0.0, SMOKE_SPEED_MPS, 0.0])
        frames_geo.append(
            {"t": t, "sat_pos": sat_pos, "sat_vel": sat_vel, "ue": ue, "ris": ris_elements}
        )
    return frames_geo


def iss_overpass_geometry(
    n_frames: int,
    dt_s: float,
    n_elem: int,
    min_elev_deg: float = 30.0,
    search_days: float = 7.0,
    ue_geodetic: tuple[float, float] = (39.9042, 116.4074),
) -> list[dict]:
    """First ISS overpass above the UE (default Beijing) after the TLE epoch.

    SGP4 returns TEME states; all geometry here is rotated into ECEF
    (including the earth-rate term in the velocity) so elevation, Doppler
    and channels are computed in one consistent frame.
    """
    sat = satrec_from_tle(*ISS_TLE)
    ue_ecef = frames.geodetic_to_ecef(ue_geodetic[0], ue_geodetic[1], 0.0)
    jd0 = sat.jdsatepoch + sat.jdsatepochF
    step_s = 30.0
    n_steps = int(search_days * 86400 / step_s)
    window: list[float] = []
    for i in range(n_steps):
        jd = jd0 + i * step_s / 86400.0
        e, r_km, _ = sat.sgp4(jd, 0.0)
        if e != 0:
            continue
        r_ecef = frames.eci_to_ecef(np.asarray(r_km, dtype=float) * 1e3, jd)
        if frames.elevation_deg(ue_ecef, r_ecef) >= min_elev_deg:
            window.append(i * step_s)
        elif window:
            break
    if len(window) * step_s < n_frames * dt_s:
        raise RuntimeError(
            f"no {n_frames}-frame overpass above {min_elev_deg} deg in {search_days} days"
        )
    t0 = window[0]

    wavelength = C_LIGHT / FC_HZ
    up = ue_ecef / np.linalg.norm(ue_ecef)
    east = np.cross(np.array([0.0, 0.0, 1.0]), up)
    east /= np.linalg.norm(east)
    ris_center = ue_ecef + up * RIS_ELEM_OFFSET_M
    elem_spacing = wavelength / 2.0
    ris_elements = np.array(
        [ris_center + east * (i - (n_elem - 1) / 2.0) * elem_spacing for i in range(n_elem)]
    )
    out = []
    for k in range(n_frames):
        jd = jd0 + (t0 + k * dt_s) / 86400.0
        e, r_km, v_km_s = sat.sgp4(jd, 0.0)
        if e != 0:
            raise RuntimeError(f"sgp4 error {e}")
        r_eci = np.asarray(r_km, dtype=float) * 1e3
        v_eci = np.asarray(v_km_s, dtype=float) * 1e3
        r_ecef = frames.eci_to_ecef(r_eci, jd)
        v_ecef = _rotation_matrix(jd) @ v_eci - np.cross([0.0, 0.0, EARTH_ROT_RATE], r_ecef)
        out.append(
            {
                "t": t0 + k * dt_s,
                "sat_pos": r_ecef,
                "sat_vel": v_ecef,
                "ue": ue_ecef,
                "ris": ris_elements,
            }
        )
    return out


def _rotation_matrix(jd_utc: float) -> np.ndarray:
    """TEME->ECEF rotation about +z by GMST."""
    theta = frames.gmst_rad(jd_utc)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def frame_paths(geo: dict, blockage_db: float) -> tuple[np.ndarray, complex, float]:
    """Per-element two-hop channel, direct channel (with blockage), Doppler."""
    sat_pos, sat_vel = geo["sat_pos"], geo["sat_vel"]
    ue, ris = geo["ue"], geo["ris"]
    h_d = free_space_channel(FC_HZ, float(np.linalg.norm(sat_pos - ue)), GT_DBI, GR_DBI)
    h_d *= 10 ** (-blockage_db / 20.0)
    elem_gain = 10 ** (array_gain_db(1) / 20.0)
    a = np.empty(ris.shape[0], dtype=complex)
    for i, r in enumerate(ris):
        g = free_space_channel(FC_HZ, float(np.linalg.norm(sat_pos - r)), GT_DBI, 0.0)
        h = free_space_channel(FC_HZ, float(np.linalg.norm(r - ue)), 0.0, GR_DBI)
        a[i] = elem_gain * g * h
    rr = range_rate_mps(sat_pos, sat_vel, ue, np.zeros(3))
    f_d = doppler_shift_hz(rr, FC_HZ)
    return a, h_d, f_d


def snr_db(a: np.ndarray, h_d: complex, phi: np.ndarray) -> float:
    field = h_d + np.sum(a * np.exp(1j * np.asarray(phi, dtype=float)))
    return 10.0 * np.log10(PT_W * abs(field) ** 2 / noise_power_w())


def run_closedloop(
    n_frames: int = 32,
    dt_s: float = 0.5,
    n_elem: int = N_ELEMENTS,
    blockage_db: float = 40.0,
    smoke: bool = False,
    seed: int = 42,
    segments: tuple[int, ...] = (1, 2, 4, 8, 16),
    log: logging.Logger | None = None,
) -> dict:
    log = log or logging.getLogger("kimi_isac.closedloop")
    seed_all(seed)
    rng = np.random.default_rng(seed)
    if smoke:
        geometry = smoke_geometry(n_frames, dt_s, n_elem)
    else:
        geometry = iss_overpass_geometry(n_frames, dt_s, n_elem)

    paths = [frame_paths(g, blockage_db) for g in geometry]
    tracked = np.array(
        [align_phases(a, h_d, seed=int(rng.integers(1 << 31))) for a, h_d, _ in paths]
    )
    random_phi = rng.uniform(0.0, 2.0 * np.pi, size=(n_frames, n_elem))

    variants: dict[str, np.ndarray] = {
        "direct_only": np.zeros((n_frames, n_elem)),
        "random": random_phi,
        "tracked": tracked,
    }
    for k in segments:
        variants[f"segmented_{k}"] = segment_track(tracked, k)

    snr = {
        name: np.array([snr_db(a, h_d, phi[k]) for k, (a, h_d, _) in enumerate(paths)])
        for name, phi in variants.items()
    }
    greedy = np.array([greedy_bound(a, h_d) for a, h_d, _ in paths])
    greedy_snr = 10.0 * np.log10(PT_W * greedy**2 / noise_power_w())

    summary = {}
    base = snr["tracked"].mean()
    for name, s in snr.items():
        summary[name] = {
            "mean_snr_db": float(s.mean()),
            "gain_vs_random_db": float(s.mean() - snr["random"].mean()),
            "frac_of_tracked": float(10 ** ((s.mean() - base) / 10.0)),
            "frac_of_greedy": float(10 ** ((s.mean() - greedy_snr.mean()) / 10.0)),
        }
    summary["greedy_bound"] = {"mean_snr_db": float(greedy_snr.mean())}

    result = {
        "scenario": "smoke" if smoke else "iss_overpass",
        "n_frames": n_frames,
        "dt_s": dt_s,
        "n_elem": n_elem,
        "blockage_db": blockage_db,
        "fc_hz": FC_HZ,
        "mean_doppler_hz": float(np.mean([fd for _, _, fd in paths])),
        "panel_gain_db": array_gain_db(n_elem),
        "summary": summary,
        "per_frame_snr_db": {k: v.tolist() for k, v in snr.items()},
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--n-elem", type=int, default=N_ELEMENTS)
    parser.add_argument("--blockage-db", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args(argv)

    logging_setup.setup_logging()
    log = logging.getLogger("kimi_isac.closedloop")
    result = run_closedloop(
        n_frames=args.frames,
        dt_s=args.dt,
        n_elem=args.n_elem,
        blockage_db=args.blockage_db,
        smoke=args.smoke,
        seed=args.seed,
        log=log,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "closedloop.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    log.info(
        "scenario=%s blockage=%.0f dB panel_gain=%.1f dB",
        result["scenario"],
        result["blockage_db"],
        result["panel_gain_db"],
    )
    for name, s in result["summary"].items():
        if "gain_vs_random_db" in s:
            log.info(
                "%-14s mean SNR %7.2f dB | vs random %+6.2f dB | %.1f%% of tracked | %.1f%% of greedy",
                name,
                s["mean_snr_db"],
                s["gain_vs_random_db"],
                100 * s["frac_of_tracked"],
                100 * s["frac_of_greedy"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
