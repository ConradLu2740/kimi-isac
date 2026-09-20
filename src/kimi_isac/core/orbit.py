"""Orbit propagation via SGP4, plus published ISS TLE for verification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sgp4.api import Satrec

from kimi_isac.core import frames

# Fixed historical-grade ISS TLE (fetched from Celestrak 2026-09-20,
# epoch 26262.82). Frozen on purpose: verification must not depend on the
# network or on the current element set.
ISS_TLE = (
    "1 25544U 98067A   26262.82001782  .00006637  00000+0  12771-3 0  9998",
    "2 25544  51.6308 191.7362 0004804 159.3703 200.7480 15.49182662586428",
)


@dataclass(frozen=True)
class OrbitSample:
    """One propagated state."""

    t_sec: float
    r_eci_m: np.ndarray
    v_eci_mps: np.ndarray
    r_ecef_m: np.ndarray
    speed_mps: float
    geocentric_alt_m: float
    geodetic_alt_m: float


def satrec_from_tle(line1: str, line2: str) -> Satrec:
    sat = Satrec.twoline2rv(line1, line2)
    if sat.error != 0:
        raise ValueError(f"TLE rejected by SGP4 (error code {sat.error})")
    return sat


def propagate(sat: Satrec, jd_utc: float) -> tuple[np.ndarray, np.ndarray]:
    """SGPropagate -> (r_eci_m, v_eci_mps)."""
    e, r_km, v_km_s = sat.sgp4(jd_utc, 0.0)
    if e != 0:
        raise RuntimeError(f"sgp4 propagation error code {e}")
    return np.asarray(r_km, dtype=float) * 1e3, np.asarray(v_km_s, dtype=float) * 1e3


def sample_orbit(
    sat: Satrec, jd_start: float, t_sec_start: float, t_sec: float, dt: float = 60.0
) -> OrbitSample:
    """Sample the orbit at time t (seconds from the start instant)."""
    jd = jd_start + (t_sec - t_sec_start) / 86400.0
    r_eci, v_eci = propagate(sat, jd)
    r_ecef = frames.eci_to_ecef(r_eci, jd)
    _, _, geo_alt = frames.ecef_to_geodetic(r_ecef)
    return OrbitSample(
        t_sec=t_sec,
        r_eci_m=r_eci,
        v_eci_mps=v_eci,
        r_ecef_m=r_ecef,
        speed_mps=float(np.linalg.norm(v_eci)),
        geocentric_alt_m=frames.geocentric_altitude_m(r_ecef),
        geodetic_alt_m=geo_alt,
    )
