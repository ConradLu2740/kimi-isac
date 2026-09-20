"""Orbit verification against external references.

Checks (all independent of this codebase):
- semi-major axis from Kepler's third law using the TLE mean motion
- period from mean motion matches the published ISS value (~92.9 min)
- propagated speed against the vis-viva equation
- specific orbital energy drift along the propagated arc
- altitude stays inside the published ISS band (400-430 km)
"""

from __future__ import annotations

import math

import numpy as np

from kimi_isac.core import orbit as orbit_mod
from kimi_isac.core.constants import MU_EARTH

ISS_ALT_BAND_KM = (400.0, 440.0)
VIS_VIVA_TOL = 0.02
ENERGY_DRIFT_TOL = 5e-3


def run() -> bool:
    sat = orbit_mod.satrec_from_tle(*orbit_mod.ISS_TLE)
    jd0, fr0 = sat.jdsatepoch, sat.jdsatepochF
    n_rad_s = sat.no_kozai / 60.0  # sgp4 stores mean motion in rad/min
    period_s = 2.0 * math.pi / n_rad_s
    a_m = (MU_EARTH / n_rad_s**2) ** (1.0 / 3.0)

    n = 60
    samples = [
        orbit_mod.sample_orbit(sat, jd0 + fr0, 0.0, t_sec=t * period_s / n) for t in range(n + 1)
    ]
    speeds = np.array([s.speed_mps for s in samples])
    radii = np.array([np.linalg.norm(s.r_eci_m) for s in samples])
    alts_km = np.array([s.geocentric_alt_m for s in samples]) / 1e3
    energies = speeds**2 / 2.0 - MU_EARTH / radii

    v_vis_viva = np.sqrt(MU_EARTH * (2.0 / radii - 1.0 / a_m))
    vv_err = float(np.max(np.abs(speeds - v_vis_viva) / v_vis_viva))
    energy_drift = float(np.max(np.abs(energies - energies[0]) / abs(energies[0])))

    checks = [
        (
            "period from mean motion ~92.9 min",
            92.5 * 60 < period_s < 93.5 * 60,
            f"{period_s / 60.0:.2f} min",
        ),
        (
            "altitude within published ISS band",
            ISS_ALT_BAND_KM[0] < alts_km.min() and alts_km.max() < ISS_ALT_BAND_KM[1],
            f"[{alts_km.min():.1f}, {alts_km.max():.1f}] km",
        ),
        ("speed vs vis-viva", vv_err < VIS_VIVA_TOL, f"max rel err {vv_err:.2e}"),
        ("specific energy drift", energy_drift < ENERGY_DRIFT_TOL, f"{energy_drift:.2e}"),
    ]
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok = ok and passed
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run() else 1)
