"""Doppler sign-convention verification.

Physical requirement: a transmitter approaching the receiver produces a
POSITIVE received frequency shift (f_D = -range_rate / lambda). The
analytic construction below is independent of any simulator.
"""

from __future__ import annotations

import numpy as np

from kimi_isac.core.channel import C_LIGHT, doppler_shift_hz, range_rate_mps

FREQ_HZ = 2.2e9  # S-band
H_M = 7.02e6
D0_M = 2.0e6
V_MPS = 7_000.0


def _sat_state(t: float) -> tuple[np.ndarray, np.ndarray]:
    """Straight-line approaching fly-by starting at y = -D0, velocity +y."""
    return np.array([0.0, -D0_M + V_MPS * t, H_M]), np.array([0.0, V_MPS, 0.0])


def run() -> bool:
    target = np.zeros(3)
    v_target = np.zeros(3)

    # numeric range rate from finite differences of the synthetic range
    ts = np.linspace(0.0, 4.0, 2001)
    ranges = np.array([np.linalg.norm(_sat_state(t)[0] - target) for t in ts])
    t_ref = ts[1000]
    rng_rate_num = float(np.gradient(ranges, ts)[1000])

    # analytic geometry at the same instant
    r_tx, v_tx = _sat_state(t_ref)
    analytic_rng = float(np.dot(-v_tx, (target - r_tx) / np.linalg.norm(target - r_tx)))

    rng_rate_code = range_rate_mps(r_tx, v_tx, target, v_target)
    f_d = doppler_shift_hz(rng_rate_code, FREQ_HZ)

    approaching_ok = analytic_rng < 0
    sign_ok = f_d > 0
    numeric_ok = abs(rng_rate_num - analytic_rng) < 1.0
    formula_ok = abs(f_d - (-analytic_rng / (C_LIGHT / FREQ_HZ))) < 1e-6

    checks = [
        (
            "approaching satellite has negative range rate",
            approaching_ok,
            f"{analytic_rng:.3f} m/s",
        ),
        (
            "approaching satellite has POSITIVE Doppler",
            sign_ok,
            f"{f_d:+.3f} Hz at {FREQ_HZ / 1e9:.1f} GHz",
        ),
        (
            "finite-difference range rate matches analytic",
            numeric_ok,
            f"num {rng_rate_num:.3f} vs {analytic_rng:.3f} m/s",
        ),
        ("f_D equals -range_rate/lambda", formula_ok, f"{f_d:.6f} Hz"),
    ]
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok = ok and passed
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run() else 1)
