"""Wireless channel model (single source of truth for path loss / phase).

Conventions (chosen deliberately; the original project under review used
the opposite Doppler sign):
- Complex channel uses the IEEE time convention ``exp(-j k r)`` with
  phase decreasing along the propagation path.
- Doppler shift follows ``f_D = -range_rate / lambda``, so a transmitter
  approaching the receiver produces a POSITIVE received frequency shift.
"""

from __future__ import annotations

import numpy as np

from kimi_isac.core.constants import C_LIGHT


def free_space_channel(
    freq_hz: float, dist_m: float, gt_dbi: float = 0.0, gr_dbi: float = 0.0
) -> complex:
    """Scalar complex channel H for one path: sqrt(Gt Gr) * lambda/(4 pi d) * exp(-j 2 pi d / lambda)."""
    lam = C_LIGHT / freq_hz
    amp = (10.0 ** (gt_dbi / 20.0)) * (10.0 ** (gr_dbi / 20.0)) * lam / (4.0 * np.pi * dist_m)
    return amp * np.exp(-1j * 2.0 * np.pi * dist_m / lam)


def doppler_shift_hz(range_rate_mps: float, freq_hz: float) -> float:
    """Received Doppler shift. range_rate > 0 means receding (range growing)."""
    return -range_rate_mps / (C_LIGHT / freq_hz)


def range_rate_mps(
    r_tx_m: np.ndarray, v_tx_mps: np.ndarray, r_rx_m: np.ndarray, v_rx_mps: np.ndarray
) -> float:
    """d|r_rx - r_tx|/dt: positive when the link is stretching."""
    u = np.asarray(r_rx_m, dtype=float) - np.asarray(r_tx_m, dtype=float)
    dist = float(np.linalg.norm(u))
    return float(np.dot(np.asarray(v_rx_mps) - np.asarray(v_tx_mps), u / dist))


def los_delay_s(dist_m: float) -> float:
    return dist_m / C_LIGHT
