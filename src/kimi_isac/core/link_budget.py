"""Link budget: Friis transmission equation, noise floor, SNR.

All functions take/return SI units or dB-quantities marked ``_db``.
Noise power is kTB (IEEE reference temperature) plus an optional noise
figure, so SNR values are physical, not calibrated by magic constants.
"""

from __future__ import annotations

import math

from kimi_isac.core.constants import C_LIGHT, K_BOLTZMANN, T_NOISE_REF

KTB_1HZ_DBW = 10.0 * math.log10(K_BOLTZMANN * T_NOISE_REF)  # -203.98 dBW/Hz


def wavelength_m(freq_hz: float) -> float:
    return C_LIGHT / freq_hz


def free_space_path_loss_db(freq_hz: float, dist_m: float) -> float:
    """FSPL = 20 log10(4 pi d / lambda). Textbook check: 92.45 dB at 1 GHz, 1 km."""
    lam = wavelength_m(freq_hz)
    return 20.0 * math.log10(4.0 * math.pi * dist_m / lam)


def friis_received_dbw(
    pt_dbw: float, gt_dbi: float, gr_dbi: float, freq_hz: float, dist_m: float
) -> float:
    """Received power in dBW (Friis, free space, polarization matched)."""
    return pt_dbw + gt_dbi + gr_dbi - free_space_path_loss_db(freq_hz, dist_m)


def eirp_dbw(pt_dbw: float, gt_dbi: float) -> float:
    return pt_dbw + gt_dbi


def thermal_noise_dbw(bandwidth_hz: float, noise_figure_db: float = 0.0) -> float:
    """N = kTB + NF (IEEE 290 K reference)."""
    n_dbw_hz = 10.0 * math.log10(K_BOLTZMANN * T_NOISE_REF)
    return n_dbw_hz + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


def snr_db(
    pt_dbw: float,
    gt_dbi: float,
    gr_dbi: float,
    freq_hz: float,
    dist_m: float,
    bandwidth_hz: float,
    noise_figure_db: float = 0.0,
) -> float:
    """Received SNR in dB: Pr / N."""
    pr = friis_received_dbw(pt_dbw, gt_dbi, gr_dbi, freq_hz, dist_m)
    return pr - thermal_noise_dbw(bandwidth_hz, noise_figure_db)
