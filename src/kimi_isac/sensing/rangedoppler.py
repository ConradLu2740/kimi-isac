"""Range-Doppler processing of OFDM slow-time symbol stacks."""

from __future__ import annotations

import numpy as np

from kimi_isac.core.waveform import OFDMConfig


def range_doppler_map(symbols: np.ndarray, cfg: OFDMConfig) -> dict[str, np.ndarray]:
    """Map slow-time frequency-domain symbols to a range-Doppler magnitude map.

    Parameters
    ----------
    symbols:
        Complex array of shape ``(n_slow, n_subcarriers)``; row ``k`` holds
        the subcarrier response at slow-time index ``k``.
    cfg:
        OFDM configuration (sets the range/Doppler grids).

    Returns
    -------
    dict with:
        - ``map``: magnitude, shape ``(n_subcarriers, n_slow)`` — [range, doppler]
        - ``range_bins_m``, ``doppler_bins_hz``: axis grids
    """
    symbols = np.asarray(symbols, dtype=complex)
    if symbols.ndim != 2 or symbols.shape[1] != cfg.n_subcarriers:
        raise ValueError(f"expected (n_slow, {cfg.n_subcarriers}), got {symbols.shape}")
    range_profiles = np.fft.ifft(symbols, axis=1)  # range per slow-time symbol
    rd = np.fft.fftshift(np.fft.fft(range_profiles, axis=0), axes=0)
    n_slow = symbols.shape[0]
    doppler_bins = np.fft.fftshift(np.fft.fftfreq(n_slow, d=cfg.useful_symbol_s))
    return {
        "map": np.abs(rd).T,
        "range_bins_m": cfg.range_bins_m(),
        "doppler_bins_hz": doppler_bins,
    }


def inject_target(
    cfg: OFDMConfig,
    n_slow: int,
    range_m: float,
    doppler_hz: float,
    amplitude: complex = 1.0 + 0.0j,
) -> np.ndarray:
    """Synthesize a single-target slow-time symbol stack.

    Row k, subcarrier n: amplitude * exp(-j 2 pi f_n 2R / c) * exp(+j 2 pi f_d k T_sym)
    """
    freqs = np.arange(cfg.n_subcarriers) * cfg.subcarrier_spacing_hz
    slow = np.arange(n_slow)
    range_phase = np.exp(-1j * 2.0 * np.pi * freqs * (2.0 * range_m) / 299_792_458.0)
    doppler_phase = np.exp(1j * 2.0 * np.pi * doppler_hz * slow * cfg.useful_symbol_s)
    return amplitude * np.outer(doppler_phase, range_phase)
