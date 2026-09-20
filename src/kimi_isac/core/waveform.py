"""OFDM sensing waveform parameters and range processing.

Resolution relations (round-trip propagation):
- range resolution          dR = c / (2 B)
- unambiguous range         R_max = c / (2 * subcarrier_spacing)
- Doppler resolution        1 / T_obs
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kimi_isac.core.constants import C_LIGHT


@dataclass(frozen=True)
class OFDMConfig:
    n_subcarriers: int
    bandwidth_hz: float

    @property
    def subcarrier_spacing_hz(self) -> float:
        return self.bandwidth_hz / self.n_subcarriers

    @property
    def useful_symbol_s(self) -> float:
        return 1.0 / self.subcarrier_spacing_hz

    @property
    def sample_interval_s(self) -> float:
        return self.useful_symbol_s / self.n_subcarriers

    @property
    def range_resolution_m(self) -> float:
        return C_LIGHT / (2.0 * self.bandwidth_hz)

    @property
    def max_unambiguous_range_m(self) -> float:
        return C_LIGHT / (2.0 * self.subcarrier_spacing_hz)

    def range_bins_m(self) -> np.ndarray:
        return np.arange(self.n_subcarriers) * self.range_resolution_m


def doppler_nyquist_hz(symbol_interval_s: float) -> float:
    """Nyquist Doppler for slow-time sampling every ``symbol_interval_s``."""
    return 1.0 / (2.0 * symbol_interval_s)


def range_profile(h_freq: np.ndarray, config: OFDMConfig) -> tuple[np.ndarray, np.ndarray]:
    """Frequency response -> (range bins, profile magnitude).

    A point target at range R appears at bin round(2 R f_s / c) of the
    IFFT of H(f); the bin spacing is the range resolution c / (2B).
    """
    h = np.asarray(h_freq, dtype=complex)
    if h.shape != (config.n_subcarriers,):
        raise ValueError(f"expected ({config.n_subcarriers},) subcarriers, got {h.shape}")
    prof = np.abs(np.fft.ifft(h))
    return config.range_bins_m(), prof


def point_target_response(range_m: float, freqs_hz: np.ndarray) -> np.ndarray:
    """Round-trip channel response H(f) = exp(-j 2 pi f * 2R / c)."""
    freqs = np.asarray(freqs_hz, dtype=float)
    return np.exp(-1j * 2.0 * np.pi * freqs * (2.0 * range_m) / C_LIGHT)
