"""Synthetic single-dominant-target sensing scenario.

Every sample is generated deterministically from its index and the split
seed, so fixed splits stay reproducible across processes. The range axis
is cropped to the ROI ``[0, range_max_m]``; the velocity-to-Doppler
conversion goes through the carrier frequency (f_D = 2 v / lambda); and
the per-sample SNR is drawn from ``snr_levels`` so one trained model
must cope with the whole difficulty range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kimi_isac.core.constants import C_LIGHT
from kimi_isac.core.waveform import OFDMConfig
from kimi_isac.sensing.rangedoppler import inject_target, range_doppler_map

RANGE_TRAIN = (80.0, 360.0)
VEL_TRAIN = (-35.0, 35.0)
RANGE_OOD = ((20.0, 60.0), (380.0, 480.0))
VEL_OOD = ((-60.0, -45.0), (45.0, 60.0))
SNR_LEVELS = (6.0, 1.0, 0.25)

CLASS_NAMES = ("ground", "slow", "fast")


def classify_velocity(vel_mps: float) -> int:
    if abs(vel_mps) < 2.0:
        return 0
    if abs(vel_mps) < 12.0:
        return 1
    return 2


@dataclass(frozen=True)
class ScenarioConfig:
    carrier_hz: float = 30.0e9
    n_subcarriers: int = 1024
    bandwidth_hz: float = 30.72e6
    n_slow: int = 32
    snr_levels: tuple[float, ...] = SNR_LEVELS
    target_prob: float = 0.5
    range_max_m: float = 480.0
    vel_max_mps: float = 75.0  # Nyquist Doppler of the slow-time grid, in m/s
    range_pool: int = 2  # 98 ROI range bins -> 49
    doppler_pool: int = 2  # 32 doppler bins -> 16

    @property
    def ofdm(self) -> OFDMConfig:
        return OFDMConfig(n_subcarriers=self.n_subcarriers, bandwidth_hz=self.bandwidth_hz)

    @property
    def range_resolution_m(self) -> float:
        return self.ofdm.range_resolution_m

    @property
    def n_range_full(self) -> int:
        return int(round(self.range_max_m / self.range_resolution_m))

    @property
    def doppler_bins_mps(self) -> np.ndarray:
        """Slow-time Doppler axis converted to radial velocity via v = f_D * lambda / 2."""
        f = np.fft.fftshift(np.fft.fftfreq(self.n_slow, d=self.ofdm.useful_symbol_s))
        return f * C_LIGHT / (2.0 * self.carrier_hz)

    def vel_to_doppler_hz(self, vel_mps: float) -> float:
        return 2.0 * vel_mps * self.carrier_hz / C_LIGHT


def _draw_range(rng: np.random.Generator, ood: bool) -> float:
    if not ood:
        return float(rng.uniform(*RANGE_TRAIN))
    band = RANGE_OOD[int(rng.integers(0, 2))]
    return float(rng.uniform(*band))


def _draw_velocity(rng: np.random.Generator, ood: bool) -> float:
    if not ood:
        return float(rng.uniform(*VEL_TRAIN))
    band = VEL_OOD[int(rng.integers(0, 2))]
    return float(rng.uniform(*band))


def _pool(m: np.ndarray, range_pool: int, doppler_pool: int) -> np.ndarray:
    """Average pool (range, doppler) -> (range//range_pool, doppler//doppler_pool)."""
    r = m.shape[0] // range_pool * range_pool
    d = m.shape[1] // doppler_pool * doppler_pool
    m = m[:r, :d]
    return m.reshape(r // range_pool, range_pool, d // doppler_pool, doppler_pool).mean(axis=(1, 3))


def make_sample(idx: int, seed: int, cfg: ScenarioConfig, ood: bool = False) -> dict:
    """One sample: pooled map (ML input), full-res map (classical input), labels."""
    rng = np.random.default_rng((seed * 1_000_003 + idx) % (2**63))
    has_target = bool(rng.random() < cfg.target_prob)
    range_m = _draw_range(rng, ood)
    vel = _draw_velocity(rng, ood)
    snr = float(cfg.snr_levels[int(rng.integers(0, len(cfg.snr_levels)))])
    amp = 1.0 if has_target else 0.0
    symbols = inject_target(
        cfg.ofdm, cfg.n_slow, range_m, cfg.vel_to_doppler_hz(vel), amplitude=amp + 0.0j
    )
    noise_std = 1.0 / np.sqrt(snr)
    noise = (
        noise_std
        * (rng.normal(size=symbols.shape) + 1j * rng.normal(size=symbols.shape))
        / np.sqrt(2.0)
    )
    rd = range_doppler_map(symbols + noise, cfg.ofdm)["map"]  # (1024, 32)
    rd = rd[: cfg.n_range_full]  # crop to the ROI
    # Fixed scale: log1p of absolute magnitude. Never divide by the per-sample
    # max (that would erase the absolute level the energy detector needs).
    log_map = np.log1p(rd)
    pooled = _pool(log_map, cfg.range_pool, cfg.doppler_pool)
    return {
        "map": pooled.astype(np.float32),
        "map_full": log_map.astype(np.float32),
        "has_target": np.float32(1.0 if has_target else 0.0),
        "range_m": np.float32(range_m),
        "vel_mps": np.float32(vel),
        "class": np.int64(classify_velocity(vel)),
        "snr": np.float32(snr),
        "ood": np.bool_(ood),
    }
