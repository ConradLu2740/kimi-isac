"""Waveform verification: range resolution, target localization, Doppler
estimation, and Parseval energy conservation."""

from __future__ import annotations

import numpy as np

from kimi_isac.core.constants import C_LIGHT
from kimi_isac.core.waveform import (
    OFDMConfig,
    doppler_nyquist_hz,
    point_target_response,
    range_profile,
)


def run() -> bool:
    cfg = OFDMConfig(n_subcarriers=1024, bandwidth_hz=30.72e6)
    freqs = np.arange(cfg.n_subcarriers) * cfg.subcarrier_spacing_hz

    dR = cfg.range_resolution_m
    r_max = cfg.max_unambiguous_range_m
    t_sym = cfg.useful_symbol_s

    resolution_ok = abs(dR - C_LIGHT / (2.0 * 30.72e6)) < 1e-6
    rmax_ok = abs(r_max - C_LIGHT / (2.0 * 30e3)) < 1e-6

    # single target localization: peak within one bin of the injected range
    target_m = 150.0
    _, prof = range_profile(point_target_response(target_m, freqs), cfg)
    peak_range = float(np.argmax(prof) * dR)
    peak_ok = abs(peak_range - target_m) < dR

    # resolvability (Rayleigh-type): two targets 2 bins apart must show a
    # dip below half the peak between them; 0.5 bin apart they share the
    # main lobe (both target bins inside the contiguous -3 dB region)
    def profile_for(sep_m: float) -> np.ndarray:
        h = point_target_response(target_m, freqs) + point_target_response(target_m + sep_m, freqs)
        return range_profile(h, cfg)[1]

    p_res = profile_for(2.0 * dR)
    pk_res = int(np.argmax(p_res))
    b1, b2 = int(round(target_m / dR)), int(round((target_m + 2.0 * dR) / dR))
    inner = p_res[min(b1, b2) + 1 : max(b1, b2)]
    resolved_ok = len(inner) > 0 and bool(inner.min() < 0.5 * p_res[pk_res])

    p_mrg = profile_for(0.5 * dR)
    pk_mrg = int(np.argmax(p_mrg))
    main_lobe = p_mrg > 0.707 * p_mrg[pk_mrg]
    i, j = pk_mrg, pk_mrg
    while i > 0 and main_lobe[i - 1]:
        i -= 1
    while j < len(main_lobe) - 1 and main_lobe[j + 1]:
        j += 1
    b1, b2 = int(round(target_m / dR)), int(round((target_m + 0.5 * dR) / dR))
    merged_ok = i <= b1 <= j and i <= b2 <= j

    # Doppler from slow-time phase progression
    f_d_in = 2_000.0
    m_symbols = 32
    k = np.arange(m_symbols)
    x = np.exp(1j * 2.0 * np.pi * f_d_in * k * t_sym) * (0.6 + 0.8j)
    phase_steps = np.angle(x[1:] * np.conj(x[:-1]))
    f_d_hat = float(np.mean(phase_steps) / (2.0 * np.pi * t_sym))
    doppler_ok = abs(f_d_hat - f_d_in) < 1.0 / (m_symbols * t_sym) / 2
    nyquist_ok = abs(doppler_nyquist_hz(t_sym) - 1.0 / (2.0 * t_sym)) < 1e-12

    # Parseval for numpy's unnormalized ifft: sum|x|^2 == sum|X|^2 / N
    rng = np.random.default_rng(0)
    x_spec = rng.normal(size=cfg.n_subcarriers) + 1j * rng.normal(size=cfg.n_subcarriers)
    x_time = np.fft.ifft(x_spec)
    parseval_err = abs(
        np.sum(np.abs(x_time) ** 2) - np.sum(np.abs(x_spec) ** 2) / cfg.n_subcarriers
    ) / (np.sum(np.abs(x_spec) ** 2) / cfg.n_subcarriers)

    checks = [
        ("range resolution dR = c/2B", resolution_ok, f"{dR:.3f} m"),
        ("unambiguous range = c/2df", rmax_ok, f"{r_max:.1f} m"),
        ("point target localized within 1 bin", peak_ok, f"{peak_range:.2f} m vs {target_m}"),
        ("2-bin targets resolve (dip < 0.5 peak)", resolved_ok, f"sep {2.0 * dR:.2f} m"),
        ("0.5-bin targets share one main lobe", merged_ok, f"sep {0.5 * dR:.2f} m"),
        ("Doppler estimate matches input", doppler_ok, f"{f_d_hat:.2f} vs {f_d_in} Hz"),
        ("Doppler Nyquist = 1/2Tsym", nyquist_ok, f"{doppler_nyquist_hz(t_sym):.1f} Hz"),
        ("Parseval energy conservation", parseval_err < 1e-10, f"rel err {parseval_err:.2e}"),
    ]
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok = ok and passed
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run() else 1)
