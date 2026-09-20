"""Echo model: 3-D cloud -> per-frame complex echo + 10-dim features.

Path model (all via core primitives):
- direct sensing path:   sat -> scatterer -> UE (two free-space hops)
- RIS-assisted path:     sat -> panel element -> scatterer -> UE (three hops),
                         panel phases aligned to the ROI centroid (aligned),
                         uniform random (random), panel off (none), or
                         per-scatterer ideal alignment (oracle; upper bound).
SNR definition: the noiseless direct-path power of frame 0 is normalized
to 1; per-frame noise std is 1/sqrt(snr) relative to that.
Feature vector per frame: [amp_dB, sin(angle Y), cos(angle Y), doppler_kHz,
delay_ms, dist_sat_roi, elev_norm, rcs_log10, irs_sin, irs_cos].
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from kimi_isac.closedloop import FC_HZ, iss_overpass_geometry
from kimi_isac.core.channel import C_LIGHT, doppler_shift_hz
from kimi_isac.core.frames import elevation_deg
from kimi_isac.core.ris import array_gain_db
from kimi_isac.gen import scene
from kimi_isac.opt.phase_opt import align_phases

REFLECTIVITY = {
    "building": 0.9,
    "vehicle": 0.7,
    "uav": 0.4,
    "tank": 0.8,
    "tower": 0.6,
    "cubesat": 0.5,
    "bicycle": 0.3,
    "pedestrian": 0.3,
    "bridge": 0.85,
    "windmill": 0.6,
}


@dataclass(frozen=True)
class EchoConfig:
    carrier_hz: float = FC_HZ
    n_elem: int = 256
    tau: int = 8
    dt_s: float = 0.5
    roi_offset_m: float = scene.ROI_OFFSET_M
    ris_offset_m: float = scene.RIS_OFFSET_M


@functools.lru_cache(maxsize=8)
def _geometry_cached(n_elem: int, tau: int, dt_s: float) -> tuple:
    frames_geo = iss_overpass_geometry(n_frames=tau, dt_s=dt_s, n_elem=n_elem)
    ue, up, east = scene.ue_and_frame()
    anchor = scene.roi_anchor(ue, up)
    elem_spacing = (C_LIGHT / FC_HZ) / 2.0
    ris = np.array(
        [
            anchor + up * scene.RIS_OFFSET_M + east * (i - (n_elem - 1) / 2.0) * elem_spacing
            for i in range(n_elem)
        ]
    )
    return tuple(
        {**g, "ue": ue, "up": up, "east": east, "roi": anchor, "ris": ris} for g in frames_geo
    )


def frame_geometry(cfg: EchoConfig) -> list[dict]:
    return list(_geometry_cached(cfg.n_elem, cfg.tau, cfg.dt_s))


def _fs_amp(dist_m):
    """Free-space amplitude lambda/(4 pi d) (0 dBi antennas), vectorized."""
    return (C_LIGHT / FC_HZ) / (4.0 * np.pi * np.asarray(dist_m, dtype=float))


def _scatterer_paths(fr: dict, scat: np.ndarray):
    """(d1, d2, bistatic range rate) arrays; rate = d/dt(d1 + d2)."""
    sat, v, ue = fr["sat_pos"], fr["sat_vel"], fr["ue"]
    d1 = np.linalg.norm(sat - scat, axis=1)  # (N,)
    d2 = np.linalg.norm(scat - ue, axis=1)  # (N,)
    u1 = (sat - scat) / d1[:, None]
    rate = np.einsum("j,nj->n", v, u1)  # scatterers are static
    return d1, d2, rate


def _elem_chain(fr: dict, scat: np.ndarray, d2: np.ndarray, elem_gain_lin: float):
    """(E, N) complex chain sat->elem->scatterer->UE (amplitude and phase)."""
    sat = fr["sat_pos"]
    ris = fr["ris"]  # (E, 3)
    d_sat_elem = np.linalg.norm(sat - ris, axis=1)  # (E,)
    d_elem_scat = np.linalg.norm(ris[:, None, :] - scat[None, :, :], axis=2)  # (E, N)
    amp = (
        math.sqrt(elem_gain_lin)
        * _fs_amp(d_sat_elem)[:, None]
        * _fs_amp(d_elem_scat)
        * _fs_amp(d2)[None, :]
    )
    phase = 2.0 * np.pi * (d_sat_elem[:, None] + d_elem_scat + d2[None, :])
    return amp * np.exp(-1j * phase / (C_LIGHT / FC_HZ))


def _aligned_phases(fr: dict, centroid: np.ndarray, elem_gain_lin: float, rng) -> np.ndarray:
    """Panel phases coherently illuminating the centroid (what a real
    controller can do: it only knows the dominant target)."""
    sat, ue = fr["sat_pos"], fr["ue"]
    ris = fr["ris"]
    d_sat_elem = np.linalg.norm(sat - ris, axis=1)
    d_elem_c = np.linalg.norm(ris - centroid, axis=1)
    d_c_ue = float(np.linalg.norm(centroid - ue))
    a = (
        math.sqrt(elem_gain_lin)
        * _fs_amp(d_sat_elem)
        * _fs_amp(d_elem_c)
        * _fs_amp(d_c_ue)
        * np.exp(-1j * 2.0 * np.pi * (d_sat_elem + d_elem_c + d_c_ue) / (C_LIGHT / FC_HZ))
    )
    return align_phases(a, 0.0 + 0.0j, seed=int(rng.integers(1 << 31)))


def echo_and_features(
    cloud_local: np.ndarray,
    class_name: str,
    ris_mode: str,
    seed: int,
    snr: float,
    cfg: EchoConfig,
) -> dict:
    if ris_mode not in ("aligned", "random", "none", "oracle"):
        raise ValueError(f"unknown ris_mode {ris_mode!r}")
    rng = np.random.default_rng(seed)
    frames = frame_geometry(cfg)
    ue, up, east = frames[0]["ue"], frames[0]["up"], frames[0]["east"]
    scat = scene.scatterers_ecef(cloud_local, ue, up, east)
    centroid = scat.mean(axis=0)
    rho = REFLECTIVITY[class_name]
    elem_gain_lin = 10 ** (array_gain_db(1) / 10.0)
    lam = C_LIGHT / FC_HZ

    # frame-0 noiseless direct-path power is the SNR reference
    d1_0, d2_0, _ = _scatterer_paths(frames[0], scat)
    ref_power = float(np.sum((rho * _fs_amp(d1_0) * _fs_amp(d2_0)) ** 2))
    noise_std = math.sqrt(ref_power / snr)

    Y = np.empty(cfg.tau, dtype=complex)
    Y_ris_arr = np.zeros(cfg.tau, dtype=complex)
    Y_oracle = np.full(cfg.tau, np.nan, dtype=complex)
    feats = np.empty((cfg.tau, 10), dtype=float)
    for t, fr in enumerate(frames):
        d1, d2, rate = _scatterer_paths(fr, scat)
        # f_D matches core.channel.doppler_shift_hz(rate, FC_HZ)
        f_d = -rate / lam
        doppler_phase = np.exp(1j * 2.0 * np.pi * f_d * fr["t"])
        amp_direct = rho * _fs_amp(d1) * _fs_amp(d2)
        Y_direct = float(np.sum(amp_direct * doppler_phase))

        Y_ris = 0.0 + 0.0j
        phases = None
        if ris_mode in ("aligned", "random", "oracle"):
            if ris_mode == "random":
                phases = rng.uniform(0.0, 2.0 * np.pi, size=fr["ris"].shape[0])
            elif ris_mode == "aligned":
                phases = _aligned_phases(fr, centroid, elem_gain_lin, rng)
            chain = _elem_chain(fr, scat, d2, elem_gain_lin)  # (E, N)
            if ris_mode == "oracle":
                # each scatterer's own element chain perfectly aligned
                Y_ris = float(np.sum(np.abs(chain.sum(axis=0)) * doppler_phase))
                Y_oracle[t] = Y_direct + Y_ris
            else:
                if phases is None:
                    raise RuntimeError("phases must be set for aligned/random modes")
                Y_ris = float(np.sum(chain * np.exp(1j * phases)[:, None] * doppler_phase[None, :]))

        Y[t] = Y_direct + Y_ris
        Y_ris_arr[t] = Y_ris
        Y[t] += noise_std * (rng.normal() + 1j * rng.normal()) / math.sqrt(2.0)

        # features (centroid quantities; index 3 is the centroid Doppler)
        d_sat_c = float(np.linalg.norm(fr["sat_pos"] - centroid))
        d_c_ue = float(np.linalg.norm(centroid - fr["ue"]))
        rr_c = float(np.dot(fr["sat_vel"], (fr["sat_pos"] - centroid) / d_sat_c))
        f_d_c = doppler_shift_hz(rr_c, FC_HZ)
        delay = (d_sat_c + d_c_ue) / C_LIGHT
        feats[t] = [
            20.0 * math.log10(abs(Y[t]) / noise_std + 1e-12),
            math.sin(math.atan2(Y[t].imag, Y[t].real)),
            math.cos(math.atan2(Y[t].imag, Y[t].real)),
            f_d_c / 1e3,
            delay * 1e3,
            float(np.linalg.norm(fr["sat_pos"] - centroid)) / 1e2,
            elevation_deg(fr["ue"], fr["sat_pos"]) / 90.0,
            math.log10(abs(Y[t] / noise_std) ** 2 + 1e-12),
            math.sin(phases[0]) if phases is not None else 0.0,
            math.cos(phases[0]) if phases is not None else 0.0,
        ]
    return {
        "features": feats,
        "Y": Y,
        "Y_ris": Y_ris_arr,
        "Y_oracle": None if np.all(np.isnan(Y_oracle)) else Y_oracle,
    }
