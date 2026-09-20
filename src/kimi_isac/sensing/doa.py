"""ULA direction finding with MUSIC."""

from __future__ import annotations

import numpy as np


def ula_steering(
    n_elements: int, angles_deg: np.ndarray, spacing_wavelength: float = 0.5
) -> np.ndarray:
    """Steering vectors: column k is a(angle_k), shape (n_elements, n_angles)."""
    idx = np.arange(n_elements)[:, None]
    return np.exp(
        1j * 2.0 * np.pi * spacing_wavelength * idx * np.sin(np.radians(angles_deg)[None, :])
    )


def snapshot_matrix(
    angles_deg: np.ndarray,
    n_snapshots: int,
    n_elements: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Incoherent sources at ``angles_deg`` observed over ``n_snapshots`` snapshots."""
    a = ula_steering(n_elements, np.asarray(angles_deg, dtype=float))
    symbols = (
        rng.normal(size=(len(angles_deg), n_snapshots))
        + 1j * rng.normal(size=(len(angles_deg), n_snapshots))
    ) / np.sqrt(2.0)
    signal = a @ symbols
    noise = (
        noise_std
        * (
            rng.normal(size=(n_elements, n_snapshots))
            + 1j * rng.normal(size=(n_elements, n_snapshots))
        )
        / np.sqrt(2.0)
    )
    return signal + noise


def music_spectrum(
    x: np.ndarray, n_sources: int, grid_deg: np.ndarray, spacing_wavelength: float = 0.5
) -> np.ndarray:
    """MUSIC pseudo-spectrum on ``grid_deg`` for snapshot matrix ``x``."""
    r = (x @ x.conj().T) / x.shape[1]
    eigvals, eigvecs = np.linalg.eigh(r)
    noise_subspace = eigvecs[:, :-n_sources]
    a = ula_steering(x.shape[0], grid_deg, spacing_wavelength)
    proj = noise_subspace.conj().T @ a
    denom = np.sum(np.abs(proj) ** 2, axis=0)
    return 1.0 / np.maximum(denom, 1e-30)


def estimate_doa(
    x: np.ndarray, n_sources: int, spacing_wavelength: float = 0.5, grid_step_deg: float = 0.05
) -> np.ndarray:
    """Peak locations (degrees) of the MUSIC spectrum, strongest first."""
    grid = np.arange(-90.0, 90.0 + grid_step_deg, grid_step_deg)
    spec = music_spectrum(x, n_sources, grid, spacing_wavelength)
    order = np.argsort(spec)[::-1]
    peaks: list[float] = []
    for i in order:
        if all(abs(grid[i] - p) > 1.0 for p in peaks):
            peaks.append(float(grid[i]))
        if len(peaks) == n_sources:
            break
    return np.array(sorted(peaks))
