"""RIS phase optimization (coordinate ascent) and segmented reconfiguration.

The optimized objective is the received field at the UE,

    maximize_phi  | h_d + sum_n a_n e^{j phi_n} |^2 ,  |e^{j phi_n}| = 1,

with ``a_n = sqrt(G_elem) * g_sr,n * h_rd,n`` the per-element two-hop
channel (the per-element amplitude carries ``sqrt(G_elem)`` so a coherent
sum reproduces the aperture-derived panel gain ``G = N G_elem`` used in
``core.ris``). Coordinate ascent converges to the unit-modulus optimum;
the greedy bound ``|h_d| + sum|a_n|`` is unreachable but reported.
"""

from __future__ import annotations

import numpy as np


def align_phases(
    a: np.ndarray,
    h_d: complex,
    max_iter: int = 200,
    tol: float = 1e-7,
    seed: int = 0,
) -> np.ndarray:
    """Unit-modulus phase alignment (vectorized Jacobi sweeps).

    With the current total field ``S = h_d + sum_n a_n e^{j phi_n}``, the
    simultaneous update moves every element to
    ``phi_n = angle(S - a_n e^{j phi_n}) - angle(a_n)``; iterating raises
    ``|S|`` monotonically toward the coherent alignment.
    """
    a = np.asarray(a, dtype=complex)
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=a.size)
    field = h_d + np.sum(a * np.exp(1j * phi))
    for _ in range(max_iter):
        phi_new = np.angle(field - a * np.exp(1j * phi)) - np.angle(a)
        field_new = h_d + np.sum(a * np.exp(1j * phi_new))
        converged = abs(abs(field_new) - abs(field)) < tol * abs(field)
        phi, field = phi_new, field_new
        if converged:
            break
    return phi


def received_field(a: np.ndarray, h_d: complex, phi: np.ndarray) -> complex:
    a = np.asarray(a, dtype=complex)
    return h_d + np.sum(a * np.exp(1j * np.asarray(phi, dtype=float)))


def greedy_bound(a: np.ndarray, h_d: complex) -> float:
    """Unreachable upper bound |h_d| + sum|a_n| (independent per-element max)."""
    return abs(h_d) + float(np.sum(np.abs(a)))


def segment_track(tracked_phi: np.ndarray, n_segments: int) -> np.ndarray:
    """Freeze phases into ``n_segments`` consecutive blocks along frames.

    Each block uses the phases optimized at its first frame (a real
    controller can only reconfigure at segment boundaries).
    """
    tracked = np.asarray(tracked_phi, dtype=float)
    n_frames = tracked.shape[0]
    n_segments = max(1, min(n_segments, n_frames))
    bounds = np.linspace(0, n_frames, n_segments + 1, dtype=int)
    out = np.empty_like(tracked)
    for k in range(n_segments):
        lo, hi = bounds[k], bounds[k + 1]
        out[lo:hi] = tracked[lo]
    return out
