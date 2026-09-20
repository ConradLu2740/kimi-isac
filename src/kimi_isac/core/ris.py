"""RIS / reflective-array model with aperture-derived gain.

The gain is derived from the effective aperture
``A = N * (d/lambda)^2 * eta`` and ``G = 4 pi A / lambda^2``, never a
hand-picked constant. Phases are unit-modulus by construction.
"""

from __future__ import annotations

import math

import numpy as np


def array_gain_db(
    n_elements: int, spacing_wavelength: float = 0.5, aperture_efficiency: float = 0.8
) -> float:
    """Peak directivity of an N-element planar grid with the given spacing."""
    a_over_lambda2 = n_elements * spacing_wavelength**2 * aperture_efficiency
    return 10.0 * math.log10(4.0 * math.pi * a_over_lambda2)


class RISPanel:
    """A passive reflective panel of unit-modulus phase-shifting elements."""

    def __init__(
        self,
        n_elements: int,
        spacing_wavelength: float = 0.5,
        aperture_efficiency: float = 0.8,
    ) -> None:
        if n_elements < 1:
            raise ValueError("n_elements must be >= 1")
        self.n_elements = n_elements
        self.gain_db = array_gain_db(n_elements, spacing_wavelength, aperture_efficiency)
        self.phases = np.zeros(n_elements, dtype=float)

    def set_phases(self, phases: np.ndarray) -> None:
        phases = np.asarray(phases, dtype=float)
        if phases.shape != (self.n_elements,):
            raise ValueError(f"expected ({self.n_elements},) phases, got {phases.shape}")
        self.phases = phases.copy()

    def random_phases(self, rng: np.random.Generator) -> np.ndarray:
        self.set_phases(rng.uniform(0.0, 2.0 * math.pi, self.n_elements))
        return self.phases

    @staticmethod
    def unit_modulus(phases: np.ndarray) -> np.ndarray:
        """exp(j*phases): the only operation phases may undergo (|.| == 1)."""
        return np.exp(1j * np.asarray(phases, dtype=float))
