"""Physical constants. SI units unless noted.

Values are from CODATA / WGS84 / IERS references; no project-specific
magic numbers belong in this module.
"""

from __future__ import annotations

# Exact (SI defining constants)
C_LIGHT = 299_792_458.0  # m/s
K_BOLTZMANN = 1.380649e-23  # J/K

# WGS84 terrestrial reference system
MU_EARTH = 3.986_004_418e14  # m^3/s^2, geocentric gravitational constant
EARTH_R_EQ = 6_378_137.0  # m, semi-major axis a
EARTH_FLATTENING = 1.0 / 298.257_223_563  # f
EARTH_R_POL = EARTH_R_EQ * (1.0 - EARTH_FLATTENING)  # m, derived
EARTH_ROT_RATE = 7.292_115_0e-5  # rad/s, IERS nominal

# IEEE noise reference temperature (290 K)
T_NOISE_REF = 290.0
