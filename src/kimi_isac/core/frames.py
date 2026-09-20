"""Coordinate frames: ECI (inertial) <-> ECEF (earth-fixed) <-> WGS84 geodetic.

Conventions
-----------
- ECEF z-axis points to the geographic north pole; ECI differs from ECEF
  only by the Greenwich apparent sidereal time rotation about +z.
- Geodetic conversion is full WGS84 (elliptical), not spherical.
"""

from __future__ import annotations

import math

import numpy as np

from kimi_isac.core.constants import EARTH_FLATTENING, EARTH_R_EQ

_ARCSEC_PER_DEG = 3600.0


def julian_date_from_datetime(dt) -> float:
    """Julian date (UTC treated as UT1) for a timezone-aware/naive datetime."""
    year = dt.year
    month = dt.month
    day = dt.day
    frac = (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
    day_with_frac = day + frac
    if month <= 2:
        year -= 1
        month += 12
    a = math.floor(year / 100)
    b = 2 - a + math.floor(a / 4)
    jd = (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day_with_frac
        + b
        - 1524.5
    )
    return jd


def gmst_rad(jd_utc: float) -> float:
    """Greenwich mean sidereal time (IAU 1982), radians in [0, 2pi)."""
    d_ut1 = jd_utc - 2451545.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * d_ut1 / 86400.0
        + 0.093104 * d_ut1**2 / 86400.0
        - 6.2e-6 * d_ut1**3 / 86400.0
    )
    return (gmst_sec % 86400.0) / 86400.0 * 2.0 * math.pi


def elevation_deg(observer_ecef: np.ndarray, target_ecef: np.ndarray) -> float:
    """Elevation angle of the target as seen from the observer (degrees).

    The local up direction is the WGS84 geodetic normal at the observer
    (not the geocentric radial — the difference matters at mid-latitudes).
    """
    lat, lon, _ = ecef_to_geodetic(observer_ecef)
    lat_r, lon_r = np.radians(lat), np.radians(lon)
    up = np.array([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
    v = np.asarray(target_ecef, dtype=float) - np.asarray(observer_ecef, dtype=float)
    sin_el = float(np.dot(v, up) / (np.linalg.norm(v) + 1e-12))
    return float(np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0))))


def eci_to_ecef(r_eci: np.ndarray, jd_utc: float) -> np.ndarray:
    """Rotate an inertial-frame vector to the earth-fixed frame."""
    theta = gmst_rad(jd_utc)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot @ np.asarray(r_eci, dtype=float)


def ecef_to_eci(r_ecef: np.ndarray, jd_utc: float) -> np.ndarray:
    """Rotate an earth-fixed vector to the inertial frame."""
    theta = gmst_rad(jd_utc)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot @ np.asarray(r_ecef, dtype=float)


def ecef_to_geodetic(r_ecef: np.ndarray) -> tuple[float, float, float]:
    """WGS84 ECEF -> (latitude_deg, longitude_deg, altitude_m), closed form."""
    x, y, z = (float(v) for v in r_ecef)
    a = EARTH_R_EQ
    f = EARTH_FLATTENING
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)  # second eccentricity squared
    p = math.hypot(x, y)
    lon = math.atan2(y, x)
    theta = math.atan2(z * a, p * a * (1.0 - f))
    lat = math.atan2(
        z + ep2 * a * (1.0 - f) * math.sin(theta) ** 3,
        p - e2 * a * math.cos(theta) ** 3,
    )
    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    alt = p / math.cos(lat) - n if abs(math.cos(lat)) > 1e-12 else abs(z) - n * (1.0 - f)
    return math.degrees(lat), math.degrees(lon), alt


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
    """WGS84 geodetic -> ECEF (m)."""
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    a, f = EARTH_R_EQ, EARTH_FLATTENING
    e2 = f * (2.0 - f)
    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    x = (n + alt_m) * math.cos(lat) * math.cos(lon)
    y = (n + alt_m) * math.cos(lat) * math.sin(lon)
    z = (n * (1.0 - e2) + alt_m) * math.sin(lat)
    return np.array([x, y, z])


def geocentric_altitude_m(r_ecef: np.ndarray) -> float:
    """Altitude above the mean earth radius (the value usually quoted for ISS)."""
    return float(np.linalg.norm(r_ecef) - 6371.0e3)
