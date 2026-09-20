"""ROI placement and the local ENU frame for the echo scenario."""

from __future__ import annotations

import numpy as np

from kimi_isac.core import frames

UE_GEODETIC = (39.9042, 116.4074)
ROI_OFFSET_M = 200.0
RIS_OFFSET_M = 30.0


def ue_and_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ue_ecef, up, east) for the fixed UE location."""
    ue = frames.geodetic_to_ecef(UE_GEODETIC[0], UE_GEODETIC[1], 0.0)
    up = ue / np.linalg.norm(ue)
    east = np.cross(np.array([0.0, 0.0, 1.0]), up)
    east /= np.linalg.norm(east)
    return ue, up, east


def roi_anchor(ue: np.ndarray, up: np.ndarray) -> np.ndarray:
    return ue + up * ROI_OFFSET_M


def scatterers_ecef(
    cloud_local: np.ndarray, ue: np.ndarray, up: np.ndarray, east: np.ndarray
) -> np.ndarray:
    """ROI-local meters (x east, y north, z up) -> ECEF."""
    north = np.cross(up, east)
    basis = np.stack([east, north, up], axis=1)
    anchor = roi_anchor(ue, up)
    return anchor + cloud_local @ basis.T
