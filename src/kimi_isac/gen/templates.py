"""Procedural point-cloud templates: parameterized 3-D object classes.

Radar sees surfaces, so every builder samples points on the *surface* of
composed primitives (boxes standing in for cylinders/rings). Each instance
is parameterized by the rng, so a class yields infinitely many instances
while staying recognizable. All clouds have exactly N_POINTS points,
normalized to the ROI extent [-1, 1]^3 (see gen/scene.py for the metric
interpretation).
"""

from __future__ import annotations

import numpy as np

TRAIN_CLASSES: tuple[str, ...] = (
    "building",
    "vehicle",
    "uav",
    "tank",
    "tower",
    "cubesat",
    "bicycle",
    "pedestrian",
)
OOD_CLASSES: tuple[str, ...] = ("bridge", "windmill")

N_POINTS = 512
ROI_EXTENT_M = 80.0


def all_class_names() -> list[str]:
    return list(TRAIN_CLASSES) + list(OOD_CLASSES)


def class_index(name: str) -> int:
    names = all_class_names()
    if name not in names:
        raise ValueError(f"unknown class {name!r}")
    return names.index(name)


def _box_surface(
    rng: np.random.Generator, center: np.ndarray, size: np.ndarray, n: int
) -> np.ndarray:
    """Uniform points on the surface of an axis-aligned box."""
    center = np.asarray(center, dtype=float)
    size = np.asarray(size, dtype=float)
    side = rng.integers(0, 6, size=n)
    uv = rng.uniform(-0.5, 0.5, size=(n, 2))
    pts = np.repeat(center[None, :], n, axis=0)
    specs = [(0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0)]
    for s, (axis, sign) in enumerate(specs):
        m = side == s
        if not m.any():
            continue
        p = np.repeat(center[None, :], int(m.sum()), axis=0)
        others = [a for a in range(3) if a != axis]
        p[:, others[0]] += uv[m, 0] * size[others[0]]
        p[:, others[1]] += uv[m, 1] * size[others[1]]
        p[:, axis] += sign * size[axis] / 2.0
        pts[m] = p
    return pts


def _ring(
    rng: np.random.Generator, center: np.ndarray, radius: float, height: float, n: int
) -> np.ndarray:
    """Points on a thin vertical ring (cylinder wall), approximated by a 12-gon."""
    n_gon = 12
    angles = np.arange(n_gon) * 2.0 * np.pi / n_gon
    idx = rng.integers(0, n_gon, size=n)
    x = center[0] + radius * np.cos(angles[idx])
    y = center[1] + radius * np.sin(angles[idx])
    z = center[2] + rng.uniform(-height / 2.0, height / 2.0, size=n)
    return np.stack([x, y, z], axis=1)


def _distribute(rng: np.random.Generator, parts: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Concatenate parts with weights, then resample/trim to exactly N_POINTS."""
    total_w = sum(w for _, w in parts)
    counts = [max(1, int(round(N_POINTS * w / total_w))) for _, w in parts]
    clouds = []
    for (pts, _), c in zip(parts, counts):
        if len(pts) >= c:
            choice = rng.choice(len(pts), size=c, replace=False)
            clouds.append(pts[choice])
        else:
            extra = rng.choice(len(pts), size=c - len(pts), replace=True)
            clouds.append(np.concatenate([pts, pts[extra]], axis=0))
    out = np.concatenate(clouds, axis=0)
    if len(out) > N_POINTS:
        out = out[rng.choice(len(out), size=N_POINTS, replace=False)]
    elif len(out) < N_POINTS:
        extra = out[rng.choice(len(out), size=N_POINTS - len(out), replace=True)]
        out = np.concatenate([out, extra], axis=0)
    return out


def _ground_offset(rng: np.random.Generator, footprint: float) -> np.ndarray:
    return np.array(
        [
            rng.uniform(-1.0, 1.0) * (ROI_EXTENT_M / 2.0 - footprint),
            rng.uniform(-1.0, 1.0) * (ROI_EXTENT_M / 2.0 - footprint),
            0.0,
        ]
    )


def _b_building(rng: np.random.Generator) -> np.ndarray:
    n_boxes = int(rng.integers(1, 4))
    h0 = rng.uniform(10.0, 20.0)
    w = rng.uniform(8.0, 20.0)
    base = _ground_offset(rng, w / 2.0)
    parts = []
    z = 0.0
    for _ in range(n_boxes):
        hh = h0 * rng.uniform(0.6, 1.4)
        center = base + np.array([0.0, 0.0, z + hh / 2.0])
        parts.append((_box_surface(rng, center, np.array([w, w, hh]), 256), 1.0))
        z += hh
    return _distribute(rng, parts)


def _b_vehicle(rng: np.random.Generator) -> np.ndarray:
    length = rng.uniform(4.0, 6.0)
    width = rng.uniform(1.8, 2.2)
    height = rng.uniform(1.4, 1.8)
    base = _ground_offset(rng, length / 2.0) + np.array([0.0, 0.0, height / 2.0])
    parts = [(_box_surface(rng, base, np.array([length, width, height]), 320), 3.0)]
    for dx in (-length / 2.0 + 0.5, length / 2.0 - 0.5):
        for dy in (-width / 2.0, width / 2.0):
            wheel = base + np.array([dx, dy, -height / 2.0 + 0.35])
            parts.append((_box_surface(rng, wheel, np.array([0.7, 0.25, 0.7]), 24), 1.0))
    return _distribute(rng, parts)


def _b_uav(rng: np.random.Generator) -> np.ndarray:
    body = _box_surface(rng, np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.25]), 200)
    parts = [(body, 2.0)]
    arm = rng.uniform(0.6, 1.0)
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            tip = np.array([sx * arm, sy * arm, 0.0])
            mid = tip / 2.0 + np.array([0.0, 0.0, -0.05])
            parts.append((_box_surface(rng, mid, np.array([arm, 0.06, 0.06]), 32), 1.0))
            parts.append((_box_surface(rng, tip, np.array([0.5, 0.5, 0.02]), 24), 1.0))
    pts = _distribute(rng, parts)
    off = _ground_offset(rng, 2.0) + np.array([0.0, 0.0, rng.uniform(5.0, 30.0)])
    return pts + off


def _b_tank(rng: np.random.Generator) -> np.ndarray:
    hull = _box_surface(rng, np.array([0.0, 0.0, 0.9]), np.array([6.5, 3.2, 1.2]), 280)
    turret = _ring(rng, np.array([-0.3, 0.0, 1.9]), 1.2, 0.7, 120)
    barrel = _box_surface(rng, np.array([3.4, 0.0, 1.9]), np.array([4.0, 0.18, 0.18]), 40)
    return _distribute(rng, [(hull, 3.0), (turret, 1.0), (barrel, 0.5)])


def _b_tower(rng: np.random.Generator) -> np.ndarray:
    h = rng.uniform(15.0, 30.0)
    shaft = _box_surface(rng, np.array([0.0, 0.0, h / 2.0]), np.array([2.0, 2.0, h]), 300)
    parts = [(shaft, 2.0)]
    for i in range(3):
        z = h * (i + 1) / 4.0
        parts.append(
            (_box_surface(rng, np.array([0.0, 0.0, z]), np.array([5.0, 5.0, 0.15]), 40), 1.0)
        )
    return _distribute(rng, parts)


def _b_cubesat(rng: np.random.Generator) -> np.ndarray:
    body = _box_surface(rng, np.array([0.0, 0.0, 0.0]), np.array([0.3, 0.3, 0.4]), 200)
    panels = []
    for sy in (-1.0, 1.0):
        panels.append(
            (
                _box_surface(rng, np.array([0.0, sy * 0.45, 0.0]), np.array([0.6, 0.02, 0.7]), 80),
                1.0,
            )
        )
    pts = _distribute(rng, [(body, 2.0), *panels])
    off = _ground_offset(rng, 1.0) + np.array([0.0, 0.0, rng.uniform(40.0, 60.0)])
    return pts + off


def _b_bicycle(rng: np.random.Generator) -> np.ndarray:
    parts = []
    for dx in (-0.55, 0.55):
        parts.append((_ring(rng, np.array([dx, 0.0, 0.35]), 0.35, 0.03, 60), 1.0))
    parts.append(
        (_box_surface(rng, np.array([0.0, 0.0, 0.75]), np.array([0.9, 0.05, 0.05]), 40), 1.0)
    )
    parts.append(
        (_box_surface(rng, np.array([-0.2, 0.0, 1.05]), np.array([0.25, 0.12, 0.05]), 20), 1.0)
    )
    return _distribute(rng, parts) + _ground_offset(rng, 1.0)


def _b_pedestrian(rng: np.random.Generator) -> np.ndarray:
    parts = [
        (_box_surface(rng, np.array([0.0, 0.0, 1.62]), np.array([0.22, 0.22, 0.24]), 40), 1.0),
        (_box_surface(rng, np.array([0.0, 0.0, 1.25]), np.array([0.45, 0.28, 0.6]), 140), 2.0),
        (_box_surface(rng, np.array([-0.1, 0.0, 0.62]), np.array([0.3, 0.28, 0.7]), 120), 2.0),
        (_box_surface(rng, np.array([0.12, 0.0, 0.62]), np.array([0.3, 0.28, 0.7]), 120), 2.0),
    ]
    return _distribute(rng, parts)


def _b_bridge(rng: np.random.Generator) -> np.ndarray:
    span = rng.uniform(40.0, 70.0)
    deck = _box_surface(rng, np.array([0.0, 0.0, 4.0]), np.array([span, 8.0, 0.6]), 300)
    parts = [(deck, 2.0)]
    for i in range(int(rng.integers(2, 5))):
        x = -span / 2.0 + span * (i + 0.5) / 4.0
        parts.append(
            (_box_surface(rng, np.array([x, 0.0, 2.0]), np.array([1.2, 1.2, 4.0]), 40), 1.0)
        )
    return _distribute(rng, parts)


def _b_windmill(rng: np.random.Generator) -> np.ndarray:
    h = rng.uniform(20.0, 40.0)
    shaft = _box_surface(rng, np.array([0.0, 0.0, h / 2.0]), np.array([2.5, 2.5, h]), 220)
    nacelle = _box_surface(rng, np.array([0.0, 0.0, h]), np.array([3.0, 2.5, 2.5]), 60)
    parts = [(shaft, 2.0), (nacelle, 1.0)]
    for k in range(3):
        ang = k * 2.0 * np.pi / 3.0
        direction = np.array([np.cos(ang), np.sin(ang), 0.0])
        mid = np.array([0.0, 0.0, h]) + direction * 11.0
        parts.append((_box_surface(rng, mid, np.array([22.0, 0.8, 0.15]), 60), 1.0))
    return _distribute(rng, parts)


_BUILDERS = {
    "building": _b_building,
    "vehicle": _b_vehicle,
    "uav": _b_uav,
    "tank": _b_tank,
    "tower": _b_tower,
    "cubesat": _b_cubesat,
    "bicycle": _b_bicycle,
    "pedestrian": _b_pedestrian,
    "bridge": _b_bridge,
    "windmill": _b_windmill,
}


def _normalize(pts: np.ndarray) -> np.ndarray:
    half = ROI_EXTENT_M / 2.0
    return np.clip(pts / half, -1.0, 1.0)


def make_template(class_name: str, rng: np.random.Generator) -> np.ndarray:
    """One parameterized instance of ``class_name`` as (N_POINTS, 3) in [-1,1]^3.

    Instances vary in shape parameters, ROI position, and yaw orientation —
    yaw matters: without it the class prior alone would nearly determine the
    cloud and 'reconstruction' would degenerate into template recall.
    """
    if class_name not in _BUILDERS:
        raise ValueError(f"unknown class {class_name!r}")
    pts = _BUILDERS[class_name](rng)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pts = pts @ rot.T
    if pts.shape != (N_POINTS, 3):
        raise RuntimeError(f"builder {class_name} produced {pts.shape}")
    return _normalize(pts)
