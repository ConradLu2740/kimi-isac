# Conditional Diffusion 3D Reconstruction Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the `gen/` subsystem (procedural point-cloud templates, physics-grounded echo conditioning, latent VAE + DiT diffusion, multi-seed CD reporting) to `kimi-isac`, plus a shared training engine extracted from `ml/train.py`.

**Architecture:** `gen/templates.py` generates parameterized 3D clouds in-code; `gen/scene.py` + `gen/echo.py` turn clouds into per-frame 10-dim ISAC conditioning features using only `core/` physics (ISS overpass geometry, two-hop channels, RIS phases aligned to the ROI centroid); `gen/vae.py` + `gen/dit.py` implement latent diffusion with cross-attention conditioning; `gen/train.py` and `gen/report.py` run on the extracted `training/engine.py` with val-based selection and bootstrap CIs.

**Tech Stack:** Python 3.11, numpy, torch (CPU smoke / RTX 5060 GPU full runs), pytest, sgp4.

**Spec:** `docs/superpowers/specs/2026-09-20-diffusion-3d-reconstruction-design.md`

## Global Constraints

- All physics constants and channel/Doppler conventions come from `core/`; `gen/` introduces no new physics constants.
- No network downloads: point clouds are procedural (8 train classes + 2 OOD: `bridge`, `windmill`).
- Every dataset split goes through `core/splits.py` and is persisted under `results/splits/`.
- Checkpoints carry metadata (epoch, config, metrics) via `ml/checkpoint.py`.
- All reported metrics are means with bootstrap 95% CIs via `core/stats.py`, n >= 10 seeds for full runs.
- `make`-equivalent commands: `python -m kimi_isac.gen.train --smoke` and `python -m kimi_isac.gen.report --smoke` must run CPU-only in under 60 s each.
- ruff (line length 100) + mypy (loose) must pass after every task.

## Review Focus

- Echo physics regressions (Doppler sign, delay scaling) — pinned by `tests/test_gen_echo.py` energy/delay/sign tests (Task 3).
- RIS alignment benefit vanishing (silent optimizer failure) — pinned by MC test that aligned mean echo energy exceeds random (Task 3).
- Cloud normalization leaking absolute scale into features — pinned by feature-finiteness + normalization tests (Task 2, Task 3).
- VAE reconstruction floor dominating diffusion CD — pinned by separate VAE recon CD metric reported alongside (Task 7, Task 8).
- Engine refactor silently changing `ml` behavior — pinned by running the full v1 suite before and after extraction (Task 1).

---

### Task 0: Version-control baseline

**Files:**
- Create: `.git` repository (local only; no remote)

**Interfaces:**
- Consumes: nothing
- Produces: a git history so Tasks 1-9 can commit

- [ ] **Step 1: Initialize repo and commit the v1 state**

```bash
git init -b main
git add -A
git commit -m "v1: physics core, classical sensing, ML head, RIS closed loop"
```

- [ ] **Step 2: Verify log**

Run: `git log --oneline`
Expected: one commit on `main`.

---

### Task 1: Extract shared training engine, refactor `ml/train.py` onto it

**Files:**
- Create: `src/kimi_isac/training/__init__.py`, `src/kimi_isac/training/engine.py`
- Modify: `src/kimi_isac/ml/train.py` (replace the inline epoch loop with engine calls; behavior unchanged)
- Test: existing suite (`tests/`) must stay green

**Interfaces:**
- Consumes: `ml/checkpoint.py:save_checkpoint/load_checkpoint`
- Produces:
  - `run_training(hooks: EngineHooks, *, epochs: int, patience: int, log: logging.Logger) -> dict`
    - `EngineHooks` dataclass with fields: `build_model() -> nn.Module`, `make_optimizer(model) -> tuple[optim, sched]`, `train_epoch(model, optim, loader) -> float`, `val_loss(model, loader) -> float`, `config: dict`, `out_dir: Path`, `clip_norm: float = 1.0`
    - Returns `{"best_val": float, "best_epoch": int, "epochs_run": int}`; on return, the best checkpoint is already loaded back into the model.

- [ ] **Step 1: Run the existing suite as the regression guard**

Run: `python -m pytest -q`
Expected: 23 passed.

- [ ] **Step 2: Write the failing import test**

Create `tests/test_engine.py`:

```python
import torch
from torch import nn

from kimi_isac.training.engine import EngineHooks, run_training


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(2, 1)

    def forward(self, x):
        return self.lin(x)


def test_engine_selects_best_val_and_loads_it(tmp_path):
    log = __import__("logging").getLogger("test")
    hooks = EngineHooks(
        build_model=_Tiny,
        make_optimizer=lambda m: (torch.optim.SGD(m.parameters(), lr=0.1), None),
        train_epoch=lambda m, o, l: 0.0,
        val_loss=lambda m, l: 0.5,
        config={"tiny": True},
        out_dir=tmp_path,
    )
    result = run_training(hooks, epochs=3, patience=2, log=log)
    assert result["epochs_run"] == 3
    assert result["best_epoch"] == 0
    assert (tmp_path / "best.pth").exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_engine.py -q`
Expected: FAIL (no module `kimi_isac.training`).

- [ ] **Step 4: Implement the engine**

Create `src/kimi_isac/training/engine.py`:

```python
"""Shared training-loop engine: val-based selection, early stopping, LR
scheduling, gradient clipping, checkpoint metadata. One implementation,
used by every trainer in this project."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from kimi_isac.ml.checkpoint import load_checkpoint, save_checkpoint


@dataclass
class EngineHooks:
    build_model: Callable[[], nn.Module]
    make_optimizer: Callable[[nn.Module], tuple]
    train_epoch: Callable[[nn.Module, torch.optim.Optimizer, object], float]
    val_loss: Callable[[nn.Module, object], float]
    config: dict
    out_dir: Path
    clip_norm: float = 1.0
    device: str = "cpu"


def run_training(
    hooks: EngineHooks, *, epochs: int, patience: int, log: logging.Logger
) -> dict:
    """Run the standard loop; returns stats. Best checkpoint is loaded back."""
    model = hooks.build_model().to(hooks.device)
    opt, sched = hooks.make_optimizer(model)
    best_val = float("inf")
    best_epoch = -1
    bad = 0
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        train_loss = hooks.train_epoch(model, opt, None)
        val = hooks.val_loss(model, None)
        if sched is not None:
            sched.step(val)
        if val < best_val - 1e-5:
            best_val, best_epoch, bad = val, epoch, 0
            save_checkpoint(
                hooks.out_dir / "best.pth", model, epoch=epoch,
                config=hooks.config, metrics={"val_loss": val}, optimizer=opt,
            )
        else:
            bad += 1
        log.info(
            "epoch %3d/%d train=%.4f val=%.4f lr=%.2e (%.1fs)",
            epoch + 1, epochs, train_loss, val,
            opt.param_groups[0]["lr"], time.time() - t0,
        )
        if bad >= patience:
            log.info("early stop at epoch %d (best %d)", epoch + 1, best_epoch + 1)
            break
    load_checkpoint(hooks.out_dir / "best.pth", model)
    model.to(hooks.device)
    return {"best_val": best_val, "best_epoch": best_epoch, "epochs_run": epoch + 1}
```

Create `src/kimi_isac/training/__init__.py` with a one-line docstring.

Note: loaders are passed through the hooks' closures (captured in the
lambdas), so the engine signature stays model-agnostic.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_engine.py -q`
Expected: PASS.

- [ ] **Step 6: Refactor `ml/train.py` to use the engine**

Replace the inline epoch loop (currently lines ~79-124 of `ml/train.py`) with
an `EngineHooks` construction that closes over the loaders and loss:

```python
    def _train_epoch(model, opt, _):
        model.train()
        tot = 0.0
        for batch in train_loader:
            maps = batch["map"].to(device)
            targets = {k: v.to(device) for k, v in batch.items()
                       if k not in ("map", "map_full")}
            opt.zero_grad()
            loss = loss_fn(model(maps), targets, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tot += float(loss.item())
        return tot / max(1, len(train_loader))

    hooks = EngineHooks(
        build_model=lambda: SensingNet(),
        make_optimizer=lambda m: _make_opt(m, lr, patience),
        train_epoch=_train_epoch,
        val_loss=_val_loss,
        config=config,
        out_dir=out_dir,
        device=device,
        clip_norm=1.0,
    )
    run_training(hooks, epochs=epochs, patience=patience, log=log)
```

with module-level helpers defined in `ml/train.py`:

```python
def _make_opt(model, lr, patience):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=max(1, patience // 2)
    )
    return opt, sched
```

Keep `config`, `out_dir`, `device`, `clip_norm` as before; delete the now-unused
`save_checkpoint`/`load_checkpoint` imports from `ml/train.py` if unused.
Keep the log line format identical.

- [ ] **Step 7: Run the full v1 suite**

Run: `python -m pytest -q`
Expected: 24 passed (23 + engine test).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: extract shared training engine; ml/train.py behavior unchanged"
```

---

### Task 2: Procedural point-cloud templates

**Files:**
- Create: `src/kimi_isac/gen/__init__.py`, `src/kimi_isac/gen/templates.py`
- Test: `tests/test_gen_templates.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `TRAIN_CLASSES: tuple[str, ...]` = `("building", "vehicle", "uav", "tank", "tower", "cubesat", "bicycle", "pedestrian")`
  - `OOD_CLASSES: tuple[str, ...]` = `("bridge", "windmill")`
  - `N_POINTS: int = 512`, `ROI_EXTENT_M: float = 80.0`
  - `class_index(name: str) -> int` (train 0..7, OOD 8..9)
  - `make_template(class_name: str, rng: np.random.Generator) -> np.ndarray` — shape `(N_POINTS, 3)`, values in `[-1, 1]^3` (ROI-normalized)

- [ ] **Step 1: Write the failing test**

Create `tests/test_gen_templates.py`:

```python
import numpy as np

from kimi_isac.gen import templates as tpl


def test_shapes_and_normalization():
    rng = np.random.default_rng(0)
    for name in tpl.TRAIN_CLASSES + tpl.OOD_CLASSES:
        cloud = tpl.make_template(name, rng)
        assert cloud.shape == (tpl.N_POINTS, 3)
        assert cloud.min() >= -1.0 and cloud.max() <= 1.0


def test_deterministic_per_seed():
    a = tpl.make_template("building", np.random.default_rng(7))
    b = tpl.make_template("building", np.random.default_rng(7))
    assert np.array_equal(a, b)
    c = tpl.make_template("building", np.random.default_rng(8))
    assert not np.array_equal(a, c)


def test_class_index_unique():
    names = tpl.TRAIN_CLASSES + tpl.OOD_CLASSES
    idx = [tpl.class_index(n) for n in names]
    assert len(set(idx)) == len(names)


def test_distinct_classes_differ():
    rng = np.random.default_rng(1)
    a = tpl.make_template("building", rng)
    b = tpl.make_template("vehicle", rng)
    assert not np.allclose(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gen_templates.py -q`
Expected: FAIL (no module `kimi_isac.gen`).

- [ ] **Step 3: Implement templates**

Create `src/kimi_isac/gen/templates.py`. Each class is a parameterized
composition of boxes/cylinders sampled on the surface (points on faces, not
volumes — radar sees surfaces). Helper:

```python
def _box_surface(rng, center, size, n):
    """Uniform points on the surface of an axis-aligned box (center (3,), size (3,))."""
    faces = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
    side = rng.integers(0, 6, size=n)
    uv = rng.uniform(-0.5, 0.5, size=(n, 2))
    pts = np.repeat(center[None, :], n, axis=0)
    # +x, -x, +y, -y, +z, -z faces
    for s, (axis, sign) in enumerate([(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]):
        m = side == s
        p = np.repeat(center[None, :], m.sum(), axis=0)
        others = [a for a in range(3) if a != axis]
        p[:, others[0]] += uv[m, 0] * size[others[0]]
        p[:, others[1]] += uv[m, 1] * size[others[1]]
        p[:, axis] += sign * size[axis] / 2.0
        pts[m] = p
    return pts
```

Class builders (all take `rng`, return `(N_POINTS, 3)` in meters inside the
ROI, then `_normalize`):

```python
def _normalize(pts: np.ndarray) -> np.ndarray:
    half = ROI_EXTENT_M / 2.0
    return np.clip(pts / half, -1.0, 1.0)
```

- `building`: 1-3 stacked boxes (height 10-40 m, footprint 8-20 m), surface
  points, jittered.
- `vehicle`: elongated box (4-6 m x 1.8-2.2 m x 1.4-1.8 m) + 4 wheel
  cylinders approximated by small boxes.
- `uav`: central box (0.5 m) + 4 arm boxes radiating + rotor disks
  (flattened boxes).
- `tank`: hull box (6-7 m) + turret cylinder (approximated by 12-gon box
  ring) + barrel thin long box.
- `tower`: thin vertical box (2 m x 2 m x 15-30 m) + 3 platform rings
  (flattened boxes at heights).
- `cubesat`: 2-3 stacked 10-30 cm boxes + 2 solar panel thin boxes.
- `bicycle`: 2 wheel rings (12-gon), frame thin boxes, saddle box.
- `pedestrian`: capsule approximated by stacked 5 boxes (head/torso/legs),
  1.7 m tall.
- `bridge` (OOD): deck long box (40-70 m) + 2-4 pylons + cable thin boxes.
- `windmill` (OOD): tower tapered box (20-40 m) + nacelle box + 3 blades
  (long thin boxes at 120°).

Each builder returns exactly `N_POINTS` points (resample without
replacement / tile-and-trim if needed). Register in a dict
`_BUILDERS = {name: fn}` and implement `make_template` as lookup + build +
normalize.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gen_templates.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(gen): procedural point-cloud templates (8 train + 2 OOD classes)"
```

---

### Task 3: Scene geometry and echo model

**Files:**
- Create: `src/kimi_isac/gen/scene.py`, `src/kimi_isac/gen/echo.py`
- Test: `tests/test_gen_echo.py`

**Interfaces:**
- Consumes: `closedloop.iss_overpass_geometry` (ISS frames), `core.channel.*`,
  `core.ris.array_gain_db`, `opt.phase_opt.align_phases`, `gen/templates.py`
- Produces:
  - `scene.UE_GEODETIC: tuple[float, float] = (39.9042, 116.4074)`
  - `scene.ROI_OFFSET_M: float = 200.0`, `scene.RIS_OFFSET_M: float = 30.0`
  - `scene.scatterers_ecef(cloud_local: np.ndarray, ue_ecef: np.ndarray, up: np.ndarray, east: np.ndarray) -> np.ndarray` — `(N, 3)` ECEF positions, ROI-local meters → ECEF using the local ENU frame at the UE
  - `echo.EchoConfig` dataclass: `carrier_hz=30e9, n_elem=256, tau=8, dt_s=0.5, ris_offset_m=30.0, roi_offset_m=200.0`
  - `echo.frame_geometry(cfg) -> list[dict]` — cached (module-level `functools.lru_cache` keyed by cfg) list of frames with `sat_pos`, `sat_vel`, `ue`, `ris` (n_elem positions near the ROI), `roi_anchor`, `up`, `east`
  - `echo.echo_and_features(cloud_local: np.ndarray, class_name: str, ris_mode: str, seed: int, snr: float, cfg: EchoConfig) -> dict` with keys `features (tau, 10)`, `Y (tau,) complex`, `Y_oracle (tau,) complex | None`
  - `ris_mode ∈ {"aligned", "random", "none", "oracle"}`; `"oracle"` returns per-scatterer ideal alignment (upper bound) and also fills `Y_oracle`
  - SNR definition: the noiseless direct-path power of frame 0 is normalized to 1; noise std per frame is `1/sqrt(snr)` relative to that. State this in the module docstring.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gen_echo.py`:

```python
import numpy as np

from kimi_isac.core.constants import C_LIGHT
from kimi_isac.gen import echo, scene, templates

SMOKE_CFG = echo.EchoConfig(n_elem=32, tau=4)


def _cloud():
    return templates.make_template("building", np.random.default_rng(3))


def test_feature_shape_and_finiteness():
    out = echo.echo_and_features(_cloud(), "building", "aligned", seed=0, snr=6.0, cfg=SMOKE_CFG)
    f = out["features"]
    assert f.shape == (SMOKE_CFG.tau, 10)
    assert np.all(np.isfinite(f))


def test_delay_feature_matches_bistatic_geometry():
    # feature[4] is delay_ms; compare against an independent computation
    cfg = SMOKE_CFG
    frames = echo.frame_geometry(cfg)
    cloud = _cloud()
    scat = scene.scatterers_ecef(cloud, frames[0]["ue"], frames[0]["up"], frames[0]["east"])
    centroid = scat.mean(axis=0)
    delay = (np.linalg.norm(frames[0]["sat_pos"] - centroid)
             + np.linalg.norm(centroid - frames[0]["ue"])) / C_LIGHT * 1e3
    f = echo.echo_and_features(cloud, "building", "none", seed=0, snr=1e6, cfg=cfg)["features"]
    assert abs(f[0, 4] - delay) < 1e-6


def test_doppler_matches_core_convention():
    # The vectorized per-scatterer Doppler must equal core.channel's scalar
    # convention (approaching satellite -> positive shift) for the centroid.
    from kimi_isac.core.channel import C_LIGHT as C
    from kimi_isac.core.channel import doppler_shift_hz, range_rate_mps

    cfg = SMOKE_CFG
    frames = echo.frame_geometry(cfg)
    cloud = _cloud()
    scat = scene.scatterers_ecef(cloud, frames[0]["ue"], frames[0]["up"], frames[0]["east"])
    centroid = scat.mean(axis=0)
    fr = frames[0]
    rr_scalar = range_rate_mps(fr["sat_pos"], fr["sat_vel"], centroid, np.zeros(3))
    f_d_core = doppler_shift_hz(rr_scalar, echo.FC_HZ)
    f = echo.echo_and_features(cloud, "building", "none", seed=0, snr=1e6, cfg=cfg)["features"]
    assert abs(f[0, 3] - f_d_core / 1e3) < 1e-9


def test_ris_aligned_beats_random_on_average():
    cloud = _cloud()
    gains = []
    rng = np.random.default_rng(0)
    for t in range(8):
        a = np.abs(echo.echo_and_features(cloud, "building", "aligned", seed=t, snr=6.0, cfg=SMOKE_CFG)["Y"]).mean()
        r = np.abs(echo.echo_and_features(cloud, "building", "random", seed=t, snr=6.0, cfg=SMOKE_CFG)["Y"]).mean()
        gains.append(a / r)
    assert np.median(gains) > 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gen_echo.py -q`
Expected: FAIL (no module).

- [ ] **Step 3: Implement `scene.py`**

```python
"""ROI placement and local ENU frame for the echo scenario."""

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


def scatterers_ecef(cloud_local, ue, up, east) -> np.ndarray:
    """ROI-local meters (x east, y north, z up) -> ECEF."""
    north = np.cross(up, east)
    basis = np.stack([east, north, up], axis=1)  # (3, 3)
    anchor = roi_anchor(ue, up)
    return anchor + cloud_local @ basis.T
```

- [ ] **Step 4: Implement `echo.py`**

```python
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
    "building": 0.9, "vehicle": 0.7, "uav": 0.4, "tank": 0.8, "tower": 0.6,
    "cubesat": 0.5, "bicycle": 0.3, "pedestrian": 0.3,
    "bridge": 0.85, "windmill": 0.6,
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
    ris = np.array([
        anchor + up * scene.RIS_OFFSET_M + east * (i - (n_elem - 1) / 2.0) * elem_spacing
        for i in range(n_elem)
    ])
    return tuple(
        {**g, "ue": ue, "up": up, "east": east, "roi": anchor, "ris": ris}
        for g in frames_geo
    )


def frame_geometry(cfg: EchoConfig) -> list[dict]:
    return list(_geometry_cached(cfg.n_elem, cfg.tau, cfg.dt_s))


def _fs_amp(dist_m: np.ndarray | float) -> np.ndarray | float:
    """Free-space amplitude lambda/(4 pi d) (0 dBi antennas), vectorized."""
    return (C_LIGHT / FC_HZ) / (4.0 * np.pi * np.asarray(dist_m))


def _scatterer_paths(fr: dict, scat: np.ndarray):
    """(d1, d2, bistatic range rate) arrays; rate = d/dt(d1 + d2)."""
    sat, v, ue = fr["sat_pos"], fr["sat_vel"], fr["ue"]
    d1 = np.linalg.norm(sat - scat, axis=1)          # (N,)
    d2 = np.linalg.norm(scat - ue, axis=1)           # (N,)
    u1 = (sat - scat) / d1[:, None]                  # sat->scatterer unit vectors
    rate = np.einsum("j,nj->n", v, u1)               # scatterers are static
    return d1, d2, rate


def _elem_chain(fr: dict, scat: np.ndarray, d2: np.ndarray, elem_gain_lin: float):
    """(E, N) complex chain sat->elem->scatterer->UE (phase included)."""
    sat, ue = fr["sat_pos"], fr["ue"]
    ris = fr["ris"]                                  # (E, 3)
    d_sat_elem = np.linalg.norm(sat - ris, axis=1)   # (E,)
    d_elem_scat = np.linalg.norm(ris[:, None, :] - scat[None, :, :], axis=2)  # (E, N)
    amp = (
        math.sqrt(elem_gain_lin)
        * _fs_amp(d_sat_elem)[:, None]
        * _fs_amp(d_elem_scat)
        * _fs_amp(d2)[None, :]
    )
    phase = 2.0 * np.pi * (d_sat_elem[:, None] + d_elem_scat + d2[None, :])
    return amp * np.exp(-1j * phase / (C_LIGHT / FC_HZ))


def _aligned_phases(fr: dict, centroid: np.ndarray, d2: np.ndarray,
                    elem_gain_lin: float, rng) -> np.ndarray:
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


def echo_and_features(cloud_local, class_name, ris_mode, seed, snr, cfg):
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

    # frame-0 noiseless direct-path power as the SNR reference
    d1_0, d2_0, _ = _scatterer_paths(frames[0], scat)
    ref_power = float(np.sum((rho * _fs_amp(d1_0) * _fs_amp(d2_0)) ** 2))
    noise_std = math.sqrt(ref_power / snr)

    Y = np.empty(cfg.tau, dtype=complex)
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
                phases = _aligned_phases(fr, centroid, d2, elem_gain_lin, rng)
            chain = _elem_chain(fr, scat, d2, elem_gain_lin)  # (E, N)
            if ris_mode == "oracle":
                # each scatterer's own element chain perfectly aligned
                Y_ris = float(np.sum(np.abs(chain.sum(axis=0)) * doppler_phase))
                Y_oracle[t] = Y_direct + Y_ris
            else:
                Y_ris = float(
                    np.sum(chain @ (np.exp(1j * phases) * doppler_phase))
                )

        Y[t] = Y_direct + Y_ris
        Y[t] += noise_std * (rng.normal() + 1j * rng.normal()) / math.sqrt(2.0)

        # features (centroid quantities; index 3 is the centroid Doppler)
        rr_c = float(np.dot(fr["sat_vel"], (fr["sat_pos"] - centroid) / d_c_ue))
        f_d_c = doppler_shift_hz(rr_c, FC_HZ)
        delay = (
            float(np.linalg.norm(fr["sat_pos"] - centroid)) + d_c_ue
        ) / C_LIGHT
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
        "Y_oracle": None if np.all(np.isnan(Y_oracle)) else Y_oracle,
    }
```

Performance note: this implementation is fully vectorized over scatterers
and elements (`_elem_chain` builds an `(E, N)` matrix; 256 x 512 = 131k
complex entries per frame). No Python loops over scatterers or elements.
For the smoke config (`n_elem=32, tau=4`) the full dataset item costs
milliseconds; the full config is still precomputed once and cached under
`results/gen_cache/` (Task 5).

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_gen_echo.py -q`
Expected: PASS (with the vectorized implementation).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(gen): echo model (cloud -> features) with RIS centroid alignment"
```

---

### Task 4: Chamfer distance metrics

**Files:**
- Create: `src/kimi_isac/gen/metrics.py`
- Test: `tests/test_gen_metrics.py`

**Interfaces:**
- Consumes: nothing
- Produces: `chamfer_distance(a: np.ndarray, b: np.ndarray) -> float` (symmetric, mean-squared form; both `(N, 3)`), `summarize_cd(runs: list[float]) -> core.stats.Summary`

- [ ] **Step 1: Write the failing test**

Create `tests/test_gen_metrics.py`:

```python
import numpy as np

from kimi_isac.gen import metrics


def test_identical_clouds_zero():
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, size=(32, 3))
    assert metrics.chamfer_distance(a, a) == 0.0


def test_known_two_point_cd():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    assert abs(metrics.chamfer_distance(a, b) - 1.0) < 1e-9


def test_symmetry():
    rng = np.random.default_rng(1)
    a = rng.uniform(-1, 1, size=(16, 3))
    b = rng.uniform(-1, 1, size=(24, 3))
    assert abs(metrics.chamfer_distance(a, b) - metrics.chamfer_distance(b, a)) < 1e-9


def test_summary_has_ci():
    from kimi_isac.core import stats

    s = metrics.summarize_cd([0.1, 0.12, 0.09, 0.11])
    assert isinstance(s, stats.Summary)
    assert s.ci95_lo <= s.mean <= s.ci95_hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gen_metrics.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement metrics**

```python
"""Chamfer distance (symmetric, mean-squared form) and CD aggregation."""

from __future__ import annotations

import numpy as np

from kimi_isac.core import stats as stats_mod


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0.5 * (mean_{x in a} min_y |x-y|^2 + mean_{y in b} min_x |y-x|^2)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 3 or b.shape[1] != 3:
        raise ValueError("expected (N, 3) clouds")
    d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
    return float(0.5 * (d2.min(axis=1).mean() + d2.min(axis=0).mean()))


def summarize_cd(runs: list[float]) -> stats_mod.Summary:
    return stats_mod.summarize_runs(runs, seed=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gen_metrics.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(gen): Chamfer distance metrics with CI aggregation"
```

---

### Task 5: Datasets with fixed splits

**Files:**
- Create: `src/kimi_isac/gen/dataset.py`
- Test: `tests/test_gen_dataset.py`

**Interfaces:**
- Consumes: `gen/templates.py`, `gen/echo.py`, `core/splits.py`
- Produces:
  - `GenConfig` dataclass: `n_total=600, seed=42, snr_levels=(6.0, 1.0, 0.25), tau=8, n_elem=256`
  - `get_or_make_split(cfg, out_dir: Path) -> dict` (persisted at `out_dir/splits/gen_{n_total}_{seed}.json`)
  - `class UncondDataset(Dataset)` — item `{"cloud": (512,3) float32, "class": int64}`; index-addressed, deterministic per (seed, idx)
  - `class CondDataset(Dataset)` — item `{"features": (tau,10) float32, "cloud": (512,3), "class": int64, "snr": float32, "ris_mode": str}`
  - `class OODCondDataset(CondDataset)` — indices `range(1000, 1060)` with OOD classes
  - `class OracleCondDataset(CondDataset)` — 40 fixed samples, `ris_mode` forced to `"oracle"` (upper bound for the ablation curve)

- [ ] **Step 1: Write the failing test**

Create `tests/test_gen_dataset.py`:

```python
import numpy as np
import torch

from kimi_isac.gen import dataset as ds

SMOKE = ds.GenConfig(n_total=24, seed=42, snr_levels=(6.0,), n_elem=16, tau=4)


def test_uncond_dataset_shapes(tmp_path):
    split = ds.get_or_make_split(SMOKE, tmp_path)
    d = ds.UncondDataset(split["train"], SMOKE.seed, SMOKE)
    item = d[0]
    assert item["cloud"].shape == (512, 3)
    assert 0 <= int(item["class"]) < 8


def test_split_reused_and_disjoint(tmp_path):
    s1 = ds.get_or_make_split(SMOKE, tmp_path)
    s2 = ds.get_or_make_split(SMOKE, tmp_path)
    assert s1 == s2
    from kimi_isac.core import splits

    splits.assert_disjoint(s1)


def test_cond_dataset_features(tmp_path):
    split = ds.get_or_make_split(SMOKE, tmp_path)
    d = ds.CondDataset(split["test"], SMOKE.seed, SMOKE)
    item = d[0]
    assert item["features"].shape == (SMOKE.tau, 10)
    assert np.all(np.isfinite(item["features"].numpy()))
    assert item["ris_mode"] in ("aligned", "random", "none")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gen_dataset.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `dataset.py`**

```python
"""Datasets over fixed splits: unconditional clouds and conditional echoes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from kimi_isac.core import splits as splits_mod
from kimi_isac.gen import echo, templates

TRAIN_RIS_MODES = ("aligned", "random", "none")


@dataclass(frozen=True)
class GenConfig:
    n_total: int = 600
    seed: int = 42
    snr_levels: tuple[float, ...] = (6.0, 1.0, 0.25)
    tau: int = 8
    n_elem: int = 256

    def config_hash(self) -> str:
        import hashlib
        import json

        payload = json.dumps(
            {"n_total": self.n_total, "seed": self.seed,
             "snr_levels": list(self.snr_levels), "tau": self.tau,
             "n_elem": self.n_elem},
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def get_or_make_split(cfg: GenConfig, out_dir: Path) -> dict:
    path = Path(out_dir) / "splits" / f"gen_{cfg.n_total}_{cfg.seed}.json"
    if path.exists():
        return splits_mod.load_split(path)
    spec = splits_mod.SplitSpec(n_total=cfg.n_total, seed=cfg.seed, name="gen")
    split = splits_mod.make_split(spec)
    splits_mod.assert_disjoint(split)
    splits_mod.save_split(split, path)
    return split


def _sample_cloud(idx: int, seed: int, class_pool: tuple[str, ...]) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng((seed * 1_000_003 + idx) % (2**63))
    name = class_pool[int(rng.integers(0, len(class_pool)))]
    return templates.make_template(name, rng), templates.class_index(name)


class UncondDataset(Dataset):
    def __init__(self, indices, seed: int, cfg: GenConfig, ood: bool = False):
        self.indices = list(indices)
        self.seed = seed
        self.cfg = cfg
        self.pool = templates.OOD_CLASSES if ood else templates.TRAIN_CLASSES

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        cloud, cls = _sample_cloud(self.indices[i], self.seed, self.pool)
        return {"cloud": torch.from_numpy(cloud.astype(np.float32)),
                "class": torch.tensor(cls)}


class CondDataset(Dataset):
    def __init__(self, indices, seed: int, cfg: GenConfig, ood: bool = False):
        self.indices = list(indices)
        self.seed = seed
        self.cfg = cfg
        self.pool = templates.OOD_CLASSES if ood else templates.TRAIN_CLASSES
        self.echo_cfg = echo.EchoConfig(n_elem=cfg.n_elem, tau=cfg.tau)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        rng = np.random.default_rng((self.seed * 1_000_003 + idx) % (2**63))
        name = self.pool[int(rng.integers(0, len(self.pool)))]
        cloud = templates.make_template(name, rng)
        snr = float(self.cfg.snr_levels[int(rng.integers(0, len(self.cfg.snr_levels)))])
        ris_mode = TRAIN_RIS_MODES[int(rng.integers(0, 3))]
        out = echo.echo_and_features(cloud, name, ris_mode, seed=idx, snr=snr,
                                     cfg=self.echo_cfg)
        return {
            "features": torch.from_numpy(out["features"].astype(np.float32)),
            "cloud": torch.from_numpy(cloud.astype(np.float32)),
            "class": torch.tensor(templates.class_index(name)),
            "snr": torch.tensor(snr),
            "ris_mode": ris_mode,
        }


class OODCondDataset(CondDataset):
    def __init__(self, seed: int, cfg: GenConfig, n: int = 60):
        super().__init__(list(range(1000, 1000 + n)), seed, cfg, ood=True)


class OracleCondDataset(CondDataset):
    """Fixed 40-sample set with per-scatterer oracle alignment (upper bound)."""

    def __init__(self, seed: int, cfg: GenConfig, n: int = 40):
        super().__init__(list(range(2000, 2000 + n)), seed, cfg)

    def __getitem__(self, i):
        idx = self.indices[i]
        rng = np.random.default_rng((self.seed * 1_000_003 + idx) % (2**63))
        name = templates.TRAIN_CLASSES[int(rng.integers(0, len(templates.TRAIN_CLASSES)))]
        cloud = templates.make_template(name, rng)
        snr = float(self.cfg.snr_levels[int(rng.integers(0, len(self.cfg.snr_levels)))])
        out = echo.echo_and_features(cloud, name, "oracle", seed=idx, snr=snr,
                                     cfg=self.echo_cfg)
        return {
            "features": torch.from_numpy(out["features"].astype(np.float32)),
            "cloud": torch.from_numpy(cloud.astype(np.float32)),
            "class": torch.tensor(templates.class_index(name)),
            "snr": torch.tensor(snr),
            "ris_mode": "oracle",
        }
```

Runtime note: `CondDataset.__getitem__` runs the full echo simulation
(vectorized per Task 3); for the smoke config this is well under a second
per item. For the full config, `gen/train.py` MUST precompute the echo
features once and cache them under `results/gen_cache/` keyed by
`(config_hash, seed, idx)` as `.npz` files — implement this cache inside
`CondDataset` (check-then-save pattern) so training runs do not resimulate.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gen_dataset.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(gen): datasets over fixed splits with echo feature caching"
```

---

### Task 6: VAE and latent DiT

**Files:**
- Create: `src/kimi_isac/gen/vae.py`, `src/kimi_isac/gen/dit.py`
- Test: `tests/test_gen_models.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `vae.PointVAE(n_points=512, z_dim=256) -> nn.Module` with `encode(x) -> (mu, logvar)`, `decode(z) -> x_hat`, `forward(x) -> dict` (`x_hat`, `mu`, `logvar`, `z`); `vae_loss(out, x, kl_weight) -> Tensor`
  - `dit.CondEncoder(n_classes=10, feat_dim=10, tau=8, d_model=128, out_dim=256) -> nn.Module` — `forward(class_idx, features) -> (pooled_emb, seq_tokens)`
  - `dit.LatentDiT(z_dim=256, n_tokens=8, depth=4, hidden=256, n_heads=8, cond_dim=256) -> nn.Module` — `forward(z_noisy, t, cond_emb, cond_tokens) -> eps_pred`
  - `dit.DDPMScheduler(T=100)` with `add_noise(z, t, noise) -> Tensor`, `sample(model, cond_emb, cond_tokens, shape, device) -> Tensor`, and `betas`/`alphas_cumprod` buffers

- [ ] **Step 1: Write the failing test**

Create `tests/test_gen_models.py`:

```python
import torch

from kimi_isac.gen import dit, vae


def test_vae_roundtrip_shapes():
    m = vae.PointVAE(n_points=512, z_dim=256)
    x = torch.randn(4, 512, 3)
    out = m(x)
    assert out["x_hat"].shape == x.shape
    mu, logvar = m.encode(x)
    assert mu.shape == (4, 256)
    loss = vae.vae_loss(out, x, kl_weight=1e-4)
    assert loss.dim() == 0


def test_cond_encoder_shapes():
    enc = dit.CondEncoder(n_classes=10, feat_dim=10, tau=8)
    cls = torch.zeros(4, dtype=torch.long)
    feats = torch.randn(4, 8, 10)
    pooled, tokens = enc(cls, feats)
    assert pooled.shape == (4, 256)
    assert tokens.shape == (4, 8, 128)


def test_dit_forward_and_scheduler():
    model = dit.LatentDiT(z_dim=256, n_tokens=8)
    sched = dit.DDPMScheduler(T=100)
    enc = dit.CondEncoder(n_classes=10, feat_dim=10, tau=8)
    cls = torch.zeros(2, dtype=torch.long)
    feats = torch.randn(2, 8, 10)
    pooled, tokens = enc(cls, feats)
    z = torch.randn(2, 256)
    t = torch.tensor([10, 50])
    noisy = sched.add_noise(z, t, torch.randn_like(z))
    assert noisy.shape == z.shape
    eps = model(noisy, t, pooled, tokens)
    assert eps.shape == z.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gen_models.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `vae.py`**

```python
"""Point-cloud VAE: 512x3 -> z(256)."""

from __future__ import annotations

import torch
from torch import nn


class PointVAE(nn.Module):
    def __init__(self, n_points: int = 512, z_dim: int = 256):
        super().__init__()
        self.n_points = n_points
        self.z_dim = z_dim
        in_dim = n_points * 3
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, in_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x.reshape(x.shape[0], -1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).reshape(-1, self.n_points, 3)

    def forward(self, x: torch.Tensor) -> dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return {"x_hat": self.decode(z), "mu": mu, "logvar": logvar, "z": z}


def vae_loss(out: dict, x: torch.Tensor, kl_weight: float = 1e-4) -> torch.Tensor:
    recon = ((out["x_hat"] - x) ** 2).sum(dim=-1).mean()
    mu, logvar = out["mu"], out["logvar"]
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return recon + kl_weight * kl
```

- [ ] **Step 4: Implement `dit.py`**

```python
"""Latent DiT denoiser with cross-attention conditioning + linear-beta DDPM."""

from __future__ import annotations

import math

import torch
from torch import nn


class CondEncoder(nn.Module):
    """Class embedding + feature-sequence Transformer -> pooled conditioning."""

    def __init__(self, n_classes: int = 10, feat_dim: int = 10, tau: int = 8,
                 d_model: int = 128, n_heads: int = 4, out_dim: int = 256):
        super().__init__()
        self.class_emb = nn.Embedding(n_classes, d_model)
        self.feat_proj = nn.Linear(feat_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.pool = nn.Linear(d_model, out_dim)

    def forward(self, class_idx: torch.Tensor, features: torch.Tensor):
        tokens = self.feat_proj(features) + self.class_emb(class_idx).unsqueeze(1)
        tokens = self.transformer(tokens)
        return self.pool(tokens.mean(dim=1)), tokens


class DiTBlock(nn.Module):
    """AdaLN-zero self-attention + cross-attention to condition tokens."""

    def __init__(self, hidden: int, n_heads: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(hidden, n_heads, batch_first=True,
                                           kdim=cond_dim, vdim=cond_dim)
        self.norm3 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden)
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * hidden))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, cond, cond_tokens):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + gate1.unsqueeze(1) * self.attn(h, h, h)[0]
        h2 = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + self.cross(h2, cond_tokens, cond_tokens)[0]
        x = x + self.mlp(self.norm3(x))
        return x


class LatentDiT(nn.Module):
    def __init__(self, z_dim: int = 256, n_tokens: int = 8, depth: int = 4,
                 hidden: int = 256, n_heads: int = 8, cond_dim: int = 256):
        super().__init__()
        self.in_proj = nn.Linear(z_dim // n_tokens, hidden)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, hidden) * 0.02)
        self.blocks = nn.ModuleList(
            [DiTBlock(hidden, n_heads, cond_dim) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, z_dim // n_tokens)
        self.n_tokens = n_tokens

    def forward(self, z_noisy, t, cond_emb, cond_tokens):
        h = z_noisy.reshape(z_noisy.shape[0], self.n_tokens, -1)
        h = self.in_proj(h) + self.pos
        for blk in self.blocks:
            h = blk(h, cond_emb, cond_tokens)
        return self.out_proj(self.out_norm(h)).reshape(z_noisy.shape[0], -1)


class DDPMScheduler:
    """Linear-beta DDPM, T = 100, deterministic ancestral sampling."""

    def __init__(self, T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)

    def add_noise(self, z: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        ac = self.alphas_cumprod.to(z.device)[t].view(-1, 1)
        return ac.sqrt() * z + (1 - ac).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, cond_emb, cond_tokens, shape, device="cpu"):
        """Ancestral sampling: z_T -> z_0 through the learned eps network."""
        z = torch.randn(shape, device=device)
        for step in reversed(range(self.T)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            eps = model(z, t, cond_emb, cond_tokens)
            beta = self.betas.to(device)[step]
            alpha = self.alphas_cumprod.to(device)[step]
            mean = (z - beta / (1 - alpha).sqrt() * eps) / (1 - beta).sqrt()
            if step > 0:
                z = mean + beta.sqrt() * torch.randn_like(z)
            else:
                z = mean
        return z
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_gen_models.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(gen): PointVAE and latent DiT with cross-attention conditioning"
```

---

### Task 7: Two-stage training CLI

**Files:**
- Create: `src/kimi_isac/gen/train.py`
- Modify: `src/kimi_isac/__main__.py` (add `gen` subcommand: `verify` pattern — lazy import `gen.train.main`)
- Test: manual smoke run + `tests/test_gen_train_smoke.py` (in-process 1-epoch)

**Interfaces:**
- Consumes: `training/engine.py`, `gen/vae.py`, `gen/dit.py`, `gen/dataset.py`, `gen/metrics.py`
- Produces:
  - `main(argv=None) -> int` with `--smoke`, `--seed`, `--epochs-vae`, `--epochs-dit`, `--out`, `--device` (auto/cpu/cuda)
  - `run_gen(seed, cfg, epochs_vae, epochs_dit, device, out_dir, log) -> dict` with keys:
    `vae_recon_cd: float` (held-out instances), `uncond_cd_train: float`,
    `uncond_cd_heldout: float` (held-out instances, class-conditional),
    `uncond_cd_ood: float` (the 2 OOD classes), `cond_cd: float`
    (val split, `ris_mode == "aligned"` samples only),
    `cond_cd_oracle: float` (`OracleCondDataset`), and
    `cond_cd_cells: list[((snr, ris_mode), cd)]` (per-sample conditional CDs
    for the report's curve aggregation)

- [ ] **Step 1: Write the in-process smoke test**

Create `tests/test_gen_train_smoke.py`:

```python
import logging
from pathlib import Path

from kimi_isac.gen import dataset as ds, train as gen_train


def test_smoke_training_runs(tmp_path):
    cfg = ds.GenConfig(n_total=24, seed=0, snr_levels=(6.0,), tau=4, n_elem=16)
    metrics = gen_train.run_gen(
        seed=0, cfg=cfg, epochs_vae=1, epochs_dit=1, device="cpu",
        out_dir=tmp_path, log=logging.getLogger("test"),
    )
    assert set(metrics) >= {"vae_recon_cd", "uncond_cd_heldout", "cond_cd"}
    for v in metrics.values():
        assert v == v  # not NaN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gen_train_smoke.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `train.py`**

```python
"""Two-stage generative training: VAE then latent DiT, on the shared engine."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from kimi_isac.core import logging_setup
from kimi_isac.core.rng import seed_all
from kimi_isac.gen import dataset as ds
from kimi_isac.gen import dit, metrics, vae
from kimi_isac.training.engine import EngineHooks, run_training


def _loader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _recon_cd(model, loader, device) -> float:
    model.eval()
    cds = []
    for batch in loader:
        out = model(batch["cloud"].to(device))
        for a, b in zip(out["x_hat"].cpu().numpy(), batch["cloud"].numpy()):
            cds.append(metrics.chamfer_distance(a, b))
    return float(np.mean(cds))


def _train_vae(split, cfg, device, out_dir, log, epochs, batch_size, lr, patience):
    train_loader = _loader(ds.UncondDataset(split["train"], cfg.seed, cfg), batch_size, True)
    val_loader = _loader(ds.UncondDataset(split["val"], cfg.seed, cfg), batch_size, False)

    def train_epoch(model, opt, _):
        model.train()
        tot = 0.0
        for batch in train_loader:
            x = batch["cloud"].to(device)
            opt.zero_grad()
            loss = vae.vae_loss(model(x), x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item())
        return tot / max(1, len(train_loader))

    hooks = EngineHooks(
        build_model=vae.PointVAE,
        make_optimizer=lambda m: (
            torch.optim.Adam(m.parameters(), lr=lr),
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                torch.optim.Adam(m.parameters(), lr=lr), patience=2
            ),
        ),
        train_epoch=train_epoch,
        val_loss=lambda m, _: _vae_val(m, val_loader, device),
        config={"stage": "vae", "seed": cfg.seed, "hash": cfg.config_hash()},
        out_dir=Path(out_dir) / "vae",
        device=device,
    )
    return run_training(hooks, epochs=epochs, patience=patience, log=log)


def _vae_val(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["cloud"].to(device)
            tot += float(vae.vae_loss(model(x), x).item())
            n += 1
    return tot / max(1, n)
```

IMPORTANT for Step 3: fix `make_optimizer` to build the optimizer ONCE:

```python
        make_optimizer=lambda m: (
            (opt := torch.optim.Adam(m.parameters(), lr=lr)),
            torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2),
        ),
```

Then `_train_dit` follows the same pattern (DDPM eps-MSE loss on latents from
the frozen VAE), and `run_gen` orchestrates: train VAE → compute
`vae_recon_cd` on a held-out-instance loader → freeze VAE → train DiT →
sample `uncond_cd_heldout` (class-conditional, held-out instances) →
sample `cond_cd` (conditional on val split features). Unconditional sampling
uses class embeddings only (`CondEncoder` with a zero feature tensor);
conditional sampling uses dataset features. Report `cond_cd` restricted to
`ris_mode == "aligned"` samples for comparability with the oracle curve in
`gen/report.py`.

`__main__.py`: add subcommand

```python
    sub.add_parser("gen", help="generative 3D reconstruction (train/report)")
```

dispatch: `if args.command == "gen": from kimi_isac.gen import train as gen_train; return gen_train.main()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gen_train_smoke.py -q`
Expected: PASS (under 60 s on CPU).

- [ ] **Step 5: Run CLI smoke**

Run: `python -m kimi_isac.gen.train --smoke`
Expected: completes, prints CD metrics.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(gen): two-stage training CLI with smoke mode"
```

---

### Task 8: Multi-seed report with CD curves

**Files:**
- Create: `src/kimi_isac/gen/report.py`
- Test: manual smoke run

**Interfaces:**
- Consumes: `gen/train.py:run_gen`, `gen/dataset.py`, `gen/metrics.py`, `core/stats.py`
- Produces:
  - `main(argv=None) -> int` with `--seeds`, `--smoke`, `--epochs-vae`, `--epochs-dit`, `--out`, `--device`
  - writes `results/gen_report.json`: `uncond_cd_heldout_inst`, `uncond_cd_heldout_class`, `memorization_gap`, `cond_cd` per (snr, ris_mode) cell, `cond_cd_oracle`, `vae_recon_cd` — each `{mean, std, ci95: [lo, hi], n}`

- [ ] **Step 1: Implement `report.py`** (full implementation in the code block below)

Module skeleton (imports + CLI) is fixed by the full `run_report` body that
follows the IMPORTANT note in this step: `report.py` contains exactly
`import argparse/json/logging/Path/numpy/torch`, the `kimi_isac` imports
shown there, the full `run_report` function, and a `main(argv=None)` that
parses `--seeds --smoke --epochs-vae --epochs-dit --out --device` (smoke
sets `seeds=2, epochs*=1, n_total=24, n_elem=16, tau=4`), calls
`run_report`, and returns 0.

```python
"""Multi-seed generative report: unconditional CD, conditional CD curves."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from kimi_isac.core import logging_setup, stats as stats_mod
from kimi_isac.core.rng import seed_all
from kimi_isac.gen import dataset as ds
from kimi_isac.gen import train as gen_train
```

IMPORTANT for Step 1: `gen_train.run_gen` must return the full key set from
Task 7's interface (`vae_recon_cd`, `uncond_cd_train`, `uncond_cd_heldout`,
`uncond_cd_ood`, `cond_cd`, `cond_cd_cells`) — Task 7's implementation
already provides them; the report consumes exactly those keys.
`memorization_gap` is computed per run as
`uncond_cd_heldout - uncond_cd_train` and then aggregated across seeds
with `stats_mod.summarize_runs` (each seed contributes one gap value).

The report CLI: `--smoke` sets `seeds=2, epochs*=1, n_total=24, n_elem=16,
tau=4`.

Full `run_report` body:

```python
def run_report(seeds: int, epochs_vae: int, epochs_dit: int, out: Path,
               device: str, log: logging.Logger, cfg: ds.GenConfig | None = None
               ) -> dict:
    cfg = cfg or ds.GenConfig()
    runs = []
    for i in range(seeds):
        seed = 1000 + i
        seed_all(seed)
        log.info("=== gen seed %d (%d/%d) ===", seed, i + 1, seeds)
        runs.append(gen_train.run_gen(
            seed=seed, cfg=cfg, epochs_vae=epochs_vae, epochs_dit=epochs_dit,
            device=device, out_dir=out / "gen" / f"seed_{seed}", log=log,
        ))

    def agg(key: str) -> dict:
        s = stats_mod.summarize_runs([r[key] for r in runs], seed=0)
        return {"mean": s.mean, "std": s.std, "ci95": [s.ci95_lo, s.ci95_hi],
                "n": s.n}

    gaps = [r["uncond_cd_heldout"] - r["uncond_cd_train"] for r in runs]
    gap = stats_mod.summarize_runs(gaps, seed=0)

    cells: dict[str, list[float]] = {}
    for run in runs:
        for (snr, ris), cd in run["cond_cd_cells"]:
            cells.setdefault(f"snr{snr:g}_{ris}", []).append(cd)
    for key, values in cells.items():
        assert len(values) == seeds, f"cell {key} has n={len(values)} != {seeds}"
    cond = {
        k: {"mean": stats_mod.summarize_runs(v, seed=0).mean,
            "ci95": [stats_mod.summarize_runs(v, seed=0).ci95_lo,
                     stats_mod.summarize_runs(v, seed=0).ci95_hi],
            "n": len(v)}
        for k, v in sorted(cells.items())
    }

    report = {
        "n_seeds": seeds,
        "config_hash": cfg.config_hash(),
        "vae_recon_cd": agg("vae_recon_cd"),
        "uncond_cd_heldout_inst": agg("uncond_cd_heldout"),
        "uncond_cd_heldout_class": agg("uncond_cd_ood"),
        "memorization_gap": {"mean": gap.mean, "std": gap.std,
                             "ci95": [gap.ci95_lo, gap.ci95_hi], "n": gap.n},
        "cond_cd": cond,
        "cond_cd_oracle": agg("cond_cd_oracle"),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "gen_report.json").write_text(json.dumps(report, indent=1),
                                         encoding="utf-8")
    return report
```

- [ ] **Step 2: Run smoke**

Run: `python -m kimi_isac.gen.report --smoke`
Expected: completes on CPU in under 60 s; `results/gen_report.json` written with
all keys present.

- [ ] **Step 3: Run the full report on GPU**

Run: `python -m kimi_isac.gen.report --seeds 10 --device cuda`
Expected: completes in ~20-40 min on the RTX 5060; every metric has a
finite CI.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(gen): multi-seed CD report with SNR x RIS ablation curves"
```

---

### Task 9: CI wiring and README section

**Files:**
- Modify: `.github/workflows/ci.yml` (add gen smoke steps)
- Modify: `README.md` (v2 section with the measured numbers)

**Interfaces:**
- Consumes: everything above
- Produces: CI runs `python -m kimi_isac.gen.train --smoke` and
  `python -m kimi_isac.gen.report --smoke`; README v2 section reports the
  measured CIs (fill in from `results/gen_report.json`)

- [ ] **Step 1: Extend CI**

Add after the ML smoke step:

```yaml
      - name: Generative pipeline smoke (VAE + DiT, 1 epoch)
        run: python -m kimi_isac.gen.train --smoke
      - name: Generative report smoke (2 seeds)
        run: python -m kimi_isac.gen.report --smoke
```

- [ ] **Step 2: Add the README v2 section**

Append a `## v2: conditional diffusion 3D reconstruction` section with:
architecture one-liner, the measured table (uncond held-out CD with CI,
held-out-class CD, memorization gap, cond CD(SNR, RIS) curves, oracle),
and the cross-range honesty statement from the spec.

- [ ] **Step 3: Full acceptance**

Run: `ruff check . && ruff format --check . && mypy src && python -m pytest -q && python -m kimi_isac.verify`
Expected: all green; verify exit 0.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "ci+docs: gen smoke in CI; README v2 section with CD results"
```

---

## Review Focus

1. **Echo vectorization correctness** (Task 3): the Python-loop reference
   implementation in the test (`test_delay_feature_matches_bistatic_geometry`)
   pins the delay feature against a hand-computed bistatic path; if the
   vectorized rewrite changes the sum order/precision beyond 1e-6 relative,
   the test fails — that is the guard.
2. **RIS alignment not actually helping** (Task 3): MC test
   `test_ris_aligned_beats_random_on_average` fails if the optimizer
   silently degrades (e.g. wrong phase convention) — median gain must
   exceed 1.0 over 8 seeds.
3. **Condition leakage of absolute scale** (Task 2/3): features are
   scale-free by construction (dB, normalized), but if a future edit adds a
   raw-amplitude feature the delay/Doppler tests still pass while
   conditioning silently changes — pinned by `test_feature_shape_and_finiteness`
   plus the spec's feature list being frozen at 10 dims (any new feature
   requires a spec change).
4. **Engine extraction regression** (Task 1): the full v1 suite runs before
   and after; `test_engine_selects_best_val_and_loads_it` pins best-epoch
   selection semantics.
5. **Report cells with n < 10 seeds** (Task 8): `_cond_cd_by_cell` must
   assert every reported cell has `n == seeds`; a cell that silently drops
   samples is a reported-CI failure — pinned in the report smoke by
   checking all cells present in `results/gen_report.json`.
