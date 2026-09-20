"""Datasets over fixed splits: unconditional clouds and conditional echoes.

Conditional samples run the full echo simulation on first access and are
cached as .npz under ``cache_dir`` (keyed by config hash, seed, index) so
repeated training runs never resimulate.
"""

from __future__ import annotations

import hashlib
import json
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
        payload = json.dumps(
            {
                "n_total": self.n_total,
                "seed": self.seed,
                "snr_levels": list(self.snr_levels),
                "tau": self.tau,
                "n_elem": self.n_elem,
            },
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


def _draw_sample(seed: int, idx: int, pool: tuple[str, ...]) -> tuple[str, np.ndarray]:
    rng = np.random.default_rng((seed * 1_000_003 + idx) % (2**63))
    name = pool[int(rng.integers(0, len(pool)))]
    return name, templates.make_template(name, rng)


class UncondDataset(Dataset):
    """(cloud, class) samples addressed by split index."""

    def __init__(self, indices, seed: int, cfg: GenConfig, ood: bool = False):
        self.indices = list(indices)
        self.seed = seed
        self.cfg = cfg
        self.pool = templates.OOD_CLASSES if ood else templates.TRAIN_CLASSES

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        name, cloud = _draw_sample(self.seed, self.indices[i], self.pool)
        return {
            "cloud": torch.from_numpy(cloud.astype(np.float32)),
            "class": torch.tensor(templates.class_index(name)),
        }


class CondDataset(Dataset):
    """(features, cloud, class, snr, ris_mode) samples; echo cached on disk."""

    def __init__(
        self,
        indices,
        seed: int,
        cfg: GenConfig,
        ood: bool = False,
        cache_dir: str | Path | None = None,
    ):
        self.indices = list(indices)
        self.seed = seed
        self.cfg = cfg
        self.pool = templates.OOD_CLASSES if ood else templates.TRAIN_CLASSES
        self.echo_cfg = echo.EchoConfig(n_elem=cfg.n_elem, tau=cfg.tau)
        self.cache_dir = Path(cache_dir) / cfg.config_hash() if cache_dir is not None else None

    def __len__(self) -> int:
        return len(self.indices)

    def _cache_path(self, idx: int) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"s{self.seed}_{idx}.npz"

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        s = (self.seed * 1_000_003 + idx) % (2**63)
        # decorrelated stream: the class draw inside _draw_sample uses default_rng(s);
        # reusing the same seed here would alias class and SNR through the same
        # underlying random word (observed as a spurious class<->SNR coupling).
        rng = np.random.default_rng(s ^ 0x9E3779B9)
        path = self._cache_path(idx)
        if path is not None and path.exists():
            data = np.load(path)
            features = data["features"]
            cloud = data["cloud"]
            cls = int(data["cls"])
            snr = float(data["snr"])
            ris_mode = str(data["ris_mode"])
        else:
            name, cloud = _draw_sample(self.seed, idx, self.pool)
            snr = float(self.cfg.snr_levels[int(rng.integers(0, len(self.cfg.snr_levels)))])
            ris_mode = TRAIN_RIS_MODES[int(rng.integers(0, len(TRAIN_RIS_MODES)))]
            out = echo.echo_and_features(
                cloud, name, ris_mode, seed=idx, snr=snr, cfg=self.echo_cfg
            )
            features = out["features"].astype(np.float32)
            cls = templates.class_index(name)
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    path,
                    features=features,
                    cloud=cloud.astype(np.float32),
                    cls=int(cls),
                    snr=snr,
                    ris_mode=ris_mode,
                )
        return {
            "features": torch.from_numpy(np.asarray(features, dtype=np.float32)),
            "cloud": torch.from_numpy(np.asarray(cloud, dtype=np.float32)),
            "class": torch.tensor(int(cls)),
            "snr": torch.tensor(float(snr)),
            "ris_mode": str(ris_mode),
        }


class OODCondDataset(CondDataset):
    """Held-out classes (bridge, windmill) for the conditional path."""

    def __init__(self, seed: int, cfg: GenConfig, n: int = 60, cache_dir: str | Path | None = None):
        super().__init__(list(range(1000, 1000 + n)), seed, cfg, ood=True, cache_dir=cache_dir)


class OracleCondDataset(CondDataset):
    """Fixed 40-sample set with per-scatterer oracle alignment (upper bound)."""

    def __init__(self, seed: int, cfg: GenConfig, n: int = 40, cache_dir: str | Path | None = None):
        super().__init__(list(range(2000, 2000 + n)), seed, cfg, cache_dir=cache_dir)

    def __getitem__(self, i: int) -> dict:
        idx = self.indices[i]
        s = (self.seed * 1_000_003 + idx) % (2**63)
        # decorrelated stream: the class draw inside _draw_sample uses default_rng(s);
        # reusing the same seed here would alias class and SNR through the same
        # underlying random word (observed as a spurious class<->SNR coupling).
        rng = np.random.default_rng(s ^ 0x9E3779B9)
        path = self._cache_path(idx)
        if path is not None and path.exists():
            data = np.load(path)
            return {
                "features": torch.from_numpy(np.asarray(data["features"], dtype=np.float32)),
                "cloud": torch.from_numpy(np.asarray(data["cloud"], dtype=np.float32)),
                "class": torch.tensor(int(data["cls"])),
                "snr": torch.tensor(float(data["snr"])),
                "ris_mode": "oracle",
            }
        name, cloud = _draw_sample(self.seed, idx, templates.TRAIN_CLASSES)
        snr = float(self.cfg.snr_levels[int(rng.integers(0, len(self.cfg.snr_levels)))])
        out = echo.echo_and_features(cloud, name, "oracle", seed=idx, snr=snr, cfg=self.echo_cfg)
        features = out["features"].astype(np.float32)
        cls = templates.class_index(name)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                features=features,
                cloud=cloud.astype(np.float32),
                cls=int(cls),
                snr=snr,
                ris_mode="oracle",
            )
        return {
            "features": torch.from_numpy(features),
            "cloud": torch.from_numpy(cloud.astype(np.float32)),
            "class": torch.tensor(cls),
            "snr": torch.tensor(snr),
            "ris_mode": "oracle",
        }


CELL_EVAL_BASE = 4000


def make_cell_eval_items(
    cfg: GenConfig, seed: int, per_class: int = 2, cache_dir: str | Path | None = None
) -> list[dict]:
    """Stratified evaluation set: every (snr, ris_mode, class) combo appears
    ``per_class`` times, so CD cells isolate the SNR/RIS effect from class mix."""
    items: list[dict] = []
    counter = 0
    echo_cfg = echo.EchoConfig(n_elem=cfg.n_elem, tau=cfg.tau)
    cache = Path(cache_dir) / cfg.config_hash() / "cells" if cache_dir is not None else None
    for snr in cfg.snr_levels:
        for mode in TRAIN_RIS_MODES:
            for cls_name in templates.TRAIN_CLASSES:
                for _ in range(per_class):
                    idx = CELL_EVAL_BASE + counter
                    counter += 1
                    path = cache / f"s{seed}_{idx}.npz" if cache is not None else None
                    if path is not None and path.exists():
                        data = np.load(path)
                        items.append(
                            {
                                "features": torch.from_numpy(
                                    np.asarray(data["features"], dtype=np.float32)
                                ),
                                "cloud": torch.from_numpy(
                                    np.asarray(data["cloud"], dtype=np.float32)
                                ),
                                "class": torch.tensor(int(data["cls"])),
                                "snr": torch.tensor(float(data["snr"])),
                                "ris_mode": str(data["ris_mode"]),
                            }
                        )
                        continue
                    rng = np.random.default_rng((seed * 1_000_003 + idx) % (2**63))
                    cloud = templates.make_template(cls_name, rng)
                    out = echo.echo_and_features(
                        cloud, cls_name, mode, seed=idx, snr=float(snr), cfg=echo_cfg
                    )
                    features = out["features"].astype(np.float32)
                    cls = templates.class_index(cls_name)
                    if path is not None:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        np.savez(
                            path,
                            features=features,
                            cloud=cloud.astype(np.float32),
                            cls=int(cls),
                            snr=float(snr),
                            ris_mode=mode,
                        )
                    items.append(
                        {
                            "features": torch.from_numpy(features),
                            "cloud": torch.from_numpy(cloud.astype(np.float32)),
                            "class": torch.tensor(cls),
                            "snr": torch.tensor(float(snr)),
                            "ris_mode": mode,
                        }
                    )
    return items


UNCOND_EVAL_BASE = 5000


def make_uncond_eval_items(
    cfg: GenConfig, seed: int, ood: bool = False, per_class: int = 2
) -> list[dict]:
    """Stratified unconditional evaluation references (cloud, class)."""
    pool = templates.OOD_CLASSES if ood else templates.TRAIN_CLASSES
    items: list[dict] = []
    counter = 0
    for cls_name in pool:
        for _ in range(per_class):
            idx = UNCOND_EVAL_BASE + counter
            counter += 1
            rng = np.random.default_rng((seed * 1_000_003 + idx) % (2**63))
            cloud = templates.make_template(cls_name, rng)
            items.append(
                {
                    "cloud": torch.from_numpy(cloud.astype(np.float32)),
                    "class": torch.tensor(templates.class_index(cls_name)),
                }
            )
    return items
