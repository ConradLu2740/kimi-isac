"""Fixed train/val/test index splits, persisted as JSON artifacts.

Splits are generated once from a master seed + config hash and saved to
disk; training and evaluation load the SAME artifact, so evaluation never
depends on re-drawing the generator.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kimi_isac.core.rng import seed_all


@dataclass(frozen=True)
class SplitSpec:
    """Requested split geometry."""

    n_total: int
    train_frac: float = 0.7
    val_frac: float = 0.15
    seed: int = 42
    name: str = "default"

    def config_hash(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


def make_split(spec: SplitSpec) -> dict:
    """Sample disjoint train/val/test index sets; deterministic in the seed."""
    if not 0.0 < spec.train_frac < 1.0 or spec.val_frac <= 0.0:
        raise ValueError("bad fractions")
    if spec.train_frac + spec.val_frac >= 1.0:
        raise ValueError("train + val must leave room for test")
    seed_all(spec.seed)
    perm = np.random.default_rng(spec.seed).permutation(spec.n_total)
    n_train = int(round(spec.train_frac * spec.n_total))
    n_val = int(round(spec.val_frac * spec.n_total))
    train = sorted(int(i) for i in perm[:n_train])
    val = sorted(int(i) for i in perm[n_train : n_train + n_val])
    test = sorted(int(i) for i in perm[n_train + n_val :])
    return {
        "name": spec.name,
        "seed": spec.seed,
        "config_hash": spec.config_hash(),
        "n_total": spec.n_total,
        "train": train,
        "val": val,
        "test": test,
    }


def assert_disjoint(split: dict) -> None:
    """Fail loudly if the three index sets overlap or cover indices wrongly."""
    train, val, test = (set(split[k]) for k in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test), "overlapping splits"
    assert len(train) + len(val) + len(test) == split["n_total"], "split does not cover n_total"
    for k in ("train", "val", "test"):
        assert all(0 <= i < split["n_total"] for i in split[k]), "index out of range"


def save_split(split: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split, indent=1), encoding="utf-8")
    return path


def load_split(path: str | Path) -> dict:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_disjoint(split)
    return split
