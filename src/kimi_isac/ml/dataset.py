"""Dataset over fixed split indices."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from kimi_isac.ml.scenario import ScenarioConfig, make_sample


class SensingDataset(Dataset):
    """Samples addressed by split index; generated deterministically per index."""

    def __init__(self, indices: list[int], seed: int, cfg: ScenarioConfig, ood: bool = False):
        self.indices = list(indices)
        self.seed = seed
        self.cfg = cfg
        self.ood = ood

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        s = make_sample(self.indices[i], self.seed, self.cfg, ood=self.ood)
        return {
            "map": torch.from_numpy(s["map"]),
            "map_full": torch.from_numpy(s["map_full"]),
            "has_target": torch.tensor(s["has_target"]),
            "range_m": torch.tensor(s["range_m"]),
            "vel_mps": torch.tensor(s["vel_mps"]),
            "class": torch.tensor(s["class"]),
            "snr": torch.tensor(s["snr"]),
        }


def stack_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
