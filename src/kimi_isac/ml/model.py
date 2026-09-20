"""Small multi-head sensing network (torch)."""

from __future__ import annotations

import torch
from torch import nn


class SensingNet(nn.Module):
    """Multi-head CNN over the pooled range-Doppler map.

    Heads: detection logit, range (normalized), velocity (normalized),
    class logits.
    """

    def __init__(self, n_classes: int = 3, n_doppler: int = 16, n_range: int = 49) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.head_detect = nn.Linear(32 * (n_range // 4) * (n_doppler // 4), 1)
        self.head_range = nn.Linear(32 * (n_range // 4) * (n_doppler // 4), 1)
        self.head_vel = nn.Linear(32 * (n_range // 4) * (n_doppler // 4), 1)
        self.head_class = nn.Linear(32 * (n_range // 4) * (n_doppler // 4), n_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        z = self.features(x).flatten(1)
        return {
            "detect": self.head_detect(z).squeeze(-1),
            "range": self.head_range(z).squeeze(-1),
            "vel": self.head_vel(z).squeeze(-1),
            "class": self.head_class(z),
        }
