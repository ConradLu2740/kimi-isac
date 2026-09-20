"""Tests for the shared training engine."""

import logging

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
    hooks = EngineHooks(
        build_model=_Tiny,
        make_optimizer=lambda m: (torch.optim.SGD(m.parameters(), lr=0.1), None),
        train_epoch=lambda m, o, _loader: 0.0,
        val_loss=lambda m, _loader: 0.5,
        config={"tiny": True},
        out_dir=tmp_path,
    )
    result = run_training(hooks, epochs=3, patience=2, log=logging.getLogger("test"))
    assert result["epochs_run"] == 3
    assert result["best_epoch"] == 0
    assert (tmp_path / "best.pth").exists()
