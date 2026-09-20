"""Deterministic seeding helpers.

Call ``seed_all`` once at process start so numpy/builtin random/torch
draw the same streams across runs. PYTHONHASHSEED is intentionally not
touched at runtime (it only affects interpreter start); avoid relying on
hash ordering for reproducibility.
"""

from __future__ import annotations

import random

import numpy as np


def seed_all(seed: int) -> None:
    """Seed python ``random``, numpy, and torch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
