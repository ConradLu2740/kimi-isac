"""Logging setup: levels, timestamps, optional file sink into results/."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FMT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure root logging. Repeated calls replace handlers, never duplicate."""
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    stream_handler: logging.Handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(logging.Formatter(_FMT))
    root.addHandler(stream_handler)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(_FMT))
        root.addHandler(file_handler)
    root.setLevel(level)
