"""Shared CLI argument helpers."""

from __future__ import annotations

import argparse


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the seed/device/out flags every entry point must expose."""
    parser.add_argument("--seed", type=int, default=42, help="master seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="compute device for torch-backed paths (default: auto)",
    )
    parser.add_argument("--out", type=str, default="results", help="output directory")
    parser.add_argument("--log-level", type=str, default="INFO", help="logging level")
