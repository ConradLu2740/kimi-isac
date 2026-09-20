"""CLI entry: ``python -m kimi_isac <subcommand>``."""

from __future__ import annotations

import argparse
import logging
import sys

from kimi_isac.core import logging_setup
from kimi_isac.core.rng import seed_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kimi-isac", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="external-truth physics verification suite")
    sub.add_parser("closedloop", help="sensing-communication closed-loop demo")
    sub.add_parser("gen", help="generative 3D reconstruction (VAE + latent DiT)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging_setup.setup_logging(logging.INFO)
    log = logging.getLogger("kimi_isac")

    # Subcommands grow in later phases; dispatch is explicit so imports stay lazy.
    if args.command == "verify":
        from kimi_isac.verify import run_all

        seed_all(0)
        ok = run_all()
        log.info("verify: %s", "ALL PASS" if ok else "FAILURES PRESENT")
        return 0 if ok else 1
    if args.command == "closedloop":
        from kimi_isac.closedloop import main as closedloop_main

        return closedloop_main()
    if args.command == "gen":
        from kimi_isac.gen import train as gen_train

        return gen_train.main()
    log.error("unknown command %s", args.command)
    return 2


if __name__ == "__main__":
    sys.exit(main())
