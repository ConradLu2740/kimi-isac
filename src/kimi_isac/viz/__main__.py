"""Generate all demo assets: python -m kimi_isac.viz"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from kimi_isac.core import logging_setup
from kimi_isac.viz import closedloop_demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, default="docs/demos")
    parser.add_argument(
        "--skip-gen",
        action="store_true",
        help="skip the reconstruction viewer (needs a trained checkpoint)",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="results/gen/seed_1001")
    args = parser.parse_args(argv)

    logging_setup.setup_logging()
    log = logging.getLogger("kimi_isac.viz")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    closedloop_demo.main(out_dir=str(out))

    if not args.skip_gen:
        ckpt = Path(args.checkpoint_dir)
        if (ckpt / "dit" / "best.pth").exists():
            from kimi_isac.viz import gen_demo

            gen_demo.main(["--checkpoint-dir", str(ckpt), "--out", str(out)])
        else:
            log.warning("no checkpoint at %s; skipping reconstruction viewer", ckpt)
    log.info("demo assets in %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
