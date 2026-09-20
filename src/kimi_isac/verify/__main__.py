import sys

from kimi_isac.verify import run_all

sys.exit(0 if run_all() else 1)
