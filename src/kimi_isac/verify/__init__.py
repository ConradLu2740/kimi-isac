"""kimi_isac.verify: physics verification against external ground truth."""

from __future__ import annotations


def run_all() -> bool:
    """Run all verification checks. Returns True iff every check passes.

    Checks are grouped so CI can report per-group:
    - orbit: SGP4 propagation of a fixed ISS TLE vs published values
    - doppler: sign convention f_D = -range_rate / wavelength
    - link_budget: Friis formula vs textbook reference values
    - waveform: OFDM range resolution = c / (2B), energy conservation

    Self-consistency (regression) checks live in tests/, never here.
    """
    from kimi_isac.verify import orbit, doppler, link_budget, waveform

    groups = [
        ("orbit", orbit.run),
        ("doppler", doppler.run),
        ("link_budget", link_budget.run),
        ("waveform", waveform.run),
    ]
    all_ok = True
    for name, fn in groups:
        ok = fn()
        all_ok = all_ok and ok
        print(f"[{name}] {'PASS' if ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run_all() else 1)
