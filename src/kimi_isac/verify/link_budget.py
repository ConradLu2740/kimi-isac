"""Link-budget verification against textbook references."""

from __future__ import annotations

from kimi_isac.core import link_budget as lb

TOL_DB = 0.05


def run() -> bool:
    # Textbook free-space path loss figures:
    # FSPL(1 GHz, 1 km) = 20 log10(4 pi d / lambda) = 92.45 dB
    # FSPL(2.4 GHz, 1 km) = 100.05 dB (rule of thumb: ~100 dB)
    fspl_1g = lb.free_space_path_loss_db(1.0e9, 1.0e3)
    fspl_24g = lb.free_space_path_loss_db(2.4e9, 1.0e3)

    # kTB: -203.98 dBW in 1 Hz at 290 K; -174 dBm/Hz convention
    noise_1hz = lb.thermal_noise_dbw(1.0)
    noise_20mhz = lb.thermal_noise_dbw(20.0e6)  # -174 dBm/Hz + 73.01 dB = -100.97 dBm

    # Friis identity: Pr = Pt + Gt + Gr - FSPL
    pr = lb.friis_received_dbw(30.0, 2.0, 2.0, 2.4e9, 1.0e3)
    pr_manual = 30.0 + 2.0 + 2.0 - fspl_24g

    snr = lb.snr_db(30.0, 2.0, 2.0, 2.4e9, 1.0e3, 20.0e6, noise_figure_db=5.0)
    snr_manual = pr - (noise_20mhz + 5.0)

    checks = [
        ("FSPL @ 1 GHz, 1 km = 92.45 dB", abs(fspl_1g - 92.45) < TOL_DB, f"{fspl_1g:.2f} dB"),
        ("FSPL @ 2.4 GHz, 1 km = 100.05 dB", abs(fspl_24g - 100.05) < TOL_DB, f"{fspl_24g:.2f} dB"),
        (
            "kTB noise in 1 Hz = -203.98 dBW",
            abs(noise_1hz + 203.98) < TOL_DB,
            f"{noise_1hz:.2f} dBW",
        ),
        (
            "noise in 20 MHz = -130.97 dBW",
            abs(noise_20mhz + 130.97) < TOL_DB,
            f"{noise_20mhz:.2f} dBW",
        ),
        ("Friis identity holds", abs(pr - pr_manual) < 1e-9, f"{pr_manual:.2f} dBW"),
        ("SNR definition Pr/N consistent", abs(snr - snr_manual) < 1e-9, f"{snr:.2f} dB"),
    ]
    ok = True
    for name, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")
        ok = ok and passed
    return ok


if __name__ == "__main__":
    import sys

    sys.exit(0 if run() else 1)
