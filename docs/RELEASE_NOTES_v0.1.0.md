# v0.1.0 — First public release

**kimi-isac: Physics-Grounded, Statistically Honest ISAC Simulation**

A reference implementation of RIS-aided Integrated Sensing and
Communication spanning LEO satellite links (SGP4), OFDM sensing, RIS
phase control, and a sensing–communication closed loop — where every
headline number carries its statistical footing and every verification
compares against *external* ground truth, never against the code itself.

## Highlights

- **Physics core with external-truth verification** — 26 checks that
  pass against textbooks, published ISS elements, and analytic bounds
  (orbit period 92.95 min, Friis 92.45/100.05 dB, Doppler sign
  convention, Parseval conservation), kept structurally separate from
  self-consistency regressions.
- **Classical sensing done right** — CA-CFAR with Monte-Carlo-validated
  false-alarm calibration; delay estimation that achieves the
  Cramér–Rao bound (1.03×), plus a measured demonstration that envelope
  detection costs exactly 2× versus the coherent estimator.
- **ML vs classical, fairly compared** — 10 seeds, bootstrap 95% CIs,
  fixed on-disk splits, val-based checkpoint selection, and an OOD suite
  that runs by default. Result: the classical pipeline wins single-target
  localization (2.7 m vs 33 m) and dominates out-of-distribution; the ML
  head's only edge is detection accuracy.
- **RIS closed loop over a real ISS overpass** — per-frame phase
  tracking delivers +43.5 dB over random phases (99.7% of the greedy
  bound); segmented reconfiguration (K ≤ 8) loses essentially all gain at
  30 GHz LEO Doppler rates.
- **Conditional latent diffusion 3D reconstruction** — procedural
  templates, physics-grounded echo conditioning, VAE + latent DiT,
  stratified (SNR × RIS mode × class) ablation grid. Honest negative
  result: conditional reconstruction does not beat the class prior on
  single-station echoes, because the echo physically lacks cross-range
  information.

## Reproducibility

```bash
pip install -e ".[dev,ml]"
python -m kimi_isac.verify        # 26 external-truth checks (~10 s)
python -m pytest -q              # 56 tests
python -m kimi_isac.ml.report --smoke        # ML pipeline smoke
python -m kimi_isac.gen.report --smoke       # generative pipeline smoke
python -m kimi_isac.viz                     # regenerate demo assets
```

Full reports (GPU, ~15 min each):
`python -m kimi_isac.ml.report --seeds 10 --device cuda`,
`python -m kimi_isac.gen.report --seeds 10 --device cuda`.

## Notes

- All data and weights are synthesized in code; no downloads.
- CI: ruff + mypy + pytest + physics verification + pipeline smokes.
- `docs/superpowers/` contains the design spec and implementation plan
  for the generative module, kept for methodology transparency.
- See TECH_REPORT.md for the full write-up and the three negative
  results.
