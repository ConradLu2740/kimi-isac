# Design: Conditional Diffusion 3D Reconstruction Module (`gen/`)

Date: 2026-09-20
Status: approved for planning
Supersedes: none
Related: README v2 roadmap item "conditional diffusion 3D reconstruction"

## 1. Intent and success criteria

Add the v2 generative subsystem to `kimi-isac`: procedural 3D point-cloud
templates, a physics-grounded echo model that turns a cloud into ISAC
observations, and a conditional latent diffusion model (VAE + DiT) that
reconstructs the cloud from those observations — while inheriting every
discipline the project established in v1 (single-source physics, fixed
splits, val-based selection, multi-seed bootstrap CIs, honest baselines).

Success criteria:

1. Unconditional class-conditional generation reports held-out-instance and
   held-out-class Chamfer distance with bootstrap 95% CIs (n >= 10 seeds),
   plus a train-vs-held-out memorization check.
2. Conditional reconstruction reports CD as a function of SNR and RIS phase
   mode (aligned / random / none) with CIs, and an oracle upper bound
   (per-scatterer alignment).
3. Every physical quantity in the echo model comes from `core/`; no new
   physics constants or conventions are introduced.
4. CI runs a CPU smoke of the whole chain in under 60 s; the full report
   runs on the RTX 5060 Laptop GPU.
5. The shared training engine extracted for this module keeps `ml/train.py`
   behavior identical (guarded by the existing 23 tests).

## 2. Scope

In scope (v2.1):

- Procedural point-cloud templates: 8 training classes + 2 OOD classes,
  parameterized, generated in-code, 512 points each, ROI-normalized to
  [-1, 1]^3.
- Single scene geometry: ISS overpass above Beijing (reuses
  `closedloop.iss_overpass_geometry`), UE fixed, 80 m x 80 m ground ROI,
  tau = 8 slow-time frames (dt = 0.5 s from the overpass start).
- Echo model: per-scatterer two-hop channels through `core.channel`,
  Doppler via `core.channel.doppler_shift_hz`, RIS panel from
  `core.ris` / `opt.phase_opt` aligned to the ROI centroid.
- 10-dim per-frame conditioning features (amplitude/phase/Doppler/delay/
  distance/elevation/RCS/IRS-phase) on the scalar BS-UE link.
- VAE (z = 256) + latent DDPM DiT with cross-attention conditioning on
  (class embedding, feature sequence).
- Multi-seed report: unconditional CD (held-out instances/classes) and
  conditional CD curves over SNR x RIS mode, with oracle upper bound.

Out of scope (v2.2+ candidates, recorded so they are not smuggled in):

- Flow matching, RD-map CNN conditioning, scatterer-token conditioning.
- Multi-scene / GEO / MEO geometry, multi-target clouds, clutter.
- Cross-range estimation claims: the conditioning signal physically
  lacks cross-range information (single-station geometry); the model's
  cross-range content comes from the class prior. This is a finding to
  report, not a gap to hide.

## 3. Architecture

New package `src/kimi_isac/gen/`:

```
gen/
├── templates.py   # procedural point-cloud classes (8 + 2 OOD), parameterized
├── scene.py       # ROI anchor -> ECEF scatterers; ISS frames (delegates to closedloop geometry)
├── echo.py        # scatterers -> per-frame complex echo + 10-dim features; RIS modes
├── dataset.py     # fixed splits (core/splits); unconditional + conditional views
├── vae.py         # PointVAE: 512x3 -> z(256)
├── dit.py         # 1D latent DiT denoiser + condition encoder (class + feature seq)
├── train.py       # two-stage training on the shared engine
├── metrics.py     # symmetric Chamfer distance + aggregation helpers
└── report.py      # multi-seed report: unconditional CD, conditional CD curves, oracle
```

New shared engine `src/kimi_isac/training/engine.py`:

- Callback-based epoch loop: `build_model()`, `train_step(batch) -> loss`,
  `val_loss() -> float`, plus optimizer/scheduler construction hooks.
- Keeps: val-based best-checkpoint selection, early stopping patience,
  ReduceLROnPlateau, gradient clipping (max_norm=1.0), checkpoint metadata
  (epoch, config, metrics, optimizer) via `ml/checkpoint.py`.
- `ml/train.py` is refactored onto the engine with identical behavior;
  the existing test suite guards the refactor. `gen/train.py` builds on the
  same engine, so there is exactly one training-loop implementation.

## 4. Physics: echo model and RIS coupling

Scene (per sample): satellite state from the frozen ISS TLE propagated
over the first qualifying overpass; UE at Beijing (39.9042 N, 116.4074 E);
ROI anchored 200 m from the UE, 80 m x 80 m on the ground plane.

Per scatterer i (3-D position p_i in ECEF, reflectivity rho_i per class):

- Two-hop channel: `g_i(t) = free_space_channel(f, d_sat(t)->p_i) *
  free_space_channel(f, d_p_i->ue) * rho_i`.
- Bistatic range rate -> Doppler via `range_rate_mps` + `doppler_shift_hz`
  (IEEE sign convention: approaching positive).
- Received complex signal: `Y(t) = sum_i g_i(t) * exp(j 2 pi f_Di t_rel)`
  plus complex Gaussian noise at the configured per-sample SNR.

RIS coupling (the scientific core):

- Controller aligns panel phases to the ROI **centroid** scatterer via
  `opt.align_phases` (what a real controller can do: it only knows the
  dominant target). The aligned element sum multiplies the centroid
  contribution coherently; off-center scatterers accumulate phase error
  proportional to their bistatic excess path, so wider clouds degrade.
- `random` mode: uniform random phases (seeded per sample).
- `none` mode: RIS off (panel contributes nothing).
- Oracle variant: per-scatterer ideal alignment (each scatterer phased
  independently) — an unreachable upper bound, reported separately.

Conditioning features per frame on the scalar BS-UE link (10 dims):

`amp_dB, sin(angle Y), cos(angle Y), doppler_kHz, delay_ms,
dist_sat_roi, elev_norm, rcs_log10, irs_sin, irs_cos` — deliberately
mirroring the interface the original IRS-Diffu-ISAC repo exposed, so its
reported numbers stay comparable, while every value is computed by
`core/` primitives.

Cross-range honesty: the feature vector contains no cross-range
information about the cloud (monostatic-ish bistatic geometry). The
reconstruction's cross-range content is supplied by the class prior
learned by the diffusion model. The report states this explicitly.

## 5. Models

- **PointVAE**: encoder MLP 512x3 -> 256 -> z (256); decoder mirror.
  KL warmup 5 epochs, weight 1e-4 (matches `ml` conventions).
- **Condition encoder**: class embedding (learned, 64) + feature sequence
  (tau x 10) through a 2-layer Transformer (d_model 128, 4 heads) ->
  pooled conditioning embedding (256).
- **Latent DiT**: 1D latent length 8 (z reshaped 256 -> 8 x 32 tokens),
  depth 4, hidden 256, 8 heads, AdaLN-zero conditioning on the pooled
  embedding; cross-attention to the feature-sequence tokens.
- **DDPM**: T = 100, linear beta schedule, deterministic sampling
  (implemented in `gen/dit.py`; no dependency on legacy code).
- Classifier-free guidance is out of scope for v2.1 (YAGNI).

## 6. Evaluation

`gen/report.py --seeds N` writes `results/gen_report.json`:

| Metric | Definition |
|---|---|
| `uncond_cd_heldout_inst` | CD of unconditional-class samples vs held-out instances |
| `uncond_cd_heldout_class` | CD on the 2 OOD classes (class prior absent) |
| `memorization_gap` | train CD vs held-out-instance CD difference with CI |
| `cond_cd[snr][ris]` | conditional reconstruction CD per SNR level and RIS mode |
| `cond_cd_oracle` | CD under per-scatterer oracle alignment |
| `cond_cd_none` | CD with RIS off (noise-only conditioning sanity) |

All metrics: n >= 10 seeds, mean with bootstrap 95% CI via
`core/stats.py`. Classical 3D reconstruction from a single-station
range-Doppler observation is physically impossible (no cross-range), so no
classical baseline is fabricated; the honest comparison is the oracle
bound and the SNR/RIS curves.

## 7. Testing and CI

- `tests/test_gen_templates.py`: template shapes, point counts,
  normalization range, determinism, OOD class disjointness.
- `tests/test_gen_echo.py`: echo energy scaling (single scatterer at known
  range produces the expected delay), Doppler sign, RIS aligned > random
  in mean echo energy (MC over seeds), feature dimensions and finiteness.
- `tests/test_gen_metrics.py`: Chamfer distance properties (zero for
  identical clouds, symmetry, known closed form for two-point sets).
- CI step: `python -m kimi_isac.gen.train --smoke` (CPU, tiny, < 60 s)
  plus `python -m kimi_isac.gen.report --smoke` (2 seeds, tiny).
- Full report on GPU: `python -m kimi_isac.gen.report --seeds 10`.

## 8. Risks

- **VAE error floor**: latent diffusion CD includes VAE reconstruction
  error. Report VAE recon CD separately so the diffusion contribution is
  visible. If the floor dominates, v2.2 upgrades the VAE (not the scope
  here).
- **Class prior leakage**: held-out-class CD is the honest measure of
  generalization; memorization gap guards instance-level leakage.
- **Runtime**: 10 seeds x (VAE + DiT) on the RTX 5060 must finish the
  report in ~20 min; the smoke path caps samples/epochs to keep CI fast.
- **Engine refactor regression**: mitigated by running the full v1 test
  suite before and after the extraction.
