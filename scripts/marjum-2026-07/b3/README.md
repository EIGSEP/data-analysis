# B3 — beam chromaticity mode budget, and the horizon accuracy requirement

`forward-modeler`. v0 of the budget; B4 repeats it with the measured beam.

Two questions, one for the instrument model and one that sets another
agent's requirement:

1. **How many spectral eigenmodes describe the EIGSEP bowtie beam's frequency
   evolution to ~1 part in 1e4** (the foreground/signal dynamic range)?
2. **How accurately must the horizon be known** before horizon error, not
   thermal noise, limits the recovered 21-cm signal?

## Scripts

| script | produces |
|---|---|
| `beam_mode_budget.py` | `b3_mode_budget.json` — uncentered SVD over the frequency axis of the HFSS beam power map, residual vs mode count K against the 1e-4 line. Geometry-free: no terrain, horizon, or sky weighting. |
| `horizon_sensitivity.py` | `b3_horizon_sensitivity.json` — perturbs the measured Marjum horizon (rigid bearing rotation, elevation error, and geometer's own eight displaced profiles) and reports the induced antenna-temperature error. |
| `make_figures.py` | `b3_fig1_residual_vs_modes.png`, `b3_fig2_mode_budget.png` |

Run from this directory; each writes its JSON beside itself and caches sky
models as `_sky_cache_*.npz` / `_gsm_cache_*.npz` (gitignored, ~44 MB,
rebuilt on first run).

## Inputs, and why this study is not in a package repo

It reads across three repositories, which is why it lives under
`marjum-2026-07/analysis/` rather than inside `eigsep_sim`:

| input | repo |
|---|---|
| `eigsep_data/hfss_beam_maps/bowtie_beam.npz` | eigsep_data |
| `eigsep_sim/src/eigsep_sim/data/eigsep_bowtie_v000.npz` | eigsep_sim |
| `eigsep_sim/src/eigsep_sim/data/models_21cm.npz` | eigsep_sim |
| `marjum-2026-07/curation/horizon_profiles.npz` | this repo |

Nothing here imports `eigsep_sim` as a package — only its data files, plus
numpy, healpy and scipy.

## The two beam products disagree

`beam_mode_budget.py` loads **both** candidate beam products and reports
their disagreement per band in `product_disagreement`. It is the measurement
of that dispute, not a consumer of either product as truth:

| band | rms frac. difference | max |
|---|---|---|
| trough 60–100 MHz | 8.2% | 10.6% |
| cosmology 50–110 MHz | 9.7% | 14.3% |
| midband 50–130 MHz | 13.3% | 19.9% |
| **fullband 50–250 MHz** | **28.9%** | **62.5%** |

The commonly quoted "8–13% rms" is the in-band figure. Across the full
50–250 MHz band the two products differ by 28.9% rms and up to 62.5%, so the
disagreement is substantially worse outside the cosmology bands.

**This is unresolved.** Until it is, the mode budget is reported per product
rather than as a single number, and no downstream consumer should treat
either beam as the beam.

## Status

`BRANCH_MERGE_PLAN.md` §3.6 held this work back on the grounds that it is
"exactly the place that disputed input would land." That rationale does not
match the code: B3 measures the disagreement rather than consuming it. The
holdout came from an explicit dispatch instruction, so it stands until
whoever issued it confirms — but the stated technical reason should not be
the basis for it.
