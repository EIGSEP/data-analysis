# notebooks/dominic/vector_beam — vector-beam solve on Marjum 2026-07

Recovers the sky antenna's complex vector beam `u` (three Cartesian
components per pixel) from the transmitter cross-correlations, at the
pixels the raster sampled, and checks it against HFSS and against held-out
data. The method is `eigsep_data.vector_beam`; this directory holds the
campaign inputs and the checks.

## Setup

Three antennas: the rotating sky bowtie (the unknown `u`), a stationary
ground bowtie (the correlation partner, never modelled; its fixed
response to each transmitter arm is absorbed into a per-channel complex
D-term `g`), and the stationary cross-dipole transmitter. Both bowties
are single-port; only the transmitter has two arms. Per sample:

    Ex = E1 . u,   Ey = g (E2 . u),   r_hat . u = 0 (far-field prior)

The component of `u` along `r_hat` is never measured, only set by the
prior.

## Files

| File | Role |
|---|---|
| `pipeline.py` | Campaign inputs: data window, cached geometry, the sky/ground RFI recipe, the cross reduction constants (`SHIFT`, `TAU_NS`, trust mask). `load()` returns the geometry and the 30 `freq_map` channels. |
| `regression_check.py` | Re-runs closure `g`, the HFSS comparison and the leave-one-out check on every channel and compares with `reference/`. |
| `reference/` | The September 2026 results this must reproduce: `all_channels_xyz_validation.json`, `loo_forward_check_results.json`. |

## Inputs not yet updated

`pipeline.py` reproduces the September study's inputs on purpose, so a
change in a result can be traced to the method or to the inputs, never
both. It still uses potentiometer azimuth via `extract_beam_mapping_data`
and the geometry in `joint_fit_cache.npz` (set `VECTOR_BEAM_CACHE`; it is
not committed), fitted by the since-removed `fit_multi_freq_joint` in
`beam_sim`'s rotation convention. Moving to `load_bundle`, pointing table
v3 and geometry release v0004 is the next step.

## Known results (September 2026, reproduced 2026-10-20)

- HFSS comparison: recovered maps reproduce HFSS structure in x, y, z and
  total power for most of the band; y is the weakest component
  everywhere.
- Held-out check: median power correlation 0.912 over 30 channels, 0.923
  without d86-89 (222.7-228.5 MHz), which fail from an interleaved-comb
  fill defect. d75/77/84 lose Ey alone; cause unknown.

## Recent changes

- 2026-10-20: added; the solver moved into `eigsep_data.vector_beam`.
