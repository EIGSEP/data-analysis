# notebooks/dominic/vector_beam — vector-beam solve on Marjum 2026-07

Recovers the sky antenna's complex vector beam `u` (three Cartesian
components per pixel) from the transmitter cross-correlations, at the
pixels the raster sampled, and checks it against HFSS and against held-out
data. The method is `eigsep_data.beam_mapping.vector_beam`; this
directory holds the campaign inputs and the checks.

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

## Extrapolated fills (2026-10-21)

Each channel measures one transmitter arm; the other arm's value is a
DPSS fill from neighbouring tones. Every fill >100x its channel's median
(up to ~1e8x, d75-d89) was an extrapolation: a same-arm tone missing on
one side, or, at d89, the untrusted 230.5 MHz column above it. Fills with
a trusted measured tone on both sides stay within the measured spread.
`run_phase2.py` now NaNs the extrapolated fills (the default of
`vb.reduce_cross` and `pipeline.load`) and writes
`phase2_bracketed_results.json`; `BRACKETED_FILL=0` gives the 2026-10-20
behaviour. `regression_check.py` passes `require_bracketed_fill=False`.
Selection B medians, Phase 2 -> bracketed:
LOO power 0.958 -> 0.970, power 0.954 -> 0.977, y 0.930 -> 0.958,
z 0.907 -> 0.981; d77 and d82 LOO 0.43 -> 0.96, d85-88 ~0 -> 0.91-0.94; no
channel worse. Cost: median 91% of pixels kept (min 72%); d89 has no
valid samples. Figures and the scripts behind them are in
`bracketed_fill_check/` (`dump_inputs.py` caches the inputs they read).

## Recent changes

- 2026-10-21: the solver moved to `eigsep_data.beam_mapping.vector_beam`;
  the bracketed-fill rule is the default.
- 2026-10-21: bracketed-fill rule; d77/d82 and d85-88 LOO failures were
  extrapolated DPSS fills.
- 2026-10-20: added; the solver moved into `eigsep_data.vector_beam`.
