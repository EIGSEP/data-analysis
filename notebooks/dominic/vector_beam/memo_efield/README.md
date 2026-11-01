# memo_efield — scripts behind the E-field beam memo

Every number and figure of the memo "Marjum 2026-07 vector E-field beam"
(EIGSEP.github.io, `memos/marjum_efield_beam/`). The method is
`eigsep_data.beam_mapping.vector_beam`; the mount comes from
`eigsep_data.beam_mapping` (`TxGeometryModel` with `axle_tilt_deg`,
`AZ_SENSE`), so this needs eigsep_data at or after EIGSEP/eigsep_data#20.

## Method as used in the memo

- Per-pixel solve, no global basis and no HFSS prior. Per integration:
  `Ex = E1 . u`, `Ey = g (E2 . u)`, `r_hat . u = 0`, with `r_hat` the
  body-frame arrival direction of that integration (a local transversality
  row, not a fixed axis).
- One complex closure `g` per tooth, refitted per mount.
- No per-dwell phase calibration: `vb.self_calibrate_phase` constrains only
  4200 of 6810 integrations and would drop ~45% of pixels.
- Bracketed-fill rule: extrapolated DPSS fills are NaN, never replaced.
- Mount L, the memo's: left-handed azimuth (memo 003, pointing table v5),
  memo 002's v0005 posterior-mean positions, az zero +15.8 deg, alpha0 8 deg,
  el zero +0.05 deg and v0016's axle tilt +4.7 deg. `mounts.py` also defines
  R (right-handed, v0013, the first draft) and L0 (L without the tilt) for
  comparison.
- Maps use one real power scale per tooth: `compare_to_truth`'s single
  complex scale understates the amplitude by 6-8 dB when per-pixel phases
  differ from HFSS.

## Running

From this directory, with eigsep_data's environment:

    # once: inputs (~6 min; needs the pointing table and the joint-fit cache)
    cd ../bracketed_fill_check
    POINTING_TABLE=.../pointing_table.parquet VECTOR_BEAM_CACHE=.../joint_fit_cache.npz \
        PYTHONPATH=.. python dump_inputs.py ../memo_efield/inputs.npz
    cd ../memo_efield
    python compute_beams.py          # output/scores.json, beams_maps.npz (~10 min)
    python dwell_repeatability.py    # output/dwell_repeatability.json (~5 min)
    VECTOR_BEAM_CACHE=... python count_auto_pixels.py   # output/auto_pixels.json
    python make_figures.py           # output/figures/*.pdf, output/values.tex

`output/values.tex` is the memo's `values.tex`. Set `EFIELD_INPUTS` to use
inputs elsewhere.

| File | Role |
|---|---|
| `mounts.py` | Loads the inputs (bracketed-fill rule) and builds the R / L0 / L geometries. |
| `compute_beams.py` | Solve and HFSS score on every tooth under each mount; maps for L. |
| `dwell_repeatability.py` | Per-dwell solves on multi-dwell pixels: repeat rms vs rms against HFSS, by HFSS share (memo, "Noise or systematics at the nulls"). |
| `count_auto_pixels.py` | Pixels covered by the auto-power method vs the cross inputs and the solve (memo, "Comparison with HFSS"). |
| `make_figures.py` | Figures and the number macros. |
| `supporting/` | Tilt scan, error budget and phase-jump analyses (memo Figs. 5-7), run under the earlier right-handed convention; see its README. |

## Results (mount L, 29 teeth, 2026-11-01)

Median correlation with HFSS: x 0.986, y 0.985, z 0.983, total power 0.981
(R: 0.985 / 0.958 / 0.981 / 0.977; L0: 0.985 / 0.944 / 0.976 / 0.976).
Maps at 175.8, 199.2 and 222.7 MHz: 0.42, 0.49, 0.72 dB power-weighted rms.
At the y and z nulls, 74% and 71% of the variance of the solve - HFSS
difference repeats between independent dwells: a direction-dependent
systematic, not noise.
