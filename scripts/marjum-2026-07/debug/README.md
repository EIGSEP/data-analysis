# debug/ — superseded beam-fit generators

Kept for provenance; not maintained, and not expected to run from here (each
loads its siblings with `SOURCE.with_name(...)`, which now resolves inside
`debug/`). The current generator is `../fit_beam.py`, which still loads
`../empirical_beam_spectral_no_el0.py` and `../empirical_beam.py`.

Moved here 2026-09-24:

| File | Generated | Superseded because |
|---|---|---|
| `empirical_beam_commanded.py` | `derived/beam/empirical_raster_v0006`, `v0007` | Renamed to `../fit_beam.py`, which reads `curation/pointing_table.parquet` (v2.x) instead of `point_table_v2-beta.npz`. This copy is byte-identical to the hash in v0007's `provenance.json`. |
| `empirical_beam_spectral.py` | `empirical_raster_v0002` | Included the opening el≈0 slew; replaced by `empirical_beam_spectral_no_el0.py`. |
| `empirical_beam_ell_scan.py`, `empirical_beam_ell_export.py` (+ `.orig`) | `empirical_raster_v0004` | ℓmax scan on pot-azimuth pointing. |
| `empirical_beam_pca_polemask.py` | `empirical_raster_v0005` | PCA fit with a masked pole region, on pot-azimuth pointing; not carried forward. |

Moved here 2026-09-25, when the beam fit moved into `eigsep_data.beam_mapping`
and `../fit_beam.py` and `../build_tooth_selection.py` became thin drivers on
it. These are the frozen generators of existing products; each is
byte-identical to the hash its products record under their original names.
To re-run them, copy the set back to `scripts/marjum-2026-07/` under their
original names: `empirical_beam.py` locates the campaign from its own path
(`parents[3]`), and each loads its siblings by name.

| File | Original name | Generated |
|---|---|---|
| `fit_beam_legacy.py` | `fit_beam.py` | `derived/beam/empirical_raster_v0008`-`v0011` (v0008/v0009 with `--convention legacy`) |
| `empirical_beam.py`, `empirical_beam_spectral_no_el0.py` | same | loaded by `fit_beam_legacy.py` and `empirical_beam_commanded.py` |
| `tooth_background.py` | same | the background of v0009-v0011 and of the tooth selections (floor(2NW)+1 DPSS modes) |
| `build_tooth_selection_legacy.py` | `build_tooth_selection.py` | `derived/beam/tooth_selection_v0001`, `v0002` |

`eigsep_data`'s `tests/test_beam_mapping_v0011_regression.py` checks that the
package reproduces v0011's model predictions from its inputs.

