# marjum-2026-07 product generators and one-off studies

Moved here from `marjum-2026-07/{curation,flagging,experiments,analysis}/`
on 2026-09-19. The campaign repository holds data products and
annotations; the code that generates them lives here, and the reusable
parts live in `eigsep_data`.

## Point at a campaign first

Nothing here anchors on its own `__file__` any more — these scripts used
to sit inside the campaign tree and walk up to it. Set the campaign once:

```sh
export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
```

or in Python, `eigsep_data.set_campaign_root(...)`. The older
`MARJUM_DATA_ROOT` still wins where it was already honoured, for a
worktree that lacks the gitignored raw data.

## Layout

| Directory | What it generates |
|---|---|
| `curation/` | The campaign's tracked tables: `file_state.csv`, `mode_table.jsonl`, `cal_windows.jsonl`, `boundaries.jsonl`, the comb tables (`boxair_emi_presence`, `transmitter_presence`, `comb_eras`, `beamscan_comb_identity`, `boxair_emi_matched`), `overflow_channels.jsonl`. |
| `flagging/` | B15/B16 flag studies, the comb inventory, the box-gnd transmitter per-integration trace and on/off episodes (`scan_transmitter_per_integration.py`, `derive_transmitter_transitions.py`), `make_figures.py` for flags/v0, and the B16 DPSS trial. |
| `experiments/` | The B10 natural-experiment scan/summary pairs behind MEMO-005…011. |
| `b3/` | The beam mode-budget / horizon-sensitivity study, formerly `marjum-2026-07/analysis/b3/`. |
| `tcal/` | The `derived/tcal/` box-air temperature calibration: build and validate. |
| `ground_sky/` | Ground temperature and sky solved from an assumed beam (`eigsep_sim.design_matrix`); for now, the degeneracy study on the real geometry. |

Outputs still land in the campaign repository, beside the data they
describe. These scripts write there; they do not keep their own copies.

## Comb names

Memo 001 ("Combs, and which one is the transmitter") names the two combs;
code here uses its names. The **transmitter** is the 8-channel
(1.953125 MHz) comb on channels == 0 mod 8 from 07-17 15:36:22, on both
antennas. **`boxair_emi`** is box-air's own 1.000 MHz self-EMI on 07-16
01:16:57-16:49:03. Before 2026-10-03 these scripts called them the other way
round ("digital self-comb" for the transmitter, "tx"/`tx_comb`/`tx_on` for
the EMI). Renamed in that change:

| Old | New |
|---|---|
| `curation/scan_tx_presence.py` → `tx_presence.jsonl` (`tx_on`) | `scan_boxair_emi_presence.py` → `boxair_emi_presence.jsonl` (`boxair_emi`) |
| `curation/build_self_comb_presence.py` → `self_comb_presence.jsonl` | `build_transmitter_presence.py` → `transmitter_presence.jsonl` (`transmitter`) |
| `curation/scan_tx_comb.py` → `tx_comb_eras.jsonl` | `scan_comb_eras.py` → `comb_eras.jsonl` |
| `curation/build_tx_state_matched.py` → `tx_state_matched.jsonl` (`tx_state`) | `build_boxair_emi_matched.py` → `boxair_emi_matched.jsonl` (`boxair_emi_state`); it detects the 1 MHz EMI, so it is kept under that name, not retired |
| `flagging/scan_self_comb_per_integration.py`, `derive_self_comb_transitions.py` | `scan_transmitter_per_integration.py`, `derive_transmitter_transitions.py` |
| `mode_table.jsonl` column `tx_comb` | `boxair_emi`, plus a new `transmitter` column |
| `b15_event_survey.self_comb_channel_mask` | `comb_teeth_channel_mask` (old name kept as an alias) |

Frozen scripts in `debug/` and the notebooks keep the old names.

## What is *not* here

`detectors.py`, `build_masks.py`, `validate.py` and `select_files.py`
moved into `eigsep_data` — they have callers beyond any one study, and
`eigsep_data.products.flags` was already the reader for what
`build_masks` writes. Import them:

```python
from eigsep_data.flagging import detectors, build_masks, validate
from eigsep_data import select_files
```

## B16 is a trial, not production

`flagging/b16_dpss_model.py`, `package_dpss_into_mask.py` and
`run_b16_full_campaign.py` are here rather than in `eigsep_data`
deliberately. The product's own manifest says "not validated for
production use as-is", refinement made the fit worse for nearly every
file (1/28, 0/28 improved), and its output bit in `flags/v2` has a known
uncentred-threshold defect. See `marjum-2026-07/flags/v2/README.md`.

## Beam fits (top level)

| Script | Role |
|---|---|
| `fit_beam.py` | Thin driver on `eigsep_data.beam_mapping`; generator of `derived/beam/empirical_raster_v0012` onward. Campaign choices (raster window, pointing, flags, site geometry) live here, the method in the package. `--concentration-min legacy` uses the v0009–v0011 background (15 DPSS modes, which cannot represent a constant); the default 1e-6 keeps 21. |
| `build_tooth_selection.py` | Thin driver on `beam_mapping.tx_teeth.select_teeth`; builds `derived/beam/tooth_selection_vNNNN.{json,csv}`. With `--concentration-min legacy` it reproduces `tooth_selection_v0002` exactly; the corrected background selects the same 64 teeth. |
| `build_raster_flags.py` | Builds `derived/beam/raster_flags_v0001.npz`. |

The generators of v0008–v0011 and `tooth_selection_v0001/v0002` (`fit_beam_legacy.py`,
`empirical_beam.py`, `empirical_beam_spectral_no_el0.py`, `tooth_background.py`,
`build_tooth_selection_legacy.py`) are frozen in `debug/`, byte-identical to the hashes
those products record.

Superseded generators are in `debug/` (see `debug/README.md`).

## Recent changes

- 2026-10-03: comb labels follow memo 001 (see "Comb names"): curation and
  flagging generators renamed, `build_mode_table.py` now writes `boxair_emi`
  and a per-file `transmitter` column, its science census excludes both
  combs, and consumers read the new column names.

- 2026-10-02: `build_known_quantities_xlsx.py` cites the living memos 002, 003 and 005
  instead of legacy MEMO-012/013 and the removed `AZIMUTH_CONVENTIONS.md`.
- 2026-10-01: added `ground_sky/`, the start of the ground-temperature and
  sky solve; its first study shows what the Marjum geometry can constrain.
- 2026-10-01: added `tcal/`, which builds and validates the first field
  temperature calibration product, `derived/tcal/v0000/`.
