# debug/ — superseded and single-question notebooks

Notebooks that are kept for provenance but are **not** maintained interfaces to
the campaign data. They may reference caches, paths, or package APIs that no
longer exist; do not expect them to run.

For a working view of the data, use the notebooks one level up. Each of those
reads `marjum-2026-07/` and the installed packages at run time and caches
nothing.

Moved here 2026-09-21 when the `arp/marjum-2026-07` set was tidied:

| File | Superseded by | Why |
|---|---|---|
| `rfi_dev_v0.ipynb` | `../rfi_dev_v2.1.ipynb` | First pass at the DPSS background model. |
| `rfi_dev_v1.ipynb` | `../rfi_dev_v2.1.ipynb` | Pre-`eigsep_data.rfi_supported`; carried the algorithm inline. |
| `rfi_dev_v2.ipynb` | `../rfi_dev_v2.1.ipynb` | Same method, before the package API and the antenna-resolution policy. |
| `rfi_explorer.ipynb` | `../rfi_dev_v2.1.ipynb` | Earlier viewer, narrower set of views. |
| `rfi_explorer_bak.ipynb` | `../rfi_dev_v2.1.ipynb` | Backup copy. |
| `rfi_flag_prototype.ipynb` | `../rfi_dev_v2.1.ipynb` | Prototype detector study; imports the local `rfi_proto.py`. |
| `rfi_dev_v2.1_delay_scratch.ipynb` | — | In-painting / delay-transform exploration split out of `rfi_dev_v2.1`. Needs that notebook's sections 1–3 run first. |
| `beam_explorer_bak.ipynb` | `../beam_explorer.ipynb` | Backup copy. |
| `geometry_explorer_bak.ipynb` | `../geometry_explorer.ipynb` | Backup copy. |
| `camera_proposal_repair_20261007.ipynb` (+ `.html`) | memo 002 | October proposal code and its executed rendering; memo 002 supersedes its interpretation, and old terrain paths are not maintained. |

Also here:

| File | Why |
|---|---|
| `build_lidar_explorer_cache.py` | Built `lidar_explorer_cache.npz` for the pre-2026-09-21 `lidar_explorer.ipynb`. That notebook now reads the DEM and the pointing table directly, so there is no cache to build. Kept because it documents the verification that the notebook's old hand-rolled ray march reproduced `marjum_lidar_constraint.march` exactly. |

Moved here 2026-09-24, when `scripts/marjum-2026-07/fit_beam.py` became the
beam-fit generator. Nothing in the directory above imports any of these. They
import each other by bare module name, so they are kept together:

| Files | What they were |
|---|---|
| `fit_v007_pca_beam.py`, `fit_v007_multichannel_consensus.py`, `grow_v007_beam_consensus.py`, `screen_v007_tx_channels.py` | v007 PCA / consensus beam fits on the hardcoded `files[-185:-150]` slice and motor pointing. |
| `fit_beam_v2.py`, `run_fit_beam_v2_pointingv1geom.py`, `beam_fits_v2_report.json`, `beam_fits_v2_pointingv1geom_report.json`, `build_antenna_mask.py` | Beam fits v2: the wide 226-file window on `pointing_table@v1` pot azimuth. |
| `explore_basis.py`, `explore_basis_v2.py`, `explore_basis_v3.py`, `cv_ridge.py` | Basis and ridge studies on the v007 fit. |
| `make_all_freq_comparison.py`, `make_aug_beammap.py`, `make_azel_comparison.py`, `make_beammap_style.py`, `make_combined_comparison.py`, `fast_mollview.py` | Figures from the v007 PCA fit. |
| `explorer_loader.py`, `explorer_model.py`, `build_explorer_cache.py`, `verify_explorer_loader.py`, `build_cal_gain.py`, `patch_explorer_*.py` | Cache, model and patches for the pre-2026-09-22 `beam_explorer.ipynb`. |
| `build_outlier_checkpoint.py`, `build_cal_toggle_checkpoint.py`, `patch_v2_checkpoint_stage{1,2}.py`, `render_{cal,outlier,v2}_checkpoint.sh`, `detect_el_slew_glitch.py`, `build_sample_sidecar.py` | Builders of the three `beam_*_checkpoint` notebooks here. |
| `check_comb_off_in_fit.py`, `check_rescore_repaired.py` | Checks on the v007 fit (comb-off files in the slice; wrap repair). |
| `plot_real_tx_beam.py`, `compare_real_tx.py` | Measured-vs-HFSS TX comparisons against the 92.5 m model. |

Superseded by `../../../../scripts/marjum-2026-07/fit_beam.py` (fits) and
`../beam_explorer.ipynb` (viewing).

Added 2026-09-24, the tooth background and tooth selection for beam fit v0009. These are live diagnostics that do run:

| File | What it answers |
|---|---|
| `tooth_background_debug.ipynb` (+ `.html`) | Why the v3-beta smooth model undershoots (it rejects positive excess); gap differencing against local DPSS on a transmitter-off control; delay spectra. Aaron approved replacing gap differencing with DPSS. |
| `tooth_background_basis_debug.ipynb` (+ `.html`) | Choice of DPSS basis: 150 ns smooth, no reflection modes. |
| `tooth_selection_debug.ipynb` (+ `.html`) | How `tooth_selection_v0001` fared in v0009; the neighbour-coherence test proposed for v0002. |

## Recent changes

- 2026-10-07: Preserved the camera-proposal diagnostic notebook and rendered
  HTML here after its results were incorporated into memo 002.
