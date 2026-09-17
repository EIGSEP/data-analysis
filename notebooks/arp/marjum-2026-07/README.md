# notebooks/arp/marjum-2026-07 — beam-mapping analysis

Aaron's beam PCA / rotation-fit / plotting work for the EIGSEP **Marjum Pass,
UT** deployment of 12–18 July 2026 (`deployment5` in older tooling).

Imported here 2026-09-12 from `~/projects/eigsep/marjum-2026-07/` where it had
grown up alongside the campaign's annotated ground truth. The two are now
separated: this directory holds the exploratory analysis; the campaign
directory (a different git repo) holds only the JSONL/Markdown ground truth
and the curation scripts that produce it. See
`~/projects/eigsep/marjum-2026-07/INDEX.md` for the campaign inventory and
data-selection playbook.

## Layout

Flat — 29 files, no subpackage structure. Nine are import targets used by
others in this directory; the rest are runnable scripts, demos, tests, or
a notebook:

| Import target (× importers) | Purpose |
|---|---|
| `rotation_beam.py` (18) | Rotation-axis beam-fitting core. |
| `tx_beam_sim.py` (18) | HFSS beam-set loading + TX simulation. |
| `v007_beam_diagnostic.py` (14) | Data loaders + diagnostic fits for the v007 beam. |
| `beam_pca.py` (10) | Low-order PCA/POD spectral basis over the HFSS beam. |
| `fit_v007_pca_beam.py` (9) | PCA-basis beam fitting used by the `make_*` scripts. |
| `data_space_rfi.py` (9) | Model-independent RFI flagging from raw correlator data. |
| `fast_mollview.py` (2) | Faster healpy Mollweide plotting shim. |
| `hpm.py` (1) | Local HPM copy; imported by `sim.py`. |
| `sim.py` | Chain-of-imports helper on top of `hpm.py`. |

Data-integrity checks (added 2026-09-13, answering MEMO-001 and the
comb-spacing question):

| Script | Question it answers |
|---|---|
| `check_wrap_impact.py` | Which fit samples are corrupted by int32 accumulator wrap, and by how much. |
| `check_wrap_survives_flagging.py` | Whether `gross_power_time_flags` already rejects them. |
| `check_rescore_repaired.py` | How much the fitted geometry and per-channel gains move once wrap is repaired. |
| `check_comb_eras.py` | The TX comb spacing per campaign era, measured on raw spectra. |
| `check_comb_presence_scan.py` | Per-file TX comb on/off map across the beam-scan window → `comb_presence_beam_scan.json`. |
| `check_comb_off_in_fit.py` | Impact of the comb-off files that leak into the hardcoded `load_v007_data` slice. |

The last two found the load-bearing problem: **the TX comb is off for 65 of the
227 beam-scan files**, and three of them sit inside `files[-185:-150]`, the
slice `load_v007_data` hardcodes. They were not flagged, and they biased the
fitted TX heading by 10.6°. `load_v007_data` now gates on `comb_present()`;
pass `require_comb=False` to reproduce pre-2026-09-13 results.

Entry points: `compare_real_tx.py`, `cv_ridge.py`, `demo_*`, `explore_basis*`,
`fit_v007_multichannel_consensus.py`, `grow_v007_beam_consensus.py`,
`make_*`, `plot_real_tx_beam.py`, `screen_v007_tx_channels.py`,
`test_*`, `fit.py` (orphaned utility, no importers). Notebook:
`EIGSEP_data_explore_v007_beammap.ipynb`.

`regen/` and `regen2/` are Aaron's manual output directories from successive
`fit_v007_multichannel_consensus.py` runs.

## Paths

- Beam maps: `BEAM_FILE = "../../../hfss_beam_maps/bowtie_beam.npz"` — three
  hops up lands in `eigsep_data/`, then into `hfss_beam_maps/`.
- Correlator data: pass `--data ~/projects/eigsep/marjum-2026-07/data/` on the
  CLI. Aaron's scripts take data paths as arguments; nothing is hardcoded.
- Analysis outputs (`*.png`, `v007_*.json`, `regen/v007_*.json`) are
  gitignored — regenerable from the scripts. Add specific ones with
  `git add -f` if worth committing.

## Runtime

Use the shared analysis interpreter:

```bash
ARP=/home/aparsons/.local/share/mamba/envs/arp/bin/python3
$ARP make_aug_beammap.py --data ~/projects/eigsep/marjum-2026-07/data/ --output aug_beammap.png
```

## Future

Per `~/projects/eigsep/REORG_PLAN.md` §"Working Directories", the mature
lib-shaped modules here (`rotation_beam`, `tx_beam_sim`, `beam_pca`,
`fit_v007_pca_beam`, etc.) are candidates for eventual promotion into an
`eigsep_data.beam_mapping/` subpackage. This flat layout is an intentional
intermediate state — not the endpoint.
