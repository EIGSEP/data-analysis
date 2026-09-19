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
| `curation/` | The campaign's tracked tables: `file_state.csv`, `mode_table.jsonl`, `cal_windows.jsonl`, `boundaries.jsonl`, the TX/comb presence tables, `overflow_channels.jsonl`. |
| `flagging/` | B15/B16 flag studies, the comb inventory, `make_figures.py` for flags/v0, and the B16 DPSS trial. |
| `experiments/` | The B10 natural-experiment scan/summary pairs behind MEMO-005…011. |
| `b3/` | The beam mode-budget / horizon-sensitivity study, formerly `marjum-2026-07/analysis/b3/`. |

Outputs still land in the campaign repository, beside the data they
describe. These scripts write there; they do not keep their own copies.

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
