# marjum-2026-07 flag studies (rfi-analyst)

Review notebooks for the RFI flagging work on the Marjum Pass 2026-07
campaign, moved here from `marjum-2026-07/flags/{b15,b16,qa,v1}/` on
2026-09-19. The campaign repository holds data products and annotations;
notebooks and the code that builds them live here.

Each study is a notebook plus its rendered `.html` and `.pdf` — the PDF
is the review artifact Aaron reads, so all three travel together.

| Study | Notebook | Was |
|---|---|---|
| B15 external RFI characterization | `external_rfi_characterization` | `flags/b15/` |
| B16 DPSS smooth model + residual PCA | `dpss_smooth_model_and_residual_pca` | `flags/b16/` |
| B16 full-campaign checkpoint | `full_campaign_checkpoint` | `flags/b16/` |
| Waterfall sanity check | `waterfall_sanity_check` | `flags/qa/` |
| B8 coincidence validation | `checkpoint_coincidence_validation` | `flags/v1/` |

`flags/v1/` was never a mask product — it was this unfinished B8
investigation occupying a version number. Only `flags/v0` and `flags/v2`
are products.

## The `build_*.py` scripts are frozen provenance

They generated the notebooks above and are kept so the artifacts can be
traced, **not** so they can be re-run. Their embedded cell source still
refers to the pre-2026-09-19 layout (`sys.path` into
`marjum-2026-07/flagging/`, `MARJUM_DATA_ROOT`, and in
`build_b16_full_campaign_checkpoint.py` a `/tmp/rfi-b16-linear-wt`
worktree that no longer exists).

Those paths were deliberately **not** rewritten. A builder that no longer
matches the artifact it produced is worse than one that is honestly
stale: the rendered notebooks cannot be re-executed to validate a
rewrite, because their inputs (`flags/b15/_scratch/`, the b16 worktree)
are gitignored scratch or gone. If you need one of these to run again,
port it deliberately onto `eigsep_data.set_campaign_root()` and record
that you did.

`rfi_explorer_core.py` and `build_rfi_explorer.py` are likewise
superseded: the current `notebooks/arp/marjum-2026-07/rfi_explorer.ipynb`
reads through `eigsep_data.load_bundle` directly and imports neither.

## What moved to the package instead

`detectors.py`, `build_masks.py` and `validate.py` — the flags/v0
producer — are now `eigsep_data.flagging`. `select_files.py` is now
`eigsep_data.select_files`. Scripts here import them from there.
