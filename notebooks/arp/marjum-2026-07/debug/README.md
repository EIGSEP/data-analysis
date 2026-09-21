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
