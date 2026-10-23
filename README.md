# data-analysis

Notebooks and exploratory analysis for EIGSEP. **Not an installable
package** — there is no `pyproject.toml` here and nothing in this repo is
importable.

## The `eigsep_data` package moved

As of 2026-09-17 the package half of this repo lives in its own
repository: **https://github.com/EIGSEP/eigsep_data** (`src/eigsep_data/`,
`tests/`, `docs/`). Its history came across with `git filter-repo`, so
`git log` on any package file still works there.

The import name did not change. Install it into whatever environment you
run notebooks in and keep writing `import eigsep_data` exactly as before:

```sh
git clone git@github.com:EIGSEP/eigsep_data.git
pip install -e eigsep_data
```

Why the split: the package is ~600 KB of library code that other repos
depend on; the notebooks are ~470 MB of embedded outputs. They have
different audiences, different review needs, and different clone costs.

## One-time setup: notebook output stripping

Notebook outputs are embedded PNGs and dominate this repo — 60% of all
blob bytes in history are `.ipynb`. `.gitattributes` routes every
notebook through `nbstripout`, so what git stores is source only.

Run this once per clone, or your notebooks commit with their outputs:

```sh
pip install nbstripout
nbstripout --install          # defines the filter this clone uses
```

It is a **clean filter**, not a pre-commit hook: your working notebooks
keep their plots and nothing on disk is rewritten. Only the blob going
into git is stripped.

The filter records the interpreter that ran `--install`, so rerun
`nbstripout --install` after rebuilding or upgrading that Python
environment; otherwise every git command that touches a notebook fails
with `clean filter 'nbstripout' failed`.

CI (`.github/workflows/notebooks-stripped.yml`) runs `nbstripout --verify`
on every committed notebook for each pull request and push to `main`, so
a clone without the filter cannot get outputs in. If it fails on your
branch, install the filter and run `git add --renormalize -- '*.ipynb'`.

Because committed notebooks are therefore unexecuted, **a review
checkpoint's evidence is its rendered `.html` and `.pdf`**, which are
committed alongside and are not stripped. Commit all three.

## What is still here

- `notebooks/` — per-person exploratory work, including
  `notebooks/arp/marjum-2026-07/` (the July-2026 beam analysis).
- `hfss_beam_maps/` — see its README; the canonical copy now lives in
  `eigsep_data`.

## Recent changes

- 2026-10-23: Every committed notebook is now stripped, including the four
  in `notebooks/christian/` that kept their figures (Christian agreed); 28
  notebooks had outputs in git. A CI check now fails any pull request or
  push to `main` that commits a notebook with outputs.
