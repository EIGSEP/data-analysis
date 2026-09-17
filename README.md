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

## What is still here

- `notebooks/` — per-person exploratory work, including
  `notebooks/arp/marjum-2026-07/` (the July-2026 beam analysis).
- `hfss_beam_maps/` — see its README; the canonical copy now lives in
  `eigsep_data`.
