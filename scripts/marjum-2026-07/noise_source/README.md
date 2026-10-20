# noise_source — the bench-measured noise-source product

Builds `marjum-2026-07/derived/noise_source/<version>/`: the noise source's
temperature at reference plane P, measured on the bench with a hot load
switched through the field switch board, independent of the receiver. `tcal`
from v0004 builds its noise-source excess from it.

| Script | What it does |
|---|---|
| `build_noise_source.py` | Runs `eigsep_data.noise_source_bench.measure` on one bench run and writes the product with `eigsep_data.products.noise_source.write`: `solutions.npz`, `manifest.json` (inputs with SHA-256, code commits, the wiring and thresholds) and `validation.json`. Holds only this run's choices: where its files are, how the bench was wired, the thresholds. |

The measurement (dwells, cycles, the mismatch-corrected fit, the mask) is in
`eigsep_data` and the physics in `eigsep_cal.noise_source`; the bench run's
as-built record is `notebooks/charlie/bench_runs/2026-10-08_interim/`, and the
analysis was developed in `notebooks/charlie/bench_tns/` (charlie-scratch
branch). What the product is valid for is in its own `README.md`.

## Recent changes

- 2026-10-20: added, for `noise_source` v0001 from the 2026-10-08 interim bench run.
