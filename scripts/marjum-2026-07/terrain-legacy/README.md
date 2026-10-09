# Historical Marjum terrain analysis scripts

These 62 scripts were copied byte-for-byte from the separate `terrain/`
checkout on 2026-10-08, before removing their superseded copies there.
`source_hashes.json` records each source SHA-256. They cover camera
initialization and validation, transmitter and antenna refinements, LIDAR
and feed checks, and the MCMC pilot, review, and finalization drivers.
They are provenance for older studies, not a maintained pipeline: many
use working-directory paths and caches that have since moved.

The five files used by the published geometry-posterior v0004 model
(`marjum_mcmc_b21.py`, `marjum_mcmc.py`, `marjum_bundle.py`,
`marjum_camera.py`, `marjum_fitio.py`) are byte-identical copies in
`eigsep_terrain/src/eigsep_terrain/marjum_geometry/`, and the v0004 drivers
are in `../geometry/`. The current geometry result and limitations are in
`memos/memo-002-marjum-2026-07-geometry/`; inputs and chain diagnostics
are in the versioned campaign products. Future analysis should start from
those records rather than treating these exploratory scripts as current.

## Recent changes

- 2026-10-08: Pointed to the new homes of the v0004 model files and drivers.
- 2026-10-08: Archived the superseded terrain analysis code with exact
  source hashes so the separate terrain checkout can be reduced without
  losing the method behind historical results.
