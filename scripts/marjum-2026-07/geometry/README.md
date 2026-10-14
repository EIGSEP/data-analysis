# Marjum geometry posterior drivers

These drivers produced `marjum-2026-07/derived/geometry_posterior/v0004`,
the UTM-frame joint camera, antenna and transmitter fit that memo 002
adopts. They moved here from that product directory on 2026-10-08 and still
read and write it.

| Script | Step |
|---|---|
| `make_product.py` | `prepare` the checksum-pinned inputs and starting states, run each `chain`, and `verify` the sources |
| `audit_run.py` | check the five completed checkpoints against their chain arrays and inputs; writes `run_integrity.json` |
| `finalize_product.py` | publish the higher-density state (chains or re-evaluated v0003) with convergence diagnostics |
| `review_gps.py` | compare old- and UTM-frame camera positions with HEIC GPS fixes; writes `gps_comparison.json` |
| `geometry_checks.py` | held-out refits of v0004 (each antenna label left out, IMG_2210/2211 transmitter labels withheld, HEIC GPS withheld); writes `derived/geometry_checks/vNNNN` |
| `lidar_constraint.py` | box-air LIDAR ranges ray-traced through the DEM from a geometry's antenna, bearings applied on the UTM grid (`--frame legacy` reproduces `lidar_constraint` v0001); writes `derived/lidar_constraint/vNNNN` |
| `tx_feed_orientation.py` | transmitter-feed (rx6/rx1) axes from the IMG_2203 corner picks at a geometry's poses and chain draws (`--geometry v0001` reproduces `tx_feed_orientation` v0001; `--geometry v0004` reports grid and true bearings); writes `derived/tx_feed_orientation/vNNNN` |

Set `EIGSEP_CAMPAIGN_ROOT` to the campaign. The model code is the five
frozen files in `eigsep_terrain/src/eigsep_terrain/marjum_geometry/`, found
through the installed `eigsep_terrain` (put a worktree's `src` first on
`PYTHONPATH` to use another checkout). `input_manifest.json` still records
those files under their old `terrain/` paths; `PINS.json` there maps each to
its copy, and `make_product.resolve` applies that map to manifest keys and
to the absolute paths stored in the chain checkpoints. The manifest's own
hash for `make_product.py` is the original driver, checked from the
top-level repository at commit `dd37b15`. The package-commit check now
compares the pinned `eigsep_terrain` files at the recorded commit, so a
later commit that leaves them unchanged still verifies.

To check the published run without overwriting it:

```sh
export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
python audit_run.py --out /tmp/run_integrity.json
cmp /tmp/run_integrity.json $EIGSEP_CAMPAIGN_ROOT/derived/geometry_posterior/v0004/run_integrity.json
```

`finalize_product.py` refuses to replace a published fit, and resuming a
chain would see new absolute model paths in its signature, so rerunning
those steps means a new release version.

## Recent changes

- 2026-10-10: `tx_feed_orientation.py`, ported from the retired terrain script; it reproduces v0001 to 3e-5 deg per draw and re-solves the feed at release v0004, reporting true bearings (the raster axes are UTM grid axes, 1.52 deg from true north).
- 2026-10-09: added `geometry_checks.py` and `lidar_constraint.py`, the independent checks that give v0004 its bounds in memo 002 (`geometry_checks` v0001, `lidar_constraint` v0002). The LIDAR port applies true bearings as UTM grid bearings, which the retired terrain version did not.
- 2026-10-08: The drivers read the model inputs from the v0004 product's own `inputs/model_v0002/` copy instead of the unpublished `geometry_posterior/v0002/inputs`; recorded paths to the old directory are mapped there, and the frozen DEM is checked against `dem/v0001`.
- 2026-10-08: Moved here from the v0004 product directory and repointed at
  the frozen model copies in `eigsep_terrain`, so v0004 verifies without the
  retired `terrain/` checkout.
