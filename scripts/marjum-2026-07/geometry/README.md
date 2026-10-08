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

- 2026-10-08: Moved here from the v0004 product directory and repointed at
  the frozen model copies in `eigsep_terrain`, so v0004 verifies without the
  retired `terrain/` checkout.
