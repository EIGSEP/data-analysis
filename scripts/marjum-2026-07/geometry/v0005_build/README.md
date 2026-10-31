# Geometry v0005 build scripts, as run

These scripts built geometry release v0005
(`marjum-2026-07/imgs/fits/v0005_marjum_geometry`,
`derived/geometry_posterior/v0005`) and the v0005 checks that memo 002
quotes. Aaron ran them through an Antigravity (Gemini) session from 2026-10-16
to 2026-10-21, in that session's scratch directory
(`~/.gemini/antigravity-cli/brain/265525af-560e-4e5c-83cb-3c1cc78e2d96/scratch`).
They are byte-identical copies, frozen so that the release has its code.
`CODE_SHA256SUMS` holds their checksums. They have not been reviewed or
tidied, and they still name the scratch directory and
`/home/aparsons/projects/eigsep` in their paths.

| Script | Step |
|---|---|
| `solve_v5_map.py` | joint MAP on the 29-camera network with the new tie points, starting from v0004 |
| `test_polish_and_loo.py`, `test_surrogate_mh.py`, `test_exact_gn_basis.py`, `test_ess_mh_5chains.py` | camera polishing, the surrogate model and the 211-parameter unpacking that the samplers import |
| `build_vol_diffeo.py` | the volume-preserving reparametrisation about the joint mode (`diffeo_vol_system.pkl`) |
| `fast_numba_kernels.py`, `test_fast_vol_mcmc.py` | numba likelihood and gradient kernels and the block/slice steps |
| `test_hmc_vol_mcmc.py` | the five HMC chains (`posterior_v5_hmc_chains.npz`) |
| `test_schur_mcmc.py` | `build_model()`: the exact v0005 posterior, also used by `../geometry_checks.py --geometry ...v0005...` |
| `package_v0005.py` | diagnostics and packaging of the release and of `geometry_checks` v0002, `lidar_constraint` v0003 and `tx_feed_orientation` v0003 (2026-10-17) |
| `write_v5_posterior.py` | rewrote `posterior_samples.npz` and `provisional_geometry.npz` on 2026-10-21 |

The data files they read or write by absolute scratch path are copied,
unchanged, to the gitignored
`marjum-2026-07/derived/geometry_posterior/v0005/inputs/v0005_build/`;
`DATA_SHA256SUMS` holds their checksums. Replace the scratch prefix in a path
with that directory to find a file. `fit_transmitter_v5_clean.npz` is the
same file as the product's `inputs/model_v0005/fit_transmitter.npz`.
Not kept: the session's copy of the Python environment, its exploratory
scripts and image crops, and `test_run_v5/` (trial chains).

Two products were made before the 2026-10-21 rewrite and record inputs that
are no longer on disk. `tx_feed_orientation` v0003 records
`posterior_samples.npz` 66480fab... and `chain_diagnostics.json` 4436d9d0...;
the product now holds 3d3cd030... and 5772d801.... `geometry_checks` v0002
records `provisional_geometry.npz` 8871d36e...; the product now holds
e005f141....

## Recent changes

- 2026-10-26: frozen here from the Antigravity session's scratch directory, which is being cleaned up, so the v0005 release keeps its code and inputs.
