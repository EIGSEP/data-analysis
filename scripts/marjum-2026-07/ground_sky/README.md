# Ground temperature and sky from an assumed beam

Infrastructure for solving for the ground temperature and a sky model from
Marjum spectra, given a beam model. This is the EIGSEP counterpart of
bloom21cm's per-frequency linear inversion. The method is in
`eigsep_sim.design_matrix`; this directory holds the Marjum-specific studies
that drive it.

## The model

At each frequency, every time bin is

    T(t) = Σ_p A_sky[t,p] T_sky[p] + A_gnd[t] T_gnd + T_off

The sky is in Galactic HEALPix pixels. The ground is everything below the
DEM horizon (`curation/horizon_profiles_v0002.npz`, one profile per height
era), at one uniform temperature. `T_off` is an additive offset. Rows are
normalized by the full-sphere beam integral, so sky plus ground weights sum
to one. As a result, a uniform shift of the sky and ground against the
offset is an exact null direction.

Inputs:

- Site: `eigsep_base.const.MARJUM_PASS`.
- Mount: `eigsep_base.rotations.mount_rotation(az_table + az_offset, el, psi)`,
  with `az_offset` and `psi` read from `empirical_raster_v0012`.
- Pointing: `curation/pointing_table.parquet`, using phase C and
  `quality == "ok"` rows only.
- Beam: pluggable. Either the HFSS bowtie or an `empirical_beam.npz`, through
  `HealpixBeam.from_npz`.

## Scripts

| Script | What it does |
|---|---|
| `degeneracy_study.py` | Builds the design matrix on the real geometry. Reports Fisher errors on T_gnd, the offset and the sky mean under four prior choices (sky free or 10 % GSM; offset free or known to 1 K). Then runs a simulate-and-recover check: GSM truth at nside 16, fitted at nside 8, with HFSS or the empirical beam as the true beam. Writes `summary.json`. No measured spectra are used. |

Run with `EIGSEP_CAMPAIGN_ROOT` set. The script takes a few minutes.

## What `summary.json` says (2026-10-01, defaults)

These figures use 120 s static bins and radiometer noise only.

- **With the sky free, T_gnd is never constrained.** That holds even with
  the offset known: within one pointing and era, the ground column is
  constant and trades against a uniform sky shift.
- **With a 10 % GSM sky prior and the offset free**, T_gnd and the offset
  are anticorrelated at ρ ≈ −0.97 to −1.00. σ(T_gnd) at 58.6 / 78.1 /
  101.6 / 148.4 / 199.2 MHz:

  | slice | σ(T_gnd), K |
  |---|---|
  | parked, 30 m only | 100 / 48 / 27 / 16 / 18 |
  | parked, all eras | 21 / 10 / 5 / 3 / 1 |
  | all of phase C | 16 / 7.5 / 3.9 / 1.7 / 0.7 |

  Height changes, which move the horizon, are what break the degeneracy.
- **The tcal v0000 windows alone** (71 bins) are weak. With the offset free,
  σ(T_gnd) is 185 K at 58.6 MHz and 89 K at 148.4 MHz; with a known offset,
  it is 2–11 K.
- **Beam error goes into T_gnd without raising χ².** With HFSS as the true
  beam, the recovered T_gnd is unbiased (≤ 1σ). With the empirical beam as
  the true beam and HFSS assumed, T_gnd is biased by up to 39 K (16σ) at
  78 MHz. χ² per row moves only from about 3.0 to 3.3. The empirical beam
  below the horizon is the ℓmax = 8 extrapolation of a raster that does not
  sample it, so treat the size of this bias as illustrative. The result
  that a good χ² cannot vouch for T_gnd holds regardless.

## Recent changes

- 2026-10-01: new directory; `degeneracy_study.py` and its first `summary.json`.
