# Ground temperature and sky from an assumed beam

Infrastructure for solving for the ground temperature and a sky model from
Marjum spectra, given a beam model. This is the EIGSEP counterpart of
bloom21cm's per-frequency linear inversion. The method is in
`eigsep_sim.design_matrix`; this directory holds the Marjum-specific
pipeline (calibrated spectra → bins → fit) and the studies that drive it.

## The model

At each frequency, every time bin is

    T(t) = Σ_p A_sky[t,p] T_sky[p] + A_gnd[t] T_gnd + T_off[regime(t)]

The sky is in Galactic HEALPix pixels. For data, `T_sky = a·GSM + δ`: the
amplitude `a` is free (it absorbs the front-end loss, since `tcal`
temperatures are at plane P, and any GSM scale error) and only `δ` has a
prior. The ground is everything below the
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
| `bin_spectra.py` | Builds `derived/ground_sky/binned_vNNNN`: `tcal` box-air temperatures, `flags@v2` bits 0–7, transmitter-comb channels dropped in its span, averaged over ±2 channels at each HFSS beam frequency and into static 120 s bins of one height era and receiver regime. Noise is the in-bin scatter. |
| `fit_ground_sky.py` | Builds `derived/ground_sky/fit_vNNNN` from a binned product: per-frequency fit of `a`, T_gnd and one offset per receiver regime, on night bins only (Sun below −10°; it is not modelled). Reports which columns the data actually constrain, the per-regime pedestal ⟨f_gnd⟩·T_gnd + T_off that is constrained even when its parts are not, and χ² on held-out alternating 30-min blocks. |
| `common.py` | Paths, provenance, and per-file height era (`mode_table.jsonl`, which fills the pointing table's blank eras) and receiver regime (`cal_windows.jsonl`). |
| `degeneracy_study.py` | Builds the design matrix on the real geometry. Reports Fisher errors on T_gnd, the offset and the sky mean under four prior choices (sky free or 10 % GSM; offset free or known to 1 K). Then runs a simulate-and-recover check: GSM truth at nside 16, fitted at nside 8, with HFSS or the empirical beam as the true beam. Writes `summary.json`. No measured spectra are used. |

Run with `EIGSEP_CAMPAIGN_ROOT` set. Each takes a few minutes; `--out-dir`
writes a trial anywhere instead of a new product version.

## First fit to data (2026-10-01; `binned_v0001`, HFSS beam)

Trial runs, not yet a product: `fit_v0001` waits for eigsep_sim#6.

- **The calibrated night data cannot measure T_gnd.** `tcal@v0002` covers
  07-17 04:11–16:22 (zenith, 87.5 m, rx-A) and 07-18 01:27–02:56 (mostly
  near nadir, 91 m, rx-B). The second window is all at dusk, with the Sun
  above −10°, so the night fit is one pointing, one height and one regime:
  165 bins. There, T_gnd and the offset are exactly degenerate (as
  `degeneracy_study.py` predicts). Only their pedestal is determined:
  ⟨f_gnd⟩·T_gnd + T_off ≈ +50 to +80 K across 110–165 MHz, and −700 to
  −60 K below 75 MHz. Between 125 and 207 MHz, the ±1° wobble in el
  formally separates the two, giving T_gnd from −1260 to +1510 K with
  quoted errors of 60–1100 K. That is fitted model error, not a
  measurement.
- **The data vary about twice as much as GSM through the HFSS beam
  predicts.** The template amplitude is a = 1.3 at 51 MHz, 1.8–2.3 from 62
  to 210 MHz and 2.6–3.0 at 211–227 MHz, each ±0.02 statistically. A direct
  forward check (GSM, T_gnd = 300 K, no offset) gives data/model of 1.1–1.9
  in both receiver regimes, peaking near 78 MHz, and steady through the
  night. Candidates: the `tcal` kelvin scale (it assumes the nameplate
  noise-source excess, 917 K, and corrects mismatch most where the antenna
  is worst matched), or a sky fraction that the HFSS beam underestimates.
  The fit cannot tell these apart.
- **The model is not yet adequate at the noise level.** χ² per bin is
  70–280 in clean bands. Residuals are about 10× the in-bin scatter, which
  is itself 2–6× radiometric. Held-out χ² is 2–5× the training χ², so the
  10 % GSM prior on δ lets the sky absorb structure it cannot predict.
- **The Sun must be cut or modelled.** It lifts data/model at 199 MHz from
  1.06 to 1.80 within an hour of rising.

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
- **The tcal v0000 windows alone** (71 bins) are weak, and that count
  includes the dusk rx-B bins that the Sun cut removes. With the offset free,
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

- 2026-10-01: added the data pipeline (`bin_spectra.py`, `fit_ground_sky.py`,
  `common.py`) and the first fit's findings: night calibrated data fix the sky
  scale but not T_gnd.
- 2026-10-01: new directory; `degeneracy_study.py` and its first `summary.json`.
