# Ground temperature and sky from an assumed beam

Infrastructure for solving for the ground temperature and a sky model from
Marjum spectra, given a beam model. This is the EIGSEP counterpart of
bloom21cm's per-frequency linear inversion. The method is in
`eigsep_sim.design_matrix`; this directory holds the Marjum-specific
pipeline (raw spectra → bins → calibration → fit) and the studies that drive it.

Binning and temperature calibration are separate steps. `bin_raw.py` bins raw
correlator power and nothing else. `common.calibrate` then puts a binned
product in kelvin with any `tcal` version, at each bin's time and on each
channel, so a new calibration needs no new bins.

## The model

At each frequency, every time bin is

    T(t) = Σ_p A_sky[t,p] T_sky[p] + A_gnd[t] T_gnd + T_off[regime(t)]
           [+ A_sun[t] S_sun]

The optional Sun term is its flux S_sun (SFU) times the beam gain toward it
and its terrain visibility (`sun.py`). The sky is in Galactic HEALPix pixels. For data, `T_sky = a·GSM + δ`: the
amplitude `a` is free (it absorbs the front-end loss, since `tcal`
temperatures are at plane P, and any GSM scale error) and only `δ` has a
prior. The ground is everything below the
DEM horizon (`curation/horizon_profiles_v0003.npz`, one profile per height
era), at one uniform temperature. `T_off` is an additive offset. Rows are
normalized by the full-sphere beam integral, so sky plus ground weights sum
to one. As a result, a uniform shift of the sky and ground against the
offset is an exact null direction.

Inputs:

- Site: `eigsep_base.const.MARJUM_PASS`.
- Mount: `common.beam_rotation(az_table, el)`, the beam fit's own rotation
  `Rz(psi) Ry(tilt) Rx(el + el_zero) Rz(az + az_offset + az_zero)` (built by
  `eigsep_data.beam_mapping.TxGeometryModel`), with the az offset, the fitted az-
  and el-zero deltas, the fitted along-axle tilt and `psi` read from the beam product
  named once in `common.BEAM_PRODUCT` (`empirical_raster_v0017`, the reference beam of
  memo 005: pointing table v5, geometry release v0005, tilt +4.74°); see
  `common.beam_geometry`. Applying them all puts the sky in the frame the beam was
  fitted in.
- Pointing: `curation/pointing_table.parquet` at `common.POINTING` (v5, whose
  azimuth is left-handed about the boresight, as the beam was fitted), using phase C
  and `quality == "ok"` rows only.
- Horizon bearings: the horizon-profile bearings are UTM 12N grid
  bearings (the DEM's frame), 1.52° from true at the site. The fit now
  rotates them to true bearings; the first trials did not.
- Beam: pluggable. Either the HFSS bowtie or `common.BEAM_PRODUCT`'s `empirical_beam.npz`, through
  `HealpixBeam.from_npz`.

## Scripts

| Script | What it does |
|---|---|
| `fit_ground_sky.py` | Builds `derived/ground_sky/fit_vNNNN` from a raw binned product, calibrated with `--tcal` (default v0004; bins with tcal, pointing, a horizon-profile era and at least `--min-rows` rows): per-frequency fit of `a`, T_gnd and one offset per receiver regime. `--sun model` adds the Sun as a column; `--sun cut-flat` (Sun below −10°) or `cut-terrain` (Sun behind the ridge) instead chooses which bins are fitted. The model and residual are evaluated on every bin either way, so cut bins are out-of-sample predictions. Reports which columns the data actually constrain, the per-regime pedestal ⟨f_gnd⟩·T_gnd + T_off that is constrained even when its parts are not, and χ² on held-out alternating 30-min blocks. |
| `sun.py` | The Sun from the antenna: ephemeris; the DEM traced along the Sun's bearing (horizon elevation and edge distance); knife-edge diffraction at that edge; the K-per-SFU column through the beam; the UTM grid-to-true bearing rotation. |
| `sun_events.py` | Tests whether the out-of-sample residual of a `cut-flat` fit around the 07-17 terrain sunrise is the Sun: a local line plus S_sun × Sun column on transmitter-off bins only, a flat-horizon null at 12:26, and a timing scan. |
| `sun_bursts.py` | Builds `derived/ground_sky/bursts_vNNNN`: the solar bursts of `curation/solar_bursts_v0001.json` (e-CALLISTO) in raw, unflagged box-air, with excess per band and sub-band, cutouts, and a random-daytime null. |
| `impulsive_scan.py` | Builds `impulsive_vNNNN`: a blind scan of raw box-air for impulsive events, with rates by whether the Sun is visible, behind the ridge, or down. |
| `sun_transitions.py` | Builds `transitions_vNNNN`: the raw context, minute by minute, of every terrain sunrise and sunset of phase C, plus a raw-power line + Sun fit with a timing scan, and when the ground under the antenna crosses. |
| `sun_profile.py` | Builds `sun_profile_vNNNN`: the Sun column through the beam, minute by minute over phase C. |
| `raster_sun.py` | Builds `raster_sun_vNNNN`: regression of raw power during the 07-17 raster on sky, ground, transmitter and Sun columns, with fake Suns as the null. Once per beam. |
| `joint_fit.py` | Builds `joint_vNNNN`: the calibrated night fit tied across frequency (one T_gnd, a smooth offset) by summing per-frequency marginal likelihoods; T_gnd's stability against the offset order is the test. |
| `sky_cal.py` | Builds `skycal_vNNNN`: parked night stretches calibrated on the sky (gain from GSM's drift through the beam), and the receiver/ground separation across stretches of different ground fraction. |
| `raster_sky.py` | Builds `raster_sky_vNNNN`: gain-free regression of raw power in moving windows on GSM, ground, constant, transmitter, Sun and drift columns; T_gnd as the ground/GSM coefficient ratio; low-order sky corrections (Y_lm) tested against ground-fixed ones on held-out blocks; optionally a specularly reflected sky term. |
| `ground_excess.py` | Builds `ground_excess_vNNNN`: the rotating-antenna regression with the ground split into regions (a band below the terrain horizon against the deeper ground, or azimuth sectors), each region's brightness in GSM kelvin; locates the low-frequency ground excess. |
| `bin_raw.py` | Builds `derived/ground_sky/raw_binned_vNNNN`: raw box-air power over all of phase C (not only where `tcal` exists), at each HFSS beam frequency in the six channels midway between transmitter teeth, `flags@v3` (any bit but 9 and 10) and box-air's self-EMI masked, in bins of up to 60 s that close when the pointing moves 1°. Stores pointing (filled across a gap only where both sides agree) and its spread within the bin, the means of six housekeeping streams (system current, RF-switch and load temperatures, elevation-IMU accelerations), transmitter on-fraction, and each offset channel's own mean and sample count (`yc`, `nc`) for `common.calibrate`. No calibration is applied or stored. It reads `common.POINTING` by default (`--pointing`). `--delay-filter-ns` low-passes each integration in delay before binning (`delay_filter.py`); `--union-mask-only` is its control. |
| `delay_filter.py` | The delay low-pass `bin_raw.py --delay-filter-ns` applies: each integration's fluctuation about a 600 s running mean, fitted with DPSS vectors spanning \|delay\| < τ in two sub-bands (50–85, 110–250 MHz) under one flag mask per 5 s block (the block's union); blocks over half flagged are dropped. Memo 007 § In flight has the motivation and the test. |
| `global_fit.py` | Builds `global_vNNNN` from a raw binned product: one model of every bin of the chosen receiver regimes (default rx-A) per frequency, gain × (GSM + T_gnd × ground fraction + low-order sky + transmitter + Sun + offset). The gain is smooth in time (log gain piecewise linear, knots every 3 h, a 5 % prior per knot step) or free per window; offsets per regime or per window; Sun fitted or cut. Gates drop windows whose level or sky-following fails. Held-out blocks score each configuration; gain × `tcal` scale gives the GSM amplitude a; fits where the gain collapses are flagged `diverged`. Beam columns are cached under a key of everything that defines them. `--beam-lmax L` adds a body-frame Y_lm correction to the beam (`beam_corr.py`). `--gain-regressors` adds bin columns (e.g. housekeeping means) to log gain; `--regressor-shift-s` is their time-shifted control. `--fast-knot-s` adds a fast gain and tilt shared across frequencies, only within fixed-pointing runs, scored on held-out frequencies; `--fast-max-el-deg` loosens the elevation limit on a run. `--swing-max-el-std` / `--swing-max-el-dev` drop bins where the box swings. `--fit-eras` / `--fit-utc` fit a subset of the bins with the same columns. `--mock-y` fits mock data from `inject_mock.py` instead. `--split-streams` splits a bin column into 3-h slow and fast parts for `--gain-regressors`. `--max-tx-frac` drops bins in transmitter episodes; `--exclude-utc` drops a time range (e.g. the 07-16 23:50-00:02 solar flare). `--offset-prior` and `--gain-anchor` put priors on the offsets and on the GSM amplitude (per-frequency fit only). `--only-mhz` reuses the full grid's column cache. `--ground diffuse` replaces the constant ground temperature with diffusely reflected sky, (1 − R) T_phys + R × the cos-weighted open-sky GSM mean, with `--diffuse-lst-shift-h` as a wrong-sky control; `--ground beam` (below-horizon beam response really on the sky) and `--ground pickup` (sky by a path that does not turn with the antenna) are the alternatives at the same freedom, and `--ground hk` with `--hk`/`--hk-stream` tests a housekeeping stream (a daily systematic) in the same slot; `--hk-extra` adds housekeeping streams as extra columns beside any ground choice, for joint sky + temperature fits. `--orient add|mult` adds elevation harmonics, as kelvin or as a fractional change of the model with tilt, to test an orientation-dependent term against the ground column. `--ground-group window|era` gives each fitted window or height era its own ground temperature, so differences between groups cannot be carried by a sky column (per window is degenerate on phase C, whose windows are mostly parked). `--ground-level-prior MEAN SIGMA` puts a Gaussian prior on the free ground level of `--ground diffuse-free`, which otherwise trades with the receiver offset and runs off. `--night-only`, `--t-range` and `--t-exclude` restrict the fitted bins (the Sun below the horizon; a UTC range; UTC spans left out); `--gain-segment-by-era` gives each height era its own gain curve. Bins after the 07-18 01:41:38 elevation-drive failure are kept (rx-B only). |
| `inject_mock.py` | Mock binned data for injection tests of `global_fit.py --mock-y`: a fit's own prediction (so its T_gnd and gain are the truth) plus radiometer noise, optionally times another fit's real residuals rolled in time; `--fast-gain-cm` multiplies in a frequency-flat fast gain (from `fast_gain_cm.py`), and `--resid-key` reads a shared fast-gain fit's residuals. |
| `fast_gain_cm.py` | A frequency-flat fast log gain for `inject_mock.py --fast-gain-cm`: a fit's knot log gain averaged over its converged frequencies in a band, minus a 3-h piecewise-linear trend per gain segment. |
| `tgnd_table.py` | T_gnd spectrum from one global fit, with an error per frequency from the injection refits (rms of recovered minus true), and excluded or diverged frequencies marked. |
| `hk_streams.py` | Per-integration housekeeping streams (thermistors, currents, IMU, motors, pot, switch state) from the correlator metadata, saved for `gain_drivers.py`. |
| `gain_drivers.py` | What drives the sub-hour gain: a fit's 20-min common-mode log gain (3-h detrended) against every housekeeping stream on the same knots, in two disjoint stretches, plus event-triggered residuals around cal cycles, VNA and motor moves. |
| `cal_gain.py` | The receiver gain after the RF switch from the tcal@v0003 load and noise-source visits, against a global fit's knot gain, the sky drift and the switch thermistor. |
| `cal_gain_mock.py` | Binned data divided by that calibration gain, for `global_fit.py --mock-y`: fixes the gain to the calibration's. |
| `cal_gain_30m.py` | The same receiver-gain proxy (load power over load temperature) from raw RFAMB visits in the 30 m era, which tcal does not cover, against a fit's knot gain; drops contaminated visits and records the raw files' SHA-256. |
| `beam_corr.py` | Columns for a low-order beam correction B0 (1 + Σ c_lm Y_lm) in the antenna body frame: per l,m a sky part (GSM above the terrain horizon) and a ground part (solid angle below it), normalized like the design matrix. At l = 0 it reproduces C_GSM and C_gnd, which `global_fit.py` prints as a check. |
| `plot_global.py` | Figures for a `global_vNNNN` directory: fitted parameters against frequency per configuration, and residuals against time. |
| `common.py` | Paths, provenance, and per-file height era (`mode_table.jsonl`, which fills the pointing table's blank eras) and receiver regime (`cal_windows.jsonl`). |
| `degeneracy_study.py` | Builds the design matrix on the real geometry. Reports Fisher errors on T_gnd, the offset and the sky mean under four prior choices (sky free or 10 % GSM; offset free or known to 1 K). Then runs a simulate-and-recover check: GSM truth at nside 16, fitted at nside 8, with HFSS or the empirical beam as the true beam. Writes `summary.json`. No measured spectra are used. |

Run with `EIGSEP_CAMPAIGN_ROOT` set. Each takes a few minutes; `--out-dir`
writes a trial anywhere instead of a new product version.

## Fit to data and the Sun (2026-10-06; `flags@v3`, `tcal@v0003`, HFSS beam)

**The reference for these results is now memo 008**
(`memos/memo-008-marjum-2026-07-image-domain/`), built from the published
products `derived/ground_sky/{binned_v0001,binned_v0002,fit_v0001..v0004,sun_events_v0001}`.
It also compares them with concurrent RSTN solar monitoring. The summary below
is kept for orientation.

These are trial runs written with `--out-dir`, not yet products. Coverage is
the same as tcal v0002: 07-17 04:11–16:22 (87.5 m, rx-A, parked) and
07-18 01:27–02:56 (91 m, rx-B).

**When the Sun is visible.** The ridges are high and close: 39° at 350 m
toward sunrise and 17° at 170 m toward sunset (DEM v0001, traced along the
Sun's bearing). The Sun clears the ridge at **07-17 15:55 UTC** (87.5 m),
3.5 h after it crosses 0° altitude (12:26). It drops behind the ridge at
**07-18 01:20** (91 m), 1.5 h before 0°. The rx-B window starts after that,
so the calibrated data see the direct Sun only from 15:55 to 16:22 on 07-17.
At these wavelengths the ridge edge diffracts over √(λd/2) ≈ 2–6°, so the
Sun ramps in over 10–30 min, slower at low frequency. The 30 m era has its
own times (rise 16:25, set 00:01). The first trials' "the Sun lifts 199 MHz
within an hour of rising" cannot be the direct Sun: that hour is behind
the ridge.

**Ground temperature.** No change from the first trials. Only one pointing,
one height and one regime survive the night cut, and T_gnd trades exactly
against the offset. Where the solver calls T_gnd constrained (125–207 MHz),
it ranges from −6700 to +1400 K, which is fitted model error. The GSM
amplitude is a ≈ 1.3–3.0, χ² per bin is ~75–300, and held-out χ² is 1.5–5×
the training χ². Fitting the Sun-behind-ridge morning bins as well
(`cut-terrain`) raises the median χ² only from 119 to 136.

**The Sun around 15:55** (`sun_events.py` on a 60 s `cut-flat` fit):

- **Method.** The residual from −85 to +27 min is fitted locally, as a line
  plus S_sun × the predicted Sun column. Only transmitter-off bins are fitted
  (the transmitter is on 15:36–16:14 with gaps, per the box-gnd
  transitions). That leaves 62 bins, of which **only 7 have the Sun above the
  ridge** (16:14–16:22).
- **Spectrum.** At 110–235 MHz, S_sun rises smoothly: 7 ± 1 SFU (109 MHz),
  14 ± 1 (148), 30 ± 2 (199), 22 ± 1 (219). Errors are scaled by
  √(χ²/dof). Below 90 MHz the result is unstable: the residual there drifts
  ~70 K across the window, and the null below fails.
- **Flat-horizon null.** The same local fit around 12:26, with the Sun
  visible from 0°, gives −1 to −6 SFU at 110–235 MHz. No step appears when
  the Sun rises behind the ridge, as the terrain predicts.
- **Timing.** χ² against a shift of the Sun column is lowest at τ = −4 min,
  and within +5 of that minimum from −8 to +3 min. Shifts of 20 min or more
  earlier are strongly disfavoured.
- **Kelvin scale.** The night fit needs a ≈ 2: the data vary twice as much
  as GSM through HFSS predicts. If that factor lies in the kelvin scale or
  the beam, it scales S_sun too, and S/a is 3 SFU (109 MHz), 8 (148) and
  13 (199). Either reading is in the quiet-to-active Sun range.
- **Not separated.** Anything else that changes during 16:14–16:22 is not
  separated, for example the box or the ground under the antenna coming into
  sunlight. All 7 post-ridge bins come right after the transmitter switches
  off. Transmitter-on bins sit a further +10 to +170 K above the Sun model.
  So this is an excess that coincides in time, spectrum and geometry with the
  predicted sunrise. It is not an identification of the Sun.
- **Global fit not usable.** A global fit with the Sun column on every bin
  (`--sun model`) fits the transmitter-on bins, and its held-out χ² reaches
  10⁵.

**Next checks.**

- Compare with external solar radio monitors at 150–245 MHz for 07-17
  ~16 UTC.
- Repeat at the uncalibrated sunsets (07-16 and 07-17 00:01 at 30 m;
  07-18 01:20 at 91 m), using the knife-edge ramp shape, not a level. In
  raw-data quick looks, the step at 07-17 00:01 is abrupt and rippled below
  100 MHz, with a matching jump on box-gnd. That is a system change, not
  the Sun.

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

## Phase-C model, first trials (2026-10-09; not yet a product)

`bin_raw.py` → `global_fit.py` (run before the v0013 / geometry v0004 port: beam-v0012
mount offsets, horizon profiles v0002), HFSS beam, rx-A only (07-15 to 07-17 19:41;
6,088 bins at 30, 87.5 and 91 m), smooth gain with 3 h knots.

- **What the data support.** Free per-window gains with free offsets are not
  identifiable: where the model's variation misses the data, the fit sends a
  window's gain to zero and its offset to infinity. Per-window rx-A gains
  nonetheless stayed within ±4 % over three days, so the smooth gain is
  justified. rx-B and the rx-transition span (which holds the raster) need
  their own offsets and gains and are not stable yet; the raster stays with
  `raster_sky.py`. 07-15 14:49-16:56 is 5-10x low in level and is gated out.
- **T_gnd** (GSM kelvin) falls from about 420 K at 117 MHz to 210-250 K at
  195-203 MHz, the same with the empirical beam (within 10 K) and close to the
  rotating-antenna values of memo 008. It moves by 50-100 K with the gain
  knot spacing (1.5, 3, 6 h), and l <= 2 sky corrections pull it to 0-280 K,
  so it is not yet robust. Below 85 MHz it is thousands of kelvin, the
  low-frequency excess again; 211-227 MHz diverge.
- **a = gain x tcal scale** is 2.0-2.6 at 117-195 MHz, as memo 008's
  calibrated fit found. T_gnd near the physical ground temperature in GSM
  kelvin is consistent with GSM being right and tcal's kelvin scale about
  twice high (memo 004's open T_NS question); that is not tested here.
- **Fit quality.** rms 1.9-2.6 % of power at 117-203 MHz, held out 2.0-2.6 %,
  about 100 radiometer sigma. The 07-17 87.5 m night fits to about 1 %; the
  07-16 30 m night (box-air self-EMI on) leaves slow ±5-9 % waves.

## Recent changes

- 2026-10-31: the reference beam is `empirical_raster_v0017` (memo 005), and every script reads pointing table v5 (`common.POINTING`), the left-handed azimuth that beam was fitted on. Its mount has an along-axle tilt (+4.74°), which `eigsep_base.mount_rotation` cannot express, so every rotation now comes from `common.beam_rotation` (the beam fit's own `TxGeometryModel`), and `common.mount_offsets` is gone; manifests record `common.beam_geometry`. `sun_events.py` and `joint_fit.py` rebuild the rotation from the fit they read, so older fits (no tilt) still reproduce. `sky_cal.py` defaults to tcal v0004.
- 2026-10-31: binning and temperature calibration are separate steps. `bin_raw.py` no longer reads `tcal`; it stores each offset channel's bin mean, and `common.calibrate` applies a `tcal` version (default v0004, the bench-measured noise source) to them afterwards. `fit_ground_sky.py` now reads a raw binned product and calibrates it; `global_fit.py`, `cal_gain.py` and `cal_gain_mock.py` take `--tcal`. `bin_spectra.py`, which calibrated rows before binning, is retired; its products (`binned_v0001`–`v0006`) are still read as they are. `bin_raw.py` takes `--pointing` (default v4: the reference beam's az offset is still right-handed). `raw_binned_v0005` is the first product built this way.
- 2026-10-23: every script here reads `curation/horizon_profiles_v0005` (traced from geometry release v0005's antenna, so the transmitter comes from v0005 too) and `pointing@v4`; the 91 m re-raise (`~91m-reraise`) has its own horizon profile, so a window after the re-raise is fitted with `--era 91m-reraise`. The empirical beam (`empirical_raster_v0013`) is still the one fitted on geometry v0004.
- 2026-10-22: fixed-height analyses use only rows at that height. `common.rows_at_height` keeps the rows
  whose mode-table era has the requested horizon profile; `raster_sky.py`, `ground_excess.py` and
  `raster_sun.py` apply it to `--era` and record the eras their window held, and refuse a window with
  none (the 07-17 00:21-01:08 "dusk at 30 m" window is on the ground in the rebuilt mode table).
  `sky_cal.py` gives each stretch a height and skips one with no rows at it: the 07-15 night is on
  the ground, which has no horizon profile yet.
- 2026-10-22: every script here reads `pointing@v3` (era and phase by source file, the nadir-wrap fix,
  elevation-glitch rows `suspect`) instead of v2.0, and `degeneracy_study.py` takes height eras from the
  mode table by source file like the others, not from the pointing table's own column (v3's was built
  before the 2026-10-18 mode-table rebuild). Products made from here on use the rebuilt height eras.
- 2026-10-21: `sky_cal.py --fit-from DATE=UTC` fits one night's stretch only from a later time, still
  binning it from its start. The 07-15 night's raw power is a few percent of its later level before
  about 04:45, so `skycal_v0005` fits that night from 05:12 (memo 008).
- 2026-10-21: the two lines of `global_fit.py` work (daily-term and receiver-gain tests from branch
  `cal-gain-daily-term-v0004`; the shared fast gain from `sky-terrain-model-v0004`) are merged. Both had
  built `--gain-anchor`, `--mock-y` and fast-gain injection; one implementation is kept: priors are rows
  P p - b, with `--ground-level-prior`'s named-coefficient priors first, so each branch's runs build
  the same penalty as before. Bins after the 07-18 elevation failure are kept, as on the
  shared-fast-gain line (they are rx-B; memo 008's fits are rx-A).
- 2026-10-21: `degeneracy_study.py` reads its horizons from `common.HORIZON_PROFILES`; it had its own
  hard-coded `horizon_profiles_v0003`, so the 2026-10-19 rerun described as v0004 was still on v0003.
  `sun_transitions.py` defaults to the v0004 products (`sun_events_v0004`, `fit_v0015`).
- 2026-10-20: `global_fit.py --fast-solver gn` (Levenberg-Marquardt on the shared fast gain and every training
  frequency's parameters at once, Schur complement on the shared block) and `--fast-drop-collapsed`. The
  alternation never converged; the joint fit reaches steps below ~1e-6 once training frequencies that collapse
  during it (a_tcal < 0.5 or diverged: 203 MHz, and 74 or 51 MHz in full-band training) are dropped by rule. At
  convergence the linear shared gain trained on 117-203 and on 51-203 MHz agree within 28 % at 117-164 MHz (148 MHz
  579 and 695 K) and within a factor 1.7 at 172-195 MHz: the 4.4x band coupling reported from the alternation was
  an artefact of non-convergence and collapsed training frequencies. The quadratic basis drops 187-203 MHz from its
  narrow training and is unstable there.
- 2026-10-20: `global_fit.py --fast-tol`. Run to 450 iterations, the shared-gain alternation still takes steps of
  2-4e-4 in log gain: it drifts rather than converges. High-band T_gnd moves at most a few per cent for the
  narrow linear and full quadratic fits (16 % at 117-125 MHz for the narrow quadratic); the split fit's T_gnd at
  74-82 MHz changes by over 1000 K. The band-coupling conclusion holds (ratios 4.4 linear, 2.6 quadratic).
- 2026-10-20: `global_fit.py --fast-freq-basis` and `--fast-train`. The shared fast gain's frequency dependence:
  linear (gain + tilt), quadratic in log f, or split (its own gain and tilt below and above 100 MHz). Split reproduces
  the 117-203-trained fit exactly at high frequency and gives the best held-out rms at 51-82 MHz (2.6-3.7 %). A
  linear or quadratic dependence across 51-203 MHz still pulls high-band T_gnd down (quadratic: 117 MHz 654 -> 297 K,
  125 MHz 735 -> 331 K) and worsens its held-out rms. With three injection rolls per truth and training, the
  quadratic's narrow-vs-full difference exceeds the injection scatter at 117-156 and 188-195 MHz (e.g. 140 MHz 231 K
  against <= 59 K) and is within it at 164-180 MHz. The fast term is not one gain smooth in frequency across both bands.
- 2026-10-20: shared fast gain trained on 51-203 MHz instead of 117-203. The gain grows to 7.4 % rms (tilt 4.1 %)
  and T_gnd at 117-195 MHz drops 1.2-3.3x (148 MHz 541 -> 311 K; 117 MHz 823 -> 250 K); 203 MHz diverges. Three
  injection rolls from each training's own fit, refitted with both trainings: at 117-195 MHz either training
  recovers either truth within 25-150 K rms, so the 100-570 K difference between the two real fits at 117-180 MHz
  is not noise: T_gnd depends on which frequencies define the shared gain. At 51-82 MHz each training recovers its
  own truth best. Not adopted.
- 2026-10-20: `global_fit.py --fast-mode all` (with `--fast-tilt-clip`): one fast gain and tilt shared across
  frequencies, knots every 20 min over all of phase C, high-passed against the per-frequency 3-h gain. On terrain
  v0004 it removes the collapse at 51-74, 188-195 and 234 MHz (a_tcal 1.9-3.4); held-out rms at 117-180 MHz
  matches per-frequency 20-min knots and is better at 188-195 MHz. T_gnd at 117-203 MHz (148 MHz 541 K; 188 and
  195 MHz 164 K) is the same for fast priors 0.02-0.1, but needs ~150 ALS iterations to settle (530 K at 117 MHz
  after 12, 823 K after 150). Injection (one roll each): a mock with the shared gain is recovered by the shared
  model within 60 K at 133-195 MHz (125-250 K off at 117 and 203 MHz); a mock with per-frequency 20-min gains,
  refitted with the shared model, lands within 120 K of its truth at 140-180 MHz but 330 K high at 125 MHz.
  `inject_mock.py --resid-key resid_frac_fast` builds mocks from a shared-gain fit.
- 2026-10-19: `cal_gain_30m.py`. In the 30 m era the only clean pair of load visits, 07-16 01:41 and 02:24 (the 01:14 visit is contaminated at 6 frequencies), shows the gain after the switch changing by -0.9 to +0.5 % at 117-203 MHz while a 20-min fit's gain changes by up to 7 % (8-19 % at 51-82 MHz), with the antenna moving.
- 2026-10-19: every script here now reads `curation/horizon_profiles_v0004` and `derived/dem/v0004`, which extends the DEM 6 km south; v0003 cut the southern horizon short by up to 5.5 deg. Products made from here on carry the v0004 terrain.
- 2026-10-18: `cal_gain.py`, `cal_gain_mock.py`. On 07-17 04:10-16:22 (87.5 m, the only stretch with dense
  calibration cycles) the gain after the RF switch varies about 1 % (load proxy) to 3 % (noise-source proxy, which
  includes the source's drift) and follows the switch thermistor (load proxy -1.2 %/K); after a 3-h detrend it has
  no structure that follows the sky. A global fit's 20-min knot gain moves 8-50 % there and tracks the sky. Fixing
  the gain to either proxy leaves 1.7-3 % residuals, too rough to test the daily term.
- 2026-10-18: `global_fit.py --gain-anchor A SIGMA` and `--mock-y` (ported from the `sky-terrain-model` branch), and `inject_mock.py` with a `--fast-gain-cm` option fed by the new `fast_gain_cm.py`: test whether a sub-hour gain could make the low-frequency daily term, by refitting with faster gain knots and by injecting a frequency-flat fast gain into mocks.
- 2026-10-18: `global_fit.py --t-exclude START END ...`: leave out UTC spans, such as the 07-16 look-down park that dominates the misfit in the second half of phase C.
- 2026-10-18: `global_fit.py --night-only` (Sun below the geometric horizon) and `--gain-segment-by-era` (a separate gain curve per height era), to find out why the second half of phase C fits badly at low frequency.
- 2026-10-18: `global_fit.py --t-range START END`: fit only bins in a UTC time range, to repeat a model comparison on disjoint halves of phase C.
- 2026-10-18: `global_fit.py --ground-level-prior MEAN SIGMA`: a Gaussian prior on the free ground level, so a sky column with a free level converges and the real and LST-shifted skies can be compared at equal freedom.
- 2026-10-18: `global_fit.py --split-streams`, `inject_mock.py --fast-gain-from`, `gain_drivers.py` JSON output.
  Slow and fast switch-temperature terms on 3-h knots give the same low T_gnd as one term (148 MHz 297 K). A mock
  with that model's gain times the temperature-free fast gain of the 20-min fit is refitted by it at 14 K (true
  297) and by 20-min knots at 504 K; at 117-180 MHz 20-min knots stay within about 100 K. The unexplained fast gain
  is what breaks slow knots.
- 2026-10-18: `hk_streams.py`, `gain_drivers.py`. 20-min knots plus the switch-temperature regressor leave T_gnd
  within 8 % of 20-min knots alone at 117-203 MHz, and refit a 20-min mock identically (within 2 %): no narrower
  scatter, no bias, so the regressor is redundant there. The fast (sub-3-h) common-mode gain, 2.9-3.2 % rms and
  consistent between disjoint frequency halves (r 0.88-0.90), follows the switch thermistor at -3.2 %/K in both
  disjoint stretches (07-15/16 and 07-17; R2 0.29-0.46), about twice the 3-h fit's slow coefficient. No other
  stream (current, IMU, motors, cal or VNA cycles, flags) correlates consistently in both stretches.
- 2026-10-18: `global_fit.py --exclude-utc`. Tried a gain that cannot follow the diurnal sky: 3-h knots plus the
  RF-switch temperature as a regressor (coefficient -0.02 to -0.05 per sigma, smooth across the band). It stops the
  collapse at 51-74 and 187-195 MHz but lowers T_gnd at 117-180 MHz 2-4x and triples chi2/dof. Injection: a mock
  with the 20-min gain, refitted this way, reproduces the real low values (e.g. 148 MHz 260 vs 287 K), and does so
  at 66-74 and 187-195 MHz even though that mock's truth there is the collapsed solution; a mock with the
  temperature gain, refitted with 20-min knots, stays low (148 MHz 475 K is the one exception), not the real
  472-734 K. So the data favour the 20-min gain, and the temperature model's T_gnd is not a measurement.
- 2026-10-18: `tgnd_table.py`, `global_fit.py --max-tx-frac`. Working choice is 20-min knots; T_gnd errors come
  from injection draws (40-100 K at 125-203 MHz, 140-270 K at 58-117 MHz), 2-4x the fit's own. Frequencies where
  the fit collapses (51, 66, 74, 187-227 MHz) do contain the sky: in long parked stretches raw power follows the
  GSM column (corr 0.85-0.97) with a slope smooth across the band, 25-3000x the fit's gain there. The fit has two
  near-equal-chi2 solutions; in the collapsed one the 20-min gain (low band) or Sun and transmitter terms (high
  band) mimic the sky's drift. Cutting transmitter bins, or transmitter and Sun-visible bins, does not cure it.
- 2026-10-17: `global_fit.py --ground-group window|era`: one ground temperature per fitted window or height era, alone or beside a sky column, and the residual file now carries S̄ when it was computed. Per window diverges on phase C (parked windows make it a per-window offset); per era is the identifiable form.
- 2026-10-17: `global_fit.py --orient add|mult --orient-n N`: elevation harmonics as extra columns, additive or as a gain/match change with tilt, to check whether an orientation-dependent term rather than the sky carries the low-frequency ground term.
- 2026-10-17: `global_fit.py --offset-prior/--gain-anchor`. The full-band fit (20-min knots) diverges at 51, 66,
  74, 187-219 MHz because the gain collapses (GSM amplitude a_tcal 0.001-0.03 against 3-8 at 117-203 MHz) while
  the ground, transmitter and Sun terms grow, not only through T_gnd trading with the offset. An offset prior
  stops the runaway offset; an anchor holds a_tcal up. Held-out rms is the same to 0.01 % with or without
  either, the anchored a_tcal stays 0.3-1.9 even at sigma 0.2, and T_gnd swings 460-1750 K at 187-211 MHz
  with the prior. These frequencies are unconstrained by this model; leave them out of T_gnd spectra.
- 2026-10-17: `inject_mock.py` and `global_fit.py --mock-y`: injection tests of the gain model. On real data
  T_gnd (117-180 MHz) falls 2-8x from 20-min to 3-h gain knots. Mocks with a 3-h gain are recovered by every
  knot spacing; mocks with the 20-min fit's gain are recovered only by 20-min knots, and 3-h knots then
  reproduce the real 3-h values. So the real data favour 20-min knots. Adding the real residuals (rolled in
  time) scatters the recovered T_gnd by 60-80 K rms per frequency (2-4x the quoted errors), mean -20 to -50 K.
- 2026-10-16: `global_fit.py --hk-extra <stream>...`: housekeeping streams as extra columns on top of any `--ground`, so a sky term and a temperature can be fitted together and the sky term checked for surviving the temperature.
- 2026-10-16: `global_fit.py --fit-eras/--fit-utc`: fit one era or time range alone. At 117-180 MHz (rx-A,
  20-min gain knots) only the whole 30 m era gives a finite T_gnd by itself, and it differs from the all-era
  fit by 2x at 117 and 180 MHz. Each half of the 30 m era, and the 87.5 m era, diverges alone (T_gnd trades
  with the offset). T_gnd comes from level contrasts across many hours, so it rests on the gain model there.
- 2026-10-15: `global_fit.py --ground hk --hk <npz> --hk-stream <name>`: a per-bin housekeeping stream as the additive term, because over phase C the sky's LST cycle and the day share a phase; bins missing a stream are not fitted.
- 2026-10-15: `global_fit.py --swing-max-el-std/--swing-max-el-dev`: drop 30 m bins where the box swings (IMU
  elevation scatter within a bin, or offset from its 10-min median). Tried at 0.2 deg / 1 deg on 117-180 MHz:
  the kept bins' held-out residual falls only 1-5 %, while T_gnd moves by -25 % to +100 % (172 MHz 345 -> 689 K
  with both cuts). The swinging bins carry much of the elevation leverage on the ground term; not adopted.
- 2026-10-13: `global_fit.py --ground beam|pickup`: two alternatives to diffuse reflection at the same freedom, so a better fit can be attributed to the ground and not to any term that follows the sky with LST.
- 2026-10-13: `global_fit.py --ground diffuse|diffuse-free`: the ground can reflect the sky diffusely, so its temperature follows the hemisphere-averaged sky with LST; `--diffuse-lst-shift-h` builds the same column from the wrong sky as a control.
- 2026-10-13: `global_fit.py --fast-max-el-deg`: separate elevation tolerance for the fast gain's fixed-pointing
  runs (azimuth now compared with wrap). At 30 m the recorded elevation (IMU) wobbles by degrees while the
  elevation motor holds still, so most "moving" 30 m bins are this wobble. Loosening to 4 or 8 deg cuts those
  bins' held-out residual by 13-17 %, but moves T_gnd at 117 MHz from 552 K (0.5 deg) to 833 / 988 K: the fast
  gain then trades against the ground term. Keep the default (same as `--fast-max-move-deg`).
- 2026-10-13: `global_fit.py --fast-knot-s`: a fast (3 min) log gain and tilt shared across frequencies, only
  inside fixed-pointing runs, fitted on alternate frequencies and scored on the others. First trial
  (117-180 MHz, rx-A): at held-out frequencies it cuts the fixed-pointing bins' residual by 26 % at 30 m and
  87.5 m, as much as at the training frequencies; it cannot touch the moving 30 m bins, which hold most of
  the residual. Note that `--only-mhz` changes the window gate (a median over the frequencies fitted), so
  its bins differ from a full-band run's.
- 2026-10-11: `bin_raw.py` stores per-bin means of six housekeeping streams and the pointing spread;
  `global_fit.py --gain-regressors` puts them in log gain, with `--regressor-shift-s` as the control.
  First trial (148 MHz, rx-A): RF-switch temperature is a significant gain term (about -1 % per K,
  near zero when shifted), but it barely changes the held-out score with 20 min knots, and none of the
  streams explains the fast fluctuation at 30 m. `global_fit.py` no longer drops bins after the
  07-18 01:41:38 elevation-drive failure: the IMU elevation there is real (affects rx-B fits only).
- 2026-10-10: `delay_filter.py` and `bin_raw.py --delay-filter-ns` / `--union-mask-only`: an optional delay
  low-pass of every integration before binning, aimed at box-air's long-delay (300–900 ns at 87.5 m)
  fluctuation (memo 007). Its union masks drop blocks more than half flagged, which removes about half
  of the rx-A bins at 148 MHz; compare filtered bins with `--union-mask-only` bins, not with unfiltered ones.
- 2026-10-10: `beam_corr.py` and `global_fit.py --beam-lmax`: an optional
  low-order beam correction fitted with the rest of the model. In a trial at
  148 MHz (rx-A, 10-min gain knots) a correction to l = 2 gave 1.12 % rms and
  1.95 % held out, against 1.15 % and 1.70 % without it: the beam is not
  what limits the fit. Most of the residual is a minute-scale fluctuation in
  sub-5 s bins, which is being characterized before averaging.
- 2026-10-09: `bin_raw.py`, `global_fit.py`, `plot_global.py`: one model of
  raw box-air power over phase C, with a smooth gain; first trials above. `global_fit.py` now reads the beam, mount
  offsets, horizon profiles and transmitter through `common`/`sun` like the other scripts
  (merged from main); the first trials ran on v0012 offsets and horizon profiles v0002.
- 2026-10-09: The beam product is named once (`common.BEAM_PRODUCT`, now `empirical_raster_v0013`, the reference beam refitted on geometry v0004), and every mount rotation now applies the beam fit's full pointing solution (`common.mount_offsets`: az offset plus fitted az-zero delta, el-zero delta, psi). Before, the scripts applied v0012's `az_offset` and `psi` only, leaving out its +0.56 deg el-zero delta (v0013: +0.93 deg). Products from `fit_v0009`, `sun_events_v0003` and the other `_v0003` versions on use this.
- 2026-10-08: Every script reads `common.HORIZON_PROFILES` (`curation/horizon_profiles_v0003`, the release v0004 antenna) and `common.DEM_PATH` (`derived/dem/v0003`, the mosaic 1 km larger to the west and south), so the antenna, transmitter, horizon and the Sun's terrain trace all come from geometry v0004. `sun_transitions.py` takes `--sun-events` and `--fit` instead of hard-coding v0001/v0003. Products before the `_v0002` (or `fit_v0005`) versions are on v0001 geometry.
- 2026-10-08: The transmitter now comes from the geometry release the horizon profiles were built on (`sun.transmitter_enu()`), not `curation/transmitter_position.json`, so the antenna and transmitter always come from one fit. That is still release v0001 (same transmitter position as before, so existing products are unchanged); these scripts move to the current release v0004 only when `curation/horizon_profiles` is rebuilt at the v0004 antenna.
- 2026-10-07: `ground_excess.py`: the low-frequency ground excess is spread over the ground, not
  concentrated at the horizon or in particular azimuths (memo 008).

- 2026-10-07: sky recovery (`joint_fit.py`, `sky_cal.py`, `raster_sky.py`). Frequency smoothness
  alone does not fix T_gnd, and the parked nights' height lever is too weak. The rotating antenna
  gives a ground term that repeats across three 91 m windows, but it falls with frequency and is
  not reflected sky. Low-order sky corrections are not separable from ground-fixed ones (memo 008).

- 2026-10-07: solar bursts, impulsive-event rates, raw sunrise/sunset context and fits, the Sun's
  path through the beam, and the raster test (`sun_bursts.py`, `impulsive_scan.py`,
  `sun_transitions.py`, `sun_profile.py`, `raster_sun.py`). Box-air records solar bursts. The
  calibrated 07-17 sunrise excess is larger than the concurrent Sun (memo 008).

- 2026-10-06: published the products memo 008 reads; added the campaign
  sunrise/sunset table and an npz to `sun_events.py`.

- 2026-10-06: v3 flags and tcal v0003 are now the defaults. Fixed a 35 MHz
  frequency offset in `bin_spectra.py`, caused by v3's band-limited bundles.
  Horizon bearings are rotated from the UTM grid to true. The Sun is added
  (`sun.py`, `sun_events.py`, `--sun`). First calibrated excess consistent
  with the Sun at the 07-17 terrain sunrise.

- 2026-10-01: added the data pipeline (`bin_spectra.py`, `fit_ground_sky.py`,
  `common.py`) and the first fit's findings: night calibrated data fix the sky
  scale but not T_gnd.
- 2026-10-01: new directory; `degeneracy_study.py` and its first `summary.json`.
