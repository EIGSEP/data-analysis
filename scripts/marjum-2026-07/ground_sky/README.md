# Ground temperature and sky from an assumed beam

Infrastructure for solving for the ground temperature and a sky model from
Marjum spectra, given a beam model. This is the EIGSEP counterpart of
bloom21cm's per-frequency linear inversion. The method is in
`eigsep_sim.design_matrix`; this directory holds the Marjum-specific
pipeline (calibrated spectra → bins → fit) and the studies that drive it.

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
- Mount: `eigsep_base.rotations.mount_rotation(az_table + az_offset, el, psi)`,
  with `az_offset` and `psi` read from `empirical_raster_v0012`.
- Pointing: `curation/pointing_table.parquet`, using phase C and
  `quality == "ok"` rows only.
- Horizon bearings: the horizon-profile bearings are UTM 12N grid
  bearings (the DEM's frame), 1.52° from true at the site. The fit now
  rotates them to true bearings; the first trials did not.
- Beam: pluggable. Either the HFSS bowtie or an `empirical_beam.npz`, through
  `HealpixBeam.from_npz`.

## Scripts

| Script | What it does |
|---|---|
| `bin_spectra.py` | Builds `derived/ground_sky/binned_vNNNN`: `tcal@v0003` box-air temperatures, `flags@v3` (any bit but 9; `flags@v2` bits 0–7 with `--flags v2`), transmitter-comb channels dropped in its span, averaged over ±2 channels at each HFSS beam frequency and into static 120 s bins of one height era and receiver regime. Noise is the in-bin scatter. |
| `fit_ground_sky.py` | Builds `derived/ground_sky/fit_vNNNN` from a binned product: per-frequency fit of `a`, T_gnd and one offset per receiver regime. `--sun model` adds the Sun as a column; `--sun cut-flat` (Sun below −10°) or `cut-terrain` (Sun behind the ridge) instead chooses which bins are fitted. The model and residual are evaluated on every bin either way, so cut bins are out-of-sample predictions. Reports which columns the data actually constrain, the per-regime pedestal ⟨f_gnd⟩·T_gnd + T_off that is constrained even when its parts are not, and χ² on held-out alternating 30-min blocks. |
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

## Recent changes

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
