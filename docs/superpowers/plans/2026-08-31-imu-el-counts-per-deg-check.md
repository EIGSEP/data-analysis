# Empirical counts/deg from the Jul 17 raster IMU — Follow-up

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans (or subagent-driven-development) to work this task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Status:** not started. Written 2026-08-31 as a deferred follow-up; data availability verified, analysis not yet run.

**Goal:** Measure stepper counts-per-degree empirically from the elevation IMU over the 2026-07-17 motor_scan raster, and compare against the nominal 62.7̄.

**Why this is worth doing (and what it actually tests):** the motor scale is *already* confirmed on hardware for deployments 4 and 5 — `step_angle_deg=1.8`, `gear_teeth=113`, `microstep=1` → `113/1.8 = 62.7̄` counts/deg (0.0159292 deg/step), `±11300 counts = ±180.0°`. So this check is **not** a calibration of the motor. It is an independent validation of `imu_el_from_accel`'s SVD plane fit: if the recovered slope departs from 62.7̄, the fault is in the IMU/plane-fit path, not the gearbox.

**Why the measurement is free of circularity:** in `src/eigsep_data/data.py:424` (`imu_el_from_accel`), `counts_per_deg` is used *only* to pick the two-fold sign (via `np.cov`) and the median offset. The returned `imu_el_deg` is scaled purely by gravity through the SVD. A regression of `imu_el_deg` against `el_pos` therefore measures counts/deg rather than assuming it.

**Not a firmware question:** `pico-firmware/src/motor.c:52` `m->max_pulses = 60; // pulses per command, ~1 deg` is the per-`motor_op()` pulse batch cap (used at `motor.c:97`), not a conversion factor. The firmware does no degree math; host/pico exchange raw pulse counts. Ignore the 60.

---

## Verified data facts (checked 2026-08-31, do not re-derive)

- Raster = 29 files, `corr_20260717_202824Z.h5` .. `corr_20260717_212831Z.h5` in `data/deployment5_filtered/`, 6960 integrations over 62.3 min.
- `metadata/imu_el`: **6960/6960 `status == "update"`, zero missing accel.** `|accel|` = 9.834 ± 0.075 m/s². Component std = (0.48, 6.78, 7.03) — motion is confined to a plane, so the SVD is well conditioned.
- `metadata/imu_az`: **100% `status == "error"`, every field null.** Unusable; do not touch it. (An earlier note said "IMU status=error during the scan" — that was azimuth only.)
- `el_counts` sweeps the full −11300 → +11300 (i.e. −180° → +180°), 53 turnarounds / 54 passes, median 180 counts (≈2.87°) per integration. Full-circle leverage on the fit.
- The sidecar `notebooks/christian/deployment5/motor_scan_20260717_key4.h5` has `el_counts`/`az_counts`/`times` but **no accelerometer**. Either re-read `metadata/imu_el` from `data/deployment5_filtered/`, or extend the sidecar with an `imu_accel` (6960, 3) dataset — the latter is preferable, it keeps the check reproducible without the raw tree.

---

## Method

- [ ] **Step 1: Get accel onto the raster time base.** Add `imu_accel` (6960, 3, float64) to the sidecar, ordered by the same `sort_index` as `el_counts`. Guard the extraction the way the existing notebook does (skip if the dataset is present unless `FORCE_EXTRACT`).

- [ ] **Step 2: Recover the in-plane angle.** Reuse the front half of `imu_el_from_accel`: SVD of the accel rows, project onto `Vt[0]`/`Vt[1]`, `np.unwrap(np.degrees(np.arctan2(...)), period=360)`. Do **not** call the full function — its output has already had the offset absorbed; you want `raw_deg`. Consider factoring that half out into a small helper so both the function and this check share it rather than duplicating the algebra.

- [ ] **Step 3: Fit per monotonic pass, not globally.** `el_pos` is the *commanded* position and the platform lags it. A constant lag is harmless mid-sweep (it shifts the offset) but biases the slope across the 53 turnarounds. Split on `np.sign(np.diff(el_counts))`, drop a few samples either side of each reversal, and fit `raw_deg = a * el_counts + b` per pass.

- [ ] **Step 4: Report.** `counts_per_deg = 1 / |a|`, with the spread across the 54 passes as the uncertainty. Compare up-passes against down-passes separately — a systematic split between the two directions is gear backlash, which is a real mechanical number worth having, not noise.

- [ ] **Step 5: Cross-check the unwrap.** Because el traverses a full 360°, a mis-unwrap shows up as a slope that is wrong by a clean ratio rather than as scatter. Sanity-check that each pass spans ≈360° of `raw_deg` before trusting its slope.

**Pass criterion:** per-pass `counts_per_deg` consistent with 62.7̄ to within ~1%. A clean 4× discrepancy (251.1) would mean something in the chain picked up the legacy `eigsep-motor-control` `microstep = 4` (`stepper_pico.py:43-46`, `stepper_rpi.py:27-29`, `scripts/sender.py:12-13`) instead of picohost's `microstep = 1` — that library is superseded, do not use it for conversions.

**Deliverable:** follow the repo's existing shape — the numeric work in a small tested helper next to `src/eigsep_data/data.py`, the narrative and plots in a notebook under `notebooks/christian/deployment5/`. Plot `raw_deg` vs `el_counts` coloured by pass direction; the backlash, if present, will be visible as two offset lines.
