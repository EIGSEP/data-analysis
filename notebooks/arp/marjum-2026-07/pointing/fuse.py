"""Fuse the Marjum 2026-07 pointing streams into an az/el/height timeline.

Sensor roles for this campaign (established empirically, see MEMO):

``potmon.pot_az_angle``
    The **only** absolute azimuth available in the 07-17/18 window --
    ``imu_az`` returns ``status='error'`` for 100% of samples.  Noisy:
    sigma ~1.73 deg measured from platform-static runs.  ``pot_az_near_rail``
    never asserts, and the calibration slope/intercept are constant, so it is
    unbiased but coarse.

``motor.az_pos`` / ``motor.el_pos``
    Smooth and finely quantised (0.0159 deg/step) but **relative** and
    unreliable open-loop: the motor-vs-pot offset jumps by up to 45 deg at
    discrete slip/re-home events and drifts ~34 deg/hr during the scan.
    Used for short-term motion only, never as an absolute angle.

``imu_el.el_deg``
    Gravity-referenced absolute elevation, sigma ~0.24 deg, validated against
    ``imu_elevation_deg(accel)`` (median diff 0.016 deg) with
    ``|accel|`` = 9.838 +/- 0.091 m/s^2.  The authority on elevation, and the
    only sensor that reveals the EL drive failure.

``lidar.distance_m``
    Platform-mounted rangefinder, not a continuous altimeter: 77% of finite
    readings are the exact out-of-range sentinel 250.00 m, and ground returns
    only occur when the feed points down (el ~75-90 deg).  Used to *check*
    height during the 01:32-01:41 sweeps, not to track it.

Fusion strategy
---------------
Azimuth uses a complementary filter rather than a weighted mean: the motor
supplies relative motion, the potentiometer supplies the absolute level via a
robust rolling median of ``pot - motor`` computed **within** slip segments so
a slip event is never smoothed across.  This keeps the motor's fine structure
(the 5 deg scan steps) while cutting the 1.73 deg pot noise by ~sqrt(N).

Elevation is taken directly from the IMU where available; the motor adds
nothing, being both noisier in absolute terms and outright false after the EL
drive fails.
"""

from __future__ import annotations

import numpy as np

MOTOR_DEG_PER_STEP = 180.0 / 1.13e4

# Measured sensor noise (deg, 1-sigma), from platform-static runs in-window.
POT_AZ_SIGMA = 1.73
IMU_EL_SIGMA = 0.24
MOTOR_QUANT_SIGMA = MOTOR_DEG_PER_STEP / np.sqrt(12.0)

# Irreducible azimuth error floor: genuine platform sway on the 91 m tether
# plus residual slip drift inside the smoothing window.  Calibrated from the
# scatter of the fused azimuth across 52 scan plateaus where the commanded
# azimuth was provably constant (observed 0.408 deg; white-noise model alone
# predicts 0.176 deg), so this is a held-out validation residual against a
# known-constant truth, not the training residual of the filter.
AZ_FLOOR_SIGMA = float(np.sqrt(0.4078 ** 2 - 0.176 ** 2))  # ~0.368 deg

# LIDAR out-of-range sentinel and plausible-ground-return band (m).
LIDAR_SENTINEL = 250.0
LIDAR_MIN_VALID = 20.0
LIDAR_MAX_VALID = 200.0

# Quality flag bits.
FLAG_OK = 0
FLAG_NO_METADATA = 1 << 0       # correlator file carries no metadata group
FLAG_AZ_NO_POT = 1 << 1         # no absolute azimuth reference this sample
FLAG_AZ_SLIP_EVENT = 1 << 2     # within a motor az slip/re-home transient
FLAG_EL_NO_IMU = 1 << 3         # elevation from motor only, unvalidated
FLAG_EL_STUCK = 1 << 4          # EL drive commanding but antenna not moving
FLAG_UNCOMMANDED = 1 << 5       # antenna moving with no motor command
FLAG_NO_ESTIMATE = 1 << 6       # no sensor available; value is NaN, NOT filled
FLAG_HEIGHT_ASSUMED = 1 << 7    # height nominal, not measured at this sample
FLAG_EL_POST_FAILURE = 1 << 8   # at/after the EL drive failure; el is parked

FLAG_NAMES = {
    FLAG_NO_METADATA: "NO_METADATA",
    FLAG_AZ_NO_POT: "AZ_NO_POT",
    FLAG_AZ_SLIP_EVENT: "AZ_SLIP_EVENT",
    FLAG_EL_NO_IMU: "EL_NO_IMU",
    FLAG_EL_STUCK: "EL_STUCK",
    FLAG_UNCOMMANDED: "UNCOMMANDED_MOTION",
    FLAG_NO_ESTIMATE: "NO_ESTIMATE",
    FLAG_HEIGHT_ASSUMED: "HEIGHT_ASSUMED",
    FLAG_EL_POST_FAILURE: "EL_POST_FAILURE",
}

# v0 never extrapolates: where no sensor supports a sample the value is NaN
# and FLAG_NO_ESTIMATE is set.  A gap flag is preferable to a smooth fit
# through absent data.
EXTRAPOLATION_POLICY = "none: unsupported samples are NaN + NO_ESTIMATE"


def wrap180(x):
    return (np.asarray(x, float) + 180.0) % 360.0 - 180.0


def wrap360(x):
    return np.asarray(x, float) % 360.0


def _runs(mask):
    """Contiguous [start, stop) index runs where boolean ``mask`` is True."""
    mask = np.asarray(mask, bool)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def rolling_median(values, valid, half_window):
    """Rolling median of ``values`` over +/- ``half_window`` samples.

    Only ``valid`` entries contribute; positions with no valid neighbour
    return NaN.  Implemented with an explicit loop over a sorted window --
    the arrays here are ~5e4 samples, so clarity beats cleverness.
    """
    values = np.asarray(values, float)
    n = values.size
    out = np.full(n, np.nan)
    idx = np.flatnonzero(valid)
    if idx.size == 0:
        return out
    vals = values[idx]
    for i in range(n):
        lo = np.searchsorted(idx, i - half_window, "left")
        hi = np.searchsorted(idx, i + half_window, "right")
        if hi > lo:
            out[i] = np.median(vals[lo:hi])
    return out


def detect_az_slip(motor_az_deg, pot_az_deg, half_window=50, jump_deg=1.0,
                   merge_gap=200):
    """Find discrete motor-azimuth slip / re-home events.

    Slip shows up as a step in ``pot - motor``; a smoothed version of that
    offset is differenced and thresholded.  Returns a boolean per-sample mask
    marking the slip transients themselves.
    """
    offset = wrap180(pot_az_deg - motor_az_deg)
    valid = np.isfinite(offset)
    smooth = rolling_median(offset, valid, half_window)
    jump = np.abs(np.diff(smooth, prepend=smooth[0]))
    mask = np.nan_to_num(jump) > jump_deg
    # Widen each detection so the transient itself is covered, then merge.
    out = mask.copy()
    for s, e in _runs(mask):
        out[max(0, s - merge_gap // 2):min(mask.size, e + merge_gap // 2)] = True
    return out, smooth


def segment_bounds(slip_mask, n):
    """Index boundaries of stable (non-slip) segments."""
    bounds = [0]
    for s, e in _runs(slip_mask):
        bounds.extend([s, e])
    bounds.append(n)
    bounds = sorted(set(np.clip(bounds, 0, n)))
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] > bounds[i]]


def fuse_azimuth(motor_az_steps, pot_az_deg, half_window=56):
    """Complementary-filter azimuth: motor motion anchored to pot level.

    ``half_window`` of 56 samples is +/-30 s at the 0.537 s cadence, which
    cuts the 1.73 deg pot noise to ~0.23 deg while limiting smeared slip
    drift (~34 deg/hr during the scan) to a comparable ~0.28 deg.
    """
    motor_deg = MOTOR_DEG_PER_STEP * np.asarray(motor_az_steps, float)
    pot = np.asarray(pot_az_deg, float)
    n = motor_deg.size

    slip_mask, _ = detect_az_slip(motor_deg, pot)
    have_both = np.isfinite(motor_deg) & np.isfinite(pot)

    offset = np.full(n, np.nan)
    offset_scatter = np.full(n, np.nan)
    raw_offset = wrap180(pot - motor_deg)

    # Estimate the motor->absolute offset independently inside each stable
    # segment so a slip step is never averaged across.
    for s, e in segment_bounds(slip_mask, n):
        seg_valid = have_both[s:e]
        if not seg_valid.any():
            continue
        seg_raw = raw_offset[s:e]
        # Unwrap about the segment median so the rolling median is linear-safe.
        centre = np.median(seg_raw[seg_valid])
        seg_rel = wrap180(seg_raw - centre)
        smooth = rolling_median(seg_rel, seg_valid, half_window)
        offset[s:e] = centre + smooth
        resid = seg_rel - smooth
        # Robust local scatter -> empirical pot noise, per segment.
        good = seg_valid & np.isfinite(resid)
        if good.sum() > 10:
            offset_scatter[s:e] = 1.4826 * np.median(np.abs(resid[good]))

    az = wrap360(motor_deg + offset)

    # Where the motor is absent but the pot is not, fall back to raw pot.
    pot_only = ~np.isfinite(az) & np.isfinite(pot)
    az[pot_only] = wrap360(pot[pot_only])

    flags = np.zeros(n, dtype=np.int32)
    flags[slip_mask] |= FLAG_AZ_SLIP_EVENT
    flags[~np.isfinite(pot)] |= FLAG_AZ_NO_POT

    # Uncertainty: smoothed-pot standard error, floored by the single-sample
    # pot noise where we could not smooth, plus motor quantisation.
    n_eff = np.full(n, np.nan)
    for s, e in segment_bounds(slip_mask, n):
        cnt = np.cumsum(have_both[s:e].astype(float))
        lo = np.maximum(0, np.arange(e - s) - half_window)
        hi = np.minimum(e - s - 1, np.arange(e - s) + half_window)
        n_eff[s:e] = np.maximum(cnt[hi] - np.where(lo > 0, cnt[lo - 1], 0.0), 1.0)
    sigma_pot = np.where(np.isfinite(offset_scatter), offset_scatter, POT_AZ_SIGMA)
    sigma_az = np.sqrt((sigma_pot / np.sqrt(np.nan_to_num(n_eff, nan=1.0))) ** 2
                       + MOTOR_QUANT_SIGMA ** 2 + AZ_FLOOR_SIGMA ** 2)
    sigma_az[pot_only] = np.hypot(POT_AZ_SIGMA, AZ_FLOOR_SIGMA)
    sigma_az[~np.isfinite(az)] = np.nan
    return az, sigma_az, flags, offset


def detect_el_stuck(imu_el_deg, motor_el_steps, window=200, imu_ptp_deg=3.0,
                    motor_ptp_deg=30.0):
    """Flag samples where the EL motor commands motion the antenna refuses.

    The failure signature is unambiguous: the motor count sweeps a large
    range while the gravity-referenced IMU elevation stays put.
    """
    imu = np.asarray(imu_el_deg, float)
    motor = MOTOR_DEG_PER_STEP * np.asarray(motor_el_steps, float)
    n = imu.size
    stuck = np.zeros(n, bool)
    unwrapped = np.full(n, np.nan)
    ok = np.isfinite(imu)
    if ok.any():
        unwrapped[ok] = np.rad2deg(np.unwrap(np.deg2rad(imu[ok])))
    for i in range(0, n, 10):
        lo, hi = max(0, i - window), min(n, i + window)
        seg_i = unwrapped[lo:hi]
        seg_m = motor[lo:hi]
        gi, gm = np.isfinite(seg_i), np.isfinite(seg_m)
        if gi.sum() < 20 or gm.sum() < 20:
            continue
        if np.ptp(seg_i[gi]) < imu_ptp_deg and np.ptp(seg_m[gm]) > motor_ptp_deg:
            stuck[lo:hi] = True
    return stuck


def detect_el_motion(imu_el_deg, window=100, ptp_deg=3.0):
    """Per-sample mask of genuine (IMU-observed) elevation motion."""
    imu = np.asarray(imu_el_deg, float)
    n = imu.size
    ok = np.isfinite(imu)
    unwrapped = np.full(n, np.nan)
    if ok.any():
        unwrapped[ok] = np.rad2deg(np.unwrap(np.deg2rad(imu[ok])))
    out = np.zeros(n, bool)
    for i in range(n):
        lo = max(0, i - window)
        seg = unwrapped[lo:i + 1]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 10 and np.ptp(seg) > ptp_deg:
            out[i] = True
    return out


def detect_uncommanded(imu_el_deg, motor_el_steps, window=200,
                       imu_ptp_deg=20.0, motor_ptp_deg=1.0):
    """Flag antenna motion with no corresponding motor command."""
    imu = np.asarray(imu_el_deg, float)
    motor = MOTOR_DEG_PER_STEP * np.asarray(motor_el_steps, float)
    n = imu.size
    out = np.zeros(n, bool)
    unwrapped = np.full(n, np.nan)
    ok = np.isfinite(imu)
    if ok.any():
        unwrapped[ok] = np.rad2deg(np.unwrap(np.deg2rad(imu[ok])))
    for i in range(0, n, 10):
        lo, hi = max(0, i - window), min(n, i + window)
        seg_i, seg_m = unwrapped[lo:hi], motor[lo:hi]
        gi, gm = np.isfinite(seg_i), np.isfinite(seg_m)
        if gi.sum() < 20 or gm.sum() < 20:
            continue
        if np.ptp(seg_i[gi]) > imu_ptp_deg and np.ptp(seg_m[gm]) < motor_ptp_deg:
            out[lo:hi] = True
    return out


def fuse_elevation(imu_el_deg, motor_el_steps):
    """Elevation from the IMU, with a flagged motor-only fallback."""
    imu = np.asarray(imu_el_deg, float)
    motor_deg = wrap180(MOTOR_DEG_PER_STEP * np.asarray(motor_el_steps, float))
    n = imu.size

    el = np.where(np.isfinite(imu), wrap180(imu), np.nan)
    sigma = np.full(n, np.nan)
    sigma[np.isfinite(el)] = IMU_EL_SIGMA
    flags = np.zeros(n, dtype=np.int32)

    stuck = detect_el_stuck(imu, motor_el_steps)
    flags[stuck] |= FLAG_EL_STUCK
    flags[detect_uncommanded(imu, motor_el_steps)] |= FLAG_UNCOMMANDED

    # Once the EL drive has demonstrably failed, every later sample is
    # parked-or-suspect: mark the whole tail so no consumer fits through it.
    # The stall detector uses a +/-window, so its first trigger can precede
    # the last genuine motion; start the tail at the later of the two so the
    # flag never contradicts an observed rotation.
    if stuck.any():
        first_stuck = int(np.flatnonzero(stuck)[0])
        moving = detect_el_motion(imu)
        prior = np.flatnonzero(moving[:min(n, first_stuck + 2 * 200)])
        start = max(first_stuck, int(prior[-1]) + 1) if prior.size else first_stuck
        flags[start:] |= FLAG_EL_POST_FAILURE

    # IMU-less samples: motor only, and with no way to validate it.
    fallback = ~np.isfinite(el) & np.isfinite(motor_deg)
    el[fallback] = motor_deg[fallback]
    flags[~np.isfinite(imu)] |= FLAG_EL_NO_IMU
    # Unvalidated motor elevation: quote the observed motor-vs-IMU spread.
    sigma[fallback] = np.nan  # filled by caller from the measured disagreement
    return el, sigma, flags


def lidar_height(distance_m, imu_el_deg, el_lo=70.0, el_hi=95.0):
    """Ground-return height estimates from the platform LIDAR.

    Returns a masked array of plausible heights: sentinel and structure-hit
    returns removed, restricted to feed-down orientations.
    """
    d = np.asarray(distance_m, float)
    el = np.asarray(imu_el_deg, float)
    good = (np.isfinite(d) & (d != LIDAR_SENTINEL)
            & (d > LIDAR_MIN_VALID) & (d < LIDAR_MAX_VALID)
            & np.isfinite(el) & (np.abs(el) >= el_lo) & (np.abs(el) <= el_hi))
    out = np.full(d.size, np.nan)
    out[good] = d[good]
    return out
