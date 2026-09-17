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
    discrete slip/re-home events, and loses 27.4 deg in a single 12.3-min
    episode during the 07-17 scan (20:41:24-20:53:43, -133 deg/hr) while
    tracking the command to within a few degrees either side of it.
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
FLAG_AZ_SLIP_RAMP = 1 << 9      # sustained az slip: platform losing ground to motor
FLAG_EL_SOLUTION_GLITCH = 1 << 10  # elevation slew faster than the drive can move

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
    FLAG_AZ_SLIP_RAMP: "AZ_SLIP_RAMP",
    FLAG_EL_SOLUTION_GLITCH: "EL_SOLUTION_GLITCH",
}

# Sustained-slip detector. AZ_SLIP_EVENT catches *steps* in the motor-vs-pot
# offset and is blind to *ramps* -- which is how it missed both events that
# actually moved the 07-17 scan off its commanded grid (a ~70 s startup
# transient worth +8.7 deg, and the 12.3-min episode worth -27.4 deg).
# A +/-60 s window separates the episode (median 133 deg/hr) from the quiet
# phase (p95 50 deg/hr) by 2.6x; shorter windows are swamped by the
# platform's per-step lag behind the motor, longer ones blur the episode.
SLIP_RAMP_HALF_S = 60.0
SLIP_RAMP_DEG_PER_HR = 90.0

# Unphysical-elevation-slew detector (beam-analyst, beam_metric_outliers_
# checkpoint.ipynb / detect_el_slew_glitch.py). In the post-EL-failure wrap
# cluster the antenna is parked at el ~ +/-180 and the IMU elevation solver
# intermittently emits a single spurious sample at |el| ~ 59-60 (or ~0) before
# returning to the park -- 42 of 45 such samples in the 07-17/18 beam-scan
# window have an immediate same-file neighbour at |el| > 150, and the
# implied slew rate (p90 ~220 deg/s) is far beyond anything the drive can
# do. Commanded scan slew is ~5 deg/s (campaign p99 of the
# in-file rate); 20 deg/s is 4x that and sits in a sparse valley between real
# motion and the glitch population (counts barely move between the 20, 30
# and 50 deg/s thresholds). None of the existing flags catch these: they pass
# as quality=="ok" with no UNCOMMANDED_MOTION.
#
# Scope caveat: the wrap-cluster mechanism is established for the
# post-EL-failure era. Applied campaign-wide, pre-failure firings of this
# same criterion are a different, less-understood population (they rarely
# show the |el|~59 signature and are mostly already quality=="suspect" for
# other reasons) -- so this flag is named and thresholded by its criterion
# ("elevation solution moved faster than the drive can"), not by the park
# mechanism, and consumers working pre-failure data should not assume it
# means the specific wrap-park glitch.
EL_SLEW_MAX_DEG_S = 20.0
EL_SLEW_DT_MIN_S = 0.1
EL_SLEW_DT_MAX_S = 2.0

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
    cuts the 1.73 deg pot noise to ~0.23 deg.  Slip smeared into that window
    is negligible outside slip episodes and reaches ~1.1 deg at the peak
    episode rate (-133 deg/hr); those samples carry AZ_SLIP_EVENT, and the
    segment-wise offset estimate keeps the episode from contaminating the
    quiet phases either side of it.
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


def detect_az_slip_ramp(az_deg, motor_az_steps, times,
                        half_s=SLIP_RAMP_HALF_S,
                        thresh_deg_per_hr=SLIP_RAMP_DEG_PER_HR,
                        gap_s=5.0):
    """Flag sustained azimuth slip -- the platform losing ground to the motor.

    Measures the rate of change of (fused azimuth - motor azimuth) over a
    +/-``half_s`` window and thresholds it.  Complements
    :func:`detect_az_slip`, which sees steps but not ramps.

    Operates per contiguous time segment so a data gap is never differenced
    across, and shrinks the window at segment edges so a transient in the
    first minute is still visible -- the 07-17 startup transient sits
    entirely inside one window half-width of the scan start.
    """
    az = np.asarray(az_deg, float)
    motor = MOTOR_DEG_PER_STEP * np.asarray(motor_az_steps, float)
    times = np.asarray(times, float)
    n = az.size
    out = np.zeros(n, bool)

    ok = np.isfinite(az) & np.isfinite(motor)
    if not ok.any():
        return out
    idx = np.flatnonzero(ok)
    # Split into contiguous runs; unwrapping across a gap is meaningless.
    breaks = np.flatnonzero(np.diff(times[idx]) > gap_s)
    starts = np.concatenate(([0], breaks + 1))
    stops = np.concatenate((breaks + 1, [idx.size]))

    for s, e in zip(starts, stops):
        sub = idx[s:e]
        if sub.size < 20:
            continue
        t_sub = times[sub]
        div = (np.rad2deg(np.unwrap(np.deg2rad(az[sub])))
               - np.rad2deg(np.unwrap(np.deg2rad(motor[sub]))))
        # Window edges by time, clipped to the segment -- this is what keeps
        # the detector alive within half_s of a segment boundary.
        lo = np.searchsorted(t_sub, t_sub - half_s, "left")
        hi = np.searchsorted(t_sub, t_sub + half_s, "right") - 1
        dt_win = t_sub[hi] - t_sub[lo]
        valid = dt_win > max(half_s * 0.5, 1.0)
        rate = np.zeros(sub.size)
        rate[valid] = ((div[hi[valid]] - div[lo[valid]])
                       / (dt_win[valid] / 3600.0))
        out[sub[valid & (np.abs(rate) > thresh_deg_per_hr)]] = True
    return out


def detect_el_solution_glitch(el_deg, times, file_index,
                              thresh_deg_per_s=EL_SLEW_MAX_DEG_S,
                              dt_min=EL_SLEW_DT_MIN_S, dt_max=EL_SLEW_DT_MAX_S):
    """Flag samples adjacent to an unphysical elevation slew.

    For each pair of temporally adjacent samples (i, i+1) in the same file
    with ``dt_min < dt < dt_max`` and both elevations finite, computes
    ``|el[i+1] - el[i]| / dt`` and flags **both** i and i+1 if it exceeds
    ``thresh_deg_per_s`` -- the transition identifies a bad pair; which
    member is the bad one is not determined by the rate alone.

    Reference implementation: beam-analyst's
    ``notebooks/arp/marjum-2026-07/detect_el_slew_glitch.py``, which this
    reproduces exactly (verified against ``disc_mask.npy``, 604/604 agree
    over the 07-17/18 beam-scan window). Do not substitute a "differs from
    both neighbours" test: it misses glitch runs of 2-3 consecutive samples
    and only partially heals the affected elevation bands.
    """
    el = np.asarray(el_deg, float)
    t = np.asarray(times, float)
    fi = np.asarray(file_index)
    n = el.size

    same_file = fi[1:] == fi[:-1]
    dt = np.diff(t)
    d_el = np.abs(np.diff(el))
    testable = (same_file & np.isfinite(d_el) & np.isfinite(dt)
                & (dt > dt_min) & (dt < dt_max))

    bad = np.zeros(n - 1, bool)
    bad[testable] = (d_el[testable] / dt[testable]) > thresh_deg_per_s

    out = np.zeros(n, bool)
    out[:-1] |= bad
    out[1:] |= bad
    return out


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
