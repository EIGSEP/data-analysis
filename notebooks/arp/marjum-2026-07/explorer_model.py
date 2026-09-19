"""Headless copy of beam_explorer.ipynb's model, so scripts can reproduce a
hand-fit exactly without a kernel.

Lifted verbatim from the notebook's cells 1/3/5 (the coupling maths, the
amplitude fit, the normalized-RMS metric and the display binning) so that any
number produced here is the same number the sliders show.
"""
import os

import numpy as np
import healpy as hp
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Which cache to read. Defaults to the live one, but an analysis that must be
# reproducible against a *specific* cache generation can pin it:
#
#     os.environ["BEAM_EXPLORER_CACHE"] = "beam_explorer_cache_pre20260917.npz"
#
# before importing this module. beam_metric_outliers_checkpoint does exactly
# that: it is the record of the investigation that found the calibration-window
# leak, so it has to run against the cache that still contained it. Re-running
# it against the corrected cache would have it "discover" a defect that had
# already been removed.
CACHE = Path(os.environ.get("BEAM_EXPLORER_CACHE",
                            str(HERE / "beam_explorer_cache.npz")))
if not CACHE.is_absolute():
    CACHE = HERE / CACHE

C = np.load(CACHE)
AZ = C["az_deg"].astype(float)
EL = C["el_deg"].astype(float)
Y = C["measured_tx"]
USED = C["used"]
CHANS = C["channels"]
ARMS = C["arms"]
FREQS = C["freqs_mhz"]
A_HFSS = C["a_hfss"]
BEAM = C["beam_cart"].astype(np.complex128)
NSIDE = int(C["nside"])
NCOMP = BEAM.shape[0]
SURVEYED_DELTA = C["surveyed_delta_enu"]


def rotations(az_deg, el_deg):
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    ca, sa = np.cos(az), -np.sin(az)
    ce, se = np.cos(el), np.sin(el)
    R = np.empty((az.size, 3, 3))
    R[:, 0] = np.stack([ca, -sa, np.zeros_like(ca)], axis=1)
    R[:, 1] = np.stack([ce * sa, ce * ca, -se], axis=1)
    R[:, 2] = np.stack([se * sa, se * ca, ce], axis=1)
    return R


def field_top(alpha_deg, arm):
    a = np.deg2rad(alpha_deg + (90.0 if arm else 0.0))
    return np.array([-np.sin(a), np.cos(a), 0.0])


def coupling(az_deg, el_deg, heading, alpha_deg, arm):
    R = rotations(az_deg, el_deg)
    Rt = R.transpose(0, 2, 1)
    rhat = np.einsum("nij,j->ni", Rt, heading)
    th = np.arccos(np.clip(rhat[:, 2], -1, 1))
    ph = np.mod(np.arctan2(rhat[:, 1], rhat[:, 0]), 2 * np.pi)
    px = hp.ang2pix(NSIDE, th, ph)
    w = np.moveaxis(BEAM[:, :, px], 1, -1)
    e = np.einsum("nij,j->ni", Rt, field_top(alpha_deg, arm))
    e = e - np.sum(e * rhat, axis=1, keepdims=True) * rhat
    w = w - np.sum(w * rhat[None], axis=2, keepdims=True) * rhat[None]
    return np.einsum("fni,ni->fn", np.conj(w), e)


def householder(u):
    e0 = np.zeros(u.size, complex)
    e0[0] = 1.0
    alpha = -np.exp(1j * np.angle(u[0])) if abs(u[0]) > 1e-12 else -1.0
    v = u - alpha * e0
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return np.eye(u.size, dtype=complex)
    v = v / nv
    return np.eye(u.size, dtype=complex) - 2 * np.outer(v, np.conj(v))


# Sign of the elevation rotation.
#
# The model rotates elevation about the FIXED +x = East shaft, so its +el tips
# the boresight from zenith toward -N (SOUTH). Christian's hardware convention
# (2026-09-17) is that +EL is the right-hand rule about the highline vector
# pointing WEST, which tips the boresight toward NORTH. The model is therefore
# backwards, and `el_sign = -1` corrects it.
#
# Default is +1, i.e. the UNCORRECTED model, so that every number published
# before 2026-09-17 -- fit_beam_v2, both review checkpoints, and the pinned
# beam_metric_outliers_checkpoint -- stays reproducible. Pass el_sign=-1 for
# the physically correct convention.
EL_SIGN_UNCORRECTED = 1
EL_SIGN_CORRECTED = -1


def model_power(ch_index, heading, alpha_deg, arm, shape_re, shape_im,
                gain=None, az_off=0.0, el_off=0.0, apply_cal=False,
                el_sign=EL_SIGN_UNCORRECTED):
    cpl = coupling(AZ + az_off, el_sign * EL + el_off, heading, alpha_deg, arm)
    a = A_HFSS[ch_index]
    g_hfss = np.linalg.norm(a)
    basis = householder(a / max(g_hfss, 1e-30))
    cpl_rot = basis.conj().T @ cpl
    params = np.r_[1.0, np.array(shape_re) + 1j * np.array(shape_im)]
    m = np.abs(np.conj(params) @ cpl_rot) ** 2
    u = cal_used(ch_index) if apply_cal else USED[ch_index]
    d = data(ch_index, apply_cal=apply_cal)
    if gain is None:
        denom = np.sum(m[u] * m[u])
        A = np.sum(d[u] * m[u]) / denom if denom > 0 else 0.0
    else:
        A = gain
    return A * m, A


def normalized_rms(ch_index, model, el_cut=180.0, extra_mask=None,
                   apply_cal=False):
    u = cal_used(ch_index) if apply_cal else USED[ch_index]
    u = u & (np.abs(EL) <= el_cut)
    if extra_mask is not None:
        u = u & extra_mask
    d = data(ch_index, apply_cal=apply_cal)[u]
    r = d - model[u]
    return float(np.sqrt(np.mean(r ** 2)) / max(np.sqrt(np.mean(d ** 2)), 1e-30))


# --------------------------------------------------------------- calibration
# B7's per-cycle receiver gain, interpolated onto the sample times.
#
# measured_tx is a channel difference, so with auto = g_rx * (T_sky + T_rx) and
# g_rx smooth over three adjacent 244 kHz channels, it carries g_rx as a LINEAR
# multiplicative factor. Dividing it out removes a drift the single per-channel
# fit amplitude cannot absorb (x1.61 at 173.83 MHz across this window).
#
# CAL_GAIN is NaN outside the cycle span -- no extrapolation. CAL_ANCHOR_MIN is
# the distance in minutes to the nearest real cycle, which matters because the
# cycles are not evenly spread: a 342-minute hole (07-17 19:42 -> 07-18 01:24)
# holds 44% of the used samples, and it is exactly where B7 brackets the
# receiver regime change. Inside it the interpolation is a straight line
# through an unknown transition, not a measurement.
_CALF = HERE / "cal_gain.npz"
CAL_AVAILABLE = _CALF.exists() and "times" in C
CAL_GAIN = None
CAL_ANCHOR_MIN = None
CAL_GAP = (None, None)

if CAL_AVAILABLE:
    _cal = np.load(_CALF, allow_pickle=True)
    TIMES = C["times"].astype(float)
    _st = _cal["sol_times"].astype(float)
    _gc = _cal["gain_cycles"].astype(float)          # (ncyc, nchan)
    CAL_GAIN = np.full((len(CHANS), TIMES.size), np.nan)
    _tok = TIMES > 0
    for _j in range(len(CHANS)):
        _ok = np.isfinite(_gc[:, _j]) & (_gc[:, _j] > 0)
        if _ok.sum() < 2:
            continue
        _s, _g = _st[_ok], _gc[_ok, _j]
        _in = _tok & (TIMES >= _s[0]) & (TIMES <= _s[-1])
        CAL_GAIN[_j, _in] = np.interp(TIMES[_in], _s, _g)
    CAL_ANCHOR_MIN = np.full(TIMES.size, np.nan)
    CAL_ANCHOR_MIN[_tok] = np.min(
        np.abs(TIMES[_tok][:, None] - _st[None, :]), axis=1) / 60.0
    CAL_GAP = (str(_cal["gap_start_utc"]), str(_cal["gap_end_utc"]))


def data(ch_index, apply_cal=False):
    """Measured quantity for a channel: raw counts, or counts / g_rx."""
    d = Y[ch_index].astype(float)
    if not apply_cal:
        return d
    if not CAL_AVAILABLE:
        raise RuntimeError("cal_gain.npz not available; run build_cal_gain.py")
    with np.errstate(invalid="ignore", divide="ignore"):
        return d / CAL_GAIN[ch_index]


def cal_used(ch_index):
    """USED restricted to samples that have a gain solution (no extrapolation)."""
    if not CAL_AVAILABLE:
        return USED[ch_index]
    return USED[ch_index] & np.isfinite(CAL_GAIN[ch_index])


def heading_from_enu(dE, dN, dU):
    v = np.array([dE, dN, dU], float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.array([0.0, 0.0, -1.0])


BIN_DEG = 4.0
AZ_EDGES = np.arange(0.0, 360.0 + BIN_DEG, BIN_DEG)
EL_EDGES = np.arange(-180.0, 180.0 + BIN_DEG, BIN_DEG)
NAZ, NEL = len(AZ_EDGES) - 1, len(EL_EDGES) - 1
_ai = np.clip(np.digitize(AZ, AZ_EDGES) - 1, 0, NAZ - 1)
_ei = np.clip(np.digitize(EL, EL_EDGES) - 1, 0, NEL - 1)
_flat = _ai * NEL + _ei


def grid(values, mask):
    s = np.bincount(_flat[mask], weights=values[mask], minlength=NAZ * NEL)
    n = np.bincount(_flat[mask], minlength=NAZ * NEL)
    out = np.full(NAZ * NEL, np.nan)
    ok = n > 0
    out[ok] = s[ok] / n[ok]
    return out.reshape(NAZ, NEL).T


AARON_CASE = dict(
    channel=712, arm=1, alpha_deg=51.0, dE=0.0, dN=8.0, dU=-93.5,
    gain=1.82e11, shape_re=[0.0] * (NCOMP - 1), shape_im=[0.0] * (NCOMP - 1),
    el_cut=180.0,
)


def aaron_case():
    """Return (ch_index, model_power_array, amplitude, normalized_rms)."""
    k = AARON_CASE
    i = int(np.argmin(np.abs(CHANS - k["channel"])))
    h = heading_from_enu(k["dE"], k["dN"], k["dU"])
    m, A = model_power(i, h, k["alpha_deg"], k["arm"], k["shape_re"],
                       k["shape_im"], gain=k["gain"])
    return i, m, A, normalized_rms(i, m, el_cut=k["el_cut"])
