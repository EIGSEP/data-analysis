"""One phenomenological model of raw box-air power over all of phase C, per frequency.

Builds ``derived/ground_sky/global_vNNNN`` from ``raw_binned_vNNNN``. Every
bin b, at each frequency, is modelled as

    y_b = G_w [ C_GSM,b + T_g C_gnd,b + sum_lm d_lm C_lm,b + x_tx C_tx,b
                + S C_sun,b + o_k ]
          (+ L_w (t_b - t_w))

- ``G``: receiver gain. By default (``--gain smooth``) log G is piecewise linear in
  time, knots every ``--knot-s`` per receiver regime and accumulation length,
  with a Gaussian prior ``--gain-step-sigma`` on each step between knots (memo
  004: the load power holds to 0.85 % rms over 12 h). Free per-window gains
  (``--gain window``, the original ``G_w`` below) let a window whose variation
  the model misses set its gain to zero and its offset to infinity.
- ``G_w``: receiver gain of gain window w (raw counts per GSM kelvin), free per
  window. A window is a stretch of one height era, receiver-regime group and
  accumulation length, at most ``--window-s`` long.
- ``C_GSM``: GSM2016 through the beam (sky part of the design matrix, rows
  normalized by the full-sphere beam integral, so sky + ground weights sum to
  one). Its coefficient is fixed at 1: the gains carry the scale, so every
  other term is in GSM kelvin.
- ``T_g``: one ground temperature below the DEM horizon, ``C_gnd`` the ground
  fraction. A uniform shift of the sky is degenerate with ground and offset
  (sky fraction = 1 - ground fraction), so ``T_g`` is relative to GSM's
  monopole, and the sky corrections start at l = 1.
- ``d_lm``: real spherical harmonics in Galactic coordinates, 1 <= l <= ``--lmax``.
- ``x_tx``: the transmitter, the beam toward it times the fraction of the bin it
  was on (box-gnd transitions).
- ``S``: the Sun's flux (``--sun model``), one value per frequency for the whole
  campaign; ``C_sun`` is the antenna temperature per SFU through the beam, with
  knife-edge diffraction at the terrain edge along the Sun's bearing (``sun.py``).
  The quiet Sun was steady over 07-15 to 07-18 (RSTN, memo 008).
- ``o_k``: offset (receiver noise and anything else additive), per offset
  group. ``--offsets regime``: one per receiver regime, rx-A and rx-B, and one
  per window inside ``rx-transition`` (the regime changed there at an unknown
  time). ``--offsets window``: one per window; then only pointing changes
  within a window constrain ``T_g``.
- ``L_w``: an optional additive drift per window (``--drift``), first order in a
  gain drift.

The fit is nonlinear only through the products G_w x (...). It is solved by
alternating least squares to a starting point and then Levenberg-Marquardt
(``scipy.optimize.least_squares``) on all parameters; the convergence status,
iterations and final cost are recorded.

Noise: radiometer, y / sqrt(dnu tau) over the bin's unmasked samples, plus a
model-error floor ``--floor`` x y in quadrature (without it, parked 60 s bins
at 0.03 % radiometer noise would outweigh the moving data a hundredfold and
the fit would chase parked-night systematics). Residuals are also reported in
radiometer sigma.

Selection, decided before fitting:

- bins with pointing (measured, or filled across a gap where the pointing on
  either side agrees, ``bin_raw.py``), before the elevation drive failed at
  07-18 01:41:38;
- with ``--sun cut``, only bins with the Sun more than ``--sun-margin-deg``
  behind the terrain edge, and no Sun column; ``--sun model`` (the default) keeps
  the daytime bins, which hold the 07-17 raster;
- a window with an offset of its own (all windows with ``--offsets window``;
  the rx-transition windows otherwise) only if its ground fraction changes by
  ``--min-fgnd-spread`` (5-95 percentiles): on a parked window that offset and
  the gain trade almost exactly;
- receiver regimes ``--regimes`` (default rx-A: 07-15 to 07-17 19:41, at
  30, 87.5 and the first 50 min at 91 m);
- gain windows whose level, raw power over C_GSM, is within ``--max-level-ratio``
  of their receiver regime's median (07-15 14:49-16:56 sits 5-10x low);
- gain windows whose raw power the sky follows: a per-window, per-frequency
  regression on (C_GSM, 1, t) must leave a median residual under
  ``--max-window-resid`` of the power, in the median over frequencies (the
  07-15 night does not follow the sky, memo 008 section 7.2).

Validation: bins in alternating ``--block-s`` blocks are held out; the model
is fitted on the rest and scored on them. Gains and per-window terms come
from each window's training bins.

Where ``tcal@v0003`` covers a window, its scale (K at plane P per count)
times the window's gain gives a, the GSM amplitude in tcal's kelvin.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python global_fit.py v0001 --binned <raw_binned dir> --beam hfss --offsets regime
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from scipy.optimize import least_squares

import eigsep_data
import eigsep_sim
from eigsep_base.const import MARJUM_PASS
from eigsep_base.rotations import mount_rotation
from eigsep_sim.design_matrix import HorizonProfile, build_design_matrix
from eigsep_sim.observer import EarthSurface

import sun as sunmod
from common import (
    CHANNEL_MHZ, DEM_PATH, ERAS, HERE, HORIZON_PROFILES, campaign_root, git_rev,
    mount_offsets, sha256,
)
from fit_ground_sky import gsm_maps, load_beam
from raster_sky import real_ylm_maps

FM = (85.0, 110.0)
#: Elevation drive failure (curation/pointing_table.README.md, known defects).
EL_FAILURE = pd.Timestamp("2026-07-18T01:41:38Z").timestamp()
#: Y_lm columns are always built (and cached) to at least this l.
LMAX_CACHE = 2


def windows_of(df, window_s):
    """Gain window of each bin: one era, regime and accumulation length,
    consecutive, at most window_s long."""
    key = (df.era + "|" + df.regime + "|" + df.acc.astype(str)).to_numpy()
    t = df.t.to_numpy()
    w = np.zeros(len(df), int)
    cur, start = 0, t[0]
    for i in range(1, len(df)):
        if key[i] != key[i - 1] or t[i] - start > window_s or t[i] - t[i - 1] > 1800:
            cur += 1
            start = t[i]
        w[i] = cur
    return w


def columns(df, freqs, beam, nside, lmax, campaign, geom, chunk=1500):
    """Per-bin columns at each frequency: GSM, ground fraction, Y_lm,
    transmitter, and the Sun (K per SFU, knife-edge terrain visibility)."""
    az_off, el_off, psi = mount_offsets(campaign)
    eras = sorted(set(df.era))
    hz = [sunmod.true_horizon(HorizonProfile.from_npz(
        campaign / f"{HORIZON_PROFILES}.npz", ERAS[e])) for e in eras]
    hidx = np.array([eras.index(e) for e in df.era])
    lat, lon, hgt = MARJUM_PASS
    t = df.t.to_numpy()
    rg = EarthSurface(lat, lon, hgt).rot_gal2top_stack(Time(t, format="unix")).astype(float)
    rb = mount_rotation(df.az.to_numpy() + az_off, df.el.to_numpy() + el_off, psi)
    gsm = gsm_maps(freqs, nside)
    ylm, labels = real_ylm_maps(nside, lmax) if lmax else (np.zeros((0, gsm.shape[1])), [])
    nf, nb = len(freqs), len(df)
    cG, cN = np.zeros((nf, nb)), np.zeros((nf, nb))
    cY = np.zeros((nf, len(labels), nb))
    for a in range(0, nb, chunk):
        s = slice(a, min(a + chunk, nb))
        dm = build_design_matrix(beam, hz, rg[s], rb[s], nside, horizon_index=hidx[s],
                                 offset_groups=False, nside_int=64)
        Ask = dm.A[:, :, dm.sky]
        cG[:, s] = np.einsum("ftp,fp->ft", Ask, gsm)
        cY[:, :, s] = np.einsum("ftp,kp->fkt", Ask, ylm)
        cN[:, s] = dm.A[:, :, dm.ground][:, :, 0]
        print(f"  columns {s.stop}/{nb}", flush=True)
    tx = sunmod.transmitter_enu()
    gm = np.radians(sunmod.grid_to_true_deg())
    R = np.array([[np.cos(gm), -np.sin(gm), 0], [np.sin(gm), np.cos(gm), 0], [0, 0, 1]])
    cT = np.zeros((nf, nb))
    for e in eras:
        m = (df.era == e).to_numpy()
        v = R @ ((tx - sunmod.antenna_enu(ERAS[e])) / np.linalg.norm(tx - sunmod.antenna_enu(ERAS[e])))
        cT[:, m] = beam(np.einsum("tji,j->it", rb[m], v))
    cT *= df.tx_frac.to_numpy()[None, :]
    cS = sunmod.sun_column(beam, rb, geom, "knife")
    return cG, cN, cY, labels, cT, cS


def cached_columns(key, path, build):
    """Columns from ``path`` if its stored key matches ``key`` exactly, else
    built and stored. Never falls back to a different key."""
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            if str(z["key"]) == key:
                print("  columns from cache", flush=True)
                return (z["cG"], z["cN"], z["cY"], [str(x) for x in z["labels"]], z["cT"], z["cS"])
    cG, cN, cY, labels, cT, cS = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, key=key, cG=cG, cN=cN, cY=cY, labels=np.array(labels, dtype=str), cT=cT, cS=cS)
    return cG, cN, cY, labels, cT, cS


def sun_geometry_bins(df, step_s=120.0):
    """``sun.sun_geometry`` for every bin: the Sun's position exactly, the
    terrain edge along its bearing traced on a step_s grid and interpolated
    (it moves 0.5 deg in 120 s). Below -5 deg altitude the edge is not traced
    and theta is set to -90."""
    t = df.t.to_numpy()
    alt, az = sunmod.sun_altaz(t)
    hz, edge = np.full(t.shape, 90.0), np.full(t.shape, 1000.0)
    for e in sorted(set(df.era)):
        m = (df.era == e).to_numpy()
        grid = np.arange(t[m].min(), t[m].max() + step_s, step_s)
        galt, gaz = sunmod.sun_altaz(grid)
        up = galt > -5
        if not up.any():
            continue
        gh, ge = np.full(grid.shape, 90.0), np.full(grid.shape, 1000.0)
        gh[up], ge[up] = sunmod.trace_horizon(90.0 - gaz[up], ERAS[e])
        hz[m], edge[m] = np.interp(t[m], grid, gh), np.interp(t[m], grid, ge)
    a, b = np.radians(alt), np.radians(90.0 - az)
    enu = np.array([np.cos(a) * np.cos(b), np.cos(a) * np.sin(b), np.sin(a)])
    theta = np.where(alt > -5, alt - hz, -90.0)
    return {"alt": alt, "az": az, "enu": enu, "horizon_deg": hz, "edge_m": edge, "theta_deg": theta}


class Model:
    """The bilinear model at one frequency, with its parameter layout."""

    def __init__(self, cG, X, xnames, win, groups, n_groups, tt, drift, lin_offset=False):
        self.cG, self.X, self.xnames = cG, X, xnames      # X: (nb, nx) shared columns
        self.lin_offset = lin_offset                      # offsets per window (groups == win)
        self.win, self.nw = win, win.max() + 1
        self.groups, self.ng = groups, n_groups
        self.tt, self.drift = tt, drift
        self.nx = X.shape[1]

    def split(self, p):
        i = 0
        th = p[i:i + self.nx]; i += self.nx
        o = p[i:i + self.ng]; i += self.ng
        G = p[i:i + self.nw]; i += self.nw
        L = p[i:i + self.nw] if self.drift else np.zeros(self.nw)
        return th, o, G, L

    def inner(self, th, o):
        return self.cG + self.X @ th + o[self.groups]

    def predict(self, p):
        th, o, G, L = self.split(p)
        return G[self.win] * self.inner(th, o) + L[self.win] * self.tt

    def jac(self, p, rows, w):
        th, o, G, L = self.split(p)
        n = rows.sum()
        g = G[self.win][rows]
        J = [self.X[rows] * g[:, None]]
        Jo = np.zeros((n, self.ng))
        Jo[np.arange(n), self.groups[rows]] = g
        J.append(Jo)
        JG = np.zeros((n, self.nw))
        JG[np.arange(n), self.win[rows]] = self.inner(th, o)[rows]
        J.append(JG)
        if self.drift:
            JL = np.zeros((n, self.nw))
            JL[np.arange(n), self.win[rows]] = self.tt[rows]
            J.append(JL)
        return np.hstack(J) * w[rows][:, None]

    def varpro(self, q, y, w, rows):
        """Variable projection. ``q`` holds the shared parameters (the X
        coefficients, then the group offsets unless they are per window). For
        each window, G (and the window offset and drift, where fitted) solve
        a linear least-squares problem exactly. Returns the weighted residual
        on ``rows`` and the full parameter vector."""
        th = q[:self.nx]
        o = np.zeros(self.ng) if self.lin_offset else q[self.nx:]
        u = self.cG + self.X @ th + (0.0 if self.lin_offset else o[self.groups])
        G, L, o = np.ones(self.nw), np.zeros(self.nw), o.copy()
        r = np.zeros(len(y))
        for k in range(self.nw):
            m = rows & (self.win == k)
            if not m.any():
                continue
            cols = [u[m]] + ([np.ones(m.sum())] if self.lin_offset else []) + ([self.tt[m]] if self.drift else [])
            A = np.stack(cols, 1) * w[m][:, None]
            c, *_ = np.linalg.lstsq(A, y[m] * w[m], rcond=None)
            r[m] = y[m] * w[m] - A @ c
            G[k] = c[0]
            if self.lin_offset:
                o[k] = c[1] / c[0]
            if self.drift:
                L[k] = c[-1]
        return r[rows], np.r_[th, o, G, L if self.drift else []]

    def n_shared(self):
        return self.nx + (0 if self.lin_offset else self.ng)


def gain_basis(t, seg, knot_s):
    """Piecewise-linear basis for log gain: one curve per segment, knots every
    knot_s seconds over the segment's span (knots with no bin within knot_s
    dropped), constant beyond the end knots. Returns (H, knot_t, knot_seg)."""
    cols, kt, ks = [], [], []
    for sname in np.unique(seg):
        m = seg == sname
        ts = t[m]
        k = np.arange(ts.min(), ts.max() + knot_s, knot_s)
        k = k[[np.any(np.abs(ts - x) <= knot_s) for x in k]]
        if k.size == 1:
            k = np.array([ts.min(), ts.max() + 1.0])
        I = np.eye(k.size)
        for j in range(k.size):
            c = np.zeros(len(t))
            c[m] = np.interp(ts, k, I[j])
            cols.append(c)
            kt.append(float(k[j]))
            ks.append(str(sname))
    return np.stack(cols, 1), np.array(kt), np.array(ks)


class SmoothGainModel:
    """y = exp(H a) [C_GSM + X th + o_group]: the gain is smooth in time."""

    def __init__(self, cG, X, xnames, groups, n_groups, H, knot_seg, step_sigma):
        self.cG, self.X, self.xnames = cG, X, xnames
        self.groups, self.ng, self.H = groups, n_groups, H
        self.nx, self.nk = X.shape[1], H.shape[1]
        # Prior on log-gain steps between adjacent knots of one segment.
        rows = [k for k in range(self.nk - 1) if knot_seg[k] == knot_seg[k + 1]]
        self.D = np.zeros((len(rows), self.nk))
        for r, k in enumerate(rows):
            self.D[r, k], self.D[r, k + 1] = -1.0 / step_sigma, 1.0 / step_sigma

    def split(self, p):
        return p[:self.nx], p[self.nx:self.nx + self.ng], p[self.nx + self.ng:]

    def inner(self, th, o):
        return self.cG + self.X @ th + o[self.groups]

    def gain(self, p):
        return np.exp(self.H @ self.split(p)[2])

    def predict(self, p):
        th, o, _ = self.split(p)
        return self.gain(p) * self.inner(th, o)

    def jac(self, p, rows, w):
        th, o, _ = self.split(p)
        g = self.gain(p)[rows]
        n = rows.sum()
        Jo = np.zeros((n, self.ng))
        Jo[np.arange(n), self.groups[rows]] = g
        Ja = (g * self.inner(th, o)[rows])[:, None] * self.H[rows]
        return np.hstack([self.X[rows] * g[:, None], Jo, Ja]) * w[rows][:, None]

    def start(self, y, rows):
        """Gains from y / C_GSM with no ground, offsets or corrections."""
        a, *_ = np.linalg.lstsq(self.H[rows], np.log(y[rows] / self.cG[rows]), rcond=None)
        return np.r_[np.zeros(self.nx + self.ng), a]

    def penalty(self, p):
        return self.D @ self.split(p)[2]

    def fit(self, y, w, rows):
        zero = np.zeros((self.D.shape[0], self.nx + self.ng))
        return least_squares(
            lambda p: np.r_[((self.predict(p) - y) * w)[rows], self.penalty(p)], self.start(y, rows),
            jac=lambda p: np.vstack([self.jac(p, rows, w), np.hstack([zero, self.D])]),
            method="trf", x_scale="jac", xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=3000)


def window_gate(y, cG, win, tt, rows):
    """Median over frequencies of each window's median |residual| / power, for
    raw power regressed on (C_GSM, 1, t) within the window."""
    nw = win.max() + 1
    out = np.full((y.shape[0], nw), np.nan)
    for i in range(y.shape[0]):
        for k in range(nw):
            m = rows & (win == k) & np.isfinite(y[i])
            if m.sum() < 5:
                continue
            A = np.stack([cG[i, m], np.ones(m.sum()), tt[m]], 1)
            c, *_ = np.linalg.lstsq(A, y[i, m], rcond=None)
            out[i, k] = np.median(np.abs(y[i, m] - A @ c)) / np.mean(y[i, m])
    return np.nanmedian(out, 0)


def fit_smooth(out, y0, w, fin, train, sig_rad, X, xnames, cG, groups, gnames, H, knot_t,
               knot_seg, sc, ok_bins, resid_store, i, nbins, step_sigma):
    """Fit the smooth-gain model on the training rows and on all rows."""
    M = SmoothGainModel(cG, X, xnames, groups, len(gnames), H, knot_seg, step_sigma)
    for name, rows in (("train", fin & train), ("all", fin)):
        sol = M.fit(y0, w, rows)
        p = sol.x
        th, o, a = M.split(p)
        r = y0 - M.predict(p)
        dof = rows.sum() - len(p)
        chi = float(np.sum((r * w)[rows] ** 2) / dof)
        J = M.jac(p, rows, w)
        JtJ = J.T @ J
        JtJ[M.nx + M.ng:, M.nx + M.ng:] += M.D.T @ M.D
        err = np.sqrt(np.abs(np.diag(np.linalg.pinv(JtJ)))) * np.sqrt(chi)
        res = {"status": int(sol.status), "nfev": int(sol.nfev), "cost": float(sol.cost),
               "optimality": float(sol.optimality), "cond": float(np.linalg.cond(JtJ)),
               "chi2_per_dof": chi, "n_rows": int(rows.sum()), "n_params": len(p),
               "theta": dict(zip(xnames, th.tolist())),
               "theta_err_scaled": dict(zip(xnames, err[:M.nx].tolist())),
               "offsets": dict(zip(gnames, o.tolist())),
               "offsets_err_scaled": dict(zip(gnames, err[M.nx:M.nx + M.ng].tolist())),
               "rms_frac": float(np.sqrt(np.mean((r / np.where(fin, y0, 1))[rows] ** 2)))}
        # Diverged: a gain knot below 1 % of the median knot, or an offset over
        # 10x the median GSM column (the gain -> 0, offset -> infinity mode
        # the data fall into where the model's variation does not match).
        gk = np.exp(a)
        res["diverged"] = bool(np.any(gk < 0.01 * np.median(gk))
                               or np.any(np.abs(o) > 10 * np.median(cG[rows])))
        if name == "train":
            # Score held-out bins only where every gain knot they lean on has
            # training bins (a stretch inside one held-out block has none).
            sup = (H[rows] > 0.25).sum(0) > 0
            te = fin & ~train & ~np.any((H > 0) & ~sup[None, :], axis=1)
            res["n_test"] = int(te.sum())
            res["test_rms_frac"] = float(np.sqrt(np.mean((r / np.where(fin, y0, 1))[te] ** 2)))
            res["test_chi2"] = float(np.mean((r * w)[te] ** 2))
            res["test_rms_radiometer_sigma"] = float(np.sqrt(np.mean((r / sig_rad)[te] ** 2)))
        else:
            res["gain_knots"] = {"t": knot_t.tolist(), "segment": knot_seg.tolist(),
                                 "gain": np.exp(a).tolist()}
            res["rms_radiometer_sigma"] = float(np.sqrt(np.mean((r / sig_rad)[rows] ** 2)))
            tmp = np.full(nbins, np.nan)
            tmp[np.flatnonzero(ok_bins)[fin]] = (r / y0)[fin]
            resid_store[i] = tmp
            g = M.gain(p)
            ok = np.isfinite(sc) & fin
            # GSM amplitude in tcal's kelvin: gain (counts per GSM K) x tcal scale (K per count).
            res["a_tcal"] = ({"mean": float(np.mean((g * sc)[ok])),
                              "p5_p95": np.percentile((g * sc)[ok], [5, 95]).tolist(),
                              "n_bins": int(ok.sum())} if ok.any() else None)
        out[name] = res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--binned", type=Path, required=True)
    ap.add_argument("--beam", choices=["hfss", "empirical"], default="hfss")
    ap.add_argument("--offsets", choices=["regime", "window"], default="regime")
    ap.add_argument("--lmax", type=int, default=2)
    ap.add_argument("--drift", action="store_true")
    ap.add_argument("--regimes", nargs="+", default=["rx-A"],
                    help="receiver regimes to fit (rx-A, rx-B, rx-transition). rx-transition, where the "
                         "regime changed at an unknown time, and the short rx-B span each need their own "
                         "offset and gain, which their data do not pin down")
    ap.add_argument("--gain", choices=["smooth", "window"], default="smooth",
                    help="log gain piecewise linear in time with knots every --knot-s per receiver "
                         "regime and accumulation length (smooth), or free per window (window)")
    ap.add_argument("--knot-s", type=float, default=10800.0)
    ap.add_argument("--gain-step-sigma", type=float, default=0.05,
                    help="prior on the log-gain step between adjacent knots (memo 004: load power "
                         "steady to 0.85%% rms over 12 h; gain drift ~2.4%% per 3 h)")
    ap.add_argument("--window-s", type=float, default=3600.0)
    ap.add_argument("--floor", type=float, default=0.005)
    ap.add_argument("--block-s", type=float, default=1800.0)
    ap.add_argument("--sun", choices=["model", "cut"], default="model",
                    help="fit the Sun's flux per frequency (model), or drop bins with the "
                         "Sun less than --sun-margin-deg behind the terrain (cut)")
    ap.add_argument("--sun-margin-deg", type=float, default=2.0)
    ap.add_argument("--max-window-resid", type=float, default=0.05)
    ap.add_argument("--max-level-ratio", type=float, default=2.0,
                    help="drop windows whose raw/C_GSM level is this far from their regime's median")
    ap.add_argument("--min-fgnd-spread", type=float, default=0.03,
                    help="windows with an offset of their own need this 5-95%% spread in ground fraction")
    ap.add_argument("--fmin", type=float, default=50.0)
    ap.add_argument("--fmax", type=float, default=235.0)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--only-mhz", type=float, nargs="*", help="fit only these frequencies (diagnostics)")
    ap.add_argument("--nside", type=int, default=8)
    ap.add_argument("--label", default=None, help="name of this configuration in the output")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    campaign = campaign_root()
    label = args.label or f"{args.beam}_{args.offsets}_l{args.lmax}{'_drift' if args.drift else ''}"
    out_dir = args.out_dir or campaign / "derived/ground_sky" / f"global_{args.version}"
    if (out_dir / f"{label}.json").exists():
        sys.exit("exists; versions are immutable")
    z = dict(np.load(args.binned / "binned.npz", allow_pickle=False))
    nb = len(z["t"])
    df = pd.DataFrame({k: z[k] for k in z if np.ndim(z[k]) == 1 and len(z[k]) == nb})
    keep = np.isfinite(df.az.to_numpy()) & np.isfinite(df.el.to_numpy()) & df.era.isin(list(ERAS)).to_numpy()
    df = df[keep].reset_index(drop=True)
    # The accumulation length, from the integration time (0.268 s or 0.537 s;
    # it doubled at 07-15 15:55); bin_raw's ``acc`` column is not filled.
    df["acc"] = np.rint(df.tau.to_numpy() * 1e3).astype(int)
    import hashlib
    gkey = hashlib.sha256(json.dumps({"binned": sha256(args.binned / "binned.npz"),
                                      "bins": hashlib.sha256(df.t.to_numpy().tobytes()).hexdigest(),
                                      "eras": sorted(set(df.era)), "dem": DEM_PATH,
                                      "horizon": HORIZON_PROFILES}).encode()).hexdigest()
    gpath = args.binned / "column_cache" / f"sun_{gkey[:16]}.npz"
    geom = None
    if gpath.exists():
        with np.load(gpath) as zg:
            if str(zg["key"]) == gkey:
                geom = {k: zg[k] for k in zg.files if k != "key"}
    if geom is None:
        geom = sun_geometry_bins(df)
        gpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(gpath, key=gkey, **geom)
    hidden = geom["theta_deg"] < -args.sun_margin_deg
    sel_counts = {"bins": int(nb), "with_pointing": int(keep.sum()), "sun_hidden": int(hidden.sum())}
    # --sun cut drops the Sun-visible bins from the fit, not from the columns,
    # so every configuration shares one column cache.
    fit_mask = hidden if args.sun == "cut" else np.ones(len(df), bool)
    # The elevation drive failed at 07-18 01:41:38 UTC (pointing_table README,
    # known defects); the elevation after it is not known. A fit mask, like
    # the Sun cut, so the column cache is shared.
    fit_mask = fit_mask & (df.t.to_numpy() < EL_FAILURE)
    fit_mask &= df.regime.isin(args.regimes).to_numpy()
    sel_counts["before_el_failure"] = int((df.t.to_numpy() < EL_FAILURE).sum())
    Yall, nsamp = z["y"][keep], z["nsamp"][keep]
    scale = z["scale"][keep]

    beam, beam_path = load_beam(args.beam, campaign)
    bf = beam.freqs_hz / 1e6
    # Frequencies from the binned list (the HFSS slices), so every beam is
    # fitted at the same ones; then those the beam has (and, for the
    # empirical beam, constrains).
    grid = [x for x in z["freqs_mhz"] if args.fmin <= x <= args.fmax and not FM[0] < x < FM[1]][::args.every]
    use = []
    for x in grid:
        j = int(np.argmin(np.abs(bf - x)))
        if abs(bf[j] - x) < 1e-3 and (args.beam == "hfss" or beam.meta["empirical_frequency_mask"][j]):
            use.append(j)
    if args.only_mhz:
        use = [j for j in use if np.min(np.abs(bf[j] - np.array(args.only_mhz))) < 1.0]
    freqs = bf[use]
    beam = beam.select(use)
    fi = [int(np.argmin(np.abs(z["freqs_mhz"] - x))) for x in freqs]
    assert np.allclose(z["freqs_mhz"][fi], freqs)
    Y, NS, SC = Yall[:, fi].T, nsamp[:, fi].T, scale[:, fi].T

    win = windows_of(df, args.window_s)
    t = df.t.to_numpy()
    tmid = np.array([t[win == k].mean() for k in range(win.max() + 1)])
    tt = (t - tmid[win]) / 3600.0
    blocks = ((t - t.min()) // args.block_s).astype(int)
    train = blocks % 2 == 0
    print(f"{len(df)} bins, {win.max() + 1} windows, {len(freqs)} frequencies", flush=True)

    # The columns are cached with Y_lm to at least LMAX_CACHE, so runs at a
    # lower lmax share one cache; the key holds everything that defines them.
    lmax_c = max(args.lmax, LMAX_CACHE)
    key = hashlib.sha256(json.dumps({
        "binned": sha256(args.binned / "binned.npz"), "beam": sha256(beam_path),
        "freqs": [round(float(x), 6) for x in freqs], "nside": args.nside, "lmax": lmax_c,
        "bins": hashlib.sha256(df.t.to_numpy().tobytes()).hexdigest(),
        "mount": mount_offsets(campaign), "horizon": HORIZON_PROFILES,
        "tx": [round(float(x), 3) for x in sunmod.transmitter_enu()]}).encode()).hexdigest()
    cG, cN, cY, ylabels, cT, cS = cached_columns(
        key, args.binned / "column_cache" / f"{key[:16]}.npz",
        lambda: columns(df, freqs, beam, args.nside, lmax_c, campaign, geom))
    n_lm = (args.lmax + 1) ** 2 - 1          # real Y_lm are ordered by l
    cY, ylabels = cY[:, :n_lm], ylabels[:n_lm]

    gate = window_gate(Y, cG, win, tt, fit_mask)
    good_w = gate < args.max_window_resid       # NaN (no bins to test) is not kept
    # Level gate: raw power over GSM through the beam, per window (median over
    # bins and frequencies), against the median of its receiver regime. The sky gate tests
    # only each window's shape; on 07-15 14:49-16:56 the level is 5-10x low.
    # (Raw counts do not scale with the accumulation length: 07-15 00-03,
    # before it doubled, sits at the later level.)
    lev_b = np.nanmedian(Y / cG, axis=0)
    lev_w = np.array([np.nanmedian(lev_b[(win == k) & fit_mask]) if ((win == k) & fit_mask).any() else np.nan
                      for k in range(win.max() + 1)])
    wreg0 = np.array([df.regime[win == k].iloc[0] for k in range(win.max() + 1)])
    lev_ratio = np.array([lev_w[k] / np.nanmedian(lev_w[wreg0 == wreg0[k]]) for k in range(win.max() + 1)])
    level_bad = ~(np.abs(np.log(lev_ratio)) < np.log(args.max_level_ratio))
    good_w &= ~level_bad
    # A window with an offset of its own (every window with --offsets window;
    # rx-transition windows with --offsets regime) can separate that offset
    # from its gain only if its ground fraction changes: on a parked window
    # the two trade almost exactly. Such windows need a 5-95 percentile spread
    # in ground fraction (at the middle frequency) of --min-fgnd-spread.
    imid = len(freqs) // 2
    nwin = win.max() + 1
    spread = np.array([np.ptp(np.percentile(cN[imid][(win == k) & fit_mask], [5, 95]))
                       if ((win == k) & fit_mask).any() else 0.0 for k in range(nwin)])
    wreg = np.array([df.regime[win == k].iloc[0] for k in range(nwin)])
    own = np.full(nwin, args.offsets == "window") | ~np.isin(wreg, ["rx-A", "rx-B"])
    static_own = own & (spread < args.min_fgnd_spread)
    good_w &= ~static_own
    ok_bins = good_w[win] & fit_mask
    # Renumber windows over the bins kept.
    kept_w = np.unique(win[ok_bins])
    remap = -np.ones(win.max() + 1, int)
    remap[kept_w] = np.arange(kept_w.size)
    wk = remap[win[ok_bins]]

    reg = df.regime.to_numpy()[ok_bins]
    if args.offsets == "window":
        groups, gnames = wk, [f"w{k}" for k in range(wk.max() + 1)]
    else:
        names = []
        for r_, k in zip(reg, wk):
            names.append(r_ if r_ in ("rx-A", "rx-B") else f"{r_}:w{k}")
        gnames = sorted(set(names))
        groups = np.array([gnames.index(n) for n in names])

    if args.gain == "smooth":
        seg = (df.regime + "|" + df.acc.astype(str)).to_numpy()[ok_bins]
        H, knot_t, knot_seg = gain_basis(t[ok_bins], seg, args.knot_s)
        print(f"  smooth gain: {H.shape[1]} knots in {len(set(seg))} segments", flush=True)
    else:
        H = knot_t = knot_seg = None
    results = []
    resid_store = np.full(Y.shape, np.nan)
    for i, x in enumerate(freqs):
        y = Y[i][ok_bins]
        fin = np.isfinite(y) & (y > 0)
        sig_rad = y / np.sqrt(CHANNEL_MHZ * 1e6 * df.tau.to_numpy()[ok_bins] * np.maximum(NS[i][ok_bins], 1))
        sig = np.sqrt(sig_rad**2 + (args.floor * y) ** 2)
        w = np.where(fin, 1 / sig, 0.0)
        y0 = np.where(fin, y, 0.0)
        cols = [cN[i][ok_bins], *cY[i][:, ok_bins]]
        xnames = ["T_gnd"] + [f"d_{s}" for s in ylabels]
        if np.any(cT[i][ok_bins] != 0):
            cols.append(cT[i][ok_bins]); xnames.append("tx")
        if args.sun == "model" and np.any(cS[i][ok_bins] > 0):
            cols.append(cS[i][ok_bins]); xnames.append("S_sun")
        X = np.stack(cols, 1)
        out = {"freq_mhz": float(x)}
        if args.gain == "smooth":
            fit_smooth(out, y0, w, fin, train[ok_bins], sig_rad, X, xnames, cG[i][ok_bins], groups,
                       gnames, H, knot_t, knot_seg, SC[i][ok_bins], ok_bins, resid_store, i, len(df),
                       args.gain_step_sigma)
            results.append(out)
            a_ = out["all"]
            print(f"{x:6.1f} MHz  T_gnd {a_['theta']['T_gnd']:8.1f}  offsets "
                  + " ".join(f"{k}:{v:.0f}" for k, v in list(a_["offsets"].items())[:4])
                  + f"  chi2/dof {a_['chi2_per_dof']:.1f}  rms {100 * a_['rms_frac']:.2f}%"
                  f"  held-out rms {100 * out['train']['test_rms_frac']:.2f}%  status {a_['status']}"
                  + ("  DIVERGED" if a_["diverged"] or out["train"]["diverged"] else ""), flush=True)
            continue
        M = Model(cG[i][ok_bins], X, xnames, wk, groups, len(gnames), tt[ok_bins], args.drift,
                  lin_offset=args.offsets == "window")
        for name, rows in (("train", fin & train[ok_bins]), ("all", fin)):
            sol = least_squares(lambda q: M.varpro(q, y0, w, rows)[0], np.zeros(M.n_shared()),
                                method="lm", x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12,
                                max_nfev=20000)
            p = M.varpro(sol.x, y0, w, rows)[1]
            th, o, G, L = M.split(p)
            r = (y0 - M.predict(p))
            dof = rows.sum() - len(p)
            chi = float(np.sum((r * w)[rows] ** 2) / dof)
            # Errors, scaled by chi2/dof. Shared parameters: from the projected
            # Jacobian, which marginalizes the per-window gains; an unconstrained
            # parameter shows as a huge error. Per-window offsets (--offsets
            # window): from the full Jacobian.
            JtJ = sol.jac.T @ sol.jac
            err_q = np.sqrt(np.abs(np.diag(np.linalg.pinv(JtJ)))) * np.sqrt(chi)
            cond = float(np.linalg.cond(JtJ))
            if M.lin_offset:
                J = M.jac(p, rows, w)
                err_o = np.sqrt(np.abs(np.diag(np.linalg.pinv(J.T @ J))))[M.nx:M.nx + M.ng] * np.sqrt(chi)
                err = np.r_[err_q[:M.nx], err_o]
            else:
                err = err_q
            res = {"status": int(sol.status), "nfev": int(sol.nfev), "cost": float(sol.cost),
                   "optimality": float(sol.optimality), "cond_shared": cond,
                   "chi2_per_dof": chi, "n_rows": int(rows.sum()), "n_params": len(p),
                   "theta": dict(zip(M.xnames, th.tolist())),
                   "theta_err_scaled": dict(zip(M.xnames, err[:M.nx].tolist())),
                   "offsets": dict(zip(gnames, o.tolist())),
                   "offsets_err_scaled": dict(zip(gnames, err[M.nx:].tolist())),
                   "rms_frac": float(np.sqrt(np.mean((r / y0)[rows] ** 2)))}
            if name == "train":
                te = fin & ~train[ok_bins]
                te &= (np.bincount(wk[rows], minlength=M.nw) > 0)[wk]   # windows with a fitted gain
                res["test_rms_frac"] = float(np.sqrt(np.mean((r / np.where(fin, y0, 1))[te] ** 2)))
                res["test_chi2"] = float(np.mean((r * w)[te] ** 2))
                res["test_rms_radiometer_sigma"] = float(np.sqrt(np.mean((r / sig_rad)[te] ** 2)))
            else:
                res["gain"] = G.tolist()
                res["rms_radiometer_sigma"] = float(np.sqrt(np.mean((r / sig_rad)[rows] ** 2)))
                tmp = np.full(len(df), np.nan)
                tmp[np.flatnonzero(ok_bins)[fin]] = (r / y0)[fin]
                resid_store[i] = tmp
                # GSM amplitude in tcal kelvin, per window with tcal: a = G x scale.
                sc = SC[i][ok_bins]
                a = [float(np.nanmean(sc[wk == k]) * G[k]) for k in range(len(G))
                     if np.isfinite(sc[wk == k]).mean() > 0.5]
                res["a_tcal_windows"] = a
            out[name] = res
        results.append(out)
        a_ = out["all"]
        print(f"{x:6.1f} MHz  T_gnd {a_['theta']['T_gnd']:8.1f}  offsets "
              + " ".join(f"{k}:{v:.0f}" for k, v in list(a_["offsets"].items())[:3])
              + f"  chi2/dof {a_['chi2_per_dof']:.1f}  rms {100 * a_['rms_frac']:.2f}%"
              f"  held-out rms {100 * out['train']['test_rms_frac']:.2f}%  status {a_['status']}",
              flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    win_info = [{"window": int(k), "t0": float(t[win == k].min()), "t1": float(t[win == k].max()),
                 "era": df.era[win == k].iloc[0], "regime": df.regime[win == k].iloc[0],
                 "n_bins": int((win == k).sum()), "gate_resid": float(gate[k]),
                 "fgnd_spread": float(spread[k]), "dropped_static_own_offset": bool(static_own[k]),
                 "level_ratio": float(lev_ratio[k]), "dropped_level": bool(level_bad[k]),
                 "kept": bool(remap[k] >= 0), "fit_index": int(remap[k])} for k in range(win.max() + 1)]
    np.savez_compressed(out_dir / f"{label}_resid.npz", freqs_mhz=freqs, t=t, resid_frac=resid_store,
                        kept=ok_bins, window=win, era=df.era.to_numpy().astype(str),
                        regime=df.regime.to_numpy().astype(str), az=df.az.to_numpy(), el=df.el.to_numpy(),
                        cGSM=cG, cGnd=cN)
    (out_dir / f"{label}.json").write_text(json.dumps({
        "provenance": {
            "product": "ground_sky_global", "version": args.version, "label": label,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "data-analysis/scripts/marjum-2026-07/ground_sky/global_fit.py",
            "argv": sys.argv[1:],
            "code": {"data-analysis": git_rev(HERE), "eigsep_data": git_rev(eigsep_data.__path__[0]),
                     "eigsep_sim": git_rev(Path(eigsep_sim.__file__).parent)},
            "inputs": {"beam_sha256": sha256(beam_path),
                       "binned_sha256": sha256(args.binned / "binned.npz"),
                       "mount_offsets_deg": mount_offsets(campaign),
                       "horizon_profiles": HORIZON_PROFILES, "dem": DEM_PATH}},
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                   if k not in ("version", "out_dir")},
        "selection": sel_counts | {"windows": int(win.max() + 1), "windows_kept": int(kept_w.size),
                                   "windows_dropped_static_own_offset": int(static_own.sum()),
                                   "windows_dropped_level": int(level_bad.sum()),
                                   "bins_fitted": int(ok_bins.sum())},
        "offset_groups": gnames, "windows": win_info, "ylm_labels": ylabels,
        "results": results}, indent=1))


if __name__ == "__main__":
    main()
