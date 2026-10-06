"""Is the calibrated excess around the 07-17 terrain sunrise the Sun?

Reads a ``fit_ground_sky.py`` output fitted with ``--sun cut-flat``, so every
bin with the Sun above 0 deg is out of sample, and tests its residual (data
minus fitted sky, ground and offset) in a local window around the time the
Sun clears the ridge (07-17 15:55 UTC at 87.5 m, ``sun.py``).

The residual drifts by up to ~100 K over hours (the night model extrapolated
in LST), so each frequency is fitted locally:

    resid(t) = c0 + c1 (t - t_ref) + S_sun * A_sun(t - tau)

on **transmitter-off bins only** (box-gnd transitions,
``curation/transmitter_transitions_boxgnd.jsonl``). In this window that is
everything before 15:36:22 (the Sun 4-12 deg behind the ridge) and
16:14:18-16:22:11 (the Sun 4-5 deg above it). Transmitter-on bins are
shown but never fitted. ``A_sun`` is evaluated on a dense grid from the
geometry (``sun.sun_column`` at the window's fixed pointing), so it can be
shifted by ``tau`` without running off the sampled bins.

Checks that could come out the other way:

1. **No step at 0 deg altitude.** A flat-horizon Sun (visible from 12:26)
   predicts a rise at 12:26 and a smooth climb after it; the same local fit
   with the flat-horizon column (window moved to bracket 12:26) should give
   S_sun consistent with zero if the ridge hides the Sun.
2. **Timing.** chi2(tau) over the transmitter-off bins should be smallest
   near tau = 0, not at a shift that would mean the excess has another cause.
3. **Spectrum.** S_sun(nu) should be positive and smooth, of the order of the
   quiet-to-active Sun (a few to ~100 SFU over 50-230 MHz).

Errors on S_sun are scaled by sqrt(chi2/dof) of the local fit. Anything that
changes in the same few minutes as the Sun (the box warming in sunlight, the
ground under the antenna being lit) is not separated by this test; a result
here is "consistent with" the Sun, not an identification.

It also lists every terrain and 0-deg sunrise and sunset during each height
era of phase C (``sun.terrain_events``; era spans from
``curation/mode_table.jsonl``).

Writes ``sun_events.json``, ``sun_events.npz`` (everything plotted) and
``sun_events.png`` into ``--out-dir``.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python sun_events.py FIT_DIR --out-dir OUT
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sun as sunmod
from common import ERAS, campaign_root
from fit_ground_sky import load_beam

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#2f9e59"
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#c9c8c4"


def ts(iso):
    return pd.Timestamp(iso).timestamp()


def tx_on_bins(t, half_bin_s, campaign):
    """True for bins overlapping any box-gnd transmitter episode."""
    on = np.zeros(len(t), bool)
    for line in open(campaign / "curation/transmitter_transitions_boxgnd.jsonl"):
        ep = json.loads(line)
        if "t_start_utc" not in ep:
            continue
        a, b = ts(ep["t_start_utc"]), ts(ep["t_end_utc"])
        on |= (t + half_bin_s >= a) & (t - half_bin_s <= b)
    return on


def dense_columns(beam, az, el, era_key, t_grid, az_offset, psi):
    """Sun columns (K/SFU) on a dense grid at fixed pointing: knife, flat."""
    from eigsep_base.rotations import mount_rotation

    rot = np.broadcast_to(mount_rotation(az + az_offset, el, psi),
                          (len(t_grid), 3, 3))
    geom = sunmod.sun_geometry(t_grid, era_key)
    return (sunmod.sun_column(beam, rot, geom, "knife"),
            sunmod.sun_column(beam, rot, geom, "flat"), geom)


def local_fit(t, y, s, col_fn, tau, t_ref):
    """Weighted fit of c0 + c1 (t - t_ref) + S col(t - tau)."""
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
    if ok.sum() < 5:
        return None
    c = col_fn(t[ok] - tau)
    A = np.stack([np.ones(ok.sum()), (t[ok] - t_ref) / 3600, c], 1) / s[ok, None]
    b = y[ok] / s[ok]
    cov = np.linalg.pinv(A.T @ A)
    x = cov @ A.T @ b
    chi2 = float(np.sum((b - A @ x) ** 2))
    dof = max(int(ok.sum()) - 3, 1)
    return {"S": float(x[2]), "sigma_S": float(np.sqrt(cov[2, 2] * max(chi2 / dof, 1))),
            "c0": float(x[0]), "c1": float(x[1]), "chi2": chi2, "dof": dof,
            "n": int(ok.sum())}


def campaign_events(campaign):
    """Terrain and 0-deg sunrises and sunsets while each height era was flown."""
    rows = [json.loads(line) for line in open(campaign / "curation/mode_table.jsonl")]
    rows = pd.DataFrame([r for r in rows if "file_first" in r and r.get("phase") == "C"])
    out = []
    for era, g in rows.groupby("height_era"):
        if era not in ERAS:
            continue
        t0, t1 = ts(g.t_start_utc.min()), ts(g.t_end_utc.max())
        for e in sunmod.terrain_events(t0, t1, ERAS[era]):
            e["era"] = ERAS[era]
            e["utc"] = pd.Timestamp(e["t_unix"], unit="s").isoformat()
            out.append(e)
    return sorted(out, key=lambda e: e["t_unix"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("fit_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--window", nargs=2, default=["2026-07-17T14:30:00Z",
                                                  "2026-07-17T16:23:00Z"],
                    help="UTC window around the terrain sunrise")
    ap.add_argument("--null-window", nargs=2, default=["2026-07-17T11:30:00Z",
                                                       "2026-07-17T13:30:00Z"],
                    help="UTC window bracketing the 0-deg sunrise (12:26)")
    ap.add_argument("--taus-min", type=float, nargs=3, default=[-40, 40, 1])
    ap.add_argument("--bin-s", type=float, default=60.0)
    ap.add_argument("--plot-mhz", type=float, nargs="+",
                    default=[74.2, 113.3, 148.4, 175.8, 199.2, 218.8])
    args = ap.parse_args()

    campaign = campaign_root()
    z = np.load(args.fit_dir / "fit.npz", allow_pickle=True)
    man = json.loads((args.fit_dir / "manifest.json").read_text())
    if man["params"]["sun"] != "cut-flat":
        raise SystemExit("needs a fit made with --sun cut-flat")
    freqs, t = z["freqs_mhz"], z["t"]
    resid, sig = z["residual_k"], z["sigma_k"]
    tx = tx_on_bins(t, args.bin_s / 2, campaign)

    w0, w1 = map(ts, args.window)
    win = (t >= w0) & (t <= w1)
    eras = set(z["era"][win])
    if len(eras) != 1 or np.ptp(z["az"][win]) > 3 or np.ptp(z["el"][win]) > 3:
        raise SystemExit(f"window must be one era and one pointing (az, el within 3 deg): {eras}")
    era_key = ERAS[eras.pop()]
    az, el = float(np.median(z["az"][win])), float(np.median(z["el"][win]))

    beam, _ = load_beam(man["provenance"]["inputs"]["beam"]["name"], campaign)
    bidx = [int(np.argmin(np.abs(beam.freqs_hz / 1e6 - f))) for f in freqs]
    beam = beam.select(bidx)
    geo = man["geometry"]
    n0, n1 = map(ts, args.null_window)
    grid = np.arange(min(n0, w0) - 3 * 3600, w1 + 3 * 3600, 30.0)
    col_k, col_f, geom = dense_columns(beam, az, el, era_key, grid,
                                       geo["az_offset_deg"], geo["psi_deg"])
    rise = grid[np.argmax(geom["theta_deg"] > 0)]
    alt0 = grid[np.argmax(geom["alt"] > 0)]

    def interp(col, fi):
        return lambda tt: np.interp(tt, grid, col[fi])

    taus = np.arange(args.taus_min[0], args.taus_min[1] + 1e-9,
                     args.taus_min[2]) * 60
    use = win & ~tx
    nul = (t >= n0) & (t <= n1) & ~tx
    out = {"fit_dir": str(args.fit_dir), "window_utc": args.window,
           "null_window_utc": args.null_window,
           "pointing_table_az_el_deg": [az, el], "era": era_key,
           "ridge_clear_utc": pd.Timestamp(rise, unit="s").isoformat(),
           "sun_alt0_utc": pd.Timestamp(alt0, unit="s").isoformat(),
           "bins": {"window": int(win.sum()), "tx_off_fitted": int(use.sum()),
                    "tx_off_sun_above_ridge": int((use & (t > rise)).sum()),
                    "null_window_tx_off": int(nul.sum())},
           "per_frequency": []}
    chi2_tau = np.full((len(freqs), len(taus)), np.nan)
    for fi, f in enumerate(freqs):
        y, s = resid[fi], sig[fi]
        r = local_fit(t[use], y[use], s[use], interp(col_k, fi), 0.0, rise)
        rf = local_fit(t[nul], y[nul], s[nul], interp(col_f, fi), 0.0, alt0)
        for k, tau in enumerate(taus):
            rt = local_fit(t[use], y[use], s[use], interp(col_k, fi), tau, rise)
            chi2_tau[fi, k] = rt["chi2"] if rt else np.nan
        # Out-of-sample: transmitter-on bins against the TX-off fit.
        txw = win & tx
        pred_on = (r["c0"] + r["c1"] * (t[txw] - rise) / 3600
                   + r["S"] * interp(col_k, fi)(t[txw])) if r else None
        out["per_frequency"].append({
            "freq_mhz": float(f), "terrain": r, "flat_null": rf,
            "tau_best_min": (float(taus[np.nanargmin(chi2_tau[fi])] / 60)
                             if np.isfinite(chi2_tau[fi]).any() else None),
            "tx_on_excess_k_median": (float(np.nanmedian(y[txw] - pred_on))
                                      if r is not None and txw.any() else None),
        })
    norm = chi2_tau - np.nanmin(chi2_tau, axis=1, keepdims=True)
    good = np.isfinite(norm).all(axis=1)
    dchi = np.nanmean(norm[good] / np.array(
        [max(p["terrain"]["chi2"] / p["terrain"]["dof"], 1)
         for p, g in zip(out["per_frequency"], good) if g])[:, None], axis=0)
    out["taus_min"] = (taus / 60).tolist()
    out["mean_scaled_delta_chi2_vs_tau"] = dchi.tolist()
    out["tau_best_min_all"] = float(taus[np.argmin(dchi)] / 60)
    out["events"] = campaign_events(campaign)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sun_events.json").write_text(json.dumps(out, indent=1))
    pf = out["per_frequency"]

    def arr(key, sub):
        return np.array([p[key][sub] if p[key] else np.nan for p in pf])

    np.savez_compressed(
        args.out_dir / "sun_events.npz",
        freqs_mhz=freqs, t=t, tx_on=tx, window=win, fitted_local=use,
        null_fitted=nul, residual_k=resid, sigma_k=sig,
        grid_t=grid, sun_column_knife=col_k, sun_column_flat=col_f,
        grid_alt=geom["alt"], grid_az=geom["az"], grid_theta=geom["theta_deg"],
        grid_horizon=geom["horizon_deg"], grid_edge_m=geom["edge_m"],
        ridge_clear_t=rise, alt0_t=alt0,
        S=arr("terrain", "S"), sigma_S=arr("terrain", "sigma_S"),
        c0=arr("terrain", "c0"), c1=arr("terrain", "c1"),
        chi2=arr("terrain", "chi2"), dof=arr("terrain", "dof"),
        S_null=arr("flat_null", "S"), sigma_S_null=arr("flat_null", "sigma_S"),
        tau_best_min=np.array([p["tau_best_min"] if p["tau_best_min"] is not None
                               else np.nan for p in pf]),
        taus_min=taus / 60, chi2_tau=chi2_tau, mean_scaled_dchi2=dchi,
        gsm_amplitude=z["gsm_amplitude"])
    for p in out["per_frequency"]:
        r, rf = p["terrain"], p["flat_null"]
        if r is None:
            continue
        print(f"{p['freq_mhz']:7.2f} MHz  S_sun {r['S']:7.1f} ± {r['sigma_S']:5.1f} SFU"
              f" (chi2/dof {r['chi2'] / r['dof']:7.1f}, n {r['n']})  flat-horizon null "
              + (f"{rf['S']:7.1f} ± {rf['sigma_S']:5.1f}" if rf else "  n/a")
              + f"  tau* {p['tau_best_min']:+.0f} min  TX-on excess "
              f"{p['tx_on_excess_k_median']:+.1f} K")
    print(json.dumps(out["bins"]), "ridge", out["ridge_clear_utc"],
          "tau best", out["tau_best_min_all"])
    plot(args, z, t, tx, win, freqs, out, grid, col_k, rise, taus, dchi)


def plot(args, z, t, tx, win, freqs, out, grid, col_k, rise, taus, dchi):
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    sel = list(dict.fromkeys(int(np.argmin(np.abs(freqs - f))) for f in args.plot_mhz))
    fig, axes = plt.subplots(len(sel) + 2, 1, figsize=(7.5, 1.8 * len(sel) + 4.5))
    xm = lambda tt: (tt - rise) / 60  # noqa: E731
    gw = (grid >= t[win].min()) & (grid <= t[win].max())
    for ax, fi in zip(axes, sel):
        p = out["per_frequency"][fi]
        r = p["terrain"]
        y, s = z["residual_k"][fi], z["sigma_k"][fi]
        off, on = win & ~tx, win & tx
        ax.errorbar(xm(t[off]), y[off], s[off], fmt="o", ms=2.5, lw=0.6,
                    color=INK, ecolor=FAINT, label="transmitter off (fitted)")
        ax.plot(xm(t[on]), y[on], "o", ms=2.5, mfc="none", color=ORANGE, mew=0.7,
                label="transmitter on (not fitted)")
        if r:
            base = r["c0"] + r["c1"] * (grid[gw] - rise) / 3600
            ax.plot(xm(grid[gw]), base, color=MUTED, lw=0.8, ls="--")
            ax.plot(xm(grid[gw]), base + r["S"] * col_k[fi, gw], color=BLUE, lw=1.2,
                    label=f"baseline + Sun, {r['S']:.0f} ± {r['sigma_S']:.0f} SFU")
        ax.axvline(0, color=INK, lw=0.6, ls=":")
        ax.set_ylabel(f"{freqs[fi]:.1f} MHz\nresidual (K)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=6, loc="upper left")
    axes[len(sel) - 1].set_xlabel("minutes from the Sun clearing the ridge (07-17 15:55 UTC, 87.5 m)")
    ax = axes[-2]
    pf = [p for p in out["per_frequency"] if p["terrain"]]
    f = np.array([p["freq_mhz"] for p in pf])
    ax.errorbar(f, [p["terrain"]["S"] for p in pf], [p["terrain"]["sigma_S"] for p in pf],
                fmt="o", ms=3, color=BLUE, label="terrain Sun, around 15:55 (TX off)")
    pn = [p for p in pf if p["flat_null"]]
    ax.errorbar([p["freq_mhz"] for p in pn], [p["flat_null"]["S"] for p in pn],
                [p["flat_null"]["sigma_S"] for p in pn], fmt="s", ms=3, color=ORANGE,
                label="flat-horizon Sun, around 12:26 (should be 0)")
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel("S_sun (SFU)")
    ax.set_xlabel("frequency (MHz)")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[-1]
    ax.plot(taus / 60, dchi, color=BLUE)
    ax.axvline(0, color=INK, lw=0.6, ls=":")
    ax.set_xlabel("shift τ of the Sun column (min; >0 = later than the DEM predicts)")
    ax.set_ylabel("mean Δχ²/(χ²/dof)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out_dir / "sun_events.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
