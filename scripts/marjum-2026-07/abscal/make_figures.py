#!/usr/bin/env python3
"""Residual and diagnostic figures for MEMO-003 / MEMO-004.

Every panel states its band and whether the quantity plotted is held-out or
in-sample, because a residual without that label is not evidence.

    python make_figures.py            # writes figs/*.png
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"

# Palette: validated (light mode, surface #fcfcfb) via the dataviz validator.
# All checks PASS; aqua carries a contrast WARN so it is only used with a
# visible direct label.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": "#e6e5e1",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 10,
    "axes.titlesize": 11,
    "legend.frameon": False,
    "figure.dpi": 150,
})


def recess(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def dt(ts):
    return [datetime.fromtimestamp(t, timezone.utc) for t in ts]


# ---------------------------------------------------------------- figure 1
def fig_closure(z, fmin, fmax):
    """Cross-closure residual vs frequency. HELD OUT: cycle i's solution is
    applied to cycle i+1's ambient load, which that solution never saw."""
    f, g, trx = z["freqs"], z["gain"], z["t_rx"]
    t_amb, p_amb = z["t_amb"], z["p_amb"]
    band = (f >= fmin) & (f <= fmax)
    fb = f[band]

    res = []
    for i in range(len(g) - 1):
        rec = p_amb[i + 1][band] / g[i][band] - trx[i][band]
        res.append(rec - t_amb[i + 1])
    res = np.array(res)

    fig, (a0, a1) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), height_ratios=[2, 1.5], sharex=True)

    for r in res:
        a0.plot(fb, r, color=BLUE, lw=0.6, alpha=0.18)
    med = np.nanmedian(res, axis=0)
    a0.plot(fb, med, color=BLUE, lw=2.0, label="median of 37 pairs")
    a0.axhline(0, color=INK2, lw=1.0, ls=":")
    rms = np.nanstd(res)
    a0.axhspan(-rms, rms, color=ORANGE, alpha=0.12)
    a0.text(fb[-1], rms, f"  ±{rms:.2f} K rms", color=ORANGE,
            va="bottom", ha="right", fontsize=9)
    a0.set_ylabel("closure residual  [K]")
    a0.set_title("Cross-closure residual — HELD OUT\n"
                 "cycle $i$ solution applied to cycle $i{+}1$ ambient load",
                 loc="left")
    a0.legend(loc="lower left")
    recess(a0)

    im = a1.imshow(res, aspect="auto", origin="lower", cmap="RdBu_r",
                   vmin=-5, vmax=5,
                   extent=[fb[0], fb[-1], 0, len(res)])
    a1.set_xlabel(f"frequency  [MHz]   (band {fmin:.0f}–{fmax:.0f} MHz)")
    a1.set_ylabel("adjacent pair index")
    cb = fig.colorbar(im, ax=a1, pad=0.01)
    cb.set_label("residual  [K]")
    cb.outline.set_visible(False)
    recess(a1)

    fig.tight_layout()
    out = FIGS / "fig1_closure_residual.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}: rms {rms:.2f} K over {len(res)} pairs")
    return out


# ---------------------------------------------------------------- figure 2
def fig_spectrum(z, fmin, fmax):
    """Calibrated spectrum + power-law residual. IN-SAMPLE: the power law is
    fitted to this same median spectrum. It is a shape diagnostic, not a
    validation of the calibration."""
    f, T = z["freqs"], z["T_ant"]
    band = (f >= fmin) & (f <= fmax)
    fb = f[band]
    Tm = np.nanmedian(T, axis=0)[band]
    ok = np.isfinite(Tm) & (Tm > 0)
    cf = np.polyfit(np.log(fb[ok]), np.log(Tm[ok]), 1)
    beta = cf[0]
    fit = np.exp(np.polyval(cf, np.log(fb)))
    resid = 100 * (Tm - fit) / fit

    fig, (a0, a1) = plt.subplots(
        2, 1, figsize=(7.2, 6.2), height_ratios=[2.2, 1], sharex=True)

    a0.plot(fb, Tm, color=BLUE, lw=2.0, label="$T_{\\rm ant}^{\\rm (sw)}$, median of 215 spectra")
    a0.plot(fb, fit, color=ORANGE, lw=2.0, ls="--",
            label=f"power law, $\\beta = {beta:+.3f}$")
    a0.plot(fb, Tm[0] * (fb / fb[0]) ** (-2.5), color=MUTED, lw=1.6, ls="-.",
            label="synchrotron $\\beta = -2.5$ (for reference)")
    a0.set_ylabel("$T_{\\rm ant}^{\\rm (sw)}$  [K]")
    a0.set_title("Calibrated spectrum at the RF SWITCH COMMON PORT — not a sky "
                 "brightness\n"
                 f"band {fmin:.0f}–{fmax:.0f} MHz; measured $\\beta={beta:+.2f}$ "
                 "where synchrotron requires $-2.5$", loc="left")
    a0.legend(loc="upper left")
    recess(a0)

    a1.plot(fb, resid, color=BLUE, lw=1.6)
    a1.axhline(0, color=INK2, lw=1.0, ls=":")
    a1.set_xlabel(f"frequency  [MHz]   (band {fmin:.0f}–{fmax:.0f} MHz)")
    a1.set_ylabel("residual  [%]")
    a1.text(0.01, 0.06,
            f"in-sample: {np.nanstd(resid):.2f} % rms, "
            f"{np.nanmax(resid)-np.nanmin(resid):.1f} % ptp",
            transform=a1.transAxes, color=INK2, fontsize=9)
    recess(a1)

    fig.tight_layout()
    out = FIGS / "fig2_spectrum_residual.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}: beta {beta:+.3f} ({fmin:.0f}-{fmax:.0f} MHz)")
    return out


# ---------------------------------------------------------------- figure 3
def fig_control(z, fmin, fmax, nseg=59):
    """The Claim-2 control. Both series normalised to their own mean, so this
    is one axis, not a dual axis."""
    f = z["freqs"]
    band = (f >= fmin) & (f <= fmax)
    # Difference FIRST, then reduce: the excess is a per-channel quantity.
    # Reducing each state before subtracting compares different channels (the
    # median is not linear). That is about which quantity is computed -- it is
    # NOT a claim that the resulting correlation scalar is well determined.
    # Per channel, r(control, excess) spans [-0.15, +0.52], median +0.36.
    PA, PO = z["p_amb"][:, band], z["p_on"][:, band]
    pa = np.nanmedian(PA, axis=1)
    t = z["sol_times"]
    s = slice(0, nseg)
    pd_ = np.nanmedian(PO - PA, axis=1)

    ca, cd = pa[s] / pa[s].mean(), pd_[s] / pd_[s].mean()
    x = dt(t[s])

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(x, ca, color=BLUE, lw=2.0, marker="o", ms=3.5,
            label="$P_{\\rm amb}$ — fixed passive load (CONTROL)")
    ax.plot(x, cd, color=ORANGE, lw=2.0, marker="o", ms=3.5,
            label="$P_{\\rm on}-P_{\\rm amb}$ — injected excess")
    ax.axhline(1.0, color=INK2, lw=1.0, ls=":")

    ax.annotate(f"{100*ca.std():.2f} % rms", (x[-1], ca[-1]), color=BLUE,
                xytext=(6, -2), textcoords="offset points", fontsize=9,
                va="center")
    ax.annotate(f"{100*cd.std():.2f} % rms", (x[-1], cd[-1]), color=ORANGE,
                xytext=(6, 0), textcoords="offset points", fontsize=9,
                va="center")

    ax.set_ylabel("power, normalised to segment mean  [–]")
    ax.set_xlabel("time  [UTC]")
    ax.set_title("The receiver is five times steadier than the injection\n"
                 f"band {fmin:.0f}–{fmax:.0f} MHz, {nseg} cycles, "
                 "2026-07-17 04:11→16:23 UTC (in-sample)", loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="upper left")
    recess(ax)

    fig.tight_layout()
    out = FIGS / "fig3_control_vs_excess.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}: control {100*ca.std():.2f} % rms vs "
          f"excess {100*cd.std():.2f} % rms")
    return out


# ---------------------------------------------------------------- figure 4
def fig_trx(z, fmin, fmax):
    f = z["freqs"]
    band = (f >= fmin) & (f <= fmax)
    tb = np.nanmedian(z["t_rx"][:, band], axis=1)
    t = z["sol_times"]
    x = dt(t)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    # break the line at observing gaps > 1 h so absent data is not implied
    seg, cur = [], [0]
    for i in range(1, len(t)):
        if t[i] - t[i - 1] > 3600:
            seg.append(cur)
            cur = []
        cur.append(i)
    seg.append(cur)
    for k, sidx in enumerate(seg):
        sidx = np.array(sidx)
        ax.plot([x[i] for i in sidx], tb[sidx], color=BLUE, lw=2.0,
                marker="o", ms=4,
                label="$T_{\\rm rx}$, per cal cycle" if k == 0 else None)

    for ts, lab in ((datetime(2026, 7, 17, 22, 0, tzinfo=timezone.utc),
                     "7U battery failure ~22:00"),
                    (datetime(2026, 7, 17, 23, 35, tzinfo=timezone.utc),
                     "box-air outage ~23:35")):
        ax.axvline(ts, color=ORANGE, lw=1.6, ls="--")
        ax.text(ts, ax.get_ylim()[1], " " + lab, rotation=90, va="top",
                ha="left", color=ORANGE, fontsize=8.5)

    ax.set_ylabel("$T_{\\rm rx}$  [K]")
    ax.set_xlabel("time  [UTC]")
    ax.set_title("Receiver regime change across the 07-17 power events\n"
                 f"band {fmin:.0f}–{fmax:.0f} MHz, {len(t)} cal cycles; "
                 "line broken at observing gaps > 1 h", loc="left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.legend(loc="upper left")
    recess(ax)

    fig.tight_layout()
    out = FIGS / "fig4_trx_timeseries.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}: T_rx {tb.min():.0f}-{tb.max():.0f} K")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="window_0717.npz")
    ap.add_argument("--phasec", default="trx_phaseC.npz")
    ap.add_argument("--fmin", type=float, default=55.0)
    ap.add_argument("--fmax", type=float, default=85.0)
    args = ap.parse_args()

    FIGS.mkdir(exist_ok=True)
    w = np.load(HERE / args.window, allow_pickle=True)
    p = np.load(HERE / args.phasec, allow_pickle=True)
    print(f"figures (band {args.fmin:.0f}-{args.fmax:.0f} MHz):")
    fig_closure(w, args.fmin, args.fmax)
    fig_spectrum(w, args.fmin, args.fmax)
    fig_control(p, args.fmin, args.fmax)
    fig_trx(p, args.fmin, args.fmax)


if __name__ == "__main__":
    main()
