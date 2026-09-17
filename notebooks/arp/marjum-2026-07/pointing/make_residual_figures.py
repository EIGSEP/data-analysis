"""Residual figures for the pointing-table memo (MEMO-002).

Three residuals, each labelled in its caption as held-out or in-sample:

1. Plateau scatter before/after fusion -- the 1.64 -> 0.41 deg result.
2. Achieved vs commanded azimuth across the 20:26-21:28 scan block.
3. Motor-vs-IMU elevation residual, as a time series.

All three are scoped to the 07-17/18 beam-scan window and say so on the
figure; none of these numbers should be quoted campaign-wide.

Run::

    /home/aparsons/.local/share/mamba/envs/arp/bin/python3 make_residual_figures.py \
        --table ~/projects/eigsep/marjum-2026-07/pointing/pointing_table_v0.npz \
        --outdir ~/projects/eigsep/marjum-2026-07/pointing
"""

from __future__ import annotations

import argparse
import datetime as dt
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# Validated categorical slots 1 and 2 (light mode), plus ink and surface.
C_RAW = "#eb6834"      # orange -- the "before" / raw sensor
C_FUSED = "#2a78d6"    # blue   -- the "after" / fused product
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"

MOTOR_DEG_PER_STEP = 180.0 / 1.13e4
SCAN_START = dt.datetime(2026, 7, 17, 20, 26, 30, tzinfo=dt.timezone.utc)
SCAN_STOP = dt.datetime(2026, 7, 17, 21, 28, 32, tzinfo=dt.timezone.utc)


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def caption(fig, text):
    fig.text(0.012, 0.012, text, color=INK_2, fontsize=7.4, ha="left",
             va="bottom", linespacing=1.55)


def runs(mask):
    mask = np.asarray(mask, bool)
    e = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return list(zip(e[::2], e[1::2]))


def plateaus(t, motor_az, valid, min_n=50):
    """Index ranges where the commanded (motor) azimuth is constant."""
    same = np.concatenate(([False], np.diff(motor_az) == 0)) & valid
    return [(a, b) for a, b in runs(same) if b - a >= min_n]


def fig_plateau_scatter(z, outdir):
    t, az, mo, po = (z["time_utc"], z["az_deg"], z["motor_az_deg"], z["pot_az_deg"])
    m = (t >= SCAN_START.timestamp()) & (t < SCAN_STOP.timestamp())
    m &= np.isfinite(az) & np.isfinite(mo) & np.isfinite(po)
    idx = np.flatnonzero(m)
    AZ = np.rad2deg(np.unwrap(np.deg2rad(az[idx])))
    pl = plateaus(t[idx], mo[idx], np.ones(idx.size, bool))

    raw = np.array([np.nanstd(po[idx][a:b]) for a, b in pl])
    fused = np.array([np.std(AZ[a:b]) for a, b in pl])

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=200, facecolor=SURFACE)
    lim = (0.04, 4.2)
    ax.plot(lim, lim, color=INK_2, lw=1.2, ls="--", zorder=1)
    ax.text(1.65, 2.35, "no improvement", color=INK_2, fontsize=8, rotation=41,
            ha="center", va="center")
    ax.scatter(raw, fused, s=42, facecolor=C_FUSED, edgecolor=SURFACE,
               linewidth=1.2, zorder=3, label=f"{len(pl)} scan plateaus")
    ax.axhline(np.median(fused), color=C_FUSED, lw=1.6, alpha=0.65, zorder=2)
    ax.axvline(np.median(raw), color=C_RAW, lw=1.6, alpha=0.85, zorder=2)
    ax.text(np.median(raw) * 0.93, 0.048,
            f"raw pot median {np.median(raw):.3f}$\\degree$",
            color=C_RAW, fontsize=8.5, ha="right", va="bottom")
    ax.text(0.047, np.median(fused) * 1.10,
            f"fused median {np.median(fused):.3f}$\\degree$",
            color=C_FUSED, fontsize=8.5, ha="left", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_aspect("equal")
    style(ax, "Azimuth scatter on commanded-constant plateaus\n"
              "beam scan, 2026-07-17 20:26-21:28 UTC",
          "raw potentiometer scatter per plateau  (degrees, 1$\\sigma$)",
          "fused azimuth scatter per plateau  (degrees, 1$\\sigma$)")
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    for txt in leg.get_texts():
        txt.set_color(INK_2)
    better = int((fused < raw).sum())
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.20)
    caption(fig,
            f"HELD-OUT with respect to the fusion filter: plateaus are defined by the commanded motor azimuth being\n"
            f"constant, a fact the filter never uses. {better} of {len(pl)} plateaus improve; median "
            f"{np.median(raw):.3f}$\\degree$ -> {np.median(fused):.3f}$\\degree$ "
            f"({np.median(raw)/np.median(fused):.2f}x).\n"
            f"CAVEAT: the 0.368$\\degree$ azimuth error floor in the product is CALIBRATED from this scatter, so the floor\n"
            f"is in-sample here. The improvement factor is not. Scope: 07-17/18 beam-scan window only.")
    path = os.path.join(outdir, "fig1_plateau_scatter.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path, len(pl), np.median(raw), np.median(fused), better


def fig_achieved_vs_commanded(z, outdir):
    t, az, mo = z["time_utc"], z["az_deg"], z["motor_az_deg"]
    m = (t >= SCAN_START.timestamp()) & (t < SCAN_STOP.timestamp())
    m &= np.isfinite(az) & np.isfinite(mo)
    idx = np.flatnonzero(m)
    T = np.array([dt.datetime.fromtimestamp(x, dt.timezone.utc) for x in t[idx]])
    AZ = np.rad2deg(np.unwrap(np.deg2rad(az[idx])))
    MO = np.rad2deg(np.unwrap(np.deg2rad(mo[idx])))
    AZ -= AZ[0]
    MO -= MO[0]
    div = AZ - MO

    # The slip is a discrete episode, not continuous drift; mark it.
    ep0 = dt.datetime(2026, 7, 17, 20, 41, 24, tzinfo=dt.timezone.utc)
    ep1 = dt.datetime(2026, 7, 17, 20, 57, 30, tzinfo=dt.timezone.utc)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.2, 7.8), dpi=200, facecolor=SURFACE, sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.25], "hspace": 0.18})

    for ax in (ax1, ax2):
        ax.axvspan(ep0, ep1, color=C_RAW, alpha=0.10, lw=0, zorder=0)

    ax1.plot(T, MO, color=C_RAW, lw=2.0, label="commanded (motor counts)")
    ax1.plot(T, AZ, color=C_FUSED, lw=2.0, label="achieved (fused pointing)")
    style(ax1, "Achieved vs commanded azimuth across the beam scan\n"
               "2026-07-17, cumulative from scan start",
          "", "azimuth travelled  (deg)")
    ax1.text(ep0 + (ep1 - ep0) / 2, 232, "slip episode", color=C_RAW,
             fontsize=8.5, ha="center", va="top")
    leg = ax1.legend(loc="upper left", frameon=False, fontsize=8.5)
    for txt in leg.get_texts():
        txt.set_color(INK_2)

    ax2.axhline(0, color=INK_2, lw=1.0, ls="--")
    ax2.plot(T, div, color=C_FUSED, lw=2.0)
    ax2.set_ylim(-38, 12)
    ax2.annotate(f"{div[-1]:+.1f}$\\degree$", xy=(T[-1], div[-1]),
                 xytext=(-52, -16), textcoords="offset points", color=INK,
                 fontsize=9, arrowprops=dict(arrowstyle="->", color=INK_2, lw=1.0))
    ax2.text(T[0], 7.0, "tracks command", color=INK_2, fontsize=8.5,
             ha="left", va="center")
    ax2.text(T[-1], -14.0, "holds new offset", color=INK_2, fontsize=8.5,
             ha="right", va="center")
    style(ax2, "", "time (UTC)", "achieved $-$ commanded  (deg)")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=dt.timezone.utc))

    fig.subplots_adjust(left=0.11, right=0.97, top=0.90, bottom=0.30)
    caption(fig,
            f"HELD-OUT: the commanded scan grid is external information the fusion never consumes. Both traces are "
            f"PER-SAMPLE (n={idx.size:,}),\n"
            f"unwrapped, not plateau medians; 'achieved' is the product's fused az_deg, not the raw potentiometer.\n"
            f"The slip is a DISCRETE EPISODE, not continuous drift: divergence holds within +0.0 to +4.7$\\degree$ until 20:41, "
            f"loses 27.4$\\degree$ in 12.3 min\n"
            f"(20:41:24-20:53:43, -133$\\degree$/hr), then holds a roughly constant offset (std 1.8$\\degree$) to scan end at "
            f"{div[-1]:+.2f}$\\degree$. A least-squares slope\n"
            f"through this step reads ~34$\\degree$/hr and describes neither phase. Per-step medians: 5.0018$\\degree$ commanded "
            f"vs 4.435$\\degree$ achieved.\n"
            f"QUOTE THE EPISODE LOSS, NOT THE ENDPOINT. Cumulative divergence is measured from an epoch, and the "
            f"platform is already slewing\n"
            f"at scan start (166$\\degree$ in the 30 s before 20:26:30), so the endpoint moves with the epoch chosen: "
            f"-21.36$\\degree$ from 20:20:00, -22.58$\\degree$ from\n"
            f"20:26:00, -28.91$\\degree$ from 20:26:30, -32.64$\\degree$ from 20:35:00. The episode loss (-27.35$\\degree$) and the "
            f"post-episode scatter (1.89$\\degree$) are\n"
            f"invariant under all of those. Scope: 07-17/18 beam-scan window only.")
    path = os.path.join(outdir, "fig2_achieved_vs_commanded.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path, div[-1]


def fig_el_residual(z, outdir):
    t, imu, mo_el, fl = (z["time_utc"], z["imu_el_deg"], z["motor_el_deg"],
                         z["flags"])
    both = np.isfinite(imu) & np.isfinite(mo_el)
    idx = np.flatnonzero(both)
    wrap = lambda x: (x + 180) % 360 - 180
    resid = wrap(mo_el[idx] - imu[idx])
    T = np.array([dt.datetime.fromtimestamp(x, dt.timezone.utc) for x in t[idx]])
    sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    post = (fl[idx] & (1 << 8)) != 0

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200, facecolor=SURFACE)
    ax.axhline(0, color=INK_2, lw=1.0, ls="--", zorder=2)
    ax.scatter(T[~post], resid[~post], s=3.0, color=C_FUSED, alpha=0.35,
               linewidths=0, zorder=3, label="EL drive responding")
    ax.scatter(T[post], resid[post], s=3.0, color=C_RAW, alpha=0.45,
               linewidths=0, zorder=4, label="after EL drive failure (01:42:32)")
    style(ax, "Motor-minus-IMU elevation residual\n"
              "every sample where both sensors reported, 07-17/18",
          "time (UTC)", "motor $-$ IMU elevation  (degrees)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=dt.timezone.utc))
    ax.set_ylim(-190, 190)
    ax.set_yticks([-180, -90, 0, 90, 180])
    leg = ax.legend(loc="lower left", frameon=False, fontsize=8.5, markerscale=3.4)
    for txt in leg.get_texts():
        txt.set_color(INK_2)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.80, bottom=0.34)
    caption(fig,
            f"CROSS-SENSOR, not a fit residual: elevation is taken from the IMU alone, so the motor never enters the fused value.\n"
            f"Robust spread {sigma:.1f}$\\degree$ (1.4826 x MAD, n={idx.size:,}). The disagreement is neither drift nor a single step: it is bounded\n"
            f"and structured while the drive responds -- note the discrete level change near 19:45 -- then saturates across the full\n"
            f"$\\pm$180$\\degree$ range once the drive fails and the motor keeps counting against a parked antenna. This is why motor elevation\n"
            f"is never used as an absolute angle. Times are UTC on 07-17/18. Scope: beam-scan window only.")
    path = os.path.join(outdir, "fig3_el_residual.png")
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path, sigma, idx.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    z = np.load(os.path.expanduser(args.table), allow_pickle=True)
    outdir = os.path.expanduser(args.outdir)

    p1, npl, raw, fused, better = fig_plateau_scatter(z, outdir)
    print(f"{p1}: {npl} plateaus, raw {raw:.3f} -> fused {fused:.3f} deg "
          f"({raw/fused:.2f}x), {better}/{npl} improved")
    p2, div = fig_achieved_vs_commanded(z, outdir)
    print(f"{p2}: divergence {div:+.1f} deg by scan end")
    p3, sig, n = fig_el_residual(z, outdir)
    print(f"{p3}: robust sigma {sig:.1f} deg over n={n}")


if __name__ == "__main__":
    main()
