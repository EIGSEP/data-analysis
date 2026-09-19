"""Diagnostic figures for the flags@v0 memo (MEMO-009).

Three figures, each answering a question a reader would otherwise have to
take on trust:

  fig1_completeness   What survives the cut?      (leakage / false negatives)
  fig2_tradeoff       What does the threshold cost? (sky removed vs leakage)
  fig3_residual       Does masking actually remove the RFI, and only the RFI?

Figure 3 is the before/after residual the memo series requires: without it
the statistics are unanchored.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from eigsep_data.flagging import detectors as D  # noqa: E402
from eigsep_data.flagging import validate as V  # noqa: E402

from eigsep_data.paths import get_campaign_root


def _campaign_root():
    """Campaign root; ``MARJUM_DATA_ROOT`` wins, else the package setting.

    This script anchored on its own ``__file__`` until it moved out of
    the campaign tree on 2026-09-19.
    """
    env = os.environ.get("MARJUM_DATA_ROOT")
    if env:
        return env
    return str(get_campaign_root(required=True))


ROOT = _campaign_root()
OUT = os.path.join(ROOT, "flags", "v0", "figures")

# Validated categorical slots 1-3 (see dataviz references/palette.md).
# All-pairs validated in light mode; aqua carries a contrast WARN, so every
# series is direct-labelled rather than identified by colour alone.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURFACE = "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(True, color=MUTED, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)


def _interp_x(x, y, target):
    """x at which y first crosses target, by linear interpolation."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(len(y) - 1):
        if y[i] <= target <= y[i + 1] and y[i + 1] != y[i]:
            return x[i] + (target - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    return np.nan


# --------------------------------------------------------------- figure 1

def fig_completeness(val):
    lk = val["leakage"]
    amps = sorted(float(k.replace("sigma", "")) for k in lk)
    prob = [lk[f"{int(a)}sigma"]["detection_prob"] for a in amps]
    n = lk[f"{int(amps[0])}sigma"]["n_trials"]

    fig, ax = plt.subplots(figsize=(7.0, 4.3), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot(amps, prob, color=BLUE, linewidth=2, marker="o", markersize=8,
            markerfacecolor=BLUE, markeredgecolor=SURFACE,
            markeredgewidth=2, zorder=3)

    p50, p90 = _interp_x(amps, prob, 0.5), _interp_x(amps, prob, 0.9)
    for xv, lab, yy in ((p50, f"50% at {p50:.1f}σ", 0.5),
                        (p90, f"90% at {p90:.1f}σ", 0.9)):
        if np.isfinite(xv):
            ax.plot([xv, xv], [0, yy], color=MUTED, linewidth=1,
                    linestyle=(0, (4, 3)), zorder=1)
            ax.annotate(lab, (xv, yy), xytext=(6, -12),
                        textcoords="offset points", color=INK2, fontsize=9)

    ax.axvspan(0, 4, color=ORANGE, alpha=0.09, zorder=0, linewidth=0)
    ax.annotate("effectively blind\nbelow 4σ", (2.0, 0.72), color=ORANGE,
                fontsize=9, ha="center", va="center", weight="bold")

    ax.set_xlim(0, max(amps) + 0.4)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("injected spike amplitude  (per-channel temporal MAD, σ)",
                  color=INK2, fontsize=10)
    ax.set_ylabel("detection probability", color=INK2, fontsize=10)
    ax.set_title("What survives the cut: flagger completeness vs RFI amplitude",
                 color=INK, fontsize=12, weight="bold", loc="left", pad=12)
    ax.annotate(f"{n} injections per amplitude, into RFI-quiet data "
                f"(07-17 06:00–10:00 UTC)",
                (0, 1.0), xycoords="axes fraction", xytext=(0, 8),
                textcoords="offset points", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    p = os.path.join(OUT, "fig1_completeness.png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


# --------------------------------------------------------------- figure 2

def tradeoff_data(window=("2026-07-17T06:00", "2026-07-17T10:00"),
                  n_files=8, thresholds=(3, 4, 5, 6, 7, 8), amp=6.0,
                  n_inject=30, seed=1):
    """False-positive (sky removed) and detection rate vs clip threshold."""
    rng = np.random.default_rng(seed)
    files = V.files_between(*window)[:n_files]
    fp = {t: [] for t in thresholds}
    det = {t: [0, 0] for t in thresholds}
    for p in files:
        logp, freqs, ant = V.load_logp(p)
        if logp is None:
            continue
        sel = np.where((freqs >= D.BAND_ANALYSIS[0])
                       & (freqs <= D.BAND_ANALYSIS[1]))[0]
        _, _, scale = D.transient_track(logp, ant)
        good_t = np.where(ant)[0]
        for t in thresholds:
            pix, _, _ = D.transient_track(logp, ant, clip_sigma=t)
            fp[t].append(float(pix[:, sel].mean()))
            for _ in range(n_inject):
                ch = int(rng.choice(sel))
                ti = int(rng.choice(good_t))
                s = scale[ch]
                if not np.isfinite(s) or s <= 0:
                    continue
                test = logp.copy()
                test[ti, ch] += amp * s
                pk, _, _ = D.transient_track(test, ant, clip_sigma=t)
                det[t][1] += 1
                if pk[ti, ch]:
                    det[t][0] += 1
    return (list(thresholds),
            [float(np.median(fp[t])) * 100 for t in thresholds],
            [det[t][0] / det[t][1] * 100 if det[t][1] else np.nan
             for t in thresholds], amp)


def fig_tradeoff(data):
    th, sky, dete, amp = data
    # Two measures of different scale -> two stacked panels sharing x,
    # never a dual y-axis.
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 5.6), dpi=200,
                                 sharex=True,
                                 gridspec_kw={"hspace": 0.18})
    fig.patch.set_facecolor(SURFACE)
    for ax in (a1, a2):
        _style(ax)

    a1.plot(th, sky, color=ORANGE, linewidth=2, marker="o", markersize=8,
            markerfacecolor=ORANGE, markeredgecolor=SURFACE, markeredgewidth=2)
    a1.set_ylabel("sky removed  (%)", color=INK2, fontsize=10)
    a1.set_title("What the threshold costs, in both directions",
                 color=INK, fontsize=12, weight="bold", loc="left", pad=12)
    a1.annotate("sky removed — false positives in RFI-quiet data",
                (th[-1], sky[-1]), xytext=(-6, 14), textcoords="offset points",
                color=ORANGE, fontsize=9.5, ha="right", weight="bold")

    a2.plot(th, dete, color=BLUE, linewidth=2, marker="o", markersize=8,
            markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=2)
    a2.set_ylabel(f"detected at {amp:g}σ  (%)", color=INK2, fontsize=10)
    a2.set_xlabel("detector clip threshold  (σ)", color=INK2, fontsize=10)
    a2.annotate(f"RFI recovered — a {amp:g}σ spike",
                (th[-1], dete[-1]), xytext=(-6, 14),
                textcoords="offset points", color=BLUE, fontsize=9.5,
                ha="right", weight="bold")

    for ax in (a1, a2):
        ax.axvline(5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    a1.annotate("v0 operating point", (5, max(sky)), xytext=(8, -4),
                textcoords="offset points", color=INK2, fontsize=9)
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_threshold_tradeoff.png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    return p


# --------------------------------------------------------------- figure 3

def fig_residual(fname="corr_20260718_003149Z.h5", key="4"):
    """Before/after spectra for a file inside the 07-17/18 comb episode."""
    path = os.path.join(ROOT, "data", fname)
    with h5py.File(path, "r") as h:
        freqs = h["header/freqs"][:]
        raw = h["data/" + key][:]
        rfsw = h["metadata/rfswitch"][()] if (
            "metadata" in h and "rfswitch" in h["metadata"]) else None
    ant = D.antenna_mask(rfsw, raw.shape[0])
    ovf = D.overflow_mask(raw)
    val = np.where(ovf, raw.astype(float) + 2.0 ** 32, raw.astype(float))
    logp = np.log10(np.maximum(val, 1.0))

    pix, _, _ = D.transient_track(logp, ant)
    chan, _ = D.persistent_track(logp, ant, freqs)
    med = np.median(logp[ant], axis=0)
    combs = D.identify_combs(med, freqs)
    cat = D.categorise(pix, chan, freqs, combs, False,
                       D.broadband_times(pix, freqs),
                       D.meteor_scatter_times(pix, freqs))
    flagged = (cat & D.RFI_BITS) != 0

    before = np.median(logp[ant], axis=0)
    m = np.where(flagged, np.nan, logp)
    with np.errstate(invalid="ignore"):
        after = np.nanmedian(m[ant], axis=0)

    band = (freqs >= 50) & (freqs <= 215)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.6, 5.8), dpi=200,
                                 sharex=True, gridspec_kw={
                                     "height_ratios": [2.1, 1],
                                     "hspace": 0.16})
    fig.patch.set_facecolor(SURFACE)
    for ax in (a1, a2):
        _style(ax)

    a1.plot(freqs[band], before[band], color=ORANGE, linewidth=1.4, zorder=2)
    a1.plot(freqs[band], after[band], color=BLUE, linewidth=1.4, zorder=3)
    a1.annotate("before masking", (freqs[band][-1], before[band][-1]),
                xytext=(-4, 16), textcoords="offset points", color=ORANGE,
                fontsize=9.5, ha="right", weight="bold")
    a1.annotate("after masking", (freqs[band][-1], after[band][-1]),
                xytext=(-4, -18), textcoords="offset points", color=BLUE,
                fontsize=9.5, ha="right", weight="bold")
    a1.set_ylabel("log$_{10}$ power", color=INK2, fontsize=10)
    a1.set_title("Before / after: the 1.953125 MHz comb removed, "
                 "continuum left intact",
                 color=INK, fontsize=12, weight="bold", loc="left", pad=12)
    a1.annotate(f"{fname}, input {key} — inside the 07-17/18 comb episode",
                (0, 1.0), xycoords="axes fraction", xytext=(0, 8),
                textcoords="offset points", color=MUTED, fontsize=8.5)

    diff = before - after
    a2.plot(freqs[band], diff[band], color=AQUA, linewidth=1.4)
    a2.axhline(0, color=MUTED, linewidth=0.9)
    a2.annotate("difference (what the mask removed)",
                (freqs[band][0], np.nanmax(diff[band])),
                xytext=(4, -2), textcoords="offset points", color=AQUA,
                fontsize=9.5, weight="bold")
    a2.set_xlabel("frequency (MHz)", color=INK2, fontsize=10)
    a2.set_ylabel("Δ log$_{10}$ power", color=INK2, fontsize=10)

    fig.tight_layout()
    p = os.path.join(OUT, "fig3_before_after_residual.png")
    fig.savefig(p, facecolor=SURFACE)
    plt.close(fig)
    frac = float(flagged[:, band].mean())
    return p, frac


def main():
    os.makedirs(OUT, exist_ok=True)
    val = json.load(open(os.path.join(ROOT, "flags", "v0", "validation.json")))
    print(fig_completeness(val))
    data = tradeoff_data()
    print("threshold sweep:", list(zip(data[0], np.round(data[1], 3),
                                       np.round(data[2], 1))))
    print(fig_tradeoff(data))
    p, frac = fig_residual()
    print(p, f"flagged fraction in band = {frac:.4f}")
    with open(os.path.join(OUT, "threshold_sweep.json"), "w") as f:
        json.dump({"threshold_sigma": data[0],
                   "sky_removed_pct": data[1],
                   "detect_pct_at_6sigma": data[2]}, f, indent=2)


if __name__ == "__main__":
    main()
