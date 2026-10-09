"""Figures for a ``global_vNNNN`` directory (``global_fit.py``).

- ``params.png``: per configuration against frequency: T_gnd, the receiver
  offsets, the Sun's flux, and held-out rms against training rms.
- ``gains.png``: the fitted gain knots against time at 148 MHz, per configuration.
- ``resid_<label>.png``: one configuration's fractional residual against time
  at four frequencies, every fitted bin, coloured by height era, with
  excluded windows and the transmitter-on spans marked.

Run::

    python plot_global.py <global dir> [--resid LABEL ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

ERA_C = {"~30m": "#2a78d6", "~87.5m": "#1baf7a", "~91m": "#eb6834"}
CFG_C = ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#7a5195", "#555555"]


def load(d):
    out = {}
    for f in sorted(d.glob("*.json")):
        out[f.stem] = json.loads(f.read_text())
    return out


def as_dt(t):
    return (np.asarray(t) * 1e6).astype("datetime64[us]")


def params(d, runs):
    fig, ax = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
    for c, (lab, r) in zip(CFG_C, runs.items()):
        R = r["results"]
        f = np.array([x["freq_mhz"] for x in R])
        tg = np.array([x["all"]["theta"]["T_gnd"] for x in R])
        etg = np.array([x["all"]["theta_err_scaled"]["T_gnd"] for x in R])
        ax[0, 0].errorbar(f, tg, etg, fmt="o-", ms=3, color=c, label=lab, capsize=2)
        for g, mk in (("rx-A", "o"), ("rx-B", "s")):
            v = [x["all"]["offsets"].get(g, np.nan) for x in R]
            ax[0, 1].plot(f, v, mk + "-", ms=3, color=c, label=f"{lab} {g}" if g == "rx-A" else None,
                          alpha=1 if g == "rx-A" else 0.5)
        s = [x["all"]["theta"].get("S_sun", np.nan) for x in R]
        es = [x["all"]["theta_err_scaled"].get("S_sun", np.nan) for x in R]
        ax[1, 0].errorbar(f, s, es, fmt="o-", ms=3, color=c, capsize=2)
        ax[1, 1].plot(f, [100 * x["train"]["test_rms_frac"] for x in R], "o-", ms=3, color=c)
        ax[1, 1].plot(f, [100 * x["train"]["rms_frac"] for x in R], "x--", ms=3, color=c, alpha=0.6)
    ax[0, 0].set(ylabel="T_gnd (GSM K)", ylim=(-500, 3000))
    ax[0, 0].legend(fontsize=6)
    ax[0, 1].set(ylabel="offset (GSM K); o rx-A, s rx-B", ylim=(-1000, 3000))
    ax[1, 0].set(ylabel="Sun flux (SFU, GSM-K scale)", xlabel="frequency (MHz)")
    ax[1, 1].set(ylabel="rms residual (% of power)\no held-out, x training", xlabel="frequency (MHz)",
                 yscale="log")
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "params.png", dpi=120)


def gains(d, runs, freq=148.4):
    """Fitted gain knots against time at one frequency, per configuration."""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for c, (lab, r) in zip(CFG_C, runs.items()):
        q = min(r["results"], key=lambda x: abs(x["freq_mhz"] - freq))
        gk = q["all"].get("gain_knots")
        if not gk:
            continue
        t, g, sg = np.array(gk["t"]), np.array(gk["gain"]), np.array(gk["segment"])
        for k, seg in enumerate(sorted(set(sg))):
            m = sg == seg
            ax.plot(as_dt(t[m]), g[m], "o-", ms=3, color=c, label=lab if k == 0 else None)
    ax.set(yscale="log", ylabel=f"gain at {freq} MHz\n(counts per GSM K)", xlabel="UTC")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(d / "gains.png", dpi=120)


def resid(d, lab, run):
    z = np.load(d / f"{lab}_resid.npz", allow_pickle=False)
    f = z["freqs_mhz"]
    pick = [int(np.argmin(np.abs(f - x))) for x in (70, 125, 160, 200)]
    t = as_dt(z["t"])
    fig, ax = plt.subplots(len(pick), 1, figsize=(10, 7.5), sharex=True)
    for a, i in zip(ax, pick):
        r = 100 * z["resid_frac"][i]
        for e, c in ERA_C.items():
            m = (z["era"] == e) & np.isfinite(r)
            a.plot(t[m], r[m], ".", ms=1.5, color=c, label=e)
        for w in run["windows"]:
            if not w["kept"]:
                a.axvspan(as_dt(w["t0"]), as_dt(w["t1"]), color="0.85", lw=0)
        a.axhline(0, color="k", lw=0.5)
        lim = np.nanpercentile(np.abs(r), 99.5) if np.isfinite(r).any() else 1
        a.set(ylim=(-lim, lim), ylabel=f"{f[i]:.1f} MHz\nresid (%)")
        a.grid(alpha=0.3)
    ax[0].legend(markerscale=6, fontsize=7, ncol=3)
    ax[0].set_title(f"{lab}: (data − model)/data; grey = window excluded by the sky gate")
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax[-1].set_xlabel("UTC")
    fig.tight_layout()
    fig.savefig(d / f"resid_{lab}.png", dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--resid", nargs="*", default=[])
    a = ap.parse_args()
    runs = load(a.dir)
    params(a.dir, runs)
    gains(a.dir, runs)
    for lab in a.resid:
        resid(a.dir, lab, runs[lab])


if __name__ == "__main__":
    main()
