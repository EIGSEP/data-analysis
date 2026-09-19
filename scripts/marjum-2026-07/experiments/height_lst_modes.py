#!/usr/bin/env python
"""height_lst_modes.py — does height modulation add independent information?

PROGRAM.md §7 P3 asks whether rotation + LST + height "actually deliver
independent information, or merely re-weight the same few modes". The height
half is testable without any absolute calibration, by LST matching.

The 30 m era (2026-07-16) and the 87.5 m era (2026-07-17) cover the same UTC
hours one day apart, so at matched UTC the sky is the same to 3.9 min of LST.
Form, per matched hour, the ratio spectrum

    R(nu, LST) = S_87.5m(nu, LST) / S_30m(nu, LST)

and the logic decomposes cleanly:

  * R flat in nu and constant in LST -> a pure scalar rescaling. Height
    delivered no new information at all.
  * R structured in nu but constant in LST -> an instrumental bandpass change.
    Still no new *sky* information.
  * R varying with LST -> either the horizon/beam-weighted sky genuinely
    changed with height, or something drifted in time. Those two are NOT
    separated by this test; see the confounds in the memo.

Counting modes: PCA over the set of per-hour log-ratio spectra, with the noise
level set by the within-hour scatter, so "how many modes are above noise"
is answered against a measured floor rather than an assumed one.

box-gnd (input 0) is the science channel here: the 1.953 MHz EMI episode of
2026-07-16 01:18-16:51 is confined to box-air, which makes box-air's ratio
step by ~9x at the episode boundary and unusable for this test.

Usage:
    python height_lst_modes.py [--input 0] [--out DIR]
"""
import argparse
import datetime as dt
import glob
import json
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from eigsep_data.paths import campaign_data_dir

DATA = str(campaign_data_dir())
BAND = (50.0, 200.0)
HOURS = ["%02d" % h for h in range(2, 16)]   # 02-15 UTC, both days populated
DAY_LO, DAY_HI = "20260716", "20260717"      # ~30 m era, ~87.5 m era

CORRUPT = {"corr_20260715_044343Z.h5", "corr_20260715_213105Z.h5",
           "corr_20260718_032419Z.h5"}


def hour_spectra(day, hour, key):
    """Per-file median spectra for one UTC hour, wrap-repaired."""
    out, freqs = [], None
    for p in sorted(glob.glob(os.path.join(DATA, "corr_*.h5"))):
        b = os.path.basename(p)
        if b in CORRUPT or b[5:13] != day or b[14:16] != hour:
            continue
        with h5py.File(p, "r") as f:
            if key not in f["data"]:
                continue
            a = f["data"][key][:].astype(np.int64)
            freqs = f["header"]["freqs"][:]
        m = a < 0
        if m.any():
            a[m] += 2 ** 32
        inb = (freqs >= BAND[0]) & (freqs <= BAND[1])
        if np.median(a[:, inb]) < 1000:
            continue
        out.append(np.median(a, axis=0))
    return (np.array(out, float) if out else None), freqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="0")
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    key = args.input

    lo_list, hi_list, used, scatter = [], [], [], []
    freqs = None
    for h in HOURS:
        Slo, freqs = hour_spectra(DAY_LO, h, key)
        Shi, freqs = hour_spectra(DAY_HI, h, key)
        if Slo is None or Shi is None or len(Slo) < 8 or len(Shi) < 8:
            continue
        lo_list.append(np.median(Slo, axis=0))
        hi_list.append(np.median(Shi, axis=0))
        # within-hour scatter of the *mean* spectrum, both eras combined:
        # this is the noise on each ratio point.
        s_lo = Slo.std(axis=0, ddof=1) / np.sqrt(len(Slo))
        s_hi = Shi.std(axis=0, ddof=1) / np.sqrt(len(Shi))
        scatter.append((s_lo, s_hi, np.median(Slo, axis=0), np.median(Shi, axis=0)))
        used.append(h)

    lo = np.array(lo_list)
    hi = np.array(hi_list)
    inb = (freqs >= BAND[0]) & (freqs <= BAND[1])
    nu = freqs[inb]
    lo, hi = lo[:, inb], hi[:, inb]

    # Common RFI mask: a channel is used only if it is clean in EVERY hour of
    # BOTH eras, so all ratio spectra live on an identical channel basis.
    from scipy.ndimage import median_filter
    bad = np.zeros(lo.shape[1], bool)
    for arr in (lo, hi):
        for row in arr:
            base = median_filter(row, size=31)
            resid = row - base
            mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
            bad |= np.abs(resid) > 4.0 * 1.4826 * mad
    good = ~bad & (lo > 0).all(axis=0) & (hi > 0).all(axis=0)
    print(f"input {key}: {len(used)} matched hours ({used[0]}-{used[-1]} UTC), "
          f"{good.sum()} of {inb.sum()} channels clean in all")

    R = hi[:, good] / lo[:, good]
    lnR = np.log(R)

    # per-hour scalar level and residual shape
    level = lnR.mean(axis=1)
    shape = lnR - level[:, None]

    print(f"\nscalar level exp(<lnR>) per hour:")
    print("   " + "  ".join(f"{h}:{np.exp(l):.3f}" for h, l in zip(used, level)))
    print(f"   mean {np.exp(level.mean()):.4f}   "
          f"spread {100*level.std():.2f}%  (this is the pure-rescaling part)")

    # noise on lnR per hour from within-hour scatter
    noise = []
    for (s_lo, s_hi, m_lo, m_hi) in scatter:
        n = np.sqrt((s_lo[inb][good] / m_lo[inb][good]) ** 2
                    + (s_hi[inb][good] / m_hi[inb][good]) ** 2)
        noise.append(np.median(n))
    noise = np.array(noise)
    print(f"\nmeasured noise on lnR: median {np.median(noise):.5f} per channel")
    print(f"residual shape RMS   : {shape.std():.5f}")
    print(f"   shape/noise = {shape.std()/np.median(noise):.1f}")

    # PCA on the residual shapes: how many modes does height+LST actually span?
    u, s, vt = np.linalg.svd(shape - shape.mean(axis=0), full_matrices=False)
    var = s ** 2 / max((s ** 2).sum(), 1e-30)
    print(f"\nPCA of the LST-varying part of the ratio spectrum:")
    print("   singular values: " + "  ".join(f"{x:.4f}" for x in s[:6]))
    print("   variance frac  : " + "  ".join(f"{x:.3f}" for x in var[:6]))
    # noise-equivalent singular value for a (nhour x nchan) white matrix
    nh, nc = shape.shape
    s_noise = np.median(noise) * np.sqrt(nc)
    n_sig = int((s > s_noise).sum())
    print(f"   noise-equivalent singular value ~ {s_noise:.4f}")
    print(f"   => {n_sig} mode(s) above the measured noise floor")

    outp = os.path.join(args.out, f"height_lst_modes_input{key}.json")
    with open(outp, "w") as fh:
        json.dump({"provenance": {
            "product": "height_lst_modes", "campaign": "marjum-2026-07",
            "version": "v1",
            "generated_utc": dt.datetime.now(dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "generator": "experiments/height_lst_modes.py",
            "input": key, "band_mhz": list(BAND),
            "days": {"low": DAY_LO, "high": DAY_HI}, "hours": used,
            "notes": ("LST matched by UTC hour one day apart (3.9 min offset); "
                      "wraps repaired; common RFI mask across all hours/eras; "
                      "noise floor from within-hour scatter"),
        },
            "scalar_level": np.exp(level).tolist(),
            "level_spread_pct": float(100 * level.std()),
            "shape_rms": float(shape.std()),
            "noise_median": float(np.median(noise)),
            "singular_values": s.tolist(),
            "variance_fraction": var.tolist(),
            "n_modes_above_noise": n_sig,
            "n_channels": int(good.sum()),
        }, fh)
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
