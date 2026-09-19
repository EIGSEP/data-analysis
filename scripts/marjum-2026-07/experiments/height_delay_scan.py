#!/usr/bin/env python
"""height_delay_scan.py — what does height modulation actually buy?

PROGRAM.md §5 lists platform height as the closest ground analogue to BLOOM's
moving occulter and notes nobody has quantified what it delivers. The
calibration-independent half of that question lives in delay space:

    a reflection off the ground below the platform arrives at
    tau = 2 h cos(theta) / c  <=  2h/c,

so raising the antenna pushes reflection structure to longer delays (finer
spectral ripple) by a factor known in advance from geometry alone. Predicted
maxima: 2 m -> 13 ns, 30 m -> 200 ns, 87.5 m -> 584 ns, 91 m -> 607 ns.

METHOD NOTES, each one a trap already stepped in:

  * DETRENDING MUST BE SCALE-FREE. A median-filter high-pass has a hard
    frequency scale: a 101-channel filter cuts at 24.7 MHz and therefore
    *deletes the 2 m era's predicted 75 MHz ripple while passing the 87.5 m
    era's 1.7 MHz*, manufacturing a height trend from the estimator. A
    low-order polynomial in log-log removes only the smoothest bandpass shape
    and does not impose a delay cutoff that varies with what you are testing.
  * DIGITAL COMBS MIMIC HEIGHT. The 1.953125 MHz self-comb (exactly 8
    channels) transforms to 512 ns = an apparent 76.7 m; the 0.9766 MHz
    (4-channel) family lands at 1024 ns = 153 m. The TX comb near 1.0 MHz sits
    close to the latter. All are marked in the output.
  * RFI DOMINATES A DELAY TRANSFORM. One bright channel rings across the whole
    delay axis, so flagging is iterative sigma-clipping, and the flagged
    fraction is reported because a heavily flagged band cannot support a claim.
  * Accumulator wrap is repaired (+2^32), not masked (MEMO-002).
  * The box-air break at 2026-07-17T23:44:35Z (MEMO-004) splits the 91 m era;
    only the pre-break segment is used.

Usage:
    python height_delay_scan.py [--input 4] [--out DIR]
"""
import argparse
import datetime as dt
import glob
import json
import os

import h5py
import numpy as np
from scipy.ndimage import median_filter

HERE = os.path.dirname(os.path.abspath(__file__))
from eigsep_data.paths import campaign_data_dir

DATA = str(campaign_data_dir())
C = 299792458.0
BAND = (50.0, 200.0)

# Windows chosen from measured per-file liveness (bandpow>0 on both inputs),
# rfswitch RFANT, rot_state parked -- not from the field-note narrative.
ERAS = [
    ("~2m",     2.0,  "A", "20260713060000", "20260713120000"),
    ("~30m",   30.0,  "C", "20260715140000", "20260715180000"),
    ("~87.5m", 87.5,  "C", "20260717030000", "20260717150000"),
    ("~91m",   91.0,  "C", "20260717190000", "20260717200000"),
]

CORRUPT = {
    "corr_20260715_044343Z.h5",
    "corr_20260715_213105Z.h5",
    "corr_20260718_032419Z.h5",
}

DIGITAL = {
    "8-chan self-comb 1.953 MHz": 1e3 / 1.953125,
    "4-chan family 0.977 MHz": 1e3 / 0.9765625,
}


def era_spectrum(t0, t1, key):
    out, freqs = [], None
    for p in sorted(glob.glob(os.path.join(DATA, "corr_*.h5"))):
        if os.path.basename(p) in CORRUPT:
            continue
        b = os.path.basename(p)
        t = b[5:13] + b[14:20]
        if not (t0 <= t <= t1):
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
        if np.median(a[:, inb]) < 1000:      # dead / outage file
            continue
        out.append(np.median(a, axis=0))
    if not out:
        return None, None, 0
    return np.median(np.array(out, float), axis=0), freqs, len(out)


def flag_rfi(s, n_iter=5, thresh=4.0):
    """Iterative sigma-clip against a wide median baseline. Returns mask."""
    bad = np.zeros(len(s), bool)
    for _ in range(n_iter):
        work = s.copy()
        if bad.any():
            work[bad] = np.interp(np.flatnonzero(bad),
                                  np.flatnonzero(~bad), s[~bad])
        base = median_filter(work, size=31)
        resid = work - base
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
        new = np.abs(resid) > thresh * 1.4826 * mad
        if (new | bad).sum() == bad.sum():
            break
        bad |= new
    return bad


def delay_transform(spec, freqs):
    m = (freqs >= BAND[0]) & (freqs <= BAND[1])
    s = spec[m].astype(float)
    nu = freqs[m]
    dnu = nu[1] - nu[0]

    bad = flag_rfi(s)
    frac_flag = bad.mean()
    if bad.any():
        s = s.copy()
        s[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), s[~bad])

    # Scale-free detrend: low-order polynomial in log-log removes the smooth
    # bandpass without imposing a delay cutoff.
    good = s > 0
    coef = np.polyfit(np.log(nu[good]), np.log(s[good]), 7)
    smooth = np.exp(np.polyval(coef, np.log(nu)))
    r = s / smooth - 1.0

    w = np.blackman(len(r))
    P = np.abs(np.fft.rfft(r * w))
    tau = np.fft.rfftfreq(len(r), d=dnu) * 1e3
    return tau, P, frac_flag, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="4")
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()

    print(f"input {args.input}; band {BAND[0]}-{BAND[1]} MHz; "
          f"delay resolution {1e3/(BAND[1]-BAND[0]):.1f} ns\n")
    for k, v in DIGITAL.items():
        print(f"  digital artefact: {k} -> {v:.0f} ns "
              f"(mimics h = {C*v*1e-9/2:.1f} m)")
    print()

    results = []
    for label, h, phase, t0, t1 in ERAS:
        spec, freqs, n = era_spectrum(t0, t1, args.input)
        if spec is None:
            print(f"{label:<8}: no usable files")
            continue
        tau, P, ff, r = delay_transform(spec, freqs)
        pred = 2 * h / C * 1e9

        # Noise floor from the top of the delay axis, where no physical
        # reflection from a 100 m platform can live.
        floor = np.median(P[tau > 1500])
        sig = P / max(floor, 1e-30)

        above = tau > 30.0
        pk_i = np.argmax(P[above])
        pk_tau = tau[above][pk_i]
        pk_sig = sig[above][pk_i]

        # power in the band the geometry allows, vs beyond it
        inside = (tau > 30) & (tau <= pred)
        beyond = tau > pred
        print(f"{label:<8} h={h:5.1f} m  phase {phase}  n={n:4d} files  "
              f"RFI-flagged {100*ff:.1f}%  ripple p68={np.percentile(abs(r),68):.4f}")
        print(f"         predicted 2h/c = {pred:6.1f} ns")
        print(f"         peak >30ns  = {pk_tau:6.1f} ns  "
              f"({pk_sig:6.1f}x noise floor)  => apparent h {C*pk_tau*1e-9/2:6.1f} m")
        if inside.any():
            print(f"         median significance  30..{pred:.0f} ns : "
                  f"{np.median(sig[inside]):6.2f}x    beyond: "
                  f"{np.median(sig[beyond]):6.2f}x")
        for k, v in DIGITAL.items():
            j = np.argmin(np.abs(tau - v))
            print(f"         at {v:6.0f} ns ({k}): {sig[j]:6.1f}x")
        print()
        results.append({
            "era": label, "height_m": h, "phase": phase, "n_files": n,
            "frac_rfi_flagged": float(ff), "pred_delay_ns": float(pred),
            "peak_delay_ns": float(pk_tau), "peak_significance": float(pk_sig),
            "noise_floor": float(floor),
            "tau_ns": tau.tolist(), "P": P.tolist(),
        })

    outp = os.path.join(args.out, f"height_delay_input{args.input}.json")
    with open(outp, "w") as fh:
        json.dump({"provenance": {
            "product": "height_delay_scan", "campaign": "marjum-2026-07",
            "version": "v2",
            "generated_utc": dt.datetime.now(dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "generator": "experiments/height_delay_scan.py",
            "input": args.input, "band_mhz": list(BAND),
            "notes": ("wraps repaired +2^32; iterative sigma-clip RFI flagging; "
                      "scale-free 7th-order log-log polynomial detrend (NOT a "
                      "median high-pass, which would impose a height-dependent "
                      "delay cutoff); Blackman window"),
        }, "eras": results}, fh)
    print(f"wrote {outp}")


if __name__ == "__main__":
    main()
