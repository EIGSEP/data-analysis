"""Frequency-flat fast log gain from a global_fit's gain knots, for inject_mock.py --fast-gain-cm.

    fast_gain_cm.py FIT_JSON OUT_NPZ [--fmin 117 --fmax 204] [--detrend-s 10800]

The mean log gain over the converged fits with fmin < f < fmax, knot by knot
(every frequency of one fit shares its knots), minus a piecewise-linear fit
with knots every --detrend-s, per gain segment. Writes knot_t, knot_seg, fast
and the frequencies used.
"""
import argparse
import json

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("fit"); ap.add_argument("out")
ap.add_argument("--fmin", type=float, default=117.0)
ap.add_argument("--fmax", type=float, default=204.0)
ap.add_argument("--detrend-s", type=float, default=10800.0)
a = ap.parse_args()
R = [r for r in json.load(open(a.fit))["results"]
     if a.fmin < r["freq_mhz"] < a.fmax and not r["all"]["diverged"]]
k0 = R[0]["all"]["gain_knots"]
for r in R:
    assert r["all"]["gain_knots"]["t"] == k0["t"], "frequencies have different knots"
kt, ks = np.array(k0["t"]), np.array(k0["segment"])
lg = np.mean([np.log(r["all"]["gain_knots"]["gain"]) for r in R], 0)
fast = np.full(lg.shape, np.nan)
for sg in np.unique(ks):
    m = np.flatnonzero(ks == sg)
    o = m[np.argsort(kt[m])]
    t_, l_ = kt[o], lg[o]
    kk = np.arange(t_.min(), t_.max() + a.detrend_s, a.detrend_s)
    A = np.stack([np.interp(t_, kk, np.eye(kk.size)[i]) for i in range(kk.size)], 1)
    c, *_ = np.linalg.lstsq(A, l_, rcond=None)
    fast[o] = l_ - A @ c
np.savez(a.out, knot_t=kt, knot_seg=ks, fast=fast, freqs_mhz=[r["freq_mhz"] for r in R])
print(a.out, f"{len(R)} frequencies, rms fast log gain {np.std(fast):.4f}")
