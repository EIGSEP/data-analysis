"""Binned data divided by a calibration-derived gain, for global_fit --mock-y.

    cal_gain_mock.py BINNED_NPZ ns|load OUT.npz MHZ [MHZ ...] [--start ... --stop ...]

y / (g / median g) at each frequency, rx-A, inside [start, stop), NaN elsewhere. Proxy 'ns':
g = 1 / scale, the binned tcal@v0003 scale (noise-source based, with its S11 factors); proxy
'load': P_load / T_load from the tcal@v0003 load visits (EIGSEP_CAMPAIGN_ROOT), linear in
time between visits. Fitting it with one gain level fixes the gain to the calibration's.
"""
import argparse
import os
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("binned"); ap.add_argument("proxy", choices=["ns", "load"]); ap.add_argument("out")
ap.add_argument("mhz", type=float, nargs="+")
ap.add_argument("--start", default="2026-07-17T04:10")
ap.add_argument("--stop", default="2026-07-17T16:23")
args = ap.parse_args()
B = np.load(args.binned)
Z = np.load(Path(os.environ["EIGSEP_CAMPAIGN_ROOT"]) / "derived/tcal/v0003/solutions.npz")
t = B["t"]
t0, t1 = (np.datetime64(x).astype("datetime64[s]").astype(int) for x in (args.start, args.stop))
span = (B["regime"] == "rx-A") & (t >= t0) & (t < t1)
fz = Z["freqs"] / 1e6 if Z["freqs"].max() > 1e4 else Z["freqs"]
A = np.flatnonzero((Z["amb_regime"] == "rx-A") & (Z["amb_t"] < t1 + 600))
A = A[np.argsort(Z["amb_t"][A])]
TL = np.interp(Z["amb_t"][A], Z["therm_t"], Z["therm_k"])
y = np.full((len(t), len(args.mhz)), np.nan)
fo = []
for j, x in enumerate(args.mhz):
    c = int(np.argmin(np.abs(B["freqs_mhz"] - x)))
    fo.append(B["freqs_mhz"][c])
    if args.proxy == "ns":
        g = 1.0 / B["scale"][:, c]
    else:
        k = int(np.argmin(np.abs(fz - B["freqs_mhz"][c])))
        g = np.interp(t, Z["amb_t"][A], Z["amb_p"][A, k] / TL)
    m = span & np.isfinite(g) & (g > 0) & np.isfinite(B["y"][:, c])
    y[m, j] = B["y"][m, c] / (g[m] / np.median(g[m]))
np.savez(args.out, t=t, freqs_mhz=np.array(fo), y=y)
print(args.out, "bins per frequency", np.isfinite(y).sum(0))
