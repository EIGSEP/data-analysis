"""The receiver gain from raw load visits in the 30 m era, against a global fit's knot gain.

    cal_gain_30m.py FIT_JSON FIT_RESID_NPZ OUT.json [--start ... --stop ...]

tcal@v0003 has no calibration before 07-17, so this reads the raw box-air correlator files
(EIGSEP_CAMPAIGN_ROOT/data) between --start and --stop and finds the RF switch's load
(RFAMB) visits. Each visit's spectrum is the median of its rows, the first two after the
switch dropped. The gain proxy is that power over the load's temperature
(tempctrl_load_T_now), at each of the fit's frequencies: it measures the gain after the
switch only. A visit is excluded when, at more than --max-bad of those frequencies, its
proxy differs by more than --tol from the median over the visits (contamination such as
07-16's comb EMI). Against it: the fit's rx-A knot gain and the GSM column at the visit
times. The raw files read are listed with their SHA-256.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from eigsep_data import MetadataIndex

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("fit"); ap.add_argument("resid"); ap.add_argument("out")
ap.add_argument("--start", default="2026-07-16T01:10:00Z")
ap.add_argument("--stop", default="2026-07-16T02:35:00Z")
ap.add_argument("--tol", type=float, default=0.10)
ap.add_argument("--max-bad", type=int, default=2)
args = ap.parse_args()
campaign = Path(os.environ["EIGSEP_CAMPAIGN_ROOT"])
J = json.load(open(args.fit)); R = np.load(args.resid)
sel = MetadataIndex(campaign / "data").select(time=(args.start, args.stop))
B = sel.load_bundle(antenna="box-air")
meta = B.meta.reset_index(drop=True)
t = np.asarray(B.t, float)
rf = meta.rfswitch.astype(str).to_numpy()
TL = pd.to_numeric(meta.tempctrl_load_T_now, errors="coerce").to_numpy(float) + 273.15
f = np.asarray(B.freqs_mhz, float)
D = np.asarray(B.data, float)
fits = [r["freq_mhz"] for r in J["results"]]
ch = [int(np.argmin(np.abs(f - x))) for x in fits]

m = rf == "RFAMB"
visits = []
for s in np.flatnonzero(m & ~np.r_[False, m[:-1]]):
    e = s
    while e < len(m) and m[e]:
        e += 1
    rows = np.arange(s + 2, e)
    if rows.size >= 3:
        P = np.median(D[rows][:, ch], 0)
        visits.append({"t": float(t[rows].mean()), "n_rows": int(rows.size),
                       "T_load_K": float(np.nanmedian(TL[rows])), "proxy": (P / np.nanmedian(TL[rows])).tolist()})
G = np.array([v["proxy"] for v in visits])
med = np.median(G, 0)
for v, g in zip(visits, G):
    bad = np.abs(g / med - 1) > args.tol
    v["n_bad"] = int(bad.sum())
    v["bad_mhz"] = [fits[j] for j in np.flatnonzero(bad)]
    v["used"] = v["n_bad"] <= args.max_bad
use = [v for v in visits if v["used"]]
tu = np.array([v["t"] for v in use])
rx = R["regime"] == "rx-A"
per = []
for j, x in enumerate(fits):
    gk = J["results"][j]["all"]["gain_knots"]
    seg = np.array(gk["segment"]); ok = np.array([s.startswith("rx-A") for s in seg])
    kt, kg = np.array(gk["t"])[ok], np.array(gk["gain"])[ok]
    o = np.argsort(kt)
    fit = np.interp(tu, kt[o], kg[o])
    cal = np.array([v["proxy"][j] for v in use])
    sky = np.interp(tu, R["t"][rx], R["cGSM"][j, rx])
    per.append({"freq_mhz": x, "diverged": J["results"][j]["all"]["diverged"],
                "cal_change": float(cal[-1] / cal[0] - 1), "fit_change": float(fit[-1] / fit[0] - 1),
                "sky_change": float(sky[-1] / sky[0] - 1)})
era = [str(e) for e in np.unique(R["era"][(R["t"] >= tu.min()) & (R["t"] <= tu.max())])]
files = sorted(set(str(campaign / "data" / p) for p in sel.files))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


out = {"start": args.start, "stop": args.stop, "tol": args.tol, "max_bad": args.max_bad, "era": era,
       "visits": visits, "used_t": tu.tolist(), "per_freq": per,
       "files": [{"path": str(Path(p).relative_to(campaign)), "sha256": sha256(p)} for p in files]}
json.dump(out, open(args.out, "w"), indent=1)
print(f"{len(visits)} load visits, {len(use)} used; {len(files)} raw files; wrote {args.out}")
