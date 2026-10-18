"""The receiver gain from the RF-switch calibration visits, against a global fit's knot gain
and the sky drift.

    cal_gain.py FIT_JSON FIT_RESID_NPZ OUT.json [--start ... --stop ...]

The calibration visits are those of tcal@v0003 (derived/tcal/v0003/solutions.npz under
EIGSEP_CAMPAIGN_ROOT): rx-A load (RFAMB) and noise-source (RFNON) spectra, paired within 5 min.
Two gain proxies per visit, at each of the fit's frequencies:
  ns:   P_NS - P_load (gain x the noise source's excess, which itself drifts);
  load: P_load / T_load (gain x (T_load + T_rx) / T_load), T_load from the load's thermistor.
Both measure the gain after the RF switch only. Against each: the fit's log gain (its rx-A
knots interpolated to the visit times) and the log GSM column (the fit's residual file), over
the span and after subtracting a running 3-h mean; and the switch thermistor
(rfswitch_therm_temp_therm0 from the correlator metadata) at the visit times.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("fit"); ap.add_argument("resid"); ap.add_argument("out")
ap.add_argument("--start", default="2026-07-17T04:00")
ap.add_argument("--stop", default="2026-07-17T16:30")
ap.add_argument("--detrend-s", type=float, default=10800.0)
args = ap.parse_args()
campaign = Path(os.environ["EIGSEP_CAMPAIGN_ROOT"])
Z = np.load(campaign / "derived/tcal/v0003/solutions.npz")
J = json.load(open(args.fit)); R = np.load(args.resid)
t0, t1 = (np.datetime64(x).astype("datetime64[s]").astype(int) for x in (args.start, args.stop))
fz = Z["freqs"] / 1e6 if Z["freqs"].max() > 1e4 else Z["freqs"]
ta, tn = Z["amb_t"], Z["non_t"]
ia = np.flatnonzero((Z["amb_regime"] == "rx-A") & (ta >= t0) & (ta < t1))
inn = np.flatnonzero(Z["non_regime"] == "rx-A")
pairs = []
for i in ia:
    j = inn[np.argmin(np.abs(tn[inn] - ta[i]))]
    if abs(tn[j] - ta[i]) < 300:
        pairs.append((i, j))
pairs = sorted(pairs, key=lambda p: ta[p[0]])
t = np.array([0.5 * (ta[i] + tn[j]) for i, j in pairs])
TL = np.interp(t, Z["therm_t"], Z["therm_k"])
fits = [r["freq_mhz"] for r in J["results"]]
ch = [int(np.argmin(np.abs(fz - x))) for x in fits]
P_amb = np.array([Z["amb_p"][i][ch] for i, _ in pairs])
P_non = np.array([Z["non_p"][j][ch] for _, j in pairs])
G = {"ns": P_non - P_amb, "load": P_amb / TL[:, None]}

from eigsep_data import MetadataIndex
meta = MetadataIndex(campaign / "data").select(time=(float(t.min()) - 600, float(t.max()) + 600)).meta
import pandas as pd
tm = pd.to_numeric(meta["time"], errors="coerce").to_numpy(float)
th = pd.to_numeric(meta["rfswitch_therm_temp_therm0"], errors="coerce").to_numpy(float)
ok = np.isfinite(tm) & np.isfinite(th)
o = np.argsort(tm[ok])
T_sw = np.interp(t, tm[ok][o], th[ok][o])


def knots(j):
    gk = J["results"][j]["all"]["gain_knots"]
    seg = np.array(gk["segment"]); sel = np.array([s.startswith("rx-A") for s in seg])
    kt, kg = np.array(gk["t"])[sel], np.array(gk["gain"])[sel]
    o = np.argsort(kt)
    return np.interp(t, kt[o], kg[o])


rx = R["regime"] == "rx-A"


def dt(x):
    return np.array([x[i] - x[np.abs(t - t[i]) <= args.detrend_s / 2].mean() for i in range(len(t))])


def corr(u, v):
    return float(np.corrcoef(u, v)[0, 1])


def slope(x, y):
    return float(np.polyfit(x, y, 1)[0])


per = []
for j, x in enumerate(fits):
    a = J["results"][j]["all"]
    lf, ls = np.log(knots(j)), np.log(np.interp(t, R["t"][rx], R["cGSM"][j, rx]))
    row = {"freq_mhz": x, "diverged": a["diverged"], "a_tcal": (a.get("a_tcal") or {}).get("mean"),
           "fit_rms": float(lf.std()), "sky_rms": float(ls.std()), "r_fit_sky": corr(lf, ls),
           "r_fit_sky_fast": corr(dt(lf), dt(ls)), "fit_fast_rms": float(dt(lf).std())}
    for k, g in G.items():
        lg = np.log(g[:, j])
        if not np.all(np.isfinite(lg)):
            row[k] = None
            continue
        row[k] = {"rms": float(lg.std()), "fast_rms": float(dt(lg).std()), "r_sky": corr(lg, ls),
                  "r_sky_fast": corr(dt(lg), dt(ls)), "r_fit": corr(lg, lf), "r_fit_fast": corr(dt(lg), dt(lf)),
                  "slope_therm0": slope(T_sw, lg), "r_therm0": corr(T_sw, lg),
                  "slope_therm0_fast": slope(dt(T_sw), dt(lg)), "r_therm0_fast": corr(dt(T_sw), dt(lg))}
    per.append(row)

band = [j for j, x in enumerate(fits) if 110 < x < 205 and not J["results"][j]["all"]["diverged"]
        and all(np.all(np.isfinite(np.log(g[:, j]))) for g in G.values())]
cm = {k: np.mean([np.log(g[:, j]) - np.log(g[:, j]).mean() for j in band], 0) for k, g in G.items()}
cm["fit"] = np.mean([np.log(knots(j)) - np.log(knots(j)).mean() for j in band], 0)
common = {"freqs": [fits[j] for j in band]}
for k, v in cm.items():
    common[k] = {"rms": float(v.std()), "fast_rms": float(dt(v).std()), "slope_therm0": slope(T_sw, v),
                 "r_therm0": corr(T_sw, v), "slope_therm0_fast": slope(dt(T_sw), dt(v)),
                 "r_therm0_fast": corr(dt(T_sw), dt(v)), "series": v.tolist()}
common["r_ns_fit_fast"] = corr(dt(cm["ns"]), dt(cm["fit"]))
out = {"n_pairs": len(pairs), "t": t.tolist(), "span": [float(t.min()), float(t.max())],
       "spacing_min": float(np.median(np.diff(t)) / 60), "T_load_K": [float(TL.min()), float(TL.max())],
       "therm0_C": [float(T_sw.min()), float(T_sw.max())], "therm0_fast_rms": float(dt(T_sw).std()),
       "therm0": T_sw.tolist(), "detrend_s": args.detrend_s, "per_freq": per, "common": common,
       "series": {}}
for x in (66.4, 148.4):
    j = int(np.argmin(np.abs(np.array(fits) - x)))
    out["series"][f"{fits[j]:.1f}"] = {"ns": np.log(G["ns"][:, j]).tolist(), "load": np.log(G["load"][:, j]).tolist(),
                                       "fit": np.log(knots(j)).tolist(),
                                       "sky": np.log(np.interp(t, R["t"][rx], R["cGSM"][j, rx])).tolist()}
json.dump(out, open(args.out, "w"))
print(f"{len(pairs)} visit pairs; wrote {args.out}")
