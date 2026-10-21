"""T_gnd spectrum from one global fit, with an error from injection tests.

    tgnd_table.py FIT_JSON --inj TRUTH_JSON:INJ_JSON [...] [--exclude MHZ ...] [--out CSV]

T_gnd is the fit's ("all" rows). Each --inj pair is a mock built from TRUTH_JSON's fit
(``inject_mock.py``) and the refit of it (``global_fit.py --mock-y``); its deviation
recovered - true at each frequency is one draw of the fit's error. The error quoted is the
rms of the draws at that frequency (few draws: treat as a scale, not a precise sigma), with
the band rms of all draws alongside. Frequencies in --exclude, flagged diverged, or with no
draw are reported but marked.
"""
import argparse, json
import numpy as np

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("fit"); ap.add_argument("--inj", nargs="+", default=[])
ap.add_argument("--exclude", type=float, nargs="*", default=[])
ap.add_argument("--out", default=None)
args = ap.parse_args()
F = json.load(open(args.fit))
T = {round(r["freq_mhz"], 1): r["all"] for r in F["results"]}
dev = {f: [] for f in T}
for pair in args.inj:
    tj, ij = pair.split(":")
    tr = {round(r["freq_mhz"], 1): r["all"]["theta"]["T_gnd"] for r in json.load(open(tj))["results"]}
    for r in json.load(open(ij))["results"]:
        f = round(r["freq_mhz"], 1)
        if f in dev and f in tr and not r["all"]["diverged"]:
            dev[f].append(r["all"]["theta"]["T_gnd"] - tr[f])
alld = np.concatenate([v for v in dev.values() if v]) if any(dev.values()) else np.array([])
rows = []
for f, a in T.items():
    ex = any(abs(f - x) < 1.0 for x in args.exclude)
    d = np.array(dev[f])
    flag = "excluded" if ex else ("diverged" if a["diverged"] else ("no injection" if d.size == 0 else ""))
    rows.append((f, a["theta"]["T_gnd"], a["theta_err_scaled"]["T_gnd"],
                 float(np.sqrt(np.mean(d ** 2))) if d.size else np.nan, d.size, flag))
print(f"band rms of {alld.size} injection draws: {np.sqrt(np.mean(alld ** 2)):.0f} K, mean {alld.mean():+.0f} K")
print("  MHz    T_gnd  fit_err  inj_rms  n  note")
for f, t, e, s, n, fl in rows:
    print(f"{f:6.1f} {t:8.0f} {e:8.0f} {s:8.0f} {n:2d}  {fl}")
if args.out:
    with open(args.out, "w") as fh:
        fh.write("freq_mhz,T_gnd_K,fit_err_K,inj_rms_K,n_inj,note\n")
        for r in rows:
            fh.write(",".join(f"{x:.1f}" if isinstance(x, float) else str(x) for x in r) + "\n")
