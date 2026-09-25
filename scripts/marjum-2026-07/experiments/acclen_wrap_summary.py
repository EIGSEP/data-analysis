#!/usr/bin/env python
"""acclen_wrap_summary.py — reduce acclen_wrap_scan.jsonl to memo numbers.

Three questions, in order:
  1. which channels wrap, and how does incidence vary by era and acc_len;
  2. how big is the power error the wrap injects (lower bound);
  3. do noise statistics scale as 1/sqrt(N) on wrap-free channels across the
     2026-07-15T15:54:59Z doubling.

For (3) the contrast is matched on everything the mode table carries: phase C,
height era ~30m, rot_state parked, tx_comb off, and RF switch state, so the
only intended difference is corr_acc_len.
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOUNDARY = "2026-07-15T15:54:59Z"


def load():
    recs, prov = [], None
    with open(os.path.join(HERE, "acclen_wrap_scan.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            if "provenance" in d:
                prov = d["provenance"]
            else:
                recs.append(d)
    return prov, recs


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    prov, recs = load()
    recs = [r for r in recs if not r.get("error")]
    print(f"records: {len(recs)}  band {prov['band_mhz']} MHz\n")

    INPUTS = ["0", "3", "4", "5"]

    # ---------- 1. incidence ----------
    print("=" * 66)
    print("1. WRAP INCIDENCE")
    print("=" * 66)
    by_acc = defaultdict(lambda: [0, 0])
    by_era = defaultdict(lambda: [0, 0])
    for r in recs:
        wrapped = any(r.get(f"nneg_{k}", 0) > 0 for k in INPUTS)
        for key, tab in ((r.get("acc_len"), by_acc),
                         ((r.get("height_era"), r.get("acc_len")), by_era)):
            tab[key][0] += 1
            tab[key][1] += int(wrapped)

    print("\n  by corr_acc_len:")
    print("    acc_len        files   wrapped    rate")
    for k in sorted(by_acc, key=lambda x: (x is None, x)):
        n, w = by_acc[k]
        print(f"    {str(k):<14} {n:5d}   {w:5d}   {100*w/n:5.1f}%")

    print("\n  by height era x acc_len:")
    print("    era        acc_len        files   wrapped    rate")
    for k in sorted(by_era, key=lambda x: (str(x[0]), str(x[1]))):
        n, w = by_era[k]
        print(f"    {str(k[0]):<10} {str(k[1]):<14} {n:5d}   {w:5d}"
              f"   {100*w/n:5.1f}%")

    # which channels
    npz = os.path.join(HERE, "wrap_channels.npz")
    if os.path.exists(npz):
        z = np.load(npz)
        freqs = np.arange(1024) * (250.0 / 1024)
        print("\n  worst-offending channels (per acc_len|input group):")
        for key, counts in zip(z["keys"], z["counts"]):
            if counts.sum() == 0:
                continue
            top = np.argsort(counts)[::-1][:6]
            top = [t for t in top if counts[t] > 0]
            occupied = int((counts > 0).sum())
            print(f"    {key}:  {counts.sum():,} wrapped samples over "
                  f"{occupied} distinct channels")
            print("        " + ", ".join(
                f"{freqs[t]:.1f}MHz({counts[t]:,})" for t in top))

    # ---------- 2. magnitude ----------
    print("\n" + "=" * 66)
    print("2. POWER ERROR INJECTED BY THE WRAP  (lower bound)")
    print("=" * 66)
    print("\n  band-integrated fractional error (P_corrected - P_recorded)/P_corrected")
    print("    acc_len        input   n_wrapped_files   median      p90       max")
    for acc in sorted({r.get("acc_len") for r in recs},
                      key=lambda x: (x is None, x)):
        for k in INPUTS:
            e = [r[f"bandpow_err_{k}"] for r in recs
                 if r.get("acc_len") == acc and r.get(f"bandpow_err_{k}", 0) > 0]
            if not e:
                continue
            print(f"    {str(acc):<14} {k:<6}  {len(e):6d}          "
                  f"{np.median(e):.3e}  {pct(e,90):.3e}  {max(e):.3e}")

    # ---------- 3. noise scaling ----------
    print("\n" + "=" * 66)
    print("3. NOISE SCALING ACROSS THE DOUBLING  (wrap-free channels)")
    print("=" * 66)

    def sel(acc, before):
        out = []
        for r in recs:
            if r.get("acc_len") != acc:
                continue
            if (r["t_utc"] < BOUNDARY) != before:
                continue
            if r.get("phase") != "C" or r.get("height_era") != "~30m":
                continue
            if r.get("rot_state") != "parked" or r.get("tx_comb") != "off":
                continue
            if r.get("rfswitch") != "RFANT":
                continue
            out.append(r)
        return out

    pre = sel(67108864, True)
    post = sel(134217728, False)
    print(f"\n  matched windows (phase C, ~30m, parked, tx off, RFANT):")
    print(f"    before: {len(pre):3d} files  "
          f"{min((r['t_utc'] for r in pre), default='-')} .. "
          f"{max((r['t_utc'] for r in pre), default='-')}")
    print(f"    after : {len(post):3d} files  "
          f"{min((r['t_utc'] for r in post), default='-')} .. "
          f"{max((r['t_utc'] for r in post), default='-')}")

    print("\n    input   n_pre  n_post   fracRMS_pre   fracRMS_post"
          "    ratio   expected")
    for k in ["0", "4"]:
        a = [r[f"fracrms_med_{k}"] for r in pre if f"fracrms_med_{k}" in r]
        b = [r[f"fracrms_med_{k}"] for r in post if f"fracrms_med_{k}" in r]
        # drop files whose median fractional RMS is wildly off: those are
        # files with a state change or calibration switching mid-file.
        a = [x for x in a if 1e-5 < x < 0.05]
        b = [x for x in b if 1e-5 < x < 0.05]
        if not a or not b:
            print(f"    {k:<6}  insufficient matched files ({len(a)}, {len(b)})")
            continue
        ma, mb = np.median(a), np.median(b)
        print(f"    {k:<6}  {len(a):4d}   {len(b):4d}    {ma:.5e}   "
              f"{mb:.5e}   {mb/ma:.4f}   {1/np.sqrt(2):.4f}")
        print(f"            pre  IQR [{pct(a,25):.3e}, {pct(a,75):.3e}]"
              f"   post IQR [{pct(b,25):.3e}, {pct(b,75):.3e}]")


if __name__ == "__main__":
    main()
