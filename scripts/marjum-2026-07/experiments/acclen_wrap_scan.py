#!/usr/bin/env python
"""acclen_wrap_scan.py — experiment 2: the corr_acc_len doubling and the wrap.

MEMO-001 found the int32 auto accumulators wrap, and that the rate tracks
`corr_acc_len` (5.4% of files before the 2026-07-15T15:54:59Z doubling, 25.6%
after). This pass characterises what that did to the data.

Per file it records:

  wrap incidence
    - per-channel wrap counts, accumulated globally per (input, acc_len), so
      we can say *which* channels overflow rather than just how many files.

  magnitude of the instrumental step
    - band-integrated power as recorded, and again with wrapped samples
      repaired (a negative int32 auto is the true value minus 2^32, so adding
      2^32 back recovers it). The difference is the error the wrap injects
      into any total-power statistic. This is sky-independent: both numbers
      come from the same file, so LST drift cancels exactly and no matched
      before/after sky window is needed.
      CAVEAT: only wraps that land in negative territory are detectable. A
      channel that wrapped far enough to return to positive values is
      invisible here, so every number below is a LOWER BOUND on the damage.

  noise statistics on wrap-free channels
    - fractional RMS from successive differences along time,
      sigma = std(P[i+1]-P[i]) / sqrt(2), reported as sigma/mean. The
      lag-1 difference suppresses slow sky and gain drift, isolating the white
      component. Radiometrically sigma/mean scales as 1/sqrt(N_acc), so the
      doubling should drop it by 1/sqrt(2) = 0.7071. Computed only on channels
      with no wrapped sample anywhere in the file.

Times from filenames; `header/times` is bad in 642 files and is not used.

Usage:
    python acclen_wrap_scan.py [--limit N] [--out DIR]
"""
import argparse
import bisect
import datetime as dt
import glob
import json
import os

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
from eigsep_data.paths import campaign_data_dir

DATA = str(campaign_data_dir())
MODE_TABLE = os.path.join(HERE, "..", "curation", "mode_table.jsonl")

CORRUPT = {
    "corr_20260715_044343Z.h5",
    "corr_20260715_213105Z.h5",
    "corr_20260718_032419Z.h5",
}
WRAP = 2 ** 32
BAND = (50.0, 200.0)  # MHz; BLP-200 on the inputs, SLP-200+ on the Tx


def t_from_name(name):
    stem = os.path.basename(name).split(".")[0]
    parts = stem.split("_")
    return dt.datetime.strptime(
        parts[1] + parts[2][:6], "%Y%m%d%H%M%S"
    ).replace(tzinfo=dt.timezone.utc)


def load_mode_index():
    """(sorted starts, rows) from the mode table, for acc_len / era lookup."""
    rows = []
    with open(MODE_TABLE) as fh:
        for line in fh:
            d = json.loads(line)
            if "provenance" not in d:
                rows.append(d)
    rows.sort(key=lambda r: r["t_start_utc"])
    return [r["t_start_utc"] for r in rows], rows


def mode_of(starts, rows, t):
    i = bisect.bisect_right(starts, t) - 1
    if i < 0:
        return None
    w = rows[i]
    return w if t <= w["t_end_utc"] else None


def scan_file(path, starts, rows, wrap_hist):
    base = os.path.basename(path)
    t = t_from_name(path).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec = {"file": base, "t_utc": t}
    w = mode_of(starts, rows, t)
    if w is not None:
        rec["acc_len"] = w["corr_acc_len"]
        rec["height_era"] = w["height_era"]
        rec["phase"] = w["phase"]
        rec["rfswitch"] = w["rfswitch_dominant"]
        rec["rot_state"] = w["rot_state"]
        rec["tx_comb"] = w["tx_comb"]

    with h5py.File(path, "r") as f:
        freqs = f["header"]["freqs"][:]
        inband = (freqs >= BAND[0]) & (freqs <= BAND[1])
        for k in ("0", "2", "3", "4", "5"):
            if k not in f["data"]:
                continue
            a = f["data"][k][:]
            if a.ndim != 2:
                continue
            neg = a < 0
            nneg = int(neg.sum())
            rec[f"nneg_{k}"] = nneg

            # per-channel wrap histogram, keyed by (acc_len, input)
            if nneg:
                key = (rec.get("acc_len"), k)
                h = wrap_hist.setdefault(key, np.zeros(a.shape[1], np.int64))
                h += neg.sum(axis=0)

            corrected = a.astype(np.int64)
            corrected[neg] += WRAP

            rec_pow = a.astype(np.int64)[:, inband].sum()
            cor_pow = corrected[:, inband].sum()
            rec[f"bandpow_rec_{k}"] = int(rec_pow)
            rec[f"bandpow_cor_{k}"] = int(cor_pow)
            if cor_pow > 0:
                rec[f"bandpow_err_{k}"] = float(
                    (cor_pow - rec_pow) / cor_pow
                )

            # noise statistics on wrap-free in-band channels
            wrapfree = inband & ~neg.any(axis=0)
            rec[f"n_wrapfree_{k}"] = int(wrapfree.sum())
            if wrapfree.sum() and a.shape[0] > 8:
                p = a[:, wrapfree].astype(np.float64)
                mean = p.mean(axis=0)
                good = mean > 0
                if good.any():
                    d = np.diff(p[:, good], axis=0)
                    sig = d.std(axis=0, ddof=1) / np.sqrt(2.0)
                    frac = sig / mean[good]
                    rec[f"fracrms_med_{k}"] = float(np.median(frac))
                    rec[f"fracrms_n_{k}"] = int(good.sum())
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()

    starts, rows = load_mode_index()
    files = sorted(glob.glob(os.path.join(DATA, "corr_*.h5")))
    files = [f for f in files if os.path.basename(f) not in CORRUPT]
    if args.limit:
        files = files[:: max(1, len(files) // args.limit)][: args.limit]

    wrap_hist = {}
    recs = []
    for i, path in enumerate(files):
        try:
            recs.append(scan_file(path, starts, rows, wrap_hist))
        except Exception as exc:
            recs.append({"file": os.path.basename(path), "error": repr(exc)})
        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    out = os.path.join(args.out, "acclen_wrap_scan.jsonl")
    with open(out, "w") as fh:
        fh.write(json.dumps({"provenance": {
            "product": "acclen_wrap_scan",
            "campaign": "marjum-2026-07",
            "version": "v1",
            "generated_utc": dt.datetime.now(dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"),
            "generator": "experiments/acclen_wrap_scan.py",
            "inputs": ["marjum-2026-07/data/corr_*.h5 (filtered)",
                       "marjum-2026-07/curation/mode_table.jsonl"],
            "n_files": len(files),
            "band_mhz": list(BAND),
            "notes": ("bandpow_err is a LOWER BOUND: only wraps landing "
                      "negative are detectable. times from filenames."),
        }}) + "\n")
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {out} ({len(recs)} records)")

    if wrap_hist:
        keys = sorted(wrap_hist, key=lambda k: (str(k[0]), k[1]))
        np.savez(
            os.path.join(args.out, "wrap_channels.npz"),
            keys=np.array([f"{k[0]}|{k[1]}" for k in keys]),
            counts=np.stack([wrap_hist[k] for k in keys]),
        )
        print(f"wrote wrap_channels.npz ({len(keys)} (acc_len, input) groups)")


if __name__ == "__main__":
    main()
