"""Campaign-wide comb + band-power scan.

Diagnostic pass, not a mask product.  Its job is to locate the *real*
windows of the emitters CAMPAIGN.md describes from field notes, so the
mask categories are validated against measured behaviour rather than
against a transcribed time that may be approximate.

Emits one JSON record per (file, input) to stdout as JSONL.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np

from eigsep_data.flagging import detectors as D

SPACINGS = [0.625, 1.000, 1.250, 2.000, 4.000]
BANDS = {
    "fm": D.BAND_FM,
    "dtv_lo": D.BAND_DTV_LO,
    "dtv_hi": D.BAND_DTV_HI,
    "orbcomm": D.BAND_ORBCOMM,
    "laptop": D.BAND_LAPTOP,
}


def scan_one(path):
    fn = os.path.basename(path)
    out = []
    try:
        with h5py.File(path, "r") as h:
            freqs = h["header/freqs"][:]
            keys = [k for k in h["data"] if len(k) == 1]
            rfsw = None
            if "metadata" in h and "rfswitch" in h["metadata"]:
                rfsw = h["metadata/rfswitch"][()]
            for k in keys:
                d = h["data/" + k][:]
                nt = d.shape[0]
                ant = D.antenna_mask(rfsw, nt)
                if ant.sum() < 4:
                    ant = np.ones(nt, dtype=bool)
                logp = np.log10(np.maximum(d[ant].astype(np.float64), 1.0))
                med = np.median(logp, axis=0)
                if not np.isfinite(med).all():
                    continue
                rec = {
                    "file": fn,
                    "input": k,
                    "n_time": int(nt),
                    "n_ant": int(ant.sum()),
                    "power": round(float(np.median(med)), 4),
                }
                for s in SPACINGS:
                    band = D.BAND_LAPTOP if s == 2.000 else (50.0, 200.0)
                    rec[f"comb{s:g}"] = round(
                        D.comb_snr(med, freqs, s, band=band), 2)
                    if s == 2.000:
                        rec["comb2_wide"] = round(
                            D.comb_snr(med, freqs, s, band=(50.0, 200.0)), 2)
                for name, (lo, hi) in BANDS.items():
                    sel = (freqs >= lo) & (freqs <= hi)
                    rec[f"p_{name}"] = round(float(np.median(med[sel])), 4)
                out.append(rec)
    except Exception as exc:  # a corrupt file must not kill the scan
        out.append({"file": fn, "error": f"{type(exc).__name__}: {exc}"})
    return out


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
    n_workers = int(os.environ.get("NWORKERS", "8"))
    done = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for recs in ex.map(scan_one, files, chunksize=8):
            for r in recs:
                sys.stdout.write(json.dumps(r) + "\n")
            done += 1
            if done % 250 == 0:
                print(f"# {done}/{len(files)}", file=sys.stderr, flush=True)
    print(f"# done {done}/{len(files)}", file=sys.stderr)


if __name__ == "__main__":
    main()
