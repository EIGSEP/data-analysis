#!/usr/bin/env python
"""mux_copy_scan.py — differencing experiment: ADC mux copies (4->5, 0->1).

The SNAP ADC mux duplicates one digitised input stream into a second correlator
input slot. The same analogue signal therefore traverses two *digital* paths
(two F-engine input slots, two X-engine auto/cross products). Differencing the
two copies bounds everything the digital chain contributes downstream of the
mux tap, with the sky, the analogue chain and the ADC held exactly fixed.

What this script measures, per file:

  stage 1 (all 5,120 files, attrs only)
    - `mux_copy_4to5`, `mux_copy_0to1` as recorded by `data/filter_corr_keys.py`
      at filter time, when inputs 1 and 5 still existed. These are
      `np.array_equal(dX, dY) and dX.any()` — note the `.any()`: a dead or
      all-zero input reads as False regardless of mux state, so the attr alone
      cannot separate "mux off" from "input dead".
    - whether the retained science inputs are all-zero, which disambiguates it.

  stage 2 (phase B only, 1,111 files, full data)
    - direct bitwise comparison of `data/4` vs `data/5`, which survive the
      filter in phase B. Reports exact mismatch counts and max |difference| in
      accumulator LSB, plus the median auto power, so a null result can be
      quoted as a fractional bound rather than a bare "identical".
    - Cauchy-Schwarz check on the mux-derived cross `35`:
      |V_35| / sqrt(P_3 P_5) <= 1. The mux exists to synthesise a 3x4 cross
      that the correlator never computed directly; this validates that the
      synthesised product is physical and correctly paired.

Times come from filenames (file close time, UTC). `header/times` is bad in 642
files and is deliberately not used.

Usage:
    python mux_copy_scan.py [--limit N] [--out DIR]
"""
import argparse
import datetime as dt
import glob
import json
import os

import h5py
import numpy as np

from eigsep_data.paths import campaign_data_dir

DATA = str(campaign_data_dir())
PHASE_B_START = "corr_20260714_041043"
PHASE_C_START = "corr_20260715_003217"

# Files the curation layer flags as corrupt; skipped everywhere downstream.
CORRUPT = {
    "corr_20260715_044343Z.h5",
    "corr_20260715_213105Z.h5",
    "corr_20260718_032419Z.h5",
}


def t_from_name(name):
    """File close time, UTC, from the filename. Never from header/times."""
    stem = os.path.basename(name).split(".")[0]  # corr_20260714_100016Z[-1]
    stamp = stem.split("_")[1] + stem.split("_")[2][:6]
    return dt.datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(
        tzinfo=dt.timezone.utc
    )


def phase_of(name):
    base = os.path.basename(name)
    if base < PHASE_B_START:
        return "A"
    if base < PHASE_C_START:
        return "B"
    return "C"


def scan_file(path, deep):
    """One file -> record dict. `deep` enables the phase-B data comparison."""
    base = os.path.basename(path)
    rec = {
        "file": base,
        "t_utc": t_from_name(path).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "phase": phase_of(path),
    }
    with h5py.File(path, "r") as f:
        rec["attr_eq45"] = bool(f.attrs.get("mux_copy_4to5", False))
        rec["attr_eq01"] = bool(f.attrs.get("mux_copy_0to1", False))
        keys = list(f["data"].keys())
        rec["keys"] = ",".join(sorted(keys))

        # Zero-liveness of the retained autos, to disambiguate the attrs, and
        # accumulator-overflow counts: the autos are int32 and a
        # positive-definite quantity that reads negative has wrapped.
        for k in ("0", "2", "3", "4", "5"):
            if k in keys:
                a = f["data"][k][:]
                rec[f"live_{k}"] = bool(np.any(a))
                rec[f"nneg_{k}"] = int(np.count_nonzero(a < 0))
                pos = a[a > 0]
                rec[f"med_{k}"] = float(np.median(pos)) if pos.size else 0.0

        if not deep or not {"4", "5"}.issubset(keys):
            return rec

        d4 = f["data"]["4"][:]
        d5 = f["data"]["5"][:]
        diff = d5.astype(np.int64) - d4.astype(np.int64)
        rec["n_samp"] = int(d4.size)
        rec["n_mismatch"] = int(np.count_nonzero(diff))
        rec["max_absdiff_lsb"] = int(np.abs(diff).max())
        rec["med_power_4"] = float(np.median(d4))
        rec["med_power_5"] = float(np.median(d5))

        # Cauchy-Schwarz on the mux-derived cross, |V_35| <= sqrt(P_3 P_5).
        # Gated on power: input 3 is notched near 133 MHz and rolls off above
        # ~244 MHz, where P_3 falls to a few counts and the ratio is a
        # zero-denominator artefact rather than a statement about the cross.
        # Files where the test cannot be posed are recorded in `coh_skip`
        # rather than silently contributing violations.
        if "3" in keys and "35" in keys:
            d3 = f["data"]["3"][:].astype(np.float64)
            c = f["data"]["35"][:].astype(np.float64)
            v = np.abs(c[..., 0] + 1j * c[..., 1])
            p3, p5 = d3, d5
            # Two ways this test goes meaningless, neither about the mux:
            #  - an input is dead (median a few counts), so a *relative* power
            #    gate admits pure-noise channels;
            #  - an int32 accumulator has wrapped, so the recorded power is not
            #    the true power. Both are excluded rather than reported as
            #    Cauchy-Schwarz failures.
            # In-band only. The signal chain carries BLP-200 on the inputs and
            # SLP-200+ on the Tx (CAMPAIGN.md 07-12, 07-17), so power above
            # ~200 MHz is low-pass stopband residue; the top channel
            # (249.3 MHz) is where every residual excursion lives.
            freqs = f["header"]["freqs"][:]
            inband = (freqs >= 50.0) & (freqs <= 200.0)
            m3 = np.median(p3[:, inband]) if inband.any() else 0.0
            m5 = np.median(p5[:, inband]) if inband.any() else 0.0
            DEAD_FLOOR = 1000.0  # counts; live inputs sit at 1e4-1e6
            rec["coh_skip"] = None
            if m3 < DEAD_FLOOR or m5 < DEAD_FLOOR:
                rec["coh_skip"] = "input-dead"
            elif (d3 < 0).any() or (d5 < 0).any():
                rec["coh_skip"] = "accumulator-wrapped"
            else:
                ok = np.zeros(p3.shape, bool)
                ok[:, inband] = True
                ok &= (p3 > 0.05 * m3) & (p5 > 0.05 * m5)
                if ok.any():
                    coh = v[ok] / np.sqrt(p3[ok] * p5[ok])
                    rec["coh_max"] = float(coh.max())
                    rec["coh_med"] = float(np.median(coh))
                    rec["n_coh_gt1"] = int(np.count_nonzero(coh > 1.0 + 1e-12))
                    rec["n_coh_eval"] = int(ok.sum())
                    rec["coh_frac_masked"] = float(1.0 - ok.mean())
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=os.path.dirname(__file__))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DATA, "corr_*.h5")))
    files = [f for f in files if os.path.basename(f) not in CORRUPT]
    if args.limit:
        files = files[:: max(1, len(files) // args.limit)][: args.limit]

    recs = []
    for i, path in enumerate(files):
        deep = phase_of(path) == "B"
        try:
            recs.append(scan_file(path, deep))
        except Exception as exc:  # keep going; report at the end
            recs.append({"file": os.path.basename(path), "error": repr(exc)})
        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    out = os.path.join(args.out, "mux_copy_scan.jsonl")
    prov = {
        "provenance": {
            "product": "mux_copy_scan",
            "campaign": "marjum-2026-07",
            "version": "v1",
            "generated_utc": dt.datetime.now(dt.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "generator": "experiments/mux_copy_scan.py",
            "inputs": ["data/corr_*.h5 (filtered)"],
            "n_files": len(files),
            "notes": "times from filenames; header/times unused (bad in 642 files)",
        }
    }
    with open(out, "w") as fh:
        fh.write(json.dumps(prov) + "\n")
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {out} ({len(recs)} records)")


if __name__ == "__main__":
    main()
