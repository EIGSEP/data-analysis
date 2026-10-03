#!/usr/bin/env python3
"""Per-integration transmitter-comb state on box-gnd, across the era.

The 8-channel (1.953125 MHz) comb on channels == 0 mod 8 is the beam-mapping
transmitter (memo 001, "Combs"). Before memo 001 it was called the "digital
self-comb", this script was `scan_self_comb_per_integration.py`, and its
output was `flags/diag/self_comb_per_integration_boxgnd.jsonl`. The
measurement is unchanged.

Why this exists
---------------
`flags/diag/comb_inventory.jsonl` reports one `lock8` per (file, input),
computed from a **median over all 240 integrations** of the file. That
statistic was built to characterise combs, and it does that well, but it
is the wrong instrument for *dating transitions*, in two demonstrated
ways:

* It **misses short episodes.** `corr_20260717_163125Z.h5` publishes
  `lock8 = 0.0` on both inputs while carrying the comb for its first 72
  integrations. A feature with less than roughly half-file duty cycle
  washes out of a 240-sample median and the file reads clean.
* Where the antenna is disconnected it **dates the antenna returning**
  rather than the comb starting. `corr_20260717_162916Z.h5`'s published
  detection rests entirely on its final 29 `RFANT` integrations, which
  are simply where box-air first gets an antenna-connected look.

Both bias the era's episode count and durations downward/late.

Scope: box-gnd only
-------------------
Per Aaron's 2026-09-20 decision, **box-gnd (input `0`) is the
authoritative source for transmitter on/off state and transition timing.** It
is conducted, not antenna-switched, so it sees through the calibration
and VNA cadence that makes box-air's readings ambiguous. This changes
what determines *state*; it does not change what downstream analysis
plots, which stays both boxes plus the cross.

Method
------
The statistic is `comb_inventory.tooth_contrast_locked` at spacing 8
channels -- **imported from that module, not reimplemented**, so this
cannot silently drift from the detector that produced the published
inventory. The only change is the time support: a median over `--block`
integrations (default 8, ~4.3 s) instead of over the whole file.

int32 auto wraps are repaired (`+2**32`) before the log, via
`detectors.overflow_mask`; a wrap is a huge downward excursion in log
space and would corrupt the tooth statistic.

A note on the statistic's floor, measured while calibrating this
------------------------------------------------------------------
On comb-free data the log-residual is **exactly zero** on more than half
the in-band channels -- `median_filter` over a quantised single-block
spectrum reproduces the central value exactly. The MAD scale then
collapses to 0 and `tooth_contrast_locked` returns its guard value 0.0.

This is benign for detection but must be stated: a returned 0.0 means
"no measurable tooth structure **or** a degenerate scale", and the two
are not distinguishable from the number alone. Checked on 750 sampled
era blocks: 4.0% had a degenerate (zero) scale, and every one of them
had an unnormalised best-phase tooth excess of exactly 0.00000 -- i.e.
the degeneracy never hid a comb. `tooth_excess_log10` is written
alongside the contrast so this stays auditable downstream.

Output: one provenance line, then one JSON line per (file, block):
    {file, block, int0, n_int, t_utc, lock8, phase, scale,
     tooth_excess_log10}

Usage
-----
    python scan_transmitter_per_integration.py --out <path.jsonl>
    python scan_transmitter_per_integration.py --era-only --block 8
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "comb_inventory", HERE / "comb_inventory.py")
CI = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CI)
DET = CI.D

# The transmitter era, by file close time: the first file in which box-gnd
# sees the 8-channel comb (07-17 15:36:22) through 07-18 03:00:13. The four
# files after that are not scanned. Used only by --era-only.
ERA_LO = "20260717153744"
ERA_HI = "20260718030013"

KEY = "0"          # box-gnd; the authoritative state source
SPACING = 8        # 1.953125 MHz = exactly 8.000 channels


def campaign_root():
    env = os.environ.get("MARJUM_DATA_ROOT")
    if env:
        return Path(env)
    from eigsep_data.paths import get_campaign_root
    return get_campaign_root(required=True)


def file_key(name):
    return name[5:13] + name[14:20]


def close_ts(name):
    return (datetime.strptime(name[5:20], "%Y%m%d_%H%M%S")
            .replace(tzinfo=timezone.utc).timestamp())


def scan_file(path, block):
    """Per-block tooth contrast at 8.000 ch for one file's box-gnd."""
    name = path.name
    with h5py.File(path, "r") as h:
        if KEY not in h["data"]:
            return []
        freqs = np.asarray(h["header/freqs"][:], dtype=float)
        raw = h["data/" + KEY][:]

    val = raw.astype(np.float64)
    ovf = DET.overflow_mask(raw)
    if ovf.any():
        val = np.where(ovf, val + 2.0 ** 32, val)
    logp = np.log10(np.maximum(val, 1.0))

    n_int = logp.shape[0]
    chans = np.arange(logp.shape[1])
    in_band = (freqs >= CI.BAND[0]) & (freqs <= CI.BAND[1])
    # Integration cadence from the file's own span; filename is the
    # CLOSE time, so block start = close - (n_int - int0) * dt.
    t_close = close_ts(name)
    dt = 128.85 / n_int if n_int else 0.0

    rows = []
    for b, s in enumerate(range(0, n_int, block)):
        sl = slice(s, s + block)
        med = np.median(logp[sl], axis=0)
        resid = med - median_filter(med, size=17, mode="nearest")
        c, ph = CI.tooth_contrast_locked(resid, chans, in_band, SPACING)
        scale = float(1.4826 * np.median(
            np.abs(resid[in_band] - np.median(resid[in_band]))))
        excess = max(
            float(np.median(resid[in_band & (chans % SPACING == q)])
                  - np.median(resid[in_band & ~(chans % SPACING == q)]))
            for q in range(SPACING))
        t = t_close - (n_int - s) * dt
        rows.append({
            "file": name,
            "block": b,
            "int0": int(s),
            "n_int": int(min(block, n_int - s)),
            "t_utc": datetime.fromtimestamp(t, timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "lock8": round(float(c), 3),
            "phase": int(ph),
            "scale": round(scale, 8),
            "tooth_excess_log10": round(excess, 6),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", type=int, default=8,
                    help="integrations per block (default 8, ~4.3 s)")
    ap.add_argument("--era-only", action="store_true",
                    help="restrict to the transmitter era (ERA_LO..ERA_HI)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = campaign_root()
    data = root / "data"
    files = sorted(data.glob("corr_*.h5"))
    if args.era_only:
        files = [f for f in files if ERA_LO <= file_key(f.name) <= ERA_HI]
    out_path = Path(args.out) if args.out else (
        root / "flags" / "diag" / "transmitter_per_integration_boxgnd.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sha = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short",
                          "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain",
                            "--", "."], capture_output=True,
                           text=True).stdout.strip()
    prov = {
        "product": "transmitter_per_integration_boxgnd",
        "campaign": "marjum-2026-07",
        "generator": "data-analysis/scripts/marjum-2026-07/flagging/"
                     "scan_transmitter_per_integration.py",
        "generator_commit": f"{sha}-dirty" if dirty else sha,
        "renamed_from": {
            "file": "flags/diag/self_comb_per_integration_boxgnd.jsonl",
            "generator": "scan_self_comb_per_integration.py",
            "reason": "the 8-channel comb is the transmitter, not a digital "
                      "self-comb (memo 001)"},
        "input": KEY, "spacing_ch": SPACING, "block_integrations": args.block,
        "files": ([ERA_LO, ERA_HI] if args.era_only else "all"),
    }
    n_rows = 0
    with out_path.open("w") as fh:
        fh.write(json.dumps({"provenance": prov}) + "\n")
        for i, path in enumerate(files):
            try:
                for rec in scan_file(path, args.block):
                    fh.write(json.dumps(rec) + "\n")
                    n_rows += 1
            except Exception as exc:                      # noqa: BLE001
                print(f"  ! {path.name}: {exc}", file=sys.stderr)
            if i % 50 == 0:
                print(f"  ...{i}/{len(files)}", file=sys.stderr)
    print(f"wrote {n_rows} block records for {len(files)} files -> {out_path}")


if __name__ == "__main__":
    main()
