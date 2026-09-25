#!/usr/bin/env python3
"""Audit the B7 solved cycles for int32 accumulator wrap.

Per natural-experimenter MEMO-006 (renumbered from their self-assigned
002 -- numbers are assigned through MEMO-000, not self-assigned), the int32
auto accumulators wrap on the
brightest channels -- 11.7% of files campaign-wide, 25.6% in the post-doubling
era, which is exactly the 07-17/18 calibration window. A wrapped sample
reinterprets as a large negative int32.

This checks the actual arrays B7 consumes, per switch state, so the answer is
specific to the calibration solve rather than to the file as a whole:

  RFAMB / RFNON : feed the (g, T_rx) solve. Low-power states; expected safe.
  RFANT         : feeds T_ant. Bright sky; the one at risk.

Reports per-state negative counts, the in-band (55-85 MHz) restriction, and
which cycles/channels would need excluding.
"""
from __future__ import annotations

import argparse
import json

import h5py
import numpy as np
from eigsep_data import select_files
from eigsep_data.paths import get_campaign_root

import abscal

CURATION = get_campaign_root(required=True) / "curation"
INT32_MAX = 2**31 - 1


def allowed(phase, start, end):
    kept, _dropped, _warnings = select_files.select(phase=phase, start=start, end=end)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="C")
    ap.add_argument("--start", default="2026-07-17T04:00:00Z")
    ap.add_argument("--end", default="2026-07-18T04:00:00Z")
    ap.add_argument("--input", default=abscal.SWITCHED_INPUT)
    ap.add_argument("--fmin", type=float, default=55.0)
    ap.add_argument("--fmax", type=float, default=85.0)
    args = ap.parse_args()

    names = allowed(args.phase, args.start, args.end)
    print(f"auditing {len(names)} curated files, input {args.input}")

    states_of_interest = ("RFANT", "RFAMB", "RFNON")
    tot = {s: 0 for s in states_of_interest}
    neg = {s: 0 for s in states_of_interest}
    neg_band = {s: 0 for s in states_of_interest}
    files_hit, chans_hit = set(), set()
    worst = 0

    for n in names:
        p = abscal.DATA / n
        with h5py.File(p, "r") as f:
            if "rfswitch" not in f["metadata"]:
                continue
            st = np.array([str(x) for x in
                           json.loads(f["metadata/rfswitch"][()])])
            if args.input not in f["data"]:
                continue
            d = f["data"][args.input][:, :]          # int32, NOT cast
            freqs = f["header/freqs"][:]
        band = (freqs >= args.fmin) & (freqs <= args.fmax)
        for s in states_of_interest:
            m = st == s
            if not m.any():
                continue
            blk = d[m]
            tot[s] += blk.size
            nneg = int((blk < 0).sum())
            neg[s] += nneg
            nb = int((blk[:, band] < 0).sum())
            neg_band[s] += nb
            if nneg:
                files_hit.add(n)
                ch = np.where((blk < 0).any(axis=0))[0]
                chans_hit.update(freqs[ch].tolist())
                worst = min(worst, int(blk.min()))

    print()
    print("=" * 68)
    print("INT32 WRAP AUDIT  (negative accumulator samples = wrapped)")
    print("=" * 68)
    print(f"  {'state':<8}{'samples':>14}{'negative':>12}{'  in 55-85 MHz':>16}")
    for s in states_of_interest:
        print(f"  {s:<8}{tot[s]:>14,}{neg[s]:>12,}{neg_band[s]:>16,}")
    print()
    if sum(neg.values()) == 0:
        print("  RESULT: zero wrapped samples in any state used by the B7 solve.")
        print("  The calibration window is CLEAN. No exclusion needed.")
    else:
        print(f"  RESULT: wrap present. files affected: {len(files_hit)}")
        print(f"  distinct channels affected: {len(chans_hit)}")
        if chans_hit:
            cl = sorted(chans_hit)
            print(f"  channel range: {cl[0]:.2f} - {cl[-1]:.2f} MHz")
            inb = [c for c in cl if args.fmin <= c <= args.fmax]
            print(f"  of which inside {args.fmin:.0f}-{args.fmax:.0f} MHz: "
                  f"{len(inb)}")
        print(f"  most negative value seen: {worst}")
    print()
    print(f"  (int32 max = {INT32_MAX:,}; a wrapped sample lands just below "
          f"-{2**31:,}+value)")


if __name__ == "__main__":
    main()
