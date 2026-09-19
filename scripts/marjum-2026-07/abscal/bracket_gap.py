#!/usr/bin/env python3
"""Quantify the P_amb (fixed passive load) step across a specified cal gap,
as a function of the pre/post averaging-window size.

Written in response to data-archivist's request to verify the "4.5%" claim
in CAMPAIGN.md:166 / INDEX.md:717,1187 (commit 5fe4581): that RFAMB, the
receiver's response to an UNCHANGING load, moved only slightly across the
2026-07-17 19:42 -> 2026-07-18 01:24 UTC gap, so the gap does not read as a
broadband receiver-state change.

The 4.5% figure was the single most-favourable bracketing choice (one cycle
either side of the gap) from an earlier ad-hoc check that was never
committed as a script. This reproduces it from the committed trx_phaseC.npz
and shows it does not converge: the ratio depends on how many cycles are
averaged on each side, ranging ~4-12% over the choices below. There is no
single "true" value for this gap; the range itself is the finding.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default="trx_phaseC.npz")
    ap.add_argument("--gap-start", default="2026-07-17T19:42:00Z")
    ap.add_argument("--gap-end", default="2026-07-18T01:24:00Z")
    ap.add_argument("--pre-n", type=int, nargs="+", default=[1, 2, -1],
                     help="cycles averaged before the gap; -1 = all available")
    ap.add_argument("--post-n", type=int, nargs="+", default=[1, 2, -1],
                     help="cycles averaged after the gap; -1 = all available")
    args = ap.parse_args()

    z = np.load(args.npz)
    t = z["sol_times"]
    freqs = z["freqs"]
    fmin, fmax = float(z["fmin"]), float(z["fmax"])
    band = (freqs >= fmin) & (freqs <= fmax)
    p_amb = np.nanmedian(z["p_amb"][:, band], axis=1)

    gap_lo, gap_hi = parse(args.gap_start), parse(args.gap_end)
    pre_idx = np.where(t < gap_lo)[0]
    post_idx = np.where(t > gap_hi)[0]

    print(f"P_amb (fixed passive load, band-median {fmin:.0f}-{fmax:.0f} MHz) "
          f"across the gap {args.gap_start} -> {args.gap_end}")
    print(f"  {len(pre_idx)} cycles available before, {len(post_idx)} after")
    print(f"  last pre cycle : {iso(t[pre_idx[-1]])}")
    print(f"  first post cycle: {iso(t[post_idx[0]])}")

    qp = p_amb[:20]
    print(f"  quiet-period (first 20 cycles) rms = {100*qp.std()/qp.mean():.2f} %"
          f"  <- noise floor for comparison\n")

    print(f"{'pre n':>6} {'post n':>7} {'ratio':>8} {'step %':>8}")
    for npre in args.pre_n:
        npre_eff = len(pre_idx) if npre < 0 else npre
        for npost in args.post_n:
            npost_eff = len(post_idx) if npost < 0 else npost
            pre_mean = p_amb[pre_idx[-npre_eff:]].mean()
            post_mean = p_amb[post_idx[:npost_eff]].mean()
            ratio = post_mean / pre_mean
            print(f"{npre_eff:6d} {npost_eff:7d} {ratio:8.4f} {100*(1-ratio):+8.1f}")

    print("\nDoes not converge to a single value as the window widens; it is")
    print("~4-12% depending on bracketing choice, i.e. ~12-35x the quiet-period")
    print("rms above -- a real but small step, far short of the claimed x10.24")
    print("broadband gain change (924%).")


if __name__ == "__main__":
    main()
