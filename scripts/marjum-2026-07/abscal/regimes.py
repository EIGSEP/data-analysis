#!/usr/bin/env python3
"""Segment the Phase-C calibration history into receiver regimes, and test
what drives the drift within each.

Why this matters: (G, T_rx) are observed to move in tight anti-correlation.
That pattern is diagnostic. A pure receiver-gain drift CANNOT produce it --
if every power scales by a common factor k, then

    G_est  = k*(P_on - P_amb)/dT        ->  scales by k
    T_rx   = k*P_amb / (k*G) - T_amb    ->  INVARIANT

so gain drift cancels exactly out of T_rx. What does produce anti-correlated
(G up, T_rx down) is a change in the Y-factor itself, i.e. a change in the
*ratio* of the injected noise to the ambient load. With T_ENR held at its
nameplate value, a drifting noise-source output shows up exactly this way:

    T_rx_est = (T_amb + T_rx_true) * dT_assumed/dT_true  -  T_amb

so T_rx_est goes down precisely when the true excess goes up, and G_est goes
up by the same factor. Hence: the ratio G_est/G_ref measures the *implied*
drift of the true noise-source excess relative to nameplate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%m-%d %H:%M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default="trx_phaseC.npz")
    ap.add_argument("--gap-hours", type=float, default=1.0)
    args = ap.parse_args()

    z = np.load(args.npz)
    t = z["sol_times"]
    tb = z["band_trx"]
    gb = z["band_gain"]
    tns = z["t_ns"]
    tamb = z["t_amb"]
    t_enr = float(z["t_enr"])
    freqs = z["freqs"]
    fmin, fmax = float(z["fmin"]), float(z["fmax"])

    # split on observing gaps
    seg, cur = [], [0]
    for i in range(1, len(t)):
        if (t[i] - t[i - 1]) > args.gap_hours * 3600:
            seg.append(cur)
            cur = []
        cur.append(i)
    seg.append(cur)

    print("=" * 76)
    print("PHASE-C RECEIVER REGIMES")
    print("=" * 76)
    print(f"{'segment':>28} {'n':>3} {'T_rx K':>16} {'g_rx cts/K':>17} "
          f"{'T_ns K':>13}")
    for s in seg:
        s = np.array(s)
        print(f"  {iso(t[s[0]])} -> {iso(t[s[-1]])} {len(s):3d} "
              f"{tb[s].mean():7.1f} +/-{tb[s].std():5.1f} "
              f"{gb[s].mean():9.1f} +/-{gb[s].std():5.1f} "
              f"{tns[s].mean():7.2f}")

    print()
    print("WHAT IS ACTUALLY MOVING")
    ok = np.isfinite(tb) & np.isfinite(gb)
    print(f"  corr(T_rx, g_rx) all cycles = {np.corrcoef(tb[ok],gb[ok])[0,1]:+.4f}")
    print("    (near -1, but note T_rx and g_rx are NOT independent: T_rx is")
    print("     defined as P_amb/g - T_amb, so any error in g maps straight")
    print("     into T_rx. The anti-correlation alone proves nothing.)")
    print()
    # Use the STORED per-channel powers, and reduce in the right order.
    #
    # BAND-REDUCTION ORDER MATTERS -- for forming the excess. The excess at
    # channel i is p_on[i] - p_amb[i]. The median is NOT linear, so reducing
    # each state first and subtracting afterwards compares the median channel
    # of p_on against a different channel of p_amb (they differ in 78% of
    # cycles, by a mean of 37 of 123 channels). Difference FIRST, always:
    # that is about WHICH QUANTITY you compute, and it is not a matter of
    # taste.
    #
    # But do NOT over-read the resulting correlation scalar. The two orders
    # give +0.281 and +0.460, and an earlier version of this comment called
    # the latter "an artefact". That was too strong -- scribe went to the
    # per-channel correlation, which is what both scalars approximate:
    #     median +0.359, mean +0.248, 16-84 pct [-0.109, +0.447],
    #     full range [-0.154, +0.518] over 123 channels.
    # The channel-to-channel spread is far wider than the gap between the two
    # estimators. And it is worse than that -- the scalar is dominated by a
    # handful of end cycles. Sliding the segment boundary by four cycles:
    #
    #     n=55  per-channel median +0.578   band scalar +0.452
    #     n=56                     +0.550               +0.422
    #     n=57                     +0.522               +0.392
    #     n=58                     +0.359               +0.281   <- scribe
    #     n=59                     +0.154               +0.136   <- mine
    #
    # Monotonic, and a swing of 0.42 for four cycles out of ~59. The whole
    # disagreement between +0.281 and +0.460 was smaller than the boundary
    # sensitivity of either. r(control, excess) is NOT DETERMINED to better
    # than about +/-0.2 by this dataset; quote it as "weak-to-moderate,
    # poorly constrained" or not at all.
    #
    # This is exactly why (Y-1) below is the primary statistic: it is
    # gain-invariant by construction and so does not depend on a correlation
    # the data cannot pin down.
    #
    # (Band MEAN is unusable throughout: a few RFI channels swamp it.)
    band = (freqs >= fmin) & (freqs <= fmax)
    PA, PO = z["p_amb"][:, band], z["p_on"][:, band]
    p_amb = np.nanmedian(PA, axis=1)
    p_diff = np.nanmedian(PO - PA, axis=1)          # difference first
    y_minus_1 = np.nanmedian((PO - PA) / PA, axis=1)

    print("  PRIMARY STATISTIC -- (Y-1) = (P_on - P_amb)/P_amb, per channel.")
    print("  Invariant under a common scaling of every power, so a receiver-")
    print("  gain explanation cannot survive in it BY CONSTRUCTION. This")
    print("  retires the common-mode objection without argument.")
    for sg in seg:
        sg = np.array(sg)
        if len(sg) < 5:
            continue
        ys = y_minus_1[sg]
        print(f"    {iso(t[sg[0]])} -> {iso(t[sg[-1]])}  ({len(sg)} cycles)  "
              f"rms {100*ys.std()/ys.mean():5.2f} %  "
              f"ptp {100*np.ptp(ys)/ys.mean():6.2f} %  "
              f"r(.,t) {np.corrcoef(t[sg], ys)[0,1]:+.3f}")

    print()
    print("  Supporting decomposition, PER SEGMENT:")
    for sg in seg:
        sg = np.array(sg)
        if len(sg) < 5:
            continue
        print(f"    {iso(t[sg[0]])} -> {iso(t[sg[-1]])}  ({len(sg)} cycles)")
        for name, x in (("P_amb  (fixed passive load)", p_amb),
                        ("P_on-P_amb (injected excess)", p_diff)):
            xs = x[sg]
            print(f"      {name:<30} rms {100*xs.std()/xs.mean():5.2f} %  "
                  f"ptp {100*np.ptp(xs)/xs.mean():6.2f} %  "
                  f"corr(.,t) {np.corrcoef(t[sg], xs)[0,1]:+.3f}")
        rch = np.array([np.corrcoef(PA[sg, i], (PO - PA)[sg, i])[0, 1]
                        for i in range(PA.shape[1])])
        rch = rch[np.isfinite(rch)]
        print(f"      per-channel r(control,excess): median "
              f"{np.median(rch):+.3f}  16-84 pct "
              f"[{np.percentile(rch,16):+.3f}, {np.percentile(rch,84):+.3f}]"
              f"  <- the scalar below is ONE estimator of this")
        print(f"      r(control, excess) = "
              f"{np.corrcoef(p_amb[sg], p_diff[sg])[0,1]:+.3f}   "
              f"excess is {(p_diff[sg].std()/p_diff[sg].mean())/(p_amb[sg].std()/p_amb[sg].mean()):.1f}x "
              f"less steady than the control")

    print()
    print("  Claim: the receiver is several times STEADIER than the injection.")
    print("  Not 'the receiver is stable' -- the control has its own mild")
    print("  trend, and the weaker phrasing is the correct one.")

    print()
    print("IMPLIED NOISE-SOURCE EXCESS  (relative to the nameplate 917.1 K)")
    ref = np.median(gb[:20])          # first segment as reference
    imp = t_enr * gb / ref
    for s in seg:
        s = np.array(s)
        print(f"  {iso(t[s[0]])} -> {iso(t[s[-1]])}: "
              f"T_ENR_implied {imp[s].mean():7.1f} K "
              f"({100*(imp[s].mean()-t_enr)/t_enr:+6.1f} % vs nameplate)")

    print()
    print("WITHIN-SEGMENT THERMAL TEST")
    for s in seg:
        s = np.array(s)
        if len(s) < 5:
            print(f"  {iso(t[s[0]])}: only {len(s)} cycles, skipped")
            continue
        r = np.corrcoef(tns[s], tb[s])[0, 1]
        sl = np.polyfit(tns[s], tb[s], 1)[0]
        # is time a better predictor than temperature?
        rt = np.corrcoef(t[s], tb[s])[0, 1]
        print(f"  {iso(t[s[0]])} -> {iso(t[s[-1]])}: "
              f"corr(T_rx,T_ns)={r:+.3f} slope={sl:+7.1f} K/K | "
              f"corr(T_rx,time)={rt:+.3f}")
    print()
    print("  Interpretation: where |corr(T_rx,time)| >> |corr(T_rx,T_ns)|, the")
    print("  drift is NOT thermally driven -- it is monotonic in time and the")
    print("  thermometry is only incidentally co-trending.")


if __name__ == "__main__":
    main()
