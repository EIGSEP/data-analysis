#!/usr/bin/env python3
"""T_rx(nu, t) and g_rx(nu, t) across all of Phase C.

Consumes data-archivist's curation/cal_windows.jsonl (the authoritative
campaign-wide cal-window product) rather than re-scanning, and intersects it
with curation/select_files.py so the campaign masks still apply.

The 07-16 -> 07-18 span covers ~2 diurnal cycles, so the physical
thermometry swings far more than it does inside a single 8-hour window.
That is the lever that can separate a real LNA thermal coefficient from a
calibration artefact.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import h5py
import numpy as np
from eigsep_data import select_files
from eigsep_data.paths import get_campaign_root

import abscal

CURATION = get_campaign_root(required=True) / "curation"
CAL_WINDOWS = CURATION / "cal_windows.jsonl"


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def allowed_files(phase, start, end):
    kept, _dropped, _warnings = select_files.select(phase=phase, start=start, end=end)
    return set(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="C")
    ap.add_argument("--start", default="2026-07-16T00:00:00Z")
    ap.add_argument("--end", default="2026-07-18T04:00:00Z")
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--fmin", type=float, default=55.0)
    ap.add_argument("--fmax", type=float, default=85.0)
    ap.add_argument("--out", default="trx_phaseC.npz")
    args = ap.parse_args()

    keep = allowed_files(args.phase, args.start, args.end)
    print(f"curation allows {len(keep)} files in {args.start}..{args.end}")

    wins = [json.loads(l) for l in open(CAL_WINDOWS)]
    t0, t1 = parse(args.start), parse(args.end)
    wins = [
        w for w in wins
        if w["phase"] == args.phase and "noise" in w["cal_type"]
        and t0 <= parse(w["t_start_utc"]) <= t1
        and w["states"].get("RFNON", 0) >= args.min_samples
        and w["states"].get("RFAMB", 0) >= args.min_samples
    ]
    print(f"cal_windows.jsonl: {len(wins)} phase-{args.phase} noise windows "
          f"with >= {args.min_samples} of each state")

    # Map each window to the files between file_first and file_last that
    # survive curation. Filenames are file CLOSE times, so a window's data
    # can begin in the file *before* file_first; include one file of margin
    # and let the switch-state axis decide what actually belongs.
    all_names = sorted(keep)
    sols, skipped = [], []
    for w in wins:
        lo, hi = w["file_first"], w["file_last"]
        idx = [i for i, n in enumerate(all_names) if lo <= n <= hi]
        if not idx:
            skipped.append((w["t_start_utc"], "all files masked by curation"))
            continue
        i0 = max(0, min(idx) - 1)
        names = all_names[i0:max(idx) + 1]
        dwells = []
        for n in names:
            d = abscal.read_file(abscal.DATA / n)
            if d is not None:
                dwells.append(d)
        if not dwells:
            skipped.append((w["t_start_utc"], "no rfswitch axis"))
            continue
        with h5py.File(abscal.DATA / names[0], "r") as f:
            cal_cfg = json.loads(f["header/obs_config"][()])["calibration"]
        try:
            s = abscal.solve_gain_trx(dwells, cal_cfg)
        except ValueError as e:
            skipped.append((w["t_start_utc"], str(e)[:60]))
            continue
        if s is None or s.n_on < args.min_samples or s.n_amb < args.min_samples:
            skipped.append((w["t_start_utc"], "too few samples after masking"))
            continue
        sols.append(s)

    print(f"solved {len(sols)} cycles; skipped {len(skipped)}")
    for t, why in skipped[:10]:
        print(f"   skip {t}: {why}")
    if not sols:
        sys.exit("nothing solved")

    freqs = sols[0].freqs
    band = (freqs >= args.fmin) & (freqs <= args.fmax)
    # Use the dt-normalised gain: raw counts/K are not comparable across the
    # corr_acc_len doubling at 2026-07-15 15:55 UTC. T_rx is a ratio and is
    # unaffected either way.
    g = np.array([s.gain_per_s for s in sols])
    dts = np.array([s.dt for s in sols])
    trx = np.array([s.t_rx for s in sols])
    tmid = np.array([0.5 * (s.t0 + s.t1) for s in sols])
    t_ns = np.array([s.t_ns for s in sols])
    t_amb = np.array([s.t_amb for s in sols])

    gb = np.nanmedian(g[:, band], axis=1)
    tb = np.nanmedian(trx[:, band], axis=1)

    span_h = (tmid[-1] - tmid[0]) / 3600.0
    print()
    print("=" * 72)
    print(f"PHASE C  {iso(tmid[0])} -> {iso(tmid[-1])}   ({span_h:.1f} h)")
    print(f"band {args.fmin:.0f}-{args.fmax:.0f} MHz, {len(sols)} cal cycles")
    print("=" * 72)
    u = np.unique(np.round(dts, 4))
    print(f"  integration dt   : {', '.join(f'{x:.4f}s' for x in u)}"
          f"{'   <== spans the corr_acc_len doubling' if len(u) > 1 else ''}")
    print(f"  g_rx  median {np.nanmedian(gb):8.2f} counts/s/K  "
          f"rms {100*np.nanstd(gb)/np.nanmedian(gb):5.2f} %   "
          f"ptp {100*np.ptp(gb)/np.nanmedian(gb):5.2f} %")
    print(f"  T_rx  median {np.nanmedian(tb):8.2f} K           "
          f"rms {np.nanstd(tb):5.2f} K   ptp {np.ptp(tb):5.2f} K")
    print(f"  T_ns  {t_ns.min():6.2f} - {t_ns.max():6.2f} K  (ptp {np.ptp(t_ns):.2f})")
    print(f"  T_amb {t_amb.min():6.2f} - {t_amb.max():6.2f} K  (ptp {np.ptp(t_amb):.2f})")

    # ---- the thermal-lever question ----
    print()
    print("THERMAL LEVER  (does T_rx track physical temperature?)")
    for label, x in (("T_ns", t_ns), ("T_amb", t_amb)):
        ok = np.isfinite(tb) & np.isfinite(x)
        r = np.corrcoef(x[ok], tb[ok])[0, 1]
        slope, icept = np.polyfit(x[ok], tb[ok], 1)
        resid = tb[ok] - (slope * x[ok] + icept)
        print(f"  T_rx vs {label:5s}: r={r:+.3f}  slope={slope:+7.2f} K/K  "
              f"resid rms={np.std(resid):5.2f} K")
    ok = np.isfinite(tb)
    r_g = np.corrcoef(t_ns[ok], gb[ok])[0, 1]
    print(f"  g_rx vs T_ns : r={r_g:+.3f}  "
          f"slope={np.polyfit(t_ns[ok], gb[ok],1)[0]:+.3f} (counts/K)/K")

    np.savez_compressed(
        args.out, freqs=freqs, sol_times=tmid, gain=g, t_rx=trx,
        t_ns=t_ns, t_amb=t_amb, t_hot=[s.t_hot for s in sols], dts=dts,
        # Raw measured powers per cycle -- the Claim-2 control. P_amb is the
        # receiver on a FIXED passive load; P_on - P_amb is the injected
        # excess. Shipping both makes the exclusion argument checkable
        # without re-running the scan.
        p_amb=np.array([s.p_amb for s in sols]),
        p_on=np.array([s.p_on for s in sols]),
        n_on=np.array([s.n_on for s in sols]),
        n_amb=np.array([s.n_amb for s in sols]),
        t_enr=sols[0].t_enr, band_gain=gb, band_trx=tb,
        fmin=args.fmin, fmax=args.fmax,
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
