#!/usr/bin/env python3
"""Produce calibrated T_ant(nu, t) for one Phase-C window.

Each RFNON/RFAMB dwell pair yields one (G, T_rx) solution; the RFANT
integrations bracketed by consecutive solutions are calibrated with the
time-nearest solution. Output is an .npz plus a printed budget.

Usage:
  run_window.py --start 2026-07-17T04:00:00Z --end 2026-07-17T12:00:00Z \
      --phase C --out window_0717.npz
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
from eigsep_data import select_files
from eigsep_data.paths import get_campaign_root

import abscal

CURATION = get_campaign_root(required=True) / "curation"


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def select(phase, start, end):
    """Delegate file selection to the curation tool -- never glob directly."""
    kept, _dropped, _warnings = select_files.select(phase=phase, start=start, end=end)
    return [abscal.DATA / n for n in kept]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="C")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", default="window.npz")
    ap.add_argument("--sig-enr-db", type=float, default=1.0,
                    help="1-sigma on effective ENR (dB). Default 1.0 = "
                         "uncharacterised datasheet diode + pad.")
    ap.add_argument("--sig-t-amb", type=float, default=2.0)
    ap.add_argument("--sig-t-ns", type=float, default=2.0)
    # Default band matches validate.py, budget.py and the memo. Mismatched
    # defaults across these scripts produced three separate band-leak errors
    # (stat:syst, T_rx stability, closure rms) -- keep them aligned.
    ap.add_argument("--fmin", type=float, default=55.0)
    ap.add_argument("--fmax", type=float, default=85.0)
    args = ap.parse_args()

    files = select(args.phase, args.start, args.end)
    print(f"curation selected {len(files)} files")

    dwells = []
    for p in files:
        d = abscal.read_file(p)
        if d is not None:
            dwells.append(d)
    print(f"  with rfswitch axis: {len(dwells)}")
    if not dwells:
        sys.exit("no switched files in range")

    cal_cfg = None
    import h5py
    with h5py.File(files[0], "r") as f:
        cal_cfg = json.loads(f["header/obs_config"][()])["calibration"]
    print(f"  calibration cfg: {cal_cfg}")

    # ---- group dwells into cal cycles: any file holding BOTH states ----
    cycles, cur = [], []
    for d in dwells:
        has_cal = ("RFNON" in d.by_state) or ("RFAMB" in d.by_state)
        if has_cal:
            cur.append(d)
        elif cur:
            cycles.append(cur)
            cur = []
    if cur:
        cycles.append(cur)

    sols = []
    for c in cycles:
        try:
            s = abscal.solve_gain_trx(c, cal_cfg)
        except ValueError as e:
            print(f"  skip cycle {iso(c[0].t0)}: {e}")
            continue
        if s is not None and s.n_on >= 20 and s.n_amb >= 20:
            sols.append(s)
    print(f"  usable cal cycles: {len(sols)} / {len(cycles)}")
    if not sols:
        sys.exit("no usable cal cycles")

    freqs = sols[0].freqs
    band = (freqs >= args.fmin) & (freqs <= args.fmax)

    # ---- calibrate every RFANT integration with the nearest solution ----
    sol_t = np.array([0.5 * (s.t0 + s.t1) for s in sols])
    rows_T, rows_sig, rows_stat, rows_t = [], [], [], []
    for d in dwells:
        if "RFANT" not in d.by_state:
            continue
        p = d.by_state["RFANT"]
        k = int(np.argmin(np.abs(sol_t - 0.5 * (d.t0 + d.t1))))
        s = sols[k]
        p_mean = np.nanmean(p, axis=0)
        T = abscal.calibrate(p_mean, s)
        u = abscal.propagate(
            p_mean, s, n_ant=len(p), var_ant=np.nanvar(p, axis=0, ddof=1),
            sig_enr_db=args.sig_enr_db, sig_t_amb=args.sig_t_amb,
            sig_t_ns=args.sig_t_ns,
        )
        rows_T.append(T)
        rows_sig.append(u["total"])
        rows_stat.append(u["stat"])
        rows_t.append(0.5 * (d.t0 + d.t1))

    T = np.array(rows_T)
    sig = np.array(rows_sig)
    stat = np.array(rows_stat)
    tt = np.array(rows_t)
    order = np.argsort(tt)
    T, sig, stat, tt = T[order], sig[order], stat[order], tt[order]

    # ---- report ----
    g = np.array([s.gain for s in sols])
    trx = np.array([s.t_rx for s in sols])
    print("\n" + "=" * 68)
    print(f"WINDOW {iso(tt[0])} -> {iso(tt[-1])}   phase {args.phase}")
    print(f"  reference plane : RF switch common port (see abscal.py docstring)")
    print(f"  T_ENR           : {sols[0].t_enr:.1f} K "
          f"(= 290 * 10^(({cal_cfg['noise_diode_enr_db']}"
          f"-{cal_cfg['noise_source_atten_db']})/10))")
    print(f"  T_ns  (mean)    : {np.mean([s.t_ns for s in sols]):.2f} K")
    print(f"  T_amb (mean)    : {np.mean([s.t_amb for s in sols]):.2f} K")
    print(f"  T_hot (mean)    : {np.mean([s.t_hot for s in sols]):.2f} K")
    print(f"  cal cycles      : {len(sols)}   RFANT spectra: {len(T)}")
    print(f"\n  band {args.fmin:.0f}-{args.fmax:.0f} MHz:")
    print(f"    G     median  : {np.nanmedian(g[:, band]):.2f} counts/K")
    print(f"    G     drift   : {100*np.nanstd(np.nanmedian(g[:,band],axis=1))/np.nanmedian(g[:,band]):.2f} % rms over window")
    print(f"    T_rx  median  : {np.nanmedian(trx[:, band]):.1f} K")
    print(f"    T_rx  spread  : {np.nanstd(np.nanmedian(trx[:,band],axis=1)):.1f} K rms over window")
    print(f"    T_ant median  : {np.nanmedian(T[:, band]):.1f} K")
    print(f"    sigma stat    : {np.nanmedian(stat[:, band]):.2f} K")
    print(f"    sigma total   : {np.nanmedian(sig[:, band]):.1f} K "
          f"({100*np.nanmedian(sig[:,band])/np.nanmedian(T[:,band]):.1f} %)")

    np.savez_compressed(
        args.out, freqs=freqs, times=tt, T_ant=T, sigma_total=sig,
        sigma_stat=stat, gain=g, t_rx=trx,
        # Raw measured powers per cycle. Shipped so the controls in the memo
        # are auditable without re-running the scan: P_amb is the receiver's
        # response to a FIXED passive load, and is the control that excludes
        # a receiver-gain explanation for the injection drift.
        p_amb=np.array([s.p_amb for s in sols]),
        p_on=np.array([s.p_on for s in sols]),
        n_on=np.array([s.n_on for s in sols]),
        n_amb=np.array([s.n_amb for s in sols]),
        dts=np.array([s.dt for s in sols]),
        sol_times=sol_t, t_hot=[s.t_hot for s in sols],
        t_amb=[s.t_amb for s in sols], t_ns=[s.t_ns for s in sols],
        t_enr=sols[0].t_enr, cal_cfg=json.dumps(cal_cfg),
        sig_enr_db=args.sig_enr_db,
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
