#!/usr/bin/env python3
"""Validation + uncertainty budget for a calibrated window.

Three independent checks, none of which is the training residual of the fit
that produced the calibration:

1. CLOSURE. Calibrate the RFAMB samples with the solution derived from that
   same cycle -- must return T_amb. This is only a consistency check (it is
   algebraically forced to hold to machine precision for the mean), so the
   informative version is *cross*-closure: calibrate cycle i's RFAMB with
   cycle j's solution. That measures how much the calibration drifts between
   cycles, in Kelvin, and it is NOT forced.

2. SPECTRAL INDEX. The Galactic foreground is a smooth power law with
   beta ~ -2.5. Fitting T_ant(nu) should recover that. A wrong gain shape
   shows up here immediately; a wrong absolute scale does not.

3. TEMPORAL. T_ant should drift smoothly with LST, and T_rx should not
   track it. Correlation between T_rx(t) and T_ant(t) is a leakage test.
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--fmin", type=float, default=55.0)
    ap.add_argument("--fmax", type=float, default=85.0)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    f = z["freqs"]
    T = z["T_ant"]
    g = z["gain"]
    trx = z["t_rx"]
    t_amb = z["t_amb"]
    t_enr = float(z["t_enr"])
    band = (f >= args.fmin) & (f <= args.fmax)
    fb = f[band]

    print("=" * 68)
    print(f"BAND {args.fmin:.0f}-{args.fmax:.0f} MHz  --  every number below is "
          f"band-dependent; quote it WITH the band")
    print("=" * 68)
    print("1. CROSS-CLOSURE  (cycle i solution applied to cycle j ambient)")
    # T_amb recovered by cycle j's solution using cycle i's gain/trx:
    #   T = P_amb_j / G_i - Trx_i
    # Use the STORED P_amb. It has been shipped in the npz since af3b167
    # precisely so this check runs on a measured quantity.
    #
    # The fallback below reconstructs P_amb = G*(T_amb + T_rx) for products
    # cut before that. It is numerically identical -- which is the point, and
    # the trap: that expression is an algebraic IDENTITY, not an independent
    # path to P_amb. Never treat agreement between the two as corroboration.
    n = len(g)
    if "p_amb" in z.files:
        p_amb = z["p_amb"]
    else:
        p_amb = g * (t_amb[:, None] + trx)
        print("   (legacy product: P_amb reconstructed, not measured)")
    errs = []
    for i in range(n - 1):
        rec = p_amb[i + 1] / g[i] - trx[i]
        errs.append(np.nanmedian(rec[band]) - t_amb[i + 1])
    errs = np.array(errs)
    print(f"   adjacent-cycle closure error: median {np.median(errs):+.2f} K,"
          f"  rms {np.std(errs):.2f} K,  max |.| {np.max(np.abs(errs)):.2f} K")
    print(f"   -> this is the K-level cost of interpolating the calibration")
    print(f"      across one 12-min cadence step.")

    print()
    print("2. SPECTRAL INDEX  (Galactic foreground should be ~ -2.5)")
    Tm = np.nanmedian(T, axis=0)[band]
    ok = np.isfinite(Tm) & (Tm > 0)
    cf = np.polyfit(np.log(fb[ok]), np.log(Tm[ok]), 1)
    beta = cf[0]
    fit = np.exp(np.polyval(cf, np.log(fb[ok])))
    resid = (Tm[ok] - fit) / fit
    print(f"   beta = {beta:+.3f}   (expected ~ -2.5 for synchrotron)")
    print(f"   fractional residual about the power law: {100*np.std(resid):.2f} %")
    print(f"   peak-to-peak residual: {100*(resid.max()-resid.min()):.1f} %")

    print()
    print("3. T_rx / T_ant LEAKAGE")
    trx_b = np.nanmedian(trx[:, band], axis=1)
    # nearest-cycle T_ant per solution time
    Tb = np.nanmedian(T[:, band], axis=1)
    m = min(len(trx_b), len(Tb))
    c = np.corrcoef(trx_b[:m], Tb[:m])[0, 1]
    print(f"   corr(T_rx, T_ant) over window = {c:+.3f}")
    print(f"   T_rx range {trx_b.min():.1f} - {trx_b.max():.1f} K "
          f"(peak-to-peak {np.ptp(trx_b):.1f} K)")
    print(f"   T_ant range {Tb.min():.1f} - {Tb.max():.1f} K "
          f"(peak-to-peak {np.ptp(Tb):.1f} K)")

    print()
    print("=" * 68)
    print("UNCERTAINTY BUDGET  (band median, 1-sigma, Kelvin)")
    Tmed = np.nanmedian(T[:, band])
    R = Tmed / np.nanmedian(t_amb + t_enr)   # rough R for reporting
    print(f"   T_ant  = {Tmed:.1f} K")
    print(f"   {'source':<34}{'sigma [K]':>11}{'  % of T_ant':>13}")
    print("   " + "-" * 58)
    stat = float(np.nanmedian(z["sigma_stat"][:, band]))
    tot = float(np.nanmedian(z["sigma_total"][:, band]))
    syst = np.sqrt(max(tot**2 - stat**2, 0.0))
    rows = [
        ("radiometric (stat, per 8s spectrum)", stat),
        ("systematic (refs), total", syst),
        ("  of which: effective ENR", np.nan),
        ("TOTAL", tot),
    ]
    for k, v in rows:
        if np.isnan(v):
            continue
        print(f"   {k:<34}{v:>11.2f}{100*v/Tmed:>12.2f}%")
    print()
    print(f"   stat:syst ratio = 1 : {syst/stat:.0f}")


if __name__ == "__main__":
    main()
