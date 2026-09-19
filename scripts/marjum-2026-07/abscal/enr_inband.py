#!/usr/bin/env python3
"""In-band measurement of the effective ENR, from the RFNOFF/RFAMB/RFNON triplet.

The nameplate T_ENR (917.1 K from obs_config) is the single largest term in
the B7 budget and has never been measured. But where an RFNOFF dwell exists
alongside RFAMB and RFNON, the campaign data can measure it in situ:

  RFAMB  : ambient load, physical temperature T_amb   (thermistor)
  RFNOFF : noise-source path, diode OFF, physical temperature T_ns (thermistor)
  RFNON  : same path, diode ON

RFAMB and RFNOFF are BOTH passive loads at independently known temperatures.
Their difference gives the gain without involving the diode at all:

    g = (P_noff - P_amb) / (T_ns - T_amb)

and then the diode excess follows, referenced to the *same physical path*
(so path loss and mismatch largely cancel, unlike the P_non - P_amb route):

    T_ENR = (P_non - P_noff) / g

VERDICT FOR THIS CAMPAIGN: the route is arithmetically sound but NOT
executable on marjum-2026-07. T_ENR scales linearly with the lever
(T_ns - T_amb), and the measured lever is +4.30 K -- while the two
thermometers involved come from different families with a FIXED offset of
+4.561 +/- 0.133 K (therm2 - T_now, measured over 167 files / ~6 h, during
which each sensor individually swings 2.1-2.4 K). The lever and the offset
are the same size, so they are degenerate: nothing here distinguishes "the
noise-source load really is ~4.5 K hotter than the ambient load" from "these
two sensors disagree by ~4.5 K". sigma(lever) ~ lever, hence
sigma(T_ENR) ~ T_ENR, i.e. unconstrained.

Do NOT read the printed T_ENR as a measurement. It is reported so the
degeneracy is visible, and because the two cheap fixes it implies are the
actionable result:

  1. Cross-calibrate the tempctrl_load thermistor against the rfswitch_therm
     family (bench, one afternoon, no field time).
  2. Create a REAL lever by heating the ambient load. tempctrl was disabled
     all campaign (use_tempctrl: false, and the LOAD channel carried a
     sticky sensor_tripped latch). A commanded 20 K rise would turn a
     degenerate 4 K lever into a 20 K one and make this an in-band ENR
     measurement good to a few percent -- against the term that currently
     dominates the entire B7 budget at 354 K.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import abscal


def load(paths, input_key):
    states, spec, therm_ns, therm_amb = [], [], [], []
    freqs = None
    for p in paths:
        with h5py.File(p, "r") as f:
            st = [str(x) for x in json.loads(f["metadata/rfswitch"][()])]
            d = f["data"][input_key][:, :].astype(np.float64)
            freqs = f["header/freqs"][:]
            tl = abscal._jmeta(f, "tempctrl_load")
            th = abscal._jmeta(f, "rfswitch_therm")
        states += st
        spec.append(d)
        therm_ns.append(abscal._stream_mean(th, "temp_therm2")[0])
        therm_amb.append(abscal._stream_mean(tl, "T_now")[0])
    return (np.array(states), np.concatenate(spec, axis=0), freqs,
            np.nanmean(therm_ns), np.nanmean(therm_amb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--input", default="4")
    ap.add_argument("--fmin", type=float, default=55.0)
    ap.add_argument("--fmax", type=float, default=85.0)
    ap.add_argument("--sig-lever", type=float, default=4.56,
                    help="1-sigma on (T_ns - T_amb), K. Default is the "
                         "measured inter-sensor offset between the "
                         "rfswitch_therm and tempctrl_load families, which "
                         "is the honest number until they are cross-"
                         "calibrated. Pass a smaller value only if you have "
                         "that cross-calibration in hand.")
    args = ap.parse_args()

    paths = [abscal.DATA / f if not Path(f).exists() else Path(f)
             for f in args.files]
    st, d, freqs, t_ns_c, t_amb_c = load(paths, args.input)
    band = (freqs >= args.fmin) & (freqs <= args.fmax)

    need = ("RFAMB", "RFNOFF", "RFNON")
    n = {s: int((st == s).sum()) for s in need}
    print(f"samples: {n}   input {args.input}")
    if min(n.values()) < 5:
        raise SystemExit("need >= 5 samples of each of RFAMB/RFNOFF/RFNON")

    P = {s: d[st == s].mean(axis=0) for s in need}
    V = {s: d[st == s].var(axis=0, ddof=1) for s in need}

    t_ns = t_ns_c + abscal.CELSIUS_TO_KELVIN
    t_amb = t_amb_c + abscal.CELSIUS_TO_KELVIN
    dT = t_ns - t_amb
    print(f"T_ns  = {t_ns:.2f} K   T_amb = {t_amb:.2f} K   "
          f"lever T_ns-T_amb = {dT:+.2f} K")
    if abs(dT) < 1.0:
        print("  WARNING: lever < 1 K. g is essentially unconstrained.")

    # gain from the two passive loads only -- no diode involved
    g = (P["RFNOFF"] - P["RFAMB"]) / dT
    t_enr = (P["RFNON"] - P["RFNOFF"]) / g

    gb = np.nanmedian(g[band])
    eb = np.nanmedian(t_enr[band])

    # --- error propagation ---
    # T_ENR = (P_non - P_noff) * dT / (P_noff - P_amb)
    A = P["RFNON"] - P["RFNOFF"]
    B = P["RFNOFF"] - P["RFAMB"]
    nb = {s: max(n[s], 1) for s in need}
    var = (
        (dT / B) ** 2 * (V["RFNON"] / nb["RFNON"])
        + (dT * (-1 / B - A / B**2)) ** 2 * (V["RFNOFF"] / nb["RFNOFF"])
        + (dT * A / B**2) ** 2 * (V["RFAMB"] / nb["RFAMB"])
    )
    sig_rad = np.sqrt(np.nanmedian(var[band]))
    # the lever enters multiplicatively: T_ENR is linear in dT
    sig_lever = abs(eb) * (args.sig_lever / abs(dT))
    nameplate = abscal.ENR_REF_K * 10 ** 0.5
    tot = np.hypot(sig_rad, sig_lever)

    print()
    print("=" * 68)
    print(f"  g (passive pair)      = {gb:.2f} counts/K")
    print(f"  T_ENR (in-band, RAW)  = {eb:.1f} K")
    print(f"  T_ENR (nameplate)     = {nameplate:.1f} K")
    print()
    print(f"  sigma, radiometric    = {sig_rad:8.1f} K")
    print(f"  sigma, lever          = {sig_lever:8.1f} K   "
          f"(+/-{args.sig_lever} K on a {abs(dT):.2f} K lever)")
    print(f"  sigma, TOTAL          = {tot:8.1f} K   "
          f"({100*tot/abs(eb):.0f} % of the value)")
    print("=" * 68)
    print()
    if tot > 0.5 * abs(eb):
        print("  *** NOT A MEASUREMENT ***")
        print(f"  The uncertainty is {100*tot/abs(eb):.0f} % of the value. The lever")
        print(f"  ({abs(dT):.2f} K) is the same size as the known inter-sensor")
        print(f"  offset ({args.sig_lever} K), so the two are degenerate: this")
        print("  cannot distinguish a real temperature gap between the loads")
        print("  from a disagreement between two thermometer families.")
        lo, hi = eb - 2 * tot, eb + 2 * tot
        print(f"  2-sigma interval: [{max(lo,0):.0f}, {hi:.0f}] K, which "
              f"{'INCLUDES' if lo <= nameplate <= hi else 'EXCLUDES'} "
              f"the nameplate {nameplate:.0f} K.")
        print("  Do not quote the ratio to nameplate as a result.")
    else:
        print(f"  ratio to nameplate = {eb/nameplate:.3f} "
              f"+/- {tot/nameplate:.3f}")
    print()
    print("  Fixes, both cheap and both next-deployment:")
    print("   1. Cross-calibrate tempctrl_load against rfswitch_therm (bench).")
    print("   2. Heat the ambient load to create a REAL lever. tempctrl was")
    print("      disabled all campaign; a commanded 20 K rise turns this into")
    print("      an in-band ENR good to a few percent -- against the term that")
    print("      dominates the whole B7 budget at 354 K.")


if __name__ == "__main__":
    main()
