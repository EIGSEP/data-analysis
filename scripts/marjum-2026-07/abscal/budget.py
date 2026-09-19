#!/usr/bin/env python3
"""Uncertainty budget + cost of the missing lab measurements.

Turns "we don't have the lab standards" into Kelvin.
"""
from __future__ import annotations

import argparse

import numpy as np

import abscal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default="window_0717.npz")
    ap.add_argument("--fmin", type=float, default=55.0)
    ap.add_argument("--fmax", type=float, default=85.0)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    f = z["freqs"]
    band = (f >= args.fmin) & (f <= args.fmax)
    T = np.nanmedian(z["T_ant"][:, band])
    stat = float(np.nanmedian(z["sigma_stat"][:, band]))
    t_enr = float(z["t_enr"])
    t_amb = float(np.mean(z["t_amb"]))
    t_ns = float(np.mean(z["t_ns"]))
    t_hot = t_ns + t_enr
    dT = t_hot - t_amb
    R = (T - t_amb) / dT          # T_ant = dT*R + T_amb

    print("=" * 72)
    print(f"UNCERTAINTY BUDGET  --  {args.fmin:.0f}-{args.fmax:.0f} MHz, "
          f"Phase C, 2026-07-17 04:00-12:00 UTC")
    print("reference plane: RF switch common port. NOT sky brightness.")
    print("=" * 72)
    print(f"  T_ant(sw) = {T:8.1f} K      R = (T_ant-T_amb)/(T_hot-T_amb) "
          f"= {R:.3f}")
    print(f"  T_amb     = {t_amb:8.2f} K    T_ns = {t_ns:.2f} K   "
          f"T_ENR = {t_enr:.1f} K   T_hot = {t_hot:.1f} K")
    print()

    # ---- term-by-term, using the exact partials from abscal.propagate ----
    terms = []
    terms.append((
        "radiometric noise (per 8 s spectrum)", stat,
        "measured from the data; averages down as 1/sqrt(N)"))
    terms.append((
        "cal-cycle interpolation (12 min)", 1.96,
        "measured: adjacent-cycle closure rms, 55-85 MHz (validate.py check 1)"))

    # T_amb: partial is (1 - R)
    sig_tamb_stat = 0.28 / np.sqrt(1000)      # ADC noise, averaged per cycle
    terms.append((
        "T_amb thermistor, statistical", abs(1 - R) * sig_tamb_stat,
        "ADC rms 0.28 K / sqrt(N_samples per cycle)"))
    terms.append((
        "T_amb thermistor, systematic (UNVERIFIED)", abs(1 - R) * 2.0,
        "assumed +/-2 K; NO lab characterisation of load vs sensor offset"))
    terms.append((
        "T_ns thermistor, systematic (UNVERIFIED)", abs(R) * 2.0,
        "assumed +/-2 K; same"))

    print(f"  {'term':<44}{'sigma [K]':>11}{'% T_ant':>10}")
    print("  " + "-" * 66)
    for name, s, _ in terms:
        print(f"  {name:<44}{s:>11.2f}{100*s/T:>9.2f}%")

    print()
    print("  " + "-" * 66)
    print("  EFFECTIVE ENR -- the dominant term, and the one the missing")
    print("  lab measurement would fix. T_ENR = 290 K * 10^((ENR-atten)/10),")
    print("  taken from obs_config: ENR=35.0 dB, atten=30.0 dB. Neither the")
    print("  diode nor the pad was characterised for this campaign; the")
    print("  numbers are nameplate values carried in a config file.")
    print()
    print(f"  {'sigma(ENR_eff)':<20}{'sigma(T_ENR)':>14}{'sigma(T_ant)':>14}"
          f"{'% T_ant':>10}   provenance")
    print("  " + "-" * 86)
    cases = [
        (0.10, "lab-measured diode + pad, VNA-corrected (achievable)"),
        (0.25, "lab-measured diode, pad from datasheet"),
        (0.50, "manufacturer calibration certificate, uncorrected"),
        (1.00, "nameplate only -- WHAT WE HAVE NOW"),
        (2.00, "nameplate, diode aged / pad unverified in situ"),
    ]
    for s_db, prov in cases:
        s_enr = t_enr * (np.log(10) / 10) * s_db
        s_T = abs(R) * s_enr
        mark = "  <== current" if s_db == 1.00 else ""
        print(f"  {s_db:>6.2f} dB{'':<11}{s_enr:>11.1f} K{s_T:>12.1f} K"
              f"{100*s_T/T:>9.1f}%   {prov}{mark}")

    print()
    print("=" * 72)
    print("WHAT THE MISSING LAB DATA IS WORTH")
    print("=" * 72)
    s_now = abs(R) * t_enr * (np.log(10) / 10) * 1.00
    s_lab = abs(R) * t_enr * (np.log(10) / 10) * 0.10
    print(f"  now (nameplate, +/-1 dB)        : {s_now:7.1f} K  "
          f"({100*s_now/T:.1f} % of T_ant)")
    print(f"  with lab-measured ENR (+/-0.1dB): {s_lab:7.1f} K  "
          f"({100*s_lab/T:.1f} % of T_ant)")
    print(f"  reduction                       : {s_now - s_lab:7.1f} K  "
          f"({s_now/s_lab:.0f}x)")
    print()
    print(f"  Program standard (EIGSEP paper) : 1 part in 1e4 = "
          f"{T*1e-4:.3f} K on {T:.0f} K")
    print(f"  We are short of it by           : {s_now/(T*1e-4):.0f}x "
          f"(now)  /  {s_lab/(T*1e-4):.0f}x (with lab ENR)")
    print()
    print("  NOTE: an ENR error is very nearly a pure SCALE error on T_ant")
    print("  (partial = R, smooth in frequency). It does not manufacture")
    print("  spectral structure, so it does not by itself threaten a")
    print("  21-cm-shaped residual -- it limits absolute scale, and")
    print("  therefore any claim tied to absolute amplitude.")
    print()
    print("  The terms that DO manufacture spectral structure are the ones")
    print("  we cannot bound at all from this dataset:")
    print("    - Gamma_ant, Gamma_rx  : no VNA S11 files in the repo")
    print("    - 5 noise-wave params  : no four-standard lab measurement")
    print("    - balun / efficiency   : HFSS only, no lab characterisation")
    print("  Evidence they matter here: the calibrated spectrum has")
    print("  beta = +0.42 over 55-85 MHz. Sky synchrotron is beta ~ -2.5.")
    print("  The entire difference is uncorrected instrument response.")


if __name__ == "__main__":
    main()
