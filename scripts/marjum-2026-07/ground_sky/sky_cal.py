"""Calibrate raw box-air nights on the sky, then separate receiver and ground.

Builds ``derived/ground_sky/skycal_vNNNN``. Most of phase C has no kelvin
scale. But over a night the Galaxy drifts through the beam, so in a stretch of
fixed pointing the raw power at each frequency follows

    P(t) = G * C_GSM(t) + K + L * (t - t_mid)

where C_GSM is GSM through the beam (the design matrix's sky weights times
GSM; its units are kelvin of GSM), G is the gain in raw units per GSM kelvin,
K the constant part, and L a linear drift. The ground term is constant within a
stretch, since the horizon and the pointing are fixed, so it sits in K.

In GSM kelvin, K / G = T_rx + f_gnd * T_gnd + (anything else constant), where
f_gnd is the beam's ground fraction for that stretch's height and pointing.
Across stretches in one receiver regime, a straight-line fit of K/G against
f_gnd gives T_gnd (slope) and T_rx (intercept): the height lever the
degeneracy study asks for, without a kelvin calibration.

A stretch enters the regression only if GSM through the beam fits it: median
residual below ``--max-resid`` of the power.

Checks that could fail:

- **Repeatability.** Two stretches at the same height and nearly the same
  pointing (07-15 and 07-16 nights, 30 m) should give the same K/G.
- **Gain against tcal.** On 07-17, G divided by tcal's gain (raw per kelvin at
  plane P) is the GSM amplitude a in tcal's scale; the night fit (fit_v0001)
  found a ~ 2. Agreement checks this pipeline, not a's origin.
- **Physical T_gnd.** A ground temperature of 250-330 K, smooth across
  frequency, is what physics allows; wild values mean the model is missing
  something.

Data per stretch: box-air antenna rows, ``flags@v3`` (any bit but 9), with
the box-air self-EMI teeth (integer MHz, memo 006) masked within ±0.15 MHz
on the night it was on. The rows are averaged over ±2 channels at each HFSS beam
frequency and into ``--bin-s`` bins. Rows of files recorded with the switch
daemon off (all ``MISSING``; the 07-15 night) count as antenna rows, as in
``flags@v3``. The Sun is below ``--max-sun-alt`` (it is
behind the ridge long before that, ``sun.py``). Pointing comes from
``pointing@v2.0``, or the stretch median where the table has none (the
antenna is parked).

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python sky_cal.py v0001 --beam hfss
    python sky_cal.py v0001 --beam empirical
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time

import eigsep_data
import eigsep_sim
from eigsep_base.const import MARJUM_PASS
from eigsep_base.rotations import mount_rotation
from eigsep_data import MetadataIndex
from eigsep_sim.design_matrix import HorizonProfile, build_design_matrix
from eigsep_sim.observer import EarthSurface

import sun as sunmod
from common import CHANNEL_MHZ, ERAS, HERE, campaign_root, git_rev, sha256
from fit_ground_sky import gsm_maps, load_beam

# Night stretches of fixed pointing in receiver regime rx-A (mode table, pointing table).
STRETCHES = [
    {"name": "07-15 night, 30 m", "start": "2026-07-15T03:20:00Z", "stop": "2026-07-15T12:00:00Z",
     "era": "~30m", "self_emi": False},
    {"name": "07-16 night, 30 m", "start": "2026-07-16T03:20:00Z", "stop": "2026-07-16T12:00:00Z",
     "era": "~30m", "self_emi": True},
    {"name": "07-17 night, 87.5 m", "start": "2026-07-17T03:30:00Z", "stop": "2026-07-17T12:00:00Z",
     "era": "~87.5m", "self_emi": False},
]


def ts(iso):
    return pd.Timestamp(iso).timestamp()


def bin_stretch(index, st, freqs, bin_s, max_sun_alt, tcal):
    a, b = ts(st["start"]), ts(st["stop"])
    rows = []
    for c in np.arange(a, b, 3600.0):
        sel = index.select(time=(c, min(c + 3600.0, b)))
        if sel.nrows == 0:
            continue
        prods = ["flags@v3", "pointing@v2.0"] + ([f"tcal@{tcal}"] if tcal else [])
        B = sel.load_bundle(antenna="box-air", products=prods)
        f = B.freqs_mhz
        ch = np.rint(f / CHANNEL_MHZ).astype(int)
        D = B.data.astype(float)
        fl = B.flags
        codes = np.where(np.isnan(fl), 0, fl).astype(np.uint16) if fl.dtype.kind == "f" else fl.astype(np.uint16)
        bad = ((codes & ~np.uint16(1 << 9)) != 0) | (D <= 0)
        if st["self_emi"]:
            bad |= (np.abs(f - np.round(f)) < 0.15)[None, :]
        D[bad] = np.nan
        rf = B.meta.rfswitch.to_numpy()
        # Files recorded with the switch daemon off carry no switch state; the
        # switch then rests on the antenna (flags@v3 treats them as sky).
        daemon_off = B.meta.groupby("file").rfswitch.transform(lambda x: (x == "MISSING").all()).to_numpy()
        ok = (rf == "RFANT") | daemon_off
        p = B.pointing
        az = np.mod(p.az_deg.to_numpy() + 180, 360) - 180
        el = p.el_deg.to_numpy()
        good_p = p.quality.isin(["ok", "suspect"]).to_numpy()
        Y = np.array([np.nanmean(D[:, np.abs(ch - int(round(x / CHANNEL_MHZ))) <= 2], axis=1)
                      for x in freqs]).T
        if tcal:
            scale = B.products["tcal"]["scale"]
            g = np.array([np.nanmean(scale[:, np.abs(ch - int(round(x / CHANNEL_MHZ))) <= 2], axis=1)
                          for x in freqs]).T
        else:
            g = np.full_like(Y, np.nan)
        k = ((B.t - a) // bin_s).astype(int)
        for kk in np.unique(k[ok]):
            s = ok & (k == kk)
            if s.sum() < 0.5 * bin_s / 0.6:
                continue
            gp = s & good_p & np.isfinite(az) & np.isfinite(el)
            rows.append({"t": B.t[s].mean(), "y": np.nanmean(Y[s], axis=0),
                         "tcal_scale": np.nanmean(g[s], axis=0),
                         "az": np.nanmedian(az[gp]) if gp.sum() > 5 else np.nan,
                         "el": np.nanmedian(el[gp]) if gp.sum() > 5 else np.nan,
                         "n": int(s.sum())})
        print(" ", st["name"], pd.Timestamp(c, unit="s"), len(rows), flush=True)
    df = pd.DataFrame(rows)
    alt, _ = sunmod.sun_altaz(df.t.values)
    df = df[alt < max_sun_alt].reset_index(drop=True)
    df["pointing_filled"] = ~np.isfinite(df.az) | ~np.isfinite(df.el)
    df["az"] = df.az.fillna(df.az.median())
    df["el"] = df.el.fillna(df.el.median())
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--beam", choices=["hfss", "empirical"], required=True)
    ap.add_argument("--bin-s", type=float, default=600.0)
    ap.add_argument("--max-sun-alt", type=float, default=-10.0)
    ap.add_argument("--tcal", default="v0003")
    ap.add_argument("--max-resid", type=float, default=0.05,
                    help="use a stretch in the regression only if the sky fit's median residual "
                         "is below this fraction of the power")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    campaign = campaign_root()
    out_dir = args.out_dir or campaign / "derived/ground_sky" / f"skycal_{args.version}"
    if (out_dir / f"skycal_{args.beam}.json").exists():
        sys.exit(f"{out_dir}/skycal_{args.beam}.json exists; versions are immutable")
    beam, beam_path = load_beam(args.beam, campaign)
    bf = beam.freqs_hz / 1e6
    use = [i for i, x in enumerate(bf) if 45 < x < 235 and not 87 < x < 109
           and (args.beam == "hfss" or beam.meta["empirical_frequency_mask"][i])]
    beam = beam.select(use)
    freqs = beam.freqs_hz / 1e6
    gsm = gsm_maps(freqs, 8)
    with np.load(campaign / "derived/beam/empirical_raster_v0012/dpss/diagnostics.npz") as d:
        az_off, psi = float(d["az_offset_deg"]), float(d["psi_deg"])
    lat, lon, hgt = MARJUM_PASS
    index = MetadataIndex(campaign / "data")

    results, arrays = [], {}
    for si, st in enumerate(STRETCHES):
        print(st["name"], flush=True)
        df = bin_stretch(index, st, freqs, args.bin_s, args.max_sun_alt,
                         args.tcal if "87.5" in st["era"] else None)
        t = df.t.values
        rg = EarthSurface(lat, lon, hgt).rot_gal2top_stack(Time(t, format="unix")).astype(float)
        rb = mount_rotation(df.az.values + az_off, df.el.values, psi)
        hz = sunmod.true_horizon(HorizonProfile.from_npz(
            campaign / "curation/horizon_profiles_v0002.npz", ERAS[st["era"]]))
        dm = build_design_matrix(beam, [hz], rg, rb, 8, offset_groups=False, nside_int=64)
        colG = np.einsum("ftp,fp->ft", dm.A[:, :, dm.sky], gsm)
        fgnd = dm.A[:, :, dm.ground][:, :, 0]
        Y = np.stack(df.y.values).T  # (nfreq, nbin)
        TS = np.stack(df.tcal_scale.values).T
        tt = (t - t.mean()) / 3600
        per = []
        for fi, x in enumerate(freqs):
            y = Y[fi]
            m = np.isfinite(y)
            if m.sum() < 10:
                per.append(None)
                continue
            A = np.stack([colG[fi, m], np.ones(m.sum()), tt[m]], 1)
            co, *_ = np.linalg.lstsq(A, y[m], rcond=None)
            r = y[m] - A @ co
            s2 = r @ r / (m.sum() - 3)
            cov = np.linalg.inv(A.T @ A) * s2
            G, K = co[0], co[1]
            kg = K / G
            # K/G error by propagation
            J = np.array([-K / G**2, 1 / G, 0.0])
            ekg = float(np.sqrt(J @ cov @ J))
            rec = {"freq_mhz": float(x), "n_bins": int(m.sum()), "G": float(G),
                   "sigma_G": float(np.sqrt(cov[0, 0])), "K": float(K), "L_per_h": float(co[2]),
                   "K_over_G_k": float(kg), "sigma_K_over_G_k": ekg,
                   "f_gnd": float(np.mean(fgnd[fi, m])), "gsm_range_k": float(np.ptp(colG[fi, m])),
                   "resid_frac_rms": float(np.sqrt(np.mean(r**2)) / np.mean(y[m]))}
            if np.isfinite(TS[fi]).any():
                gt = 1.0 / np.nanmean(TS[fi])  # raw per tcal kelvin
                rec["a_vs_tcal"] = float(G / gt)
                rec["K_over_G_tcal_k"] = float(kg * G / gt)
            per.append(rec)
        results.append({"stretch": st, "n_bins": int(len(df)),
                        "pointing_az_el": [float(df.az.median()), float(df.el.median())],
                        "pointing_filled_frac": float(df.pointing_filled.mean()),
                        "per_frequency": per})
        arrays[f"t_{si}"], arrays[f"y_{si}"] = t, Y.astype(np.float32)
        arrays[f"colG_{si}"], arrays[f"fgnd_{si}"] = colG.astype(np.float32), fgnd.astype(np.float32)

    # Cross-stretch regression K/G = T_rx + f_gnd T_gnd, per frequency, over
    # stretches the sky actually fits.
    for r in results:
        rr = [p["resid_frac_rms"] for p in r["per_frequency"] if p]
        r["median_resid_frac"] = float(np.median(rr)) if rr else np.nan
        r["used_in_regression"] = bool(r["median_resid_frac"] < args.max_resid)
    reg = []
    for fi, x in enumerate(freqs):
        pts = [(r["per_frequency"][fi]["f_gnd"], r["per_frequency"][fi]["K_over_G_k"],
                r["per_frequency"][fi]["sigma_K_over_G_k"]) for r in results
               if r["per_frequency"][fi] and r["used_in_regression"]]
        if len(pts) < 2:
            reg.append(None)
            continue
        fg, kg, e = np.array(pts).T
        w = 1 / np.maximum(e, 1e-6) ** 2
        A = np.stack([np.ones(len(fg)), fg], 1)
        if np.ptp(fg) == 0:
            reg.append(None)
            continue
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        co = cov @ A.T @ (w * kg)
        chi2 = float(np.sum(w * (kg - A @ co) ** 2))
        reg.append({"freq_mhz": float(x), "T_rx_k": float(co[0]), "T_gnd_k": float(co[1]),
                    "sigma_T_rx_k": float(np.sqrt(cov[0, 0])), "sigma_T_gnd_k": float(np.sqrt(cov[1, 1])),
                    "chi2": chi2, "dof": len(fg) - 2,
                    "f_gnd": fg.tolist(), "K_over_G": kg.tolist(), "sigma": e.tolist()})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"skycal_{args.beam}.json").write_text(json.dumps({
        "provenance": {
            "product": "ground_sky_skycal", "version": args.version, "beam": args.beam,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "data-analysis/scripts/marjum-2026-07/ground_sky/sky_cal.py",
            "argv": sys.argv[1:],
            "code": {"data-analysis": git_rev(HERE), "eigsep_data": git_rev(eigsep_data.__path__[0]),
                     "eigsep_sim": git_rev(Path(eigsep_sim.__file__).parent)},
            "inputs": {"beam_sha256": sha256(beam_path),
                       "horizon_sha256": sha256(campaign / "curation/horizon_profiles_v0002.npz")}},
        "params": {k: v for k, v in vars(args).items() if k not in ("version", "out_dir")},
        "stretches": results, "regression": reg}, indent=1))
    np.savez_compressed(out_dir / f"skycal_{args.beam}.npz", freqs_mhz=freqs, **arrays)
    for r in reg:
        if r:
            print(f"{r['freq_mhz']:6.1f} MHz  T_gnd {r['T_gnd_k']:8.1f} ± {r['sigma_T_gnd_k']:6.1f}  "
                  f"T_rx {r['T_rx_k']:8.1f} ± {r['sigma_T_rx_k']:5.1f}  chi2/dof {r['chi2']:.1f}/{r['dof']}  "
                  f"K/G {np.round(r['K_over_G'], 0)} f {np.round(r['f_gnd'], 3)}")


if __name__ == "__main__":
    main()
