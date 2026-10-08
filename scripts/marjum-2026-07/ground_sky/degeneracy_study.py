"""What the Marjum geometry can constrain: sky, ground temperature, offset.

Builds the linear sky / ground / offset design matrix
(``eigsep_sim.design_matrix``) on the real Marjum observing geometry: site,
roll mount, phase-C pointing table, and the DEM horizon of each height era.
It then reports, per data slice and frequency:

1. **Fisher**: marginal errors on the ground temperature, the offset and the
   observed-sky mean, and the ground-offset correlation, under several prior
   assumptions (offset free or known; sky free or GSM-informed).
2. **Simulate and recover**: GSM truth at a finer pixelization (and,
   optionally, the empirical beam as truth) recovered with the HFSS beam at
   the fit resolution. Reports the ground-temperature bias in units of its
   error.

No measured spectra are used: this decides which data and priors a real fit
needs before trusting one.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python degeneracy_study.py [--freqs-mhz 60 80 100 150 200] [--out summary.json]

Writes ``summary.json`` beside itself unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import healpy
import numpy as np
import pandas as pd
from astropy.time import Time

import eigsep_base
import eigsep_sim

from eigsep_base.const import MARJUM_PASS
from eigsep_base.rotations import mount_rotation
from eigsep_sim.design_matrix import (
    HealpixBeam,
    HorizonProfile,
    build_design_matrix,
    fisher_summary,
    solve,
)
from eigsep_sim.observer import EarthSurface

HERE = Path(__file__).resolve().parent
CAMPAIGN = Path(os.environ["EIGSEP_CAMPAIGN_ROOT"])
WORKSPACE = CAMPAIGN.parent
POINTING = CAMPAIGN / "curation" / "pointing_table.parquet"
HORIZONS = CAMPAIGN / "curation" / "horizon_profiles_v0003.npz"   # common.HORIZON_PROFILES
EMPIRICAL = CAMPAIGN / "derived/beam/empirical_raster_v0012/dpss"
HFSS = WORKSPACE / "data-analysis/hfss_beam_maps/bowtie_beam.npz"

ERAS = {"~30m": "30m", "~87.5m": "87.5m", "~91m": "91m"}
# tcal v0000 calibrated spans (derived/tcal/v0000/README.md § Coverage).
TCAL_SPANS = [
    ("2026-07-17T04:11:00", "2026-07-17T16:22:00"),
    ("2026-07-18T01:27:00", "2026-07-18T02:56:00"),
]
CHANNEL_HZ = 250e6 / 1024


def load_bins(bin_s, max_spread_deg):
    """Phase-C, quality-ok pointing, averaged in static ``bin_s`` bins."""
    pt = pd.read_parquet(
        POINTING,
        columns=["t_utc_s", "az_deg", "el_deg", "quality", "height_era", "phase"],
    )
    pt = pt[(pt.phase == "C") & (pt.quality == "ok")
            & pt.height_era.isin(list(ERAS))].dropna(subset=["az_deg", "el_deg"])
    pt = pt.assign(bin=(pt.t_utc_s // bin_s).astype(np.int64))
    g = pt.groupby(["bin", "height_era"])
    bins = g.agg(
        t=("t_utc_s", "mean"),
        az=("az_deg", "median"),
        el=("el_deg", "median"),
        az_spread=("az_deg", lambda a: np.ptp(a)),
        el_spread=("el_deg", lambda a: np.ptp(a)),
        n=("t_utc_s", "size"),
    ).reset_index()
    static = (bins.az_spread < max_spread_deg) & (bins.el_spread < max_spread_deg)
    bins = bins[static & (bins.n >= 0.5 * bins.n.max())].reset_index(drop=True)
    # Rows that rarely move: round so repeated parked pointings share one
    # beam evaluation in build_design_matrix.
    bins["az"] = bins.az.round(1)
    bins["el"] = bins.el.round(1)
    return bins


def slices(bins):
    """Named row selections over the binned table."""
    era = bins.height_era
    moving = bins.groupby(["az", "el", "height_era"]).t.transform("size") < 5
    t = Time(bins.t.values, format="unix")
    in_tcal = np.zeros(len(bins), bool)
    for a, b in TCAL_SPANS:
        in_tcal |= (t >= Time(a)) & (t <= Time(b))
    return {
        "parked_30m": (era == "~30m") & ~moving,
        "parked_all_eras": ~moving,
        "all_phase_c": np.ones(len(bins), bool),
        "tcal_v0000_windows": in_tcal,
    }


def gsm_maps(freqs_hz, nside):
    from pygdsm import GlobalSkyModel16

    gsm = GlobalSkyModel16(freq_unit="MHz")
    return np.stack([
        healpy.ud_grade(gsm.generate(f / 1e6), nside) for f in freqs_hz
    ])


def nearest(beam, freqs_mhz):
    return [int(np.argmin(np.abs(beam.freqs_hz - f * 1e6))) for f in freqs_mhz]


def git_commit(path):
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "describe", "--always", "--dirty"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--freqs-mhz", type=float, nargs="+",
                    default=[60, 80, 100, 150, 200])
    ap.add_argument("--bin-s", type=float, default=120.0)
    ap.add_argument("--max-spread-deg", type=float, default=1.0)
    ap.add_argument("--nside-sky", type=int, default=8)
    ap.add_argument("--nside-truth", type=int, default=16)
    ap.add_argument("--nside-int", type=int, default=64)
    ap.add_argument("--sky-prior-frac", type=float, default=0.1,
                    help="GSM-informed sky prior width, fraction of GSM")
    ap.add_argument("--offset-prior-k", type=float, default=1.0,
                    help="'offset known' prior width, K")
    ap.add_argument("--t-ground", type=float, default=290.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "summary.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    bins = load_bins(args.bin_s, args.max_spread_deg)
    sel = slices(bins)
    print(f"{len(bins)} static {args.bin_s:.0f}-s bins; "
          + ", ".join(f"{k} {int(v.sum())}" for k, v in sel.items()))

    lat, lon, hgt = MARJUM_PASS
    times = Time(bins.t.values, format="unix")
    rot_g2t = EarthSurface(lat, lon, hgt).rot_gal2top_stack(times).astype(float)
    with np.load(EMPIRICAL / "diagnostics.npz") as d:
        az_offset, psi = float(d["az_offset_deg"]), float(d["psi_deg"])
    rot_b2e = mount_rotation(bins.az.values + az_offset, bins.el.values, psi)
    era_keys = list(ERAS)
    horizons = [HorizonProfile.from_npz(HORIZONS, ERAS[e]) for e in era_keys]
    hidx = bins.height_era.map(era_keys.index).values

    hfss = HealpixBeam.from_npz(HFSS, drop_last=True)
    hfss = hfss.select(nearest(hfss, args.freqs_mhz))
    emp = HealpixBeam.from_npz(EMPIRICAL / "empirical_beam.npz")
    emp = emp.select(nearest(emp, hfss.freqs_hz / 1e6))
    emp_fitted = emp.meta["empirical_frequency_mask"].astype(bool)
    freqs = hfss.freqs_hz

    kw = dict(horizon_index=hidx, nside_int=args.nside_int)
    print("building design matrices ...")
    dm = build_design_matrix(hfss, horizons, rot_g2t, rot_b2e,
                             args.nside_sky, **kw)
    dm_truth = {
        "hfss": build_design_matrix(hfss, horizons, rot_g2t, rot_b2e,
                                    args.nside_truth, **kw),
        "empirical": build_design_matrix(emp, horizons, rot_g2t, rot_b2e,
                                         args.nside_truth, **kw),
    }
    gsm_fit = gsm_maps(freqs, args.nside_sky)
    gsm_truth = gsm_maps(freqs, args.nside_truth)

    tau = args.bin_s
    results = []
    for name, rows in sel.items():
        rows = np.asarray(rows)
        if rows.sum() < 10:
            continue
        sub = type(dm)(
            A=dm.A[:, rows], freqs_hz=freqs, nside_sky=dm.nside_sky,
            ground_names=dm.ground_names, offset_names=dm.offset_names,
            sky_observed=np.any(dm.A[:, rows, :dm.npix] > 0, axis=(0, 1)),
        )
        for fi, f in enumerate(freqs):
            x_truth = dm_truth["hfss"].pack(gsm_truth, args.t_ground, 0.0)
            y0 = dm_truth["hfss"].predict(x_truth)[fi, rows]
            sigma = y0 / np.sqrt(CHANNEL_HZ * tau)
            free = np.inf
            sky_gsm = args.sky_prior_frac * gsm_fit[fi]
            priors = {
                "sky_free/offset_free": sub.pack(free, free, free)[fi],
                "sky_free/offset_known": sub.pack(free, free, args.offset_prior_k)[fi],
                "sky_gsm/offset_free": sub.pack(sky_gsm, free, free)[fi],
                "sky_gsm/offset_known": sub.pack(sky_gsm, free, args.offset_prior_k)[fi],
            }
            entry = {"slice": name, "freq_mhz": f / 1e6, "n_rows": int(rows.sum()),
                     "n_sky_observed": int(sub.sky_observed.sum()),
                     "median_noise_k": float(np.median(sigma)), "fisher": {}}
            for pname, prior in priors.items():
                fs = fisher_summary(sub, fi, sigma, prior_sigma=prior)
                entry["fisher"][pname] = {
                    "sigma_ground_k": float(fs["sigma"][0]),
                    "sigma_offset_k": float(fs["sigma"][1]),
                    "corr_ground_offset": float(fs["corr"][0, 1]),
                    "sigma_sky_mean_k": fs["sigma_sky_mean"],
                    "n_null": int(fs["null_vectors"].shape[0]),
                }
            # Simulate with finer truth, recover at nside_sky with HFSS,
            # using the GSM-informed sky prior centred on GSM and a known offset.
            prior_mean = sub.pack(gsm_fit[fi], 0.0, 0.0)[fi]
            prior = priors["sky_gsm/offset_known"]
            entry["recovery"] = {}
            for truth in ("hfss", "empirical"):
                if truth == "empirical" and not emp_fitted[fi]:
                    continue
                dmt = dm_truth[truth]
                y = dmt.predict(dmt.pack(gsm_truth, args.t_ground, 0.0))[fi, rows]
                y = y + sigma * rng.standard_normal(rows.sum())
                fit = solve(sub.A[fi], y, sigma, prior_sigma=prior,
                            prior_mean=prior_mean)
                g = sub.ground.start
                err = float(np.sqrt(fit["cov"][g, g]))
                entry["recovery"][f"truth_{truth}"] = {
                    "t_ground_k": float(fit["x"][g]),
                    "bias_k": float(fit["x"][g] - args.t_ground),
                    "sigma_k": err,
                    "offset_k": float(fit["x"][sub.offset.start]),
                    "chi2_per_row": float(np.mean(
                        ((y - sub.A[fi] @ fit["x"]) / sigma) ** 2)),
                }
            results.append(entry)
            fk = entry["fisher"]
            print(f"{name:20s} {f/1e6:6.1f} MHz  rows {entry['n_rows']:4d}  "
                  + "  ".join(f"{p}: σg {v['sigma_ground_k']:.3g}"
                              for p, v in fk.items()))

    summary = {
        "provenance": {
            "generator": "data-analysis/scripts/marjum-2026-07/ground_sky/degeneracy_study.py",
            "args": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items()},
            "commits": {
                "data-analysis": git_commit(HERE),
                "eigsep_sim": git_commit(Path(eigsep_sim.__file__).parent),
                "eigsep_base": git_commit(Path(eigsep_base.__file__).parent),
                "campaign": git_commit(CAMPAIGN),
            },
            "inputs": [str(POINTING), str(HORIZONS), str(HFSS), str(EMPIRICAL)],
            "pointing": {"az_offset_deg": az_offset, "psi_deg": psi,
                         "convention": "mount_rotation(az_table + az_offset, el, psi)"},
            "noise": "radiometer: T_model / sqrt(channel_width * bin_s)",
        },
        "results": results,
    }
    args.out.write_text(json.dumps(summary, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
