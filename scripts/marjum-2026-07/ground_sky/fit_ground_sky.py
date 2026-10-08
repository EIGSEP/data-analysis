"""Fit ground temperature and sky to binned calibrated spectra, per frequency.

Reads ``derived/ground_sky/binned_vNNNN`` and writes
``derived/ground_sky/fit_vNNNN``. At each beam frequency the model is

    T(t) = A_sky(t) . (a * GSM + delta) + A_gnd(t) T_gnd + T_off[regime(t)]
           [+ A_sun(t) S_sun]

built by ``eigsep_sim.design_matrix`` on the real geometry (site, roll mount
with the beam fit's az offset, per-era DEM horizon). ``a`` is free and absorbs
the front-end loss (temperatures are at plane P) and any GSM scale error;
``delta`` has a Gaussian prior of ``--sky-prior-frac`` x GSM; ``T_gnd`` and one
offset per receiver regime are free. Noise is the binned product's in-bin
scatter.

The Sun (``--sun``, module ``sun``) is either cut or modelled:

- ``model`` (default): one more column, the antenna temperature per SFU of
  solar flux through the beam and the terrain (``sun.sun_column``; knife-edge
  diffraction at the ridge along the Sun's bearing, traced in the DEM). Its
  amplitude ``S_sun`` (SFU) is free. All bins are fitted.
- ``cut-terrain``: fit only bins where the Sun's knife-edge visibility is at
  most ``--max-sun-visibility`` at every fitted frequency. Bins with the Sun
  up but behind the ridge are fitted.
- ``cut-flat``: fit only bins with the Sun below ``--max-sun-alt-deg`` (the
  rule of the first trials, which ignores the terrain).

A cut only decides which bins are fitted: the model and residual are
evaluated on every bin in the window, so the residual on cut bins is an
out-of-sample prediction (``fitted`` in ``fit.npz`` marks the fitted bins).

Horizon bearings are rotated from the DEM's UTM grid to true bearings
(``sun.grid_to_true_deg``, +1.52 deg) unless ``--no-grid-rotation``.

Frequencies whose in-bin scatter exceeds ``--max-noise-ratio`` times the
radiometer expectation (the binned manifest's ``noise_over_radiometer_median``)
are skipped; that removes FM and the strongest RFI channels.

Validation: bins are split into alternating ``--block-s`` blocks in time; the
model fitted to the even blocks predicts the odd ones. ``chi2_test`` close to
``chi2_train`` means the fit predicts data it did not see.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python fit_ground_sky.py v0001 --binned v0001 [--beam hfss|empirical]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import healpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    solve,
)
from eigsep_sim.observer import EarthSurface

import sun as sunmod
from common import ERAS, HERE, HORIZON_PROFILES, campaign_root, git_rev, sha256, workspace_root

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED = "#0b0b0b", "#52514e"


def load_beam(name, campaign):
    if name == "hfss":
        path = workspace_root() / "data-analysis/hfss_beam_maps/bowtie_beam.npz"
        return HealpixBeam.from_npz(path, drop_last=True), path
    path = campaign / "derived/beam/empirical_raster_v0012/dpss/empirical_beam.npz"
    return HealpixBeam.from_npz(path), path


def gsm_maps(freqs_mhz, nside):
    from pygdsm import GlobalSkyModel16

    gsm = GlobalSkyModel16(freq_unit="MHz")
    return np.stack([healpy.ud_grade(gsm.generate(f), nside) for f in freqs_mhz])


def sun_altitude(t_unix):
    from astropy.coordinates import AltAz, EarthLocation, get_sun
    import astropy.units as u

    lat, lon, hgt = MARJUM_PASS
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=hgt * u.m)
    t = Time(t_unix, format="unix")
    return get_sun(t).transform_to(AltAz(obstime=t, location=loc)).alt.deg


def fit_one(dm, fi, y, sigma, prior, rows):
    """Solve on ``rows``; return the solution and chi2 per row on all rows."""
    sub = dm.select_rows(rows)
    sol = solve(sub.A[fi], y[rows], sigma[rows], prior_sigma=prior)
    resid = (y - dm.A[fi] @ sol["x"]) / sigma
    return sol, resid


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--binned", required=True,
                    help="binned product version, or a path to a trial directory")
    ap.add_argument("--beam", choices=["hfss", "empirical"], default="hfss")
    ap.add_argument("--nside-sky", type=int, default=8)
    ap.add_argument("--nside-int", type=int, default=64)
    ap.add_argument("--sky-prior-frac", type=float, default=0.1)
    ap.add_argument("--max-noise-ratio", type=float, default=10.0)
    ap.add_argument("--block-s", type=float, default=1800.0)
    ap.add_argument("--sun", choices=["model", "cut-terrain", "cut-flat"],
                    default="model")
    ap.add_argument("--sun-visibility", choices=["knife", "hard"], default="knife",
                    help="terrain visibility of the Sun (model and cut-terrain)")
    ap.add_argument("--max-sun-visibility", type=float, default=0.02,
                    help="cut-terrain: drop bins above this knife-edge visibility")
    ap.add_argument("--max-sun-alt-deg", type=float, default=-10.0,
                    help="cut-flat: drop bins with the Sun above this altitude")
    ap.add_argument("--no-grid-rotation", action="store_true",
                    help="use horizon bearings as true bearings (the v0001 behaviour)")
    ap.add_argument("--utc", nargs=2, action="append", metavar=("START", "STOP"),
                    help="fit only bins inside these UTC spans (repeatable)")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    campaign = campaign_root()
    out_dir = args.out_dir or campaign / "derived/ground_sky" / f"fit_{args.version}"
    if out_dir.exists():
        sys.exit(f"{out_dir} exists; versions are immutable")
    bdir = (Path(args.binned) if Path(args.binned).is_dir()
            else campaign / "derived/ground_sky" / f"binned_{args.binned}")
    z = dict(np.load(bdir / "binned.npz"))
    bman = json.loads((bdir / "manifest.json").read_text())
    n_all = len(z["t"])
    keep = np.ones(n_all, bool)
    if args.utc:
        keep[:] = False
        for a, b in args.utc:
            keep |= (z["t"] >= pd.Timestamp(a).timestamp()) & (
                z["t"] <= pd.Timestamp(b).timestamp())
    n_window = int(keep.sum())
    z = {k: (v[keep] if v.ndim and len(v) == n_all else v) for k, v in z.items()}

    # Frequencies: radiometric ones that the beam covers.
    ratio = np.array([bman["noise_over_radiometer_median"][f"{f:.3f}"]
                      for f in z["freqs_mhz"]])
    beam, beam_path = load_beam(args.beam, campaign)
    bidx = [int(np.argmin(np.abs(beam.freqs_hz / 1e6 - f))) for f in z["freqs_mhz"]]
    on_grid = np.abs(beam.freqs_hz[bidx] / 1e6 - z["freqs_mhz"]) < 1e-6
    fitted = on_grid & (ratio <= args.max_noise_ratio)
    if "empirical_frequency_mask" in beam.meta:
        fitted &= beam.meta["empirical_frequency_mask"][bidx].astype(bool)
    fsel = np.flatnonzero(fitted)
    beam = beam.select([bidx[i] for i in fsel])
    freqs = z["freqs_mhz"][fsel]
    Y, S = z["T"][:, fsel].T, z["sigma"][:, fsel].T  # (nfreq, nbin)

    # Geometry.
    with np.load(campaign / "derived/beam/empirical_raster_v0012/dpss/diagnostics.npz") as d:
        az_offset, psi = float(d["az_offset_deg"]), float(d["psi_deg"])
    lat, lon, hgt = MARJUM_PASS
    times = Time(z["t"], format="unix")
    rot_g2t = EarthSurface(lat, lon, hgt).rot_gal2top_stack(times).astype(float)
    rot_b2e = mount_rotation(np.round(z["az"], 1) + az_offset,
                             np.round(z["el"], 1), psi)
    era_keys = np.array([ERAS[e] for e in z["era"]], dtype=object)
    geom = sunmod.sun_geometry(z["t"], era_keys)
    A_sun = sunmod.sun_column(beam, rot_b2e, geom, args.sun_visibility)
    A_sun_flat = sunmod.sun_column(beam, rot_b2e, geom, "flat")
    if args.sun == "cut-terrain":
        vis = sunmod.visibility(geom, beam.freqs_hz, args.sun_visibility)
        fitmask = ~(vis > args.max_sun_visibility).any(axis=0)
    elif args.sun == "cut-flat":
        fitmask = geom["alt"] <= args.max_sun_alt_deg
    else:
        fitmask = np.ones(len(z["t"]), bool)
    eras = sorted(set(z["era"]))
    hpath = campaign / f"{HORIZON_PROFILES}.npz"
    horizons = [HorizonProfile.from_npz(hpath, ERAS[e]) for e in eras]
    if not args.no_grid_rotation:
        horizons = [sunmod.true_horizon(h) for h in horizons]
    hidx = np.array([eras.index(e) for e in z["era"]])
    regimes = sorted(set(z["regime"]))
    ridx = np.array([regimes.index(r) for r in z["regime"]])

    print(f"{len(z['t'])} bins ({fitmask.sum()} fitted), {len(freqs)} of {len(z['freqs_mhz'])} "
          f"frequencies, beam {args.beam}; building design matrix ...", flush=True)
    dm = build_design_matrix(
        beam, horizons, rot_g2t, rot_b2e, args.nside_sky,
        horizon_index=hidx, offset_groups=ridx,
        offset_names=[f"offset[{r}]" for r in regimes],
        nside_int=args.nside_int,
    )
    gsm = gsm_maps(freqs, args.nside_sky)
    dm = dm.with_sky_templates(gsm, names=["gsm_amplitude"])
    if args.sun == "model":
        dm = dataclasses.replace(
            dm, A=np.concatenate([dm.A, A_sun[:, :, None]], axis=2),
            template_names=dm.template_names + ["sun_flux_sfu"])

    block = ((z["t"] - z["t"].min()) // args.block_s).astype(int)
    train = (block % 2 == 0) & fitmask
    g, a = dm.ground.start, dm.template.start
    off = dm.offset

    res = {k: [] for k in (
        "t_ground_k", "sigma_t_ground_k", "gsm_amplitude", "sigma_gsm_amplitude",
        "offsets_k", "sigma_offsets_k", "pedestal_k", "sigma_pedestal_k",
        "ground_constrained", "chi2_full", "chi2_train", "chi2_test", "n_bins",
        "sun_flux_sfu", "sigma_sun_flux_sfu")}
    model_parts = {k: np.full(Y.shape, np.nan)
                   for k in ("sky", "ground", "offset", "sun")}
    resid_full = np.full(Y.shape, np.nan)
    for fi, f in enumerate(freqs):
        y, s = Y[fi], S[fi]
        ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
        prior = dm.pack(args.sky_prior_frac * gsm[fi], np.inf, np.inf,
                        template=np.inf)[fi]
        if not np.isfinite(A_sun[fi][ok]).all():
            sys.exit(f"non-finite Sun column at {f} MHz")
        sol, r = fit_one(dm, fi, np.where(ok, y, 0), np.where(ok, s, 1), prior,
                         ok & fitmask)
        _, r_tr = fit_one(dm, fi, np.where(ok, y, 0), np.where(ok, s, 1),
                          prior, ok & train)
        x, cov, con = sol["x"], sol["cov"], sol["constrained"]
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
        sig[~con] = np.inf

        def value(i):
            return x[i] if con[i] else np.nan

        # Pedestal per regime: the mean ground + offset contribution, which
        # the data fix even when ground and offset trade off against each other.
        ped, sped = [], []
        for k in range(len(regimes)):
            w = np.zeros(len(x))
            rows = ok & fitmask & (ridx == k)
            w[g] = dm.A[fi, rows, g].mean() if rows.any() else 0.0
            w[off.start + k] = 1.0
            ped.append(w @ x)
            sped.append(np.sqrt(max(w @ cov @ w, 0.0)))
        res["t_ground_k"].append(value(g))
        res["sigma_t_ground_k"].append(sig[g])
        res["ground_constrained"].append(bool(con[g]))
        res["gsm_amplitude"].append(value(a))
        res["sigma_gsm_amplitude"].append(sig[a])
        res["offsets_k"].append([value(i) for i in range(off.start, off.stop)])
        res["sigma_offsets_k"].append(sig[off])
        res["pedestal_k"].append(ped)
        res["sigma_pedestal_k"].append(sped)
        res["chi2_full"].append(np.mean(r[ok & fitmask] ** 2))
        res["chi2_train"].append(np.mean(r_tr[ok & train] ** 2))
        res["chi2_test"].append(np.mean(r_tr[ok & fitmask & ~train] ** 2))
        res["n_bins"].append(int((ok & fitmask).sum()))
        if args.sun == "model":
            j = dm.template.start + 1
            res["sun_flux_sfu"].append(value(j))
            res["sigma_sun_flux_sfu"].append(sig[j])
        else:
            res["sun_flux_sfu"].append(np.nan)
            res["sigma_sun_flux_sfu"].append(np.nan)
        Af = dm.A[fi]
        sky_cols = np.r_[dm.sky, a]
        model_parts["sky"][fi] = Af[:, sky_cols] @ x[sky_cols]
        model_parts["ground"][fi] = Af[:, g] * x[g]
        model_parts["offset"][fi] = Af[:, off] @ x[off]
        model_parts["sun"][fi] = (Af[:, dm.template.start + 1]
                                  * x[dm.template.start + 1]
                                  if args.sun == "model" else 0.0)
        resid_full[fi, ok] = (r * s)[ok]
        tg = (f"T_gnd {x[g]:8.1f} ± {sig[g]:6.1f} K" if con[g]
              else "T_gnd unconstrained   ")
        sn = (f"  S_sun {res['sun_flux_sfu'][-1]:7.2f} ± "
              f"{res['sigma_sun_flux_sfu'][-1]:.2f} SFU" if args.sun == "model" else "")
        print(f"{f:7.2f} MHz  {tg}  a {x[a]:.3f} ± {sig[a]:.3f}{sn}  pedestal "
              + " ".join(f"{p:7.1f} ± {e:.1f}" for p, e in zip(ped, sped))
              + f"  chi2 {res['chi2_full'][-1]:7.1f}  train/test "
              f"{res['chi2_train'][-1]:.1f}/{res['chi2_test'][-1]:.1f}", flush=True)
    res = {k: np.asarray(v) for k, v in res.items()}

    out_dir.mkdir(parents=True)
    np.savez_compressed(
        out_dir / "fit.npz", freqs_mhz=freqs, regimes=np.array(regimes),
        residual_k=resid_full, t=z["t"], era=z["era"], regime=z["regime"],
        az=z["az"], el=z["el"], train=train, fitted=fitmask, data_k=Y, sigma_k=S, sun_column_k_per_sfu=A_sun,
        sun_column_flat_k_per_sfu=A_sun_flat,
        sun_alt_deg=geom["alt"], sun_theta_deg=geom["theta_deg"],
        sun_edge_m=geom["edge_m"],
        **{f"model_{k}_k": v for k, v in model_parts.items()}, **res)
    plot(out_dir, freqs, res, regimes)

    manifest = {
        "provenance": {
            "product": "ground_sky_fit",
            "version": args.version,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "data-analysis/scripts/marjum-2026-07/ground_sky/fit_ground_sky.py",
            "argv": sys.argv[1:],
            "code": {
                "data-analysis": git_rev(HERE),
                "eigsep_sim": git_rev(Path(eigsep_sim.__file__).parent),
                "eigsep_base": git_rev(Path(eigsep_base.__file__).parent),
            },
            "inputs": {
                "binned": {"version": args.binned,
                           "sha256": sha256(bdir / "binned.npz")},
                "beam": {"name": args.beam, "path": str(beam_path.relative_to(workspace_root())),
                         "sha256": sha256(beam_path)},
                "horizons": {"path": f"marjum-2026-07/{HORIZON_PROFILES}.npz",
                             "sha256": sha256(hpath)},
                "sky_template": "pygdsm GlobalSkyModel16, ud_grade to nside_sky",
                "dem": {"path": f"marjum-2026-07/{sunmod.DEM_PATH}",
                        "sha256": sha256(campaign / sunmod.DEM_PATH)},
            },
        },
        "params": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(args).items() if k not in ("version", "out_dir")},
        "geometry": {"az_offset_deg": az_offset, "psi_deg": psi,
                     "horizon_grid_to_true_deg": (
                         0.0 if args.no_grid_rotation else sunmod.grid_to_true_deg()),
                     "eras": eras, "regimes": regimes},
        "bins": {
            "binned": n_all,
            "in_utc_window": n_window,
            "fitted": int(fitmask.sum()),
            "sun_rule": args.sun,
            "fitted_by_era_regime": {
                f"{e}/{r}": int(((z["era"] == e) & (z["regime"] == r)).sum())
                for e in eras for r in regimes},
        },
        "frequencies": {
            "fitted_mhz": freqs.round(3).tolist(),
            "skipped_mhz": z["freqs_mhz"][~fitted].round(3).tolist(),
            "rule": f"noise/radiometer <= {args.max_noise_ratio}, on the beam grid"
                    + (", inside the empirical beam's fitted band"
                       if args.beam == "empirical" else ""),
        },
        "validation": {
            "split": f"alternating {args.block_s:.0f} s blocks; fit even, test odd",
            "chi2_train_median": float(np.median(res["chi2_train"])),
            "chi2_test_median": float(np.median(res["chi2_test"])),
        },
        "ground_constrained_at_mhz": freqs[res["ground_constrained"]].round(3).tolist(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out_dir}")


def plot(out_dir, freqs, res, regimes):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    with_sun = np.isfinite(res["sun_flux_sfu"]).any()
    fig, axes = plt.subplots(5 if with_sun else 4, 1,
                             figsize=(7, 11 if with_sun else 9), sharex=True)

    def panel(ax, y, e, color, label=None):
        ax.errorbar(freqs, y, e, fmt="o", ms=3, lw=1, color=color,
                    ecolor=color, elinewidth=1, capsize=0, label=label)
        ax.grid(True, color="#e6e5e1", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    if res["ground_constrained"].any():
        panel(axes[0], res["t_ground_k"], res["sigma_t_ground_k"], BLUE)
    else:
        axes[0].text(0.5, 0.5, "T_gnd not constrained by these data",
                     transform=axes[0].transAxes, ha="center", va="center",
                     color=MUTED)
        axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("T_gnd (K)")
    panel(axes[1], res["gsm_amplitude"], res["sigma_gsm_amplitude"], BLUE)
    axes[1].set_ylabel("GSM amplitude a")
    for k, (r, c) in enumerate(zip(regimes, (BLUE, ORANGE))):
        panel(axes[2], res["pedestal_k"][:, k], res["sigma_pedestal_k"][:, k], c, r)
    axes[2].set_ylabel("⟨f_gnd⟩T_gnd + T_off (K)")
    if len(regimes) > 1:
        axes[2].legend(frameon=False, loc="best")
    if with_sun:
        panel(axes[4], res["sun_flux_sfu"], res["sigma_sun_flux_sfu"], BLUE)
        axes[4].axhline(0, color=MUTED, lw=0.8)
        axes[4].set_ylabel("S_sun (SFU)")
        axes[4].set_xlabel("frequency (MHz)")
    axes[3].semilogy(freqs, res["chi2_train"], "o-", ms=3, lw=1, color=BLUE, label="train")
    if with_sun:
        panel(axes[4], res["sun_flux_sfu"], res["sigma_sun_flux_sfu"], BLUE)
        axes[4].axhline(0, color=MUTED, lw=0.8)
        axes[4].set_ylabel("S_sun (SFU)")
        axes[4].set_xlabel("frequency (MHz)")
    axes[3].semilogy(freqs, res["chi2_test"], "o-", ms=3, lw=1, color=ORANGE, label="held out")
    axes[3].set_ylabel("χ² per bin")
    axes[3].legend(frameon=False, loc="best")
    axes[3].grid(True, color="#e6e5e1", lw=0.6)
    axes[3].spines[["top", "right"]].set_visible(False)
    axes[3].set_xlabel("frequency (MHz)")
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(out_dir / "fit_summary.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
