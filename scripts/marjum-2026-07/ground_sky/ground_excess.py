"""Where on the ground does the excess come from? Ground regions from the rotating antenna.

Builds ``derived/ground_sky/ground_excess_vNNNN``. ``raster_sky.py`` finds that,
as the antenna moves, raw power rises with the beam's ground fraction far more
than a 300 K ground allows below ~140 MHz (memo 008 § 7.3). Two explanations
predict different places for that excess:

- **Terrestrial radio noise** (man-made, or lightning) arrives from near the
  horizon and from particular directions on the landscape. It would concentrate
  in a band just below the terrain horizon, and its azimuth pattern would stay
  fixed on the landscape from one window to the next.
- **A beam error toward the sky** (a back lobe that sees more sky than the
  model) would be blamed on whatever ground the beam sees. It would be spread
  over the ground, not concentrated at the horizon, and its azimuth pattern
  would follow the sky, moving between windows hours apart in sidereal time.

Here the below-horizon region is split (``--split``):

- ``horizon``: within ``--band-deg`` below the terrain horizon, against the
  deeper ground;
- ``azimuth``: ``--n-az`` sectors of true azimuth.

Raw power per integration (channels midway between the transmitter's comb
teeth; ``flags@v3`` bits other than 9 and 10, or none with ``--no-flags``) is
regressed per frequency on GSM through the beam, one column per ground region,
a constant, the transmitter's leakage, the Sun and a drift. Each region's
brightness is its coefficient over the GSM coefficient, in GSM kelvin, with a
jackknife over ``--block-s`` blocks.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python ground_excess.py v0001 --split horizon --label raster
    python ground_excess.py v0001 --split azimuth --label raster
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import EarthLocation
from astropy.time import Time

import eigsep_data
import eigsep_sim
from eigsep_base.const import MARJUM_PASS
from eigsep_base.rotations import mount_rotation
from eigsep_data import MetadataIndex
from eigsep_sim.design_matrix import HorizonProfile, build_design_matrix
from eigsep_sim.observer import EarthSurface

import sun as sunmod
from common import (
    CHANNEL_MHZ, HERE, HORIZON_PROFILES, campaign_root, git_rev, mount_offsets, sha256,
)
from fit_ground_sky import gsm_maps, load_beam


def labeller(split, horizon, band_deg, n_az):
    """Ground-region label for topocentric ENU unit vectors (3, n); names."""
    if split == "horizon":
        def lab(d):
            x, y, z = d
            el = np.degrees(np.arcsin(np.clip(z, -1, 1)))
            hz = np.degrees(horizon.elevation(np.degrees(np.arctan2(y, x))))
            return np.where(el > hz - band_deg, 0, 1)
        return lab, [f"within {band_deg:g} deg below the horizon", "deeper ground"]
    width = 360.0 / n_az

    def lab(d):
        x, y, _ = d
        az = np.mod(90.0 - np.degrees(np.arctan2(y, x)), 360.0)  # true azimuth, E of N
        return np.floor(az / width).astype(int) % n_az
    return lab, [f"az {k * width:.0f}-{(k + 1) * width:.0f}" for k in range(n_az)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--split", choices=["horizon", "azimuth"], required=True)
    ap.add_argument("--band-deg", type=float, default=10.0)
    ap.add_argument("--n-az", type=int, default=8)
    ap.add_argument("--beam", choices=["hfss", "empirical"], default="empirical")
    ap.add_argument("--label", default="raster")
    ap.add_argument("--start", default="2026-07-17T20:26:00Z")
    ap.add_argument("--stop", default="2026-07-17T21:28:40Z")
    ap.add_argument("--era", default="91m")
    ap.add_argument("--fmax", type=float, default=180.0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--row-step", type=int, default=1)
    ap.add_argument("--block-s", type=float, default=120.0)
    ap.add_argument("--self-emi", action="store_true")
    ap.add_argument("--no-flags", action="store_true")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    campaign = campaign_root()
    out_dir = args.out_dir or campaign / "derived/ground_sky" / f"ground_excess_{args.version}"
    name = f"{args.label}_{args.split}{'_noflags' if args.no_flags else ''}_{args.beam}.json"
    if (out_dir / name).exists():
        sys.exit("exists; versions are immutable")
    a, b = pd.Timestamp(args.start).timestamp(), pd.Timestamp(args.stop).timestamp()
    B = MetadataIndex(campaign / "data").select(time=(a, b)).load_bundle(
        antenna="box-air", products=["flags@v3", "pointing@v2.0"])
    f = B.freqs_mhz
    ch = np.rint(f / CHANNEL_MHZ).astype(int)
    D = B.data.astype(float)
    D[D <= 0] = np.nan
    if not args.no_flags:
        D[(B.flags.astype(np.uint16) & ~np.uint16((1 << 9) | (1 << 10))) != 0] = np.nan
    if args.self_emi:
        D[:, np.abs(f - np.round(f)) < 0.15] = np.nan
    p = B.pointing
    ok = ((B.meta.rfswitch == "RFANT").to_numpy() & p.quality.isin(["ok", "suspect"]).to_numpy()
          & np.isfinite(p.az_deg.to_numpy()) & np.isfinite(p.el_deg.to_numpy()))
    beam, beam_path = load_beam(args.beam, campaign)
    bf = beam.freqs_hz / 1e6
    usable = [(i, x) for i, x in enumerate(bf) if 60 < x < args.fmax and not 85 < x < 110
              and (args.beam == "hfss" or beam.meta["empirical_frequency_mask"][i])][::args.every]
    freqs = np.array([x for _, x in usable])
    beam = beam.select([i for i, _ in usable])
    Y = np.array([np.nanmean(D[:, np.isin(np.abs(ch - int(round(x / CHANNEL_MHZ))), [3, 4, 5])], axis=1)
                  for x in freqs])[:, ok]
    t, az, el = B.t[ok], p.az_deg.to_numpy()[ok], p.el_deg.to_numpy()[ok]
    sl = slice(None, None, args.row_step)
    Y, t, az, el = Y[:, sl], t[sl], az[sl], el[sl]
    az_off, el_off, psi = mount_offsets(campaign)
    lat, lon, hgt = MARJUM_PASS
    rg = EarthSurface(lat, lon, hgt).rot_gal2top_stack(Time(t, format="unix")).astype(float)
    rb = mount_rotation(az + az_off, el + el_off, psi)
    hz = sunmod.true_horizon(HorizonProfile.from_npz(
        campaign / f"{HORIZON_PROFILES}.npz", args.era))
    lab, names = labeller(args.split, hz, args.band_deg, args.n_az)
    print(f"{len(t)} rows, {len(freqs)} frequencies, regions {names}", flush=True)
    dm = build_design_matrix(beam, [hz], rg, rb, 8, offset_groups=False, nside_int=64,
                             ground_labels=lab, ground_names=names)
    gsm = gsm_maps(freqs, 8)
    colG = np.einsum("ftp,fp->ft", dm.A[:, :, dm.sky], gsm)
    colR = dm.A[:, :, dm.ground]                         # (nfreq, ntime, nregion)
    geom = sunmod.sun_geometry(t, args.era)
    colS = sunmod.sun_column(beam, rb, geom, "knife")
    tx = sunmod.transmitter_enu()
    v = tx - sunmod.antenna_enu(args.era)
    gm = np.radians(sunmod.grid_to_true_deg())
    v = np.array([[np.cos(gm), -np.sin(gm), 0], [np.sin(gm), np.cos(gm), 0], [0, 0, 1]]) @ (v / np.linalg.norm(v))
    colT = beam(np.einsum("tji,j->it", rb, v))
    tt = (t - t.mean()) / 3600
    blk = ((t - t.min()) // args.block_s).astype(int)
    nreg = colR.shape[2]
    res = []
    for i, x in enumerate(freqs):
        y = Y[i]
        m = np.isfinite(y)
        A = np.stack([colG[i]] + [colR[i, :, k] for k in range(nreg)]
                     + [np.ones_like(y), colT[i], colS[i], tt], 1)
        co, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
        jk = []
        for bb in np.unique(blk[m]):
            s_ = m & (blk != bb)
            c_, *_ = np.linalg.lstsq(A[s_], y[s_], rcond=None)
            jk.append(c_[1:1 + nreg] / c_[0])
        jk = np.array(jk)
        n = len(jk)
        err = np.sqrt((n - 1) / n * np.sum((jk - jk.mean(0)) ** 2, axis=0))
        frac = [float(np.nanmean(colR[i, m, k])) for k in range(nreg)]
        span = [float(np.nanmax(colR[i, m, k]) - np.nanmin(colR[i, m, k])) for k in range(nreg)]
        res.append({"freq_mhz": float(x), "n_rows": int(m.sum()),
                    "T_region_gsm_k": [float(c / co[0]) for c in co[1:1 + nreg]],
                    "sigma_T_region_gsm_k": err.tolist(),
                    "mean_fraction": frac, "fraction_span": span})
        print(f"{x:6.1f} MHz  " + "  ".join(f"{nm}: {c / co[0]:7.0f}±{e:5.0f} (f {fr:.3f}±{sp:.3f})"
                                          for nm, c, e, fr, sp in zip(names, co[1:1 + nreg], err, frac, span)),
              flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_text(json.dumps({
        "provenance": {
            "product": "ground_sky_ground_excess", "version": args.version,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "data-analysis/scripts/marjum-2026-07/ground_sky/ground_excess.py",
            "argv": sys.argv[1:],
            "code": {"data-analysis": git_rev(HERE), "eigsep_data": git_rev(eigsep_data.__path__[0]),
                     "eigsep_sim": git_rev(Path(eigsep_sim.__file__).parent)},
            "inputs": {"beam_sha256": sha256(beam_path)}},
        "params": {k: (str(v_) if isinstance(v_, Path) else v_) for k, v_ in vars(args).items()
                   if k not in ("version", "out_dir")},
        "regions": names, "lst_mid_h": float(Time(t.mean(), format="unix", location=EarthLocation.from_geodetic(
            lon, lat, hgt)).sidereal_time("apparent").hour),
        "results": res}, indent=1))


if __name__ == "__main__":
    main()
