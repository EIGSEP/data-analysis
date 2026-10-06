"""Can the Sun be separated from sky and ground while the antenna rotates?

Builds ``derived/ground_sky/raster_sun_vNNNN``. During the 07-17 beam-scan
raster (20:26-21:28 UTC, 91 m) the antenna sweeps in elevation at ~5 deg/s,
so the beam's gain toward the Sun changes within seconds, faster than the
receiver gain drifts. Raw power per integration, in channels midway between the
transmitter's comb teeth (3-5 channels from each HFSS beam frequency;
``flags@v3`` bits other than 9 and 10 masked), is regressed per frequency on

    c_G (A_sky . GSM) + c_N A_gnd + c_0 + c_T B(tx) + c_t t + c_S A_sun

with the columns built through the beam on the real pointing
(``eigsep_sim.design_matrix``; horizon rotated to true bearings). B(tx) is the
beam power toward the transmitter, for its leakage outside the comb. Since the
receiver gain multiplies every column, c_S / c_G is the Sun's flux on GSM's
brightness scale, independent of any kelvin calibration.

Null: the same fit with the Sun column built for fake Suns at the same
altitude, rotated in azimuth by 30-330 deg. A real detection should improve
chi2 more than every fake.

Run once per beam::

    python raster_sun.py v0001 --beam hfss
    python raster_sun.py v0001 --beam empirical    # writes into the same product
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
from common import CHANNEL_MHZ, HERE, campaign_root, git_rev, sha256
from fit_ground_sky import gsm_maps, load_beam


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--beam", choices=["hfss", "empirical"], required=True)
    ap.add_argument("--start", default="2026-07-17T20:26:00Z")
    ap.add_argument("--stop", default="2026-07-17T21:28:40Z")
    ap.add_argument("--era", default="91m")
    ap.add_argument("--every", type=int, default=3, help="use every Nth beam frequency")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    campaign = campaign_root()
    out_dir = args.out_dir or campaign / "derived/ground_sky" / f"raster_sun_{args.version}"
    if (out_dir / f"raster_{args.beam}.json").exists():
        sys.exit(f"{out_dir}/raster_{args.beam}.json exists; versions are immutable")
    a, b = pd.Timestamp(args.start).timestamp(), pd.Timestamp(args.stop).timestamp()
    B = MetadataIndex(campaign / "data").select(time=(a, b)).load_bundle(
        antenna="box-air", products=["flags@v3", "pointing@v2.0"])
    f = B.freqs_mhz
    ch = np.rint(f / CHANNEL_MHZ).astype(int)
    D = B.data.astype(float)
    D[((B.flags.astype(np.uint16) & ~np.uint16((1 << 9) | (1 << 10))) != 0) | (D <= 0)] = np.nan
    p = B.pointing
    ok = ((B.meta.rfswitch == "RFANT").to_numpy() & p.quality.isin(["ok", "suspect"]).to_numpy()
          & np.isfinite(p.az_deg.to_numpy()) & np.isfinite(p.el_deg.to_numpy()))
    beam, beam_path = load_beam(args.beam, campaign)
    bf = beam.freqs_hz / 1e6
    usable = [(i, x) for i, x in enumerate(bf) if 60 < x < 235 and not 85 < x < 110
              and (args.beam == "hfss" or beam.meta["empirical_frequency_mask"][i])]
    usable = usable[::args.every]
    freqs = np.array([x for _, x in usable])
    beam = beam.select([i for i, _ in usable])
    Y = np.array([np.nanmean(D[:, np.isin(np.abs(ch - int(round(x / CHANNEL_MHZ))), [3, 4, 5])], axis=1)
                  for x in freqs])[:, ok]
    t, az, el = B.t[ok], p.az_deg.to_numpy()[ok], p.el_deg.to_numpy()[ok]

    with np.load(campaign / "derived/beam/empirical_raster_v0012/dpss/diagnostics.npz") as d:
        az_off, psi = float(d["az_offset_deg"]), float(d["psi_deg"])
    lat, lon, hgt = MARJUM_PASS
    rg = EarthSurface(lat, lon, hgt).rot_gal2top_stack(Time(t, format="unix")).astype(float)
    rb = mount_rotation(az + az_off, el, psi)
    hz = sunmod.true_horizon(HorizonProfile.from_npz(
        campaign / "curation/horizon_profiles_v0002.npz", args.era))
    print(f"{len(t)} rows, {len(freqs)} frequencies; building design matrix ...", flush=True)
    dm = build_design_matrix(beam, [hz], rg, rb, 8, offset_groups=False, nside_int=64)
    gsm = gsm_maps(freqs, 8)
    colG = np.einsum("ftp,fp->ft", dm.A[:, :, dm.sky], gsm)
    colN = dm.A[:, :, dm.ground][:, :, 0]
    geom = sunmod.sun_geometry(t, args.era)
    colS = sunmod.sun_column(beam, rb, geom, "knife")
    tx = np.array(json.loads((campaign / "curation/transmitter_position.json").read_text())
                  ["best_estimate_enu_m"])
    v = tx - sunmod.antenna_enu(args.era)
    g = np.radians(sunmod.grid_to_true_deg())
    v = np.array([[np.cos(g), -np.sin(g), 0], [np.sin(g), np.cos(g), 0], [0, 0, 1]]) @ (v / np.linalg.norm(v))
    colT = beam(np.einsum("tji,j->it", rb, v))

    def fake(dphi):
        g2 = dict(geom)
        al, b_ = np.radians(geom["alt"]), np.radians(90 - geom["az"] + dphi)
        g2["enu"] = np.array([np.cos(al) * np.cos(b_), np.cos(al) * np.sin(b_), np.sin(al)])
        return sunmod.sun_column(beam, rb, g2, "flat")

    dphis = list(range(30, 331, 30))
    fakes = [fake(x) for x in dphis]
    tt = (t - t.mean()) / 3600
    res = []
    for i, x in enumerate(freqs):
        y = Y[i]
        m = np.isfinite(y)

        def fit(extra):
            A = np.stack([colG[i], colN[i], np.ones_like(y), colT[i], tt] + extra, 1)[m]
            co, *_ = np.linalg.lstsq(A, y[m], rcond=None)
            r = y[m] - A @ co
            return co, float(r @ r), A

        _, c0, _ = fit([])
        co, c1, A1 = fit([colS[i]])
        s2 = c1 / (m.sum() - A1.shape[1])
        cov = np.linalg.inv(A1.T @ A1) * s2
        dfake = [(c0 - fit([fk[i]])[1]) / s2 for fk in fakes]
        res.append({"freq_mhz": float(x), "n_rows": int(m.sum()),
                    "S_on_gsm_scale_sfu": float(co[5] / co[0]),
                    "sigma_sfu": float(np.sqrt(cov[5, 5]) / abs(co[0])),
                    "dchi2_sun": float((c0 - c1) / s2), "dchi2_fake": [float(z) for z in dfake],
                    "fake_rank": int(np.sum(np.array(dfake) >= (c0 - c1) / s2)),
                    "resid_frac_rms": float(np.sqrt(c1 / m.sum()) / np.nanmean(y[m]))})
        print(f"{x:6.1f} MHz S {res[-1]['S_on_gsm_scale_sfu']:8.1f}  dchi2 {res[-1]['dchi2_sun']:7.1f} "
              f"fakes above {res[-1]['fake_rank']}/{len(dphis)}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"raster_{args.beam}.json").write_text(json.dumps({
        "provenance": {
            "product": "ground_sky_raster_sun", "version": args.version, "beam": args.beam,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "data-analysis/scripts/marjum-2026-07/ground_sky/raster_sun.py",
            "argv": sys.argv[1:],
            "code": {"data-analysis": git_rev(HERE), "eigsep_data": git_rev(eigsep_data.__path__[0]),
                     "eigsep_sim": git_rev(Path(eigsep_sim.__file__).parent)},
            "inputs": {"beam_sha256": sha256(beam_path)}},
        "fake_azimuth_offsets_deg": dphis, "results": res}, indent=1))
    np.savez_compressed(out_dir / f"raster_{args.beam}.npz", freqs_mhz=freqs, t=t, az=az, el=el,
                        y=Y.astype(np.float32), colG=colG.astype(np.float32),
                        colN=colN.astype(np.float32), colS=colS.astype(np.float32),
                        colT=colT.astype(np.float32))


if __name__ == "__main__":
    main()
