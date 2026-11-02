"""Which change removed the raster's low-frequency ground excess? One variable at a time.

Builds ``derived/ground_sky/excess_swap_vNNNN``. On beam ``empirical_raster_v0013``
(right-handed pointing v4, no axle tilt) the flagged 07-17 raster's ground term
was ~3600 GSM K at 64 MHz; on ``empirical_raster_v0017`` (left-handed pointing v5,
a +4.74 deg along-axle tilt, el offset 0.05 instead of 0.93 deg) it is ~300 K
(``raster_sky_v0006`` and ``v0007``, memo 008). Four things changed together:

- **beam**: the fitted beam pattern (v0013 or v0017's ``empirical_beam.npz``);
- **hand**: the azimuth handedness, pointing v4 with v0013's az offset, or v5
  with v0017's;
- **tilt**: the along-axle tilt, 0 or v0017's;
- **el**: the el-zero offset, v0013's or v0017's.

All 16 combinations are fitted with ``raster_sky.py``'s base model (GSM through
the beam, one ground column, a constant, the transmitter's leakage, the Sun and
a drift; no Y_lm, no reflection), on the same rows, flags and frequencies, so
only the four choices differ. T_gnd is the ground coefficient over the GSM
coefficient, in GSM kelvin, with a jackknife over ``--block-s`` blocks. Rows are
those with finite pointing in both tables, so the two handedness choices use
the same integrations.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python excess_swap.py v0001
"""

from __future__ import annotations

import argparse
import itertools
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
from eigsep_data import MetadataIndex
from eigsep_sim.design_matrix import HealpixBeam, HorizonProfile, build_design_matrix
from eigsep_sim.observer import EarthSurface

import sun as sunmod
from common import (
    CHANNEL_MHZ, HERE, HORIZON_PROFILES, beam_rotation, campaign_root, git_rev, rows_at_height, sha256,
)
from fit_ground_sky import gsm_maps

OLD, NEW = "derived/beam/empirical_raster_v0013", "derived/beam/empirical_raster_v0017"
POINTINGS = {OLD: "v4", NEW: "v5"}


def geometry(product, campaign):
    """az_add / el_add / tilt / psi of a beam product's fit (``common.beam_geometry``
    for any product)."""
    from eigsep_data.beam_mapping.tx_fit import PARAM_NAMES

    with np.load(campaign / product / "dpss/diagnostics.npz") as d:
        vals = np.asarray(d["params"], float)
        p = dict(zip(PARAM_NAMES, np.pad(vals, (0, len(PARAM_NAMES) - vals.size))))
        return {"az_add_deg": float(d["az_offset_deg"] + p["az_zero_delta_deg"]),
                "el_add_deg": float(p["el_zero_delta_deg"]),
                "axle_tilt_deg": float(p["axle_tilt_delta_deg"]), "psi_deg": float(d["psi_deg"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--start", default="2026-07-17T20:26:00Z")
    ap.add_argument("--stop", default="2026-07-17T21:28:40Z")
    ap.add_argument("--era", default="91m")
    ap.add_argument("--freqs", type=float, nargs="+", default=[64.5, 70.0, 75.0, 82.0, 125.0, 140.0, 162.0])
    ap.add_argument("--nside", type=int, default=8)
    ap.add_argument("--block-s", type=float, default=120.0)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    campaign = campaign_root()
    out_dir = args.out_dir or campaign / "derived/ground_sky" / f"excess_swap_{args.version}"
    if (out_dir / "excess_swap.json").exists():
        sys.exit("exists; versions are immutable")
    a, b = pd.Timestamp(args.start).timestamp(), pd.Timestamp(args.stop).timestamp()
    sel = MetadataIndex(campaign / "data").select(time=(a, b))
    bundles = {v: sel.load_bundle(antenna="box-air", products=["flags@v3", f"pointing@{v}"]) for v in ("v4", "v5")}
    B = bundles["v5"]
    assert np.array_equal(B.t, bundles["v4"].t)
    f = B.freqs_mhz
    ch = np.rint(f / CHANNEL_MHZ).astype(int)
    D = B.data.astype(float)
    D[((B.flags.astype(np.uint16) & ~np.uint16((1 << 9) | (1 << 10))) != 0) | (D <= 0)] = np.nan
    ok = (B.meta.rfswitch == "RFANT").to_numpy()
    for bb in bundles.values():
        p = bb.pointing
        ok &= (p.quality.isin(["ok", "suspect"]).to_numpy()
               & np.isfinite(p.az_deg.to_numpy()) & np.isfinite(p.el_deg.to_numpy()))
    at_height, era_counts = rows_at_height(B.meta.file.to_numpy(), args.era, campaign)
    ok &= at_height
    t = B.t[ok]
    pt = {v: (bb.pointing.az_deg.to_numpy()[ok], bb.pointing.el_deg.to_numpy()[ok]) for v, bb in bundles.items()}
    d_el = pt["v4"][1] - pt["v5"][1]
    s_az = np.mod(pt["v4"][0] + pt["v5"][0] + 180, 360) - 180
    print(f"{len(t)} rows; el v4-v5 {np.nanmin(d_el):.3f}..{np.nanmax(d_el):.3f}; "
          f"az v4+v5 {np.nanmin(s_az):.2f}..{np.nanmax(s_az):.2f} deg", flush=True)

    beams = {k: HealpixBeam.from_npz(campaign / k / "dpss/empirical_beam.npz") for k in (OLD, NEW)}
    geoms = {k: geometry(k, campaign) for k in (OLD, NEW)}
    # the same beam slices in both: the nearest usable slice of the new beam to each target
    bf = beams[NEW].freqs_hz / 1e6
    idx = []
    for x in args.freqs:
        cand = [i for i in range(len(bf)) if beams[NEW].meta["empirical_frequency_mask"][i]
                and beams[OLD].meta["empirical_frequency_mask"][i]]
        idx.append(min(cand, key=lambda i: abs(bf[i] - x)))
    assert np.allclose(beams[OLD].freqs_hz[idx], beams[NEW].freqs_hz[idx])
    freqs = bf[idx]
    beams = {k: v.select(idx) for k, v in beams.items()}
    Y = np.array([np.nanmean(D[:, np.isin(np.abs(ch - int(round(x / CHANNEL_MHZ))), [3, 4, 5])], axis=1)
                  for x in freqs])[:, ok]
    lat, lon, hgt = MARJUM_PASS
    rg = EarthSurface(lat, lon, hgt).rot_gal2top_stack(Time(t, format="unix")).astype(float)
    hz = sunmod.true_horizon(HorizonProfile.from_npz(campaign / f"{HORIZON_PROFILES}.npz", args.era))
    gsm = gsm_maps(freqs, args.nside)
    geom = sunmod.sun_geometry(t, args.era)
    tx = sunmod.transmitter_enu()
    v = tx - sunmod.antenna_enu(args.era)
    gm = np.radians(sunmod.grid_to_true_deg())
    v = np.array([[np.cos(gm), -np.sin(gm), 0], [np.sin(gm), np.cos(gm), 0], [0, 0, 1]]) @ (v / np.linalg.norm(v))
    tt = (t - t.mean()) / 3600
    blk = ((t - t.min()) // args.block_s).astype(int)

    res = []
    for bk, hk, tk, ek in itertools.product((OLD, NEW), repeat=4):
        g = {"az_add_deg": geoms[hk]["az_add_deg"], "el_add_deg": geoms[ek]["el_add_deg"],
             "axle_tilt_deg": geoms[tk]["axle_tilt_deg"], "psi_deg": geoms[NEW]["psi_deg"]}
        az, el = pt[POINTINGS[hk]]
        rb = beam_rotation(az, el, geometry=g)
        beam = beams[bk]
        dm = build_design_matrix(beam, [hz], rg, rb, args.nside, offset_groups=False, nside_int=64)
        colG = np.einsum("ftp,fp->ft", dm.A[:, :, dm.sky], gsm)
        colN = dm.A[:, :, dm.ground][:, :, 0]
        colS = sunmod.sun_column(beam, rb, geom, "knife")
        colT = beam(np.einsum("tji,j->it", rb, v))
        tag = {"beam": bk[-5:], "hand": POINTINGS[hk], "tilt": tk[-5:], "el": ek[-5:]}
        rows = []
        for i, x in enumerate(freqs):
            y = Y[i]
            m = np.isfinite(y)
            A = np.stack([colG[i], colN[i], np.ones_like(y), colT[i], colS[i], tt], 1)
            co, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
            jk = []
            for bb in np.unique(blk[m]):
                s_ = m & (blk != bb)
                c_, *_ = np.linalg.lstsq(A[s_], y[s_], rcond=None)
                jk.append(c_[1] / c_[0])
            jk = np.array(jk)
            n = len(jk)
            rms = float(np.sqrt(np.mean((y[m] - A[m] @ co) ** 2)) / np.mean(y[m]))
            rows.append({"freq_mhz": float(x), "T_gnd_gsm_k": float(co[1] / co[0]),
                         "sigma_T_gnd_gsm_k": float(np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))),
                         "rms_frac": rms, "f_gnd_range": [float(colN[i][m].min()), float(colN[i][m].max())]})
        res.append({**tag, "geometry": g, "results": rows})
        print(" ".join(f"{k}={v_}" for k, v_ in tag.items()) + "  "
              + "  ".join(f"{r['freq_mhz']:.0f}:{r['T_gnd_gsm_k']:.0f}±{r['sigma_T_gnd_gsm_k']:.0f}"
                          f"({100 * r['rms_frac']:.2f}%)" for r in rows), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "excess_swap.json").write_text(json.dumps({
        "provenance": {
            "product": "ground_sky_excess_swap", "version": args.version,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "data-analysis/scripts/marjum-2026-07/ground_sky/excess_swap.py",
            "argv": sys.argv[1:],
            "code": {"data-analysis": git_rev(HERE), "eigsep_data": git_rev(eigsep_data.__path__[0]),
                     "eigsep_sim": git_rev(Path(eigsep_sim.__file__).parent)},
            "inputs": {k: sha256(campaign / k / "dpss/empirical_beam.npz") for k in (OLD, NEW)},
            "height": {"era": args.era, "rows": len(t), "mode_table_eras": era_counts}},
        "params": {k: v_ for k, v_ in vars(args).items() if k not in ("version", "out_dir")},
        "geometries": {k[-5:]: g for k, g in geoms.items()},
        "results": res}, indent=1))


if __name__ == "__main__":
    main()
