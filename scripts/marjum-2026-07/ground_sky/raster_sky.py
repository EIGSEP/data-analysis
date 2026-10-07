"""Low-order sky recovery from the rotating antenna, and whether it is really sky.

Builds ``derived/ground_sky/raster_sky_vNNNN``. During the 07-17 raster the
beam sweeps the sky in seconds, faster than the receiver gain drifts. Raw
power per integration, in channels midway between the transmitter's comb
teeth, is regressed per frequency on

    c_G (A_sky . GSM) + sum_lm c_lm (A_sky . Y_lm) + c_N A_gnd + c_0 + c_T B(tx)
    + c_S A_sun + c_t t

with real spherical harmonics Y_lm (l <= ``--lmax``) in Galactic coordinates.
The receiver gain multiplies everything, so c_lm / c_G is a correction to GSM
in GSM's own kelvin.

The catch: over an hour the sky turns only ~15 deg relative to the ground. A
pattern fixed on the sky and one fixed on the ground (terrain, horizon or beam
errors, which rotate with the antenna in the same way) then give nearly the
same columns. So the same fit is repeated with the Y_lm frozen at the mid-raster
sky orientation, i.e. fixed to the ground. Rows are split into alternating
``--block-s`` blocks: fitted on even blocks, scored on odd ones.

Checks that could fail:

- **Held out**: the sky Y_lm must lower the held-out residual, not just the
  training residual.
- **Sky, not ground**: the sky-fixed Y_lm must predict held-out data better
  than the ground-fixed ones. If they do equally well, the structure is not
  identified as sky.
- **Beams agree**: the recovered corrections should agree between the HFSS
  and empirical beams (``--beam``); beam errors would not.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python raster_sky.py v0001 --beam empirical
    python raster_sky.py v0001 --beam hfss
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import healpy
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


def real_ylm_maps(nside, lmax):
    """Real spherical harmonics on HEALPix, (n_lm, npix), unit rms, with labels."""
    maps, labels = [], []
    nalm = healpy.Alm.getsize(lmax)
    for ell in range(1, lmax + 1):
        for m in range(0, ell + 1):
            for part in ((("re",) if m == 0 else ("re", "im"))):
                alm = np.zeros(nalm, complex)
                alm[healpy.Alm.getidx(lmax, ell, m)] = 1.0 if part == "re" else 1j
                mp = healpy.alm2map(alm, nside, lmax=lmax)
                maps.append(mp / np.sqrt(np.mean(mp**2)))
                labels.append(f"l{ell}m{m}{part}")
    return np.array(maps), labels


def reflected_sky(beam, horizon, rot_gal2top, rot_body2enu, sky_maps, nside_int=32):
    """Beam-weighted sky reflected specularly by flat ground, per row: (nfreq, ntime).

    Directions below the local horizontal (z < 0) that the terrain does not
    already block from above are mirrored to (x, y, -z); the reflected
    brightness is the sky map there if that mirror direction is above the
    terrain horizon. Terrain-blocked directions above the horizontal emit but do
    not reflect the sky here. Normalized like the design matrix (full-sphere
    beam integral), so the column is in kelvin of the given sky maps.
    """
    nint = healpy.nside2npix(nside_int)
    d = np.array(healpy.pix2vec(nside_int, np.arange(nint)))
    vis = horizon.visible(d)
    refl = (d[2] < 0) & ~vis
    mir = d.copy()
    mir[2] = -mir[2]
    ok = refl & horizon.visible(mir)
    nside_sky = healpy.npix2nside(sky_maps.shape[1])
    out = np.zeros((sky_maps.shape[0], len(rot_gal2top)))
    for k in range(len(rot_gal2top)):
        B = beam(rot_body2enu[k].T @ d)
        B = B / B.sum(axis=1, keepdims=True)
        g = rot_gal2top[k].T @ mir[:, ok]
        pix = healpy.vec2pix(nside_sky, *g)
        out[:, k] = np.einsum("fn,fn->f", B[:, ok], sky_maps[:, pix])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--beam", choices=["hfss", "empirical"], required=True)
    ap.add_argument("--start", default="2026-07-17T20:26:00Z")
    ap.add_argument("--stop", default="2026-07-17T21:28:40Z")
    ap.add_argument("--era", default="91m")
    ap.add_argument("--label", default="raster", help="name of this window in the output file")
    ap.add_argument("--reflect", action="store_true",
                    help="also fit a specularly reflected sky term (ground reflectivity R)")
    ap.add_argument("--lmax", type=int, default=3)
    ap.add_argument("--nside", type=int, default=8)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--block-s", type=float, default=120.0)
    ap.add_argument("--row-step", type=int, default=1, help="use every Nth integration")
    ap.add_argument("--self-emi", action="store_true",
                    help="mask box-air's self-EMI teeth (integer MHz ± 0.15, memo 006)")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    campaign = campaign_root()
    out_dir = args.out_dir or campaign / "derived/ground_sky" / f"raster_sky_{args.version}"
    if (out_dir / f"{args.label}_{args.beam}.json").exists():
        sys.exit("exists; versions are immutable")
    a, b = pd.Timestamp(args.start).timestamp(), pd.Timestamp(args.stop).timestamp()
    B = MetadataIndex(campaign / "data").select(time=(a, b)).load_bundle(
        antenna="box-air", products=["flags@v3", "pointing@v2.0"])
    f = B.freqs_mhz
    ch = np.rint(f / CHANNEL_MHZ).astype(int)
    D = B.data.astype(float)
    D[((B.flags.astype(np.uint16) & ~np.uint16((1 << 9) | (1 << 10))) != 0) | (D <= 0)] = np.nan
    if args.self_emi:
        D[:, np.abs(f - np.round(f)) < 0.15] = np.nan
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
    sl = slice(None, None, args.row_step)
    Y, t, az, el = Y[:, sl], t[sl], az[sl], el[sl]
    with np.load(campaign / "derived/beam/empirical_raster_v0012/dpss/diagnostics.npz") as d:
        az_off, psi = float(d["az_offset_deg"]), float(d["psi_deg"])
    lat, lon, hgt = MARJUM_PASS
    rg = EarthSurface(lat, lon, hgt).rot_gal2top_stack(Time(t, format="unix")).astype(float)
    rg_frozen = np.broadcast_to(rg[len(rg) // 2], rg.shape).copy()
    rb = mount_rotation(az + az_off, el, psi)
    hz = sunmod.true_horizon(HorizonProfile.from_npz(
        campaign / "curation/horizon_profiles_v0002.npz", args.era))
    gsm = gsm_maps(freqs, args.nside)
    ylm, labels = real_ylm_maps(args.nside, args.lmax)
    print(f"{len(t)} rows, {len(freqs)} frequencies, {len(labels)} Y_lm", flush=True)

    def columns(rot):
        dm = build_design_matrix(beam, [hz], rot, rb, args.nside, offset_groups=False, nside_int=64)
        Ask = dm.A[:, :, dm.sky]
        return (np.einsum("ftp,fp->ft", Ask, gsm), np.einsum("ftp,kp->fkt", Ask, ylm),
                dm.A[:, :, dm.ground][:, :, 0])

    colG, colY, colN = columns(rg)
    _, colYf, _ = columns(rg_frozen)
    geom = sunmod.sun_geometry(t, args.era)
    colS = sunmod.sun_column(beam, rb, geom, "knife")
    tx = np.array(json.loads((campaign / "curation/transmitter_position.json").read_text())
                  ["best_estimate_enu_m"])
    v = tx - sunmod.antenna_enu(args.era)
    gm = np.radians(sunmod.grid_to_true_deg())
    v = np.array([[np.cos(gm), -np.sin(gm), 0], [np.sin(gm), np.cos(gm), 0], [0, 0, 1]]) @ (v / np.linalg.norm(v))
    colT = beam(np.einsum("tji,j->it", rb, v))
    colR = reflected_sky(beam, hz, rg, rb, gsm) if args.reflect else None
    tt = (t - t.mean()) / 3600
    train = ((t - t.min()) // args.block_s).astype(int) % 2 == 0

    res = []
    for i, x in enumerate(freqs):
        y = Y[i]
        m = np.isfinite(y)
        base = [colG[i], colN[i], np.ones_like(y), colT[i], colS[i], tt]

        def fit(extra):
            A = np.stack(base + list(extra), 1)
            tr, te = m & train, m & ~train
            co, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
            rt = y[tr] - A[tr] @ co
            rv = y[te] - A[te] @ co
            return co, float(np.sqrt(np.mean(rt**2))), float(np.sqrt(np.mean(rv**2)))

        def tgnd(extra):
            """T_gnd in GSM kelvin (ground coefficient / GSM coefficient) on all rows,
            with a jackknife error over the time blocks."""
            A = np.stack(base + list(extra), 1)
            blk = ((t - t.min()) // args.block_s).astype(int)
            co, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
            jk = []
            for bb in np.unique(blk[m]):
                s_ = m & (blk != bb)
                c_, *_ = np.linalg.lstsq(A[s_], y[s_], rcond=None)
                jk.append(c_[1] / c_[0])
            jk = np.array(jk)
            n = len(jk)
            return float(co[1] / co[0]), float(np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2)))

        c0, tr0, te0 = fit([])
        cs, trs, tes = fit(colY[i])
        cf, trf, tef = fit(colYf[i])
        mean = float(np.nanmean(y[m]))
        tg0, etg0 = tgnd([])
        refl = None
        if colR is not None:
            A = np.stack(base + [colR[i]], 1)
            blk = ((t - t.min()) // args.block_s).astype(int)
            co, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
            jr, jn = [], []
            for bb in np.unique(blk[m]):
                s_ = m & (blk != bb)
                c_, *_ = np.linalg.lstsq(A[s_], y[s_], rcond=None)
                jr.append(c_[-1] / c_[0])
                jn.append(c_[1] / c_[0])
            nb_ = len(jr)
            jk = lambda v: float(np.sqrt((nb_ - 1) / nb_ * np.sum((np.array(v) - np.mean(v)) ** 2)))  # noqa: E731
            R = co[-1] / co[0]
            refl = {"R": float(R), "sigma_R": jk(jr),
                    "emission_k": float(co[1] / co[0]), "sigma_emission_k": jk(jn),
                    "T_phys_k": float(co[1] / co[0] / (1 - R)) if R < 1 else None}
        tgs, etgs = tgnd(colY[i])
        tgf, etgf = tgnd(colYf[i])
        res.append({"freq_mhz": float(x),
                    "train_rms_frac": {"base": tr0 / mean, "sky_ylm": trs / mean, "ground_ylm": trf / mean},
                    "test_rms_frac": {"base": te0 / mean, "sky_ylm": tes / mean, "ground_ylm": tef / mean},
                    "gsm_coef": float(cs[0]),
                    "T_gnd_gsm_k": {"base": tg0, "sky_ylm": tgs, "ground_ylm": tgf},
                    "sigma_T_gnd_gsm_k": {"base": etg0, "sky_ylm": etgs, "ground_ylm": etgf},
                    "f_gnd_range": [float(np.nanmin(colN[i][m])), float(np.nanmax(colN[i][m]))],
                    "reflection": refl,
                    "delta_ylm_k": [float(c / cs[0]) for c in cs[len(base):]]})
        print(f"{x:6.1f} MHz held-out rms (% of power): base {100*te0/mean:.2f}  sky Y_lm "
              f"{100*tes/mean:.2f}  ground Y_lm {100*tef/mean:.2f}   T_gnd (GSM K) {tg0:.0f}±{etg0:.0f} "
              f"/ {tgs:.0f}±{etgs:.0f} / {tgf:.0f}±{etgf:.0f}"
              + (f"   R {refl['R']:.2f}±{refl['sigma_R']:.2f} emission {refl['emission_k']:.0f}±"
                 f"{refl['sigma_emission_k']:.0f} T_phys {refl['T_phys_k']}" if refl else ""), flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.label}_{args.beam}.json").write_text(json.dumps({
        "provenance": {
            "product": "ground_sky_raster_sky", "version": args.version, "beam": args.beam,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "data-analysis/scripts/marjum-2026-07/ground_sky/raster_sky.py",
            "argv": sys.argv[1:],
            "code": {"data-analysis": git_rev(HERE), "eigsep_data": git_rev(eigsep_data.__path__[0]),
                     "eigsep_sim": git_rev(Path(eigsep_sim.__file__).parent)},
            "inputs": {"beam_sha256": sha256(beam_path)}},
        "params": {k: v for k, v in vars(args).items() if k not in ("version", "out_dir")},
        "ylm_labels": labels, "results": res}, indent=1))


if __name__ == "__main__":
    main()
