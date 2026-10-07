"""Joint fit across frequency: one ground temperature, a smooth receiver offset.

Builds ``derived/ground_sky/joint_vNNNN``. The per-frequency fits
(``fit_ground_sky.py``) leave T_gnd and the offset exactly degenerate at each
frequency, because one pointing and one height give a constant ground column.
Across frequency they are not quite the same: the beam's ground fraction
f_gnd(nu) changes with frequency. So if

- T_gnd is one temperature at all frequencies (thermal emission from soil, with
  an emissivity that changes slowly), and
- the receiver offset is a smooth function of frequency (Legendre polynomials
  in log nu, order ``--orders``),

the pedestal f_gnd(nu) T_gnd + T_off(nu) can separate them, if f_gnd(nu) has
structure the offset's basis cannot copy.

Method: at each frequency the per-frequency model (sky pixels with a 10% GSM
prior, the GSM amplitude a, T_gnd and the offset) is reduced to an effective
Gaussian likelihood for (T_gnd, offset) by marginalizing everything else
(Schur complement of the precision), with errors scaled by sqrt(chi2/dof) of the
per-frequency fit. The joint parameters (T_gnd, offset coefficients) then
follow from summing those likelihoods over frequency.

Checks that could fail: T_gnd should not move with the offset order, nor
between the lower and upper halves of the band, and should be physical
(250-330 K for ground at night, scaled by memo 008's a if the factor is in the
kelvin scale). Reads the night bins of ``fit_vNNNN`` (``--fit``) and its
``binned`` input.

Run::

    export EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07
    python joint_fit.py v0001 --fit v0001
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from astropy.time import Time
from numpy.polynomial import legendre

import eigsep_sim
from eigsep_base.const import MARJUM_PASS
from eigsep_base.rotations import mount_rotation
from eigsep_sim.design_matrix import HorizonProfile, build_design_matrix
from eigsep_sim.observer import EarthSurface

import sun as sunmod
from common import ERAS, HERE, campaign_root, git_rev, sha256
from fit_ground_sky import gsm_maps, load_beam


def effective(A, y, s, prior_sigma, keep):
    """Marginal precision and information for columns ``keep`` at one frequency."""
    w = 1 / s
    Aw = A * w[:, None]
    L = Aw.T @ Aw + np.diag(1 / prior_sigma**2)
    b = Aw.T @ (y * w)
    n = np.setdiff1d(np.arange(A.shape[1]), keep)
    Lnn = L[np.ix_(n, n)]
    sol = np.linalg.solve(Lnn, np.c_[L[np.ix_(n, keep)], b[n]])
    P = L[np.ix_(keep, keep)] - L[np.ix_(keep, n)] @ sol[:, :-1]
    h = b[keep] - L[np.ix_(keep, n)] @ sol[:, -1]
    x = np.linalg.lstsq(L, b, rcond=None)[0]
    chi2 = float(np.sum(((y - A @ x) * w) ** 2))
    return P, h, chi2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("version")
    ap.add_argument("--fit", default="v0001", help="fit_vNNNN whose fitted (night) bins to use")
    ap.add_argument("--beam", choices=["hfss", "empirical"], default="hfss")
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--sky-prior-frac", type=float, default=0.1)
    ap.add_argument("--fmin", type=float, default=105.0)
    ap.add_argument("--fmax", type=float, default=235.0)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    campaign = campaign_root()
    out_dir = args.out_dir or campaign / "derived/ground_sky" / f"joint_{args.version}"
    if (out_dir / f"joint_{args.beam}.json").exists():
        sys.exit("exists; versions are immutable")
    fdir = campaign / "derived/ground_sky" / f"fit_{args.fit}"
    man = json.loads((fdir / "manifest.json").read_text())
    z = dict(np.load(fdir / "fit.npz", allow_pickle=True))
    keep = z["fitted"] & (z["regime"] == "rx-A")
    freqs_all = z["freqs_mhz"]
    fsel = np.flatnonzero((freqs_all >= args.fmin) & (freqs_all <= args.fmax))
    beam, beam_path = load_beam(args.beam, campaign)
    bidx = [int(np.argmin(np.abs(beam.freqs_hz / 1e6 - freqs_all[i]))) for i in fsel]
    if args.beam == "empirical":
        m = beam.meta["empirical_frequency_mask"][bidx].astype(bool)
        fsel, bidx = fsel[m], list(np.array(bidx)[m])
    beam = beam.select(bidx)
    freqs = freqs_all[fsel]
    Y = z["data_k"][fsel][:, keep]
    S = z["sigma_k"][fsel][:, keep]
    t = z["t"][keep]
    eras = sorted(set(z["era"][keep]))
    assert len(eras) == 1, eras
    with np.load(campaign / "derived/beam/empirical_raster_v0012/dpss/diagnostics.npz") as d:
        az_off, psi = float(d["az_offset_deg"]), float(d["psi_deg"])
    lat, lon, hgt = MARJUM_PASS
    rg = EarthSurface(lat, lon, hgt).rot_gal2top_stack(Time(t, format="unix")).astype(float)
    rb = mount_rotation(np.round(z["az"][keep], 1) + az_off, np.round(z["el"][keep], 1), psi)
    hz = sunmod.true_horizon(HorizonProfile.from_npz(
        campaign / "curation/horizon_profiles_v0002.npz", ERAS[eras[0]]))
    dm = build_design_matrix(beam, [hz], rg, rb, man["params"]["nside_sky"], nside_int=64)
    gsm = gsm_maps(freqs, man["params"]["nside_sky"])
    dm = dm.with_sky_templates(gsm, names=["gsm_amplitude"])
    g, o = dm.ground.start, dm.offset.start
    keep_cols = np.array([g, o])
    per = []
    for fi, x in enumerate(freqs):
        y, s = Y[fi], S[fi]
        ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
        prior = dm.pack(args.sky_prior_frac * gsm[fi], np.inf, np.inf, template=np.inf)[fi].copy()
        prior[~np.isfinite(prior)] = 1e9
        A = dm.A[fi][ok]
        P, h, chi2 = effective(A, y[ok], s[ok], prior, keep_cols)
        scale = max(chi2 / max(ok.sum() - 2, 1), 1.0)
        per.append({"freq_mhz": float(x), "P": (P / scale).tolist(), "h": (h / scale).tolist(),
                    "chi2_dof": chi2 / max(ok.sum() - 2, 1),
                    "f_gnd": float(np.mean(A[:, g]))})
    lnu = np.log(freqs)
    u = 2 * (lnu - lnu.min()) / np.ptp(lnu) - 1

    def solve(order, sel):
        nb = order + 1
        Lt = np.zeros((1 + nb, 1 + nb))
        bt = np.zeros(1 + nb)
        for fi in np.flatnonzero(sel):
            M = np.zeros((2, 1 + nb))
            M[0, 0] = 1.0
            M[1, 1:] = legendre.legvander(u[fi:fi + 1], order)[0]
            P = np.array(per[fi]["P"])
            hv = np.array(per[fi]["h"])
            Lt += M.T @ P @ M
            bt += M.T @ hv
        cov = np.linalg.pinv(Lt)
        th = cov @ bt
        return float(th[0]), float(np.sqrt(max(cov[0, 0], 0))), th[1:].tolist()

    allf = np.ones(len(freqs), bool)
    lo = freqs <= np.median(freqs)
    res = []
    for order in args.orders:
        tg, et, coef = solve(order, allf)
        tl, el_, _ = solve(order, lo)
        th, eh, _ = solve(order, ~lo)
        res.append({"order": order, "T_gnd_k": tg, "sigma_T_gnd_k": et, "offset_coeffs": coef,
                    "T_gnd_low_half_k": tl, "sigma_low": el_, "T_gnd_high_half_k": th,
                    "sigma_high": eh})
        print(f"order {order}: T_gnd {tg:8.1f} ± {et:6.1f}   low half {tl:8.1f} ± {el_:6.1f}   "
              f"high half {th:8.1f} ± {eh:6.1f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"joint_{args.beam}.json").write_text(json.dumps({
        "provenance": {
            "product": "ground_sky_joint", "version": args.version, "beam": args.beam,
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "data-analysis/scripts/marjum-2026-07/ground_sky/joint_fit.py",
            "argv": sys.argv[1:],
            "code": {"data-analysis": git_rev(HERE),
                     "eigsep_sim": git_rev(Path(eigsep_sim.__file__).parent)},
            "inputs": {"fit_npz_sha256": sha256(fdir / "fit.npz"), "beam_sha256": sha256(beam_path)}},
        "params": {k: v for k, v in vars(args).items() if k not in ("version", "out_dir")},
        "per_frequency": per, "orders": res}, indent=1))


if __name__ == "__main__":
    main()
