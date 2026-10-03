"""Diagnose the remaining misfit of the delay model (fit_models.py, model B).

1. Noise: is the block-to-block scatter of Q larger than the within-block
   standard error the fit uses? If so the fit's chi2 overstates the misfit;
   refit with the block-to-block scatter as the uncertainty.
2. S11 at the spare port: does the noise-wave fit want a correction to the
   measured Gamma of the SP1 port? A small complex delta(nu), shared by the
   open and the short (a property of the port, like a directivity error), is
   added to both measured Gammas and fitted together with the model-B
   parameters (variable projection: the linear parameters are solved inside).
   A physical mismatch at the port is already in an accurate measurement,
   so a fit that needs delta says the measurement is off. delta is learned
   from the load plus the open only and the short is then predicted, so it
   cannot simply absorb the short's misfit.

Usage::

    EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07 python diagnose.py --out DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as Cb
from scipy.optimize import least_squares

import eigsep_data
import fit_models as M
import fit_tier2 as F

N_DELTA = 3          # Chebyshev terms per real and imaginary part of delta


def block_scatter(f, groups):
    """Per termination: median over channels of (block-to-block variance of Q)
    / (mean within-block SE^2)."""
    m = (f >= F.BAND[0]) & (f < F.BAND[1])
    out = {}
    for cal in ("sp1_open", "sp1_short", "amb"):
        gs = [g for g in groups if g["cal"] == cal]
        q = np.array([g["Q"][m] for g in gs])
        se = np.array([g["Q_se"][m] for g in gs])
        if len(gs) < 3:
            continue
        ratio = np.nanvar(q, axis=0, ddof=1) / np.nanmean(se ** 2, axis=0)
        out[cal] = {"n_blocks": len(gs), "median_var_ratio": float(np.nanmedian(ratio)),
                    "p90_var_ratio": float(np.nanpercentile(ratio, 90))}
    return out


def inflate(groups, f):
    """Replace each group's SE by the block-to-block standard deviation of its
    termination (per channel), when that is larger."""
    new = []
    for cal in ("sp1_open", "sp1_short", "amb"):
        gs = [g for g in groups if g["cal"] == cal]
        if len(gs) >= 3:
            sd = np.nanstd(np.array([g["Q"] for g in gs]), axis=0, ddof=1)
            for g in gs:
                new.append({**g, "Q_se": np.fmax(g["Q_se"], sd)})
        else:
            new.extend(gs)
    return new


def with_delta(G, f, d):
    m = (f >= F.BAND[0]) & (f < F.BAND[1])
    x = 2 * (f - F.BAND[0]) / (F.BAND[1] - F.BAND[0]) - 1
    delta = Cb.chebval(x, d[:N_DELTA]) + 1j * Cb.chebval(x, d[N_DELTA:])
    G2 = dict(G)
    for k in ("sp1_open", "sp1_short"):
        G2[k] = G[k] + np.where(m, delta, 0)
    return G2, delta


def resid_vec(f, groups, G, t_ns, use, tau):
    """Whitened residuals of model B at its linear least-squares solution."""
    m = (f >= F.BAND[0]) & (f < F.BAND[1])
    B = F.basis(f[m])
    rows, ys = [], []
    for g in groups:
        if g["cal"] not in use:
            continue
        X = F.xcols(G[g["cal"]][m], G["rec"][m])
        sig = t_ns * X["T_L"] * g["Q_se"][m]
        y = g["T_load"] - t_ns * g["Q"][m] * X["T_L"]
        ok = np.isfinite(sig) & (sig > 0) & np.isfinite(y)
        A = np.hstack([t[:, None] * B for t in M.terms("B", G[g["cal"]][m], G["rec"][m], f[m], tau)])
        rows.append(A[ok] / sig[ok, None])
        ys.append(y[ok] / sig[ok])
    A, y = np.vstack(rows), np.concatenate(ys)
    sc = np.linalg.norm(A, axis=0)
    c, *_ = np.linalg.lstsq(A / sc, y, rcond=None)
    return (A / sc) @ c - y


def fit_delta(f, groups, G, t_ns, use, tau):
    def resid(d):
        G2, _ = with_delta(G, f, d)
        return resid_vec(f, groups, G2, t_ns, use, tau)
    r = least_squares(resid, np.zeros(2 * N_DELTA), method="trf", x_scale=0.01)
    G2, delta = with_delta(G, f, r.x)
    c, chi2, kappa, m = M.solve("B", f, groups, G2, t_ns, use, tau)
    return c, chi2, kappa, m, G2, delta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = eigsep_data.get_campaign_root(required=True)
    t_ns = float(json.load(open(root / "derived" / "tcal" / "v0002" / "manifest.json"))["model"]["t_ns_k"])
    f, groups, ant = F.load_calibrators(root, t_ns)
    G = F.gammas(root, f)
    allcal = ("amb", "sp1_open", "sp1_short")
    res = {"block_scatter": block_scatter(f, groups)}

    # 1. Refit model B with block-to-block uncertainties.
    gi = inflate(groups, f)
    c, chi2, kappa, m, tau, _ = M.best("B", f, gi, G, t_ns, allcal)
    res["model_B_inflated_errors"] = {"chi2_red": chi2, "tau_ns": tau}

    # 2. S11 correction delta at the spare port, on the inflated errors.
    _, chi2d, kd, md, G2, delta = fit_delta(f, gi, G, t_ns, allcal, tau)
    mb = (f >= F.BAND[0]) & (f < F.BAND[1])
    res["model_B_plus_delta"] = {"chi2_red": chi2d, "kappa": kd, "tau_ns": tau,
                                 "max_abs_delta": float(np.nanmax(np.abs(delta[mb]))),
                                 "median_abs_delta": float(np.nanmedian(np.abs(delta[mb])))}
    # Held out: delta and parameters from the load + open; predict the short.
    held = {}
    for keep, drop in (("sp1_open", "sp1_short"), ("sp1_short", "sp1_open")):
        _, _, _, mh, th, _ = M.best("B", f, gi, G, t_ns, ("amb", keep))
        c0, _, _, mh = M.solve("B", f, gi, G, t_ns, ("amb", keep), th)
        r0 = np.concatenate([M.predict_ts("B", c0, G[drop], G["rec"], g["Q"], f, mh, t_ns, th) - g["T_load"]
                             for g in gi if g["cal"] == drop])
        c1, _, _, mh1, G2h, _ = fit_delta(f, gi, G, t_ns, ("amb", keep), th)
        r1 = np.concatenate([M.predict_ts("B", c1, G2h[drop], G2h["rec"], g["Q"], f, mh1, t_ns, th) - g["T_load"]
                             for g in gi if g["cal"] == drop])
        held[f"predict_{drop}"] = {
            "without_delta_median_abs_k": float(np.nanmedian(np.abs(r0))),
            "with_delta_median_abs_k": float(np.nanmedian(np.abs(r1))),
            "without_delta_p95_k": float(np.nanpercentile(np.abs(r0), 95)),
            "with_delta_p95_k": float(np.nanpercentile(np.abs(r1), 95))}
    res["held_out"] = held
    (out / "diagnose.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
