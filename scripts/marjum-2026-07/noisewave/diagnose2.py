"""Second round of diagnostics for the delay model (model B), with
block-to-block uncertainties (diagnose.py):

1. Gamma_rec: does the fit want a small smooth complex correction to the
   receiver's measured reflection (rx-B has one receiver sweep)? Plausible
   if it is small (< ~0.03) and improves held-out prediction.
2. Drift: each smooth function gains a term linear in time across the
   40 minutes of blocks (N_SLOPE Chebyshev terms in frequency).
3. A delay on T_unc: T_unc gains a(nu) cos(2 pi nu tau_u) + b(nu) sin(...),
   a and b smooth (N_RIPPLE terms), tau_u scanned.

Each variant is held-out tested: fitted to the load plus one termination,
used to predict the other.

Usage::

    EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07 python diagnose2.py --out DIR
"""

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev as Cb
from scipy.optimize import least_squares

import eigsep_data
import diagnose as D
import fit_models as M
import fit_tier2 as F

TAU = 8.5            # ns, model B's delay
N_SLOPE = 3
N_RIPPLE = 3
N_DELTA = 3
TAU_U = np.arange(2.0, 80.01, 1.0)


def design(f, groups, G, t_ns, use, tau=TAU, slope=False, tau_u=None, t0=None):
    """Whitened design matrix and data for model B plus optional features.
    Returns A, y, and a closure that predicts T_s for (gs, q, t)."""
    m = (f >= F.BAND[0]) & (f < F.BAND[1])
    fb = f[m]
    B = F.basis(fb)
    Bs, Br = B[:, :N_SLOPE], B[:, :N_RIPPLE]
    span = 2400.0

    def cols(gs, t):
        tt = M.terms("B", gs[m], G["rec"][m], fb, tau)
        out = [t_[:, None] * B for t_ in tt]
        if slope:
            dt = (t - t0) / span
            out += [t_[:, None] * Bs * dt for t_ in tt]
        if tau_u is not None:
            xu = F.xcols(gs[m], G["rec"][m])["T_unc"]
            ph = 2 * np.pi * fb * 1e6 * tau_u * 1e-9
            out += [(xu * np.cos(ph))[:, None] * Br, (xu * np.sin(ph))[:, None] * Br]
        return np.hstack(out)

    rows, ys = [], []
    for g in groups:
        if g["cal"] not in use:
            continue
        X = F.xcols(G[g["cal"]][m], G["rec"][m])
        sig = t_ns * X["T_L"] * g["Q_se"][m]
        y = g["T_load"] - t_ns * g["Q"][m] * X["T_L"]
        ok = np.isfinite(sig) & (sig > 0) & np.isfinite(y)
        A = cols(G[g["cal"]], g["t"])
        rows.append(A[ok] / sig[ok, None])
        ys.append(y[ok] / sig[ok])
    A, y = np.vstack(rows), np.concatenate(ys)

    def predictor(c):
        def pred(gs, q, t):
            X = F.xcols(gs[m], G["rec"][m])
            return cols(gs, t) @ c + t_ns * q[m] * X["T_L"]
        return pred
    return A, y, predictor


def lsq(A, y):
    sc = np.linalg.norm(A, axis=0)
    c, *_ = np.linalg.lstsq(A / sc, y, rcond=None)
    sv = np.linalg.svd(A / sc, compute_uv=False)
    r = (A / sc) @ c - y
    return c / sc, float(r @ r / (len(y) - len(c))), float(sv[0] / sv[-1]), r


def held_out(f, groups, G, t_ns, **kw):
    out = {}
    for keep, drop in (("sp1_open", "sp1_short"), ("sp1_short", "sp1_open")):
        A, y, P = design(f, groups, G, t_ns, ("amb", keep), **kw)
        c, *_ = lsq(A, y)
        pred = P(c)
        r = np.concatenate([pred(G[drop], g["Q"], g["t"]) - g["T_load"] for g in groups if g["cal"] == drop])
        out[f"predict_{drop}"] = {"median_abs_k": float(np.nanmedian(np.abs(r))),
                                  "p95_abs_k": float(np.nanpercentile(np.abs(r), 95))}
    return out


def rec_delta(f, G, d):
    m = (f >= F.BAND[0]) & (f < F.BAND[1])
    x = 2 * (f - F.BAND[0]) / (F.BAND[1] - F.BAND[0]) - 1
    delta = Cb.chebval(x, d[:N_DELTA]) + 1j * Cb.chebval(x, d[N_DELTA:])
    return {**G, "rec": G["rec"] + np.where(m, delta, 0)}, delta


def fit_rec_delta(f, groups, G, t_ns, use, **kw):
    def resid(d):
        G2, _ = rec_delta(f, G, d)
        A, y, _ = design(f, groups, G2, t_ns, use, **kw)
        return lsq(A, y)[3]
    r = least_squares(resid, np.zeros(2 * N_DELTA), method="trf", x_scale=0.01)
    return rec_delta(f, G, r.x)


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
    gi = D.inflate(groups, f)
    t0 = float(np.mean([g["t"] for g in gi]))
    allcal = ("amb", "sp1_open", "sp1_short")
    m = (f >= F.BAND[0]) & (f < F.BAND[1])
    res = {}

    def report(name, **kw):
        A, y, _ = design(f, gi, G if "G" not in kw else kw.pop("G"), t_ns, allcal, t0=t0, **kw)
        c, chi2, kappa, _ = lsq(A, y)
        res[name] = {"chi2_red": chi2, "kappa": kappa, "n_par": len(c)}

    report("B")
    res["B"]["held_out"] = held_out(f, gi, G, t_ns)

    # 1. Gamma_rec correction.
    G2, delta = fit_rec_delta(f, gi, G, t_ns, allcal)
    A, y, _ = design(f, gi, G2, t_ns, allcal)
    c, chi2, kappa, _ = lsq(A, y)
    hold = {}
    for keep, drop in (("sp1_open", "sp1_short"), ("sp1_short", "sp1_open")):
        Gh, dh = fit_rec_delta(f, gi, G, t_ns, ("amb", keep))
        Ah, yh, Ph = design(f, gi, Gh, t_ns, ("amb", keep))
        ch, *_ = lsq(Ah, yh)
        r = np.concatenate([Ph(ch)(Gh[drop], g["Q"], g["t"]) - g["T_load"] for g in gi if g["cal"] == drop])
        hold[f"predict_{drop}"] = {"median_abs_k": float(np.nanmedian(np.abs(r))),
                                   "p95_abs_k": float(np.nanpercentile(np.abs(r), 95)),
                                   "max_abs_delta": float(np.nanmax(np.abs(dh[m])))}
    res["B+rec_delta"] = {"chi2_red": chi2, "kappa": kappa,
                          "median_abs_delta": float(np.nanmedian(np.abs(delta[m]))),
                          "max_abs_delta": float(np.nanmax(np.abs(delta[m]))), "held_out": hold}

    # 2. Drift.
    report("B+drift", slope=True)
    res["B+drift"]["held_out"] = held_out(f, gi, G, t_ns, slope=True, t0=t0)

    # 3. Delay on T_unc.
    scan = []
    for tu in TAU_U:
        A, y, _ = design(f, gi, G, t_ns, allcal, tau_u=tu)
        scan.append((float(tu), lsq(A, y)[1]))
    tu = min(scan, key=lambda x: x[1])[0]
    report("B+Tunc_delay", tau_u=tu)
    res["B+Tunc_delay"]["tau_u_ns"] = tu
    res["B+Tunc_delay"]["held_out"] = held_out(f, gi, G, t_ns, tau_u=tu)

    # All three structural additions together (drift + T_unc delay).
    report("B+drift+Tunc_delay", slope=True, tau_u=tu)
    res["B+drift+Tunc_delay"]["held_out"] = held_out(f, gi, G, t_ns, slope=True, tau_u=tu, t0=t0)

    (out / "diagnose2.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res.items():
        h = v.get("held_out", {})
        print(f"{k:22s} chi2/dof {v['chi2_red']:8.1f}  kappa {v['kappa']:.1e}  "
              + "  ".join(f"{n}: {x['median_abs_k']:.0f}/{x['p95_abs_k']:.0f} K" for n, x in h.items())
              + (f"  |delta| med {v['median_abs_delta']:.3f} max {v['max_abs_delta']:.3f}" if "median_abs_delta" in v else "")
              + (f"  tau_u {v['tau_u_ns']}" if "tau_u_ns" in v else ""))


if __name__ == "__main__":
    main()
