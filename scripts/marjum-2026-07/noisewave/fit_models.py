"""Tier-2 noise-wave models compared on the 07-18 rx-B spare-port blocks.

fit_tier2.py found that smooth (7-term Chebyshev) T_unc, T_cos, T_sin and T_L
do not fit the open/short data (chi2/dof ~ 1e4; residuals oscillate at ~50 ns).
This compares four models, each written as the frequency-dependent multipliers
of its smooth unknowns (eq. X and eq. map of eigsep_cal docs/api.md):

- A, Rogers & Bowman, smooth: T_L, T_unc, T_cos, T_sin.
- B, Rogers & Bowman with a delay: W = T_cos - i T_sin = (a + i b) exp(-2 pi i nu tau),
  a and b smooth; tau scanned.
- C, intrinsic (Bucher et al.): T_R, T_Li, complex c smooth, mapped through the
  measured Gamma_rec; the load constant as s / (1 - |Gamma_rec|^2), s smooth.
- D, intrinsic with c = (a + i b) exp(-2 pi i nu tau).

T_NS is fixed (the calibrators are isothermal). For each model: chi2/dof, the
condition number, and held-out prediction (fit the load plus one termination,
tau re-scanned on that training set, predict the other termination).

Usage::

    EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07 python fit_models.py --out DIR
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import eigsep_data
import fit_tier2 as F

TAUS = np.arange(0.0, 80.01, 0.5)       # ns


def terms(model, gs, gr, f, tau_ns):
    """Multipliers of each smooth unknown, for one calibrator (n_unknown, n_freq)."""
    X = F.xcols(gs, gr)
    Z = X["T_cos"] + 1j * X["T_sin"]
    r2 = np.abs(gr) ** 2
    sq = np.sqrt(1 - r2)
    ph = np.exp(-2j * np.pi * f * 1e6 * tau_ns * 1e-9)
    if model == "A":
        return [X["T_L"], X["T_unc"], X["T_cos"], X["T_sin"]]
    if model == "B":
        return [X["T_L"], X["T_unc"], (Z * ph).real, -(Z * ph).imag]
    kre = 2 * X["T_unc"] * gr.real + 2 * sq * Z.real
    kim = 2 * X["T_unc"] * gr.imag - 2 * sq * Z.imag
    base = [X["T_L"] / (1 - r2), X["T_unc"], X["T_unc"] * r2 + 2 * sq * (Z * gr).real]
    if model == "C":
        return base + [kre, kim]
    if model == "D":
        c, s = ph.real, -ph.imag          # c_re = a cos + b sin, c_im = -a sin + b cos
        return base + [kre * c - kim * s, kre * s + kim * c]
    raise ValueError(model)


def solve(model, f, groups, G, t_ns, use, tau):
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
        A = np.hstack([t[:, None] * B for t in terms(model, G[g["cal"]][m], G["rec"][m], f[m], tau)])
        rows.append(A[ok] / sig[ok, None])
        ys.append(y[ok] / sig[ok])
    A, y = np.vstack(rows), np.concatenate(ys)
    sc = np.linalg.norm(A, axis=0)
    c, *_ = np.linalg.lstsq(A / sc, y, rcond=None)
    c /= sc
    sv = np.linalg.svd(A / sc, compute_uv=False)
    return c, float(np.sum((A @ c - y) ** 2) / (len(y) - len(c))), float(sv[0] / sv[-1]), m


def best(model, f, groups, G, t_ns, use):
    taus = TAUS if model in ("B", "D") else [0.0]
    fits = [(solve(model, f, groups, G, t_ns, use, t), t) for t in taus]
    (c, chi2, kappa, m), tau = min(fits, key=lambda x: x[0][1])
    scan = [(t, r[1]) for r, t in fits]
    return c, chi2, kappa, m, tau, scan


def predict_ts(model, c, gs, gr, q, f, m, t_ns, tau):
    B = F.basis(f[m])
    X = F.xcols(gs[m], gr[m])
    tt = terms(model, gs[m], gr[m], f[m], tau)
    n = F.NTERMS
    return sum(t * (B @ c[i * n:(i + 1) * n]) for i, t in enumerate(tt)) + t_ns * q[m] * X["T_L"]


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
    res, scans = {}, {}
    allcal = ("amb", "sp1_open", "sp1_short")
    for model in "ABCD":
        c, chi2, kappa, m, tau, scan = best(model, f, groups, G, t_ns, allcal)
        scans[model] = scan
        held = {}
        for keep, drop in (("sp1_open", "sp1_short"), ("sp1_short", "sp1_open")):
            ch, _, kh, mh, th, _ = best(model, f, groups, G, t_ns, ("amb", keep))
            r = np.concatenate([predict_ts(model, ch, G[drop], G["rec"], g["Q"], f, mh, t_ns, th) - g["T_load"]
                                for g in groups if g["cal"] == drop])
            held[f"predict_{drop}"] = {"tau_ns": th, "kappa": kh,
                                       "median_abs_k": float(np.nanmedian(np.abs(r))),
                                       "p95_abs_k": float(np.nanpercentile(np.abs(r), 95))}
        ta = predict_ts(model, c, G["ant"], G["rec"], ant["Q"], f, m, t_ns, tau)
        fb = f[m]
        res[model] = {"chi2_red": chi2, "kappa": kappa, "tau_ns": tau, "held_out": held,
                      "antenna_over_tier1_at": {str(x): float(np.interp(x, fb, ta / ant["T_tier1"][m]))
                                                for x in (60, 80, 100, 150, 200)}}
        print(model, json.dumps(res[model]))
    fig, ax = plt.subplots(figsize=(6, 3))
    for model in "BD":
        t, x = zip(*scans[model])
        ax.semilogy(t, x, label=model)
    ax.set(xlabel="tau (ns)", ylabel="chi2/dof", title="delay scan")
    ax.legend()
    fig.savefig(out / "delay_scan.png", dpi=120, bbox_inches="tight")
    (out / "models.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
