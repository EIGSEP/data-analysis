"""Tier-2 noise-wave fit on the 07-18 rx-B spare-port blocks (M003 § 7.1, item 2).

Fits the receiver noise-wave parameters T_unc, T_cos, T_sin and the load
constant T_L, each a Chebyshev series in frequency, to three calibrators seen
at reference plane P: the ambient load (RFAMB) and the spare-port cable (RFSP1)
terminated open and short. T_NS is held at the nameplate 917 K, because with
every calibrator at one temperature the set cannot determine it (M003 § 6.1).

Inputs, per calibrator and channel (eq. cal / eq. X of eigsep_cal docs/api.md):
- Q_s = (P_s - P_L) / (P_NS - P_L), from tcal@v0002's T* per integration,
  Q = (T* - T_load) / T_NS, then a masked, integration-time-weighted mean per
  termination and block;
- Gamma_s and Gamma_rec at P, from the rx-B S11 sweep (07-18 01:17), with
  the correct switch file;
- T_s: the load's thermistor reading. The open and short reflect fully, so
  their source temperature at P is the cable's physical temperature, which
  was not logged; it is taken as the load temperature and the sensitivity
  to that assumption is reported.

Checks, each of which can come out against the fit:
- synthetic: Q generated from known parameters with these exact Gammas and
  the measured noise, then fitted;
- held out: fit the load plus one termination, predict the other;
- repeatability: the four blocks fitted separately.

Usage::

    EIGSEP_CAMPAIGN_ROOT=/path/to/marjum-2026-07 python fit_tier2.py --out DIR
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial import chebyshev as C

import eigsep_cal.s11
import eigsep_data
from eigsep_cal import S11
from eigsep_data import MetadataIndex

KEY = "4"
WINDOW = ("2026-07-18T01:20:00Z", "2026-07-18T03:00:00Z")   # rx-B, tcal-covered
BAND = (50.0, 230.0)
NTERMS = 7
SAMPLE_NSIG = 5.0
PARAMS = ("T_L", "T_unc", "T_cos", "T_sin")


# ------------------------------------------------------------------ data

def wmean_masked(x, tau):
    """Integration-time-weighted mean over rows, masking per sample (non-finite,
    > SAMPLE_NSIG robust sigma from the per-channel median, which only flags).
    Returns mean, its standard error, and the effective row count."""
    x = np.asarray(x, float)
    med = np.nanmedian(x, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(x - med), axis=0)
    ok = np.isfinite(x) & ~((mad > 0) & (np.abs(x - med) > SAMPLE_NSIG * mad))
    w = np.where(ok, tau[:, None], 0.0)
    ws = w.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(ws > 0, (w * np.where(ok, x, 0)).sum(0) / ws, np.nan)
        var = (w * np.where(ok, (x - m) ** 2, 0)).sum(0) / ws
        neff = ws ** 2 / (w ** 2).sum(0)
        se = np.sqrt(var / np.maximum(neff - 1, 1))
    return m, se, neff


def load_calibrators(root, t_ns):
    idx = MetadataIndex(root / "data")
    sel = idx.select(time=WINDOW, rfswitch=["RFSP1", "RFAMB", "RFANT"])
    b = sel.load_bundle(key=KEY, root=root, products=["tcal@v0002"])
    tl = b.products["tcal"]["tcal_t_load_k"]
    q = (b.t_star - tl[:, None]) / t_ns
    st = b.meta.rfswitch.to_numpy(str)
    term = b.meta.potmon_sp1_term_name.astype(str).to_numpy()
    tau = b.meta.integration_time.to_numpy(float)
    gap = np.r_[False, np.diff(b.t) > 30]
    block = np.cumsum(gap | np.r_[False, st[1:] != st[:-1]])
    groups = []
    for kind, label in (("RFSP1", "OPEN"), ("RFSP1", "SHORT"), ("RFAMB", None)):
        rows = (st == kind) & np.isfinite(q).any(1) & np.isfinite(tl)
        if label is not None:
            rows &= term == label
        for blk in np.unique(block[rows]):
            r = rows & (block == blk)
            if r.sum() < 20:
                continue
            m, se, n = wmean_masked(q[r], tau[r])
            groups.append({"cal": {"OPEN": "sp1_open", "SHORT": "sp1_short", None: "amb"}[label],
                           "t": float(np.average(b.t[r], weights=tau[r])), "n": int(r.sum()),
                           "Q": m, "Q_se": se, "T_load": float(np.average(tl[r], weights=tau[r]))})
    inband = (b.freqs_mhz >= BAND[0]) & (b.freqs_mhz < BAND[1])
    ant = (st == "RFANT") & np.isfinite(q[:, inband]).all(1)
    qa, qa_se, _ = wmean_masked(q[ant], tau[ant])
    tant1, _, _ = wmean_masked(b.calibrated[ant], tau[ant])
    return b.freqs_mhz, groups, {"Q": qa, "Q_se": qa_se, "T_tier1": tant1,
                                 "T_load": float(np.average(tl[ant], weights=tau[ant])),
                                 "n": int(ant.sum())}


def gammas(root, f):
    s = S11(root / "data" / "s11").calibrate(root / "data" / "s11" / "cal_materials" / "switch_sparams.npz",
                                            eigsep_cal.s11.PACKAGED_FILES[2026]["osldata"])
    fs = s.freqs / 1e6
    t0 = pd.Timestamp(WINDOW[0]).timestamp()
    out = {}
    for d in ("amb", "sp1_open", "sp1_short", "rec", "ant"):
        t, g = s.get_all_s11s(d, "default")
        t = np.asarray(t, float)
        k = np.argmin(np.abs(t - t0))   # the single rx-B sweep, 07-18 01:17
        gk = np.asarray(g)[k]
        out[d] = np.interp(f, fs, gk.real, left=np.nan, right=np.nan) + \
            1j * np.interp(f, fs, gk.imag, left=np.nan, right=np.nan)
        out[f"{d}_utc"] = pd.to_datetime(t[k], unit="s").strftime("%m-%d %H:%M")
    return out


# ------------------------------------------------------------------ model

def xcols(gs, gr):
    """Eq. X columns (without X_NS = Q X_L), per channel."""
    a = 1 - np.abs(gs) ** 2
    sr = np.sqrt(1 - np.abs(gr) ** 2)
    z = gs * (1 - np.conj(gs) * np.conj(gr))
    return {"T_L": np.abs(1 - gs * gr) ** 2 / a, "T_unc": -np.abs(gs) ** 2 / a,
            "T_cos": -z.real / (a * sr), "T_sin": -z.imag / (a * sr)}


def basis(f):
    x = 2 * (f - BAND[0]) / (BAND[1] - BAND[0]) - 1
    return C.chebvander(x, NTERMS - 1)


def fit(f, groups, G, t_ns, use, t_sp1_offset=0.0):
    """Whitened least squares for the Chebyshev coefficients of PARAMS."""
    m = (f >= BAND[0]) & (f < BAND[1])
    B = basis(f[m])
    rows, y = [], []
    for g in groups:
        if g["cal"] not in use:
            continue
        X = xcols(G[g["cal"]][m], G["rec"][m])
        t_s = g["T_load"] + (t_sp1_offset if g["cal"].startswith("sp1") else 0.0)
        sig = t_ns * X["T_L"] * g["Q_se"][m]
        ok = np.isfinite(sig) & (sig > 0) & np.isfinite(g["Q"][m])
        A = np.hstack([X[p][:, None] * B for p in PARAMS])
        yy = t_s - t_ns * g["Q"][m] * X["T_L"]
        rows.append(A[ok] / sig[ok, None])
        y.append(yy[ok] / sig[ok])
    A, y = np.vstack(rows), np.concatenate(y)
    scale = np.linalg.norm(A, axis=0)
    c, *_ = np.linalg.lstsq(A / scale, y, rcond=None)
    c /= scale
    sv = np.linalg.svd(A / scale, compute_uv=False)
    chi2 = float(np.sum((A @ c - y) ** 2) / (len(y) - len(c)))
    theta = {p: B @ c[i * NTERMS:(i + 1) * NTERMS] for i, p in enumerate(PARAMS)}
    return theta, {"kappa": float(sv[0] / sv[-1]), "chi2_red": chi2, "n_rows": len(y)}, m


def predict_ts(theta, gs, gr, q, t_ns, m):
    X = xcols(gs[m], gr[m])
    return sum(X[p] * theta[p] for p in PARAMS) + t_ns * q[m] * X["T_L"]


def synthetic(f, groups, G, t_ns, rng):
    """Generate Q from known smooth parameters with the measured noise; refit."""
    m = (f >= BAND[0]) & (f < BAND[1])
    x = 2 * (f - BAND[0]) / (BAND[1] - BAND[0]) - 1
    true = {"T_L": 330 + 20 * x, "T_unc": 150 - 40 * x, "T_cos": 120 * x ** 2 - 60,
            "T_sin": 80 - 30 * x}
    fake = []
    for g in groups:
        X = xcols(G[g["cal"]], G["rec"])
        q = (g["T_load"] - sum(X[p] * true[p] for p in PARAMS)) / (t_ns * X["T_L"])
        fake.append({**g, "Q": q + rng.normal(size=q.size) * g["Q_se"]})
    theta, info, _ = fit(f, fake, G, t_ns, ("amb", "sp1_open", "sp1_short"))
    err = {p: float(np.nanmax(np.abs(theta[p] - true[p][m]))) for p in PARAMS}
    return err, info


# ------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = eigsep_data.get_campaign_root(required=True)
    t_ns = float(json.load(open(root / "derived" / "tcal" / "v0002" / "manifest.json"))["model"]["t_ns_k"])
    f, groups, ant = load_calibrators(root, t_ns)
    G = gammas(root, f)
    res = {"t_ns_k": t_ns, "band_mhz": BAND, "n_terms": NTERMS,
           "s11_sweep_utc": {d: G[f"{d}_utc"] for d in ("amb", "sp1_open", "sp1_short", "rec", "ant")},
           "groups": [{k: g[k] for k in ("cal", "n", "T_load")} | {"utc": pd.to_datetime(g["t"], unit="s").strftime("%H:%M:%S")}
                      for g in groups]}

    # Synthetic recovery with these Gammas and noise.
    err, sinfo = synthetic(f, groups, G, t_ns, np.random.default_rng(0))
    res["synthetic"] = {"max_abs_err_k": err, **sinfo}

    # The fit.
    theta, info, m = fit(f, groups, G, t_ns, ("amb", "sp1_open", "sp1_short"))
    res["fit"] = info
    fb = f[m]
    for p in PARAMS:
        res[f"{p}_at"] = {str(int(x)): float(np.interp(x, fb, theta[p])) for x in (60, 100, 150, 200)}

    # Held out: fit the load + one termination, predict the other's T_s.
    held = {}
    for keep, drop in (("sp1_open", "sp1_short"), ("sp1_short", "sp1_open")):
        th, inf, mm = fit(f, groups, G, t_ns, ("amb", keep))
        pred = [predict_ts(th, G[drop], G["rec"], g["Q"], t_ns, mm) - g["T_load"]
                for g in groups if g["cal"] == drop]
        resid = np.concatenate(pred)
        held[f"fit_amb+{keep}->predict_{drop}"] = {
            "kappa": inf["kappa"], "median_abs_resid_k": float(np.nanmedian(np.abs(resid))),
            "p95_abs_resid_k": float(np.nanpercentile(np.abs(resid), 95))}
    res["held_out"] = held

    # Repeatability: each block (one open, one short, nearest load) separately.
    sp = [g for g in groups if g["cal"] != "amb"]
    amb = [g for g in groups if g["cal"] == "amb"]
    per = []
    for g_o in [g for g in sp if g["cal"] == "sp1_open"]:
        g_s = min((g for g in sp if g["cal"] == "sp1_short"), key=lambda g: abs(g["t"] - g_o["t"]))
        g_a = min(amb, key=lambda g: abs(g["t"] - g_o["t"]))
        th, inf, _ = fit(f, [g_o, g_s, g_a], G, t_ns, ("amb", "sp1_open", "sp1_short"))
        per.append({p: th[p] for p in PARAMS})
    res["block_spread_k"] = {p: float(np.nanmedian(np.std([q[p] for q in per], axis=0))) for p in PARAMS}

    # Sensitivity to the unlogged SP1 cable temperature.
    th5, _, _ = fit(f, groups, G, t_ns, ("amb", "sp1_open", "sp1_short"), t_sp1_offset=5.0)
    res["sp1_temp_plus5k_shift_k"] = {p: float(np.nanmedian(np.abs(th5[p] - theta[p]))) for p in PARAMS}

    # Antenna: tier 2 against tier 1 (tcal), same rows.
    X = xcols(G["ant"][m], G["rec"][m])
    t2 = sum(X[p] * theta[p] for p in PARAMS) + t_ns * ant["Q"][m] * X["T_L"]
    t1 = ant["T_tier1"][m]
    res["antenna"] = {"n_rows": ant["n"], "ratio_tier2_over_tier1_at": {
        str(int(x)): float(np.interp(x, fb, t2 / t1)) for x in (60, 80, 100, 150, 200)}}

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for p, c in zip(PARAMS, ["#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"]):
        axes[0].plot(fb, theta[p], color=c, label=p)
    axes[0].set(ylabel="K", title="tier-2 parameters (T_NS fixed at %.0f K)" % t_ns)
    axes[0].legend(ncol=4, fontsize=7)
    axes[1].semilogy(fb, t1, color="#2a78d6", label="tier 1 (tcal v0002)")
    axes[1].semilogy(fb, t2, color="#1baf7a", label="tier 2 (noise waves)")
    axes[1].set(xlabel="frequency (MHz)", ylabel="antenna T at P (K)")
    axes[1].legend(fontsize=7)
    fig.savefig(out / "tier2.png", dpi=120, bbox_inches="tight")
    (out / "tier2.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("synthetic", "fit", "held_out", "block_spread_k",
                                          "sp1_temp_plus5k_shift_k", "antenna")}, indent=1))


if __name__ == "__main__":
    main()
