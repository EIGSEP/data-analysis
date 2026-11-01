"""
Noise or systematics at the nulls? On pixels seen in >= 2 dwells, solve u
separately per dwell and compare dwell A with dwell B (repeatability) and each
dwell with HFSS, per component, binned by the component's HFSS share of the
pixel's total power (how deep in its null it is). Component powers |u_k|^2 are
used, which a common per-dwell phase cannot change; they are on one real scale
per tooth (as the maps) and divided by HFSS's total power at the pixel. Errors
that depend only on the body-frame direction repeat between dwells; noise and
roll-dependent leakage (geometry, polarization, g) do not, and the second grows
with the roll difference.

Output: output/dwell_repeatability.json (~5 min).
"""

import json

import numpy as np
from scipy.stats import spearmanr

from eigsep_data.beam_mapping import vector_beam as vb
from mounts import OUT, geometry, load, power_scale, truth

DWELL_S = 30.0
BINS = [(0, 0.01, "<1%"), (0.01, 0.1, "1-10%"), (0.1, 1.01, ">10%")]

z, ch = load()
t, az = z["times"], np.asarray(z["az"], float)
res = {}
for mount in ("L0", "L"):
    pxs, r, E1, E2 = geometry(mount, z)
    rep, hf, share = ([[] for _ in range(3)] for _ in range(3))
    droll, dtot = [], []
    for c in ch:
        v = c["v"]
        g = vb.fit_closure_g(pxs, r, E1, E2, c["Ex"], c["Ey"], v)
        full = vb.solve_sampled_beam(pxs, r, E1, E2, c["Ex"], c["Ey"], v, g=g)
        T = truth(z, c["k"], full["pix"])
        s = power_scale(full["u"], T)
        hfss = dict(zip(full["pix"], T))
        for p, idx in vb.group_by_pixel(pxs, v, min_visits=2).items():
            if p not in hfss:
                continue
            dwells = np.split(idx, np.flatnonzero(np.diff(t[idx]) > DWELL_S) + 1)
            if len(dwells) < 2:
                continue
            sols = []
            for d in dwells[:2]:
                u, _, _, cond = vb.solve_pixel(d, E1, E2, r, c["Ex"], c["Ey"], g=g)
                if not np.isfinite(cond) or cond > 30:
                    break
                sols.append((s * np.abs(u) ** 2, az[d].mean()))
            if len(sols) < 2:
                continue
            h = np.abs(hfss[p]) ** 2
            ht = h.sum()
            (pa, aa), (pb, ab) = sols
            for k in range(3):
                rep[k].append((pa[k] - pb[k]) / np.sqrt(2) / ht)
                hf[k] += [(pa[k] - h[k]) / ht, (pb[k] - h[k]) / ht]
                share[k].append(h[k] / ht)
            droll.append(abs((aa - ab + 90) % 180 - 90))  # roll difference, mod the dipole's 180 deg
            dtot.append(np.sum(np.abs(pa[1:] - pb[1:])) / ht)
    out = {"n_pixel_teeth": len(droll)}
    for k, name in enumerate("xyz"):
        R, F, S = np.array(rep[k]), np.array(hf[k]), np.array(share[k])
        S2 = np.repeat(S, 2)
        for lo, hi, lab in BINS:
            mr, mf = (S >= lo) & (S < hi), (S2 >= lo) & (S2 < hi)
            if mr.sum() < 20:
                continue
            rr, rf = np.sqrt(np.mean(R[mr] ** 2)), np.sqrt(np.mean(F[mf] ** 2))
            out[f"{name} {lab}"] = dict(n=int(mr.sum()), repeat_rms=float(rr), vs_hfss_rms=float(rf),
                                        systematic_rms=float(np.sqrt(max(rf ** 2 - rr ** 2, 0))))
    rho, pval = spearmanr(droll, dtot)
    out["y+z dwell difference vs roll difference: spearman"] = [float(rho), float(pval)]
    res[mount] = out
    print(mount, json.dumps(out, indent=1), flush=True)
OUT.mkdir(exist_ok=True)
json.dump(res, open(OUT / "dwell_repeatability.json", "w"), indent=1)
